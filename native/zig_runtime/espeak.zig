const std = @import("std");
const builtin = @import("builtin");

pub const Error = error{
    InitializeFailed,
    InvalidInput,
    LibraryUnavailable,
    MissingSymbol,
    OutOfMemory,
    VoiceFailed,
};

const audio_output_sync = 2;
const text_mode_utf8 = 1;

const InitializeFn = *const fn (c_int, c_int, [*c]const u8, c_int) callconv(.c) c_int;
const SetVoiceFn = *const fn ([*c]const u8) callconv(.c) c_int;
const TextToPhonemesFn = *const fn (*[*c]const u8, c_int, c_int) callconv(.c) [*c]const u8;
const TerminateFn = *const fn () callconv(.c) c_int;

const Dyn = if (builtin.os.tag == .windows) WinDyn else StdDyn;

pub const Session = struct {
    lib: Dyn,
    initialize: InitializeFn,
    set_voice: SetVoiceFn,
    text_to_phonemes: TextToPhonemesFn,
    terminate: TerminateFn,
    phoneme_mode: c_int,
    voice: []u8 = &.{},

    pub fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        _ = self.terminate();
        if (self.voice.len > 0) allocator.free(self.voice);
        self.lib.close();
        self.* = undefined;
    }

    pub fn text(self: *Session, allocator: std.mem.Allocator, input: []const u8, voice: ?[]const u8) Error![]u8 {
        if (input.len == 0) {
            return allocator.dupe(u8, &.{}) catch error.OutOfMemory;
        }
        if (voice) |value| {
            if (value.len > 0 and !std.mem.eql(u8, value, self.voice)) {
                try self.setVoice(allocator, value);
            }
        }

        const input_z = allocator.dupeZ(u8, input) catch return error.OutOfMemory;
        defer allocator.free(input_z);
        var cursor: [*c]const u8 = input_z.ptr;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);

        while (cursor != null and cursor[0] != 0) {
            const result = self.text_to_phonemes(&cursor, text_mode_utf8, self.phoneme_mode);
            if (result == null) {
                continue;
            }
            const clause = result[0..std.mem.len(result)];
            if (clause.len == 0) {
                continue;
            }
            if (out.items.len > 0) {
                out.append(allocator, ' ') catch return error.OutOfMemory;
            }
            out.appendSlice(allocator, clause) catch return error.OutOfMemory;
        }
        return out.toOwnedSlice(allocator) catch error.OutOfMemory;
    }

    fn setVoice(self: *Session, allocator: std.mem.Allocator, value: []const u8) Error!void {
        const voice_z = allocator.dupeZ(u8, value) catch return error.OutOfMemory;
        defer allocator.free(voice_z);
        if (self.set_voice(voice_z.ptr) != 0) {
            return error.VoiceFailed;
        }
        const copy = allocator.dupe(u8, value) catch return error.OutOfMemory;
        if (self.voice.len > 0) allocator.free(self.voice);
        self.voice = copy;
    }
};

pub fn create(
    allocator: std.mem.Allocator,
    io: std.Io,
    library_path: ?[]const u8,
    data_path: ?[]const u8,
    voice: []const u8,
    phoneme_mode: i32,
) Error!*Session {
    var lib = try openLibrary(library_path);
    errdefer lib.close();

    const initialize = lib.lookup(InitializeFn, "espeak_Initialize") orelse return error.MissingSymbol;
    const set_voice = lib.lookup(SetVoiceFn, "espeak_SetVoiceByName") orelse return error.MissingSymbol;
    const text_to_phonemes = lib.lookup(TextToPhonemesFn, "espeak_TextToPhonemes") orelse return error.MissingSymbol;
    const terminate = lib.lookup(TerminateFn, "espeak_Terminate") orelse return error.MissingSymbol;

    const data_z = try resolveDataPath(allocator, io, data_path);
    defer if (data_z) |path| allocator.free(path);

    const sample_rate = initialize(audio_output_sync, 0, if (data_z) |path| path.ptr else null, 0);
    if (sample_rate <= 0) {
        return error.InitializeFailed;
    }

    const session = allocator.create(Session) catch return error.OutOfMemory;
    errdefer allocator.destroy(session);
    session.* = .{
        .lib = lib,
        .initialize = initialize,
        .set_voice = set_voice,
        .text_to_phonemes = text_to_phonemes,
        .terminate = terminate,
        .phoneme_mode = @intCast(phoneme_mode),
    };
    errdefer session.deinit(allocator);
    try session.setVoice(allocator, voice);
    return session;
}

pub fn destroy(allocator: std.mem.Allocator, session: *Session) void {
    session.deinit(allocator);
    allocator.destroy(session);
}

pub fn fromHandle(handle: ?*anyopaque) Error!*Session {
    const raw = handle orelse return error.InvalidInput;
    return @ptrCast(@alignCast(raw));
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InitializeFailed => "Zig eSpeak initialization failed.",
        error.InvalidInput => "Zig eSpeak received invalid input.",
        error.LibraryUnavailable => "Zig eSpeak could not open the eSpeak-NG dynamic library.",
        error.MissingSymbol => "Zig eSpeak could not find a required eSpeak-NG symbol.",
        error.OutOfMemory => "Zig eSpeak ran out of memory.",
        error.VoiceFailed => "Zig eSpeak failed to select the requested voice.",
    };
}

fn openLibrary(explicit: ?[]const u8) Error!Dyn {
    if (explicit) |path| {
        if (path.len > 0) {
            return Dyn.open(path);
        }
    }
    for (candidates()) |path| {
        return Dyn.open(path) catch continue;
    }
    return error.LibraryUnavailable;
}

fn candidates() []const []const u8 {
    return switch (builtin.os.tag) {
        .macos => &.{
            "libespeak-ng.dylib",
            "/opt/homebrew/lib/libespeak-ng.dylib",
            "/usr/local/lib/libespeak-ng.dylib",
        },
        .windows => &.{ "libespeak-ng.dll", "espeak-ng.dll" },
        else => &.{ "libespeak-ng.so.1", "libespeak-ng.so" },
    };
}

fn resolveDataPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    explicit: ?[]const u8,
) Error!?[:0]u8 {
    if (explicit) |path| {
        if (path.len > 0) {
            return allocator.dupeZ(u8, path) catch error.OutOfMemory;
        }
    }
    for ([_][]const u8{ "ESPEAK_DATA_PATH", "ESPEAKNG_DATA_PATH" }) |name| {
        const value = try getenvOwned(allocator, name);
        defer if (value) |bytes| allocator.free(bytes);
        if (value) |bytes| {
            return allocator.dupeZ(u8, bytes) catch error.OutOfMemory;
        }
    }
    for (dataCandidates()) |path| {
        if (isDir(io, path)) {
            return allocator.dupeZ(u8, path) catch error.OutOfMemory;
        }
    }
    return null;
}

fn dataCandidates() []const []const u8 {
    return switch (builtin.os.tag) {
        .macos => &.{
            "/opt/homebrew/share/espeak-ng-data",
            "/usr/local/share/espeak-ng-data",
            "/usr/share/espeak-ng-data",
        },
        .windows => &.{"espeak-ng-data"},
        else => &.{
            "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
            "/usr/lib/aarch64-linux-gnu/espeak-ng-data",
            "/usr/share/espeak-ng-data",
            "/usr/local/share/espeak-ng-data",
        },
    };
}

fn isDir(io: std.Io, value: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(io, value, .{}) catch return false;
    return stat.kind == .directory;
}

fn getenvOwned(allocator: std.mem.Allocator, name: []const u8) Error!?[]u8 {
    const key = allocator.dupeZ(u8, name) catch return error.OutOfMemory;
    defer allocator.free(key);
    const raw = std.c.getenv(key.ptr) orelse return null;
    const value = std.mem.span(raw);
    if (value.len == 0) {
        return null;
    }
    return allocator.dupe(u8, value) catch error.OutOfMemory;
}

const StdDyn = struct {
    inner: std.DynLib,

    fn open(path: []const u8) Error!StdDyn {
        return .{ .inner = std.DynLib.open(path) catch return error.LibraryUnavailable };
    }

    fn close(self: *StdDyn) void {
        self.inner.close();
    }

    fn lookup(self: *StdDyn, comptime T: type, name: [:0]const u8) ?T {
        return self.inner.lookup(T, name);
    }
};

const WinDyn = struct {
    handle: *anyopaque,

    extern "kernel32" fn LoadLibraryA(path: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(handle: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(handle: *anyopaque) callconv(.winapi) c_int;

    fn open(path: []const u8) Error!WinDyn {
        if (path.len >= std.fs.max_path_bytes) {
            return error.LibraryUnavailable;
        }
        var path_z: [std.fs.max_path_bytes:0]u8 = undefined;
        @memcpy(path_z[0..path.len], path);
        path_z[path.len] = 0;
        const handle = LoadLibraryA(&path_z) orelse return error.LibraryUnavailable;
        return .{ .handle = handle };
    }

    fn close(self: *WinDyn) void {
        _ = FreeLibrary(self.handle);
    }

    fn lookup(self: *WinDyn, comptime T: type, name: [:0]const u8) ?T {
        const symbol = GetProcAddress(self.handle, name.ptr) orelse return null;
        return @as(T, @ptrCast(@alignCast(symbol)));
    }
};

test "espeak reports stable errors" {
    try std.testing.expectEqualStrings(
        "Zig eSpeak received invalid input.",
        errorMessage(error.InvalidInput),
    );
}
