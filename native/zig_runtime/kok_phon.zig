const std = @import("std");
const espeak = @import("espeak.zig");
const kokoro = @import("kokoro.zig");
const phon_run = @import("phon_run.zig");
const phon_ssml = @import("phon_ssml.zig");
const phoneme = @import("phoneme.zig");

pub const Error = espeak.Error || kokoro.Error || phon_run.Error || phon_ssml.Error || phoneme.Error;

const Lang = struct {
    value: []const u8,
    owned: []u8 = &.{},

    fn deinit(self: Lang, allocator: std.mem.Allocator) void {
        if (self.owned.len > 0) allocator.free(self.owned);
    }
};

pub fn text(
    allocator: std.mem.Allocator,
    session: *espeak.Session,
    input: []const u8,
    requested_language: []const u8,
) Error![]u8 {
    const trimmed = trim(input);
    if (trimmed.len == 0) {
        return allocator.dupe(u8, &.{}) catch error.OutOfMemory;
    }

    const language = try canonicalLanguage(allocator, requested_language);
    defer language.deinit(allocator);

    const normalized = try phoneme.normText(allocator, trimmed, language.value);
    defer allocator.free(normalized);

    const route = try phon_run.route(normalized, language.value);
    if (route.mixed) {
        return mixed(allocator, session, normalized, route.language);
    }
    return spoken(allocator, session, normalized, route.language);
}

pub fn ssml(
    allocator: std.mem.Allocator,
    session: *espeak.Session,
    input: []const u8,
    requested_language: []const u8,
) Error![]u8 {
    const trimmed = trim(input);
    if (trimmed.len == 0) {
        return allocator.dupe(u8, &.{}) catch error.OutOfMemory;
    }

    const language = try canonicalLanguage(allocator, requested_language);
    defer language.deinit(allocator);

    var chunks = try phon_ssml.parse(allocator, trimmed);
    defer chunks.deinit(allocator);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (chunks.chunks) |chunk| {
        switch (chunk.kind) {
            .plain => if (trim(chunk.text).len > 0) {
                const value = try text(allocator, session, chunk.text, language.value);
                defer allocator.free(value);
                out.appendSlice(allocator, value) catch return error.OutOfMemory;
            },
            .pinyin => {
                const value = try pinyin(allocator, session, chunk.text);
                defer allocator.free(value);
                out.appendSlice(allocator, value) catch return error.OutOfMemory;
            },
            .explicit => {
                const value = try phoneme.explicitPhonemes(allocator, chunk.text);
                defer allocator.free(value);
                out.appendSlice(allocator, value) catch return error.OutOfMemory;
            },
        }
        if (chunk.space_after) {
            out.append(allocator, ' ') catch return error.OutOfMemory;
        }
    }

    const joined = out.toOwnedSlice(allocator) catch return error.OutOfMemory;
    defer allocator.free(joined);
    return kokoro.clean(allocator, joined);
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InitializeFailed => espeak.errorMessage(error.InitializeFailed),
        error.LibraryUnavailable => espeak.errorMessage(error.LibraryUnavailable),
        error.MissingSymbol => espeak.errorMessage(error.MissingSymbol),
        error.VoiceFailed => espeak.errorMessage(error.VoiceFailed),
        error.InvalidArchive,
        error.InvalidConfig,
        error.InvalidNpy,
        error.UnsupportedCompression,
        => kokoro.errorMessage(@errorCast(err)),
        error.InvalidInput => "Zig Kokoro phonemizer received invalid input.",
        error.InvalidUtf8 => "Zig Kokoro phonemizer received invalid UTF-8.",
        error.LengthOverflow => "Zig Kokoro phonemizer output length overflowed.",
        error.OutOfMemory => "Zig Kokoro phonemizer ran out of memory.",
    };
}

fn mixed(
    allocator: std.mem.Allocator,
    session: *espeak.Session,
    input: []const u8,
    default_language: []const u8,
) Error![]u8 {
    var runs = try phon_run.split(allocator, input, default_language);
    defer runs.deinit(allocator);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (runs.items) |run| {
        if (run.language.len == 0) {
            out.appendSlice(allocator, run.text) catch return error.OutOfMemory;
            continue;
        }
        const value = try spoken(allocator, session, run.text, run.language);
        defer allocator.free(value);
        out.appendSlice(allocator, value) catch return error.OutOfMemory;
    }

    const joined = out.toOwnedSlice(allocator) catch return error.OutOfMemory;
    defer allocator.free(joined);
    return kokoro.clean(allocator, joined);
}

fn pinyin(
    allocator: std.mem.Allocator,
    session: *espeak.Session,
    input: []const u8,
) Error![]u8 {
    const normalized = try phoneme.pinyinNorm(allocator, input);
    defer allocator.free(normalized);
    if (trim(normalized).len == 0) {
        return allocator.dupe(u8, &.{}) catch error.OutOfMemory;
    }
    return spoken(allocator, session, normalized, "cmn");
}

fn spoken(
    allocator: std.mem.Allocator,
    session: *espeak.Session,
    input: []const u8,
    language: []const u8,
) Error![]u8 {
    const raw = try session.text(allocator, input, language);
    defer allocator.free(raw);
    const posted = try phoneme.post(allocator, raw, language);
    defer allocator.free(posted);
    return kokoro.clean(allocator, posted);
}

fn canonicalLanguage(allocator: std.mem.Allocator, raw: []const u8) Error!Lang {
    const value = trim(raw);
    if (value.len == 0 or eqlIgnore(value, "a") or eqlIgnore(value, "en-us") or
        eqlIgnore(value, "en_us") or eqlIgnore(value, "english"))
    {
        return .{ .value = "en-us" };
    }
    if (eqlIgnore(value, "b") or eqlIgnore(value, "en") or
        eqlIgnore(value, "en-gb") or eqlIgnore(value, "en_gb") or
        eqlIgnore(value, "british"))
    {
        return .{ .value = "en" };
    }
    if (eqlIgnore(value, "z") or eqlIgnore(value, "zh") or
        eqlIgnore(value, "zh-cn") or eqlIgnore(value, "zh_cn") or
        eqlIgnore(value, "cmn") or eqlIgnore(value, "mandarin"))
    {
        return .{ .value = "cmn" };
    }

    const owned = allocator.alloc(u8, value.len) catch return error.OutOfMemory;
    for (value, 0..) |byte, index| {
        owned[index] = std.ascii.toLower(byte);
    }
    return .{ .value = owned, .owned = owned };
}

fn trim(value: []const u8) []const u8 {
    return std.mem.trim(u8, value, " \t\r\n");
}

fn eqlIgnore(left: []const u8, right: []const u8) bool {
    return std.ascii.eqlIgnoreCase(left, right);
}

test "Kokoro phonemizer canonicalizes language aliases" {
    const allocator = std.testing.allocator;
    const lang = try canonicalLanguage(allocator, " English ");
    defer lang.deinit(allocator);
    try std.testing.expectEqualStrings("en-us", lang.value);
}
