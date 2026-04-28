const std = @import("std");
const abi = @import("abi.zig");
const espeak = @import("espeak.zig");
const kok_phon = @import("kok_phon.zig");

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

fn cStringOpt(value: [*c]const u8) ?[]const u8 {
    if (value == null) {
        return null;
    }
    const len = std.mem.len(value);
    if (len == 0) {
        return null;
    }
    return value[0..len];
}

fn cString(value: [*c]const u8) espeak.Error![]const u8 {
    if (value == null) {
        return error.InvalidInput;
    }
    return value[0..std.mem.len(value)];
}

export fn dinf_esp_new(
    library_path: [*c]const u8,
    data_path: [*c]const u8,
    voice: [*c]const u8,
    phoneme_mode: i32,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    const session = espeak.create(
        std.heap.c_allocator,
        std.Io.Threaded.global_single_threaded.io(),
        cStringOpt(library_path),
        cStringOpt(data_path),
        cString(voice) catch |err| {
            setError(error_out, espeak.errorMessage(err));
            return null;
        },
        phoneme_mode,
    ) catch |err| {
        setError(error_out, espeak.errorMessage(err));
        return null;
    };
    return session;
}

export fn dinf_esp_free(handle: ?*anyopaque) void {
    const session = espeak.fromHandle(handle) catch return;
    espeak.destroy(std.heap.c_allocator, session);
}

export fn dinf_esp_text(
    handle: ?*anyopaque,
    text: [*c]const u8,
    voice: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    const session = espeak.fromHandle(handle) catch |err| {
        setError(error_out, espeak.errorMessage(err));
        return null;
    };
    const out = session.text(
        std.heap.c_allocator,
        cString(text) catch |err| {
            setError(error_out, espeak.errorMessage(err));
            return null;
        },
        cStringOpt(voice),
    ) catch |err| {
        setError(error_out, espeak.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) std.heap.c_allocator.free(out);
    return abi.copyString(out);
}

fn kokoroCall(
    handle: ?*anyopaque,
    text: [*c]const u8,
    language: [*c]const u8,
    error_out: ?*[*c]u8,
    comptime func: fn (std.mem.Allocator, *espeak.Session, []const u8, []const u8) kok_phon.Error![]u8,
) [*c]u8 {
    const session = espeak.fromHandle(handle) catch |err| {
        setError(error_out, espeak.errorMessage(err));
        return null;
    };
    const out = func(
        std.heap.c_allocator,
        session,
        cString(text) catch |err| {
            setError(error_out, espeak.errorMessage(err));
            return null;
        },
        cString(language) catch |err| {
            setError(error_out, espeak.errorMessage(err));
            return null;
        },
    ) catch |err| {
        setError(error_out, kok_phon.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) std.heap.c_allocator.free(out);
    return abi.copyString(out);
}

export fn dinf_esp_kok_text(
    handle: ?*anyopaque,
    text_value: [*c]const u8,
    language: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return kokoroCall(handle, text_value, language, error_out, kok_phon.text);
}

export fn dinf_esp_kok_ssml(
    handle: ?*anyopaque,
    ssml: [*c]const u8,
    language: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return kokoroCall(handle, ssml, language, error_out, kok_phon.ssml);
}
