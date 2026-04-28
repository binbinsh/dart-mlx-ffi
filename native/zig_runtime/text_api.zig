const std = @import("std");
const abi = @import("abi.zig");
const text = @import("text.zig");

const TextTagAbi = extern struct {
    start: i32,
    end: i32,
    kind: i32,
    value: [*c]const u8,
};

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

export fn dinf_text_ssml(
    input: [*c]const u8,
    tags: [*c]const TextTagAbi,
    tag_count: isize,
    emotion: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    if (input == null or tag_count < 0) {
        setError(error_out, text.errorMessage(error.InvalidInput));
        return null;
    }
    if (tag_count > 0 and tags == null) {
        setError(error_out, text.errorMessage(error.InvalidInput));
        return null;
    }

    const count: usize = @intCast(tag_count);
    const allocator = std.heap.c_allocator;
    const items = allocator.alloc(text.SsmlTag, count) catch {
        setError(error_out, text.errorMessage(error.OutOfMemory));
        return null;
    };
    defer allocator.free(items);

    var used: usize = 0;
    var index: usize = 0;
    while (index < count) : (index += 1) {
        const source = tags[index];
        const kind: text.SsmlKind = switch (source.kind) {
            1 => .emphasis,
            2 => .phoneme,
            3 => .sub,
            else => continue,
        };
        items[used] = .{
            .start = source.start,
            .end = source.end,
            .kind = kind,
            .value = if (source.value == null) "" else source.value[0..std.mem.len(source.value)],
        };
        used += 1;
    }

    const emotion_slice: ?[]const u8 = if (emotion == null)
        null
    else
        emotion[0..std.mem.len(emotion)];
    const out = text.composeSsml(
        allocator,
        input[0..std.mem.len(input)],
        items[0..used],
        emotion_slice,
    ) catch |err| {
        setError(error_out, text.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) allocator.free(out);
    return abi.copyString(out);
}

export fn dinf_text_strip_ssml(
    input: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return textCall(input, error_out, text.stripSsml);
}

export fn dinf_text_tn_en(
    input: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return textCall(input, error_out, text.tnEn);
}

export fn dinf_text_tn_zh(
    input: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return textCall(input, error_out, text.tnZh);
}

export fn dinf_text_has_zh(input: [*c]const u8) i32 {
    if (input == null) {
        return 0;
    }
    return if (text.hasZh(input[0..std.mem.len(input)])) 1 else 0;
}

export fn dinf_text_norm_spans(
    input: [*c]const u8,
    starts: [*c]const i32,
    ends: [*c]const i32,
    count: isize,
    out_starts: [*c]i32,
    out_ends: [*c]i32,
    out_count: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const written = out_count orelse {
        setError(error_out, text.errorMessage(error.InvalidInput));
        return 1;
    };
    written.* = 0;
    if (input == null or count < 0 or
        (count > 0 and (starts == null or ends == null or out_starts == null or out_ends == null)))
    {
        setError(error_out, text.errorMessage(error.InvalidInput));
        return 1;
    }
    const len: usize = @intCast(count);
    const start_items: []const i32 = if (len == 0) &.{} else starts[0..len];
    const end_items: []const i32 = if (len == 0) &.{} else ends[0..len];
    const out_start_items: []i32 = if (len == 0) &.{} else out_starts[0..len];
    const out_end_items: []i32 = if (len == 0) &.{} else out_ends[0..len];
    const produced = text.normSpans(
        input[0..std.mem.len(input)],
        start_items,
        end_items,
        out_start_items,
        out_end_items,
    ) catch |err| {
        setError(error_out, text.errorMessage(err));
        return 1;
    };
    written.* = @intCast(produced);
    return 0;
}

export fn dinf_text_select_tn(
    input: [*c]const u8,
    en_starts: [*c]const i32,
    en_ends: [*c]const i32,
    en_count: isize,
    zh_starts: [*c]const i32,
    zh_ends: [*c]const i32,
    zh_count: isize,
    out_sources: [*c]i32,
    out_indices: [*c]i32,
    out_count: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const written = out_count orelse {
        setError(error_out, text.errorMessage(error.InvalidInput));
        return 1;
    };
    written.* = 0;
    if (input == null or en_count < 0 or zh_count < 0 or
        (en_count > 0 and (en_starts == null or en_ends == null)) or
        (zh_count > 0 and (zh_starts == null or zh_ends == null)) or
        (en_count + zh_count > 0 and (out_sources == null or out_indices == null)))
    {
        setError(error_out, text.errorMessage(error.InvalidInput));
        return 1;
    }
    const en_len: usize = @intCast(en_count);
    const zh_len: usize = @intCast(zh_count);
    const out_len = en_len + zh_len;
    const en_start_items: []const i32 = if (en_len == 0) &.{} else en_starts[0..en_len];
    const en_end_items: []const i32 = if (en_len == 0) &.{} else en_ends[0..en_len];
    const zh_start_items: []const i32 = if (zh_len == 0) &.{} else zh_starts[0..zh_len];
    const zh_end_items: []const i32 = if (zh_len == 0) &.{} else zh_ends[0..zh_len];
    const source_items: []i32 = if (out_len == 0) &.{} else out_sources[0..out_len];
    const index_items: []i32 = if (out_len == 0) &.{} else out_indices[0..out_len];
    const produced = text.selectTn(
        input[0..std.mem.len(input)],
        en_start_items,
        en_end_items,
        zh_start_items,
        zh_end_items,
        source_items,
        index_items,
    ) catch |err| {
        setError(error_out, text.errorMessage(err));
        return 1;
    };
    written.* = @intCast(produced);
    return 0;
}

fn textCall(
    input: [*c]const u8,
    error_out: ?*[*c]u8,
    comptime func: fn (std.mem.Allocator, []const u8) text.Error![]u8,
) [*c]u8 {
    if (input == null) {
        setError(error_out, text.errorMessage(error.InvalidInput));
        return null;
    }
    const out = func(
        std.heap.c_allocator,
        input[0..std.mem.len(input)],
    ) catch |err| {
        setError(error_out, text.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) std.heap.c_allocator.free(out);
    return abi.copyString(out);
}
