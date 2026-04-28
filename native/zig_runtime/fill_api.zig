const std = @import("std");
const abi = @import("abi.zig");
const fill = @import("fill.zig");

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

export fn dinf_fill_chars_i64(
    values: [*c]i64,
    mask: [*c]i64,
    len: isize,
    offset: isize,
    width: isize,
    text: [*c]const u8,
    codes: [*c]const i32,
    ids: [*c]const i64,
    vocab_count: isize,
    pad_id: i64,
    unk_id: i64,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const out = count_out orelse {
        setError(error_out, fill.errorMessage(error.InvalidInput));
        return 1;
    };
    out.* = 0;
    if (len < 0 or offset < 0 or width < 0 or vocab_count < 0 or text == null or
        (len > 0 and (values == null or mask == null)) or
        (vocab_count > 0 and (codes == null or ids == null)))
    {
        setError(error_out, fill.errorMessage(error.InvalidInput));
        return 1;
    }
    var empty_values: [0]i64 = .{};
    const value_items = if (len == 0) empty_values[0..] else values[0..@intCast(len)];
    var empty_mask: [0]i64 = .{};
    const mask_items = if (len == 0) empty_mask[0..] else mask[0..@intCast(len)];
    const code_items: []const i32 = if (vocab_count == 0) &.{} else codes[0..@intCast(vocab_count)];
    const id_items: []const i64 = if (vocab_count == 0) &.{} else ids[0..@intCast(vocab_count)];
    const text_items = text[0..std.mem.len(text)];
    const count = fill.chars(
        value_items,
        mask_items,
        @intCast(offset),
        @intCast(width),
        text_items,
        code_items,
        id_items,
        pad_id,
        unk_id,
    ) catch |err| {
        setError(error_out, fill.errorMessage(err));
        return 1;
    };
    out.* = @intCast(count);
    return 0;
}
