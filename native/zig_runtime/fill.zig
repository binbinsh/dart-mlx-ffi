const std = @import("std");

pub const Error = error{ InvalidInput, InvalidUtf8 };

pub fn i64s(values: []i64, value: i64) void {
    for (values) |*item| {
        item.* = value;
    }
}

pub fn bytes(values: []u8, value: u8) void {
    @memset(values, value);
}

pub fn range(values: []u8, offset: usize, start: usize, end: usize, value: u8) Error!void {
    if (start > end or end > values.len or offset > values.len - end) {
        return error.InvalidInput;
    }
    @memset(values[offset + start .. offset + end], value);
}

pub fn ids(values: []u8, offset: usize, width: usize, positions: []const i32) Error!void {
    if (width > values.len or offset > values.len - width) {
        return error.InvalidInput;
    }
    const row = values[offset .. offset + width];
    @memset(row, 0);
    for (positions) |pos| {
        if (pos >= 0) {
            const idx: usize = @intCast(pos);
            if (idx < width) {
                row[idx] = 1;
            }
        }
    }
}

pub fn rows(
    values: []u8,
    offset: usize,
    width: usize,
    starts: []const i32,
    ends: []const i32,
    token_starts: ?[]const i32,
    token_ends: ?[]const i32,
) Error!void {
    if (starts.len != ends.len or width > values.len or starts.len > values.len / @max(width, 1)) {
        return error.InvalidInput;
    }
    const row_count = starts.len;
    const total = std.math.mul(usize, row_count, width) catch return error.InvalidInput;
    if (total > values.len or offset > values.len - total) {
        return error.InvalidInput;
    }
    if (token_starts != null or token_ends != null) {
        const ts = token_starts orelse return error.InvalidInput;
        const te = token_ends orelse return error.InvalidInput;
        if (ts.len != te.len) return error.InvalidInput;
        for (starts, ends, 0..) |start, end, idx| {
            try span(values, offset + idx * width, width, ts, te, @intCast(start), @intCast(end));
        }
        return;
    }
    for (starts, ends, 0..) |start_raw, end_raw, idx| {
        const row = values[offset + idx * width .. offset + (idx + 1) * width];
        @memset(row, 0);
        if (end_raw <= start_raw) continue;
        const start: usize = if (start_raw <= 0) 0 else @intCast(start_raw);
        const end_unclamped: usize = if (end_raw <= 0) 0 else @intCast(end_raw);
        const end = @min(width, end_unclamped);
        if (start < end) {
            @memset(row[start..end], 1);
        }
    }
}

pub fn candidates(
    values: []u8,
    offset: usize,
    width: usize,
    offsets: []const isize,
    positions: []const i32,
) Error!void {
    if (offsets.len == 0) return error.InvalidInput;
    const row_count = offsets.len - 1;
    const total = std.math.mul(usize, row_count, width) catch return error.InvalidInput;
    if (width > values.len or total > values.len or offset > values.len - total) {
        return error.InvalidInput;
    }
    for (0..row_count) |row_idx| {
        const start_raw = offsets[row_idx];
        const end_raw = offsets[row_idx + 1];
        if (start_raw < 0 or end_raw < start_raw) {
            return error.InvalidInput;
        }
        const start: usize = @intCast(start_raw);
        const end: usize = @intCast(end_raw);
        if (end > positions.len) return error.InvalidInput;
        if (start == end) continue;
        try ids(values, offset + row_idx * width, width, positions[start..end]);
    }
}

pub fn span(
    values: []u8,
    offset: usize,
    width: usize,
    starts: []const i32,
    ends: []const i32,
    span_start: isize,
    span_end: isize,
) Error!void {
    if (starts.len != ends.len or width > values.len or offset > values.len - width) {
        return error.InvalidInput;
    }
    const row = values[offset .. offset + width];
    @memset(row, 0);
    if (span_end <= span_start) {
        return;
    }
    for (starts, ends, 0..) |tok_start_raw, tok_end_raw, idx| {
        if (idx >= width) {
            break;
        }
        const tok_start: isize = @intCast(tok_start_raw);
        const tok_end: isize = @intCast(tok_end_raw);
        if (tok_end <= tok_start) {
            continue;
        }
        if (tok_end <= span_start or tok_start >= span_end) {
            continue;
        }
        row[idx] = 1;
    }
}

pub fn tokens(
    values: []i64,
    mask: []i64,
    offset: usize,
    width: usize,
    items: []const i64,
    pad_id: i64,
) Error!void {
    if (items.len > width or width > values.len or width > mask.len or offset > values.len - width or offset > mask.len - width) {
        return error.InvalidInput;
    }
    const value_row = values[offset .. offset + width];
    const mask_row = mask[offset .. offset + width];
    i64s(value_row, pad_id);
    i64s(mask_row, 0);
    for (items, 0..) |id, idx| {
        value_row[idx] = id;
        mask_row[idx] = if (id == pad_id) 0 else 1;
    }
}

pub fn seq(
    values: []i64,
    mask: []i64,
    offset: usize,
    width: usize,
    items: []const i64,
    pad_id: i64,
) Error!void {
    if (items.len > width or width > values.len or width > mask.len or offset > values.len - width or offset > mask.len - width) {
        return error.InvalidInput;
    }
    const value_row = values[offset .. offset + width];
    const mask_row = mask[offset .. offset + width];
    i64s(value_row, pad_id);
    i64s(mask_row, 0);
    for (items, 0..) |id, idx| {
        value_row[idx] = id;
        mask_row[idx] = 1;
    }
}

pub fn chars(
    values: []i64,
    mask: []i64,
    offset: usize,
    width: usize,
    text: []const u8,
    codes: []const i32,
    char_ids: []const i64,
    pad_id: i64,
    unk_id: i64,
) Error!usize {
    if (codes.len != char_ids.len or width > values.len or width > mask.len or offset > values.len - width or offset > mask.len - width) {
        return error.InvalidInput;
    }
    const value_row = values[offset .. offset + width];
    const mask_row = mask[offset .. offset + width];
    i64s(value_row, pad_id);
    i64s(mask_row, 0);

    var index: usize = 0;
    var count: usize = 0;
    while (index < text.len and count < width) {
        const rune = try nextRune(text, index);
        index += rune.len;
        value_row[count] = charId(rune.codepoint, codes, char_ids) orelse unk_id;
        mask_row[count] = 1;
        count += 1;
    }
    return count;
}

pub fn frame(values: []i64, items: []const i64, edge_id: i64) Error!void {
    if (items.len > values.len or values.len - items.len < 2) {
        return error.InvalidInput;
    }
    values[0] = edge_id;
    @memcpy(values[1 .. 1 + items.len], items);
    values[items.len + 1] = edge_id;
}

pub fn f32s(
    dst: []f32,
    dst_offset: usize,
    src: []const f32,
    src_offset: usize,
    count: usize,
) Error!void {
    if (count > dst.len or count > src.len or dst_offset > dst.len - count or src_offset > src.len - count) {
        return error.InvalidInput;
    }
    @memcpy(dst[dst_offset .. dst_offset + count], src[src_offset .. src_offset + count]);
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidInput => "Zig fill received an invalid output buffer.",
        error.InvalidUtf8 => "Zig fill received invalid UTF-8 text.",
    };
}

const Rune = struct {
    codepoint: u21,
    len: usize,
};

fn charId(codepoint: u21, codes: []const i32, char_ids: []const i64) ?i64 {
    const target: i32 = @intCast(@as(u32, codepoint));
    for (codes, 0..) |code, index| {
        if (code == target) {
            return char_ids[index];
        }
    }
    return null;
}

fn nextRune(input: []const u8, index: usize) Error!Rune {
    const first = input[index];
    if (first < 0x80) {
        return .{ .codepoint = @intCast(first), .len = 1 };
    }
    if ((first & 0xe0) == 0xc0) {
        if (index + 1 >= input.len or !isCont(input[index + 1])) return error.InvalidUtf8;
        const value: u32 = (@as(u32, first & 0x1f) << 6) |
            @as(u32, input[index + 1] & 0x3f);
        if (value < 0x80) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .len = 2 };
    }
    if ((first & 0xf0) == 0xe0) {
        if (index + 2 >= input.len or !isCont(input[index + 1]) or !isCont(input[index + 2])) {
            return error.InvalidUtf8;
        }
        const value: u32 = (@as(u32, first & 0x0f) << 12) |
            (@as(u32, input[index + 1] & 0x3f) << 6) |
            @as(u32, input[index + 2] & 0x3f);
        if (value < 0x800 or (value >= 0xd800 and value <= 0xdfff)) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .len = 3 };
    }
    if ((first & 0xf8) == 0xf0) {
        if (index + 3 >= input.len or
            !isCont(input[index + 1]) or
            !isCont(input[index + 2]) or
            !isCont(input[index + 3]))
        {
            return error.InvalidUtf8;
        }
        const value: u32 = (@as(u32, first & 0x07) << 18) |
            (@as(u32, input[index + 1] & 0x3f) << 12) |
            (@as(u32, input[index + 2] & 0x3f) << 6) |
            @as(u32, input[index + 3] & 0x3f);
        if (value < 0x10000 or value > 0x10ffff) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .len = 4 };
    }
    return error.InvalidUtf8;
}

fn isCont(byte: u8) bool {
    return (byte & 0xc0) == 0x80;
}

test "fill i64 values" {
    var values = [_]i64{ 1, 2, 3 };
    i64s(&values, 7);

    try std.testing.expectEqual(@as(i64, 7), values[0]);
    try std.testing.expectEqual(@as(i64, 7), values[1]);
    try std.testing.expectEqual(@as(i64, 7), values[2]);
}

test "fill byte values" {
    var values = [_]u8{ 1, 2, 3 };
    bytes(&values, 4);

    try std.testing.expectEqual(@as(u8, 4), values[0]);
    try std.testing.expectEqual(@as(u8, 4), values[1]);
    try std.testing.expectEqual(@as(u8, 4), values[2]);
}

test "fill byte range" {
    var values = [_]u8{ 0, 0, 0, 0, 0, 0 };
    try range(&values, 1, 2, 5, 1);

    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 1, 1, 1 }, &values);
}

test "fill id mask" {
    var values = [_]u8{ 1, 1, 1, 1, 1, 7 };
    try ids(&values, 0, 5, &.{ 3, -1, 6, 1 });

    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 0, 7 }, &values);
}

test "fill span rows direct and token mapped" {
    var direct = [_]u8{9} ** 10;
    try rows(&direct, 0, 5, &.{ 1, 2 }, &.{ 3, 8 }, null, null);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 1, 0, 0, 0, 0, 1, 1, 1 }, &direct);

    var token = [_]u8{9} ** 8;
    try rows(&token, 0, 4, &.{1}, &.{4}, &.{ 0, 1, 3, 5 }, &.{ 1, 3, 5, 6 });
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 1, 0, 9, 9, 9, 9 }, &token);
}

test "fill candidate rows" {
    var values = [_]u8{9} ** 8;
    try candidates(&values, 0, 4, &.{ 0, 2, 3 }, &.{ 1, 3, 2 });
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 0, 0, 1, 0 }, &values);

    try candidates(&values, 0, 4, &.{ 0, 0, 1 }, &.{2});
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 0, 1, 0, 0, 1, 0 }, &values);
}

test "fill span mask" {
    var values = [_]u8{ 9, 9, 9, 9, 7 };
    try span(
        &values,
        0,
        4,
        &.{ 0, 1, 3, 0 },
        &.{ 0, 3, 5, 0 },
        2,
        4,
    );

    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 1, 0, 7 }, &values);
}

test "fill token row and attention" {
    var values = [_]i64{ 9, 9, 9, 9, 7 };
    var mask = [_]i64{ 8, 8, 8, 8, 6 };
    try tokens(&values, &mask, 0, 4, &.{ 2, 11, 0, 0 }, 0);

    try std.testing.expectEqualSlices(i64, &.{ 2, 11, 0, 0, 7 }, &values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 0, 6 }, &mask);
}

test "fill sequence row and mask" {
    var values = [_]i64{ 9, 9, 9, 9, 7 };
    var mask = [_]i64{ 8, 8, 8, 8, 6 };
    try seq(&values, &mask, 0, 4, &.{ 4, 0 }, 0);

    try std.testing.expectEqualSlices(i64, &.{ 4, 0, 0, 0, 7 }, &values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 0, 0, 6 }, &mask);
}

test "fill char row from UTF-8 text" {
    var values = [_]i64{ 9, 9, 9, 9, 7 };
    var mask = [_]i64{ 8, 8, 8, 8, 6 };
    const codes = [_]i32{ 'A', 0x4f60 };
    const char_ids = [_]i64{ 2, 3 };
    const count = try chars(&values, &mask, 0, 4, "A你?", &codes, &char_ids, 0, 1);

    try std.testing.expectEqual(@as(usize, 3), count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 1, 0, 7 }, &values);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 0, 6 }, &mask);
}

test "fill framed i64 sequence" {
    var values = [_]i64{ 9, 9, 9, 9, 9 };
    try frame(&values, &.{ 12, 13, 14 }, 0);

    try std.testing.expectEqualSlices(i64, &.{ 0, 12, 13, 14, 0 }, &values);
}

test "fill copies f32 slice" {
    var dst = [_]f32{ 0, 0, 0, 9 };
    const src = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    try f32s(&dst, 0, &src, 1, 3);

    try std.testing.expectEqualSlices(f32, &.{ 2.5, 3.5, 4.5, 9 }, &dst);
}

test "fill validates ranges" {
    var values = [_]u8{ 0, 0, 0 };
    var i64_values = [_]i64{ 0, 0, 0 };
    var i64_mask = [_]i64{ 0, 0, 0 };
    var f32_values = [_]f32{ 0, 0, 0 };

    try std.testing.expectError(error.InvalidInput, range(&values, 0, 3, 2, 1));
    try std.testing.expectError(error.InvalidInput, range(&values, 2, 0, 2, 1));
    try std.testing.expectError(error.InvalidInput, ids(&values, 2, 2, &.{}));
    try std.testing.expectError(
        error.InvalidInput,
        span(&values, 0, 1, &.{0}, &.{}, 0, 1),
    );
    try std.testing.expectError(
        error.InvalidInput,
        tokens(&i64_values, &i64_mask, 2, 2, &.{1}, 0),
    );
    try std.testing.expectError(
        error.InvalidInput,
        seq(&i64_values, &i64_mask, 0, 2, &.{ 1, 2, 3 }, 0),
    );
    try std.testing.expectError(
        error.InvalidInput,
        frame(&i64_values, &.{ 1, 2 }, 0),
    );
    try std.testing.expectError(
        error.InvalidInput,
        f32s(&f32_values, 2, &f32_values, 0, 2),
    );
}
