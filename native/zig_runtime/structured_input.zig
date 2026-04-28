const std = @import("std");
const fill = @import("fill.zig");
const target = @import("target.zig");

pub const Error = fill.Error;

pub fn reset(
    input_ids: []i64,
    token_pad_id: i64,
    attention: []i64,
    char_ids: []i64,
    char_pad_id: i64,
    char_mask: []i64,
    homograph_targets: []u8,
    homograph_candidates: []u8,
    polyphone_targets: []u8,
    polyphone_candidates: []u8,
) void {
    fill.i64s(input_ids, token_pad_id);
    fill.i64s(attention, 0);
    fill.i64s(char_ids, char_pad_id);
    fill.i64s(char_mask, 0);
    fill.bytes(homograph_targets, 0);
    fill.bytes(homograph_candidates, 1);
    fill.bytes(polyphone_targets, 0);
    fill.bytes(polyphone_candidates, 1);
}

pub fn targets(
    target_values: []u8,
    target_offset: usize,
    target_width: usize,
    candidate_values: []u8,
    candidate_offset: usize,
    candidate_width: usize,
    starts: []const i32,
    ends: []const i32,
    token_starts: ?[]const i32,
    token_ends: ?[]const i32,
    candidate_offsets: []const isize,
    candidate_ids: []const i32,
) Error!void {
    if (candidate_offsets.len != starts.len + 1) {
        return error.InvalidInput;
    }
    try fill.rows(
        target_values,
        target_offset,
        target_width,
        starts,
        ends,
        token_starts,
        token_ends,
    );
    try fill.candidates(
        candidate_values,
        candidate_offset,
        candidate_width,
        candidate_offsets,
        candidate_ids,
    );
}

pub fn matchTargets(
    target_values: []u8,
    target_offset: usize,
    target_width: usize,
    candidate_values: []u8,
    candidate_offset: usize,
    candidate_width: usize,
    matches: []const target.Match,
    token_starts: ?[]const i32,
    token_ends: ?[]const i32,
) Error!void {
    if (matches.len == 0) {
        return;
    }
    if (target_width == 0 or candidate_width == 0) {
        return error.InvalidInput;
    }
    if (token_starts != null or token_ends != null) {
        const starts = token_starts orelse return error.InvalidInput;
        const ends = token_ends orelse return error.InvalidInput;
        if (starts.len != ends.len) return error.InvalidInput;
    }
    const target_total = std.math.mul(usize, matches.len, target_width) catch return error.InvalidInput;
    const candidate_total = std.math.mul(usize, matches.len, candidate_width) catch return error.InvalidInput;
    if (target_width > target_values.len or target_total > target_values.len or target_offset > target_values.len - target_total or
        candidate_width > candidate_values.len or candidate_total > candidate_values.len or candidate_offset > candidate_values.len - candidate_total)
    {
        return error.InvalidInput;
    }

    for (matches, 0..) |item, row_index| {
        const target_row = target_values[target_offset + row_index * target_width .. target_offset + (row_index + 1) * target_width];
        if (token_starts) |starts| {
            writeTokenSpan(target_row, starts, token_ends.?, item.start, item.end);
        } else {
            writeDirectSpan(target_row, item.start, item.end);
        }

        const candidate_row = candidate_values[candidate_offset + row_index * candidate_width .. candidate_offset + (row_index + 1) * candidate_width];
        try writeCandidateIds(candidate_row, item.ids, item.id_count);
    }
}

fn writeDirectSpan(row: []u8, start_raw: i32, end_raw: i32) void {
    @memset(row, 0);
    if (end_raw <= start_raw) {
        return;
    }
    const start: usize = if (start_raw <= 0) 0 else @intCast(start_raw);
    const end_unclamped: usize = if (end_raw <= 0) 0 else @intCast(end_raw);
    const end = @min(row.len, end_unclamped);
    if (start < end) {
        @memset(row[start..end], 1);
    }
}

fn writeTokenSpan(row: []u8, starts: []const i32, ends: []const i32, span_start_raw: i32, span_end_raw: i32) void {
    @memset(row, 0);
    if (span_end_raw <= span_start_raw) {
        return;
    }
    const span_start: isize = @intCast(span_start_raw);
    const span_end: isize = @intCast(span_end_raw);
    for (starts, ends, 0..) |tok_start_raw, tok_end_raw, idx| {
        if (idx >= row.len) {
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

fn writeCandidateIds(row: []u8, ids: [*c]const i32, id_count: isize) Error!void {
    if (id_count < 0 or (id_count > 0 and ids == null)) {
        return error.InvalidInput;
    }
    if (id_count == 0) {
        return;
    }
    @memset(row, 0);
    for (ids[0..@intCast(id_count)]) |id| {
        if (id >= 0) {
            const index: usize = @intCast(id);
            if (index < row.len) {
                row[index] = 1;
            }
        }
    }
}

test "structured input reset fills all buffers" {
    var input_ids = [_]i64{ 9, 9, 9 };
    var attention = [_]i64{ 8, 8, 8 };
    var char_ids = [_]i64{ 7, 7 };
    var char_mask = [_]i64{ 6, 6 };
    var homograph_targets = [_]u8{ 5, 5 };
    var homograph_candidates = [_]u8{ 4, 4 };
    var polyphone_targets = [_]u8{ 3, 3 };
    var polyphone_candidates = [_]u8{ 2, 2 };

    reset(
        &input_ids,
        -1,
        &attention,
        &char_ids,
        -2,
        &char_mask,
        &homograph_targets,
        &homograph_candidates,
        &polyphone_targets,
        &polyphone_candidates,
    );

    try std.testing.expectEqualSlices(i64, &.{ -1, -1, -1 }, &input_ids);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0 }, &attention);
    try std.testing.expectEqualSlices(i64, &.{ -2, -2 }, &char_ids);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &char_mask);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0 }, &homograph_targets);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1 }, &homograph_candidates);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0 }, &polyphone_targets);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1 }, &polyphone_candidates);
}

test "structured input fills target and candidate rows" {
    var target_values = [_]u8{9} ** 8;
    var candidate_values = [_]u8{9} ** 6;

    try targets(
        &target_values,
        0,
        4,
        &candidate_values,
        0,
        3,
        &.{ 1, 2 },
        &.{ 3, 7 },
        null,
        null,
        &.{ 0, 2, 3 },
        &.{ 0, 2, 1 },
    );

    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 1, 0, 0, 0, 1, 1 }, &target_values);
    try std.testing.expectEqualSlices(u8, &.{ 1, 0, 1, 0, 1, 0 }, &candidate_values);
}

test "structured input fills rows from target matches" {
    const ids_a = [_]i32{ 0, 2 };
    const ids_b = [_]i32{1};
    const matches = [_]target.Match{
        .{ .start = 1, .end = 3, .index = 0, .ids = &ids_a, .id_count = ids_a.len },
        .{ .start = 2, .end = 7, .index = 1, .ids = &ids_b, .id_count = ids_b.len },
    };
    var target_values = [_]u8{9} ** 8;
    var candidate_values = [_]u8{9} ** 6;

    try matchTargets(
        &target_values,
        0,
        4,
        &candidate_values,
        0,
        3,
        &matches,
        null,
        null,
    );

    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 1, 0, 0, 0, 1, 1 }, &target_values);
    try std.testing.expectEqualSlices(u8, &.{ 1, 0, 1, 0, 1, 0 }, &candidate_values);
}
