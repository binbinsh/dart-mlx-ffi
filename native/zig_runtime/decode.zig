const std = @import("std");

pub const Error = error{
    InvalidInput,
    OutOfBounds,
};

pub fn argmax(
    data: []const f32,
    base: usize,
    item_count: usize,
    stride: usize,
    class_count: usize,
    out: []i32,
) Error!void {
    if (item_count > out.len or stride == 0 or class_count == 0 or class_count > stride) {
        return error.InvalidInput;
    }
    if (item_count == 0) {
        return;
    }
    const last_item = std.math.mul(usize, item_count - 1, stride) catch return error.OutOfBounds;
    const needed = std.math.add(usize, base, last_item) catch return error.OutOfBounds;
    const end = std.math.add(usize, needed, class_count) catch return error.OutOfBounds;
    if (end > data.len) {
        return error.OutOfBounds;
    }

    var item: usize = 0;
    while (item < item_count) : (item += 1) {
        const offset = base + item * stride;
        var best: usize = 0;
        var best_value = data[offset];
        var class: usize = 1;
        while (class < class_count) : (class += 1) {
            const value = data[offset + class];
            if (value > best_value) {
                best_value = value;
                best = class;
            }
        }
        out[item] = @intCast(best);
    }
}

fn bestClass(data: []const f32, offset: usize, class_count: usize) usize {
    var best: usize = 0;
    var best_value = data[offset];
    var class: usize = 1;
    while (class < class_count) : (class += 1) {
        const value = data[offset + class];
        if (value > best_value) {
            best_value = value;
            best = class;
        }
    }
    return best;
}

pub fn bioes(
    data: []const f32,
    base: usize,
    item_count: usize,
    stride: usize,
    class_count: usize,
    starts: []i32,
    ends: []i32,
) Error!usize {
    if (item_count > starts.len or item_count > ends.len or stride == 0 or class_count == 0 or class_count > stride) {
        return error.InvalidInput;
    }
    if (item_count == 0) {
        return 0;
    }
    const last_item = std.math.mul(usize, item_count - 1, stride) catch return error.OutOfBounds;
    const needed = std.math.add(usize, base, last_item) catch return error.OutOfBounds;
    const end = std.math.add(usize, needed, class_count) catch return error.OutOfBounds;
    if (end > data.len) {
        return error.OutOfBounds;
    }

    var span_count: usize = 0;
    var start: ?usize = null;
    var item: usize = 0;
    while (item < item_count) : (item += 1) {
        const id = bestClass(data, base + item * stride, class_count);
        switch (id) {
            4 => {
                starts[span_count] = @intCast(item);
                ends[span_count] = @intCast(item + 1);
                span_count += 1;
                start = null;
            },
            1 => start = item,
            3 => if (start) |span_start| {
                starts[span_count] = @intCast(span_start);
                ends[span_count] = @intCast(item + 1);
                span_count += 1;
                start = null;
            },
            0 => start = null,
            else => {},
        }
    }
    return span_count;
}

pub fn spanTypes(
    data: []const f32,
    base: usize,
    item_count: usize,
    stride: usize,
    class_count: usize,
    starts: []const i32,
    ends: []const i32,
    counts: []i32,
    out: []i32,
) Error!void {
    if (starts.len != ends.len or starts.len > out.len or class_count == 0 or stride == 0 or class_count > stride or class_count > counts.len) {
        return error.InvalidInput;
    }
    if (item_count == 0 or starts.len == 0) {
        return;
    }
    const last_item = std.math.mul(usize, item_count - 1, stride) catch return error.OutOfBounds;
    const needed = std.math.add(usize, base, last_item) catch return error.OutOfBounds;
    const end = std.math.add(usize, needed, class_count) catch return error.OutOfBounds;
    if (end > data.len) {
        return error.OutOfBounds;
    }

    for (starts, ends, 0..) |span_start_raw, span_end_raw, span_index| {
        if (span_start_raw < 0 or span_end_raw < span_start_raw) {
            return error.InvalidInput;
        }
        const span_start: usize = @intCast(span_start_raw);
        const span_end: usize = @intCast(span_end_raw);
        if (span_end > item_count) {
            return error.InvalidInput;
        }
        @memset(counts[0..class_count], 0);
        var item = span_start;
        while (item < span_end) : (item += 1) {
            const best = bestClass(data, base + item * stride, class_count);
            counts[best] += 1;
        }
        var best_type: usize = 0;
        var class: usize = 1;
        while (class < class_count) : (class += 1) {
            if (counts[class] > counts[best_type]) {
                best_type = class;
            }
        }
        out[span_index] = @intCast(best_type);
    }
}

pub fn activeBest(
    data: []const f32,
    offset: usize,
    count: usize,
    threshold: f64,
    out: []i32,
    best_out: *i32,
) Error!usize {
    if (count == 0 or count > out.len) {
        return error.InvalidInput;
    }
    const end = std.math.add(usize, offset, count) catch return error.OutOfBounds;
    if (end > data.len) {
        return error.OutOfBounds;
    }

    const best = bestClass(data, offset, count);
    best_out.* = @intCast(best);
    var active_count: usize = 0;
    var index: usize = 0;
    while (index < count) : (index += 1) {
        if (sigmoidActive(data[offset + index], threshold)) {
            out[active_count] = @intCast(index);
            active_count += 1;
        }
    }
    return active_count;
}

pub fn sigmoidSpans(
    data: []const f32,
    offset: usize,
    count: usize,
    final_end: usize,
    threshold: f64,
    starts: []i32,
    ends: []i32,
) Error!usize {
    if (count > starts.len or count > ends.len or final_end < count) {
        return error.InvalidInput;
    }
    const data_end = std.math.add(usize, offset, count) catch return error.OutOfBounds;
    if (data_end > data.len) {
        return error.OutOfBounds;
    }

    var span_count: usize = 0;
    var start: ?usize = null;
    var index: usize = 0;
    while (index < count) : (index += 1) {
        const active = sigmoidActive(data[offset + index], threshold);
        if (active and start == null) {
            start = index;
        } else if (!active and start != null) {
            starts[span_count] = @intCast(start.?);
            ends[span_count] = @intCast(index);
            span_count += 1;
            start = null;
        }
    }
    if (start != null) {
        starts[span_count] = @intCast(start.?);
        ends[span_count] = @intCast(final_end);
        span_count += 1;
    }
    return span_count;
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidInput => "Zig decoder received invalid logits metadata.",
        error.OutOfBounds => "Zig decoder logits metadata exceeded the input buffer.",
    };
}

fn sigmoidActive(value: f32, threshold: f64) bool {
    if (threshold <= 0.0) {
        return true;
    }
    if (threshold > 1.0) {
        return false;
    }
    const prob = 1.0 / (1.0 + @exp(-@as(f64, value)));
    return prob >= threshold;
}

test "decode argmax writes best class per item" {
    const data = [_]f32{
        0.1, 0.8, 0.2,
        0.7, 0.3, 0.1,
    };
    var out = [_]i32{ -1, -1 };

    try argmax(&data, 0, 2, 3, 3, &out);

    try std.testing.expectEqual(@as(i32, 1), out[0]);
    try std.testing.expectEqual(@as(i32, 0), out[1]);
}

test "decode argmax honors row base and class limit" {
    const data = [_]f32{
        9.0, 0.0, 0.0,
        0.0, 9.0, 0.0,
        0.3, 0.2, 8.0,
        0.1, 0.9, 7.0,
    };
    var out = [_]i32{ -1, -1 };

    try argmax(&data, 6, 2, 3, 2, &out);

    try std.testing.expectEqual(@as(i32, 0), out[0]);
    try std.testing.expectEqual(@as(i32, 1), out[1]);
}

test "decode bioes writes span boundaries" {
    const data = [_]f32{
        0.0, 9.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 9.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 9.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 9.0,
        9.0, 0.0, 0.0, 0.0, 0.0,
    };
    var starts = [_]i32{ -1, -1, -1, -1, -1 };
    var ends = [_]i32{ -1, -1, -1, -1, -1 };

    const count = try bioes(&data, 0, 5, 5, 5, &starts, &ends);

    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqualSlices(i32, &.{ 0, 3 }, starts[0..count]);
    try std.testing.expectEqualSlices(i32, &.{ 3, 4 }, ends[0..count]);
}

test "decode span types vote by argmax majority" {
    const data = [_]f32{
        0.0, 3.0, 1.0,
        0.0, 2.0, 1.0,
        0.0, 1.0, 4.0,
        5.0, 1.0, 0.0,
    };
    var counts = [_]i32{ 0, 0, 0 };
    var out = [_]i32{ -1, -1 };

    try spanTypes(
        &data,
        0,
        4,
        3,
        3,
        &.{ 0, 3 },
        &.{ 3, 4 },
        &counts,
        &out,
    );

    try std.testing.expectEqualSlices(i32, &.{ 1, 0 }, &out);
}

test "decode active best returns threshold ids and best id" {
    const data = [_]f32{ -2.0, 0.1, 3.0 };
    var out = [_]i32{ -1, -1, -1 };
    var best: i32 = -1;

    const count = try activeBest(&data, 0, 3, 0.5, &out, &best);

    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(i32, 2), best);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, out[0..count]);
}

test "decode sigmoid spans writes boundaries" {
    const data = [_]f32{ 2.0, 2.0, -2.0, 2.0 };
    var starts = [_]i32{ -1, -1, -1, -1 };
    var ends = [_]i32{ -1, -1, -1, -1 };

    const count = try sigmoidSpans(&data, 0, 4, 6, 0.5, &starts, &ends);

    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqualSlices(i32, &.{ 0, 3 }, starts[0..count]);
    try std.testing.expectEqualSlices(i32, &.{ 2, 6 }, ends[0..count]);
}
