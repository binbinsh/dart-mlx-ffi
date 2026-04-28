const abi = @import("abi.zig");
const decode = @import("decode.zig");

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

fn dataSlice(data: [*c]const f32, len: isize, error_out: ?*[*c]u8) ?[]const f32 {
    if (len < 0 or (len > 0 and data == null)) {
        setError(error_out, decode.errorMessage(error.InvalidInput));
        return null;
    }
    return if (len == 0) &.{} else data[0..@intCast(len)];
}

export fn dinf_dec_argmax(
    data: [*c]const f32,
    data_len: isize,
    base: isize,
    item_count: isize,
    stride: isize,
    class_count: isize,
    out: [*c]i32,
    error_out: ?*[*c]u8,
) i32 {
    if (base < 0 or item_count < 0 or stride < 0 or class_count < 0 or (item_count > 0 and out == null)) {
        setError(error_out, decode.errorMessage(error.InvalidInput));
        return 1;
    }
    const values = dataSlice(data, data_len, error_out) orelse return 1;
    var empty: [0]i32 = .{};
    const output = if (item_count == 0) empty[0..] else out[0..@intCast(item_count)];
    decode.argmax(
        values,
        @intCast(base),
        @intCast(item_count),
        @intCast(stride),
        @intCast(class_count),
        output,
    ) catch |err| {
        setError(error_out, decode.errorMessage(err));
        return 1;
    };
    return 0;
}

export fn dinf_dec_bioes(
    data: [*c]const f32,
    data_len: isize,
    base: isize,
    item_count: isize,
    stride: isize,
    class_count: isize,
    starts: [*c]i32,
    ends: [*c]i32,
    span_count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    if (base < 0 or item_count < 0 or stride < 0 or class_count < 0 or
        (item_count > 0 and (starts == null or ends == null)) or span_count_out == null)
    {
        setError(error_out, decode.errorMessage(error.InvalidInput));
        return 1;
    }
    const values = dataSlice(data, data_len, error_out) orelse return 1;
    var empty_starts: [0]i32 = .{};
    var empty_ends: [0]i32 = .{};
    const start_items = if (item_count == 0) empty_starts[0..] else starts[0..@intCast(item_count)];
    const end_items = if (item_count == 0) empty_ends[0..] else ends[0..@intCast(item_count)];
    const span_count = decode.bioes(
        values,
        @intCast(base),
        @intCast(item_count),
        @intCast(stride),
        @intCast(class_count),
        start_items,
        end_items,
    ) catch |err| {
        setError(error_out, decode.errorMessage(err));
        return 1;
    };
    span_count_out.?.* = @intCast(span_count);
    return 0;
}

export fn dinf_dec_span_types(
    data: [*c]const f32,
    data_len: isize,
    base: isize,
    item_count: isize,
    stride: isize,
    class_count: isize,
    starts: [*c]const i32,
    ends: [*c]const i32,
    span_count: isize,
    counts: [*c]i32,
    out: [*c]i32,
    error_out: ?*[*c]u8,
) i32 {
    if (base < 0 or item_count < 0 or stride < 0 or class_count < 0 or span_count < 0 or
        (span_count > 0 and (starts == null or ends == null or out == null)) or
        (class_count > 0 and counts == null))
    {
        setError(error_out, decode.errorMessage(error.InvalidInput));
        return 1;
    }
    const values = dataSlice(data, data_len, error_out) orelse return 1;
    const start_items: []const i32 = if (span_count == 0) &.{} else starts[0..@intCast(span_count)];
    const end_items: []const i32 = if (span_count == 0) &.{} else ends[0..@intCast(span_count)];
    var empty_counts: [0]i32 = .{};
    const count_items = if (class_count == 0) empty_counts[0..] else counts[0..@intCast(class_count)];
    var empty_out: [0]i32 = .{};
    const out_items = if (span_count == 0) empty_out[0..] else out[0..@intCast(span_count)];
    decode.spanTypes(
        values,
        @intCast(base),
        @intCast(item_count),
        @intCast(stride),
        @intCast(class_count),
        start_items,
        end_items,
        count_items,
        out_items,
    ) catch |err| {
        setError(error_out, decode.errorMessage(err));
        return 1;
    };
    return 0;
}

export fn dinf_dec_active(
    data: [*c]const f32,
    data_len: isize,
    offset: isize,
    count: isize,
    threshold: f64,
    out: [*c]i32,
    active_count_out: ?*isize,
    best_out: ?*i32,
    error_out: ?*[*c]u8,
) i32 {
    if (offset < 0 or count <= 0 or out == null or active_count_out == null or best_out == null) {
        setError(error_out, decode.errorMessage(error.InvalidInput));
        return 1;
    }
    const values = dataSlice(data, data_len, error_out) orelse return 1;
    const output = out[0..@intCast(count)];
    const active_count = decode.activeBest(
        values,
        @intCast(offset),
        @intCast(count),
        threshold,
        output,
        best_out.?,
    ) catch |err| {
        setError(error_out, decode.errorMessage(err));
        return 1;
    };
    active_count_out.?.* = @intCast(active_count);
    return 0;
}

export fn dinf_dec_spans(
    data: [*c]const f32,
    data_len: isize,
    offset: isize,
    count: isize,
    final_end: isize,
    threshold: f64,
    starts: [*c]i32,
    ends: [*c]i32,
    span_count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    if (offset < 0 or count < 0 or final_end < 0 or
        (count > 0 and (starts == null or ends == null)) or span_count_out == null)
    {
        setError(error_out, decode.errorMessage(error.InvalidInput));
        return 1;
    }
    const values = dataSlice(data, data_len, error_out) orelse return 1;
    var empty_starts: [0]i32 = .{};
    var empty_ends: [0]i32 = .{};
    const start_items = if (count == 0) empty_starts[0..] else starts[0..@intCast(count)];
    const end_items = if (count == 0) empty_ends[0..] else ends[0..@intCast(count)];
    const span_count = decode.sigmoidSpans(
        values,
        @intCast(offset),
        @intCast(count),
        @intCast(final_end),
        threshold,
        start_items,
        end_items,
    ) catch |err| {
        setError(error_out, decode.errorMessage(err));
        return 1;
    };
    span_count_out.?.* = @intCast(span_count);
    return 0;
}
