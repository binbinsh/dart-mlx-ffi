const std = @import("std");
const policy = @import("policy.zig");

const Error = error{
    InvalidOptionsJson,
    OutOfMemory,
};

pub fn build(
    allocator: std.mem.Allocator,
    engine: i32,
    prefer_mask: i32,
    diagnostics: bool,
    num_threads: i32,
    metadata_json: ?[]const u8,
    backend_json: ?[]const u8,
) Error![:0]u8 {
    var metadata_parsed: ?std.json.Parsed(std.json.Value) = null;
    defer if (metadata_parsed) |*parsed| parsed.deinit();
    var backend_parsed: ?std.json.Parsed(std.json.Value) = null;
    defer if (backend_parsed) |*parsed| parsed.deinit();

    const metadata = try parseObject(allocator, metadata_json, &metadata_parsed);
    const backend = try parseObject(allocator, backend_json, &backend_parsed);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '{');
    var written = false;
    try appendObjectFields(allocator, &out, &written, backend, null, null);
    try appendObjectFields(allocator, &out, &written, metadata, backend, null);
    if (!hasKey(backend, "accelerators") and !hasKey(metadata, "accelerators")) {
        try beginField(allocator, &out, &written, "accelerators");
        try appendAccelerators(allocator, &out, prefer_mask);
    }
    if (!hasKey(backend, "diagnostics") and !hasKey(metadata, "diagnostics")) {
        try beginField(allocator, &out, &written, "diagnostics");
        try out.appendSlice(allocator, if (diagnostics) "true" else "false");
    }
    if (num_threads > 0 and !hasKey(backend, "numThreads") and !hasKey(metadata, "numThreads")) {
        try beginField(allocator, &out, &written, "numThreads");
        try appendInt(allocator, &out, num_threads);
    }
    if (!hasKey(backend, "engine") and !hasKey(metadata, "engine")) {
        try beginField(allocator, &out, &written, "engine");
        try appendString(allocator, &out, policy.engineName(engine));
    }
    try out.append(allocator, '}');
    return out.toOwnedSliceSentinel(allocator, 0) catch error.OutOfMemory;
}

fn parseObject(
    allocator: std.mem.Allocator,
    text: ?[]const u8,
    parsed_out: *?std.json.Parsed(std.json.Value),
) Error!?std.json.ObjectMap {
    const raw = text orelse return null;
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) {
        return null;
    }
    parsed_out.* = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        trimmed,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return error.InvalidOptionsJson;
    return switch (parsed_out.*.?.value) {
        .object => |object| object,
        else => error.InvalidOptionsJson,
    };
}

fn appendObjectFields(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    written: *bool,
    object: ?std.json.ObjectMap,
    skip_a: ?std.json.ObjectMap,
    skip_b: ?std.json.ObjectMap,
) Error!void {
    const map = object orelse return;
    var it = map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (hasKey(skip_a, key) or hasKey(skip_b, key)) {
            continue;
        }
        try beginField(allocator, out, written, key);
        try appendValue(allocator, out, entry.value_ptr.*);
    }
}

fn hasKey(object: ?std.json.ObjectMap, key: []const u8) bool {
    const map = object orelse return false;
    return map.get(key) != null;
}

fn beginField(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    written: *bool,
    key: []const u8,
) Error!void {
    if (written.*) {
        try out.append(allocator, ',');
    } else {
        written.* = true;
    }
    try appendString(allocator, out, key);
    try out.append(allocator, ':');
}

fn appendValue(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: std.json.Value,
) Error!void {
    const text = std.json.Stringify.valueAlloc(allocator, value, .{}) catch return error.OutOfMemory;
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn appendAccelerators(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    mask: i32,
) Error!void {
    try out.append(allocator, '[');
    var written = false;
    if ((mask & policy.accel_ane) != 0) {
        try appendStringItem(allocator, out, &written, "ane");
    }
    if ((mask & policy.accel_gpu) != 0) {
        try appendStringItem(allocator, out, &written, "gpu");
    }
    if ((mask & policy.accel_npu) != 0) {
        try appendStringItem(allocator, out, &written, "npu");
    }
    if ((mask & policy.accel_cpu) != 0) {
        try appendStringItem(allocator, out, &written, "cpu");
    }
    try out.append(allocator, ']');
}

fn appendStringItem(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    written: *bool,
    value: []const u8,
) Error!void {
    if (written.*) {
        try out.append(allocator, ',');
    } else {
        written.* = true;
    }
    try appendString(allocator, out, value);
}

fn appendString(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: []const u8,
) Error!void {
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, byte),
        }
    }
    try out.append(allocator, '"');
}

fn appendInt(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: i32,
) Error!void {
    var buffer: [16]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch return error.OutOfMemory;
    try out.appendSlice(allocator, text);
}

test "runtime open options JSON is assembled in Zig" {
    const options = try build(
        std.testing.allocator,
        @intFromEnum(policy.Engine.onnx),
        policy.accel_gpu | policy.accel_cpu,
        true,
        4,
        "{\"diagnostics\":false,\"provider\":\"cpu\"}",
        "{\"provider\":\"cuda\",\"zigRuntimeMode\":\"echo\"}",
    );
    defer std.testing.allocator.free(options);
    const text = options[0..options.len];
    try std.testing.expect(std.mem.indexOf(u8, text, "\"provider\":\"cuda\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"provider\":\"cpu\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"diagnostics\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"diagnostics\":true") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"accelerators\":[\"gpu\",\"cpu\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"numThreads\":4") != null);
}
