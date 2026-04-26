const std = @import("std");
const policy = @import("policy.zig");

const sep: u8 = 0x1f;

pub const Entry = extern struct {
    path: [*c]const u8,
    kind: i32,
    text: [*c]const u8,
    int_value: i64,
    double_value: f64,
    bool_value: i32,
};

const string_kind: i32 = 1;
const int_kind: i32 = 2;
const bool_kind: i32 = 3;
const map_kind: i32 = 4;
const list_kind: i32 = 5;
const double_kind: i32 = 6;
const null_kind: i32 = 7;

const Error = error{OutOfMemory};

pub fn build(
    allocator: std.mem.Allocator,
    engine: i32,
    prefer_mask: i32,
    diagnostics: bool,
    num_threads: i32,
    metadata: []const Entry,
    backend: []const Entry,
) Error![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '{');
    var written = false;
    try appendFields(allocator, &out, &written, backend, &.{});
    try appendFields(allocator, &out, &written, metadata, backend);
    if (!hasRoot(backend, "accelerators") and !hasRoot(metadata, "accelerators")) {
        try beginField(allocator, &out, &written, "accelerators");
        try appendAccelerators(allocator, &out, prefer_mask);
    }
    if (!hasRoot(backend, "diagnostics") and !hasRoot(metadata, "diagnostics")) {
        try beginField(allocator, &out, &written, "diagnostics");
        try out.appendSlice(allocator, if (diagnostics) "true" else "false");
    }
    if (num_threads > 0 and !hasRoot(backend, "numThreads") and !hasRoot(metadata, "numThreads")) {
        try beginField(allocator, &out, &written, "numThreads");
        try appendInt(allocator, &out, num_threads);
    }
    if (!hasRoot(backend, "engine") and !hasRoot(metadata, "engine")) {
        try beginField(allocator, &out, &written, "engine");
        try appendString(allocator, &out, policy.engineName(engine));
    }
    try out.append(allocator, '}');
    return out.toOwnedSliceSentinel(allocator, 0) catch error.OutOfMemory;
}

fn appendFields(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    written: *bool,
    entries: []const Entry,
    skip: []const Entry,
) Error!void {
    for (entries, 0..) |entry, index| {
        const path = entryPath(entry) orelse continue;
        if (!isRoot(path) or hasRoot(skip, path)) {
            continue;
        }
        try beginField(allocator, out, written, path);
        try appendEntry(allocator, out, entries, index);
    }
}

fn appendEntry(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    entries: []const Entry,
    index: usize,
) Error!void {
    const entry = entries[index];
    switch (entry.kind) {
        string_kind => try appendString(allocator, out, entryText(entry)),
        int_kind => try appendInt64(allocator, out, entry.int_value),
        bool_kind => try out.appendSlice(allocator, if (entry.bool_value != 0) "true" else "false"),
        map_kind => try appendMap(allocator, out, entries, entryPath(entry) orelse ""),
        list_kind => try appendList(allocator, out, entries, entryPath(entry) orelse ""),
        double_kind => try appendDouble(allocator, out, entry.double_value),
        null_kind => try out.appendSlice(allocator, "null"),
        else => try out.appendSlice(allocator, "null"),
    }
}

fn appendMap(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    entries: []const Entry,
    parent: []const u8,
) Error!void {
    try out.append(allocator, '{');
    var written = false;
    for (entries, 0..) |entry, index| {
        const path = entryPath(entry) orelse continue;
        const child = directChild(parent, path) orelse continue;
        try beginField(allocator, out, &written, child);
        try appendEntry(allocator, out, entries, index);
    }
    try out.append(allocator, '}');
}

fn appendList(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    entries: []const Entry,
    parent: []const u8,
) Error!void {
    try out.append(allocator, '[');
    var written = false;
    for (entries, 0..) |entry, index| {
        const path = entryPath(entry) orelse continue;
        _ = directChild(parent, path) orelse continue;
        if (written) {
            try out.append(allocator, ',');
        } else {
            written = true;
        }
        try appendEntry(allocator, out, entries, index);
    }
    try out.append(allocator, ']');
}

fn directChild(parent: []const u8, path: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, path, parent) or path.len <= parent.len) {
        return null;
    }
    if (path[parent.len] != sep) {
        return null;
    }
    const rest = path[parent.len + 1 ..];
    if (rest.len == 0 or std.mem.indexOfScalar(u8, rest, sep) != null) {
        return null;
    }
    return rest;
}

fn hasRoot(entries: []const Entry, key: []const u8) bool {
    for (entries) |entry| {
        const path = entryPath(entry) orelse continue;
        if (std.mem.eql(u8, root(path), key)) {
            return true;
        }
    }
    return false;
}

fn root(path: []const u8) []const u8 {
    const index = std.mem.indexOfScalar(u8, path, sep) orelse return path;
    return path[0..index];
}

fn isRoot(path: []const u8) bool {
    return std.mem.indexOfScalar(u8, path, sep) == null;
}

fn entryPath(entry: Entry) ?[]const u8 {
    if (entry.path == null) {
        return null;
    }
    const path = std.mem.span(entry.path);
    return if (path.len == 0) null else path;
}

fn entryText(entry: Entry) []const u8 {
    if (entry.text == null) {
        return "";
    }
    return std.mem.span(entry.text);
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

fn appendAccelerators(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    mask: i32,
) Error!void {
    try out.append(allocator, '[');
    var written = false;
    if ((mask & policy.accel_ane) != 0) try appendStringItem(allocator, out, &written, "ane");
    if ((mask & policy.accel_gpu) != 0) try appendStringItem(allocator, out, &written, "gpu");
    if ((mask & policy.accel_npu) != 0) try appendStringItem(allocator, out, &written, "npu");
    if ((mask & policy.accel_cpu) != 0) try appendStringItem(allocator, out, &written, "cpu");
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

fn appendString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: []const u8) Error!void {
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

fn appendInt(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: i32) Error!void {
    try appendInt64(allocator, out, value);
}

fn appendInt64(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: i64) Error!void {
    const text = std.fmt.allocPrint(allocator, "{d}", .{value}) catch return error.OutOfMemory;
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn appendDouble(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: f64) Error!void {
    const text = std.fmt.allocPrint(allocator, "{d}", .{value}) catch return error.OutOfMemory;
    defer allocator.free(text);
    try out.appendSlice(allocator, text);
}

fn textEntry(path: [*c]const u8, value: [*c]const u8) Entry {
    return .{ .path = path, .kind = string_kind, .text = value, .int_value = 0, .double_value = 0, .bool_value = 0 };
}

test "runtime open options are assembled from typed entries" {
    const backend = [_]Entry{
        textEntry("provider", "cuda"),
        textEntry("zigRuntimeMode", "echo"),
    };
    const metadata = [_]Entry{
        textEntry("provider", "cpu"),
        .{ .path = "diagnostics", .kind = bool_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 },
    };
    const options = try build(
        std.testing.allocator,
        @intFromEnum(policy.Engine.onnx),
        policy.accel_gpu | policy.accel_cpu,
        true,
        4,
        &metadata,
        &backend,
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

test "runtime open options preserve nested entries" {
    const entries = [_]Entry{
        .{ .path = "items", .kind = list_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 },
        textEntry("items\x1f0", "a"),
        .{ .path = "items\x1f1", .kind = int_kind, .text = null, .int_value = 7, .double_value = 0, .bool_value = 0 },
    };
    const options = try build(std.testing.allocator, @intFromEnum(policy.Engine.onnx), 0, false, 0, &.{}, &entries);
    defer std.testing.allocator.free(options);
    try std.testing.expect(std.mem.indexOf(u8, options, "\"items\":[\"a\",7]") != null);
}
