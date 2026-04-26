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

pub fn merge(
    allocator: std.mem.Allocator,
    engine: i32,
    prefer_mask: i32,
    diagnostics: bool,
    num_threads: i32,
    metadata: []const Entry,
    backend: []const Entry,
) Error![]Entry {
    var out: std.ArrayList(Entry) = .empty;
    errdefer out.deinit(allocator);
    try appendEntries(allocator, &out, backend, &.{});
    try appendEntries(allocator, &out, metadata, backend);
    if (!hasRoot(backend, "accelerators") and !hasRoot(metadata, "accelerators")) {
        try appendAccelerators(allocator, &out, prefer_mask);
    }
    if (!hasRoot(backend, "diagnostics") and !hasRoot(metadata, "diagnostics")) {
        try out.append(allocator, boolEntry("diagnostics", diagnostics));
    }
    if (num_threads > 0 and !hasRoot(backend, "numThreads") and !hasRoot(metadata, "numThreads")) {
        try out.append(allocator, intEntry("numThreads", num_threads));
    }
    if (!hasRoot(backend, "engine") and !hasRoot(metadata, "engine")) {
        try out.append(allocator, textEntry("engine", policy.engineName(engine).ptr));
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

pub fn textEquals(
    metadata: []const Entry,
    backend: []const Entry,
    key: []const u8,
    expected: []const u8,
) bool {
    if (rootValue(backend, key)) |entry| {
        return entry.kind == string_kind and std.mem.eql(u8, entryText(entry), expected);
    }
    if (hasRoot(backend, key)) {
        return false;
    }
    if (rootValue(metadata, key)) |entry| {
        return entry.kind == string_kind and std.mem.eql(u8, entryText(entry), expected);
    }
    return false;
}

fn appendEntries(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(Entry),
    entries: []const Entry,
    skip: []const Entry,
) Error!void {
    for (entries) |entry| {
        const path = entryPath(entry) orelse continue;
        if (hasRoot(skip, root(path))) {
            continue;
        }
        try out.append(allocator, entry);
    }
}

fn appendAccelerators(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(Entry),
    mask: i32,
) Error!void {
    try out.append(allocator, kindEntry("accelerators", list_kind));
    var index: usize = 0;
    if ((mask & policy.accel_ane) != 0) {
        try out.append(allocator, textEntry("accelerators\x1f0", "ane"));
        index += 1;
    }
    if ((mask & policy.accel_gpu) != 0) {
        try out.append(allocator, textEntry(acceleratorPath(index), "gpu"));
        index += 1;
    }
    if ((mask & policy.accel_npu) != 0) {
        try out.append(allocator, textEntry(acceleratorPath(index), "npu"));
        index += 1;
    }
    if ((mask & policy.accel_cpu) != 0) {
        try out.append(allocator, textEntry(acceleratorPath(index), "cpu"));
    }
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

fn rootValue(entries: []const Entry, key: []const u8) ?Entry {
    for (entries) |entry| {
        const path = entryPath(entry) orelse continue;
        if (std.mem.eql(u8, path, key)) {
            return entry;
        }
    }
    return null;
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

fn acceleratorPath(index: usize) [*c]const u8 {
    return switch (index) {
        0 => "accelerators\x1f0",
        1 => "accelerators\x1f1",
        2 => "accelerators\x1f2",
        else => "accelerators\x1f3",
    };
}

fn kindEntry(path: [*c]const u8, kind: i32) Entry {
    return .{ .path = path, .kind = kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 };
}

fn textEntry(path: [*c]const u8, value: [*c]const u8) Entry {
    return .{ .path = path, .kind = string_kind, .text = value, .int_value = 0, .double_value = 0, .bool_value = 0 };
}

fn intEntry(path: [*c]const u8, value: i32) Entry {
    return .{ .path = path, .kind = int_kind, .text = null, .int_value = value, .double_value = 0, .bool_value = 0 };
}

fn boolEntry(path: [*c]const u8, value: bool) Entry {
    return .{ .path = path, .kind = bool_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = if (value) 1 else 0 };
}

fn findPath(entries: []const Entry, path: []const u8) ?Entry {
    for (entries) |entry| {
        const entry_path = entryPath(entry) orelse continue;
        if (std.mem.eql(u8, entry_path, path)) {
            return entry;
        }
    }
    return null;
}

test "runtime open options are merged as typed entries" {
    const backend = [_]Entry{
        textEntry("provider", "cuda"),
        textEntry("zigRuntimeMode", "echo"),
    };
    const metadata = [_]Entry{
        textEntry("provider", "cpu"),
        .{ .path = "diagnostics", .kind = bool_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 },
    };
    const options = try merge(
        std.testing.allocator,
        @intFromEnum(policy.Engine.onnx),
        policy.accel_gpu | policy.accel_cpu,
        true,
        4,
        &metadata,
        &backend,
    );
    defer std.testing.allocator.free(options);
    try std.testing.expectEqual(@as(usize, 8), options.len);
    try std.testing.expect(std.mem.eql(u8, entryText(findPath(options, "provider").?), "cuda"));
    try std.testing.expectEqual(@as(i32, 0), findPath(options, "diagnostics").?.bool_value);
    try std.testing.expectEqual(@as(i32, list_kind), findPath(options, "accelerators").?.kind);
    try std.testing.expect(std.mem.eql(u8, entryText(findPath(options, "accelerators\x1f0").?), "gpu"));
    try std.testing.expect(std.mem.eql(u8, entryText(findPath(options, "accelerators\x1f1").?), "cpu"));
    try std.testing.expectEqual(@as(i64, 4), findPath(options, "numThreads").?.int_value);
    try std.testing.expect(std.mem.eql(u8, entryText(findPath(options, "engine").?), "onnx"));
}

test "runtime open options preserve nested entries" {
    const entries = [_]Entry{
        .{ .path = "items", .kind = list_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 },
        textEntry("items\x1f0", "a"),
        .{ .path = "items\x1f1", .kind = int_kind, .text = null, .int_value = 7, .double_value = 0, .bool_value = 0 },
    };
    const options = try merge(std.testing.allocator, @intFromEnum(policy.Engine.onnx), 0, false, 0, &.{}, &entries);
    defer std.testing.allocator.free(options);
    try std.testing.expectEqual(@as(i32, list_kind), findPath(options, "items").?.kind);
    try std.testing.expect(std.mem.eql(u8, entryText(findPath(options, "items\x1f0").?), "a"));
    try std.testing.expectEqual(@as(i64, 7), findPath(options, "items\x1f1").?.int_value);
}

test "runtime open text lookup honors backend overrides" {
    const metadata = [_]Entry{textEntry("zigRuntimeMode", "echo")};
    const backend = [_]Entry{
        .{ .path = "zigRuntimeMode", .kind = null_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 },
    };
    try std.testing.expect(textEquals(&metadata, &.{}, "zigRuntimeMode", "echo"));
    try std.testing.expect(!textEquals(&metadata, &backend, "zigRuntimeMode", "echo"));
}
