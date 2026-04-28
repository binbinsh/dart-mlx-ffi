const std = @import("std");
const abi = @import("abi.zig");
const mlx_backend = @import("mlx_backend.zig");

const path_sep: u8 = 0x1f;

pub const Entry = extern struct {
    path: [*c]u8,
    kind: i32,
    text: [*c]u8,
    int_value: i64,
    double_value: f64,
    bool_value: i32,
};

pub const string_kind: i32 = 1;
pub const int_kind: i32 = 2;
pub const bool_kind: i32 = 3;
pub const map_kind: i32 = 4;
pub const list_kind: i32 = 5;
pub const double_kind: i32 = 6;
pub const null_kind: i32 = 7;

const Error = error{OutOfMemory};

pub const Builder = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Builder) void {
        freeSlice(self.entries.items);
        self.entries.deinit(self.allocator);
    }

    pub fn finish(self: *Builder) Error![]Entry {
        const owned = self.entries.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
        self.entries = .empty;
        return owned;
    }

    pub fn addString(self: *Builder, path: []const u8, value: []const u8) Error!void {
        const owned_path = try copy(path);
        errdefer abi.freeString(owned_path);
        const owned_text = try copy(value);
        try self.add(.{ .path = owned_path, .kind = string_kind, .text = owned_text, .int_value = 0, .double_value = 0, .bool_value = 0 });
    }

    pub fn addInt(self: *Builder, path: []const u8, value: i64) Error!void {
        try self.add(.{ .path = try copy(path), .kind = int_kind, .text = null, .int_value = value, .double_value = 0, .bool_value = 0 });
    }

    pub fn addBool(self: *Builder, path: []const u8, value: bool) Error!void {
        try self.add(.{ .path = try copy(path), .kind = bool_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = if (value) 1 else 0 });
    }

    pub fn addMap(self: *Builder, path: []const u8) Error!void {
        try self.add(.{ .path = try copy(path), .kind = map_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 });
    }

    pub fn addList(self: *Builder, path: []const u8) Error!void {
        try self.add(.{ .path = try copy(path), .kind = list_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 });
    }

    pub fn addDouble(self: *Builder, path: []const u8, value: f64) Error!void {
        try self.add(.{ .path = try copy(path), .kind = double_kind, .text = null, .int_value = 0, .double_value = value, .bool_value = 0 });
    }

    pub fn addNull(self: *Builder, path: []const u8) Error!void {
        try self.add(.{ .path = try copy(path), .kind = null_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 0 });
    }

    pub fn addStringField(self: *Builder, parent: []const u8, key: []const u8, value: []const u8) Error!void {
        const path = try join(self.allocator, parent, key);
        defer self.allocator.free(path);
        try self.addString(path, value);
    }

    pub fn addIntField(self: *Builder, parent: []const u8, key: []const u8, value: i64) Error!void {
        const path = try join(self.allocator, parent, key);
        defer self.allocator.free(path);
        try self.addInt(path, value);
    }

    pub fn addBoolField(self: *Builder, parent: []const u8, key: []const u8, value: bool) Error!void {
        const path = try join(self.allocator, parent, key);
        defer self.allocator.free(path);
        try self.addBool(path, value);
    }

    fn add(self: *Builder, entry: Entry) Error!void {
        errdefer freeEntry(entry);
        self.entries.append(self.allocator, entry) catch return error.OutOfMemory;
    }
};

pub fn zigSession(
    allocator: std.mem.Allocator,
    engine: []const u8,
    mode: []const u8,
    zig_version: []const u8,
    mlx_session: ?*const mlx_backend.Session,
) Error![]Entry {
    var builder = Builder.init(allocator);
    errdefer builder.deinit();
    try addBase(&builder, engine, mode, zig_version);
    if (mlx_session) |session| {
        try addMlxSession(&builder, "mlx_session", session);
    }
    return builder.finish();
}

pub fn copyEntries(
    allocator: std.mem.Allocator,
    entries: [*c]const Entry,
    count: isize,
) Error![]Entry {
    var builder = Builder.init(allocator);
    errdefer builder.deinit();
    try appendEntries(&builder, entries, count);
    return builder.finish();
}

pub fn adapterSession(
    allocator: std.mem.Allocator,
    engine: []const u8,
    zig_version: []const u8,
    entries: [*c]const Entry,
    count: isize,
) Error![]Entry {
    var builder = Builder.init(allocator);
    errdefer builder.deinit();
    try addBase(&builder, engine, "adapter", zig_version);
    try appendEntries(&builder, entries, count);
    return builder.finish();
}

pub fn freeEntries(entries: [*c]Entry, count: isize) void {
    if (entries == null or count <= 0) {
        return;
    }
    const len: usize = @intCast(count);
    const slice = entries[0..len];
    freeSlice(slice);
    std.heap.c_allocator.free(slice);
}

fn addBase(builder: *Builder, engine: []const u8, mode: []const u8, zig_version: []const u8) Error!void {
    try builder.addString("native_backend", "zig");
    try builder.addString("engine", engine);
    try builder.addString("mode", mode);
    try builder.addString("zig_version", zig_version);
    try builder.addMap("mlx_backend");
    try builder.addStringField("mlx_backend", "owner", mlx_backend.owner);
    try builder.addStringField("mlx_backend", "api", mlx_backend.api);
    try builder.addBoolField("mlx_backend", "linked", mlx_backend.linked);
    try builder.addBoolField("mlx_backend", "enabled", mlx_backend.enabled);
    try addStringList(builder, "mlx_backend", "registered_artifacts", &.{ "mlxfn", "dart_inference_linear" });
}

fn appendEntries(
    builder: *Builder,
    entries: [*c]const Entry,
    count: isize,
) Error!void {
    if (entries == null or count <= 0) {
        return;
    }
    const slice = entries[0..@intCast(count)];
    for (slice) |entry| {
        const path = entryPath(entry) orelse continue;
        switch (entry.kind) {
            string_kind => try builder.addString(path, entryText(entry)),
            int_kind => try builder.addInt(path, entry.int_value),
            bool_kind => try builder.addBool(path, entry.bool_value != 0),
            map_kind => try builder.addMap(path),
            list_kind => try builder.addList(path),
            double_kind => try builder.addDouble(path, entry.double_value),
            null_kind => try builder.addNull(path),
            else => try builder.addNull(path),
        }
    }
}

fn addMlxSession(builder: *Builder, parent: []const u8, session: *const mlx_backend.Session) Error!void {
    try builder.addMap(parent);
    try builder.addStringField(parent, "artifact_kind", mlx_backend.artifactKindName(session.artifact_kind));
    try builder.addBoolField(parent, "function_loaded", session.imported_function.ctx != null);
    try builder.addIntField(parent, "weight_file_count", @intCast(session.weight_file_count));
    try builder.addBoolField(parent, "weights_loaded", session.weights.loaded);
    try builder.addIntField(parent, "loaded_weight_file_count", @intCast(session.weights.loaded_file_count));
    try builder.addBoolField(parent, "has_config", session.metadata.config_path != null);
    try builder.addBoolField(parent, "has_tokenizer", session.metadata.tokenizer_path != null);
    try builder.addBoolField(parent, "has_generation_config", session.metadata.generation_config_path != null);
    try builder.addBoolField(parent, "has_inputs_json", session.metadata.inputs_json_path != null);
    if (session.metadata.input_names) |names| {
        try addStringList(builder, parent, "input_order", names);
    }
    if (session.metadata.model_type) |value| {
        try builder.addStringField(parent, "model_type", value);
    }
    if (session.metadata.architecture) |value| {
        try builder.addStringField(parent, "architecture", value);
    }
    if (session.metadata.quantization_mode) |value| {
        try builder.addStringField(parent, "quantization_mode", value);
    }
    if (session.metadata.quantization_bits) |value| {
        try builder.addIntField(parent, "quantization_bits", value);
    }
    if (session.metadata.quantization_group_size) |value| {
        try builder.addIntField(parent, "quantization_group_size", value);
    }
    try builder.addStringField(parent, "executor_kind", mlx_backend.executorKind(session));
}

fn addStringList(builder: *Builder, parent: []const u8, key: []const u8, values: []const []const u8) Error!void {
    const list_path = try join(builder.allocator, parent, key);
    defer builder.allocator.free(list_path);
    try builder.addList(list_path);
    for (values, 0..) |value, index| {
        const item = std.fmt.allocPrint(builder.allocator, "{d}", .{index}) catch return error.OutOfMemory;
        defer builder.allocator.free(item);
        const item_path = try join(builder.allocator, list_path, item);
        defer builder.allocator.free(item_path);
        try builder.addString(item_path, value);
    }
}

fn join(allocator: std.mem.Allocator, parent: []const u8, child: []const u8) Error![]u8 {
    if (parent.len == 0) {
        return allocator.dupe(u8, child) catch return error.OutOfMemory;
    }
    return std.fmt.allocPrint(allocator, "{s}{c}{s}", .{ parent, path_sep, child }) catch
        return error.OutOfMemory;
}

fn copy(value: []const u8) Error![*c]u8 {
    const owned = abi.copyString(value);
    if (owned == null) {
        return error.OutOfMemory;
    }
    return owned;
}

fn freeSlice(entries: []Entry) void {
    for (entries) |entry| {
        freeEntry(entry);
    }
}

fn freeEntry(entry: Entry) void {
    abi.freeString(entry.path);
    abi.freeString(entry.text);
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

test "diagnostic entries copy from typed adapter entries" {
    const source = [_]Entry{
        .{ .path = @constCast("provider"), .kind = string_kind, .text = @constCast("CPU"), .int_value = 0, .double_value = 0, .bool_value = 0 },
        .{ .path = @constCast("names\x1f0"), .kind = string_kind, .text = @constCast("x"), .int_value = 0, .double_value = 0, .bool_value = 0 },
        .{ .path = @constCast("nested\x1fok"), .kind = bool_kind, .text = null, .int_value = 0, .double_value = 0, .bool_value = 1 },
    };
    const entries = try copyEntries(std.heap.c_allocator, source[0..].ptr, @intCast(source.len));
    defer freeEntries(entries.ptr, @intCast(entries.len));
    try std.testing.expectEqual(@as(usize, 3), entries.len);
    try std.testing.expectEqualStrings("provider", entries[0].path[0..std.mem.len(entries[0].path)]);
    try std.testing.expectEqual(string_kind, entries[0].kind);
    try std.testing.expectEqualStrings("CPU", entries[0].text[0..std.mem.len(entries[0].text)]);
    try std.testing.expectEqual(bool_kind, entries[2].kind);
    try std.testing.expectEqual(@as(i32, 1), entries[2].bool_value);
}

test "adapter diagnostics include Zig bridge base fields" {
    const source = [_]Entry{
        .{ .path = @constCast("provider"), .kind = string_kind, .text = @constCast("CUDAExecutionProvider"), .int_value = 0, .double_value = 0, .bool_value = 0 },
    };
    const entries = try adapterSession(std.heap.c_allocator, "onnx", "0.16.0", source[0..].ptr, @intCast(source.len));
    defer freeEntries(entries.ptr, @intCast(entries.len));

    var found_mode = false;
    var found_provider = false;
    for (entries) |entry| {
        const path = entry.path[0..std.mem.len(entry.path)];
        if (std.mem.eql(u8, path, "mode")) {
            found_mode = true;
            try std.testing.expectEqualStrings("adapter", entry.text[0..std.mem.len(entry.text)]);
        }
        if (std.mem.eql(u8, path, "provider")) {
            found_provider = true;
            try std.testing.expectEqualStrings("CUDAExecutionProvider", entry.text[0..std.mem.len(entry.text)]);
        }
    }
    try std.testing.expect(found_mode);
    try std.testing.expect(found_provider);
}

test "Zig diagnostics include MLX backend map" {
    const entries = try zigSession(std.heap.c_allocator, "onnx", "echo", "0.16.0", null);
    defer freeEntries(entries.ptr, @intCast(entries.len));
    var found = false;
    for (entries) |entry| {
        const path = entry.path[0..std.mem.len(entry.path)];
        if (std.mem.eql(u8, path, "mlx_backend\x1fapi")) {
            found = true;
            try std.testing.expectEqualStrings("mlx-c", entry.text[0..std.mem.len(entry.text)]);
        }
    }
    try std.testing.expect(found);
}
