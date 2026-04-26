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

pub fn fromJson(allocator: std.mem.Allocator, json: []const u8) Error![]Entry {
    var builder = Builder.init(allocator);
    errdefer builder.deinit();
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return builder.finish();
    defer parsed.deinit();
    switch (parsed.value) {
        .object => |object| {
            var it = object.iterator();
            while (it.next()) |item| {
                try addJsonValue(&builder, item.key_ptr.*, item.value_ptr.*);
            }
        },
        else => {},
    }
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

fn addJsonValue(builder: *Builder, path: []const u8, value: std.json.Value) Error!void {
    switch (value) {
        .null => try builder.addNull(path),
        .bool => |item| try builder.addBool(path, item),
        .integer => |item| try builder.addInt(path, item),
        .float => |item| try builder.addDouble(path, item),
        .number_string => |item| try builder.addString(path, item),
        .string => |item| try builder.addString(path, item),
        .array => |array| {
            try builder.addList(path);
            for (array.items, 0..) |item, index| {
                const segment = std.fmt.allocPrint(builder.allocator, "{d}", .{index}) catch return error.OutOfMemory;
                defer builder.allocator.free(segment);
                const child = try join(builder.allocator, path, segment);
                defer builder.allocator.free(child);
                try addJsonValue(builder, child, item);
            }
        },
        .object => |object| {
            try builder.addMap(path);
            var it = object.iterator();
            while (it.next()) |item| {
                const child = try join(builder.allocator, path, item.key_ptr.*);
                defer builder.allocator.free(child);
                try addJsonValue(builder, child, item.value_ptr.*);
            }
        },
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

test "diagnostic entries parse nested JSON" {
    const entries = try fromJson(std.heap.c_allocator, "{\"provider\":\"CPU\",\"names\":[\"x\"],\"nested\":{\"ok\":true,\"n\":3}}");
    defer freeEntries(entries.ptr, @intCast(entries.len));
    try std.testing.expect(entries.len >= 6);
    try std.testing.expectEqualStrings("provider", entries[0].path[0..std.mem.len(entries[0].path)]);
    try std.testing.expectEqual(string_kind, entries[0].kind);
    try std.testing.expectEqualStrings("CPU", entries[0].text[0..std.mem.len(entries[0].text)]);
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
