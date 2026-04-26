const std = @import("std");

const path = std.Io.Dir.path;
const max_config_json_bytes = 2 * 1024 * 1024;

pub const ArtifactKind = enum {
    single_mlx_function,
    directory_mlx_function,
    single_safetensors,
    directory_model_safetensors,
    directory_sharded_safetensors,
    directory_safetensors,
};

pub const SessionError = error{
    InvalidPath,
    ArtifactNotFound,
    UnsupportedArtifact,
    InvalidConfig,
    MlxCallFailed,
    OutOfMemory,
};

pub const Metadata = struct {
    config_path: ?[]u8,
    tokenizer_path: ?[]u8,
    generation_config_path: ?[]u8,
    inputs_json_path: ?[]u8,
    input_names: ?[][]u8,
    model_type: ?[]u8,
    architecture: ?[]u8,
    quantization_mode: ?[]u8,
    quantization_bits: ?i64,
    quantization_group_size: ?i64,

    pub fn empty() Metadata {
        return .{
            .config_path = null,
            .tokenizer_path = null,
            .generation_config_path = null,
            .inputs_json_path = null,
            .input_names = null,
            .model_type = null,
            .architecture = null,
            .quantization_mode = null,
            .quantization_bits = null,
            .quantization_group_size = null,
        };
    }

    pub fn deinit(self: *Metadata, allocator: std.mem.Allocator) void {
        freeOptionalString(allocator, &self.config_path);
        freeOptionalString(allocator, &self.tokenizer_path);
        freeOptionalString(allocator, &self.generation_config_path);
        freeOptionalString(allocator, &self.inputs_json_path);
        if (self.input_names) |names| {
            freeStringList(allocator, names);
            self.input_names = null;
        }
        freeOptionalString(allocator, &self.model_type);
        freeOptionalString(allocator, &self.architecture);
        freeOptionalString(allocator, &self.quantization_mode);
        self.* = Metadata.empty();
    }
};

pub const Layout = struct {
    kind: ArtifactKind,
    primary_artifact_path: []u8,
    function_path: ?[]u8,
    weight_paths: [][]u8,
    weight_file_count: usize,

    pub fn deinit(self: Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.primary_artifact_path);
        var function_path = self.function_path;
        freeOptionalString(allocator, &function_path);
        freeStringList(allocator, self.weight_paths);
    }
};

pub fn discover(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!Layout {
    const stat = std.Io.Dir.cwd().statFile(io, model_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.ArtifactNotFound,
        else => return error.InvalidPath,
    };
    return switch (stat.kind) {
        .file => discoverFileArtifact(allocator, model_path),
        .directory => discoverDirectoryArtifact(allocator, io, model_path),
        else => error.UnsupportedArtifact,
    };
}

pub fn loadMetadata(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!Metadata {
    const stat = std.Io.Dir.cwd().statFile(io, model_path, .{}) catch return error.InvalidPath;
    if (stat.kind == .file and std.mem.endsWith(u8, model_path, ".mlxfn")) {
        const parent = try parentPath(allocator, model_path);
        defer allocator.free(parent);
        return loadDirectoryMetadata(allocator, io, parent);
    }
    if (stat.kind != .directory) {
        return Metadata.empty();
    }
    return loadDirectoryMetadata(allocator, io, model_path);
}

fn loadDirectoryMetadata(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!Metadata {
    var dir = std.Io.Dir.cwd().openDir(io, model_path, .{}) catch return error.InvalidPath;
    defer dir.close(io);

    var metadata = Metadata.empty();
    errdefer metadata.deinit(allocator);
    metadata.config_path = try optionalJoinedFile(allocator, dir, io, model_path, "config.json");
    metadata.tokenizer_path = try optionalJoinedFile(allocator, dir, io, model_path, "tokenizer.json");
    metadata.generation_config_path = try optionalJoinedFile(allocator, dir, io, model_path, "generation_config.json");
    metadata.inputs_json_path = try optionalJoinedFile(allocator, dir, io, model_path, "inputs.json");
    if (metadata.config_path) |config_path| {
        try parseModelConfig(allocator, io, config_path, &metadata);
    }
    if (metadata.inputs_json_path) |inputs_json_path| {
        metadata.input_names = try parseInputNames(allocator, io, inputs_json_path);
    }
    return metadata;
}

pub fn kindName(kind: ArtifactKind) []const u8 {
    return switch (kind) {
        .single_mlx_function => "single_mlx_function",
        .directory_mlx_function => "directory_mlx_function",
        .single_safetensors => "single_safetensors",
        .directory_model_safetensors => "directory_model_safetensors",
        .directory_sharded_safetensors => "directory_sharded_safetensors",
        .directory_safetensors => "directory_safetensors",
    };
}

pub fn freeStringList(allocator: std.mem.Allocator, items: [][]u8) void {
    freeStringsOnly(allocator, items);
    allocator.free(items);
}

pub fn freeOptionalString(allocator: std.mem.Allocator, value: *?[]u8) void {
    if (value.*) |bytes| {
        allocator.free(bytes);
        value.* = null;
    }
}

fn discoverFileArtifact(
    allocator: std.mem.Allocator,
    model_path: []const u8,
) SessionError!Layout {
    if (std.mem.endsWith(u8, model_path, ".mlxfn")) {
        const primary = allocator.dupe(u8, model_path) catch return error.OutOfMemory;
        errdefer allocator.free(primary);
        const function_path = allocator.dupe(u8, model_path) catch return error.OutOfMemory;
        errdefer allocator.free(function_path);
        const weight_paths = try emptyPathList(allocator);
        errdefer allocator.free(weight_paths);
        return .{
            .kind = .single_mlx_function,
            .primary_artifact_path = primary,
            .function_path = function_path,
            .weight_paths = weight_paths,
            .weight_file_count = 0,
        };
    }
    if (!std.mem.endsWith(u8, model_path, ".safetensors")) {
        return error.UnsupportedArtifact;
    }
    const primary = allocator.dupe(u8, model_path) catch return error.OutOfMemory;
    errdefer allocator.free(primary);
    const weight_paths = try singlePathList(allocator, model_path);
    errdefer freeStringList(allocator, weight_paths);
    return .{
        .kind = .single_safetensors,
        .primary_artifact_path = primary,
        .function_path = null,
        .weight_paths = weight_paths,
        .weight_file_count = 1,
    };
}

fn discoverDirectoryArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!Layout {
    var dir = std.Io.Dir.cwd().openDir(io, model_path, .{
        .access_sub_paths = true,
        .iterate = true,
    }) catch return error.InvalidPath;
    defer dir.close(io);

    if (fileExists(dir, io, "function.mlxfn")) {
        const function_path = try joinPath(allocator, model_path, "function.mlxfn");
        errdefer allocator.free(function_path);
        const primary = allocator.dupe(u8, function_path) catch return error.OutOfMemory;
        errdefer allocator.free(primary);
        const weight_paths = try emptyPathList(allocator);
        errdefer allocator.free(weight_paths);
        return .{
            .kind = .directory_mlx_function,
            .primary_artifact_path = primary,
            .function_path = function_path,
            .weight_paths = weight_paths,
            .weight_file_count = 0,
        };
    }

    if (fileExists(dir, io, "model.safetensors")) {
        const primary = try joinPath(allocator, model_path, "model.safetensors");
        errdefer allocator.free(primary);
        return .{
            .kind = .directory_model_safetensors,
            .primary_artifact_path = primary,
            .function_path = null,
            .weight_paths = try singlePathList(allocator, primary),
            .weight_file_count = 1,
        };
    }

    if (fileExists(dir, io, "model.safetensors.index.json")) {
        const scan = try scanSafetensors(allocator, io, dir, model_path);
        errdefer scan.deinit(allocator);
        if (scan.count == 0) {
            return error.UnsupportedArtifact;
        }
        return .{
            .kind = .directory_sharded_safetensors,
            .primary_artifact_path = try joinPath(allocator, model_path, "model.safetensors.index.json"),
            .function_path = null,
            .weight_paths = scan.paths,
            .weight_file_count = scan.count,
        };
    }

    const scan = try scanSafetensors(allocator, io, dir, model_path);
    errdefer scan.deinit(allocator);
    if (scan.count > 0) {
        const primary = allocator.dupe(u8, scan.paths[0]) catch return error.OutOfMemory;
        return .{
            .kind = .directory_safetensors,
            .primary_artifact_path = primary,
            .function_path = null,
            .weight_paths = scan.paths,
            .weight_file_count = scan.count,
        };
    }
    return error.UnsupportedArtifact;
}

const SafetensorsScan = struct {
    count: usize,
    paths: [][]u8,

    fn deinit(self: SafetensorsScan, allocator: std.mem.Allocator) void {
        freeStringList(allocator, self.paths);
    }
};

fn scanSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    model_path: []const u8,
) SessionError!SafetensorsScan {
    var iter = dir.iterate();
    var paths: std.ArrayList([]u8) = .empty;
    defer paths.deinit(allocator);
    errdefer freeStringsOnly(allocator, paths.items);

    while (iter.next(io) catch return error.InvalidPath) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".safetensors")) {
            continue;
        }
        const weight_path = try joinPath(allocator, model_path, entry.name);
        errdefer allocator.free(weight_path);
        paths.append(allocator, weight_path) catch return error.OutOfMemory;
    }
    std.sort.insertion([]u8, paths.items, {}, pathLessThan);
    const owned = paths.toOwnedSlice(allocator) catch return error.OutOfMemory;
    return .{ .count = owned.len, .paths = owned };
}

fn fileExists(dir: std.Io.Dir, io: std.Io, file_name: []const u8) bool {
    const stat = dir.statFile(io, file_name, .{}) catch return false;
    return stat.kind == .file;
}

fn joinPath(
    allocator: std.mem.Allocator,
    parent: []const u8,
    child: []const u8,
) SessionError![]u8 {
    return path.join(allocator, &.{ parent, child }) catch error.OutOfMemory;
}

fn parentPath(allocator: std.mem.Allocator, value: []const u8) SessionError![]u8 {
    var index = value.len;
    while (index > 0) {
        index -= 1;
        if (value[index] == '/' or value[index] == '\\') {
            if (index == 0) {
                return allocator.dupe(u8, value[0..1]) catch return error.OutOfMemory;
            }
            return allocator.dupe(u8, value[0..index]) catch return error.OutOfMemory;
        }
    }
    return allocator.dupe(u8, ".") catch return error.OutOfMemory;
}

fn optionalJoinedFile(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    io: std.Io,
    parent: []const u8,
    file_name: []const u8,
) SessionError!?[]u8 {
    if (!fileExists(dir, io, file_name)) {
        return null;
    }
    return try joinPath(allocator, parent, file_name);
}

fn parseModelConfig(
    allocator: std.mem.Allocator,
    io: std.Io,
    config_path: []const u8,
    metadata: *Metadata,
) SessionError!void {
    const bytes = std.Io.Dir.cwd().readFileAlloc(
        io,
        config_path,
        allocator,
        .limited(max_config_json_bytes),
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidConfig,
    };
    defer allocator.free(bytes);

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        bytes,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return error.InvalidConfig;
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |value| value,
        else => return error.InvalidConfig,
    };
    metadata.model_type = try copyStringField(allocator, object, "model_type");
    metadata.architecture = try copyFirstStringField(allocator, object, "architectures");
    try parseQuantizationMetadata(allocator, object, metadata);
}

fn parseQuantizationMetadata(
    allocator: std.mem.Allocator,
    object: std.json.ObjectMap,
    metadata: *Metadata,
) SessionError!void {
    const quantization = objectField(object, "quantization") orelse
        objectField(object, "quantization_config") orelse return;
    metadata.quantization_mode = try copyStringField(allocator, quantization, "mode") orelse
        try copyStringField(allocator, quantization, "quant_method");
    metadata.quantization_bits = integerField(quantization, "bits");
    metadata.quantization_group_size = integerField(quantization, "group_size");
    if (metadata.quantization_mode == null and
        (metadata.quantization_bits != null or metadata.quantization_group_size != null))
    {
        metadata.quantization_mode = allocator.dupe(u8, "affine") catch return error.OutOfMemory;
    }
}

fn parseInputNames(
    allocator: std.mem.Allocator,
    io: std.Io,
    inputs_json_path: []const u8,
) SessionError![][]u8 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(
        io,
        inputs_json_path,
        allocator,
        .limited(max_config_json_bytes),
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidConfig,
    };
    defer allocator.free(bytes);

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        bytes,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return error.InvalidConfig;
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |value| value,
        else => return error.InvalidConfig,
    };
    if (try inputOrderField(allocator, object, "input_order")) |names| {
        return names;
    }
    if (try inputOrderField(allocator, object, "inputOrder")) |names| {
        return names;
    }
    const inputs = objectField(object, "inputs") orelse object;
    return copyObjectKeys(allocator, inputs);
}

fn inputOrderField(
    allocator: std.mem.Allocator,
    object: std.json.ObjectMap,
    key: []const u8,
) SessionError!?[][]u8 {
    const value = object.get(key) orelse return null;
    const items = switch (value) {
        .array => |array| array.items,
        else => return error.InvalidConfig,
    };
    var names: std.ArrayList([]u8) = .empty;
    defer names.deinit(allocator);
    errdefer freeStringsOnly(allocator, names.items);
    for (items) |item| {
        const text = switch (item) {
            .string => |string| string,
            else => return error.InvalidConfig,
        };
        const copy = allocator.dupe(u8, text) catch return error.OutOfMemory;
        errdefer allocator.free(copy);
        names.append(allocator, copy) catch return error.OutOfMemory;
    }
    return names.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn copyObjectKeys(
    allocator: std.mem.Allocator,
    object: std.json.ObjectMap,
) SessionError![][]u8 {
    var names: std.ArrayList([]u8) = .empty;
    defer names.deinit(allocator);
    errdefer freeStringsOnly(allocator, names.items);

    var iterator = object.iterator();
    while (iterator.next()) |entry| {
        const copy = allocator.dupe(u8, entry.key_ptr.*) catch return error.OutOfMemory;
        errdefer allocator.free(copy);
        names.append(allocator, copy) catch return error.OutOfMemory;
    }
    return names.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn objectField(
    object: std.json.ObjectMap,
    key: []const u8,
) ?std.json.ObjectMap {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .object => |item| item,
        else => null,
    };
}

fn integerField(
    object: std.json.ObjectMap,
    key: []const u8,
) ?i64 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |integer| integer,
        else => null,
    };
}

fn copyStringField(
    allocator: std.mem.Allocator,
    object: std.json.ObjectMap,
    key: []const u8,
) SessionError!?[]u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |text| allocator.dupe(u8, text) catch return error.OutOfMemory,
        else => null,
    };
}

fn copyFirstStringField(
    allocator: std.mem.Allocator,
    object: std.json.ObjectMap,
    key: []const u8,
) SessionError!?[]u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .array => |items| {
            if (items.items.len == 0) {
                return null;
            }
            return switch (items.items[0]) {
                .string => |text| allocator.dupe(u8, text) catch return error.OutOfMemory,
                else => null,
            };
        },
        else => null,
    };
}

fn pathLessThan(_: void, lhs: []u8, rhs: []u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

fn emptyPathList(allocator: std.mem.Allocator) SessionError![][]u8 {
    return allocator.alloc([]u8, 0) catch error.OutOfMemory;
}

fn singlePathList(
    allocator: std.mem.Allocator,
    item: []const u8,
) SessionError![][]u8 {
    const list = allocator.alloc([]u8, 1) catch return error.OutOfMemory;
    errdefer allocator.free(list);
    list[0] = allocator.dupe(u8, item) catch return error.OutOfMemory;
    return list;
}

fn freeStringsOnly(allocator: std.mem.Allocator, items: [][]u8) void {
    for (items) |item| {
        allocator.free(item);
    }
}

test "MLX artifact discovery prioritizes exported function bundles" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();
    const function = try tmp.dir.createFile(std.testing.io, "function.mlxfn", .{});
    function.close(std.testing.io);
    const inputs = try tmp.dir.createFile(std.testing.io, "inputs.safetensors", .{});
    inputs.close(std.testing.io);

    const model_path = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(model_path);
    const layout = try discover(std.testing.allocator, std.testing.io, model_path);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(ArtifactKind.directory_mlx_function, layout.kind);
    try std.testing.expectEqual(@as(usize, 0), layout.weight_file_count);
    try std.testing.expectEqual(@as(usize, 0), layout.weight_paths.len);
    try std.testing.expect(layout.function_path != null);
}

test "MLX artifact metadata parses safetensors directory config" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(std.testing.io, "model.safetensors", .{});
    file.close(std.testing.io);
    const config = try tmp.dir.createFile(std.testing.io, "config.json", .{});
    var config_writer = config.writer(std.testing.io, &.{});
    try config_writer.interface.writeAll("{\"model_type\":\"qwen3\",\"architectures\":[\"Qwen3ForCausalLM\"],\"quantization\":{\"bits\":4,\"group_size\":64}}");
    try config_writer.interface.flush();
    config.close(std.testing.io);
    const tokenizer = try tmp.dir.createFile(std.testing.io, "tokenizer.json", .{});
    tokenizer.close(std.testing.io);
    const generation_config = try tmp.dir.createFile(std.testing.io, "generation_config.json", .{});
    generation_config.close(std.testing.io);

    const model_path = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(model_path);
    const layout = try discover(std.testing.allocator, std.testing.io, model_path);
    defer layout.deinit(std.testing.allocator);
    var metadata = try loadMetadata(std.testing.allocator, std.testing.io, model_path);
    defer metadata.deinit(std.testing.allocator);

    try std.testing.expectEqual(ArtifactKind.directory_model_safetensors, layout.kind);
    try std.testing.expectEqual(@as(usize, 1), layout.weight_file_count);
    try std.testing.expectEqualStrings("qwen3", metadata.model_type.?);
    try std.testing.expectEqualStrings("Qwen3ForCausalLM", metadata.architecture.?);
    try std.testing.expectEqualStrings("affine", metadata.quantization_mode.?);
    try std.testing.expectEqual(@as(i64, 4), metadata.quantization_bits.?);
    try std.testing.expectEqual(@as(i64, 64), metadata.quantization_group_size.?);
    try std.testing.expect(metadata.config_path != null);
    try std.testing.expect(metadata.tokenizer_path != null);
    try std.testing.expect(metadata.generation_config_path != null);
}

test "MLX artifact metadata parses function input order" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();
    const function = try tmp.dir.createFile(std.testing.io, "function.mlxfn", .{});
    function.close(std.testing.io);
    const inputs_json = try tmp.dir.createFile(std.testing.io, "inputs.json", .{});
    var writer = inputs_json.writer(std.testing.io, &.{});
    try writer.interface.writeAll(
        "{\"inputs\":{\"pixel_values\":{\"dtype\":\"float32\"},\"input_ids\":{\"dtype\":\"int32\"}},\"input_order\":[\"input_ids\",\"pixel_values\"]}",
    );
    try writer.interface.flush();
    inputs_json.close(std.testing.io);

    const model_path = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(model_path);
    var metadata = try loadMetadata(std.testing.allocator, std.testing.io, model_path);
    defer metadata.deinit(std.testing.allocator);

    try std.testing.expect(metadata.inputs_json_path != null);
    try std.testing.expect(metadata.input_names != null);
    try std.testing.expectEqual(@as(usize, 2), metadata.input_names.?.len);
    try std.testing.expectEqualStrings("input_ids", metadata.input_names.?[0]);
    try std.testing.expectEqualStrings("pixel_values", metadata.input_names.?[1]);
}

test "MLX artifact metadata parses sibling inputs for single function files" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();
    const function = try tmp.dir.createFile(std.testing.io, "function.mlxfn", .{});
    function.close(std.testing.io);
    const inputs_json = try tmp.dir.createFile(std.testing.io, "inputs.json", .{});
    var writer = inputs_json.writer(std.testing.io, &.{});
    try writer.interface.writeAll("{\"inputs\":{\"tokens\":{\"dtype\":\"int32\"}}}");
    try writer.interface.flush();
    inputs_json.close(std.testing.io);

    const function_path = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "function.mlxfn" });
    defer std.testing.allocator.free(function_path);
    var metadata = try loadMetadata(std.testing.allocator, std.testing.io, function_path);
    defer metadata.deinit(std.testing.allocator);

    try std.testing.expect(metadata.inputs_json_path != null);
    try std.testing.expect(metadata.input_names != null);
    try std.testing.expectEqual(@as(usize, 1), metadata.input_names.?.len);
    try std.testing.expectEqualStrings("tokens", metadata.input_names.?[0]);
}

test "MLX artifact discovery sorts sharded safetensors" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();
    const index_file = try tmp.dir.createFile(std.testing.io, "model.safetensors.index.json", .{});
    index_file.close(std.testing.io);
    const shard_b = try tmp.dir.createFile(std.testing.io, "model-00002-of-00002.safetensors", .{});
    shard_b.close(std.testing.io);
    const shard_a = try tmp.dir.createFile(std.testing.io, "model-00001-of-00002.safetensors", .{});
    shard_a.close(std.testing.io);

    const model_path = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(model_path);
    const layout = try discover(std.testing.allocator, std.testing.io, model_path);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(ArtifactKind.directory_sharded_safetensors, layout.kind);
    try std.testing.expectEqual(@as(usize, 2), layout.weight_file_count);
    try std.testing.expectEqual(@as(usize, 2), layout.weight_paths.len);
    try std.testing.expect(std.mem.endsWith(u8, layout.weight_paths[0], "model-00001-of-00002.safetensors"));
}
