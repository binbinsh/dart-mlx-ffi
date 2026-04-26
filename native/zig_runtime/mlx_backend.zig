const std = @import("std");
const builtin = @import("builtin");
const mlx_c = @import("mlx_c.zig");
const mlx_output = @import("mlx_output.zig");

pub const api = mlx_c.api;
pub const owner = mlx_c.owner;
pub const linked = mlx_c.linked;
pub const enabled = mlx_c.enabled;
pub const status_json = mlx_c.status_json;
pub const MlxArray = mlx_c.MlxArray;
pub const MlxDtype = mlx_c.MlxDtype;
pub const OutputBatch = mlx_output.OutputBatch;
pub const OutputTensor = mlx_output.OutputTensor;

const MlxString = mlx_c.MlxString;
const MlxMapStringToArray = mlx_c.MlxMapStringToArray;
const MlxMapStringToString = mlx_c.MlxMapStringToString;
const MlxMapStringToArrayIterator = mlx_c.MlxMapStringToArrayIterator;
const MlxMapStringToStringIterator = mlx_c.MlxMapStringToStringIterator;
const MlxStream = mlx_c.MlxStream;

pub const InputTensor = extern struct {
    dtype: i32,
    rank: i32,
    shape: [*c]const i64,
    byte_length: isize,
    data: ?*const anyopaque,
};

const MlxManagedDtor = *const fn (?*anyopaque) callconv(.c) void;
const path = std.Io.Dir.path;
const max_config_json_bytes = 2 * 1024 * 1024;

const native = if (linked and !builtin.is_test) struct {
    extern fn mlx_string_new() MlxString;
    extern fn mlx_string_data(str: MlxString) [*c]const u8;
    extern fn mlx_string_free(str: MlxString) c_int;
    extern fn mlx_version(str: *MlxString) c_int;
    extern fn mlx_array_new_data_managed_payload(
        data: ?*anyopaque,
        shape: [*c]const c_int,
        dim: c_int,
        dtype: MlxDtype,
        payload: ?*anyopaque,
        dtor: MlxManagedDtor,
    ) MlxArray;
    extern fn mlx_array_free(arr: MlxArray) c_int;
    extern fn mlx_map_string_to_array_new() MlxMapStringToArray;
    extern fn mlx_map_string_to_array_free(map: MlxMapStringToArray) c_int;
    extern fn mlx_map_string_to_array_insert(
        map: MlxMapStringToArray,
        key: [*c]const u8,
        value: MlxArray,
    ) c_int;
    extern fn mlx_map_string_to_array_get(
        value: *MlxArray,
        map: MlxMapStringToArray,
        key: [*c]const u8,
    ) c_int;
    extern fn mlx_map_string_to_array_iterator_new(
        map: MlxMapStringToArray,
    ) MlxMapStringToArrayIterator;
    extern fn mlx_map_string_to_array_iterator_next(
        key: *[*c]const u8,
        value: *MlxArray,
        it: MlxMapStringToArrayIterator,
    ) c_int;
    extern fn mlx_map_string_to_array_iterator_free(
        it: MlxMapStringToArrayIterator,
    ) c_int;
    extern fn mlx_map_string_to_string_new() MlxMapStringToString;
    extern fn mlx_map_string_to_string_free(map: MlxMapStringToString) c_int;
    extern fn mlx_map_string_to_string_insert(
        map: MlxMapStringToString,
        key: [*c]const u8,
        value: [*c]const u8,
    ) c_int;
    extern fn mlx_map_string_to_string_iterator_new(
        map: MlxMapStringToString,
    ) MlxMapStringToStringIterator;
    extern fn mlx_map_string_to_string_iterator_next(
        key: *[*c]const u8,
        value: *[*c]const u8,
        it: MlxMapStringToStringIterator,
    ) c_int;
    extern fn mlx_map_string_to_string_iterator_free(
        it: MlxMapStringToStringIterator,
    ) c_int;
    extern fn mlx_default_gpu_stream_new() MlxStream;
    extern fn mlx_stream_free(stream: MlxStream) c_int;
    extern fn mlx_load_safetensors(
        params: *MlxMapStringToArray,
        metadata: *MlxMapStringToString,
        file: [*:0]const u8,
        stream: MlxStream,
    ) c_int;
    extern fn mlx_matmul(
        result: *MlxArray,
        lhs: MlxArray,
        rhs: MlxArray,
        stream: MlxStream,
    ) c_int;
    extern fn mlx_add(
        result: *MlxArray,
        lhs: MlxArray,
        rhs: MlxArray,
        stream: MlxStream,
    ) c_int;
} else struct {};

pub const VersionError = error{
    MlxUnavailable,
    MlxCallFailed,
    OutOfMemory,
};

pub const ArtifactKind = enum {
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

const Weights = struct {
    params: MlxMapStringToArray,
    metadata: MlxMapStringToString,
    loaded: bool,
    loaded_file_count: usize,

    fn empty() Weights {
        return .{
            .params = .{ .ctx = null },
            .metadata = .{ .ctx = null },
            .loaded = false,
            .loaded_file_count = 0,
        };
    }

    fn deinit(self: *Weights) void {
        if (linked and !builtin.is_test) {
            if (self.params.ctx != null) {
                _ = native.mlx_map_string_to_array_free(self.params);
            }
            if (self.metadata.ctx != null) {
                _ = native.mlx_map_string_to_string_free(self.metadata);
            }
        }
        self.* = Weights.empty();
    }
};

const ModelMetadata = struct {
    config_path: ?[]u8,
    tokenizer_path: ?[]u8,
    generation_config_path: ?[]u8,
    model_type: ?[]u8,
    architecture: ?[]u8,
    quantization_mode: ?[]u8,
    quantization_bits: ?i64,
    quantization_group_size: ?i64,

    fn empty() ModelMetadata {
        return .{
            .config_path = null,
            .tokenizer_path = null,
            .generation_config_path = null,
            .model_type = null,
            .architecture = null,
            .quantization_mode = null,
            .quantization_bits = null,
            .quantization_group_size = null,
        };
    }

    fn deinit(self: *ModelMetadata, allocator: std.mem.Allocator) void {
        freeOptionalString(allocator, &self.config_path);
        freeOptionalString(allocator, &self.tokenizer_path);
        freeOptionalString(allocator, &self.generation_config_path);
        freeOptionalString(allocator, &self.model_type);
        freeOptionalString(allocator, &self.architecture);
        freeOptionalString(allocator, &self.quantization_mode);
        self.* = ModelMetadata.empty();
    }
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    model_path: []u8,
    primary_weight_path: []u8,
    weight_paths: [][]u8,
    artifact_kind: ArtifactKind,
    weight_file_count: usize,
    weights: Weights,
    metadata: ModelMetadata,

    pub fn deinit(self: *Session) void {
        self.metadata.deinit(self.allocator);
        self.weights.deinit();
        freeStringList(self.allocator, self.weight_paths);
        self.allocator.free(self.model_path);
        self.allocator.free(self.primary_weight_path);
        self.allocator.destroy(self);
    }
};

const ArtifactLayout = struct {
    kind: ArtifactKind,
    primary_weight_path: []u8,
    weight_paths: [][]u8,
    weight_file_count: usize,

    fn deinit(self: ArtifactLayout, allocator: std.mem.Allocator) void {
        allocator.free(self.primary_weight_path);
        freeStringList(allocator, self.weight_paths);
    }
};

pub fn createSession(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!*Session {
    if (model_path.len == 0 or std.mem.indexOf(u8, model_path, "://") != null) {
        return error.InvalidPath;
    }
    const layout = try discoverArtifact(allocator, io, model_path);
    errdefer layout.deinit(allocator);
    var metadata = try loadModelMetadata(allocator, io, model_path);
    errdefer metadata.deinit(allocator);
    var weights = try loadWeights(allocator, layout.weight_paths);
    errdefer weights.deinit();
    const model_path_copy = allocator.dupe(u8, model_path) catch return error.OutOfMemory;
    errdefer allocator.free(model_path_copy);
    const session = allocator.create(Session) catch return error.OutOfMemory;
    errdefer allocator.destroy(session);
    session.* = .{
        .allocator = allocator,
        .model_path = model_path_copy,
        .primary_weight_path = layout.primary_weight_path,
        .weight_paths = layout.weight_paths,
        .artifact_kind = layout.kind,
        .weight_file_count = layout.weight_file_count,
        .weights = weights,
        .metadata = metadata,
    };
    return session;
}

pub fn sessionDiagnosticsJson(
    session: *const Session,
    allocator: std.mem.Allocator,
) std.mem.Allocator.Error![]u8 {
    return std.json.Stringify.valueAlloc(
        allocator,
        .{
            .artifact_kind = artifactKindName(session.artifact_kind),
            .weight_file_count = session.weight_file_count,
            .weights_loaded = session.weights.loaded,
            .loaded_weight_file_count = session.weights.loaded_file_count,
            .has_config = session.metadata.config_path != null,
            .has_tokenizer = session.metadata.tokenizer_path != null,
            .has_generation_config = session.metadata.generation_config_path != null,
            .model_type = session.metadata.model_type,
            .architecture = session.metadata.architecture,
            .quantization_mode = session.metadata.quantization_mode,
            .quantization_bits = session.metadata.quantization_bits,
            .quantization_group_size = session.metadata.quantization_group_size,
            .executor_kind = executorKind(session),
        },
        .{ .emit_null_optional_fields = false },
    );
}

pub fn sessionErrorMessage(err: SessionError) []const u8 {
    return switch (err) {
        error.InvalidPath => "Zig-owned MLX backend requires a resolved local MLX artifact path.",
        error.ArtifactNotFound => "Zig-owned MLX backend could not find the local MLX artifact.",
        error.UnsupportedArtifact => "Zig-owned MLX backend requires local MLX safetensors weights.",
        error.InvalidConfig => "Zig-owned MLX backend could not parse the local MLX config.json.",
        error.MlxCallFailed => "Zig-owned MLX backend failed while loading safetensors through mlx-c.",
        error.OutOfMemory => "Zig-owned MLX backend ran out of memory while creating the session.",
    };
}

pub fn artifactKindName(kind: ArtifactKind) []const u8 {
    return switch (kind) {
        .single_safetensors => "single_safetensors",
        .directory_model_safetensors => "directory_model_safetensors",
        .directory_sharded_safetensors => "directory_sharded_safetensors",
        .directory_safetensors => "directory_safetensors",
    };
}

pub fn versionString(allocator: std.mem.Allocator) VersionError![]u8 {
    if (!linked or builtin.is_test) {
        return error.MlxUnavailable;
    }
    var value = native.mlx_string_new();
    if (value.ctx == null) {
        return error.MlxUnavailable;
    }
    defer _ = native.mlx_string_free(value);
    if (native.mlx_version(&value) != 0) {
        return error.MlxCallFailed;
    }
    const data = native.mlx_string_data(value);
    if (data == null) {
        return error.MlxCallFailed;
    }
    const len = std.mem.len(data);
    return allocator.dupe(u8, data[0..len]) catch error.OutOfMemory;
}

fn discoverArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!ArtifactLayout {
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

fn discoverFileArtifact(
    allocator: std.mem.Allocator,
    model_path: []const u8,
) SessionError!ArtifactLayout {
    if (!std.mem.endsWith(u8, model_path, ".safetensors")) {
        return error.UnsupportedArtifact;
    }
    return .{
        .kind = .single_safetensors,
        .primary_weight_path = allocator.dupe(u8, model_path) catch return error.OutOfMemory,
        .weight_paths = try singlePathList(allocator, model_path),
        .weight_file_count = 1,
    };
}

fn discoverDirectoryArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!ArtifactLayout {
    var dir = std.Io.Dir.cwd().openDir(io, model_path, .{
        .access_sub_paths = true,
        .iterate = true,
    }) catch return error.InvalidPath;
    defer dir.close(io);

    if (fileExists(dir, io, "model.safetensors")) {
        const primary = try joinPath(allocator, model_path, "model.safetensors");
        errdefer allocator.free(primary);
        return .{
            .kind = .directory_model_safetensors,
            .primary_weight_path = primary,
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
            .primary_weight_path = try joinPath(allocator, model_path, "model.safetensors.index.json"),
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
            .primary_weight_path = primary,
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

fn loadModelMetadata(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_path: []const u8,
) SessionError!ModelMetadata {
    const stat = std.Io.Dir.cwd().statFile(io, model_path, .{}) catch return error.InvalidPath;
    if (stat.kind != .directory) {
        return ModelMetadata.empty();
    }

    var dir = std.Io.Dir.cwd().openDir(io, model_path, .{}) catch return error.InvalidPath;
    defer dir.close(io);

    var metadata = ModelMetadata.empty();
    errdefer metadata.deinit(allocator);
    metadata.config_path = try optionalJoinedFile(allocator, dir, io, model_path, "config.json");
    metadata.tokenizer_path = try optionalJoinedFile(allocator, dir, io, model_path, "tokenizer.json");
    metadata.generation_config_path = try optionalJoinedFile(allocator, dir, io, model_path, "generation_config.json");
    if (metadata.config_path) |config_path| {
        try parseModelConfig(allocator, io, config_path, &metadata);
    }
    return metadata;
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
    metadata: *ModelMetadata,
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
    metadata: *ModelMetadata,
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

fn singlePathList(
    allocator: std.mem.Allocator,
    item: []const u8,
) SessionError![][]u8 {
    const list = allocator.alloc([]u8, 1) catch return error.OutOfMemory;
    errdefer allocator.free(list);
    list[0] = allocator.dupe(u8, item) catch return error.OutOfMemory;
    return list;
}

fn freeStringList(allocator: std.mem.Allocator, items: [][]u8) void {
    freeStringsOnly(allocator, items);
    allocator.free(items);
}

fn freeStringsOnly(allocator: std.mem.Allocator, items: [][]u8) void {
    for (items) |item| {
        allocator.free(item);
    }
}

fn freeOptionalString(allocator: std.mem.Allocator, value: *?[]u8) void {
    if (value.*) |bytes| {
        allocator.free(bytes);
        value.* = null;
    }
}

fn loadWeights(
    allocator: std.mem.Allocator,
    weight_paths: []const []const u8,
) SessionError!Weights {
    if (weight_paths.len == 0) {
        return error.UnsupportedArtifact;
    }
    if (!linked or builtin.is_test) {
        return Weights.empty();
    }

    var weights = Weights{
        .params = native.mlx_map_string_to_array_new(),
        .metadata = native.mlx_map_string_to_string_new(),
        .loaded = true,
        .loaded_file_count = 0,
    };
    errdefer weights.deinit();
    if (weights.params.ctx == null or weights.metadata.ctx == null) {
        return error.MlxCallFailed;
    }

    const stream = native.mlx_default_gpu_stream_new();
    if (stream.ctx == null) {
        return error.MlxCallFailed;
    }
    defer _ = native.mlx_stream_free(stream);

    for (weight_paths, 0..) |weight_path, index| {
        const c_path = allocator.dupeZ(u8, weight_path) catch return error.OutOfMemory;
        defer allocator.free(c_path);
        if (index == 0) {
            if (native.mlx_load_safetensors(&weights.params, &weights.metadata, c_path.ptr, stream) != 0) {
                return error.MlxCallFailed;
            }
        } else {
            var shard_params = native.mlx_map_string_to_array_new();
            var shard_metadata = native.mlx_map_string_to_string_new();
            defer _ = native.mlx_map_string_to_array_free(shard_params);
            defer _ = native.mlx_map_string_to_string_free(shard_metadata);
            if (shard_params.ctx == null or shard_metadata.ctx == null) {
                return error.MlxCallFailed;
            }
            if (native.mlx_load_safetensors(&shard_params, &shard_metadata, c_path.ptr, stream) != 0) {
                return error.MlxCallFailed;
            }
            try mergeArrayMap(weights.params, shard_params);
            try mergeStringMap(weights.metadata, shard_metadata);
        }
        weights.loaded_file_count += 1;
    }
    return weights;
}

fn mergeArrayMap(target: MlxMapStringToArray, source: MlxMapStringToArray) SessionError!void {
    if (!linked or builtin.is_test) {
        return;
    }
    const iterator = native.mlx_map_string_to_array_iterator_new(source);
    if (iterator.ctx == null) {
        return error.MlxCallFailed;
    }
    defer _ = native.mlx_map_string_to_array_iterator_free(iterator);

    while (true) {
        var key: [*c]const u8 = null;
        var value = MlxArray{ .ctx = null };
        const status = native.mlx_map_string_to_array_iterator_next(&key, &value, iterator);
        if (status == 2) {
            return;
        }
        if (status != 0 or key == null or value.ctx == null) {
            if (value.ctx != null) {
                _ = native.mlx_array_free(value);
            }
            return error.MlxCallFailed;
        }
        if (native.mlx_map_string_to_array_insert(target, key, value) != 0) {
            _ = native.mlx_array_free(value);
            return error.MlxCallFailed;
        }
        _ = native.mlx_array_free(value);
    }
}

fn mergeStringMap(target: MlxMapStringToString, source: MlxMapStringToString) SessionError!void {
    if (!linked or builtin.is_test) {
        return;
    }
    const iterator = native.mlx_map_string_to_string_iterator_new(source);
    if (iterator.ctx == null) {
        return error.MlxCallFailed;
    }
    defer _ = native.mlx_map_string_to_string_iterator_free(iterator);

    while (true) {
        var key: [*c]const u8 = null;
        var value: [*c]const u8 = null;
        const status = native.mlx_map_string_to_string_iterator_next(&key, &value, iterator);
        if (status == 2) {
            return;
        }
        if (status != 0 or key == null or value == null) {
            return error.MlxCallFailed;
        }
        if (native.mlx_map_string_to_string_insert(target, key, value) != 0) {
            return error.MlxCallFailed;
        }
    }
}

pub const TensorError = error{
    MlxUnavailable,
    UnsupportedDtype,
    InvalidTensor,
    ShapeOutOfRange,
    ByteLengthMismatch,
    MlxCallFailed,
    OutOfMemory,
};

pub const ExecutionError = mlx_output.OutputError || error{
    WeightsUnavailable,
    InvalidInput,
    MissingWeight,
    UnsupportedArchitecture,
    ExecutorNotImplemented,
};

pub fn executeSession(
    allocator: std.mem.Allocator,
    session: *const Session,
    batch: InputBatch,
) ExecutionError!OutputBatch {
    if (!linked or builtin.is_test) {
        return error.MlxUnavailable;
    }
    if (!session.weights.loaded) {
        return error.WeightsUnavailable;
    }
    if (session.metadata.architecture == null and session.metadata.model_type == null) {
        return error.UnsupportedArchitecture;
    }
    if (isLinearExecutor(session)) {
        return executeLinear(allocator, session, batch);
    }
    return error.ExecutorNotImplemented;
}

pub fn executionErrorMessage(err: ExecutionError) []const u8 {
    return switch (err) {
        error.WeightsUnavailable => "Zig-owned MLX backend has no loaded weight maps for execution.",
        error.InvalidInput => "Zig-owned MLX backend received no usable executor input.",
        error.MissingWeight => "Zig-owned MLX backend could not find a required executor weight.",
        error.UnsupportedArchitecture => "Zig-owned MLX backend could not identify a supported model architecture.",
        error.ExecutorNotImplemented => "Zig-owned MLX backend has not registered an executor for this model architecture yet.",
        error.MlxUnavailable,
        error.UnsupportedDtype,
        error.InvalidArray,
        error.ShapeOutOfRange,
        error.MlxCallFailed,
        error.OutOfMemory,
        => mlx_output.errorMessage(@errorCast(err)),
    };
}

fn isLinearExecutor(session: *const Session) bool {
    if (session.metadata.model_type) |model_type| {
        if (std.mem.eql(u8, model_type, "dart_inference_linear")) {
            return true;
        }
    }
    if (session.metadata.architecture) |architecture| {
        return std.mem.eql(u8, architecture, "DartInferenceLinear");
    }
    return false;
}

fn executorKind(session: *const Session) []const u8 {
    return if (isLinearExecutor(session)) "linear" else "unregistered";
}

fn executeLinear(
    allocator: std.mem.Allocator,
    session: *const Session,
    batch: InputBatch,
) ExecutionError!OutputBatch {
    if (batch.arrays.len == 0) {
        return error.InvalidInput;
    }
    const stream = native.mlx_default_gpu_stream_new();
    if (stream.ctx == null) {
        return error.MlxCallFailed;
    }
    defer _ = native.mlx_stream_free(stream);

    const weight = try getRequiredWeight(session.weights.params, "weight");
    defer _ = native.mlx_array_free(weight);
    const bias = getOptionalWeight(session.weights.params, "bias") catch return error.MlxCallFailed;
    defer if (bias.ctx != null) {
        _ = native.mlx_array_free(bias);
    };

    var output = MlxArray{ .ctx = null };
    errdefer if (output.ctx != null) {
        _ = native.mlx_array_free(output);
    };
    if (native.mlx_matmul(&output, batch.arrays[0], weight, stream) != 0) {
        return error.MlxCallFailed;
    }
    if (bias.ctx != null) {
        var biased = MlxArray{ .ctx = null };
        if (native.mlx_add(&biased, output, bias, stream) != 0) {
            return error.MlxCallFailed;
        }
        _ = native.mlx_array_free(output);
        output = biased;
    }
    defer _ = native.mlx_array_free(output);

    const tensors = allocator.alloc(OutputTensor, 1) catch return error.OutOfMemory;
    errdefer allocator.free(tensors);
    tensors[0] = try mlx_output.materializeArray(allocator, "output", output);
    return .{ .allocator = allocator, .tensors = tensors };
}

fn getRequiredWeight(map: MlxMapStringToArray, key: [*c]const u8) ExecutionError!MlxArray {
    const value = try getOptionalWeight(map, key);
    if (value.ctx == null) {
        return error.MissingWeight;
    }
    return value;
}

fn getOptionalWeight(map: MlxMapStringToArray, key: [*c]const u8) ExecutionError!MlxArray {
    var value = MlxArray{ .ctx = null };
    const status = native.mlx_map_string_to_array_get(&value, map, key);
    if (status == 2) {
        return value;
    }
    if (status != 0) {
        return error.MlxCallFailed;
    }
    return value;
}

pub const InputBatch = struct {
    allocator: std.mem.Allocator,
    arrays: []MlxArray,

    pub fn deinit(self: InputBatch) void {
        if (linked and !builtin.is_test) {
            for (self.arrays) |array| {
                if (array.ctx != null) {
                    _ = native.mlx_array_free(array);
                }
            }
        }
        self.allocator.free(self.arrays);
    }
};

pub fn prepareInputBatch(
    allocator: std.mem.Allocator,
    tensors: [*c]const InputTensor,
    count: usize,
) TensorError!InputBatch {
    if (!linked or builtin.is_test) {
        return error.MlxUnavailable;
    }
    if (count > 0 and tensors == null) {
        return error.InvalidTensor;
    }
    const arrays = allocator.alloc(MlxArray, count) catch return error.OutOfMemory;
    errdefer allocator.free(arrays);
    var produced: usize = 0;
    errdefer {
        var index: usize = 0;
        while (index < produced) : (index += 1) {
            if (arrays[index].ctx != null) {
                _ = native.mlx_array_free(arrays[index]);
            }
        }
    }
    while (produced < count) : (produced += 1) {
        arrays[produced] = try createArray(allocator, tensors[produced]);
    }
    return .{ .allocator = allocator, .arrays = arrays };
}

fn createArray(
    allocator: std.mem.Allocator,
    tensor: InputTensor,
) TensorError!MlxArray {
    try validateInputTensor(tensor);
    const dtype = try mlxDtype(tensor.dtype);

    const rank: usize = @intCast(tensor.rank);
    var shape = allocator.alloc(c_int, rank) catch return error.OutOfMemory;
    defer allocator.free(shape);
    var index: usize = 0;
    while (index < rank) : (index += 1) {
        const dim = tensor.shape[index];
        if (dim < 0) {
            return error.InvalidTensor;
        }
        if (dim > std.math.maxInt(c_int)) {
            return error.ShapeOutOfRange;
        }
        shape[index] = @intCast(dim);
    }

    const data = if (tensor.data) |ptr| @constCast(ptr) else null;
    const array = native.mlx_array_new_data_managed_payload(
        data,
        if (rank == 0) null else shape.ptr,
        @intCast(rank),
        dtype,
        data,
        retainInputBuffer,
    );
    if (array.ctx == null) {
        return error.MlxCallFailed;
    }
    return array;
}

pub fn validateInputTensor(tensor: InputTensor) TensorError!void {
    if (!validTensor(tensor)) {
        return error.InvalidTensor;
    }
    const dtype = try mlxDtype(tensor.dtype);
    const expected = try expectedByteLength(tensor, dtypeSize(dtype));
    if (@as(usize, @intCast(tensor.byte_length)) != expected) {
        return error.ByteLengthMismatch;
    }
}

fn validTensor(tensor: InputTensor) bool {
    if (tensor.rank < 0 or tensor.byte_length < 0) {
        return false;
    }
    if (tensor.rank > 0 and tensor.shape == null) {
        return false;
    }
    if (tensor.byte_length > 0 and tensor.data == null) {
        return false;
    }
    return true;
}

fn expectedByteLength(tensor: InputTensor, item_size: usize) TensorError!usize {
    var elements: usize = 1;
    const rank: usize = @intCast(tensor.rank);
    var index: usize = 0;
    while (index < rank) : (index += 1) {
        const dim = tensor.shape[index];
        if (dim < 0) {
            return error.InvalidTensor;
        }
        const dim_usize: usize = @intCast(dim);
        elements = std.math.mul(usize, elements, dim_usize) catch
            return error.InvalidTensor;
    }
    return std.math.mul(usize, elements, item_size) catch error.InvalidTensor;
}

fn retainInputBuffer(payload: ?*anyopaque) callconv(.c) void {
    _ = payload;
}

pub fn mlxDtype(runtime_dtype: i32) TensorError!MlxDtype {
    return switch (runtime_dtype) {
        1 => .float32,
        2 => .int32,
        3 => .int64,
        4 => .uint8,
        5 => .float64,
        6 => .float16,
        7 => .bool,
        else => error.UnsupportedDtype,
    };
}

fn dtypeSize(dtype: MlxDtype) usize {
    return mlx_c.dtypeSize(dtype);
}

pub fn tensorErrorMessage(err: TensorError) []const u8 {
    return switch (err) {
        error.MlxUnavailable => unavailableMessage(),
        error.UnsupportedDtype => "Zig-owned MLX backend received an unsupported tensor dtype.",
        error.InvalidTensor => "Zig-owned MLX backend received an invalid tensor.",
        error.ShapeOutOfRange => "Zig-owned MLX backend received a tensor shape that does not fit mlx-c.",
        error.ByteLengthMismatch => "Zig-owned MLX backend received a tensor byte length that does not match dtype and shape.",
        error.MlxCallFailed => "Zig-owned MLX backend failed while converting runtime tensors through mlx-c.",
        error.OutOfMemory => "Zig-owned MLX backend ran out of memory.",
    };
}

pub fn unavailableMessage() []const u8 {
    return if (linked)
        "Zig-owned MLX backend reached mlx-c, but model execution is not implemented yet."
    else
        "Zig-owned MLX backend is only linked for Apple targets; mlx-c must be called from native/zig_runtime, not Dart.";
}

pub fn statusJson() []const u8 {
    return status_json;
}

test "MLX status remains Zig-owned" {
    try std.testing.expectEqualStrings("zig", owner);
    try std.testing.expectEqualStrings("mlx-c", api);
    try std.testing.expect(!enabled);
}

test "MLX session discovers model.safetensors directory artifacts" {
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
    const session = try createSession(std.testing.allocator, std.testing.io, model_path);
    defer session.deinit();

    try std.testing.expectEqual(ArtifactKind.directory_model_safetensors, session.artifact_kind);
    try std.testing.expectEqual(@as(usize, 1), session.weight_file_count);
    try std.testing.expect(!session.weights.loaded);
    try std.testing.expectEqualStrings("qwen3", session.metadata.model_type.?);
    try std.testing.expectEqualStrings("Qwen3ForCausalLM", session.metadata.architecture.?);
    try std.testing.expectEqualStrings("affine", session.metadata.quantization_mode.?);
    try std.testing.expectEqual(@as(i64, 4), session.metadata.quantization_bits.?);
    try std.testing.expectEqual(@as(i64, 64), session.metadata.quantization_group_size.?);
    const json = try sessionDiagnosticsJson(session, std.testing.allocator);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"artifact_kind\":\"directory_model_safetensors\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"weights_loaded\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"has_config\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"has_tokenizer\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"has_generation_config\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"model_type\":\"qwen3\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"quantization_mode\":\"affine\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"quantization_bits\":4") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"executor_kind\":\"unregistered\"") != null);
}

test "MLX session discovers sharded safetensors directory artifacts" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();
    const index_file = try tmp.dir.createFile(std.testing.io, "model.safetensors.index.json", .{});
    index_file.close(std.testing.io);
    const shard_a = try tmp.dir.createFile(std.testing.io, "model-00001-of-00002.safetensors", .{});
    shard_a.close(std.testing.io);
    const shard_b = try tmp.dir.createFile(std.testing.io, "model-00002-of-00002.safetensors", .{});
    shard_b.close(std.testing.io);

    const model_path = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(model_path);
    const session = try createSession(std.testing.allocator, std.testing.io, model_path);
    defer session.deinit();

    try std.testing.expectEqual(ArtifactKind.directory_sharded_safetensors, session.artifact_kind);
    try std.testing.expectEqual(@as(usize, 2), session.weight_file_count);
    try std.testing.expectEqual(@as(usize, 2), session.weight_paths.len);
    try std.testing.expect(std.mem.endsWith(u8, session.weight_paths[0], "model-00001-of-00002.safetensors"));
}

test "runtime dtype maps to mlx-c dtype" {
    try std.testing.expectEqual(MlxDtype.float32, try mlxDtype(1));
    try std.testing.expectEqual(MlxDtype.int32, try mlxDtype(2));
    try std.testing.expectEqual(MlxDtype.int64, try mlxDtype(3));
    try std.testing.expectEqual(MlxDtype.uint8, try mlxDtype(4));
    try std.testing.expectEqual(MlxDtype.float64, try mlxDtype(5));
    try std.testing.expectEqual(MlxDtype.float16, try mlxDtype(6));
    try std.testing.expectEqual(MlxDtype.bool, try mlxDtype(7));
    try std.testing.expectError(error.UnsupportedDtype, mlxDtype(99));
}

test "runtime tensor byte length is validated before mlx-c conversion" {
    const shape = [_]i64{ 2, 3 };
    const tensor = InputTensor{
        .dtype = 1,
        .rank = 2,
        .shape = &shape,
        .byte_length = 24,
        .data = @ptrFromInt(0x1),
    };
    try std.testing.expectEqual(@as(usize, 24), try expectedByteLength(tensor, dtypeSize(.float32)));
    try validateInputTensor(tensor);

    const bad = InputTensor{
        .dtype = 1,
        .rank = 2,
        .shape = &shape,
        .byte_length = 20,
        .data = @ptrFromInt(0x1),
    };
    try std.testing.expectError(error.ByteLengthMismatch, validateInputTensor(bad));
}
