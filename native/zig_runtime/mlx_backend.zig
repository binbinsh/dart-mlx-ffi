const std = @import("std");
const builtin = @import("builtin");
const mlx_artifact = @import("mlx_artifact.zig");
const mlx_c = @import("mlx_c.zig");
const mlx_input = @import("mlx_input.zig");
const mlx_output = @import("mlx_output.zig");

pub const api = mlx_c.api;
pub const owner = mlx_c.owner;
pub const linked = mlx_c.linked;
pub const enabled = mlx_c.enabled;
pub const MlxArray = mlx_c.MlxArray;
pub const MlxDtype = mlx_c.MlxDtype;
pub const InputBatch = mlx_input.InputBatch;
pub const InputTensor = mlx_input.InputTensor;
pub const OutputBatch = mlx_output.OutputBatch;
pub const OutputTensor = mlx_output.OutputTensor;
pub const TensorError = mlx_input.TensorError;
pub const mlxDtype = mlx_input.mlxDtype;
pub const prepareInputBatch = mlx_input.prepareInputBatch;
pub const tensorErrorMessage = mlx_input.tensorErrorMessage;

const MlxString = mlx_c.MlxString;
const MlxMapStringToArray = mlx_c.MlxMapStringToArray;
const MlxMapStringToString = mlx_c.MlxMapStringToString;
const MlxMapStringToArrayIterator = mlx_c.MlxMapStringToArrayIterator;
const MlxMapStringToStringIterator = mlx_c.MlxMapStringToStringIterator;
const MlxStream = mlx_c.MlxStream;
const MlxVectorArray = mlx_c.MlxVectorArray;
const MlxImportedFunction = mlx_c.MlxImportedFunction;

const native = if (linked and !builtin.is_test) struct {
    extern fn mlx_string_new() MlxString;
    extern fn mlx_string_data(str: MlxString) [*c]const u8;
    extern fn mlx_string_free(str: MlxString) c_int;
    extern fn mlx_version(str: *MlxString) c_int;
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
    extern fn mlx_vector_array_new_data(
        data: [*c]const MlxArray,
        size: usize,
    ) MlxVectorArray;
    extern fn mlx_vector_array_free(vec: MlxVectorArray) c_int;
    extern fn mlx_vector_array_size(vec: MlxVectorArray) usize;
    extern fn mlx_vector_array_get(
        result: *MlxArray,
        vec: MlxVectorArray,
        index: usize,
    ) c_int;
    extern fn mlx_imported_function_new(file: [*:0]const u8) MlxImportedFunction;
    extern fn mlx_imported_function_free(function: MlxImportedFunction) c_int;
    extern fn mlx_imported_function_apply(
        result: *MlxVectorArray,
        function: MlxImportedFunction,
        args: MlxVectorArray,
    ) c_int;
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

pub const ArtifactKind = mlx_artifact.ArtifactKind;
pub const SessionError = mlx_artifact.SessionError;
const ModelMetadata = mlx_artifact.Metadata;

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

pub const Session = struct {
    allocator: std.mem.Allocator,
    model_path: []u8,
    primary_artifact_path: []u8,
    function_path: ?[]u8,
    weight_paths: [][]u8,
    artifact_kind: ArtifactKind,
    weight_file_count: usize,
    weights: Weights,
    imported_function: MlxImportedFunction,
    metadata: ModelMetadata,

    pub fn deinit(self: *Session) void {
        self.metadata.deinit(self.allocator);
        if (linked and !builtin.is_test and self.imported_function.ctx != null) {
            _ = native.mlx_imported_function_free(self.imported_function);
        }
        self.weights.deinit();
        mlx_artifact.freeStringList(self.allocator, self.weight_paths);
        self.allocator.free(self.model_path);
        self.allocator.free(self.primary_artifact_path);
        mlx_artifact.freeOptionalString(self.allocator, &self.function_path);
        self.allocator.destroy(self);
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
    const layout = try mlx_artifact.discover(allocator, io, model_path);
    errdefer layout.deinit(allocator);
    var metadata = try mlx_artifact.loadMetadata(allocator, io, model_path);
    errdefer metadata.deinit(allocator);
    var weights = try loadWeights(allocator, layout.weight_paths);
    errdefer weights.deinit();
    const imported_function = try loadImportedFunction(allocator, layout.function_path);
    errdefer if (linked and !builtin.is_test and imported_function.ctx != null) {
        _ = native.mlx_imported_function_free(imported_function);
    };
    const model_path_copy = allocator.dupe(u8, model_path) catch return error.OutOfMemory;
    errdefer allocator.free(model_path_copy);
    const session = allocator.create(Session) catch return error.OutOfMemory;
    errdefer allocator.destroy(session);
    session.* = .{
        .allocator = allocator,
        .model_path = model_path_copy,
        .primary_artifact_path = layout.primary_artifact_path,
        .function_path = layout.function_path,
        .weight_paths = layout.weight_paths,
        .artifact_kind = layout.kind,
        .weight_file_count = layout.weight_file_count,
        .weights = weights,
        .imported_function = imported_function,
        .metadata = metadata,
    };
    return session;
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
    return mlx_artifact.kindName(kind);
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

fn loadWeights(
    allocator: std.mem.Allocator,
    weight_paths: []const []const u8,
) SessionError!Weights {
    if (weight_paths.len == 0) {
        return Weights.empty();
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

fn loadImportedFunction(
    allocator: std.mem.Allocator,
    function_path: ?[]const u8,
) SessionError!MlxImportedFunction {
    const path_value = function_path orelse return .{ .ctx = null };
    if (!linked or builtin.is_test) {
        return .{ .ctx = null };
    }
    const c_path = allocator.dupeZ(u8, path_value) catch return error.OutOfMemory;
    defer allocator.free(c_path);
    const function = native.mlx_imported_function_new(c_path.ptr);
    if (function.ctx == null) {
        return error.MlxCallFailed;
    }
    return function;
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

pub const ExecutionError = mlx_output.OutputError || error{
    WeightsUnavailable,
    FunctionUnavailable,
    InvalidInput,
    MissingInput,
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
    if (isFunctionArtifact(session.artifact_kind)) {
        return executeImportedFunction(allocator, session, batch);
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
        error.FunctionUnavailable => "Zig-owned MLX backend has no loaded imported function for execution.",
        error.InvalidInput => "Zig-owned MLX backend received no usable executor input.",
        error.MissingInput => "Zig-owned MLX backend could not match runtime inputs to the imported function input order.",
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

fn isFunctionArtifact(kind: ArtifactKind) bool {
    return kind == .single_mlx_function or kind == .directory_mlx_function;
}

pub fn executorKind(session: *const Session) []const u8 {
    if (isFunctionArtifact(session.artifact_kind)) {
        return "imported_function";
    }
    return if (isLinearExecutor(session)) "linear" else "unregistered";
}

fn executeImportedFunction(
    allocator: std.mem.Allocator,
    session: *const Session,
    batch: InputBatch,
) ExecutionError!OutputBatch {
    if (batch.arrays.len == 0) {
        return error.InvalidInput;
    }
    if (session.imported_function.ctx == null) {
        return error.FunctionUnavailable;
    }

    const ordered = try orderedFunctionArgs(allocator, session, batch);
    defer ordered.deinit(allocator);

    const args = native.mlx_vector_array_new_data(ordered.arrays.ptr, ordered.arrays.len);
    if (args.ctx == null) {
        return error.MlxCallFailed;
    }
    defer _ = native.mlx_vector_array_free(args);

    var result = MlxVectorArray{ .ctx = null };
    errdefer if (result.ctx != null) {
        _ = native.mlx_vector_array_free(result);
    };
    if (native.mlx_imported_function_apply(&result, session.imported_function, args) != 0) {
        return error.MlxCallFailed;
    }
    defer _ = native.mlx_vector_array_free(result);

    const output_count = native.mlx_vector_array_size(result);
    const tensors = allocator.alloc(OutputTensor, output_count) catch return error.OutOfMemory;
    errdefer allocator.free(tensors);
    var produced: usize = 0;
    errdefer {
        var index: usize = 0;
        while (index < produced) : (index += 1) {
            tensors[index].deinit();
        }
    }

    while (produced < output_count) : (produced += 1) {
        var output = MlxArray{ .ctx = null };
        if (native.mlx_vector_array_get(&output, result, produced) != 0 or output.ctx == null) {
            return error.MlxCallFailed;
        }
        defer _ = native.mlx_array_free(output);
        const name = try outputName(allocator, produced, output_count);
        defer allocator.free(name);
        tensors[produced] = try mlx_output.materializeArray(allocator, name, output);
    }
    return .{ .allocator = allocator, .tensors = tensors };
}

const FunctionArgs = struct {
    arrays: []const MlxArray,
    owned: bool,

    fn deinit(self: FunctionArgs, allocator: std.mem.Allocator) void {
        if (self.owned) {
            allocator.free(self.arrays);
        }
    }
};

fn orderedFunctionArgs(
    allocator: std.mem.Allocator,
    session: *const Session,
    batch: InputBatch,
) ExecutionError!FunctionArgs {
    const input_names = session.metadata.input_names orelse {
        return .{ .arrays = batch.arrays, .owned = false };
    };
    if (input_names.len == 0) {
        return .{ .arrays = batch.arrays, .owned = false };
    }
    if (input_names.len != batch.arrays.len or batch.names.len != batch.arrays.len) {
        return error.MissingInput;
    }

    const arrays = allocator.alloc(MlxArray, input_names.len) catch return error.OutOfMemory;
    errdefer allocator.free(arrays);
    for (input_names, 0..) |name, index| {
        const input_index = findInputIndex(batch.names, name) orelse return error.MissingInput;
        arrays[index] = batch.arrays[input_index];
    }
    return .{ .arrays = arrays, .owned = true };
}

fn findInputIndex(names: []const []const u8, wanted: []const u8) ?usize {
    for (names, 0..) |name, index| {
        if (std.mem.eql(u8, name, wanted)) {
            return index;
        }
    }
    return null;
}

fn outputName(allocator: std.mem.Allocator, index: usize, count: usize) ExecutionError![]u8 {
    if (count == 1 and index == 0) {
        return allocator.dupe(u8, "output") catch return error.OutOfMemory;
    }
    return std.fmt.allocPrint(allocator, "output_{d}", .{index}) catch return error.OutOfMemory;
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

pub fn unavailableMessage() []const u8 {
    return if (linked)
        "Zig-owned MLX backend reached mlx-c, but model execution is not implemented yet."
    else
        "Zig-owned MLX backend is only linked for Apple targets; mlx-c must be called from native/zig_runtime, not Dart.";
}

test "MLX status remains Zig-owned" {
    try std.testing.expectEqualStrings("zig", owner);
    try std.testing.expectEqualStrings("mlx-c", api);
    try std.testing.expectEqual(linked, enabled);
}

test "MLX imported function arguments follow parsed input order" {
    const allocator = std.testing.allocator;
    var input_names = try allocator.alloc([]u8, 2);
    input_names[0] = try allocator.dupe(u8, "input_ids");
    input_names[1] = try allocator.dupe(u8, "pixel_values");
    var metadata = ModelMetadata.empty();
    metadata.input_names = input_names;
    defer metadata.deinit(allocator);

    var arrays = [_]MlxArray{
        .{ .ctx = @ptrFromInt(0x11) },
        .{ .ctx = @ptrFromInt(0x22) },
    };
    var names = [_][]const u8{ "pixel_values", "input_ids" };
    const batch = InputBatch{
        .allocator = allocator,
        .names = names[0..],
        .arrays = arrays[0..],
    };
    const session = Session{
        .allocator = allocator,
        .model_path = &.{},
        .primary_artifact_path = &.{},
        .function_path = null,
        .weight_paths = &.{},
        .artifact_kind = .directory_mlx_function,
        .weight_file_count = 0,
        .weights = Weights.empty(),
        .imported_function = .{ .ctx = null },
        .metadata = metadata,
    };

    const ordered = try orderedFunctionArgs(allocator, &session, batch);
    defer ordered.deinit(allocator);
    try std.testing.expectEqual(arrays[1].ctx, ordered.arrays[0].ctx);
    try std.testing.expectEqual(arrays[0].ctx, ordered.arrays[1].ctx);
}
