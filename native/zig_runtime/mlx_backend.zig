const std = @import("std");
const builtin = @import("builtin");

pub const api = "mlx-c";
pub const owner = "zig";
pub const linked = builtin.os.tag == .macos or builtin.os.tag == .ios;
pub const enabled = false;
pub const status_json = if (linked)
    "{\"owner\":\"zig\",\"api\":\"mlx-c\",\"linked\":true,\"enabled\":false}"
else
    "{\"owner\":\"zig\",\"api\":\"mlx-c\",\"linked\":false,\"enabled\":false}";

const MlxString = extern struct {
    ctx: ?*anyopaque,
};

const MlxArray = extern struct {
    ctx: ?*anyopaque,
};

pub const InputTensor = extern struct {
    dtype: i32,
    rank: i32,
    shape: [*c]const i64,
    byte_length: isize,
    data: ?*const anyopaque,
};

const MlxDtype = enum(c_int) {
    bool = 0,
    uint8 = 1,
    int32 = 7,
    int64 = 8,
    float16 = 9,
    float32 = 10,
    float64 = 11,
};

const MlxManagedDtor = *const fn (?*anyopaque) callconv(.c) void;
const path = std.Io.Dir.path;

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
    OutOfMemory,
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    model_path: []u8,
    primary_weight_path: []u8,
    artifact_kind: ArtifactKind,
    weight_file_count: usize,

    pub fn deinit(self: *Session) void {
        self.allocator.free(self.model_path);
        self.allocator.free(self.primary_weight_path);
        self.allocator.destroy(self);
    }
};

const ArtifactLayout = struct {
    kind: ArtifactKind,
    primary_weight_path: []u8,
    weight_file_count: usize,
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
    errdefer allocator.free(layout.primary_weight_path);
    const session = allocator.create(Session) catch return error.OutOfMemory;
    errdefer allocator.destroy(session);
    session.* = .{
        .allocator = allocator,
        .model_path = allocator.dupe(u8, model_path) catch return error.OutOfMemory,
        .primary_weight_path = layout.primary_weight_path,
        .artifact_kind = layout.kind,
        .weight_file_count = layout.weight_file_count,
    };
    return session;
}

pub fn sessionDiagnosticsJson(
    session: *const Session,
    allocator: std.mem.Allocator,
) std.mem.Allocator.Error![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{{\"artifact_kind\":\"{s}\",\"weight_file_count\":{d}}}",
        .{ artifactKindName(session.artifact_kind), session.weight_file_count },
    );
}

pub fn sessionErrorMessage(err: SessionError) []const u8 {
    return switch (err) {
        error.InvalidPath => "Zig-owned MLX backend requires a resolved local MLX artifact path.",
        error.ArtifactNotFound => "Zig-owned MLX backend could not find the local MLX artifact.",
        error.UnsupportedArtifact => "Zig-owned MLX backend requires local MLX safetensors weights.",
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
        return .{
            .kind = .directory_model_safetensors,
            .primary_weight_path = try joinPath(allocator, model_path, "model.safetensors"),
            .weight_file_count = 1,
        };
    }

    if (fileExists(dir, io, "model.safetensors.index.json")) {
        const scan = try scanSafetensors(allocator, io, dir, model_path);
        defer if (scan.first_weight_path) |first| allocator.free(first);
        if (scan.count == 0) {
            return error.UnsupportedArtifact;
        }
        return .{
            .kind = .directory_sharded_safetensors,
            .primary_weight_path = try joinPath(allocator, model_path, "model.safetensors.index.json"),
            .weight_file_count = scan.count,
        };
    }

    const scan = try scanSafetensors(allocator, io, dir, model_path);
    if (scan.first_weight_path) |first| {
        return .{
            .kind = .directory_safetensors,
            .primary_weight_path = first,
            .weight_file_count = scan.count,
        };
    }
    return error.UnsupportedArtifact;
}

const SafetensorsScan = struct {
    count: usize,
    first_weight_path: ?[]u8,
};

fn scanSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    model_path: []const u8,
) SessionError!SafetensorsScan {
    var iter = dir.iterate();
    var count: usize = 0;
    var first: ?[]u8 = null;
    errdefer if (first) |value| allocator.free(value);

    while (iter.next(io) catch return error.InvalidPath) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".safetensors")) {
            continue;
        }
        count += 1;
        if (first == null) {
            first = try joinPath(allocator, model_path, entry.name);
        }
    }
    return .{ .count = count, .first_weight_path = first };
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

pub const TensorError = error{
    MlxUnavailable,
    UnsupportedDtype,
    InvalidTensor,
    ShapeOutOfRange,
    ByteLengthMismatch,
    MlxCallFailed,
    OutOfMemory,
};

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
    return switch (dtype) {
        .bool, .uint8 => 1,
        .float16 => 2,
        .int32, .float32 => 4,
        .int64, .float64 => 8,
    };
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

    const model_path = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(model_path);
    const session = try createSession(std.testing.allocator, std.testing.io, model_path);
    defer session.deinit();

    try std.testing.expectEqual(ArtifactKind.directory_model_safetensors, session.artifact_kind);
    try std.testing.expectEqual(@as(usize, 1), session.weight_file_count);
    const json = try sessionDiagnosticsJson(session, std.testing.allocator);
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"artifact_kind\":\"directory_model_safetensors\"") != null);
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
