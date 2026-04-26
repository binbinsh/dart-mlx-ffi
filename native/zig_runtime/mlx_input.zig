const std = @import("std");
const builtin = @import("builtin");
const c = @import("mlx_c.zig");

pub const InputTensor = extern struct {
    dtype: i32,
    rank: i32,
    shape: [*c]const i64,
    byte_length: isize,
    data: ?*const anyopaque,
};

pub const TensorError = error{
    MlxUnavailable,
    UnsupportedDtype,
    InvalidTensor,
    ShapeOutOfRange,
    ByteLengthMismatch,
    MlxCallFailed,
    OutOfMemory,
};

const MlxManagedDtor = *const fn (?*anyopaque) callconv(.c) void;

const native = if (c.linked and !builtin.is_test) struct {
    extern fn mlx_array_new_data_managed_payload(
        data: ?*anyopaque,
        shape: [*c]const c_int,
        dim: c_int,
        dtype: c.MlxDtype,
        payload: ?*anyopaque,
        dtor: MlxManagedDtor,
    ) c.MlxArray;
    extern fn mlx_array_free(arr: c.MlxArray) c_int;
} else struct {};

pub const InputBatch = struct {
    allocator: std.mem.Allocator,
    arrays: []c.MlxArray,

    pub fn deinit(self: InputBatch) void {
        if (c.linked and !builtin.is_test) {
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
    if (!c.linked or builtin.is_test) {
        return error.MlxUnavailable;
    }
    if (count > 0 and tensors == null) {
        return error.InvalidTensor;
    }
    const arrays = allocator.alloc(c.MlxArray, count) catch return error.OutOfMemory;
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
) TensorError!c.MlxArray {
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
    const expected = try expectedByteLength(tensor, c.dtypeSize(dtype));
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

pub fn mlxDtype(runtime_dtype: i32) TensorError!c.MlxDtype {
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

fn unavailableMessage() []const u8 {
    return if (c.linked)
        "Zig-owned MLX backend reached mlx-c, but model execution is not implemented yet."
    else
        "Zig-owned MLX backend is only linked for Apple targets; mlx-c must be called from native/zig_runtime, not Dart.";
}

test "runtime dtype maps to mlx-c dtype" {
    try std.testing.expectEqual(c.MlxDtype.float32, try mlxDtype(1));
    try std.testing.expectEqual(c.MlxDtype.int32, try mlxDtype(2));
    try std.testing.expectEqual(c.MlxDtype.int64, try mlxDtype(3));
    try std.testing.expectEqual(c.MlxDtype.uint8, try mlxDtype(4));
    try std.testing.expectEqual(c.MlxDtype.float64, try mlxDtype(5));
    try std.testing.expectEqual(c.MlxDtype.float16, try mlxDtype(6));
    try std.testing.expectEqual(c.MlxDtype.bool, try mlxDtype(7));
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
    try std.testing.expectEqual(@as(usize, 24), try expectedByteLength(tensor, c.dtypeSize(.float32)));
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
