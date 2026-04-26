const std = @import("std");
const builtin = @import("builtin");
const c = @import("mlx_c.zig");

const has_float16_data = c.linked and builtin.cpu.arch == .aarch64;

const native = if (c.linked and !builtin.is_test) struct {
    extern fn mlx_array_eval(arr: c.MlxArray) c_int;
    extern fn mlx_array_dtype(arr: c.MlxArray) c.MlxDtype;
    extern fn mlx_array_ndim(arr: c.MlxArray) usize;
    extern fn mlx_array_shape(arr: c.MlxArray) [*c]const c_int;
    extern fn mlx_array_nbytes(arr: c.MlxArray) usize;
    extern fn mlx_array_data_bool(arr: c.MlxArray) [*c]const bool;
    extern fn mlx_array_data_uint8(arr: c.MlxArray) [*c]const u8;
    extern fn mlx_array_data_int32(arr: c.MlxArray) [*c]const i32;
    extern fn mlx_array_data_int64(arr: c.MlxArray) [*c]const i64;
    extern fn mlx_array_data_float32(arr: c.MlxArray) [*c]const f32;
    extern fn mlx_array_data_float64(arr: c.MlxArray) [*c]const f64;
} else struct {};

const native_float16 = if (has_float16_data and !builtin.is_test) struct {
    extern fn mlx_array_data_float16(arr: c.MlxArray) [*c]const u16;
} else struct {};

pub const OutputError = error{
    MlxUnavailable,
    UnsupportedDtype,
    InvalidArray,
    ShapeOutOfRange,
    MlxCallFailed,
    OutOfMemory,
};

pub const OutputTensor = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    dtype: i32,
    shape: []i64,
    bytes: []u8,

    pub fn deinit(self: *OutputTensor) void {
        self.allocator.free(self.name);
        self.allocator.free(self.shape);
        self.allocator.free(self.bytes);
        self.* = .{
            .allocator = self.allocator,
            .name = &.{},
            .dtype = 0,
            .shape = &.{},
            .bytes = &.{},
        };
    }
};

pub const OutputBatch = struct {
    allocator: std.mem.Allocator,
    tensors: []OutputTensor,

    pub fn deinit(self: OutputBatch) void {
        for (self.tensors) |*tensor| {
            tensor.deinit();
        }
        self.allocator.free(self.tensors);
    }
};

pub fn materializeArray(
    allocator: std.mem.Allocator,
    name: []const u8,
    array: c.MlxArray,
) OutputError!OutputTensor {
    if (!c.linked or builtin.is_test) {
        return error.MlxUnavailable;
    }
    if (array.ctx == null) {
        return error.InvalidArray;
    }
    if (native.mlx_array_eval(array) != 0) {
        return error.MlxCallFailed;
    }

    const dtype = native.mlx_array_dtype(array);
    const runtime_dtype = try runtimeDtype(dtype);
    const rank = native.mlx_array_ndim(array);
    if (rank > std.math.maxInt(i32)) {
        return error.ShapeOutOfRange;
    }

    const shape = allocator.alloc(i64, rank) catch return error.OutOfMemory;
    errdefer allocator.free(shape);
    if (rank > 0) {
        const raw_shape = native.mlx_array_shape(array);
        if (raw_shape == null) {
            return error.MlxCallFailed;
        }
        for (shape, 0..) |*dim, index| {
            const value = raw_shape[index];
            if (value < 0) {
                return error.InvalidArray;
            }
            dim.* = value;
        }
    }

    const nbytes = native.mlx_array_nbytes(array);
    const bytes = allocator.alloc(u8, nbytes) catch return error.OutOfMemory;
    errdefer allocator.free(bytes);
    if (nbytes > 0) {
        const data = dataPointer(array, dtype) orelse return error.MlxCallFailed;
        const source: [*]const u8 = @ptrCast(data);
        @memcpy(bytes, source[0..nbytes]);
    }

    const name_copy = allocator.dupe(u8, name) catch return error.OutOfMemory;
    return .{
        .allocator = allocator,
        .name = name_copy,
        .dtype = runtime_dtype,
        .shape = shape,
        .bytes = bytes,
    };
}

pub fn runtimeDtype(dtype: c.MlxDtype) OutputError!i32 {
    return switch (dtype) {
        .float32 => 1,
        .int32 => 2,
        .int64 => 3,
        .uint8 => 4,
        .float64 => 5,
        .float16 => if (has_float16_data) 6 else error.UnsupportedDtype,
        .bool => 7,
        else => error.UnsupportedDtype,
    };
}

pub fn errorMessage(err: OutputError) []const u8 {
    return switch (err) {
        error.MlxUnavailable => "Zig-owned MLX output materialization is only available when mlx-c is linked.",
        error.UnsupportedDtype => "Zig-owned MLX backend produced an unsupported output dtype.",
        error.InvalidArray => "Zig-owned MLX backend produced an invalid output array.",
        error.ShapeOutOfRange => "Zig-owned MLX backend produced an output rank that does not fit the runtime ABI.",
        error.MlxCallFailed => "Zig-owned MLX backend failed while materializing output arrays through mlx-c.",
        error.OutOfMemory => "Zig-owned MLX backend ran out of memory while materializing outputs.",
    };
}

fn dataPointer(array: c.MlxArray, dtype: c.MlxDtype) ?*const anyopaque {
    return switch (dtype) {
        .bool => native.mlx_array_data_bool(array),
        .uint8 => native.mlx_array_data_uint8(array),
        .int32 => native.mlx_array_data_int32(array),
        .int64 => native.mlx_array_data_int64(array),
        .float16 => if (has_float16_data) native_float16.mlx_array_data_float16(array) else null,
        .float32 => native.mlx_array_data_float32(array),
        .float64 => native.mlx_array_data_float64(array),
        else => null,
    };
}

test "MLX output dtype maps to runtime dtype ids" {
    try std.testing.expectEqual(@as(i32, 1), try runtimeDtype(.float32));
    try std.testing.expectEqual(@as(i32, 2), try runtimeDtype(.int32));
    try std.testing.expectEqual(@as(i32, 3), try runtimeDtype(.int64));
    try std.testing.expectEqual(@as(i32, 4), try runtimeDtype(.uint8));
    try std.testing.expectEqual(@as(i32, 5), try runtimeDtype(.float64));
    try std.testing.expectEqual(@as(i32, 7), try runtimeDtype(.bool));
    if (has_float16_data) {
        try std.testing.expectEqual(@as(i32, 6), try runtimeDtype(.float16));
    } else {
        try std.testing.expectError(error.UnsupportedDtype, runtimeDtype(.float16));
    }
    try std.testing.expectError(error.UnsupportedDtype, runtimeDtype(.complex64));
}
