const std = @import("std");

pub const Dtype = enum(i32) {
    float32 = 1,
    int32 = 2,
    int64 = 3,
    uint8 = 4,
    float64 = 5,
    float16 = 6,
    boolean = 7,
};

pub const NativeTensor = extern struct {
    dtype: i32,
    rank: i32,
    shape: [*c]i64,
    byte_length: isize,
    data: ?*anyopaque,
};

pub const NamedTensor = extern struct {
    name: [*c]u8,
    tensor: NativeTensor,
};

pub const TensorAllocError = error{
    UnsupportedDtype,
    InvalidShape,
    ByteLengthOverflow,
    OutOfMemory,
};

pub fn copyString(value: []const u8) [*c]u8 {
    const raw = std.c.malloc(value.len + 1) orelse return null;
    const bytes: [*]u8 = @ptrCast(raw);
    @memcpy(bytes[0..value.len], value);
    bytes[value.len] = 0;
    return @ptrCast(bytes);
}

pub fn freeString(value: [*c]u8) void {
    if (value != null) {
        std.c.free(value);
    }
}

pub fn copyCString(value: [*c]const u8) [*c]u8 {
    if (value == null) {
        return copyString("");
    }
    const len = std.mem.len(value);
    return copyString(value[0..len]);
}

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = copyString(message);
    }
}

pub fn dtypeSize(dtype: i32) usize {
    return switch (dtype) {
        @intFromEnum(Dtype.float32), @intFromEnum(Dtype.int32) => 4,
        @intFromEnum(Dtype.int64), @intFromEnum(Dtype.float64) => 8,
        @intFromEnum(Dtype.uint8), @intFromEnum(Dtype.boolean) => 1,
        @intFromEnum(Dtype.float16) => 2,
        else => 0,
    };
}

pub fn tensorByteLength(
    dtype: i32,
    shape: [*c]const i64,
    rank: i32,
) TensorAllocError!usize {
    const item_size = dtypeSize(dtype);
    if (item_size == 0) {
        return error.UnsupportedDtype;
    }
    if (rank < 0) {
        return error.InvalidShape;
    }
    if (rank > 0 and shape == null) {
        return error.InvalidShape;
    }

    var elements: usize = 1;
    const count: usize = @intCast(rank);
    var index: usize = 0;
    while (index < count) : (index += 1) {
        const dim = shape[index];
        if (dim < 0) {
            return error.InvalidShape;
        }
        const dim_usize: usize = @intCast(dim);
        elements = std.math.mul(usize, elements, dim_usize) catch
            return error.ByteLengthOverflow;
    }
    return std.math.mul(usize, elements, item_size) catch
        error.ByteLengthOverflow;
}

pub fn tensorAllocErrorMessage(err: TensorAllocError) []const u8 {
    return switch (err) {
        error.UnsupportedDtype => "Zig runtime tensor allocation received an unsupported dtype.",
        error.InvalidShape => "Zig runtime tensor allocation received an invalid shape.",
        error.ByteLengthOverflow => "Zig runtime tensor allocation byte length overflowed.",
        error.OutOfMemory => "Zig runtime tensor allocation ran out of memory.",
    };
}

pub fn validTensor(tensor: NativeTensor) bool {
    if (tensor.rank < 0 or tensor.byte_length < 0) {
        return false;
    }
    if (tensor.rank > 0 and tensor.shape == null) {
        return false;
    }
    if (tensor.byte_length > 0 and tensor.data == null) {
        return false;
    }
    const expected = tensorByteLength(
        tensor.dtype,
        @ptrCast(tensor.shape),
        tensor.rank,
    ) catch return false;
    return @as(usize, @intCast(tensor.byte_length)) == expected;
}

pub fn copyTensor(input: NamedTensor) ?NamedTensor {
    if (!validTensor(input.tensor)) {
        return null;
    }

    const name = copyCString(@ptrCast(input.name));
    if (name == null) {
        return null;
    }

    var output = NamedTensor{
        .name = name,
        .tensor = .{
            .dtype = input.tensor.dtype,
            .rank = input.tensor.rank,
            .shape = null,
            .byte_length = input.tensor.byte_length,
            .data = null,
        },
    };

    if (input.tensor.rank > 0) {
        const rank: usize = @intCast(input.tensor.rank);
        const raw_shape = std.c.malloc(@sizeOf(i64) * rank) orelse {
            freeString(output.name);
            return null;
        };
        const shape: [*]i64 = @ptrCast(@alignCast(raw_shape));
        @memcpy(shape[0..rank], input.tensor.shape[0..rank]);
        output.tensor.shape = @ptrCast(shape);
    }

    if (input.tensor.byte_length > 0) {
        const byte_len: usize = @intCast(input.tensor.byte_length);
        const raw_data = std.c.malloc(byte_len) orelse {
            freeString(output.name);
            std.c.free(output.tensor.shape);
            return null;
        };
        const data: [*]u8 = @ptrCast(raw_data);
        const source: [*]const u8 = @ptrCast(input.tensor.data.?);
        @memcpy(data[0..byte_len], source[0..byte_len]);
        output.tensor.data = @ptrCast(data);
    }

    return output;
}

pub fn freeTensors(tensors: [*c]NamedTensor, count: isize) void {
    if (tensors == null) {
        return;
    }
    if (count <= 0) {
        std.c.free(tensors);
        return;
    }
    const item_count: usize = @intCast(count);
    var index: usize = 0;
    while (index < item_count) : (index += 1) {
        freeString(tensors[index].name);
        std.c.free(tensors[index].tensor.shape);
        std.c.free(tensors[index].tensor.data);
    }
    std.c.free(tensors);
}

pub fn alloc(byte_length: isize) ?*anyopaque {
    if (byte_length <= 0) {
        return null;
    }
    const len: usize = @intCast(byte_length);
    return std.c.malloc(len);
}

pub fn allocTensor(
    dtype: i32,
    shape: [*c]const i64,
    rank: i32,
    byte_length_out: ?*isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    const out = byte_length_out orelse {
        setError(error_out, "byte_length_out is null");
        return null;
    };
    out.* = 0;
    const byte_length = tensorByteLength(dtype, shape, rank) catch |err| {
        setError(error_out, tensorAllocErrorMessage(err));
        return null;
    };
    if (byte_length > std.math.maxInt(isize)) {
        setError(error_out, tensorAllocErrorMessage(error.ByteLengthOverflow));
        return null;
    }
    out.* = @intCast(byte_length);
    if (byte_length == 0) {
        return null;
    }
    return std.c.malloc(byte_length) orelse {
        out.* = 0;
        setError(error_out, tensorAllocErrorMessage(error.OutOfMemory));
        return null;
    };
}

pub fn freeBuf(value: ?*anyopaque) void {
    if (value) |ptr| {
        std.c.free(ptr);
    }
}

test "runtime tensor buffer allocation computes byte length" {
    const shape = [_]i64{ 2, 3 };
    const byte_length = try tensorByteLength(
        @intFromEnum(Dtype.float32),
        shape[0..].ptr,
        @intCast(shape.len),
    );
    try std.testing.expectEqual(@as(usize, 24), byte_length);
}

test "runtime tensor buffer allocation owns native memory" {
    const shape = [_]i64{ 2, 3 };
    var byte_length: isize = 0;
    var error_value: [*c]u8 = null;
    const pointer = allocTensor(
        @intFromEnum(Dtype.float32),
        shape[0..].ptr,
        @intCast(shape.len),
        &byte_length,
        &error_value,
    );
    defer freeBuf(pointer);
    defer freeString(error_value);
    try std.testing.expect(pointer != null);
    try std.testing.expectEqual(@as(isize, 24), byte_length);
    try std.testing.expect(error_value == null);
}

test "runtime tensor buffer allocation keeps zero-sized tensors allocation-free" {
    const shape = [_]i64{ 0, 3 };
    const byte_length = try tensorByteLength(
        @intFromEnum(Dtype.float32),
        shape[0..].ptr,
        @intCast(shape.len),
    );
    try std.testing.expectEqual(@as(usize, 0), byte_length);
}

test "runtime tensor buffer allocation rejects invalid shapes" {
    const shape = [_]i64{-1};
    try std.testing.expectError(
        error.InvalidShape,
        tensorByteLength(@intFromEnum(Dtype.float32), shape[0..].ptr, @intCast(shape.len)),
    );
}

test "runtime tensor buffer allocation reports native errors" {
    const shape = [_]i64{-1};
    var byte_length: isize = -1;
    var error_value: [*c]u8 = null;
    const pointer = allocTensor(
        @intFromEnum(Dtype.float32),
        shape[0..].ptr,
        @intCast(shape.len),
        &byte_length,
        &error_value,
    );
    defer freeString(error_value);
    try std.testing.expect(pointer == null);
    try std.testing.expectEqual(@as(isize, 0), byte_length);
    try std.testing.expect(error_value != null);
    try std.testing.expect(std.mem.indexOf(u8, std.mem.span(error_value), "invalid shape") != null);
}

test "runtime tensor validation rejects dtype shape byte mismatches" {
    var shape = [_]i64{2};
    const data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const invalid = NativeTensor{
        .dtype = @intFromEnum(Dtype.float32),
        .rank = @intCast(shape.len),
        .shape = shape[0..].ptr,
        .byte_length = 4,
        .data = @ptrCast(@constCast(data[0..].ptr)),
    };
    try std.testing.expect(!validTensor(invalid));

    var valid = invalid;
    valid.byte_length = 8;
    try std.testing.expect(validTensor(valid));
}
