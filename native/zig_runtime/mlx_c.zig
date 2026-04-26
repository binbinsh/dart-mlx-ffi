const builtin = @import("builtin");

pub const api = "mlx-c";
pub const owner = "zig";
pub const linked = builtin.os.tag == .macos or builtin.os.tag == .ios;
pub const enabled = false;
pub const status_json = if (linked)
    "{\"owner\":\"zig\",\"api\":\"mlx-c\",\"linked\":true,\"enabled\":false}"
else
    "{\"owner\":\"zig\",\"api\":\"mlx-c\",\"linked\":false,\"enabled\":false}";

pub const MlxString = extern struct {
    ctx: ?*anyopaque,
};

pub const MlxArray = extern struct {
    ctx: ?*anyopaque,
};

pub const MlxMapStringToArray = extern struct {
    ctx: ?*anyopaque,
};

pub const MlxMapStringToString = extern struct {
    ctx: ?*anyopaque,
};

pub const MlxMapStringToArrayIterator = extern struct {
    ctx: ?*anyopaque,
    map_ctx: ?*anyopaque,
};

pub const MlxMapStringToStringIterator = extern struct {
    ctx: ?*anyopaque,
    map_ctx: ?*anyopaque,
};

pub const MlxStream = extern struct {
    ctx: ?*anyopaque,
};

pub const MlxDtype = enum(c_int) {
    bool = 0,
    uint8 = 1,
    uint16 = 2,
    uint32 = 3,
    uint64 = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    int64 = 8,
    float16 = 9,
    float32 = 10,
    float64 = 11,
    bfloat16 = 12,
    complex64 = 13,
    _,
};

pub fn dtypeSize(dtype: MlxDtype) usize {
    return switch (dtype) {
        .bool, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
        else => 0,
    };
}
