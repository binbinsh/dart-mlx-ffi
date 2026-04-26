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

const native = if (linked and !builtin.is_test) struct {
    extern fn mlx_string_new() MlxString;
    extern fn mlx_string_data(str: MlxString) [*c]const u8;
    extern fn mlx_string_free(str: MlxString) c_int;
    extern fn mlx_version(str: *MlxString) c_int;
} else struct {};

pub const VersionError = error{
    MlxUnavailable,
    MlxCallFailed,
    OutOfMemory,
};

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
