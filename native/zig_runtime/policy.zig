const std = @import("std");
const builtin = @import("builtin");

pub const Engine = enum(i32) {
    mlx = 0,
    coreml = 1,
    onnx = 2,
    litert = 3,
};

pub const Platform = enum(i32) {
    ios = 0,
    macos = 1,
    windows = 2,
    linux = 3,
    android = 4,
    unknown = 5,
};

pub fn engineName(engine: i32) []const u8 {
    return switch (engine) {
        @intFromEnum(Engine.mlx) => "mlx",
        @intFromEnum(Engine.coreml) => "coreml",
        @intFromEnum(Engine.onnx) => "onnx",
        @intFromEnum(Engine.litert) => "litert",
        else => "unknown",
    };
}

pub fn platformId() i32 {
    return switch (builtin.os.tag) {
        .ios => @intFromEnum(Platform.ios),
        .macos => @intFromEnum(Platform.macos),
        .windows => @intFromEnum(Platform.windows),
        .linux => if (builtin.abi == .android)
            @intFromEnum(Platform.android)
        else
            @intFromEnum(Platform.linux),
        else => @intFromEnum(Platform.unknown),
    };
}

pub fn platformName(id: i32) []const u8 {
    return switch (id) {
        @intFromEnum(Platform.ios) => "ios",
        @intFromEnum(Platform.macos) => "macos",
        @intFromEnum(Platform.windows) => "windows",
        @intFromEnum(Platform.linux) => "linux",
        @intFromEnum(Platform.android) => "android",
        else => "unknown",
    };
}

pub fn acceleratorsJson(engine: i32) []const u8 {
    return switch (engine) {
        @intFromEnum(Engine.mlx) => "[\"gpu\",\"cpu\"]",
        @intFromEnum(Engine.coreml) => "[\"ane\",\"gpu\",\"cpu\"]",
        @intFromEnum(Engine.onnx) => "[\"gpu\",\"cpu\"]",
        @intFromEnum(Engine.litert) => "[\"gpu\",\"npu\",\"cpu\"]",
        else => "[]",
    };
}

pub fn engineOrderJson(platform: i32) []const u8 {
    return switch (platform) {
        @intFromEnum(Platform.ios),
        @intFromEnum(Platform.macos),
        => "[\"coreml\",\"mlx\",\"onnx\"]",
        @intFromEnum(Platform.windows),
        @intFromEnum(Platform.linux),
        => "[\"onnx\"]",
        @intFromEnum(Platform.android) => "[\"litert\",\"onnx\"]",
        else => "[\"coreml\",\"onnx\",\"litert\"]",
    };
}

pub fn mlxArtifactRegistered(format: ?[]const u8, artifact_path: ?[]const u8) bool {
    if (format) |value| {
        if (std.mem.eql(u8, value, "mlx-function") or std.mem.eql(u8, value, "mlxfn")) {
            return true;
        }
    }
    const value = artifact_path orelse return false;
    return std.mem.endsWith(u8, value, ".mlxfn") or
        std.mem.endsWith(u8, value, "/function.mlxfn");
}

test "runtime policy names engines and platforms" {
    try std.testing.expectEqualStrings("mlx", engineName(@intFromEnum(Engine.mlx)));
    try std.testing.expectEqualStrings("unknown", engineName(99));
    try std.testing.expectEqualStrings("linux", platformName(@intFromEnum(Platform.linux)));
    try std.testing.expectEqualStrings("unknown", platformName(99));
}

test "runtime policy orders engines by platform" {
    try std.testing.expectEqualStrings("[\"coreml\",\"mlx\",\"onnx\"]", engineOrderJson(@intFromEnum(Platform.macos)));
    try std.testing.expectEqualStrings("[\"onnx\"]", engineOrderJson(@intFromEnum(Platform.linux)));
    try std.testing.expectEqualStrings("[\"litert\",\"onnx\"]", engineOrderJson(@intFromEnum(Platform.android)));
}

test "runtime policy identifies registered MLX function artifacts" {
    try std.testing.expect(mlxArtifactRegistered("mlx-function", "model.safetensors"));
    try std.testing.expect(mlxArtifactRegistered("mlxfn", "model.safetensors"));
    try std.testing.expect(mlxArtifactRegistered(null, "function.mlxfn"));
    try std.testing.expect(mlxArtifactRegistered(null, "bundle/function.mlxfn"));
    try std.testing.expect(!mlxArtifactRegistered("mlx-safetensors", "model.safetensors"));
}
