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

pub const accel_cpu: i32 = 1 << 0;
pub const accel_gpu: i32 = 1 << 1;
pub const accel_ane: i32 = 1 << 2;
pub const accel_npu: i32 = 1 << 3;

pub fn acceleratorMask(engine: i32) i32 {
    return switch (engine) {
        @intFromEnum(Engine.mlx),
        @intFromEnum(Engine.onnx),
        => accel_gpu | accel_cpu,
        @intFromEnum(Engine.coreml) => accel_ane | accel_gpu | accel_cpu,
        @intFromEnum(Engine.litert) => accel_gpu | accel_npu | accel_cpu,
        else => 0,
    };
}

const apple_order = [_]i32{
    @intFromEnum(Engine.coreml),
    @intFromEnum(Engine.mlx),
    @intFromEnum(Engine.onnx),
};
const desktop_order = [_]i32{@intFromEnum(Engine.onnx)};
const android_order = [_]i32{
    @intFromEnum(Engine.litert),
    @intFromEnum(Engine.onnx),
};
const fallback_order = [_]i32{
    @intFromEnum(Engine.coreml),
    @intFromEnum(Engine.onnx),
    @intFromEnum(Engine.litert),
};

pub fn engineOrder(platform: i32) []const i32 {
    return switch (platform) {
        @intFromEnum(Platform.ios),
        @intFromEnum(Platform.macos),
        => apple_order[0..],
        @intFromEnum(Platform.windows),
        @intFromEnum(Platform.linux),
        => desktop_order[0..],
        @intFromEnum(Platform.android) => android_order[0..],
        else => fallback_order[0..],
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

pub fn artifactMatches(
    engine: i32,
    platform: i32,
    target_platforms: ?[]const u8,
    format: ?[]const u8,
    artifact_path: ?[]const u8,
    allow_preview_mlx: bool,
) bool {
    if (!platformAllowed(platform, target_platforms)) {
        return false;
    }
    if (engine != @intFromEnum(Engine.mlx)) {
        return true;
    }
    return allow_preview_mlx or mlxArtifactRegistered(format, artifact_path);
}

pub fn artifactPath(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    artifact_path: []const u8,
) ![]u8 {
    if (std.mem.indexOf(u8, artifact_path, "://") != null or
        std.mem.startsWith(u8, artifact_path, "/") or
        root_path.len == 0)
    {
        return allocator.dupe(u8, artifact_path) catch return error.OutOfMemory;
    }
    if (std.mem.endsWith(u8, root_path, "/")) {
        return std.fmt.allocPrint(allocator, "{s}{s}", .{ root_path, artifact_path }) catch
            return error.OutOfMemory;
    }
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ root_path, artifact_path }) catch
        return error.OutOfMemory;
}

pub fn artifactRemote(artifact_path: ?[]const u8) bool {
    const value = artifact_path orelse return false;
    if (std.mem.startsWith(u8, value, "zig://")) {
        return false;
    }
    return std.mem.indexOf(u8, value, "://") != null;
}

fn platformAllowed(platform: i32, target_platforms: ?[]const u8) bool {
    const raw = target_platforms orelse return true;
    const trimmed = std.mem.trim(u8, raw, separators());
    if (trimmed.len == 0) {
        return true;
    }
    const name = platformName(platform);
    var parts = std.mem.tokenizeAny(u8, trimmed, separators());
    while (parts.next()) |part| {
        if (std.mem.eql(u8, part, name)) {
            return true;
        }
    }
    return false;
}

fn separators() []const u8 {
    return ":;,\n\r \t";
}

test "runtime policy names engines and platforms" {
    try std.testing.expectEqualStrings("mlx", engineName(@intFromEnum(Engine.mlx)));
    try std.testing.expectEqualStrings("unknown", engineName(99));
    try std.testing.expectEqualStrings("linux", platformName(@intFromEnum(Platform.linux)));
    try std.testing.expectEqualStrings("unknown", platformName(99));
}

test "runtime policy orders engines by platform" {
    try std.testing.expectEqualSlices(
        i32,
        &.{ @intFromEnum(Engine.coreml), @intFromEnum(Engine.mlx), @intFromEnum(Engine.onnx) },
        engineOrder(@intFromEnum(Platform.macos)),
    );
    try std.testing.expectEqualSlices(
        i32,
        &.{@intFromEnum(Engine.onnx)},
        engineOrder(@intFromEnum(Platform.linux)),
    );
    try std.testing.expectEqualSlices(
        i32,
        &.{ @intFromEnum(Engine.litert), @intFromEnum(Engine.onnx) },
        engineOrder(@intFromEnum(Platform.android)),
    );
}

test "runtime policy identifies registered MLX function artifacts" {
    try std.testing.expect(mlxArtifactRegistered("mlx-function", "model.safetensors"));
    try std.testing.expect(mlxArtifactRegistered("mlxfn", "model.safetensors"));
    try std.testing.expect(mlxArtifactRegistered(null, "function.mlxfn"));
    try std.testing.expect(mlxArtifactRegistered(null, "bundle/function.mlxfn"));
    try std.testing.expect(!mlxArtifactRegistered("mlx-safetensors", "model.safetensors"));
}

test "runtime policy matches artifacts by platform and MLX status" {
    try std.testing.expect(artifactMatches(
        @intFromEnum(Engine.onnx),
        @intFromEnum(Platform.linux),
        "linux\nwindows",
        null,
        "model.onnx",
        false,
    ));
    try std.testing.expect(!artifactMatches(
        @intFromEnum(Engine.onnx),
        @intFromEnum(Platform.android),
        "linux\nwindows",
        null,
        "model.onnx",
        false,
    ));
    try std.testing.expect(!artifactMatches(
        @intFromEnum(Engine.mlx),
        @intFromEnum(Platform.macos),
        "macos",
        "mlx-safetensors",
        "model.safetensors",
        false,
    ));
    try std.testing.expect(artifactMatches(
        @intFromEnum(Engine.mlx),
        @intFromEnum(Platform.macos),
        "macos",
        "mlx-safetensors",
        "model.safetensors",
        true,
    ));
}

test "runtime policy resolves artifact paths like Dart did" {
    const allocator = std.testing.allocator;
    const absolute = try artifactPath(allocator, "/models", "/tmp/model.onnx");
    defer allocator.free(absolute);
    try std.testing.expectEqualStrings("/tmp/model.onnx", absolute);

    const uri = try artifactPath(allocator, "/models", "hf://org/repo/model.onnx");
    defer allocator.free(uri);
    try std.testing.expectEqualStrings("hf://org/repo/model.onnx", uri);

    const joined = try artifactPath(allocator, "/models", "model.onnx");
    defer allocator.free(joined);
    try std.testing.expectEqualStrings("/models/model.onnx", joined);

    const trailing = try artifactPath(allocator, "/models/", "model.onnx");
    defer allocator.free(trailing);
    try std.testing.expectEqualStrings("/models/model.onnx", trailing);
}

test "runtime policy identifies remote artifacts" {
    try std.testing.expect(artifactRemote("hf://org/repo/model.onnx"));
    try std.testing.expect(artifactRemote("s3://bucket/model.onnx"));
    try std.testing.expect(!artifactRemote("zig://echo"));
    try std.testing.expect(!artifactRemote("/models/model.onnx"));
    try std.testing.expect(!artifactRemote("model.onnx"));
    try std.testing.expect(!artifactRemote(null));
}
