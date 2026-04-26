const std = @import("std");
const policy = @import("policy.zig");

pub const Artifact = extern struct {
    engine: i32,
    path: [*c]const u8,
    format: [*c]const u8,
    target_platforms: [*c]const u8,
};

pub const Result = extern struct {
    engine: i32,
    accel_mask: i32,
    fallback_engine: i32,
};

pub const SelectError = error{
    RequestedUnavailable,
    NoArtifact,
};

pub fn select(
    platform: i32,
    requested: i32,
    allow_fallback: bool,
    prefer_mask: i32,
    artifacts: []const Artifact,
) SelectError!Result {
    if (requested >= 0) {
        if (findArtifactAbi(artifacts, requested, platform, true)) {
            return selected(requested, prefer_mask, null);
        }
        if (!allow_fallback) {
            return error.RequestedUnavailable;
        }
    }

    const fallback_engine = if (requested >= 0) requested else null;
    for (policy.engineOrder(platform)) |engine| {
        if (!findArtifactAbi(artifacts, engine, platform, false)) {
            continue;
        }
        return selected(engine, prefer_mask, fallback_engine);
    }
    return error.NoArtifact;
}

pub fn fallback(
    platform: i32,
    registered: []const i32,
    artifacts: []const Artifact,
) ?i32 {
    for (artifacts) |artifact| {
        if (!engineRegisteredAbi(registered, artifact.engine)) {
            continue;
        }
        if (!policy.artifactMatches(
            artifact.engine,
            platform,
            optionalText(artifact.target_platforms),
            optionalText(artifact.format),
            optionalText(artifact.path),
            true,
        )) {
            continue;
        }
        return artifact.engine;
    }
    return null;
}

fn selected(engine: i32, prefer_mask: i32, fallback_engine: ?i32) Result {
    return .{
        .engine = engine,
        .accel_mask = if (prefer_mask != 0) prefer_mask else policy.acceleratorMask(engine),
        .fallback_engine = fallback_engine orelse -1,
    };
}

fn findArtifactAbi(
    artifacts: []const Artifact,
    engine: i32,
    platform: i32,
    allow_preview_mlx: bool,
) bool {
    for (artifacts) |artifact| {
        if (artifact.engine != engine) {
            continue;
        }
        if (policy.artifactMatches(
            artifact.engine,
            platform,
            optionalText(artifact.target_platforms),
            optionalText(artifact.format),
            optionalText(artifact.path),
            allow_preview_mlx,
        )) {
            return true;
        }
    }
    return false;
}

fn engineRegisteredAbi(engines: []const i32, engine: i32) bool {
    for (engines) |item| {
        if (item == engine) {
            return true;
        }
    }
    return false;
}

fn optionalText(value: [*c]const u8) ?[]const u8 {
    if (value == null) {
        return null;
    }
    const len = std.mem.len(value);
    if (len == 0) {
        return null;
    }
    return value[0..len];
}

test "runtime resolver prefers Core ML on Apple platforms" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.mlx),
            .path = "model.safetensors",
            .format = "mlx-safetensors",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.coreml),
            .path = "coreml",
            .format = "",
            .target_platforms = "ios\nmacos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.onnx),
            .path = "model.onnx",
            .format = "",
            .target_platforms = "linux",
        },
    };
    const result = try select(@intFromEnum(policy.Platform.macos), -1, true, 0, artifacts[0..]);
    try std.testing.expectEqual(@as(i32, @intFromEnum(policy.Engine.coreml)), result.engine);
    try std.testing.expectEqual(
        @as(i32, policy.accel_ane | policy.accel_gpu | policy.accel_cpu),
        result.accel_mask,
    );
}

test "typed runtime resolver avoids Dart JSON assembly" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.mlx),
            .path = "model.safetensors",
            .format = "mlx-safetensors",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.coreml),
            .path = "coreml",
            .format = "",
            .target_platforms = "ios\nmacos",
        },
    };
    const result = try select(
        @intFromEnum(policy.Platform.macos),
        -1,
        true,
        policy.accel_ane,
        artifacts[0..],
    );
    try std.testing.expectEqual(@as(i32, @intFromEnum(policy.Engine.coreml)), result.engine);
    try std.testing.expectEqual(@as(i32, policy.accel_ane), result.accel_mask);
    try std.testing.expectEqual(@as(i32, -1), result.fallback_engine);
}

test "typed runtime fallback preserves registered artifact order" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.coreml),
            .path = "coreml",
            .format = "",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.mlx),
            .path = "model.safetensors",
            .format = "mlx-safetensors",
            .target_platforms = "macos",
        },
    };
    const registered = [_]i32{@intFromEnum(policy.Engine.mlx)};
    try std.testing.expectEqual(
        @as(?i32, @intFromEnum(policy.Engine.mlx)),
        fallback(@intFromEnum(policy.Platform.macos), registered[0..], artifacts[0..]),
    );
}

test "runtime resolver selects registered MLX artifacts" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.mlx),
            .path = "function.mlxfn",
            .format = "mlx-function",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.onnx),
            .path = "model.onnx",
            .format = "",
            .target_platforms = "macos",
        },
    };
    const result = try select(@intFromEnum(policy.Platform.macos), -1, true, 0, artifacts[0..]);
    try std.testing.expectEqual(@as(i32, @intFromEnum(policy.Engine.mlx)), result.engine);
}

test "runtime resolver falls back from unavailable requested engine" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.coreml),
            .path = "coreml",
            .format = "",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.litert),
            .path = "model.tflite",
            .format = "",
            .target_platforms = "android",
        },
    };
    const result = try select(
        @intFromEnum(policy.Platform.android),
        @intFromEnum(policy.Engine.coreml),
        true,
        policy.accel_npu,
        artifacts[0..],
    );
    try std.testing.expectEqual(@as(i32, @intFromEnum(policy.Engine.litert)), result.engine);
    try std.testing.expectEqual(@as(i32, policy.accel_npu), result.accel_mask);
    try std.testing.expectEqual(@as(i32, @intFromEnum(policy.Engine.coreml)), result.fallback_engine);
}

test "runtime resolver reports requested engine failures" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.coreml),
            .path = "coreml",
            .format = "",
            .target_platforms = "macos",
        },
    };
    try std.testing.expectError(
        error.RequestedUnavailable,
        select(
            @intFromEnum(policy.Platform.android),
            @intFromEnum(policy.Engine.coreml),
            false,
            0,
            artifacts[0..],
        ),
    );
}

test "runtime fallback preserves artifact order for registered engines" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.coreml),
            .path = "coreml",
            .format = "",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.mlx),
            .path = "model.safetensors",
            .format = "mlx-safetensors",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(policy.Engine.onnx),
            .path = "model.onnx",
            .format = "",
            .target_platforms = "macos",
        },
    };
    const registered = [_]i32{
        @intFromEnum(policy.Engine.mlx),
        @intFromEnum(policy.Engine.onnx),
    };
    try std.testing.expectEqual(
        @as(?i32, @intFromEnum(policy.Engine.mlx)),
        fallback(@intFromEnum(policy.Platform.macos), registered[0..], artifacts[0..]),
    );
}

test "runtime fallback skips incompatible registered artifacts" {
    const artifacts = [_]Artifact{
        .{
            .engine = @intFromEnum(policy.Engine.onnx),
            .path = "model.onnx",
            .format = "",
            .target_platforms = "linux",
        },
    };
    const registered = [_]i32{@intFromEnum(policy.Engine.onnx)};
    try std.testing.expectEqual(
        @as(?i32, null),
        fallback(@intFromEnum(policy.Platform.macos), registered[0..], artifacts[0..]),
    );
}
