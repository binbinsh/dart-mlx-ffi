const std = @import("std");
const policy = @import("policy.zig");

pub fn selectJson(allocator: std.mem.Allocator, request_json: []const u8) ![]u8 {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        request_json,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return fail(allocator, "Invalid runtime resolver request JSON.");
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |value| value,
        else => return fail(allocator, "Runtime resolver request must be a JSON object."),
    };
    const model_id = stringField(object, "modelId") orelse "model";
    const platform = intField(object, "platform") orelse policy.platformId();
    const requested = intField(object, "requestedEngine") orelse -1;
    const allow_fallback = boolField(object, "allowFallback") orelse true;
    const artifacts = arrayField(object, "artifacts") orelse
        return fail(allocator, "Runtime resolver request has no artifacts array.");
    const prefer = arrayField(object, "prefer");

    if (requested >= 0) {
        if (findArtifact(artifacts, requested, platform, true)) {
            return ok(allocator, requested, prefer, null);
        }
        if (!allow_fallback) {
            return failRequest(allocator, model_id, requested, platform);
        }
    }

    const fallback = if (requested >= 0) requested else null;
    for (policy.engineOrder(platform)) |engine| {
        if (!findArtifact(artifacts, engine, platform, false)) {
            continue;
        }
        return ok(allocator, engine, prefer, fallback);
    }
    return failNoArtifact(allocator, model_id, platform);
}

pub fn fallbackJson(allocator: std.mem.Allocator, request_json: []const u8) ![]u8 {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        request_json,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return fail(allocator, "Invalid runtime fallback request JSON.");
    defer parsed.deinit();

    const object = switch (parsed.value) {
        .object => |value| value,
        else => return fail(allocator, "Runtime fallback request must be a JSON object."),
    };
    const platform = intField(object, "platform") orelse policy.platformId();
    const artifacts = arrayField(object, "artifacts") orelse
        return fail(allocator, "Runtime fallback request has no artifacts array.");
    const registered = arrayField(object, "registeredEngines") orelse
        return fail(allocator, "Runtime fallback request has no registered engines array.");

    for (artifacts) |item| {
        const artifact = switch (item) {
            .object => |value| value,
            else => continue,
        };
        const engine = intField(artifact, "engine") orelse continue;
        if (!engineRegistered(registered, engine) or !targetsPlatform(artifact, platform)) {
            continue;
        }
        return fallbackOk(allocator, engine);
    }
    return allocator.dupe(u8, "{\"ok\":false}") catch error.OutOfMemory;
}

fn findArtifact(
    artifacts: []std.json.Value,
    engine: i32,
    platform: i32,
    allow_preview_mlx: bool,
) bool {
    for (artifacts) |item| {
        const object = switch (item) {
            .object => |value| value,
            else => continue,
        };
        if ((intField(object, "engine") orelse -1) != engine) {
            continue;
        }
        if (!targetsPlatform(object, platform)) {
            continue;
        }
        if (engine != @intFromEnum(policy.Engine.mlx)) {
            return true;
        }
        if (allow_preview_mlx or policy.mlxArtifactRegistered(
            stringField(object, "format"),
            stringField(object, "path"),
        )) {
            return true;
        }
    }
    return false;
}

fn engineRegistered(engines: []std.json.Value, engine: i32) bool {
    for (engines) |item| {
        const value = switch (item) {
            .integer => |integer| integer,
            else => continue,
        };
        if (value == engine) {
            return true;
        }
    }
    return false;
}

fn targetsPlatform(object: std.json.ObjectMap, platform: i32) bool {
    const value = object.get("targetPlatforms") orelse return true;
    return switch (value) {
        .array => |array| {
            if (array.items.len == 0) {
                return true;
            }
            const name = policy.platformName(platform);
            for (array.items) |item| {
                const target = switch (item) {
                    .string => |text| text,
                    else => continue,
                };
                if (std.mem.eql(u8, target, name)) {
                    return true;
                }
            }
            return false;
        },
        .string => |targets| policy.artifactMatches(
            @intFromEnum(policy.Engine.onnx),
            platform,
            targets,
            null,
            null,
            false,
        ),
        else => true,
    };
}

fn fallbackOk(allocator: std.mem.Allocator, engine: i32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"ok\":true,\"engine\":");
    try appendInt(allocator, &out, engine);
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn ok(
    allocator: std.mem.Allocator,
    engine: i32,
    prefer: ?[]std.json.Value,
    fallback_engine: ?i32,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"ok\":true,\"engine\":");
    try appendInt(allocator, &out, engine);
    try out.appendSlice(allocator, ",\"accelerators\":");
    try appendAccelerators(allocator, &out, engine, prefer);
    if (fallback_engine) |value| {
        try out.appendSlice(allocator, ",\"fallbackReason\":\"Requested ");
        try jsonBody(allocator, &out, policy.engineName(value));
        try out.appendSlice(allocator, " has no compatible artifact.\"");
    }
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn failRequest(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    engine: i32,
    platform: i32,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "Model ");
    try out.appendSlice(allocator, model_id);
    try out.appendSlice(allocator, " has no ");
    try out.appendSlice(allocator, policy.engineName(engine));
    try out.appendSlice(allocator, " artifact for ");
    try out.appendSlice(allocator, policy.platformName(platform));
    try out.append(allocator, '.');
    const message = try out.toOwnedSlice(allocator);
    defer allocator.free(message);
    return fail(allocator, message);
}

fn failNoArtifact(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    platform: i32,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "Model ");
    try out.appendSlice(allocator, model_id);
    try out.appendSlice(allocator, " has no runtime artifact for ");
    try out.appendSlice(allocator, policy.platformName(platform));
    try out.append(allocator, '.');
    const message = try out.toOwnedSlice(allocator);
    defer allocator.free(message);
    return fail(allocator, message);
}

fn fail(allocator: std.mem.Allocator, message: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"ok\":false,\"message\":");
    try jsonString(allocator, &out, message);
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn appendAccelerators(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    engine: i32,
    prefer: ?[]std.json.Value,
) !void {
    if (prefer) |items| {
        if (items.len > 0) {
            try appendStringArray(allocator, out, items);
            return;
        }
    }
    try out.appendSlice(allocator, policy.acceleratorsJson(engine));
}

fn appendStringArray(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    items: []std.json.Value,
) !void {
    try out.append(allocator, '[');
    var written = false;
    for (items) |item| {
        const text = switch (item) {
            .string => |value| value,
            else => continue,
        };
        if (written) {
            try out.append(allocator, ',');
        }
        try jsonString(allocator, out, text);
        written = true;
    }
    try out.append(allocator, ']');
}

fn appendInt(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: i32) !void {
    var buffer: [16]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch return error.OutOfMemory;
    try out.appendSlice(allocator, text);
}

fn intField(object: std.json.ObjectMap, key: []const u8) ?i32 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |integer| if (integer >= std.math.minInt(i32) and integer <= std.math.maxInt(i32))
            @intCast(integer)
        else
            null,
        else => null,
    };
}

fn boolField(object: std.json.ObjectMap, key: []const u8) ?bool {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .bool => |boolean| boolean,
        else => null,
    };
}

fn stringField(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |text| if (text.len == 0) null else text,
        else => null,
    };
}

fn arrayField(object: std.json.ObjectMap, key: []const u8) ?[]std.json.Value {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .array => |array| array.items,
        else => null,
    };
}

fn jsonString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: []const u8) !void {
    try out.append(allocator, '"');
    try jsonBody(allocator, out, value);
    try out.append(allocator, '"');
}

fn jsonBody(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: []const u8) !void {
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, byte),
        }
    }
}

test "runtime resolver prefers Core ML on Apple platforms" {
    const request =
        \\{"modelId":"demo","platform":1,"requestedEngine":-1,"allowFallback":true,"prefer":[],"artifacts":[
        \\{"engine":0,"path":"model.safetensors","format":"mlx-safetensors","targetPlatforms":["macos"]},
        \\{"engine":1,"path":"coreml","targetPlatforms":["ios","macos"]},
        \\{"engine":2,"path":"model.onnx","targetPlatforms":["linux"]}
        \\]}
    ;
    const result = try selectJson(std.testing.allocator, request);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"ok\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"engine\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"accelerators\":[\"ane\",\"gpu\",\"cpu\"]") != null);
}

test "runtime resolver selects registered MLX artifacts" {
    const request =
        \\{"modelId":"mlxfn","platform":1,"requestedEngine":-1,"allowFallback":true,"prefer":[],"artifacts":[
        \\{"engine":0,"path":"function.mlxfn","format":"mlx-function","targetPlatforms":["macos"]},
        \\{"engine":2,"path":"model.onnx","targetPlatforms":["macos"]}
        \\]}
    ;
    const result = try selectJson(std.testing.allocator, request);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"ok\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"engine\":0") != null);
}

test "runtime resolver falls back from unavailable requested engine" {
    const request =
        \\{"modelId":"demo","platform":4,"requestedEngine":1,"allowFallback":true,"prefer":["npu"],"artifacts":[
        \\{"engine":1,"path":"coreml","targetPlatforms":["macos"]},
        \\{"engine":3,"path":"model.tflite","targetPlatforms":["android"]}
        \\]}
    ;
    const result = try selectJson(std.testing.allocator, request);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"ok\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"engine\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"accelerators\":[\"npu\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Requested coreml has no compatible artifact") != null);
}

test "runtime resolver reports requested engine failures" {
    const request =
        \\{"modelId":"demo","platform":4,"requestedEngine":1,"allowFallback":false,"prefer":[],"artifacts":[
        \\{"engine":1,"path":"coreml","targetPlatforms":["macos"]}
        \\]}
    ;
    const result = try selectJson(std.testing.allocator, request);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"ok\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Model demo has no coreml artifact for android.") != null);
}

test "runtime fallback preserves artifact order for registered engines" {
    const request =
        \\{"platform":1,"registeredEngines":[0,2],"artifacts":[
        \\{"engine":1,"path":"coreml","targetPlatforms":["macos"]},
        \\{"engine":0,"path":"model.safetensors","format":"mlx-safetensors","targetPlatforms":["macos"]},
        \\{"engine":2,"path":"model.onnx","targetPlatforms":["macos"]}
        \\]}
    ;
    const result = try fallbackJson(std.testing.allocator, request);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"ok\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"engine\":0") != null);
}

test "runtime fallback skips incompatible registered artifacts" {
    const request =
        \\{"platform":1,"registeredEngines":[2],"artifacts":[
        \\{"engine":2,"path":"model.onnx","targetPlatforms":["linux"]}
        \\]}
    ;
    const result = try fallbackJson(std.testing.allocator, request);
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("{\"ok\":false}", result);
}
