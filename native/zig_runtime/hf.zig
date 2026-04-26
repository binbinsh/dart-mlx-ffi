const std = @import("std");
const builtin = @import("builtin");

const path = std.Io.Dir.path;

pub fn defaultCacheRoot(allocator: std.mem.Allocator) ![]u8 {
    if (try getenvOwned(allocator, "DART_INFERENCE_HF_CACHE")) |value| {
        return value;
    }
    if (builtin.os.tag == .windows) {
        if (try getenvOwned(allocator, "LOCALAPPDATA")) |base| {
            defer allocator.free(base);
            return path.join(allocator, &.{ base, "dart_inference", "huggingface" }) catch
                return error.OutOfMemory;
        }
        if (try getenvOwned(allocator, "APPDATA")) |base| {
            defer allocator.free(base);
            return path.join(allocator, &.{ base, "dart_inference", "huggingface" }) catch
                return error.OutOfMemory;
        }
    }
    if (try getenvOwned(allocator, "HOME")) |home| {
        defer allocator.free(home);
        return path.join(allocator, &.{ home, ".cache", "dart_inference", "huggingface" }) catch
            return error.OutOfMemory;
    }
    const temp_root = try tempRoot(allocator);
    defer allocator.free(temp_root);
    return path.join(allocator, &.{ temp_root, "dart_inference", "huggingface" }) catch
        return error.OutOfMemory;
}

pub fn authToken(allocator: std.mem.Allocator) !?[]u8 {
    if (try getenvOwned(allocator, "HF_TOKEN")) |value| {
        return value;
    }
    return getenvOwned(allocator, "HUGGINGFACE_HUB_TOKEN");
}

pub fn refJson(
    allocator: std.mem.Allocator,
    source_uri: ?[]const u8,
    artifact_path: ?[]const u8,
    repo_meta: ?[]const u8,
    path_meta: ?[]const u8,
    revision_meta: ?[]const u8,
) ![]u8 {
    const revision = revision_meta orelse "main";
    if (repo_meta) |repo| {
        if (repo.len > 0) {
            const artifact = normalizeArtifactPath(path_meta);
            return refObjectJson(allocator, repo, artifact, revision);
        }
    }

    const raw = source_uri orelse artifact_path orelse return allocator.dupe(u8, "{}") catch
        return error.OutOfMemory;
    const parsed = parseUri(allocator, raw) orelse return allocator.dupe(u8, "{}") catch
        return error.OutOfMemory;
    defer parsed.deinit(allocator);
    return refObjectJson(allocator, parsed.repo, parsed.artifact_path, revision);
}

pub fn cachePath(
    allocator: std.mem.Allocator,
    cache_root: []const u8,
    repo: []const u8,
    revision: []const u8,
    artifact_path: []const u8,
) ![]u8 {
    const repo_dir = try repoDir(allocator, repo);
    defer allocator.free(repo_dir);
    const safe_revision = try safeSegment(allocator, revision);
    defer allocator.free(safe_revision);

    var current = path.join(allocator, &.{
        cache_root,
        repo_dir,
        "snapshots",
        safe_revision,
    }) catch return error.OutOfMemory;
    errdefer allocator.free(current);

    if (artifact_path.len == 0 or std.mem.eql(u8, artifact_path, ".")) {
        return current;
    }
    var parts = std.mem.tokenizeScalar(u8, artifact_path, '/');
    while (parts.next()) |part| {
        const next = path.join(allocator, &.{ current, part }) catch return error.OutOfMemory;
        allocator.free(current);
        current = next;
    }
    return current;
}

pub fn isDirectoryArtifact(artifact_path: []const u8) bool {
    if (artifact_path.len == 0 or
        std.mem.eql(u8, artifact_path, ".") or
        std.mem.endsWith(u8, artifact_path, "/"))
    {
        return true;
    }
    if (endsWithIgnoreCase(artifact_path, ".mlmodelc") or
        endsWithIgnoreCase(artifact_path, ".mlpackage"))
    {
        return true;
    }
    return std.mem.indexOfScalar(u8, basename(artifact_path), '.') == null;
}

const ParsedRef = struct {
    repo: []u8,
    artifact_path: []u8,

    fn deinit(self: ParsedRef, allocator: std.mem.Allocator) void {
        allocator.free(self.repo);
        allocator.free(self.artifact_path);
    }
};

fn parseUri(allocator: std.mem.Allocator, raw: []const u8) ?ParsedRef {
    const prefix = "hf://";
    if (!std.mem.startsWith(u8, raw, prefix)) {
        return null;
    }
    const rest = raw[prefix.len..];
    if (rest.len == 0) {
        return null;
    }
    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
    const host = rest[0..slash];
    if (host.len == 0) {
        return null;
    }
    var segments: std.ArrayList([]const u8) = .empty;
    defer segments.deinit(allocator);
    if (slash < rest.len) {
        var parts = std.mem.tokenizeScalar(u8, rest[slash + 1 ..], '/');
        while (parts.next()) |part| {
            segments.append(allocator, part) catch return null;
        }
    }

    if (segments.items.len == 0) {
        const repo = allocator.dupe(u8, host) catch return null;
        const artifact = allocator.dupe(u8, ".") catch {
            allocator.free(repo);
            return null;
        };
        return .{ .repo = repo, .artifact_path = artifact };
    }

    const repo = std.fmt.allocPrint(allocator, "{s}/{s}", .{
        host,
        segments.items[0],
    }) catch return null;
    errdefer allocator.free(repo);
    const artifact = if (segments.items.len == 1)
        allocator.dupe(u8, ".") catch return null
    else
        joinSlash(allocator, segments.items[1..]) catch return null;
    return .{ .repo = repo, .artifact_path = artifact };
}

fn normalizeArtifactPath(value: ?[]const u8) []const u8 {
    const artifact = value orelse return ".";
    return if (artifact.len == 0) "." else artifact;
}

fn refObjectJson(
    allocator: std.mem.Allocator,
    repo: []const u8,
    artifact_path: []const u8,
    revision: []const u8,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"repo\":");
    try jsonString(allocator, &out, repo);
    try out.appendSlice(allocator, ",\"path\":");
    try jsonString(allocator, &out, artifact_path);
    try out.appendSlice(allocator, ",\"revision\":");
    try jsonString(allocator, &out, revision);
    try out.appendSlice(allocator, ",\"sourceUri\":");
    try out.appendSlice(allocator, "\"hf://");
    try jsonStringBody(allocator, &out, repo);
    try out.append(allocator, '/');
    try jsonStringBody(allocator, &out, artifact_path);
    try out.appendSlice(allocator, "\"}");
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn repoDir(allocator: std.mem.Allocator, repo: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "models--");
    for (repo) |byte| {
        if (byte == '/') {
            try out.appendSlice(allocator, "--");
        } else {
            try out.append(allocator, byte);
        }
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn safeSegment(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (value) |byte| {
        const safe = std.ascii.isAlphanumeric(byte) or
            byte == '.' or byte == '_' or byte == '-';
        try out.append(allocator, if (safe) byte else '_');
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn joinSlash(allocator: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (parts, 0..) |part, index| {
        if (index > 0) {
            try out.append(allocator, '/');
        }
        try out.appendSlice(allocator, part);
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn basename(value: []const u8) []const u8 {
    const slash = std.mem.lastIndexOfScalar(u8, value, '/') orelse
        std.mem.lastIndexOfScalar(u8, value, '\\') orelse return value;
    return value[slash + 1 ..];
}

fn endsWithIgnoreCase(value: []const u8, suffix: []const u8) bool {
    if (value.len < suffix.len) {
        return false;
    }
    return std.ascii.eqlIgnoreCase(value[value.len - suffix.len ..], suffix);
}

fn getenvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const key = allocator.dupeZ(u8, name) catch return error.OutOfMemory;
    defer allocator.free(key);
    const raw = std.c.getenv(key.ptr) orelse return null;
    const value = std.mem.span(raw);
    if (value.len == 0) {
        return null;
    }
    return allocator.dupe(u8, value) catch return error.OutOfMemory;
}

fn tempRoot(allocator: std.mem.Allocator) ![]u8 {
    if (builtin.os.tag == .windows) {
        if (try getenvOwned(allocator, "TEMP")) |value| {
            return value;
        }
        if (try getenvOwned(allocator, "TMP")) |value| {
            return value;
        }
        return allocator.dupe(u8, "C:\\Temp") catch return error.OutOfMemory;
    }
    if (try getenvOwned(allocator, "TMPDIR")) |value| {
        return value;
    }
    return allocator.dupe(u8, "/tmp") catch return error.OutOfMemory;
}

fn jsonString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: []const u8) !void {
    try out.append(allocator, '"');
    try jsonStringBody(allocator, out, value);
    try out.append(allocator, '"');
}

fn jsonStringBody(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: []const u8) !void {
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

test "HF default cache root is Zig-owned" {
    const value = try defaultCacheRoot(std.testing.allocator);
    defer std.testing.allocator.free(value);
    try std.testing.expect(value.len > 0);
}

test "HF auth token lookup is Zig-owned" {
    const value = try authToken(std.testing.allocator);
    if (value) |token| {
        defer std.testing.allocator.free(token);
        try std.testing.expect(token.len > 0);
    }
}

test "HF ref JSON prefers metadata" {
    const json = try refJson(
        std.testing.allocator,
        "hf://ignored/repo/file.onnx",
        "ignored",
        "acme/demo",
        "onnx/model.onnx",
        "refs/pr/1",
    );
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"repo\":\"acme/demo\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"path\":\"onnx/model.onnx\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"revision\":\"refs/pr/1\"") != null);
}

test "HF ref JSON parses hf URI" {
    const json = try refJson(
        std.testing.allocator,
        null,
        "hf://acme/demo/onnx/model.onnx",
        null,
        null,
        null,
    );
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"repo\":\"acme/demo\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"path\":\"onnx/model.onnx\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"revision\":\"main\"") != null);
}

test "HF cache path uses safe revision and repo directory" {
    const value = try cachePath(
        std.testing.allocator,
        "/cache",
        "acme/demo",
        "refs/pr/1",
        "onnx/model.onnx",
    );
    defer std.testing.allocator.free(value);
    try std.testing.expect(std.mem.indexOf(u8, value, "models--acme--demo") != null);
    try std.testing.expect(std.mem.indexOf(u8, value, "refs_pr_1") != null);
    try std.testing.expect(std.mem.endsWith(u8, value, "model.onnx"));
}

test "HF directory artifact policy matches Dart behavior" {
    try std.testing.expect(isDirectoryArtifact("."));
    try std.testing.expect(isDirectoryArtifact("bundle.mlmodelc"));
    try std.testing.expect(isDirectoryArtifact("bundle.mlpackage"));
    try std.testing.expect(isDirectoryArtifact("directory"));
    try std.testing.expect(!isDirectoryArtifact("onnx/model.onnx"));
}
