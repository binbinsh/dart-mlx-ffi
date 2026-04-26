const std = @import("std");
const builtin = @import("builtin");

const path = std.Io.Dir.path;
const max_env_json_bytes = 1024 * 1024;

const env_file_name = ".dart_inference_runtime_env.json";

const default_ort_libs = [_][]const u8{
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libnvinfer.so.10",
    "libnvinfer_plugin.so.10",
    "libnvonnxparser.so.10",
    "libnvinfer.so.9",
    "libnvinfer_plugin.so.9",
    "libnvonnxparser.so.9",
};

const Pair = struct {
    key: []u8,
    value: []u8,
};

const Env = struct {
    allocator: std.mem.Allocator,
    values: std.ArrayList(Pair) = .empty,

    fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        runtime_env_file: ?[]const u8,
        search_roots: []const u8,
    ) !Env {
        var env = Env{ .allocator = allocator };
        errdefer env.deinit();
        const file = try resolveEnvFile(allocator, io, runtime_env_file, search_roots);
        defer if (file) |value| allocator.free(value);
        if (file) |value| {
            try env.loadFile(io, value);
        }
        return env;
    }

    fn deinit(self: *Env) void {
        for (self.values.items) |item| {
            self.allocator.free(item.key);
            self.allocator.free(item.value);
        }
        self.values.deinit(self.allocator);
    }

    fn loadFile(self: *Env, io: std.Io, file_path: []const u8) !void {
        const bytes = std.Io.Dir.cwd().readFileAlloc(
            io,
            file_path,
            self.allocator,
            .limited(max_env_json_bytes),
        ) catch return;
        defer self.allocator.free(bytes);

        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            bytes,
            .{ .duplicate_field_behavior = .use_last },
        ) catch return;
        defer parsed.deinit();

        const object = switch (parsed.value) {
            .object => |value| value,
            else => return,
        };
        var iterator = object.iterator();
        while (iterator.next()) |entry| {
            const raw_value = switch (entry.value_ptr.*) {
                .string => |value| value,
                else => continue,
            };
            if (raw_value.len == 0) {
                continue;
            }
            const key = self.allocator.dupe(u8, entry.key_ptr.*) catch return error.OutOfMemory;
            errdefer self.allocator.free(key);
            const value = self.allocator.dupe(u8, raw_value) catch return error.OutOfMemory;
            errdefer self.allocator.free(value);
            self.values.append(self.allocator, .{ .key = key, .value = value }) catch return error.OutOfMemory;
        }
    }

    fn valueOwned(self: *const Env, name: []const u8) !?[]u8 {
        const process_value = try getenvOwned(self.allocator, name);
        if (process_value) |value| {
            if (value.len > 0) {
                return value;
            }
            self.allocator.free(value);
        }
        for (self.values.items) |item| {
            if (std.mem.eql(u8, item.key, name) and item.value.len > 0) {
                return self.allocator.dupe(u8, item.value) catch error.OutOfMemory;
            }
        }
        return null;
    }
};

const List = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList([]u8) = .empty,

    fn init(allocator: std.mem.Allocator) List {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *List) void {
        for (self.items.items) |item| {
            self.allocator.free(item);
        }
        self.items.deinit(self.allocator);
    }

    fn add(self: *List, value: []const u8) !void {
        const trimmed = std.mem.trim(u8, value, " \t");
        if (trimmed.len == 0 or self.contains(trimmed)) {
            return;
        }
        const copy = self.allocator.dupe(u8, trimmed) catch return error.OutOfMemory;
        errdefer self.allocator.free(copy);
        self.items.append(self.allocator, copy) catch return error.OutOfMemory;
    }

    fn addList(self: *List, raw: []const u8) !void {
        var parts = std.mem.tokenizeAny(u8, raw, separators());
        while (parts.next()) |part| {
            try self.add(part);
        }
    }

    fn contains(self: *const List, value: []const u8) bool {
        for (self.items.items) |item| {
            if (std.mem.eql(u8, item, value)) {
                return true;
            }
        }
        return false;
    }
};

pub fn ortLibsJson(
    allocator: std.mem.Allocator,
    io: std.Io,
    runtime_env_file: ?[]const u8,
    search_roots: []const u8,
    explicit_libraries: []const u8,
    library_dirs: []const u8,
    library_names: []const u8,
) ![]u8 {
    var env = try Env.load(allocator, io, runtime_env_file, search_roots);
    defer env.deinit();

    var dirs = List.init(allocator);
    defer dirs.deinit();
    try addExistingDirs(&dirs, io, library_dirs);
    try addEnvDirs(&dirs, io, &env);

    var names = List.init(allocator);
    defer names.deinit();
    try names.addList(library_names);
    if (names.items.items.len == 0) {
        for (default_ort_libs) |name| {
            try names.add(name);
        }
    }

    var libs = List.init(allocator);
    defer libs.deinit();
    try addExistingFiles(&libs, io, explicit_libraries);
    for (dirs.items.items) |dir| {
        for (names.items.items) |name| {
            const candidate = path.join(allocator, &.{ dir, name }) catch return error.OutOfMemory;
            defer allocator.free(candidate);
            if (isFile(io, candidate)) {
                try libs.add(candidate);
            }
        }
    }
    return jsonList(allocator, libs.items.items);
}

fn addEnvDirs(dirs: *List, io: std.Io, env: *const Env) !void {
    for ([_][]const u8{
        "DART_INFERENCE_NATIVE_LIBRARY_DIRS",
        "DART_INFERENCE_CUDA_LIBRARY_DIRS",
        "DART_INFERENCE_CUDA_LIB_DIRS",
        "DART_INFERENCE_TENSORRT_LIBRARY_DIRS",
    }) |name| {
        const value = try env.valueOwned(name);
        defer if (value) |item| env.allocator.free(item);
        if (value) |item| {
            try addExistingDirs(dirs, io, item);
        }
    }
    for ([_][]const u8{
        "DART_INFERENCE_CUDA_LIB_DIR",
        "DART_INFERENCE_TENSORRT_LIB_DIR",
    }) |name| {
        const value = try env.valueOwned(name);
        defer if (value) |item| env.allocator.free(item);
        if (value) |item| {
            try addExistingDir(dirs, io, item);
        }
    }
    for ([_][]const u8{
        "DART_INFERENCE_ORT_RUNTIME_LIBRARY",
        "DART_INFERENCE_ORT_LIBRARY",
    }) |name| {
        const value = try env.valueOwned(name);
        defer if (value) |item| env.allocator.free(item);
        if (value) |item| {
            try addOrtDirs(dirs, io, item);
        }
    }
}

fn addExistingDirs(dirs: *List, io: std.Io, raw: []const u8) !void {
    var parts = std.mem.tokenizeAny(u8, raw, separators());
    while (parts.next()) |part| {
        try addExistingDir(dirs, io, part);
    }
}

fn addExistingDir(dirs: *List, io: std.Io, value: []const u8) !void {
    const trimmed = std.mem.trim(u8, value, " \t");
    if (trimmed.len > 0 and isDir(io, trimmed)) {
        try dirs.add(trimmed);
    }
}

fn addExistingFiles(libs: *List, io: std.Io, raw: []const u8) !void {
    var parts = std.mem.tokenizeAny(u8, raw, separators());
    while (parts.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (trimmed.len > 0 and isFile(io, trimmed)) {
            try libs.add(trimmed);
        }
    }
}

fn addOrtDirs(dirs: *List, io: std.Io, library: []const u8) !void {
    const lib_dir = parentPath(dirs.allocator, library) catch return error.OutOfMemory;
    defer dirs.allocator.free(lib_dir);
    try addExistingDir(dirs, io, lib_dir);

    const ort_root = parentPath(dirs.allocator, lib_dir) catch return error.OutOfMemory;
    defer dirs.allocator.free(ort_root);
    const runtime_root = parentPath(dirs.allocator, ort_root) catch return error.OutOfMemory;
    defer dirs.allocator.free(runtime_root);

    const cuda = path.join(dirs.allocator, &.{ runtime_root, "cuda", "lib" }) catch return error.OutOfMemory;
    defer dirs.allocator.free(cuda);
    try addExistingDir(dirs, io, cuda);

    const tensorrt = path.join(dirs.allocator, &.{ runtime_root, "tensorrt", "lib" }) catch return error.OutOfMemory;
    defer dirs.allocator.free(tensorrt);
    try addExistingDir(dirs, io, tensorrt);
}

fn resolveEnvFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    runtime_env_file: ?[]const u8,
    search_roots: []const u8,
) !?[]u8 {
    if (runtime_env_file) |value| {
        if (value.len > 0 and isFile(io, value)) {
            return allocator.dupe(u8, value) catch error.OutOfMemory;
        }
    }
    const explicit = try getenvOwned(allocator, "DART_INFERENCE_RUNTIME_ENV_FILE");
    defer if (explicit) |value| allocator.free(value);
    if (explicit) |value| {
        if (value.len > 0 and isFile(io, value)) {
            return allocator.dupe(u8, value) catch error.OutOfMemory;
        }
    }
    if (isFile(io, env_file_name)) {
        return allocator.dupe(u8, env_file_name) catch error.OutOfMemory;
    }
    var roots = std.mem.tokenizeAny(u8, search_roots, separators());
    while (roots.next()) |root| {
        const candidate = path.join(allocator, &.{ std.mem.trim(u8, root, " \t"), env_file_name }) catch return error.OutOfMemory;
        defer allocator.free(candidate);
        if (isFile(io, candidate)) {
            return allocator.dupe(u8, candidate) catch error.OutOfMemory;
        }
    }
    return null;
}

fn isFile(io: std.Io, value: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(io, value, .{}) catch return false;
    return stat.kind == .file;
}

fn isDir(io: std.Io, value: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(io, value, .{}) catch return false;
    return stat.kind == .directory;
}

fn parentPath(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var index = value.len;
    while (index > 0) {
        index -= 1;
        if (value[index] == '/' or value[index] == '\\') {
            if (index == 0) {
                return allocator.dupe(u8, value[0..1]);
            }
            return allocator.dupe(u8, value[0..index]);
        }
    }
    return allocator.dupe(u8, ".");
}

fn getenvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const key = allocator.dupeZ(u8, name) catch return error.OutOfMemory;
    defer allocator.free(key);
    const raw = std.c.getenv(key.ptr) orelse return null;
    const value = std.mem.span(raw);
    if (value.len == 0) {
        return null;
    }
    return allocator.dupe(u8, value) catch error.OutOfMemory;
}

fn separators() []const u8 {
    return if (builtin.os.tag == .windows) ";,\n\r" else ":;,\n\r";
}

fn jsonList(allocator: std.mem.Allocator, values: []const []u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    out.append(allocator, '[') catch return error.OutOfMemory;
    for (values, 0..) |value, index| {
        if (index > 0) {
            out.append(allocator, ',') catch return error.OutOfMemory;
        }
        try jsonString(allocator, &out, value);
    }
    out.append(allocator, ']') catch return error.OutOfMemory;
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn jsonString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), value: []const u8) !void {
    out.append(allocator, '"') catch return error.OutOfMemory;
    for (value) |char| {
        switch (char) {
            '"' => out.appendSlice(allocator, "\\\"") catch return error.OutOfMemory,
            '\\' => out.appendSlice(allocator, "\\\\") catch return error.OutOfMemory,
            '\n' => out.appendSlice(allocator, "\\n") catch return error.OutOfMemory,
            '\r' => out.appendSlice(allocator, "\\r") catch return error.OutOfMemory,
            '\t' => out.appendSlice(allocator, "\\t") catch return error.OutOfMemory,
            else => out.append(allocator, char) catch return error.OutOfMemory,
        }
    }
    out.append(allocator, '"') catch return error.OutOfMemory;
}

test "ONNX preload library scan uses Zig filesystem checks" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();
    const cudart = try tmp.dir.createFile(std.testing.io, "libcudart.so.12", .{});
    cudart.close(std.testing.io);

    const root = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(root);
    const json = try ortLibsJson(
        std.testing.allocator,
        std.testing.io,
        null,
        "",
        "",
        root,
        "libcudart.so.12",
    );
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "libcudart.so.12") != null);
}

test "ONNX env file derives CUDA library directories in Zig" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();

    const root = try path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer std.testing.allocator.free(root);
    try makePath(root, &.{ "runtime", "onnxruntime", "lib" });
    try makePath(root, &.{ "runtime", "cuda", "lib" });

    const ort_path = try path.join(std.testing.allocator, &.{ root, "runtime", "onnxruntime", "lib", "libonnxruntime.so.1.25.0" });
    defer std.testing.allocator.free(ort_path);
    const cuda_path = try path.join(std.testing.allocator, &.{ root, "runtime", "cuda", "lib", "libcudart.so.12" });
    defer std.testing.allocator.free(cuda_path);

    const ort = try std.Io.Dir.cwd().createFile(std.testing.io, ort_path, .{});
    ort.close(std.testing.io);
    const cuda = try std.Io.Dir.cwd().createFile(std.testing.io, cuda_path, .{});
    cuda.close(std.testing.io);

    const env_file = try path.join(std.testing.allocator, &.{ root, env_file_name });
    defer std.testing.allocator.free(env_file);
    const text = try std.fmt.allocPrint(std.testing.allocator, "{{\"DART_INFERENCE_ORT_RUNTIME_LIBRARY\":\"{s}\"}}", .{ort_path});
    defer std.testing.allocator.free(text);
    const file = try std.Io.Dir.cwd().createFile(std.testing.io, env_file, .{});
    var writer = file.writer(std.testing.io, &.{});
    try writer.interface.writeAll(text);
    try writer.interface.flush();
    file.close(std.testing.io);

    const json = try ortLibsJson(
        std.testing.allocator,
        std.testing.io,
        env_file,
        "",
        "",
        "",
        "libcudart.so.12",
    );
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "libcudart.so.12") != null);
}

fn makePath(root: []const u8, parts: []const []const u8) !void {
    var current = try std.testing.allocator.dupe(u8, root);
    defer std.testing.allocator.free(current);
    for (parts) |part| {
        const next = try path.join(std.testing.allocator, &.{ current, part });
        std.testing.allocator.free(current);
        current = next;
        const c_path = try std.testing.allocator.dupeZ(u8, current);
        defer std.testing.allocator.free(c_path);
        if (std.c.mkdir(c_path.ptr, 0o755) != 0 and !isDir(std.testing.io, current)) {
            return error.MakeDirFailed;
        }
    }
}
