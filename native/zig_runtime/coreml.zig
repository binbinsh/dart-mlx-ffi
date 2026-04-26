const std = @import("std");

const path = std.Io.Dir.path;
const head_index = 1_000_000;

const Layout = struct {
    allocator: std.mem.Allocator,
    root_path: []u8,
    pipeline_spec_path: ?[]u8 = null,
    model_config_path: ?[]u8 = null,
    monolithic_model_path: ?[]u8 = null,
    decode_chunks: [][]u8 = &.{},
    prefill_chunks: [][]u8 = &.{},
    sidecars: [][]u8 = &.{},

    fn deinit(self: *Layout) void {
        self.allocator.free(self.root_path);
        if (self.pipeline_spec_path) |value| self.allocator.free(value);
        if (self.model_config_path) |value| self.allocator.free(value);
        if (self.monolithic_model_path) |value| self.allocator.free(value);
        freePaths(self.allocator, self.decode_chunks);
        freePaths(self.allocator, self.prefill_chunks);
        freePaths(self.allocator, self.sidecars);
    }
};

const IndexedPath = struct {
    index: usize,
    value: []u8,
};

const Scan = struct {
    allocator: std.mem.Allocator,
    decode: std.ArrayList(IndexedPath) = .empty,
    prefill: std.ArrayList(IndexedPath) = .empty,
    sidecars: std.ArrayList([]u8) = .empty,

    fn deinit(self: *Scan) void {
        for (self.decode.items) |item| self.allocator.free(item.value);
        for (self.prefill.items) |item| self.allocator.free(item.value);
        for (self.sidecars.items) |item| self.allocator.free(item);
        self.decode.deinit(self.allocator);
        self.prefill.deinit(self.allocator);
        self.sidecars.deinit(self.allocator);
    }
};

pub fn layoutJson(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
) ![]u8 {
    var layout = try discover(allocator, io, root_path);
    defer layout.deinit();
    return writeJson(allocator, &layout);
}

fn discover(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
) !Layout {
    var layout = Layout{
        .allocator = allocator,
        .root_path = allocator.dupe(u8, root_path) catch return error.OutOfMemory,
    };
    errdefer layout.deinit();

    const stat = std.Io.Dir.cwd().statFile(io, root_path, .{}) catch return layout;
    if (stat.kind == .file and std.mem.endsWith(u8, root_path, ".json")) {
        layout.pipeline_spec_path = allocator.dupe(u8, root_path) catch return error.OutOfMemory;
        return layout;
    }
    if (stat.kind != .directory) {
        return layout;
    }

    var dir = std.Io.Dir.cwd().openDir(io, root_path, .{
        .access_sub_paths = true,
        .iterate = true,
    }) catch return layout;
    defer dir.close(io);

    if (fileExists(dir, io, "model_config.json")) {
        layout.model_config_path = try join(allocator, root_path, "model_config.json");
    }

    var scan = Scan{ .allocator = allocator };
    defer scan.deinit();
    var iter = dir.iterate();
    while (iter.next(io) catch return layout) |entry| {
        try scanEntry(allocator, root_path, entry, &scan, &layout);
    }

    std.sort.insertion(IndexedPath, scan.decode.items, {}, indexLessThan);
    std.sort.insertion(IndexedPath, scan.prefill.items, {}, indexLessThan);
    std.sort.insertion([]u8, scan.sidecars.items, {}, pathLessThan);
    layout.decode_chunks = try takeIndexed(allocator, &scan.decode);
    layout.prefill_chunks = try takeIndexed(allocator, &scan.prefill);
    layout.sidecars = try takePaths(allocator, &scan.sidecars);
    return layout;
}

fn scanEntry(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    entry: std.Io.Dir.Entry,
    scan: *Scan,
    layout: *Layout,
) !void {
    const name = entry.name;
    if (isBundle(name)) {
        const full_path = try join(allocator, root_path, name);
        errdefer allocator.free(full_path);
        if (std.mem.eql(u8, name, "model.mlmodelc") or
            std.mem.eql(u8, name, "model.mlpackage"))
        {
            if (layout.monolithic_model_path) |old| allocator.free(old);
            layout.monolithic_model_path = full_path;
            return;
        }
        if (chunkIndex(name, "chunk")) |index| {
            scan.decode.append(allocator, .{ .index = index, .value = full_path }) catch return error.OutOfMemory;
            return;
        }
        if (chunkIndex(name, "prefill_chunk")) |index| {
            scan.prefill.append(allocator, .{ .index = index, .value = full_path }) catch return error.OutOfMemory;
            return;
        }
        allocator.free(full_path);
    }
    if (entry.kind == .file and !std.mem.eql(u8, name, "model_config.json")) {
        const sidecar = try join(allocator, root_path, name);
        errdefer allocator.free(sidecar);
        scan.sidecars.append(allocator, sidecar) catch return error.OutOfMemory;
    }
}

fn isBundle(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".mlmodelc") or
        std.mem.endsWith(u8, name, ".mlpackage");
}

fn chunkIndex(name: []const u8, prefix: []const u8) ?usize {
    const stem = stripBundleSuffix(name) orelse return null;
    if (std.mem.eql(u8, prefix, "chunk") and std.mem.eql(u8, stem, "chunk_head")) {
        return head_index;
    }
    if (!std.mem.startsWith(u8, stem, prefix)) {
        return null;
    }
    var rest = stem[prefix.len..];
    if (rest.len > 0 and rest[0] == '_') {
        rest = rest[1..];
    }
    if (rest.len == 0) {
        return null;
    }
    for (rest) |char| {
        if (char < '0' or char > '9') {
            return null;
        }
    }
    return std.fmt.parseInt(usize, rest, 10) catch null;
}

fn stripBundleSuffix(name: []const u8) ?[]const u8 {
    if (std.mem.endsWith(u8, name, ".mlmodelc")) {
        return name[0 .. name.len - ".mlmodelc".len];
    }
    if (std.mem.endsWith(u8, name, ".mlpackage")) {
        return name[0 .. name.len - ".mlpackage".len];
    }
    return null;
}

fn takeIndexed(allocator: std.mem.Allocator, values: *std.ArrayList(IndexedPath)) ![][]u8 {
    if (values.items.len == 0) {
        return &.{};
    }
    const out = allocator.alloc([]u8, values.items.len) catch return error.OutOfMemory;
    errdefer allocator.free(out);
    for (values.items, 0..) |item, index| {
        out[index] = item.value;
    }
    values.clearRetainingCapacity();
    return out;
}

fn takePaths(allocator: std.mem.Allocator, values: *std.ArrayList([]u8)) ![][]u8 {
    if (values.items.len == 0) {
        return &.{};
    }
    const out = allocator.alloc([]u8, values.items.len) catch return error.OutOfMemory;
    errdefer allocator.free(out);
    for (values.items, 0..) |item, index| {
        out[index] = item;
    }
    values.clearRetainingCapacity();
    return out;
}

fn freePaths(allocator: std.mem.Allocator, values: [][]u8) void {
    for (values) |value| allocator.free(value);
    if (values.len > 0) {
        allocator.free(values);
    }
}

fn fileExists(dir: std.Io.Dir, io: std.Io, file_name: []const u8) bool {
    const stat = dir.statFile(io, file_name, .{}) catch return false;
    return stat.kind == .file;
}

fn join(allocator: std.mem.Allocator, parent: []const u8, child: []const u8) ![]u8 {
    return path.join(allocator, &.{ parent, child }) catch return error.OutOfMemory;
}

fn indexLessThan(_: void, lhs: IndexedPath, rhs: IndexedPath) bool {
    return lhs.index < rhs.index;
}

fn pathLessThan(_: void, lhs: []u8, rhs: []u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

fn writeJson(allocator: std.mem.Allocator, layout: *const Layout) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"root_path\":");
    try jsonString(allocator, &out, layout.root_path);
    try out.appendSlice(allocator, ",\"pipeline_spec_path\":");
    try jsonOptString(allocator, &out, layout.pipeline_spec_path);
    try out.appendSlice(allocator, ",\"model_config_path\":");
    try jsonOptString(allocator, &out, layout.model_config_path);
    try out.appendSlice(allocator, ",\"monolithic_model_path\":");
    try jsonOptString(allocator, &out, layout.monolithic_model_path);
    try out.appendSlice(allocator, ",\"decode_chunks\":");
    try jsonList(allocator, &out, layout.decode_chunks);
    try out.appendSlice(allocator, ",\"prefill_chunks\":");
    try jsonList(allocator, &out, layout.prefill_chunks);
    try out.appendSlice(allocator, ",\"sidecars\":");
    try jsonList(allocator, &out, layout.sidecars);
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn jsonOptString(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: ?[]const u8,
) !void {
    if (value) |text| {
        try jsonString(allocator, out, text);
    } else {
        try out.appendSlice(allocator, "null");
    }
}

fn jsonList(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    values: []const []u8,
) !void {
    try out.append(allocator, '[');
    for (values, 0..) |value, index| {
        if (index > 0) {
            try out.append(allocator, ',');
        }
        try jsonString(allocator, out, value);
    }
    try out.append(allocator, ']');
}

fn jsonString(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: []const u8,
) !void {
    try out.append(allocator, '"');
    for (value) |char| {
        switch (char) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, char),
        }
    }
    try out.append(allocator, '"');
}

fn tmpRoot(allocator: std.mem.Allocator, tmp: std.testing.TmpDir) ![]u8 {
    return path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
}

fn makeDir(value: []const u8) !void {
    const c_path = try std.testing.allocator.dupeZ(u8, value);
    defer std.testing.allocator.free(c_path);
    if (std.c.mkdir(c_path.ptr, 0o755) != 0) {
        const stat = std.Io.Dir.cwd().statFile(std.testing.io, value, .{}) catch
            return error.MakeDirFailed;
        if (stat.kind != .directory) {
            return error.MakeDirFailed;
        }
    }
}

test "Core ML chunk layout is discovered in Zig" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();

    const root = try tmpRoot(std.testing.allocator, tmp);
    defer std.testing.allocator.free(root);
    const config = try tmp.dir.createFile(std.testing.io, "model_config.json", .{});
    config.close(std.testing.io);
    for ([_][]const u8{
        "chunk2.mlmodelc",
        "chunk1.mlmodelc",
        "chunk_3.mlpackage",
        "chunk_head.mlpackage",
        "chunk_0_vision.mlpackage",
        "prefill_chunk1.mlmodelc",
        "prefill_chunk_2.mlpackage",
    }) |name| {
        const full_path = try join(std.testing.allocator, root, name);
        defer std.testing.allocator.free(full_path);
        try makeDir(full_path);
    }
    const sidecar = try tmp.dir.createFile(std.testing.io, "embed_weight.bin", .{});
    sidecar.close(std.testing.io);

    var layout = try discover(std.testing.allocator, std.testing.io, root);
    defer layout.deinit();
    try std.testing.expect(layout.model_config_path != null);
    try std.testing.expectEqual(@as(usize, 4), layout.decode_chunks.len);
    try std.testing.expect(std.mem.endsWith(u8, layout.decode_chunks[0], "chunk1.mlmodelc"));
    try std.testing.expect(std.mem.endsWith(u8, layout.decode_chunks[1], "chunk2.mlmodelc"));
    try std.testing.expect(std.mem.endsWith(u8, layout.decode_chunks[2], "chunk_3.mlpackage"));
    try std.testing.expect(std.mem.endsWith(u8, layout.decode_chunks[3], "chunk_head.mlpackage"));
    try std.testing.expectEqual(@as(usize, 2), layout.prefill_chunks.len);
    try std.testing.expect(std.mem.endsWith(u8, layout.sidecars[0], "embed_weight.bin"));
}

test "Core ML pipeline JSON file is discovered in Zig" {
    var tmp = std.testing.tmpDir(.{ .access_sub_paths = true, .iterate = true });
    defer tmp.cleanup();

    const root = try tmpRoot(std.testing.allocator, tmp);
    defer std.testing.allocator.free(root);
    const spec_path = try join(std.testing.allocator, root, "pipeline.json");
    defer std.testing.allocator.free(spec_path);
    const spec = try tmp.dir.createFile(std.testing.io, "pipeline.json", .{});
    spec.close(std.testing.io);

    var layout = try discover(std.testing.allocator, std.testing.io, spec_path);
    defer layout.deinit();
    try std.testing.expect(layout.pipeline_spec_path != null);
    try std.testing.expectEqual(@as(usize, 0), layout.decode_chunks.len);
}

test "Core ML layout JSON keeps missing paths empty" {
    const json = try layoutJson(std.testing.allocator, std.testing.io, "missing_coreml_bundle");
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"root_path\":\"missing_coreml_bundle\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"decode_chunks\":[]") != null);
}
