const std = @import("std");
const abi = @import("abi.zig");

pub const Result = extern struct {
    id: [*c]u8,
    score: f64,
};

pub const Error = error{
    InvalidHandle,
    InvalidDimension,
    InvalidId,
    InvalidVector,
    OutOfMemory,
};

const Entry = struct {
    id: []u8,
    values: []f64,
};

const Candidate = struct {
    index: usize,
    score: f64,
};

pub const Index = struct {
    allocator: std.mem.Allocator,
    dim: usize,
    entries: std.ArrayList(Entry) = .empty,

    pub fn init(allocator: std.mem.Allocator, dim: usize) Index {
        return .{ .allocator = allocator, .dim = dim };
    }

    pub fn deinit(self: *Index) void {
        self.clear();
        self.entries.deinit(self.allocator);
    }

    pub fn clear(self: *Index) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.id);
            self.allocator.free(entry.values);
        }
        self.entries.clearRetainingCapacity();
    }

    pub fn put(self: *Index, id: []const u8, values: []const f64) Error!void {
        if (id.len == 0) {
            return error.InvalidId;
        }
        if (values.len != self.dim) {
            return error.InvalidVector;
        }

        const copy = self.allocator.alloc(f64, values.len) catch return error.OutOfMemory;
        errdefer self.allocator.free(copy);
        @memcpy(copy, values);

        if (self.find(id)) |index| {
            self.allocator.free(self.entries.items[index].values);
            self.entries.items[index].values = copy;
            return;
        }

        const id_copy = self.allocator.dupe(u8, id) catch return error.OutOfMemory;
        errdefer self.allocator.free(id_copy);
        self.entries.append(self.allocator, .{ .id = id_copy, .values = copy }) catch return error.OutOfMemory;
    }

    pub fn remove(self: *Index, id: []const u8) bool {
        const index = self.find(id) orelse return false;
        const entry = self.entries.orderedRemove(index);
        self.allocator.free(entry.id);
        self.allocator.free(entry.values);
        return true;
    }

    pub fn search(
        self: *const Index,
        query: []const f64,
        top_k: usize,
        min_score: f64,
    ) Error![]Result {
        if (query.len != self.dim) {
            return error.InvalidVector;
        }
        if (top_k == 0 or self.entries.items.len == 0) {
            return &.{};
        }

        const limit = @min(top_k, self.entries.items.len);
        const candidates = self.allocator.alloc(Candidate, limit) catch return error.OutOfMemory;
        defer self.allocator.free(candidates);

        var count: usize = 0;
        for (self.entries.items, 0..) |entry, index| {
            const score = cosine(query, entry.values);
            if (score < min_score) {
                continue;
            }
            var pos = count;
            while (pos > 0 and candidates[pos - 1].score < score) {
                pos -= 1;
            }
            if (pos >= limit) {
                continue;
            }
            if (count < limit) {
                count += 1;
            }
            var move = count - 1;
            while (move > pos) : (move -= 1) {
                candidates[move] = candidates[move - 1];
            }
            candidates[pos] = .{ .index = index, .score = score };
        }

        if (count == 0) {
            return &.{};
        }

        const results = self.allocator.alloc(Result, count) catch return error.OutOfMemory;
        var produced: usize = 0;
        errdefer freeResults(self.allocator, results.ptr, @intCast(produced));
        while (produced < count) : (produced += 1) {
            const entry = self.entries.items[candidates[produced].index];
            const id = abi.copyString(entry.id);
            if (id == null) {
                return error.OutOfMemory;
            }
            results[produced] = .{ .id = id, .score = candidates[produced].score };
        }
        return results;
    }

    fn find(self: *const Index, id: []const u8) ?usize {
        for (self.entries.items, 0..) |entry, index| {
            if (std.mem.eql(u8, entry.id, id)) {
                return index;
            }
        }
        return null;
    }
};

pub fn create(allocator: std.mem.Allocator, dim: isize) Error!*Index {
    if (dim < 0) {
        return error.InvalidDimension;
    }
    const index = allocator.create(Index) catch return error.OutOfMemory;
    index.* = Index.init(allocator, @intCast(dim));
    return index;
}

pub fn destroy(index: *Index) void {
    const allocator = index.allocator;
    index.deinit();
    allocator.destroy(index);
}

pub fn fromHandle(handle: ?*anyopaque) Error!*Index {
    const raw = handle orelse return error.InvalidHandle;
    return @ptrCast(@alignCast(raw));
}

pub fn freeResults(allocator: std.mem.Allocator, results: [*c]Result, count: isize) void {
    if (results == null) {
        return;
    }
    if (count <= 0) {
        return;
    }
    const len: usize = @intCast(count);
    const items: [*]Result = @ptrCast(results);
    for (items[0..len]) |result| {
        abi.freeString(result.id);
    }
    allocator.free(items[0..len]);
}

pub fn normalize(values: []f64) void {
    var norm: f64 = 0;
    for (values) |value| {
        norm += value * value;
    }
    norm = std.math.sqrt(norm);
    if (norm <= 0) {
        return;
    }
    for (values) |*value| {
        value.* /= norm;
    }
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidHandle => "Zig vector index received an invalid handle.",
        error.InvalidDimension => "Zig vector index received an invalid vector dimension.",
        error.InvalidId => "Zig vector index received an empty vector id.",
        error.InvalidVector => "Zig vector index received a vector with the wrong dimension.",
        error.OutOfMemory => "Zig vector index ran out of memory.",
    };
}

fn cosine(a: []const f64, b: []const f64) f64 {
    var dot: f64 = 0;
    var norm_a: f64 = 0;
    var norm_b: f64 = 0;
    for (a, b) |left, right| {
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    const denom = std.math.sqrt(norm_a) * std.math.sqrt(norm_b);
    return if (denom > 0) dot / denom else 0;
}

test "vector normalization keeps zero vectors stable" {
    var values = [_]f64{ 0, 0, 0 };
    normalize(&values);
    try std.testing.expectEqual(@as(f64, 0), values[0]);
    try std.testing.expectEqual(@as(f64, 0), values[1]);
    try std.testing.expectEqual(@as(f64, 0), values[2]);
}

test "vector normalization scales to unit length" {
    var values = [_]f64{ 3, 4 };
    normalize(&values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), values[0], 0.0000000001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), values[1], 0.0000000001);
}

test "vector index searches by cosine similarity" {
    var index = Index.init(std.testing.allocator, 3);
    defer index.deinit();

    try index.put("x", &.{ 1, 0, 0 });
    try index.put("y", &.{ 0, 1, 0 });
    try index.put("z", &.{ 0, 0, 1 });

    const results = try index.search(&.{ 0.9, 0.1, 0 }, 2, -1);
    defer freeResults(std.testing.allocator, results.ptr, @intCast(results.len));

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("x", results[0].id[0..std.mem.len(results[0].id)]);
    try std.testing.expect(results[0].score > 0.9);
}

test "vector index replaces and removes entries" {
    var index = Index.init(std.testing.allocator, 2);
    defer index.deinit();

    try index.put("a", &.{ 1, 0 });
    try index.put("a", &.{ 0, 1 });
    try std.testing.expectEqual(@as(usize, 1), index.entries.items.len);

    const results = try index.search(&.{ 0, 1 }, 1, -1);
    defer freeResults(std.testing.allocator, results.ptr, @intCast(results.len));
    try std.testing.expectEqualStrings("a", results[0].id[0..std.mem.len(results[0].id)]);

    try std.testing.expect(index.remove("a"));
    try std.testing.expectEqual(@as(usize, 0), index.entries.items.len);
}
