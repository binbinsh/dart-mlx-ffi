const std = @import("std");
const abi = @import("abi.zig");
const vector = @import("vector.zig");

const VecResult = vector.Result;

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

fn cStringOrEmpty(value: [*c]const u8) []const u8 {
    if (value == null) {
        return "";
    }
    return value[0..std.mem.len(value)];
}

export fn dinf_vec_new(dim: isize) ?*anyopaque {
    return vector.create(std.heap.c_allocator, dim) catch null;
}

export fn dinf_vec_free(handle: ?*anyopaque) void {
    const index = vector.fromHandle(handle) catch return;
    vector.destroy(index);
}

export fn dinf_vec_len(handle: ?*anyopaque) isize {
    const index = vector.fromHandle(handle) catch return 0;
    return @intCast(index.entries.items.len);
}

export fn dinf_vec_clear(handle: ?*anyopaque) void {
    const index = vector.fromHandle(handle) catch return;
    index.clear();
}

export fn dinf_vec_put(
    handle: ?*anyopaque,
    id: [*c]const u8,
    values: [*c]const f64,
    len: isize,
    error_out: ?*[*c]u8,
) i32 {
    const index = vector.fromHandle(handle) catch |err| {
        setError(error_out, vector.errorMessage(err));
        return 1;
    };
    if (id == null or len < 0 or (len > 0 and values == null)) {
        setError(error_out, vector.errorMessage(error.InvalidVector));
        return 1;
    }
    const items = if (len == 0) &.{} else values[0..@intCast(len)];
    index.put(cStringOrEmpty(id), items) catch |err| {
        setError(error_out, vector.errorMessage(err));
        return 1;
    };
    return 0;
}

export fn dinf_vec_put_many(
    handle: ?*anyopaque,
    ids: [*c][*c]const u8,
    values: [*c]const f64,
    count: isize,
    dim: isize,
    error_out: ?*[*c]u8,
) i32 {
    const index = vector.fromHandle(handle) catch |err| {
        setError(error_out, vector.errorMessage(err));
        return 1;
    };
    if (count < 0 or dim < 0 or (count > 0 and ids == null) or (count > 0 and dim > 0 and values == null)) {
        setError(error_out, vector.errorMessage(error.InvalidVector));
        return 1;
    }
    if (@as(usize, @intCast(dim)) != index.dim) {
        setError(error_out, vector.errorMessage(error.InvalidVector));
        return 1;
    }
    const entry_count: usize = @intCast(count);
    const vector_dim: usize = @intCast(dim);
    var offset: usize = 0;
    var entry_index: usize = 0;
    while (entry_index < entry_count) : (entry_index += 1) {
        const id = ids[entry_index];
        if (id == null) {
            setError(error_out, vector.errorMessage(error.InvalidId));
            return 1;
        }
        const items = if (vector_dim == 0) &.{} else values[offset .. offset + vector_dim];
        index.put(cStringOrEmpty(id), items) catch |err| {
            setError(error_out, vector.errorMessage(err));
            return 1;
        };
        offset += vector_dim;
    }
    return 0;
}

export fn dinf_vec_remove(handle: ?*anyopaque, id: [*c]const u8) i32 {
    const index = vector.fromHandle(handle) catch return 0;
    return if (index.remove(cStringOrEmpty(id))) 1 else 0;
}

export fn dinf_vec_search(
    handle: ?*anyopaque,
    query: [*c]const f64,
    len: isize,
    top_k: isize,
    min_score: f64,
    results_out: ?*[*c]VecResult,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const results_ptr = results_out orelse {
        setError(error_out, "Zig vector search results_out is null.");
        return 1;
    };
    const count = count_out orelse {
        setError(error_out, "Zig vector search count_out is null.");
        return 1;
    };
    results_ptr.* = null;
    count.* = 0;
    const index = vector.fromHandle(handle) catch |err| {
        setError(error_out, vector.errorMessage(err));
        return 1;
    };
    if (len < 0 or (len > 0 and query == null)) {
        setError(error_out, vector.errorMessage(error.InvalidVector));
        return 1;
    }
    if (top_k <= 0) {
        return 0;
    }
    const items = if (len == 0) &.{} else query[0..@intCast(len)];
    const results = index.search(items, @intCast(top_k), min_score) catch |err| {
        setError(error_out, vector.errorMessage(err));
        return 1;
    };
    if (results.len == 0) {
        return 0;
    }
    count.* = @intCast(results.len);
    results_ptr.* = results.ptr;
    return 0;
}

export fn dinf_vec_free_results(results: [*c]VecResult, count: isize) void {
    vector.freeResults(std.heap.c_allocator, results, count);
}

export fn dinf_vec_l2_norm(values: [*c]f64, len: isize) i32 {
    if (len < 0 or (len > 0 and values == null)) {
        return 1;
    }
    if (len == 0) {
        return 0;
    }
    vector.normalize(values[0..@intCast(len)]);
    return 0;
}
