const std = @import("std");
const abi = @import("abi.zig");
const target = @import("target.zig");

const Match = target.Match;

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

export fn dinf_tgt_new(
    homographs: [*c]const [*c]const u8,
    homograph_count: isize,
    homograph_offsets: [*c]const isize,
    homograph_ids: [*c]const i32,
    homograph_id_count: isize,
    polyphones: [*c]const [*c]const u8,
    polyphone_count: isize,
    polyphone_offsets: [*c]const isize,
    polyphone_ids: [*c]const i32,
    polyphone_id_count: isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    if (homograph_count < 0 or homograph_id_count < 0 or
        polyphone_count < 0 or polyphone_id_count < 0 or
        (homograph_count > 0 and homographs == null) or
        (homograph_count > 0 and homograph_offsets == null) or
        (homograph_id_count > 0 and homograph_ids == null) or
        (polyphone_count > 0 and polyphones == null) or
        (polyphone_count > 0 and polyphone_offsets == null) or
        (polyphone_id_count > 0 and polyphone_ids == null))
    {
        setError(error_out, target.errorMessage(error.InvalidInput));
        return null;
    }
    const allocator = std.heap.c_allocator;
    const homo = readStrings(allocator, homographs, homograph_count, error_out) orelse return null;
    defer if (homo.len > 0) allocator.free(homo);
    const homo_ids = readIdRows(
        allocator,
        homograph_offsets,
        homograph_count,
        homograph_ids,
        homograph_id_count,
        error_out,
    ) orelse return null;
    defer if (homo_ids.len > 0) allocator.free(homo_ids);
    const poly = readStrings(allocator, polyphones, polyphone_count, error_out) orelse return null;
    defer if (poly.len > 0) allocator.free(poly);
    const poly_ids = readIdRows(
        allocator,
        polyphone_offsets,
        polyphone_count,
        polyphone_ids,
        polyphone_id_count,
        error_out,
    ) orelse return null;
    defer if (poly_ids.len > 0) allocator.free(poly_ids);
    const matcher = target.create(allocator, homo, homo_ids, poly, poly_ids) catch |err| {
        setError(error_out, target.errorMessage(err));
        return null;
    };
    return @ptrCast(matcher);
}

export fn dinf_tgt_free(handle: ?*anyopaque) void {
    const matcher = target.fromHandle(handle) catch return;
    target.destroy(matcher);
}

export fn dinf_tgt_homographs(
    handle: ?*anyopaque,
    text: [*c]const u8,
    matches_out: ?*[*c]Match,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    return match(handle, text, matches_out, count_out, error_out, true);
}

export fn dinf_tgt_polyphones(
    handle: ?*anyopaque,
    text: [*c]const u8,
    matches_out: ?*[*c]Match,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    return match(handle, text, matches_out, count_out, error_out, false);
}

export fn dinf_tgt_free_matches(matches: [*c]Match, count: isize) void {
    target.freeMatches(std.heap.c_allocator, matches, count);
}

fn match(
    handle: ?*anyopaque,
    text: [*c]const u8,
    matches_out: ?*[*c]Match,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
    homographs: bool,
) i32 {
    const out = matches_out orelse {
        setError(error_out, target.errorMessage(error.InvalidInput));
        return 1;
    };
    const count = count_out orelse {
        setError(error_out, target.errorMessage(error.InvalidInput));
        return 1;
    };
    out.* = null;
    count.* = 0;
    const matcher = target.fromHandle(handle) catch |err| {
        setError(error_out, target.errorMessage(err));
        return 1;
    };
    if (text == null) {
        setError(error_out, target.errorMessage(error.InvalidInput));
        return 1;
    }
    const items = if (homographs)
        matcher.homographMatches(text[0..std.mem.len(text)])
    else
        matcher.polyphoneMatches(text[0..std.mem.len(text)]);
    const matches = items catch |err| {
        setError(error_out, target.errorMessage(err));
        return 1;
    };
    if (matches.len == 0) {
        return 0;
    }
    for (matches) |*item| {
        const ids = matcher.idsFor(homographs, item.index);
        if (ids.len > 0) {
            item.ids = ids.ptr;
            item.id_count = @intCast(ids.len);
        }
    }
    out.* = matches.ptr;
    count.* = @intCast(matches.len);
    return 0;
}

fn readStrings(
    allocator: std.mem.Allocator,
    values: [*c]const [*c]const u8,
    count: isize,
    error_out: ?*[*c]u8,
) ?[][]const u8 {
    if (count <= 0) return &.{};
    const len: usize = @intCast(count);
    const out = allocator.alloc([]const u8, len) catch {
        setError(error_out, target.errorMessage(error.OutOfMemory));
        return null;
    };
    for (out, 0..) |*item, index| {
        const value = values[index];
        if (value == null) {
            allocator.free(out);
            setError(error_out, target.errorMessage(error.InvalidInput));
            return null;
        }
        item.* = value[0..std.mem.len(value)];
    }
    return out;
}

fn readIdRows(
    allocator: std.mem.Allocator,
    offsets: [*c]const isize,
    row_count: isize,
    ids: [*c]const i32,
    id_count: isize,
    error_out: ?*[*c]u8,
) ?[][]const i32 {
    if (row_count <= 0) return &.{};
    const rows_len: usize = @intCast(row_count);
    const out = allocator.alloc([]const i32, rows_len) catch {
        setError(error_out, target.errorMessage(error.OutOfMemory));
        return null;
    };
    const id_items: []const i32 = if (id_count == 0) &.{} else ids[0..@intCast(id_count)];
    for (out, 0..) |*row, index| {
        const start_raw = offsets[index];
        const end_raw = offsets[index + 1];
        if (start_raw < 0 or end_raw < start_raw) {
            allocator.free(out);
            setError(error_out, target.errorMessage(error.InvalidInput));
            return null;
        }
        const start: usize = @intCast(start_raw);
        const end: usize = @intCast(end_raw);
        if (end > id_items.len) {
            allocator.free(out);
            setError(error_out, target.errorMessage(error.InvalidInput));
            return null;
        }
        row.* = id_items[start..end];
    }
    return out;
}
