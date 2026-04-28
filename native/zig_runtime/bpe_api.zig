const std = @import("std");
const abi = @import("abi.zig");
const bpe = @import("bpe.zig");

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

export fn dinf_bpe_new(
    vocab_keys: [*c]const [*c]const u8,
    vocab_ids: [*c]const i64,
    vocab_count: isize,
    merge_keys: [*c]const [*c]const u8,
    merge_count: isize,
    bos_id: i64,
    eos_id: i64,
    pad_id: i64,
    unk_id: i64,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    if (vocab_count < 0 or merge_count < 0 or
        (vocab_count > 0 and (vocab_keys == null or vocab_ids == null)) or
        (merge_count > 0 and merge_keys == null))
    {
        setError(error_out, bpe.errorMessage(error.InvalidInput));
        return null;
    }

    const tokenizer = bpe.create(std.heap.c_allocator, bos_id, eos_id, pad_id, unk_id) catch |err| {
        setError(error_out, bpe.errorMessage(err));
        return null;
    };
    errdefer bpe.destroy(tokenizer);

    const vocab_len: usize = @intCast(vocab_count);
    var index: usize = 0;
    while (index < vocab_len) : (index += 1) {
        const key = vocab_keys[index];
        if (key == null) {
            setError(error_out, bpe.errorMessage(error.InvalidInput));
            return null;
        }
        tokenizer.addVocab(key[0..std.mem.len(key)], vocab_ids[index]) catch |err| {
            setError(error_out, bpe.errorMessage(err));
            return null;
        };
    }

    const merge_len: usize = @intCast(merge_count);
    index = 0;
    while (index < merge_len) : (index += 1) {
        const key = merge_keys[index];
        if (key == null) {
            setError(error_out, bpe.errorMessage(error.InvalidInput));
            return null;
        }
        tokenizer.addMerge(key[0..std.mem.len(key)], index) catch |err| {
            setError(error_out, bpe.errorMessage(err));
            return null;
        };
    }

    return @ptrCast(tokenizer);
}

export fn dinf_bpe_free(handle: ?*anyopaque) void {
    const tokenizer = bpe.fromHandle(handle) catch return;
    bpe.destroy(tokenizer);
}

export fn dinf_bpe_encode(
    handle: ?*anyopaque,
    text: [*c]const u8,
    max_len: isize,
    ids: [*c]i64,
    starts: [*c]i32,
    ends: [*c]i32,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const count = count_out orelse {
        setError(error_out, bpe.errorMessage(error.InvalidInput));
        return 1;
    };
    count.* = 0;
    const tokenizer = bpe.fromHandle(handle) catch |err| {
        setError(error_out, bpe.errorMessage(err));
        return 1;
    };
    if (text == null or max_len <= 0 or ids == null or starts == null or ends == null) {
        setError(error_out, bpe.errorMessage(error.InvalidInput));
        return 1;
    }
    const len: usize = @intCast(max_len);
    const written = tokenizer.encode(
        text[0..std.mem.len(text)],
        ids[0..len],
        starts[0..len],
        ends[0..len],
    ) catch |err| {
        setError(error_out, bpe.errorMessage(err));
        return 1;
    };
    count.* = @intCast(written);
    return 0;
}

export fn dinf_bpe_fill(
    handle: ?*anyopaque,
    text: [*c]const u8,
    values: [*c]i64,
    mask: [*c]i64,
    len: isize,
    offset: isize,
    width: isize,
    starts: [*c]i32,
    ends: [*c]i32,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const count = count_out orelse {
        setError(error_out, bpe.errorMessage(error.InvalidInput));
        return 1;
    };
    count.* = 0;
    const tokenizer = bpe.fromHandle(handle) catch |err| {
        setError(error_out, bpe.errorMessage(err));
        return 1;
    };
    if (text == null or len < 0 or offset < 0 or width <= 0 or
        (len > 0 and (values == null or mask == null)) or
        starts == null or ends == null)
    {
        setError(error_out, bpe.errorMessage(error.InvalidInput));
        return 1;
    }
    const item_count: usize = @intCast(len);
    const row_offset: usize = @intCast(offset);
    const row_width: usize = @intCast(width);
    if (row_width > item_count or row_offset > item_count - row_width) {
        setError(error_out, bpe.errorMessage(error.InvalidInput));
        return 1;
    }
    const value_items = values[0..item_count];
    const mask_items = mask[0..item_count];
    const written = tokenizer.fill(
        text[0..std.mem.len(text)],
        value_items[row_offset .. row_offset + row_width],
        mask_items[row_offset .. row_offset + row_width],
        starts[0..row_width],
        ends[0..row_width],
    ) catch |err| {
        setError(error_out, bpe.errorMessage(err));
        return 1;
    };
    count.* = @intCast(written);
    return 0;
}
