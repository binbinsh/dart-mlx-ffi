// C ABI surface for the Qwen2 byte-level BPE tokenizer.  See
// qwen2_bpe.zig for algorithmic details.  Memory model mirrors bpe_api:
// - The Dart caller owns vocab/merge string arrays and special-token data
//   for the duration of the `dinf_qwen2_*_new` call only.  The Zig
//   tokenizer copies everything it needs.
// - `dinf_qwen2_bpe_encode` writes into a Dart-supplied i64 buffer; if
//   the buffer is too small the call fails with `error.InvalidInput` and
//   `count_out` reports the required length.

const std = @import("std");
const abi = @import("abi.zig");
const qwen2 = @import("qwen2_bpe.zig");

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

export fn dinf_qwen2_bpe_new(
    vocab_keys: [*c]const [*c]const u8,
    vocab_ids: [*c]const i64,
    vocab_count: isize,
    merge_keys: [*c]const [*c]const u8,
    merge_count: isize,
    special_texts: [*c]const [*c]const u8,
    special_ids: [*c]const i64,
    special_count: isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    if (vocab_count < 0 or merge_count < 0 or special_count < 0 or
        (vocab_count > 0 and (vocab_keys == null or vocab_ids == null)) or
        (merge_count > 0 and merge_keys == null) or
        (special_count > 0 and (special_texts == null or special_ids == null)))
    {
        setError(error_out, qwen2.errorMessage(error.InvalidInput));
        return null;
    }

    const tokenizer = qwen2.create(std.heap.c_allocator) catch |err| {
        setError(error_out, qwen2.errorMessage(err));
        return null;
    };
    errdefer qwen2.destroy(tokenizer);

    const vocab_len: usize = @intCast(vocab_count);
    var index: usize = 0;
    while (index < vocab_len) : (index += 1) {
        const key = vocab_keys[index];
        if (key == null) {
            setError(error_out, qwen2.errorMessage(error.InvalidInput));
            return null;
        }
        tokenizer.addVocab(key[0..std.mem.len(key)], vocab_ids[index]) catch |err| {
            setError(error_out, qwen2.errorMessage(err));
            return null;
        };
    }

    const merge_len: usize = @intCast(merge_count);
    index = 0;
    while (index < merge_len) : (index += 1) {
        const key = merge_keys[index];
        if (key == null) {
            setError(error_out, qwen2.errorMessage(error.InvalidInput));
            return null;
        }
        tokenizer.addMerge(key[0..std.mem.len(key)], index) catch |err| {
            setError(error_out, qwen2.errorMessage(err));
            return null;
        };
    }

    const special_len: usize = @intCast(special_count);
    index = 0;
    while (index < special_len) : (index += 1) {
        const text = special_texts[index];
        if (text == null) {
            setError(error_out, qwen2.errorMessage(error.InvalidInput));
            return null;
        }
        tokenizer.addSpecial(text[0..std.mem.len(text)], special_ids[index]) catch |err| {
            setError(error_out, qwen2.errorMessage(err));
            return null;
        };
    }

    return @ptrCast(tokenizer);
}

export fn dinf_qwen2_bpe_free(handle: ?*anyopaque) void {
    const tokenizer = qwen2.fromHandle(handle) catch return;
    qwen2.destroy(tokenizer);
}

/// Encode `text` (NUL-terminated UTF-8) into the caller-supplied `ids`
/// buffer.  On success returns 0 and writes the produced count to
/// `count_out`.  If `max_len` is too small, returns 1, fills `count_out`
/// with the produced count so far (always <= max_len), and writes
/// "buffer too small" to `error_out` so the caller can retry.
export fn dinf_qwen2_bpe_encode(
    handle: ?*anyopaque,
    text: [*c]const u8,
    max_len: isize,
    ids: [*c]i64,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const count = count_out orelse {
        setError(error_out, qwen2.errorMessage(error.InvalidInput));
        return 1;
    };
    count.* = 0;
    const tokenizer = qwen2.fromHandle(handle) catch |err| {
        setError(error_out, qwen2.errorMessage(err));
        return 1;
    };
    if (text == null or max_len < 0 or (max_len > 0 and ids == null)) {
        setError(error_out, qwen2.errorMessage(error.InvalidInput));
        return 1;
    }

    const allocator = std.heap.c_allocator;
    var collected: std.ArrayList(i64) = .empty;
    defer collected.deinit(allocator);
    tokenizer.encode(text[0..std.mem.len(text)], &collected) catch |err| {
        setError(error_out, qwen2.errorMessage(err));
        return 1;
    };

    const produced = collected.items.len;
    count.* = @intCast(produced);
    const cap: usize = @intCast(max_len);
    if (produced > cap) {
        setError(error_out, "qwen2 bpe output buffer too small");
        return 1;
    }
    if (produced > 0) {
        @memcpy(ids[0..produced], collected.items[0..produced]);
    }
    return 0;
}
