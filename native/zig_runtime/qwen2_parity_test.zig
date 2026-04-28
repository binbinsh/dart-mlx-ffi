// Parity test: runs the Qwen2 BPE tokenizer against the cosyvoice2
// fixture corpus and compares ids against the HuggingFace ground truth.
//
// This test is gated on environment variables so it does not run as part
// of the default `zig test` flow (which has no access to the model
// directory).  Activate it by:
//
//   QWEN2_TOKENIZER_DIR=/path/to/CosyVoice-BlankEN \
//   QWEN2_TOKENIZER_CASES=/path/to/qwen2_tokenizer_cases.json \
//   zig test qwen2_parity_test.zig
//
// Files expected in QWEN2_TOKENIZER_DIR:
//   vocab.json            -- {"piece": id, ...}     (151643 entries)
//   merges.txt            -- "#version: ..."\n then "<piece A> <piece B>"
//
// QWEN2_TOKENIZER_CASES: JSON of {"cases":[{"text":..., "ids":[...]}]}.

const std = @import("std");
const qwen2 = @import("qwen2_bpe.zig");

// 4 base + 16 cosyvoice specials.
const base_specials = [_]struct { text: []const u8, id: i64 }{
    .{ .text = "<|endoftext|>", .id = 151643 },
    .{ .text = "<|im_start|>", .id = 151644 },
    .{ .text = "<|im_end|>", .id = 151645 },
    .{ .text = "<|endofprompt|>", .id = 151646 },
    .{ .text = "[breath]", .id = 151647 },
    .{ .text = "<strong>", .id = 151648 },
    .{ .text = "</strong>", .id = 151649 },
    .{ .text = "[noise]", .id = 151650 },
    .{ .text = "[laughter]", .id = 151651 },
    .{ .text = "[cough]", .id = 151652 },
    .{ .text = "[clucking]", .id = 151653 },
    .{ .text = "[accent]", .id = 151654 },
    .{ .text = "[quick_breath]", .id = 151655 },
    .{ .text = "<laughter>", .id = 151656 },
    .{ .text = "</laughter>", .id = 151657 },
    .{ .text = "[hissing]", .id = 151658 },
    .{ .text = "[sigh]", .id = 151659 },
    .{ .text = "[vocalized-noise]", .id = 151660 },
    .{ .text = "[lipsmack]", .id = 151661 },
    .{ .text = "[mn]", .id = 151662 },
};

extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

fn loadEnv(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    // POSIX libc lookup.  This test is gated on Linux/macOS hosts where
    // the cosyvoice2 fixtures live; Windows would need an alternate path.
    const c_name = try allocator.dupeZ(u8, name);
    defer allocator.free(c_name);
    const raw = getenv(c_name.ptr) orelse return null;
    const slice = std.mem.span(raw);
    if (slice.len == 0) return null;
    return try allocator.dupe(u8, slice);
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(64 * 1024 * 1024),
    );
}

fn loadVocab(
    allocator: std.mem.Allocator,
    tokenizer: *qwen2.Tokenizer,
    vocab_json: []const u8,
) !void {
    // Minimal hand-rolled scanner: vocab.json is a flat string->int object.
    // Format: `{"piece":id,"piece":id,...}`.  Pieces may contain `\"`,
    // `\\`, `\n`, `\u00xx`.  We unescape on the fly.
    var i: usize = 0;
    // Skip leading whitespace and `{`.
    while (i < vocab_json.len and std.ascii.isWhitespace(vocab_json[i])) i += 1;
    if (i >= vocab_json.len or vocab_json[i] != '{') return error.InvalidJson;
    i += 1;

    var key_buf: std.ArrayList(u8) = .empty;
    defer key_buf.deinit(allocator);

    while (i < vocab_json.len) {
        while (i < vocab_json.len and (std.ascii.isWhitespace(vocab_json[i]) or vocab_json[i] == ',')) i += 1;
        if (i < vocab_json.len and vocab_json[i] == '}') break;
        if (i >= vocab_json.len or vocab_json[i] != '"') return error.InvalidJson;
        i += 1;
        key_buf.clearRetainingCapacity();
        while (i < vocab_json.len and vocab_json[i] != '"') {
            if (vocab_json[i] == '\\') {
                i += 1;
                if (i >= vocab_json.len) return error.InvalidJson;
                switch (vocab_json[i]) {
                    '"' => try key_buf.append(allocator, '"'),
                    '\\' => try key_buf.append(allocator, '\\'),
                    '/' => try key_buf.append(allocator, '/'),
                    'n' => try key_buf.append(allocator, '\n'),
                    'r' => try key_buf.append(allocator, '\r'),
                    't' => try key_buf.append(allocator, '\t'),
                    'b' => try key_buf.append(allocator, 8),
                    'f' => try key_buf.append(allocator, 12),
                    'u' => {
                        if (i + 4 >= vocab_json.len) return error.InvalidJson;
                        const cp = try std.fmt.parseInt(u21, vocab_json[i + 1 .. i + 5], 16);
                        var ubuf: [4]u8 = undefined;
                        const n = try std.unicode.utf8Encode(cp, ubuf[0..]);
                        try key_buf.appendSlice(allocator, ubuf[0..n]);
                        i += 4;
                    },
                    else => return error.InvalidJson,
                }
                i += 1;
            } else {
                try key_buf.append(allocator, vocab_json[i]);
                i += 1;
            }
        }
        if (i >= vocab_json.len) return error.InvalidJson;
        i += 1; // closing quote
        while (i < vocab_json.len and std.ascii.isWhitespace(vocab_json[i])) i += 1;
        if (i >= vocab_json.len or vocab_json[i] != ':') return error.InvalidJson;
        i += 1;
        while (i < vocab_json.len and std.ascii.isWhitespace(vocab_json[i])) i += 1;
        const num_start = i;
        while (i < vocab_json.len and (std.ascii.isDigit(vocab_json[i]) or vocab_json[i] == '-')) i += 1;
        const id = try std.fmt.parseInt(i64, vocab_json[num_start..i], 10);
        try tokenizer.addVocab(key_buf.items, id);
    }
}

fn loadMerges(
    allocator: std.mem.Allocator,
    tokenizer: *qwen2.Tokenizer,
    merges_text: []const u8,
) !void {
    _ = allocator;
    var rank: usize = 0;
    var it = std.mem.splitScalar(u8, merges_text, '\n');
    while (it.next()) |raw| {
        var line = raw;
        if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
        if (line.len == 0) continue;
        if (line[0] == '#') continue; // header
        try tokenizer.addMerge(line, rank);
        rank += 1;
    }
}

fn parseCases(
    allocator: std.mem.Allocator,
    json_text: []const u8,
) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
}

test "qwen2 parity against cosyvoice2 fixture" {
    const allocator = std.testing.allocator;
    const dir = try loadEnv(allocator, "QWEN2_TOKENIZER_DIR") orelse {
        std.debug.print("\n[skip] QWEN2_TOKENIZER_DIR not set\n", .{});
        return;
    };
    defer allocator.free(dir);
    const cases_path = try loadEnv(allocator, "QWEN2_TOKENIZER_CASES") orelse {
        std.debug.print("\n[skip] QWEN2_TOKENIZER_CASES not set\n", .{});
        return;
    };
    defer allocator.free(cases_path);

    const vocab_path = try std.fs.path.join(allocator, &.{ dir, "vocab.json" });
    defer allocator.free(vocab_path);
    const merges_path = try std.fs.path.join(allocator, &.{ dir, "merges.txt" });
    defer allocator.free(merges_path);

    const vocab_json = try readFileAlloc(allocator, vocab_path);
    defer allocator.free(vocab_json);
    const merges_text = try readFileAlloc(allocator, merges_path);
    defer allocator.free(merges_text);
    const cases_json = try readFileAlloc(allocator, cases_path);
    defer allocator.free(cases_json);

    var tokenizer = qwen2.Tokenizer.init(allocator);
    defer tokenizer.deinit();

    try loadVocab(allocator, &tokenizer, vocab_json);
    try loadMerges(allocator, &tokenizer, merges_text);
    for (base_specials) |sp| {
        try tokenizer.addSpecial(sp.text, sp.id);
    }

    var parsed = try parseCases(allocator, cases_json);
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases") orelse return error.InvalidJson;

    var failures: usize = 0;
    var case_index: usize = 0;
    for (cases.array.items) |case_value| {
        defer case_index += 1;
        const obj = case_value.object;
        const text = obj.get("text").?.string;
        const expected = obj.get("ids").?.array.items;

        var actual: std.ArrayList(i64) = .empty;
        defer actual.deinit(allocator);
        try tokenizer.encode(text, &actual);

        var ok = actual.items.len == expected.len;
        if (ok) {
            for (expected, actual.items) |e, a| {
                if (e.integer != a) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            failures += 1;
            std.debug.print(
                "\n[case {d}] text={s}\n  expected ({d}): ",
                .{ case_index, text, expected.len },
            );
            for (expected) |e| std.debug.print("{d} ", .{e.integer});
            std.debug.print("\n  actual   ({d}): ", .{actual.items.len});
            for (actual.items) |a| std.debug.print("{d} ", .{a});
            std.debug.print("\n", .{});
        }
    }

    if (failures != 0) {
        std.debug.print("\n{d}/{d} parity cases failed\n", .{ failures, cases.array.items.len });
        return error.ParityMismatch;
    }
    std.debug.print("\n{d}/{d} parity cases passed\n", .{ cases.array.items.len, cases.array.items.len });
}
