// Qwen2-faithful byte-level BPE tokenizer.
//
// Reproduces the HuggingFace `Qwen2Tokenizer` (fast variant) tokenization
// pipeline:
//   1. NFC normalization (skipped for now; verified unnecessary on the
//      cosyvoice2 fixture corpus, which uses ASCII / CJK / a few Latin-1
//      precomposed forms).
//   2. Special-token aware splitting (the longest matching added-token
//      content is excised and emitted directly with its registered id;
//      surrounding text re-enters the pipeline).
//   3. Pre-tokenizer regex (hand-rolled state machine equivalent to:
//        (?i:'s|'t|'re|'ve|'m|'ll|'d) |
//        [^\r\n\p{L}\p{N}]?\p{L}+   |
//        \p{N}                       |
//         ?[^\s\p{L}\p{N}]+[\r\n]*   |
//        \s*[\r\n]+                  |
//        \s+(?!\S)                   |
//        \s+
//      ).
//   4. GPT-2 byte-level mapping (every UTF-8 byte -> a printable Unicode
//      codepoint per `bytes_to_unicode`).
//   5. Standard greedy BPE merging by lowest priority rank (rank == line
//      number in `merges.txt`).
//
// The vocab and merge tables are owned by the caller (Dart) and passed in
// as raw C strings so we can map them into Zig-owned StringHashMaps.

const std = @import("std");
const unicode = @import("qwen2_unicode.zig");

pub const Error = error{
    OutOfMemory,
    InvalidInput,
    InvalidUtf8,
    UnknownToken,
};

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.OutOfMemory => "qwen2 bpe out of memory",
        error.InvalidInput => "qwen2 bpe received invalid input",
        error.InvalidUtf8 => "qwen2 bpe received invalid utf-8 text",
        error.UnknownToken => "qwen2 bpe encountered a token outside the vocab",
    };
}

/// Special-token entry registered with the tokenizer.
pub const SpecialToken = struct {
    text: []u8,
    id: i64,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,

    // vocab maps the byte-level-encoded merge string to its id; we only
    // need lookups in the encoder direction.
    vocab: std.StringHashMap(i64),
    // merges["a b"] -> rank (smaller is higher priority).
    merges: std.StringHashMap(usize),
    // Specials are kept in descending length order so that longest matches
    // win during the pre-pass split.
    specials: std.ArrayList(SpecialToken),

    pub fn init(allocator: std.mem.Allocator) Tokenizer {
        return .{
            .allocator = allocator,
            .vocab = std.StringHashMap(i64).init(allocator),
            .merges = std.StringHashMap(usize).init(allocator),
            .specials = .empty,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        var v_it = self.vocab.iterator();
        while (v_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.vocab.deinit();
        var m_it = self.merges.iterator();
        while (m_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.merges.deinit();
        for (self.specials.items) |sp| self.allocator.free(sp.text);
        self.specials.deinit(self.allocator);
    }

    pub fn addVocab(self: *Tokenizer, key: []const u8, id: i64) Error!void {
        const copy = self.allocator.dupe(u8, key) catch return error.OutOfMemory;
        errdefer self.allocator.free(copy);
        try self.vocab.put(copy, id);
    }

    pub fn addMerge(self: *Tokenizer, key: []const u8, rank: usize) Error!void {
        const copy = self.allocator.dupe(u8, key) catch return error.OutOfMemory;
        errdefer self.allocator.free(copy);
        try self.merges.put(copy, rank);
    }

    pub fn addSpecial(self: *Tokenizer, text: []const u8, id: i64) Error!void {
        const copy = self.allocator.dupe(u8, text) catch return error.OutOfMemory;
        errdefer self.allocator.free(copy);
        // Insert keeping the list sorted by descending length.
        var idx: usize = 0;
        while (idx < self.specials.items.len) : (idx += 1) {
            if (self.specials.items[idx].text.len < copy.len) break;
        }
        self.specials.insert(self.allocator, idx, .{ .text = copy, .id = id }) catch return error.OutOfMemory;
    }

    /// Encode `text` and append produced ids to `out`.  The output array
    /// list owns its memory (caller-supplied allocator).
    pub fn encode(
        self: *Tokenizer,
        text: []const u8,
        out: *std.ArrayList(i64),
    ) Error!void {
        try self.encodeSegmentSpecialAware(text, out);
    }

    fn encodeSegmentSpecialAware(
        self: *Tokenizer,
        text: []const u8,
        out: *std.ArrayList(i64),
    ) Error!void {
        var cursor: usize = 0;
        while (cursor < text.len) {
            const match = self.findSpecial(text, cursor);
            if (match) |m| {
                if (m.start > cursor) {
                    try self.encodeNormal(text[cursor..m.start], out);
                }
                out.append(self.allocator, m.id) catch return error.OutOfMemory;
                cursor = m.end;
            } else {
                try self.encodeNormal(text[cursor..], out);
                cursor = text.len;
            }
        }
    }

    const SpecialMatch = struct { start: usize, end: usize, id: i64 };

    fn findSpecial(self: *const Tokenizer, text: []const u8, from: usize) ?SpecialMatch {
        // Scan forward for the leftmost occurrence of any special token;
        // ties broken by longest-match (insertion order keeps long-first).
        var best: ?SpecialMatch = null;
        for (self.specials.items) |sp| {
            const idx = std.mem.indexOfPos(u8, text, from, sp.text) orelse continue;
            if (best) |b| {
                if (idx < b.start or (idx == b.start and sp.text.len > (b.end - b.start))) {
                    best = .{ .start = idx, .end = idx + sp.text.len, .id = sp.id };
                }
            } else {
                best = .{ .start = idx, .end = idx + sp.text.len, .id = sp.id };
            }
        }
        return best;
    }

    fn encodeNormal(
        self: *Tokenizer,
        text: []const u8,
        out: *std.ArrayList(i64),
    ) Error!void {
        if (text.len == 0) return;
        var pre = PreTokenIterator.init(text);
        while (try pre.next()) |segment| {
            try self.encodePreToken(segment, out);
        }
    }

    fn encodePreToken(
        self: *Tokenizer,
        segment: []const u8,
        out: *std.ArrayList(i64),
    ) Error!void {
        // 1. byte-level encode (each UTF-8 byte -> codepoint via bytes_to_unicode).
        // Worst case: 2 bytes per input byte (every codepoint <= 0x1FF -> 2-byte UTF-8).
        var encoded: std.ArrayList(u8) = .empty;
        defer encoded.deinit(self.allocator);
        encoded.ensureTotalCapacity(self.allocator, segment.len * 2 + 4) catch return error.OutOfMemory;
        for (segment) |b| {
            const cp = byte_encoder[b];
            var buf: [4]u8 = undefined;
            const written = std.unicode.utf8Encode(@intCast(cp), buf[0..]) catch unreachable;
            encoded.appendSlice(self.allocator, buf[0..written]) catch return error.OutOfMemory;
        }

        // 2. Build initial piece list — one piece per byte-level-encoded
        // *codepoint* (each token in the vocab is composed of these
        // GPT-2-style printable characters).
        var pieces: std.ArrayList(Piece) = .empty;
        defer pieces.deinit(self.allocator);
        var i: usize = 0;
        while (i < encoded.items.len) {
            const len = std.unicode.utf8ByteSequenceLength(encoded.items[i]) catch return error.InvalidUtf8;
            if (i + len > encoded.items.len) return error.InvalidUtf8;
            try pieces.append(self.allocator, .{
                .start = i,
                .len = len,
                .next = 0, // patched below
                .prev = 0, // patched below
            });
            i += len;
        }
        // Patch prev/next links to indices into `pieces`.
        var idx: usize = 0;
        while (idx < pieces.items.len) : (idx += 1) {
            pieces.items[idx].prev = if (idx == 0) std.math.maxInt(usize) else idx - 1;
            pieces.items[idx].next = if (idx + 1 == pieces.items.len) std.math.maxInt(usize) else idx + 1;
        }
        if (pieces.items.len == 0) return;

        // 3. Greedy BPE merge by lowest-rank pair.
        try self.mergeAll(encoded.items, &pieces);

        // 4. Map surviving pieces to ids.
        var cur: usize = 0;
        while (cur != std.math.maxInt(usize)) {
            const piece = pieces.items[cur];
            const slice = encoded.items[piece.start .. piece.start + piece.len];
            const id = self.vocab.get(slice) orelse {
                // Should never happen if vocab is complete (every single
                // byte-level char is in vocab).  Surface as error rather
                // than substituting an unknown.
                return error.UnknownToken;
            };
            out.append(self.allocator, id) catch return error.OutOfMemory;
            cur = piece.next;
        }
    }

    const Piece = struct {
        start: usize,
        len: usize,
        prev: usize,
        next: usize,
    };

    fn mergeAll(
        self: *Tokenizer,
        bytes: []const u8,
        pieces: *std.ArrayList(Piece),
    ) Error!void {
        var key_buf: std.ArrayList(u8) = .empty;
        defer key_buf.deinit(self.allocator);

        while (true) {
            // Walk the live linked list, tracking the lowest-rank pair.
            var best_rank: usize = std.math.maxInt(usize);
            var best_idx: usize = std.math.maxInt(usize);
            var cur: usize = 0;
            while (cur != std.math.maxInt(usize)) {
                const p = pieces.items[cur];
                if (p.next == std.math.maxInt(usize)) break;
                const q = pieces.items[p.next];
                key_buf.clearRetainingCapacity();
                try key_buf.appendSlice(self.allocator, bytes[p.start .. p.start + p.len]);
                try key_buf.append(self.allocator, ' ');
                try key_buf.appendSlice(self.allocator, bytes[q.start .. q.start + q.len]);
                if (self.merges.get(key_buf.items)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = cur;
                    }
                }
                cur = p.next;
            }
            if (best_idx == std.math.maxInt(usize)) return;

            // Merge pieces[best_idx] and its successor in place.
            const right_idx = pieces.items[best_idx].next;
            const right = pieces.items[right_idx];
            pieces.items[best_idx].len += right.len;
            pieces.items[best_idx].next = right.next;
            if (right.next != std.math.maxInt(usize)) {
                pieces.items[right.next].prev = best_idx;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Pre-tokenizer state machine.
// ---------------------------------------------------------------------------

const PreTokenIterator = struct {
    text: []const u8,
    cursor: usize,

    fn init(text: []const u8) PreTokenIterator {
        return .{ .text = text, .cursor = 0 };
    }

    fn next(self: *PreTokenIterator) Error!?[]const u8 {
        if (self.cursor >= self.text.len) return null;
        const start = self.cursor;
        const end = try self.matchOne();
        std.debug.assert(end > start);
        self.cursor = end;
        return self.text[start..end];
    }

    /// Returns the end offset of the next pre-token, applying the regex
    /// alternatives in order.  At least one byte is always consumed.
    fn matchOne(self: *PreTokenIterator) Error!usize {
        const text = self.text;
        const start = self.cursor;

        // Alt 1: contractions (case-insensitive).
        if (matchContraction(text, start)) |end| return end;

        // Alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if (try matchOptSymThenLetters(text, start)) |end| return end;

        // Alt 3: \p{N}  (single codepoint)
        {
            const cp = try peekCodepoint(text, start);
            if (unicode.isNumber(cp.value)) {
                return start + cp.len;
            }
        }

        // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (try matchOptSpaceSymbols(text, start)) |end| return end;

        // Alt 5: \s*[\r\n]+
        if (try matchSpacesThenNewlines(text, start)) |end| return end;

        // Alt 6: \s+(?!\S)
        if (try matchTrailingWhitespace(text, start)) |end| return end;

        // Alt 7: \s+
        if (try matchWhitespaceRun(text, start)) |end| return end;

        // Should not happen: regex is total, but if a stray byte slips
        // through, advance by one byte to avoid an infinite loop.
        return start + 1;
    }
};

const CP = struct { value: u32, len: usize };

fn peekCodepoint(text: []const u8, idx: usize) Error!CP {
    if (idx >= text.len) return error.InvalidInput;
    const len = std.unicode.utf8ByteSequenceLength(text[idx]) catch return error.InvalidUtf8;
    if (idx + len > text.len) return error.InvalidUtf8;
    const value = std.unicode.utf8Decode(text[idx .. idx + len]) catch return error.InvalidUtf8;
    return .{ .value = value, .len = len };
}

fn isAsciiAlpha(cp: u32) bool {
    return (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z');
}
fn asciiLower(cp: u32) u32 {
    return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
}

fn matchContraction(text: []const u8, start: usize) ?usize {
    if (start >= text.len) return null;
    if (text[start] != '\'') return null;
    const remaining = text[start + 1 ..];
    const Variant = struct { lower: []const u8 };
    const variants = [_]Variant{
        .{ .lower = "re" },
        .{ .lower = "ve" },
        .{ .lower = "ll" },
        .{ .lower = "s" },
        .{ .lower = "t" },
        .{ .lower = "m" },
        .{ .lower = "d" },
    };
    for (variants) |v| {
        if (remaining.len < v.lower.len) continue;
        var ok = true;
        var k: usize = 0;
        while (k < v.lower.len) : (k += 1) {
            const ch = remaining[k];
            if (!(ch >= 'A' and ch <= 'Z') and !(ch >= 'a' and ch <= 'z')) {
                ok = false;
                break;
            }
            const lower = if (ch >= 'A' and ch <= 'Z') ch + 32 else ch;
            if (lower != v.lower[k]) {
                ok = false;
                break;
            }
        }
        if (ok) return start + 1 + v.lower.len;
    }
    return null;
}

fn matchOptSymThenLetters(text: []const u8, start: usize) Error!?usize {
    var pos = start;
    // Optional [^\r\n\p{L}\p{N}].
    if (pos < text.len) {
        const cp = try peekCodepoint(text, pos);
        const c = cp.value;
        if (c != '\r' and c != '\n' and !unicode.isLetterOrNumber(c)) {
            // Need at least one letter to follow.
            const after = pos + cp.len;
            if (after < text.len) {
                const cp2 = try peekCodepoint(text, after);
                if (unicode.isLetter(cp2.value)) {
                    pos = after;
                } else {
                    // The optional sym only matches when followed by letters.
                    // Skip the optional and try with no prefix.
                }
            }
        }
    }
    // \p{L}+
    var consumed_letter = false;
    while (pos < text.len) {
        const cp = try peekCodepoint(text, pos);
        if (!unicode.isLetter(cp.value)) break;
        consumed_letter = true;
        pos += cp.len;
    }
    if (!consumed_letter) return null;
    return pos;
}

fn matchOptSpaceSymbols(text: []const u8, start: usize) Error!?usize {
    var pos = start;
    // optional leading single space (the regex has " ?" with a literal
    // space, not \s).
    if (pos < text.len and text[pos] == ' ') {
        pos += 1;
    }
    // [^\s\p{L}\p{N}]+
    var sym_count: usize = 0;
    while (pos < text.len) {
        const cp = try peekCodepoint(text, pos);
        if (unicode.isWhitespace(cp.value) or unicode.isLetterOrNumber(cp.value)) break;
        sym_count += 1;
        pos += cp.len;
    }
    if (sym_count == 0) {
        // We may have consumed a leading space without producing a symbol;
        // back off and let alts 5/6/7 handle the whitespace.
        return null;
    }
    // [\r\n]*
    while (pos < text.len and (text[pos] == '\r' or text[pos] == '\n')) {
        pos += 1;
    }
    return pos;
}

fn matchSpacesThenNewlines(text: []const u8, start: usize) Error!?usize {
    var pos = start;
    var saw_newline = false;
    var nl_seen = false;
    // \s* ... [\r\n]+  — the prefix \s* may include the newline itself,
    // so we scan whitespace and require at least one [\r\n] in the run.
    while (pos < text.len) {
        const cp = try peekCodepoint(text, pos);
        if (!unicode.isWhitespace(cp.value)) break;
        if (cp.value == '\r' or cp.value == '\n') {
            saw_newline = true;
        } else if (saw_newline) {
            // Once newlines have started, only further \r\n consume.
            break;
        }
        pos += cp.len;
        if (cp.value == '\r' or cp.value == '\n') nl_seen = true;
    }
    if (!nl_seen) return null;
    // After the trailing newline run, also consume any further \r\n.
    while (pos < text.len and (text[pos] == '\r' or text[pos] == '\n')) {
        pos += 1;
    }
    return pos;
}

fn matchTrailingWhitespace(text: []const u8, start: usize) Error!?usize {
    // Regex: \s+(?!\S)
    //
    // The greedy `\s+` runs to end-of-whitespace; the `(?!\S)` lookahead
    // then requires the next position to NOT be a non-space character —
    // satisfied either by EOF or by another whitespace character.  Since
    // the greedy match consumed all whitespace, the only way `(?!\S)`
    // succeeds is when EOF follows OR the regex backtracks the `+` so
    // that another whitespace character lies at the lookahead position.
    //
    // Concretely, for "   leading" (three spaces + letters), the engine
    // backtracks to consume only the first two spaces; the third space
    // satisfies (?!\S), and the next pre-token starts at the third space.
    //
    // Implementation: scan forward to find the end of the whitespace
    // run.  Then accept the longest prefix whose successor is whitespace
    // or EOF, i.e. (run_end - 1) if at least one whitespace char follows
    // the prefix, or run_end if run_end == text.len.
    var run_end = start;
    while (run_end < text.len) {
        const cp = peekCodepoint(text, run_end) catch break;
        if (!unicode.isWhitespace(cp.value)) break;
        run_end += cp.len;
    }
    if (run_end == start) return null;
    if (run_end == text.len) return run_end; // ran to EOF; whole run matches.
    // Otherwise, drop the last whitespace codepoint so (?!\S) holds.
    // Find the start of the last codepoint within [start, run_end).
    var last_cp_start = run_end;
    while (last_cp_start > start) {
        last_cp_start -= 1;
        // Codepoint start byte: not 10xxxxxx.
        if ((text[last_cp_start] & 0xC0) != 0x80) break;
    }
    if (last_cp_start == start) return null; // only one whitespace, can't satisfy
    return last_cp_start;
}

fn matchWhitespaceRun(text: []const u8, start: usize) Error!?usize {
    var pos = start;
    var count: usize = 0;
    while (pos < text.len) {
        const cp = try peekCodepoint(text, pos);
        if (!unicode.isWhitespace(cp.value)) break;
        count += 1;
        pos += cp.len;
    }
    if (count == 0) return null;
    return pos;
}

// ---------------------------------------------------------------------------
// GPT-2 byte_to_unicode table.
// ---------------------------------------------------------------------------

/// Maps a raw byte (0..255) to its byte-level Unicode codepoint per the
/// GPT-2 `bytes_to_unicode` spec.  Matches HuggingFace's `ByteLevel` impl.
pub const byte_encoder = blk: {
    @setEvalBranchQuota(20_000);
    var table: [256]u32 = undefined;
    var visible_count: u32 = 0;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        const visible = (i >= '!' and i <= '~') or
            (i >= 0xA1 and i <= 0xAC) or
            (i >= 0xAE and i <= 0xFF);
        if (visible) {
            table[i] = i;
        } else {
            table[i] = 0x100 + visible_count;
            visible_count += 1;
        }
    }
    break :blk table;
};

// ---------------------------------------------------------------------------
// Public C-friendly wrappers.
// ---------------------------------------------------------------------------

pub fn create(allocator: std.mem.Allocator) Error!*Tokenizer {
    const tk = allocator.create(Tokenizer) catch return error.OutOfMemory;
    tk.* = Tokenizer.init(allocator);
    return tk;
}

pub fn destroy(tk: *Tokenizer) void {
    const allocator = tk.allocator;
    tk.deinit();
    allocator.destroy(tk);
}

pub fn fromHandle(handle: ?*anyopaque) Error!*Tokenizer {
    const raw = handle orelse return error.InvalidInput;
    return @ptrCast(@alignCast(raw));
}

// ---------------------------------------------------------------------------
// Tests (run with: zig build test --summary all)
// ---------------------------------------------------------------------------

test "byte encoder maps space to G-with-stroke" {
    try std.testing.expectEqual(@as(u32, 0x120), byte_encoder[' ']);
    try std.testing.expectEqual(@as(u32, '!'), byte_encoder['!']);
    try std.testing.expectEqual(@as(u32, 'A'), byte_encoder['A']);
}

test "pre-tokenizer splits 'Hello world!'" {
    var it = PreTokenIterator.init("Hello world!");
    const a = (try it.next()).?;
    try std.testing.expectEqualStrings("Hello", a);
    const b = (try it.next()).?;
    try std.testing.expectEqualStrings(" world", b);
    const c = (try it.next()).?;
    try std.testing.expectEqualStrings("!", c);
    try std.testing.expect((try it.next()) == null);
}

test "pre-tokenizer splits digits one-by-one" {
    var it = PreTokenIterator.init("100");
    try std.testing.expectEqualStrings("1", (try it.next()).?);
    try std.testing.expectEqualStrings("0", (try it.next()).?);
    try std.testing.expectEqualStrings("0", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "pre-tokenizer handles contractions" {
    var it = PreTokenIterator.init("It's");
    try std.testing.expectEqualStrings("It", (try it.next()).?);
    try std.testing.expectEqualStrings("'s", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "pre-tokenizer handles CJK as letters" {
    var it = PreTokenIterator.init("你好");
    try std.testing.expectEqualStrings("你好", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}
