const std = @import("std");

pub const Error = error{
    InvalidHandle,
    InvalidInput,
    InvalidUtf8,
    LengthOverflow,
    OutOfMemory,
};

const marker: u21 = 0x2581;

const Piece = struct {
    text: []u8,
    start: i32,
    end: i32,
};

const Rune = struct {
    codepoint: u21,
    len: usize,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(i64),
    merges: std.StringHashMap(usize),
    bos_id: i64,
    eos_id: i64,
    pad_id: i64,
    unk_id: i64,

    pub fn init(
        allocator: std.mem.Allocator,
        bos_id: i64,
        eos_id: i64,
        pad_id: i64,
        unk_id: i64,
    ) Tokenizer {
        return .{
            .allocator = allocator,
            .vocab = std.StringHashMap(i64).init(allocator),
            .merges = std.StringHashMap(usize).init(allocator),
            .bos_id = bos_id,
            .eos_id = eos_id,
            .pad_id = pad_id,
            .unk_id = unk_id,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        freeMapKeys(i64, self.allocator, &self.vocab);
        self.vocab.deinit();
        freeMapKeys(usize, self.allocator, &self.merges);
        self.merges.deinit();
    }

    pub fn addVocab(self: *Tokenizer, key: []const u8, id: i64) Error!void {
        if (key.len == 0) return error.InvalidInput;
        const copy = self.allocator.dupe(u8, key) catch return error.OutOfMemory;
        errdefer self.allocator.free(copy);
        self.vocab.put(copy, id) catch return error.OutOfMemory;
    }

    pub fn addMerge(self: *Tokenizer, key: []const u8, rank: usize) Error!void {
        if (key.len == 0) return error.InvalidInput;
        const copy = self.allocator.dupe(u8, key) catch return error.OutOfMemory;
        errdefer self.allocator.free(copy);
        self.merges.put(copy, rank) catch return error.OutOfMemory;
    }

    pub fn encode(
        self: *Tokenizer,
        text: []const u8,
        ids: []i64,
        starts: []i32,
        ends: []i32,
    ) Error!usize {
        if (ids.len == 0 or starts.len != ids.len or ends.len != ids.len) {
            return error.InvalidInput;
        }
        @memset(ids, self.pad_id);
        @memset(starts, 0);
        @memset(ends, 0);

        if (ids.len == 1) {
            ids[0] = self.eos_id;
            return 1;
        }

        var pieces = try self.initialPieces(text);
        defer pieces.deinit(self.allocator);
        defer freePieces(self.allocator, pieces.items);

        try self.applyMerges(&pieces);

        ids[0] = self.bos_id;
        var out_index: usize = 1;
        for (pieces.items) |piece| {
            if (out_index >= ids.len - 1) {
                break;
            }
            ids[out_index] = self.tokenId(piece.text);
            starts[out_index] = piece.start;
            ends[out_index] = piece.end;
            out_index += 1;
        }
        ids[out_index] = self.eos_id;
        return ids.len;
    }

    pub fn fill(
        self: *Tokenizer,
        text: []const u8,
        ids: []i64,
        mask: []i64,
        starts: []i32,
        ends: []i32,
    ) Error!usize {
        if (mask.len != ids.len) {
            return error.InvalidInput;
        }
        const count = try self.encode(text, ids, starts, ends);
        for (ids, mask) |id, *item| {
            item.* = if (id == self.pad_id) 0 else 1;
        }
        return count;
    }

    fn initialPieces(self: *Tokenizer, text: []const u8) Error!std.ArrayList(Piece) {
        var pieces: std.ArrayList(Piece) = .empty;
        errdefer pieces.deinit(self.allocator);
        errdefer freePieces(self.allocator, pieces.items);

        try appendRune(self.allocator, &pieces, marker, 0, 0);
        const char_count = try countRunes(text);
        var index: usize = 0;
        var original_index: usize = 0;
        while (index < text.len) {
            const rune = try nextRune(text, index);
            index += rune.len;
            if (rune.codepoint == ' ') {
                const pos = try int32Size(original_index);
                try appendRune(self.allocator, &pieces, marker, pos, pos);
                original_index += 1;
                continue;
            }
            if (rune.codepoint == marker) {
                const pos = try int32Size(original_index);
                try appendRune(self.allocator, &pieces, marker, pos, pos);
                continue;
            }
            const start = try int32Size(original_index);
            const next = @min(original_index + 1, char_count);
            const end = try int32Size(next);
            try appendRune(self.allocator, &pieces, rune.codepoint, start, end);
            original_index += 1;
        }
        return pieces;
    }

    fn applyMerges(self: *Tokenizer, pieces: *std.ArrayList(Piece)) Error!void {
        while (pieces.items.len > 1) {
            var best_rank: usize = std.math.maxInt(usize);
            var best_index: ?usize = null;
            var index: usize = 0;
            while (index + 1 < pieces.items.len) : (index += 1) {
                const rank = try self.mergeRank(
                    pieces.items[index].text,
                    pieces.items[index + 1].text,
                ) orelse continue;
                if (rank < best_rank) {
                    best_rank = rank;
                    best_index = index;
                }
            }
            const merge_index = best_index orelse break;
            try mergeAt(self.allocator, pieces, merge_index);
        }
    }

    fn mergeRank(self: *Tokenizer, left: []const u8, right: []const u8) Error!?usize {
        const key_len = std.math.add(usize, left.len, right.len) catch return error.LengthOverflow;
        const total = std.math.add(usize, key_len, 1) catch return error.LengthOverflow;
        const key = self.allocator.alloc(u8, total) catch return error.OutOfMemory;
        defer self.allocator.free(key);
        @memcpy(key[0..left.len], left);
        key[left.len] = ' ';
        @memcpy(key[left.len + 1 ..], right);
        return self.merges.get(key);
    }

    fn tokenId(self: *Tokenizer, piece: []const u8) i64 {
        if (self.vocab.get(piece)) |id| {
            return id;
        }
        if (piece.len == 1) {
            var key = [_]u8{ '<', '0', 'x', '0', '0', '>' };
            const hex = "0123456789ABCDEF";
            key[3] = hex[piece[0] >> 4];
            key[4] = hex[piece[0] & 0x0f];
            if (self.vocab.get(key[0..])) |id| {
                return id;
            }
        }
        return self.unk_id;
    }
};

pub fn create(
    allocator: std.mem.Allocator,
    bos_id: i64,
    eos_id: i64,
    pad_id: i64,
    unk_id: i64,
) Error!*Tokenizer {
    const tokenizer = allocator.create(Tokenizer) catch return error.OutOfMemory;
    tokenizer.* = Tokenizer.init(allocator, bos_id, eos_id, pad_id, unk_id);
    return tokenizer;
}

pub fn destroy(tokenizer: *Tokenizer) void {
    const allocator = tokenizer.allocator;
    tokenizer.deinit();
    allocator.destroy(tokenizer);
}

pub fn fromHandle(handle: ?*anyopaque) Error!*Tokenizer {
    const raw = handle orelse return error.InvalidHandle;
    return @ptrCast(@alignCast(raw));
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidHandle => "Zig BPE tokenizer received an invalid handle.",
        error.InvalidInput => "Zig BPE tokenizer received invalid input.",
        error.InvalidUtf8 => "Zig BPE tokenizer received invalid UTF-8 text.",
        error.LengthOverflow => "Zig BPE tokenizer output length overflowed.",
        error.OutOfMemory => "Zig BPE tokenizer ran out of memory.",
    };
}

fn freeMapKeys(comptime V: type, allocator: std.mem.Allocator, map: *std.StringHashMap(V)) void {
    var keys = map.keyIterator();
    while (keys.next()) |key| {
        allocator.free(key.*);
    }
}

fn appendRune(
    allocator: std.mem.Allocator,
    pieces: *std.ArrayList(Piece),
    codepoint: u21,
    start: i32,
    end: i32,
) Error!void {
    var bytes: [4]u8 = undefined;
    const len = writeUtf8(&bytes, codepoint);
    const text = allocator.alloc(u8, len) catch return error.OutOfMemory;
    errdefer allocator.free(text);
    @memcpy(text, bytes[0..len]);
    pieces.append(allocator, .{ .text = text, .start = start, .end = end }) catch return error.OutOfMemory;
}

fn mergeAt(allocator: std.mem.Allocator, pieces: *std.ArrayList(Piece), index: usize) Error!void {
    const left = pieces.items[index];
    const right = pieces.items[index + 1];
    const len = std.math.add(usize, left.text.len, right.text.len) catch return error.LengthOverflow;
    const text = allocator.alloc(u8, len) catch return error.OutOfMemory;
    errdefer allocator.free(text);
    @memcpy(text[0..left.text.len], left.text);
    @memcpy(text[left.text.len..], right.text);
    allocator.free(left.text);
    allocator.free(right.text);
    pieces.items[index] = .{
        .text = text,
        .start = left.start,
        .end = @max(left.end, right.end),
    };
    _ = pieces.orderedRemove(index + 1);
}

fn freePieces(allocator: std.mem.Allocator, pieces: []Piece) void {
    for (pieces) |piece| {
        allocator.free(piece.text);
    }
}

fn countRunes(input: []const u8) Error!usize {
    var index: usize = 0;
    var count: usize = 0;
    while (index < input.len) {
        const rune = try nextRune(input, index);
        index += rune.len;
        count += 1;
    }
    return count;
}

fn nextRune(input: []const u8, index: usize) Error!Rune {
    const first = input[index];
    if (first < 0x80) {
        return .{ .codepoint = @intCast(first), .len = 1 };
    }
    if ((first & 0xe0) == 0xc0) {
        if (index + 1 >= input.len or !isCont(input[index + 1])) return error.InvalidUtf8;
        const value: u32 = (@as(u32, first & 0x1f) << 6) |
            @as(u32, input[index + 1] & 0x3f);
        if (value < 0x80) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .len = 2 };
    }
    if ((first & 0xf0) == 0xe0) {
        if (index + 2 >= input.len or !isCont(input[index + 1]) or !isCont(input[index + 2])) {
            return error.InvalidUtf8;
        }
        const value: u32 = (@as(u32, first & 0x0f) << 12) |
            (@as(u32, input[index + 1] & 0x3f) << 6) |
            @as(u32, input[index + 2] & 0x3f);
        if (value < 0x800 or (value >= 0xd800 and value <= 0xdfff)) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .len = 3 };
    }
    if ((first & 0xf8) == 0xf0) {
        if (index + 3 >= input.len or
            !isCont(input[index + 1]) or
            !isCont(input[index + 2]) or
            !isCont(input[index + 3]))
        {
            return error.InvalidUtf8;
        }
        const value: u32 = (@as(u32, first & 0x07) << 18) |
            (@as(u32, input[index + 1] & 0x3f) << 12) |
            (@as(u32, input[index + 2] & 0x3f) << 6) |
            @as(u32, input[index + 3] & 0x3f);
        if (value < 0x10000 or value > 0x10ffff) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .len = 4 };
    }
    return error.InvalidUtf8;
}

fn isCont(byte: u8) bool {
    return (byte & 0xc0) == 0x80;
}

fn writeUtf8(out: *[4]u8, codepoint: u21) u8 {
    const value: u32 = @intCast(codepoint);
    if (value < 0x80) {
        out[0] = @intCast(value);
        return 1;
    }
    if (value < 0x800) {
        out[0] = @intCast(0xc0 | (value >> 6));
        out[1] = @intCast(0x80 | (value & 0x3f));
        return 2;
    }
    if (value < 0x10000) {
        out[0] = @intCast(0xe0 | (value >> 12));
        out[1] = @intCast(0x80 | ((value >> 6) & 0x3f));
        out[2] = @intCast(0x80 | (value & 0x3f));
        return 3;
    }
    out[0] = @intCast(0xf0 | (value >> 18));
    out[1] = @intCast(0x80 | ((value >> 12) & 0x3f));
    out[2] = @intCast(0x80 | ((value >> 6) & 0x3f));
    out[3] = @intCast(0x80 | (value & 0x3f));
    return 4;
}

fn int32Size(value: usize) Error!i32 {
    if (value > std.math.maxInt(i32)) {
        return error.LengthOverflow;
    }
    return @intCast(value);
}

test "BPE tokenizer encodes pieces and offsets" {
    var tokenizer = Tokenizer.init(std.testing.allocator, 2, 1, 0, 3);
    defer tokenizer.deinit();
    try tokenizer.addVocab("<pad>", 0);
    try tokenizer.addVocab("<eos>", 1);
    try tokenizer.addVocab("<bos>", 2);
    try tokenizer.addVocab("<unk>", 3);
    try tokenizer.addVocab("▁", 4);
    try tokenizer.addVocab("L", 5);
    try tokenizer.addVocab("e", 6);
    try tokenizer.addVocab("a", 7);
    try tokenizer.addVocab("d", 8);

    var ids = [_]i64{0} ** 8;
    var starts = [_]i32{0} ** 8;
    var ends = [_]i32{0} ** 8;
    const count = try tokenizer.encode("Lead", &ids, &starts, &ends);

    try std.testing.expectEqual(@as(usize, 8), count);
    try std.testing.expectEqualSlices(i64, &.{ 2, 4, 5, 6, 7, 8, 1, 0 }, &ids);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 2, 3, 0, 0 }, &starts);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 1, 2, 3, 4, 0, 0 }, &ends);
}

test "BPE tokenizer applies lowest-rank merge" {
    var tokenizer = Tokenizer.init(std.testing.allocator, 2, 1, 0, 3);
    defer tokenizer.deinit();
    try tokenizer.addVocab("<pad>", 0);
    try tokenizer.addVocab("<eos>", 1);
    try tokenizer.addVocab("<bos>", 2);
    try tokenizer.addVocab("<unk>", 3);
    try tokenizer.addVocab("▁L", 9);
    try tokenizer.addVocab("e", 6);
    try tokenizer.addMerge("▁ L", 0);

    var ids = [_]i64{0} ** 5;
    var starts = [_]i32{0} ** 5;
    var ends = [_]i32{0} ** 5;
    _ = try tokenizer.encode("Le", &ids, &starts, &ends);

    try std.testing.expectEqualSlices(i64, &.{ 2, 9, 6, 1, 0 }, &ids);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 1, 0, 0 }, &starts);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 0, 0 }, &ends);
}

test "BPE tokenizer fills attention mask" {
    var tokenizer = Tokenizer.init(std.testing.allocator, 2, 1, 0, 3);
    defer tokenizer.deinit();
    try tokenizer.addVocab("<pad>", 0);
    try tokenizer.addVocab("<eos>", 1);
    try tokenizer.addVocab("<bos>", 2);
    try tokenizer.addVocab("<unk>", 3);
    try tokenizer.addVocab("▁", 4);
    try tokenizer.addVocab("A", 5);

    var ids = [_]i64{9} ** 6;
    var mask = [_]i64{8} ** 6;
    var starts = [_]i32{0} ** 6;
    var ends = [_]i32{0} ** 6;
    _ = try tokenizer.fill("A", &ids, &mask, &starts, &ends);

    try std.testing.expectEqualSlices(i64, &.{ 2, 4, 5, 1, 0, 0 }, &ids);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1, 0, 0 }, &mask);
}
