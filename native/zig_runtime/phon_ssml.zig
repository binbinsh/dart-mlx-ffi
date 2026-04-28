const std = @import("std");
const phoneme = @import("phoneme.zig");

pub const Error = phoneme.Error || error{
    OutOfMemory,
};

pub const Kind = enum(i32) {
    plain = 1,
    explicit = 2,
    pinyin = 3,
};

pub const Chunk = struct {
    kind: Kind,
    text: []u8,
    space_after: bool,
};

pub const List = struct {
    chunks: []Chunk = &.{},

    pub fn deinit(self: *List, allocator: std.mem.Allocator) void {
        for (self.chunks) |chunk| {
            allocator.free(chunk.text);
        }
        if (self.chunks.len > 0) allocator.free(self.chunks);
        self.* = .{};
    }
};

const Tag = struct {
    end: usize,
    ph: []const u8,
};

pub fn parse(allocator: std.mem.Allocator, input: []const u8) Error!List {
    var chunks: std.ArrayList(Chunk) = .empty;
    errdefer {
        for (chunks.items) |chunk| allocator.free(chunk.text);
        chunks.deinit(allocator);
    }

    var found = false;
    var cursor: usize = 0;
    var scan: usize = 0;
    while (std.mem.indexOfScalarPos(u8, input, scan, '<')) |tag_start| {
        const tag = phonemeTag(input, tag_start) orelse {
            scan = tag_start + 1;
            continue;
        };
        const close = findClose(input, tag.end) orelse {
            scan = tag_start + 1;
            continue;
        };
        found = true;
        try addPlain(allocator, &chunks, input[cursor..tag_start], true);
        try addPhoneme(allocator, &chunks, tag.ph);
        cursor = close;
        scan = close;
    }

    if (!found) {
        try addPlain(allocator, &chunks, input, false);
    } else {
        try addPlain(allocator, &chunks, input[cursor..], false);
    }

    const owned = chunks.toOwnedSlice(allocator) catch return error.OutOfMemory;
    chunks = .empty;
    return .{ .chunks = owned };
}

fn addPlain(
    allocator: std.mem.Allocator,
    chunks: *std.ArrayList(Chunk),
    fragment: []const u8,
    space_after: bool,
) Error!void {
    var text = try phoneme.plainSsml(allocator, fragment);
    errdefer allocator.free(text);
    if (trim(text).len == 0) {
        allocator.free(text);
        return;
    }
    chunks.append(allocator, .{
        .kind = .plain,
        .text = text,
        .space_after = space_after,
    }) catch return error.OutOfMemory;
    text = &.{};
}

fn addPhoneme(
    allocator: std.mem.Allocator,
    chunks: *std.ArrayList(Chunk),
    value: []const u8,
) Error!void {
    const decoded = try phoneme.plainSsml(allocator, value);
    errdefer allocator.free(decoded);
    const trimmed = trim(decoded);
    if (trimmed.len == 0) {
        allocator.free(decoded);
        return;
    }
    const text = allocator.dupe(u8, trimmed) catch return error.OutOfMemory;
    allocator.free(decoded);
    errdefer allocator.free(text);
    chunks.append(allocator, .{
        .kind = if (phoneme.looksPinyin(text)) .pinyin else .explicit,
        .text = text,
        .space_after = true,
    }) catch return error.OutOfMemory;
}

fn phonemeTag(input: []const u8, index: usize) ?Tag {
    if (index + 8 > input.len or input[index] != '<') return null;
    if (!eqlIgnore(input[index + 1 .. index + 8], "phoneme")) return null;
    const after_name = index + 8;
    if (after_name < input.len and isWord(input[after_name])) return null;
    const end = tagEnd(input, index) orelse return null;
    const ph = phAttr(input[after_name..end]) orelse return null;
    return .{ .end = end + 1, .ph = ph };
}

fn phAttr(attrs: []const u8) ?[]const u8 {
    var index: usize = 0;
    while (index + 2 <= attrs.len) : (index += 1) {
        if (!eqlIgnore(attrs[index .. index + 2], "ph")) continue;
        if (index > 0 and isWord(attrs[index - 1])) continue;
        var cursor = index + 2;
        if (cursor < attrs.len and isWord(attrs[cursor])) continue;
        while (cursor < attrs.len and isAsciiSpace(attrs[cursor])) cursor += 1;
        if (cursor >= attrs.len or attrs[cursor] != '=') continue;
        cursor += 1;
        while (cursor < attrs.len and isAsciiSpace(attrs[cursor])) cursor += 1;
        if (cursor >= attrs.len or attrs[cursor] != '"') continue;
        const start = cursor + 1;
        const end = std.mem.indexOfScalarPos(u8, attrs, start, '"') orelse return null;
        return attrs[start..end];
    }
    return null;
}

fn findClose(input: []const u8, start: usize) ?usize {
    var index = start;
    while (std.mem.indexOfScalarPos(u8, input, index, '<')) |tag_start| {
        if (tag_start + 10 <= input.len and eqlIgnore(input[tag_start .. tag_start + 10], "</phoneme>")) {
            return tag_start + 10;
        }
        index = tag_start + 1;
    }
    return null;
}

fn tagEnd(input: []const u8, start: usize) ?usize {
    return std.mem.indexOfScalarPos(u8, input, start + 1, '>');
}

fn trim(value: []const u8) []const u8 {
    var start: usize = 0;
    var end = value.len;
    while (start < end and isAsciiSpace(value[start])) start += 1;
    while (end > start and isAsciiSpace(value[end - 1])) end -= 1;
    return value[start..end];
}

fn eqlIgnore(left: []const u8, right: []const u8) bool {
    if (left.len != right.len) return false;
    for (left, right) |a, b| {
        if (lower(a) != lower(b)) return false;
    }
    return true;
}

fn isWord(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

fn isAsciiSpace(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r';
}

fn lower(byte: u8) u8 {
    return if (byte >= 'A' and byte <= 'Z') byte + 32 else byte;
}

test "phoneme ssml parser returns plain and explicit chunks" {
    var list = try parse(
        std.testing.allocator,
        "<speak><phoneme ph=\"L_IY_D\">Lead</phoneme> us <sub alias=\"one\">1</sub>.</speak>",
    );
    defer list.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), list.chunks.len);
    try std.testing.expectEqual(Kind.explicit, list.chunks[0].kind);
    try std.testing.expectEqualStrings("L_IY_D", list.chunks[0].text);
    try std.testing.expect(list.chunks[0].space_after);
    try std.testing.expectEqual(Kind.plain, list.chunks[1].kind);
    try std.testing.expectEqualStrings(" us one.", list.chunks[1].text);
    try std.testing.expect(!list.chunks[1].space_after);
}

test "phoneme ssml parser classifies pinyin chunks" {
    var list = try parse(std.testing.allocator, "银<phoneme ph=\"hang2\">行</phoneme>");
    defer list.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), list.chunks.len);
    try std.testing.expectEqual(Kind.plain, list.chunks[0].kind);
    try std.testing.expectEqualStrings("银", list.chunks[0].text);
    try std.testing.expectEqual(Kind.pinyin, list.chunks[1].kind);
    try std.testing.expectEqualStrings("hang2", list.chunks[1].text);
}

test "phoneme ssml parser falls back to plain text without phoneme tags" {
    var list = try parse(std.testing.allocator, "<speak>A <sub alias=\"one\">1</sub></speak>");
    defer list.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), list.chunks.len);
    try std.testing.expectEqual(Kind.plain, list.chunks[0].kind);
    try std.testing.expectEqualStrings("A one", list.chunks[0].text);
    try std.testing.expect(!list.chunks[0].space_after);
}
