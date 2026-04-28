const std = @import("std");

pub const Error = error{
    InvalidHandle,
    InvalidInput,
    InvalidUtf8,
    LengthOverflow,
    OutOfMemory,
};

pub const Match = extern struct {
    start: i32,
    end: i32,
    index: i32,
    ids: [*c]const i32,
    id_count: isize,
};

const Surface = struct {
    text: []u8,
    utf16_len: i32,
};

const IdSet = struct {
    items: []i32,
};

const Candidate = struct {
    start: i32,
    end: i32,
    index: i32,
};

const Rune = struct {
    codepoint: u21,
    len: usize,
};

pub const Matcher = struct {
    allocator: std.mem.Allocator,
    homographs: []Surface,
    homograph_ids: []IdSet,
    polyphones: []Surface,
    polyphone_ids: []IdSet,

    pub fn init(
        allocator: std.mem.Allocator,
        homographs: []Surface,
        homograph_ids: []IdSet,
        polyphones: []Surface,
        polyphone_ids: []IdSet,
    ) Matcher {
        return .{
            .allocator = allocator,
            .homographs = homographs,
            .homograph_ids = homograph_ids,
            .polyphones = polyphones,
            .polyphone_ids = polyphone_ids,
        };
    }

    pub fn deinit(self: *Matcher) void {
        freeSurfaces(self.allocator, self.homographs);
        freeIdSets(self.allocator, self.homograph_ids);
        freeSurfaces(self.allocator, self.polyphones);
        freeIdSets(self.allocator, self.polyphone_ids);
    }

    pub fn idsFor(self: *const Matcher, homographs: bool, index: i32) []const i32 {
        if (index < 0) return &.{};
        const id_sets = if (homographs) self.homograph_ids else self.polyphone_ids;
        const idx: usize = @intCast(index);
        if (idx >= id_sets.len) return &.{};
        return id_sets[idx].items;
    }

    pub fn homographMatches(self: *const Matcher, text: []const u8) Error![]Match {
        var candidates: std.ArrayList(Candidate) = .empty;
        defer candidates.deinit(self.allocator);

        var byte_index: usize = 0;
        var unit_index: i32 = 0;
        while (byte_index < text.len) {
            const first = text[byte_index];
            if (!asciiLetter(first)) {
                const rune = try nextRune(text, byte_index);
                byte_index += rune.len;
                unit_index = try addUnits(unit_index, utf16RuneLen(rune.codepoint));
                continue;
            }

            const start_byte = byte_index;
            const start_unit = unit_index;
            while (byte_index < text.len and asciiLetter(text[byte_index])) : (byte_index += 1) {
                unit_index = try addUnits(unit_index, 1);
            }
            while (byte_index + 1 < text.len and
                (text[byte_index] == '\'' or text[byte_index] == '-') and
                asciiLetter(text[byte_index + 1]))
            {
                byte_index += 1;
                unit_index = try addUnits(unit_index, 1);
                while (byte_index < text.len and asciiLetter(text[byte_index])) : (byte_index += 1) {
                    unit_index = try addUnits(unit_index, 1);
                }
            }

            const word = text[start_byte..byte_index];
            if (self.findHomograph(word)) |surface_index| {
                candidates.append(self.allocator, .{
                    .start = start_unit,
                    .end = unit_index,
                    .index = @intCast(surface_index),
                }) catch return error.OutOfMemory;
            }
        }
        return select(self.allocator, candidates.items);
    }

    pub fn polyphoneMatches(self: *const Matcher, text: []const u8) Error![]Match {
        var candidates: std.ArrayList(Candidate) = .empty;
        defer candidates.deinit(self.allocator);
        for (self.polyphones, 0..) |surface, surface_index| {
            if (surface.text.len == 0) continue;
            var search_start: usize = 0;
            while (search_start < text.len) {
                const found = std.mem.indexOfPos(u8, text, search_start, surface.text) orelse break;
                const start = try utf16PrefixLen(text[0..found]);
                candidates.append(self.allocator, .{
                    .start = start,
                    .end = try addUnits(start, surface.utf16_len),
                    .index = @intCast(surface_index),
                }) catch return error.OutOfMemory;
                search_start = found + 1;
            }
        }
        return select(self.allocator, candidates.items);
    }

    fn findHomograph(self: *const Matcher, word: []const u8) ?usize {
        for (self.homographs, 0..) |surface, index| {
            if (asciiLowerEql(word, surface.text)) {
                return index;
            }
        }
        return null;
    }
};

pub fn create(
    allocator: std.mem.Allocator,
    homographs: []const []const u8,
    homograph_ids: []const []const i32,
    polyphones: []const []const u8,
    polyphone_ids: []const []const i32,
) Error!*Matcher {
    if (homographs.len != homograph_ids.len or polyphones.len != polyphone_ids.len) {
        return error.InvalidInput;
    }
    const homograph_items = try copySurfaces(allocator, homographs);
    errdefer freeSurfaces(allocator, homograph_items);
    const homograph_id_items = try copyIdSets(allocator, homograph_ids);
    errdefer freeIdSets(allocator, homograph_id_items);
    const polyphone_items = try copySurfaces(allocator, polyphones);
    errdefer freeSurfaces(allocator, polyphone_items);
    const polyphone_id_items = try copyIdSets(allocator, polyphone_ids);
    errdefer freeIdSets(allocator, polyphone_id_items);
    const matcher = allocator.create(Matcher) catch return error.OutOfMemory;
    matcher.* = Matcher.init(
        allocator,
        homograph_items,
        homograph_id_items,
        polyphone_items,
        polyphone_id_items,
    );
    return matcher;
}

pub fn destroy(matcher: *Matcher) void {
    const allocator = matcher.allocator;
    matcher.deinit();
    allocator.destroy(matcher);
}

pub fn fromHandle(handle: ?*anyopaque) Error!*Matcher {
    const raw = handle orelse return error.InvalidHandle;
    return @ptrCast(@alignCast(raw));
}

pub fn freeMatches(allocator: std.mem.Allocator, matches: [*c]Match, count: isize) void {
    if (matches == null or count <= 0) return;
    const len: usize = @intCast(count);
    const items: [*]Match = @ptrCast(matches);
    allocator.free(items[0..len]);
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidHandle => "Zig target matcher received an invalid handle.",
        error.InvalidInput => "Zig target matcher received invalid input.",
        error.InvalidUtf8 => "Zig target matcher received invalid UTF-8 text.",
        error.LengthOverflow => "Zig target matcher output length overflowed.",
        error.OutOfMemory => "Zig target matcher ran out of memory.",
    };
}

fn copySurfaces(allocator: std.mem.Allocator, values: []const []const u8) Error![]Surface {
    if (values.len == 0) return &.{};
    const out = allocator.alloc(Surface, values.len) catch return error.OutOfMemory;
    var produced: usize = 0;
    errdefer {
        var index: usize = 0;
        while (index < produced) : (index += 1) {
            allocator.free(out[index].text);
        }
        allocator.free(out);
    }
    for (values) |value| {
        const copy = allocator.dupe(u8, value) catch return error.OutOfMemory;
        out[produced] = .{
            .text = copy,
            .utf16_len = try utf16PrefixLen(value),
        };
        produced += 1;
    }
    return out;
}

fn freeSurfaces(allocator: std.mem.Allocator, values: []Surface) void {
    for (values) |surface| {
        allocator.free(surface.text);
    }
    if (values.len > 0) allocator.free(values);
}

fn copyIdSets(allocator: std.mem.Allocator, values: []const []const i32) Error![]IdSet {
    if (values.len == 0) return &.{};
    const out = allocator.alloc(IdSet, values.len) catch return error.OutOfMemory;
    var produced: usize = 0;
    errdefer {
        var index: usize = 0;
        while (index < produced) : (index += 1) {
            allocator.free(out[index].items);
        }
        allocator.free(out);
    }
    for (values) |value| {
        const copy = allocator.dupe(i32, value) catch return error.OutOfMemory;
        out[produced] = .{ .items = copy };
        produced += 1;
    }
    return out;
}

fn freeIdSets(allocator: std.mem.Allocator, values: []IdSet) void {
    for (values) |ids| {
        allocator.free(ids.items);
    }
    if (values.len > 0) allocator.free(values);
}

fn select(allocator: std.mem.Allocator, candidates: []Candidate) Error![]Match {
    if (candidates.len == 0) return &.{};
    std.mem.sort(Candidate, candidates, {}, candidateLess);
    var selected: std.ArrayList(Candidate) = .empty;
    defer selected.deinit(allocator);
    for (candidates) |candidate| {
        var overlaps = false;
        for (selected.items) |item| {
            if (!(candidate.end <= item.start or candidate.start >= item.end)) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) {
            selected.append(allocator, candidate) catch return error.OutOfMemory;
        }
    }
    std.mem.sort(Candidate, selected.items, {}, selectedLess);
    const out = allocator.alloc(Match, selected.items.len) catch return error.OutOfMemory;
    for (selected.items, 0..) |item, index| {
        out[index] = .{
            .start = item.start,
            .end = item.end,
            .index = item.index,
            .ids = null,
            .id_count = 0,
        };
    }
    return out;
}

fn candidateLess(_: void, left: Candidate, right: Candidate) bool {
    const left_len = left.end - left.start;
    const right_len = right.end - right.start;
    if (left_len != right_len) return left_len > right_len;
    if (left.start != right.start) return left.start < right.start;
    if (left.end != right.end) return left.end < right.end;
    return left.index < right.index;
}

fn selectedLess(_: void, left: Candidate, right: Candidate) bool {
    if (left.start != right.start) return left.start < right.start;
    if (left.end != right.end) return left.end < right.end;
    return left.index < right.index;
}

fn asciiLowerEql(left: []const u8, right: []const u8) bool {
    if (left.len != right.len) return false;
    for (left, right) |a, b| {
        const lower = if (a >= 'A' and a <= 'Z') a + 32 else a;
        if (lower != b) return false;
    }
    return true;
}

fn asciiLetter(byte: u8) bool {
    return (byte >= 'A' and byte <= 'Z') or (byte >= 'a' and byte <= 'z');
}

fn utf16PrefixLen(text: []const u8) Error!i32 {
    var byte_index: usize = 0;
    var units: i32 = 0;
    while (byte_index < text.len) {
        const rune = try nextRune(text, byte_index);
        byte_index += rune.len;
        units = try addUnits(units, utf16RuneLen(rune.codepoint));
    }
    return units;
}

fn utf16RuneLen(codepoint: u21) i32 {
    return if (@as(u32, codepoint) > 0xffff) 2 else 1;
}

fn addUnits(left: i32, right: i32) Error!i32 {
    return std.math.add(i32, left, right) catch error.LengthOverflow;
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
        if (index + 2 >= input.len or !isCont(input[index + 1]) or !isCont(input[index + 2])) return error.InvalidUtf8;
        const value: u32 = (@as(u32, first & 0x0f) << 12) |
            (@as(u32, input[index + 1] & 0x3f) << 6) |
            @as(u32, input[index + 2] & 0x3f);
        if (value < 0x800 or (value >= 0xd800 and value <= 0xdfff)) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .len = 3 };
    }
    if ((first & 0xf8) == 0xf0) {
        if (index + 3 >= input.len or !isCont(input[index + 1]) or !isCont(input[index + 2]) or !isCont(input[index + 3])) return error.InvalidUtf8;
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

test "target matcher finds homographs" {
    const homographs = [_][]const u8{"lead"};
    const homograph_ids = [_][]const i32{&.{ 1, 2 }};
    var matcher = try create(std.testing.allocator, &homographs, &homograph_ids, &.{}, &.{});
    defer destroy(matcher);
    const out = try matcher.homographMatches("Lead the lead.");
    defer freeMatches(std.testing.allocator, out.ptr, @intCast(out.len));

    try std.testing.expectEqual(@as(usize, 2), out.len);
    try std.testing.expectEqual(@as(i32, 0), out[0].start);
    try std.testing.expectEqual(@as(i32, 4), out[0].end);
    try std.testing.expectEqual(@as(i32, 9), out[1].start);
}

test "target matcher selects longest non-overlapping polyphones" {
    const polyphones = [_][]const u8{ "行", "银行" };
    const polyphone_ids = [_][]const i32{ &.{0}, &.{ 1, 2 } };
    var matcher = try create(std.testing.allocator, &.{}, &.{}, &polyphones, &polyphone_ids);
    defer destroy(matcher);
    const out = try matcher.polyphoneMatches("银行行长");
    defer freeMatches(std.testing.allocator, out.ptr, @intCast(out.len));

    try std.testing.expectEqual(@as(usize, 2), out.len);
    try std.testing.expectEqual(@as(i32, 0), out[0].start);
    try std.testing.expectEqual(@as(i32, 2), out[0].end);
    try std.testing.expectEqual(@as(i32, 2), out[1].start);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, matcher.idsFor(false, out[0].index));
}
