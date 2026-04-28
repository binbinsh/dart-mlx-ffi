const std = @import("std");

pub const Error = error{
    InvalidInput,
    LengthOverflow,
    OutOfMemory,
};

pub const SsmlKind = enum(i32) {
    emphasis = 1,
    phoneme = 2,
    sub = 3,
};

pub const SsmlTag = struct {
    start: i32,
    end: i32,
    kind: SsmlKind,
    value: []const u8,
};

pub const Range = struct {
    start: i32,
    end: i32,
};

const SubTag = struct {
    end: usize,
    alias: ?[]const u8,
};

pub fn composeSsml(
    allocator: std.mem.Allocator,
    input: []const u8,
    tags: []const SsmlTag,
    emotion: ?[]const u8,
) Error![]u8 {
    const unit_len = try utf16Len(input);
    if (unit_len > std.math.maxInt(i32)) {
        return error.LengthOverflow;
    }
    const text_units: i32 = @intCast(unit_len);

    const sorted = allocator.dupe(SsmlTag, tags) catch return error.OutOfMemory;
    defer allocator.free(sorted);
    std.mem.sort(SsmlTag, sorted, {}, lessSsmlTag);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    out.appendSlice(allocator, "<speak>") catch return error.OutOfMemory;
    if (emotion) |label| {
        out.appendSlice(allocator, "<emotion type=\"") catch return error.OutOfMemory;
        try appendXml(allocator, &out, label);
        out.appendSlice(allocator, "\">") catch return error.OutOfMemory;
    }

    var cursor_units: i32 = 0;
    var cursor_byte: usize = 0;
    for (sorted) |tag| {
        if (tag.start < cursor_units or tag.end <= tag.start or tag.end > text_units) {
            continue;
        }
        const start_units: usize = @intCast(tag.start);
        const end_units: usize = @intCast(tag.end);
        const start_byte = byteAtUtf16(input, start_units) orelse continue;
        const end_byte = byteAtUtf16(input, end_units) orelse continue;
        if (start_byte < cursor_byte or end_byte <= start_byte) {
            continue;
        }

        try appendXml(allocator, &out, input[cursor_byte..start_byte]);
        try openSsmlTag(allocator, &out, tag);
        try appendXml(allocator, &out, input[start_byte..end_byte]);
        try closeSsmlTag(allocator, &out, tag.kind);
        cursor_units = tag.end;
        cursor_byte = end_byte;
    }
    try appendXml(allocator, &out, input[cursor_byte..]);

    if (emotion != null) {
        out.appendSlice(allocator, "</emotion>") catch return error.OutOfMemory;
    }
    out.appendSlice(allocator, "</speak>") catch return error.OutOfMemory;
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

pub fn stripSsml(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var index: usize = 0;
    while (index < input.len) {
        if (input[index] == '<') {
            if (subTag(input, index)) |tag| {
                if (findSubClose(input, tag.end)) |close| {
                    if (tag.alias) |alias| {
                        out.appendSlice(allocator, alias) catch return error.OutOfMemory;
                    }
                    index = close;
                    continue;
                }
            }
            if (tagEnd(input, index)) |end| {
                if (end > index + 1) {
                    index = end + 1;
                    continue;
                }
            }
        }
        out.append(allocator, input[index]) catch return error.OutOfMemory;
        index += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

pub fn tnEn(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    const value = trimAscii(input);
    if (try money(allocator, value)) |out| {
        return out;
    }
    if (parseCommaInt(value)) |number| {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        try appendEnglishInt(allocator, &out, number);
        return out.toOwnedSlice(allocator) catch error.OutOfMemory;
    }
    return allocator.dupe(u8, value) catch error.OutOfMemory;
}

pub fn tnZh(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (input) |byte| {
        if (digitZh(byte)) |text| {
            out.appendSlice(allocator, text) catch return error.OutOfMemory;
        } else {
            out.append(allocator, byte) catch return error.OutOfMemory;
        }
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

pub fn preferZh(input: []const u8) bool {
    var cjk: usize = 0;
    var latin: usize = 0;
    var index: usize = 0;
    while (index < input.len) {
        const rune = nextRune(input, index) orelse {
            index += 1;
            continue;
        };
        index += rune.len;
        if (isCjk(rune.codepoint)) {
            cjk += 1;
        } else if (isAsciiLetter(rune.codepoint)) {
            latin += 1;
        }
    }
    return cjk > 0 and cjk >= latin;
}

pub fn hasZh(input: []const u8) bool {
    var index: usize = 0;
    while (index < input.len) {
        const rune = nextRune(input, index) orelse {
            index += 1;
            continue;
        };
        index += rune.len;
        if (isCjk(rune.codepoint)) {
            return true;
        }
    }
    return false;
}

pub fn normSpans(
    input: []const u8,
    starts: []const i32,
    ends: []const i32,
    out_starts: []i32,
    out_ends: []i32,
) Error!usize {
    if (starts.len != ends.len or out_starts.len < starts.len or out_ends.len < starts.len) {
        return error.InvalidInput;
    }
    const units = try utf16Len(input);
    var produced: usize = 0;
    for (starts, ends) |raw_start, raw_end| {
        if (raw_start < 0 or raw_end < raw_start) {
            return error.InvalidInput;
        }
        var start: usize = @intCast(raw_start);
        var end: usize = @intCast(raw_end);
        if (end > units) {
            return error.InvalidInput;
        }
        while (start < end and isTrimUnit(input, start)) {
            start += 1;
        }
        while (end > start and isTrimUnit(input, end - 1)) {
            end -= 1;
        }
        if (end <= start) {
            continue;
        }
        if (produced > 0 and isBlankUtf16Gap(input, @intCast(out_ends[produced - 1]), start)) {
            out_ends[produced - 1] = @intCast(end);
            continue;
        }
        out_starts[produced] = @intCast(start);
        out_ends[produced] = @intCast(end);
        produced += 1;
    }
    return produced;
}

pub fn selectTn(
    input: []const u8,
    en_starts: []const i32,
    en_ends: []const i32,
    zh_starts: []const i32,
    zh_ends: []const i32,
    out_sources: []i32,
    out_indices: []i32,
) Error!usize {
    if (en_starts.len != en_ends.len or zh_starts.len != zh_ends.len) {
        return error.InvalidInput;
    }
    const max_count = en_starts.len + zh_starts.len;
    if (out_sources.len < max_count or out_indices.len < max_count) {
        return error.InvalidInput;
    }
    try validateRanges(en_starts, en_ends);
    try validateRanges(zh_starts, zh_ends);

    var produced: usize = 0;
    if (en_starts.len == 0) {
        try addTnChoices(2, zh_starts.len, out_sources, out_indices, &produced, en_starts, en_ends, zh_starts, zh_ends);
    } else if (zh_starts.len == 0) {
        try addTnChoices(1, en_starts.len, out_sources, out_indices, &produced, en_starts, en_ends, zh_starts, zh_ends);
    } else if (preferZh(input)) {
        try addTnChoices(2, zh_starts.len, out_sources, out_indices, &produced, en_starts, en_ends, zh_starts, zh_ends);
        try addTnChoices(1, en_starts.len, out_sources, out_indices, &produced, en_starts, en_ends, zh_starts, zh_ends);
    } else {
        try addTnChoices(1, en_starts.len, out_sources, out_indices, &produced, en_starts, en_ends, zh_starts, zh_ends);
        try addTnChoices(2, zh_starts.len, out_sources, out_indices, &produced, en_starts, en_ends, zh_starts, zh_ends);
    }
    sortTnChoices(out_sources[0..produced], out_indices[0..produced], en_starts, en_ends, zh_starts, zh_ends);
    return produced;
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidInput => "Zig text received invalid input.",
        error.LengthOverflow => "Zig text output length overflowed.",
        error.OutOfMemory => "Zig text ran out of memory.",
    };
}

fn money(allocator: std.mem.Allocator, value: []const u8) Error!?[]u8 {
    if (value.len < 2 or value[0] != '$') {
        return null;
    }
    var cursor: usize = 1;
    const dollar_start = cursor;
    while (cursor < value.len and std.ascii.isDigit(value[cursor])) cursor += 1;
    if (cursor == dollar_start) {
        return null;
    }
    const dollars = parseU64(value[dollar_start..cursor]) orelse return null;

    var cents: ?u64 = null;
    if (cursor < value.len) {
        if (value[cursor] != '.') {
            return null;
        }
        cursor += 1;
        const cents_start = cursor;
        while (cursor < value.len and std.ascii.isDigit(value[cursor])) cursor += 1;
        const cents_len = cursor - cents_start;
        if (cents_len == 0 or cents_len > 2 or cursor != value.len) {
            return null;
        }
        var cents_value = parseU64(value[cents_start..cursor]) orelse return null;
        if (cents_len == 1) {
            cents_value *= 10;
        }
        cents = cents_value;
    } else if (cursor != value.len) {
        return null;
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendEnglishU64(allocator, &out, dollars);
    out.appendSlice(allocator, " dollars") catch return error.OutOfMemory;
    if (cents) |amount| {
        out.appendSlice(allocator, " and ") catch return error.OutOfMemory;
        try appendEnglishU64(allocator, &out, amount);
        out.appendSlice(allocator, " cents") catch return error.OutOfMemory;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn parseCommaInt(value: []const u8) ?i64 {
    if (value.len == 0) return null;
    var index: usize = 0;
    var negative = false;
    if (value[0] == '-') {
        negative = true;
        index = 1;
    } else if (value[0] == '+') {
        index = 1;
    }
    var seen_digit = false;
    var out: i64 = 0;
    while (index < value.len) : (index += 1) {
        const byte = value[index];
        if (byte == ',') {
            continue;
        }
        if (!std.ascii.isDigit(byte)) {
            return null;
        }
        seen_digit = true;
        const digit: i64 = @intCast(byte - '0');
        out = std.math.mul(i64, out, 10) catch return null;
        out = std.math.add(i64, out, digit) catch return null;
    }
    if (!seen_digit) return null;
    return if (negative) -out else out;
}

fn parseU64(value: []const u8) ?u64 {
    if (value.len == 0) return null;
    var out: u64 = 0;
    for (value) |byte| {
        if (!std.ascii.isDigit(byte)) return null;
        out = std.math.mul(u64, out, 10) catch return null;
        out = std.math.add(u64, out, @as(u64, byte - '0')) catch return null;
    }
    return out;
}

fn appendEnglishInt(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: i64,
) Error!void {
    if (value < 0) {
        out.appendSlice(allocator, "minus ") catch return error.OutOfMemory;
        const magnitude: u64 = if (value == std.math.minInt(i64))
            @as(u64, std.math.maxInt(i64)) + 1
        else
            @intCast(-value);
        try appendEnglishU64(allocator, out, magnitude);
        return;
    }
    try appendEnglishU64(allocator, out, @intCast(value));
}

fn appendEnglishU64(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: u64,
) Error!void {
    const small = [_][]const u8{
        "zero",    "one",     "two",       "three",    "four",
        "five",    "six",     "seven",     "eight",    "nine",
        "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    };
    const tens = [_][]const u8{
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    };
    if (value < 20) {
        out.appendSlice(allocator, small[@intCast(value)]) catch return error.OutOfMemory;
    } else if (value < 100) {
        const rest = value % 10;
        out.appendSlice(allocator, tens[@intCast(value / 10)]) catch return error.OutOfMemory;
        if (rest != 0) {
            out.append(allocator, ' ') catch return error.OutOfMemory;
            out.appendSlice(allocator, small[@intCast(rest)]) catch return error.OutOfMemory;
        }
    } else if (value < 1000) {
        const rest = value % 100;
        out.appendSlice(allocator, small[@intCast(value / 100)]) catch return error.OutOfMemory;
        out.appendSlice(allocator, " hundred") catch return error.OutOfMemory;
        if (rest != 0) {
            out.append(allocator, ' ') catch return error.OutOfMemory;
            try appendEnglishU64(allocator, out, rest);
        }
    } else if (value < 1000000) {
        const rest = value % 1000;
        try appendEnglishU64(allocator, out, value / 1000);
        out.appendSlice(allocator, " thousand") catch return error.OutOfMemory;
        if (rest != 0) {
            out.append(allocator, ' ') catch return error.OutOfMemory;
            try appendEnglishU64(allocator, out, rest);
        }
    } else {
        var buffer: [32]u8 = undefined;
        const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch return error.LengthOverflow;
        out.appendSlice(allocator, text) catch return error.OutOfMemory;
    }
}

fn digitZh(byte: u8) ?[]const u8 {
    return switch (byte) {
        '0' => "零",
        '1' => "一",
        '2' => "二",
        '3' => "三",
        '4' => "四",
        '5' => "五",
        '6' => "六",
        '7' => "七",
        '8' => "八",
        '9' => "九",
        else => null,
    };
}

fn lessSsmlTag(_: void, left: SsmlTag, right: SsmlTag) bool {
    if (left.start == right.start) {
        return left.end < right.end;
    }
    return left.start < right.start;
}

fn openSsmlTag(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    tag: SsmlTag,
) Error!void {
    switch (tag.kind) {
        .emphasis => out.appendSlice(allocator, "<emphasis>") catch return error.OutOfMemory,
        .phoneme => {
            out.appendSlice(allocator, "<phoneme ph=\"") catch return error.OutOfMemory;
            try appendXml(allocator, out, tag.value);
            out.appendSlice(allocator, "\">") catch return error.OutOfMemory;
        },
        .sub => {
            out.appendSlice(allocator, "<sub alias=\"") catch return error.OutOfMemory;
            try appendXml(allocator, out, tag.value);
            out.appendSlice(allocator, "\">") catch return error.OutOfMemory;
        },
    }
}

fn closeSsmlTag(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    kind: SsmlKind,
) Error!void {
    switch (kind) {
        .emphasis => out.appendSlice(allocator, "</emphasis>") catch return error.OutOfMemory,
        .phoneme => out.appendSlice(allocator, "</phoneme>") catch return error.OutOfMemory,
        .sub => out.appendSlice(allocator, "</sub>") catch return error.OutOfMemory,
    }
}

fn appendXml(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: []const u8,
) Error!void {
    for (value) |byte| {
        switch (byte) {
            '&' => out.appendSlice(allocator, "&amp;") catch return error.OutOfMemory,
            '<' => out.appendSlice(allocator, "&lt;") catch return error.OutOfMemory,
            '>' => out.appendSlice(allocator, "&gt;") catch return error.OutOfMemory,
            '"' => out.appendSlice(allocator, "&quot;") catch return error.OutOfMemory,
            else => out.append(allocator, byte) catch return error.OutOfMemory,
        }
    }
}

fn utf16Len(input: []const u8) Error!usize {
    var units: usize = 0;
    var index: usize = 0;
    while (index < input.len) {
        const rune = nextRune(input, index) orelse return error.InvalidInput;
        index += rune.len;
        const delta: usize = if (rune.codepoint > 0xffff) 2 else 1;
        units = std.math.add(usize, units, delta) catch return error.LengthOverflow;
    }
    return units;
}

fn byteAtUtf16(input: []const u8, target_units: usize) ?usize {
    var units: usize = 0;
    var index: usize = 0;
    while (index < input.len) {
        if (units == target_units) {
            return index;
        }
        const rune = nextRune(input, index) orelse return null;
        const delta: usize = if (rune.codepoint > 0xffff) 2 else 1;
        if (units + delta > target_units) {
            return null;
        }
        units += delta;
        index += rune.len;
    }
    return if (units == target_units) input.len else null;
}

fn utf16Slice(input: []const u8, start_units: usize, end_units: usize) ?[]const u8 {
    const start = byteAtUtf16(input, start_units) orelse return null;
    const end = byteAtUtf16(input, end_units) orelse return null;
    if (end < start) return null;
    return input[start..end];
}

fn isTrimUnit(input: []const u8, unit: usize) bool {
    const slice = utf16Slice(input, unit, unit + 1) orelse return false;
    return slice.len == 1 and isTrimByte(slice[0]);
}

fn isTrimByte(byte: u8) bool {
    return switch (byte) {
        ' ', '\t', '\r', '\n', '"', '\\', '\'', '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}' => true,
        else => false,
    };
}

fn isBlankUtf16Gap(input: []const u8, start_units: usize, end_units: usize) bool {
    if (end_units <= start_units) {
        return false;
    }
    const gap = utf16Slice(input, start_units, end_units) orelse return false;
    if (gap.len == 0) {
        return false;
    }
    var index: usize = 0;
    while (index < gap.len) {
        const rune = nextRune(gap, index) orelse return false;
        if (!isSpaceRune(rune.codepoint)) {
            return false;
        }
        index += rune.len;
    }
    return true;
}

fn isSpaceRune(codepoint: u21) bool {
    return switch (@as(u32, codepoint)) {
        ' ', '\t', '\n', '\r', 0x0b, 0x0c, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn validateRanges(starts: []const i32, ends: []const i32) Error!void {
    for (starts, ends) |start, end| {
        if (start < 0 or end < start) {
            return error.InvalidInput;
        }
    }
}

fn addTnChoices(
    source: i32,
    count: usize,
    out_sources: []i32,
    out_indices: []i32,
    produced: *usize,
    en_starts: []const i32,
    en_ends: []const i32,
    zh_starts: []const i32,
    zh_ends: []const i32,
) Error!void {
    for (0..count) |index| {
        const range = choiceRange(source, @intCast(index), en_starts, en_ends, zh_starts, zh_ends) orelse return error.InvalidInput;
        var overlaps = false;
        for (0..produced.*) |used| {
            const taken = choiceRange(
                out_sources[used],
                out_indices[used],
                en_starts,
                en_ends,
                zh_starts,
                zh_ends,
            ) orelse return error.InvalidInput;
            if (!(range.end <= taken.start or range.start >= taken.end)) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) {
            out_sources[produced.*] = source;
            out_indices[produced.*] = @intCast(index);
            produced.* += 1;
        }
    }
}

fn sortTnChoices(
    sources: []i32,
    indices: []i32,
    en_starts: []const i32,
    en_ends: []const i32,
    zh_starts: []const i32,
    zh_ends: []const i32,
) void {
    var index: usize = 1;
    while (index < sources.len) : (index += 1) {
        const source = sources[index];
        const item = indices[index];
        var cursor = index;
        while (cursor > 0 and choiceLess(source, item, sources[cursor - 1], indices[cursor - 1], en_starts, en_ends, zh_starts, zh_ends)) {
            sources[cursor] = sources[cursor - 1];
            indices[cursor] = indices[cursor - 1];
            cursor -= 1;
        }
        sources[cursor] = source;
        indices[cursor] = item;
    }
}

fn choiceLess(
    left_source: i32,
    left_index: i32,
    right_source: i32,
    right_index: i32,
    en_starts: []const i32,
    en_ends: []const i32,
    zh_starts: []const i32,
    zh_ends: []const i32,
) bool {
    const left = choiceRange(left_source, left_index, en_starts, en_ends, zh_starts, zh_ends) orelse return false;
    const right = choiceRange(right_source, right_index, en_starts, en_ends, zh_starts, zh_ends) orelse return false;
    if (left.start != right.start) {
        return left.start < right.start;
    }
    return left.end < right.end;
}

fn choiceRange(
    source: i32,
    index: i32,
    en_starts: []const i32,
    en_ends: []const i32,
    zh_starts: []const i32,
    zh_ends: []const i32,
) ?Range {
    if (index < 0) return null;
    const idx: usize = @intCast(index);
    return switch (source) {
        1 => if (idx < en_starts.len) .{ .start = en_starts[idx], .end = en_ends[idx] } else null,
        2 => if (idx < zh_starts.len) .{ .start = zh_starts[idx], .end = zh_ends[idx] } else null,
        else => null,
    };
}

fn subTag(input: []const u8, index: usize) ?SubTag {
    if (index + 4 > input.len or input[index] != '<') {
        return null;
    }
    if (!eqlIgnoreCase(input[index + 1 .. index + 4], "sub")) {
        return null;
    }
    const after_name = index + 4;
    if (after_name < input.len and isWord(input[after_name])) {
        return null;
    }
    const end = tagEnd(input, index) orelse return null;
    return .{
        .end = end + 1,
        .alias = aliasAttr(input[after_name..end]),
    };
}

fn aliasAttr(attrs: []const u8) ?[]const u8 {
    var index: usize = 0;
    while (index + 5 <= attrs.len) : (index += 1) {
        if (!eqlIgnoreCase(attrs[index .. index + 5], "alias")) {
            continue;
        }
        if (index > 0 and isWord(attrs[index - 1])) {
            continue;
        }
        var cursor = index + 5;
        if (cursor < attrs.len and isWord(attrs[cursor])) {
            continue;
        }
        while (cursor < attrs.len and isAsciiSpace(attrs[cursor])) cursor += 1;
        if (cursor >= attrs.len or attrs[cursor] != '=') {
            continue;
        }
        cursor += 1;
        while (cursor < attrs.len and isAsciiSpace(attrs[cursor])) cursor += 1;
        if (cursor >= attrs.len or attrs[cursor] != '"') {
            continue;
        }
        const start = cursor + 1;
        const end = std.mem.indexOfScalarPos(u8, attrs, start, '"') orelse return null;
        return attrs[start..end];
    }
    return null;
}

fn findSubClose(input: []const u8, start: usize) ?usize {
    var index = start;
    while (std.mem.indexOfScalarPos(u8, input, index, '<')) |tag_start| {
        if (tag_start + 6 <= input.len and eqlIgnoreCase(input[tag_start .. tag_start + 6], "</sub>")) {
            return tag_start + 6;
        }
        index = tag_start + 1;
    }
    return null;
}

fn tagEnd(input: []const u8, start: usize) ?usize {
    return std.mem.indexOfScalarPos(u8, input, start + 1, '>');
}

fn eqlIgnoreCase(left: []const u8, right: []const u8) bool {
    if (left.len != right.len) return false;
    for (left, right) |a, b| {
        if (std.ascii.toLower(a) != std.ascii.toLower(b)) {
            return false;
        }
    }
    return true;
}

fn isWord(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

fn isAsciiSpace(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r';
}

fn trimAscii(value: []const u8) []const u8 {
    var start: usize = 0;
    var end = value.len;
    while (start < end and isAsciiSpace(value[start])) start += 1;
    while (end > start and isAsciiSpace(value[end - 1])) end -= 1;
    return value[start..end];
}

const Rune = struct {
    codepoint: u21,
    len: usize,
};

fn nextRune(input: []const u8, index: usize) ?Rune {
    const first = input[index];
    if (first < 0x80) {
        return .{ .codepoint = @intCast(first), .len = 1 };
    }
    if ((first & 0xe0) == 0xc0) {
        if (index + 1 >= input.len or !isCont(input[index + 1])) return null;
        const value: u32 = (@as(u32, first & 0x1f) << 6) |
            @as(u32, input[index + 1] & 0x3f);
        if (value < 0x80) return null;
        return .{ .codepoint = @intCast(value), .len = 2 };
    }
    if ((first & 0xf0) == 0xe0) {
        if (index + 2 >= input.len or !isCont(input[index + 1]) or !isCont(input[index + 2])) {
            return null;
        }
        const value: u32 = (@as(u32, first & 0x0f) << 12) |
            (@as(u32, input[index + 1] & 0x3f) << 6) |
            @as(u32, input[index + 2] & 0x3f);
        if (value < 0x800 or (value >= 0xd800 and value <= 0xdfff)) return null;
        return .{ .codepoint = @intCast(value), .len = 3 };
    }
    if ((first & 0xf8) == 0xf0) {
        if (index + 3 >= input.len or
            !isCont(input[index + 1]) or
            !isCont(input[index + 2]) or
            !isCont(input[index + 3]))
        {
            return null;
        }
        const value: u32 = (@as(u32, first & 0x07) << 18) |
            (@as(u32, input[index + 1] & 0x3f) << 12) |
            (@as(u32, input[index + 2] & 0x3f) << 6) |
            @as(u32, input[index + 3] & 0x3f);
        if (value < 0x10000 or value > 0x10ffff) return null;
        return .{ .codepoint = @intCast(value), .len = 4 };
    }
    return null;
}

fn isCont(byte: u8) bool {
    return (byte & 0xc0) == 0x80;
}

fn isAsciiLetter(codepoint: u21) bool {
    return (codepoint >= 'a' and codepoint <= 'z') or
        (codepoint >= 'A' and codepoint <= 'Z');
}

fn isCjk(codepoint: u21) bool {
    return codepoint >= 0x4e00 and codepoint <= 0x9fff;
}

test "text strips ssml sub aliases and tags" {
    const out = try stripSsml(std.testing.allocator, "<speak>I paid <sub alias=\"one\">$1</sub>.</speak>");
    defer std.testing.allocator.free(out);

    try std.testing.expectEqualStrings("I paid one.", out);
}

test "text composes ssml tags and escapes values" {
    const tags = [_]SsmlTag{
        .{ .start = 0, .end = 4, .kind = .phoneme, .value = "lead_verb" },
        .{ .start = 10, .end = 13, .kind = .sub, .value = "twelve & more" },
    };
    const out = try composeSsml(std.testing.allocator, "Lead paid $12.", &tags, "happy");
    defer std.testing.allocator.free(out);

    try std.testing.expectEqualStrings(
        "<speak><emotion type=\"happy\"><phoneme ph=\"lead_verb\">Lead</phoneme> paid <sub alias=\"twelve &amp; more\">$12</sub>.</emotion></speak>",
        out,
    );
}

test "text composes ssml using dart utf16 offsets" {
    const tags = [_]SsmlTag{
        .{ .start = 3, .end = 7, .kind = .sub, .value = "二零二六" },
    };
    const out = try composeSsml(std.testing.allocator, "🙂年2026", &tags, null);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqualStrings(
        "<speak>🙂年<sub alias=\"二零二六\">2026</sub></speak>",
        out,
    );
}

test "text strips ssml case-insensitively" {
    const out = try stripSsml(std.testing.allocator, "<SPEAK><SUB ALIAS=\"x\">y</SUB></SPEAK>");
    defer std.testing.allocator.free(out);

    try std.testing.expectEqualStrings("x", out);
}

test "text verbalizes tn fallback values" {
    const en = try tnEn(std.testing.allocator, " $123.4 ");
    defer std.testing.allocator.free(en);
    try std.testing.expectEqualStrings("one hundred twenty three dollars and forty cents", en);

    const zh = try tnZh(std.testing.allocator, "2026年");
    defer std.testing.allocator.free(zh);
    try std.testing.expectEqualStrings("二零二六年", zh);
}

test "text normalizes emphasis spans" {
    var starts = [_]i32{ 0, 10, 18 };
    var ends = [_]i32{ 7, 16, 20 };
    var out_starts = [_]i32{0} ** 3;
    var out_ends = [_]i32{0} ** 3;
    const count = try normSpans(
        "\" hello   world!  ok",
        &starts,
        &ends,
        &out_starts,
        &out_ends,
    );

    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqualSlices(i32, &.{ 2, 18 }, out_starts[0..count]);
    try std.testing.expectEqualSlices(i32, &.{ 15, 20 }, out_ends[0..count]);
}

test "text normalizes spans around utf16 surrogate pairs" {
    var starts = [_]i32{0};
    var ends = [_]i32{4};
    var out_starts = [_]i32{0};
    var out_ends = [_]i32{0};
    const count = try normSpans("🙂 hi", &starts, &ends, &out_starts, &out_ends);

    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expectEqual(@as(i32, 0), out_starts[0]);
    try std.testing.expectEqual(@as(i32, 4), out_ends[0]);
}

test "text selects tn ranges by language preference" {
    const en_starts = [_]i32{ 3, 8 };
    const en_ends = [_]i32{ 7, 10 };
    const zh_starts = [_]i32{3};
    const zh_ends = [_]i32{7};
    var sources = [_]i32{0} ** 3;
    var indices = [_]i32{0} ** 3;

    const zh_count = try selectTn(
        "今天是2026年 ok",
        &en_starts,
        &en_ends,
        &zh_starts,
        &zh_ends,
        &sources,
        &indices,
    );
    try std.testing.expectEqual(@as(usize, 2), zh_count);
    try std.testing.expectEqualSlices(i32, &.{ 2, 1 }, sources[0..zh_count]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1 }, indices[0..zh_count]);

    const en_count = try selectTn(
        "I paid $12",
        &en_starts,
        &en_ends,
        &zh_starts,
        &zh_ends,
        &sources,
        &indices,
    );
    try std.testing.expectEqual(@as(usize, 2), en_count);
    try std.testing.expectEqualSlices(i32, &.{ 1, 1 }, sources[0..en_count]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1 }, indices[0..en_count]);
}

test "text prefers chinese tn when cjk count dominates" {
    try std.testing.expect(preferZh("今天是2026年"));
    try std.testing.expect(!preferZh("I paid $12"));
    try std.testing.expect(hasZh("hello 你好"));
    try std.testing.expect(!hasZh("hello"));
}
