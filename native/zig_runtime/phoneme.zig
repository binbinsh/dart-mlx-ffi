const std = @import("std");

pub const Error = error{
    InvalidInput,
    InvalidUtf8,
    LengthOverflow,
    OutOfMemory,
};

const Rune = struct {
    codepoint: u21,
    len: usize,
};

pub fn normText(
    allocator: std.mem.Allocator,
    input: []const u8,
    language: []const u8,
) Error![]u8 {
    if (!isEnglish(language)) {
        return allocator.dupe(u8, input) catch error.OutOfMemory;
    }

    var out = try mapEnglish(allocator, input);
    errdefer allocator.free(out);
    out = try step(allocator, out, squashSpaces);
    out = try step(allocator, out, titles);
    out = try step(allocator, out, numbers);
    out = try step(allocator, out, numericCommas);
    out = try step(allocator, out, money);
    out = try step(allocator, out, pointNumbers);
    out = try step(allocator, out, digitDash);
    out = try step(allocator, out, digitS);
    out = try step(allocator, out, consonantS);
    out = try step(allocator, out, upperXS);
    out = try step(allocator, out, dottedAbbr);
    out = try step(allocator, out, letterDots);
    return trimOwned(allocator, out);
}

pub fn post(
    allocator: std.mem.Allocator,
    input: []const u8,
    language: []const u8,
) Error![]u8 {
    if (!isEnglish(language)) {
        return allocator.dupe(u8, input) catch error.OutOfMemory;
    }

    var out = allocator.dupe(u8, input) catch return error.OutOfMemory;
    errdefer allocator.free(out);
    out = try replaceStep(allocator, out, "kəkˈoːɹoʊ", "kˈoʊkəɹoʊ");
    out = try replaceStep(allocator, out, "kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ");
    out = try replaceStep(allocator, out, "ʲ", "j");
    out = try replaceStep(allocator, out, "r", "ɹ");
    out = try replaceStep(allocator, out, "x", "k");
    out = try replaceStep(allocator, out, "ɬ", "l");
    out = try step(allocator, out, hundredSpace);
    out = try step(allocator, out, zSpace);
    if (std.mem.eql(u8, language, "en-us")) {
        out = try step(allocator, out, ninetyUs);
    }
    return out;
}

pub fn plainSsml(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var index: usize = 0;
    while (index < input.len) {
        if (input[index] == '<') {
            if (subTag(input, index)) |tag| {
                if (findSubClose(input, tag.end)) |close| {
                    if (tag.alias) |alias| {
                        try appendXmlDecoded(allocator, &out, alias);
                    }
                    index = close;
                    continue;
                }
            }
            if (tagEnd(input, index)) |end| {
                index = end + 1;
                continue;
            }
        }
        if (input[index] == '&') {
            if (xmlEntity(input, index)) |entity| {
                out.appendSlice(allocator, entity.value) catch return error.OutOfMemory;
                index += entity.len;
                continue;
            }
        }
        out.append(allocator, input[index]) catch return error.OutOfMemory;
        index += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

pub fn explicitPhonemes(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    return phonemeSpaces(allocator, input, true, false);
}

pub fn pinyinNorm(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    return phonemeSpaces(allocator, input, false, true);
}

pub fn looksPinyin(input: []const u8) bool {
    if (trim(input).len == 0 or hasCjk(input)) {
        return false;
    }
    var any = false;
    var index: usize = 0;
    while (index < input.len) {
        while (index < input.len and isSpaceByte(input[index])) index += 1;
        if (index >= input.len) break;
        const start_at = index;
        while (index < input.len and !isSpaceByte(input[index])) index += 1;
        var token = input[start_at..index];
        if (std.mem.indexOfScalar(u8, token, '|')) |bar| {
            token = token[0..bar];
        }
        token = trimPinyinPunct(token);
        if (token.len == 0) {
            continue;
        }
        if (!validPinyinToken(token)) {
            return false;
        }
        any = true;
    }
    return any;
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidInput => "Zig phoneme text received invalid input.",
        error.InvalidUtf8 => "Zig phoneme text received invalid UTF-8.",
        error.LengthOverflow => "Zig phoneme text output length overflowed.",
        error.OutOfMemory => "Zig phoneme text ran out of memory.",
    };
}

fn step(
    allocator: std.mem.Allocator,
    owned: []u8,
    comptime func: fn (std.mem.Allocator, []const u8) Error![]u8,
) Error![]u8 {
    const next = try func(allocator, owned);
    allocator.free(owned);
    return next;
}

fn replaceStep(
    allocator: std.mem.Allocator,
    owned: []u8,
    needle: []const u8,
    repl: []const u8,
) Error![]u8 {
    const next = try replaceLiteral(allocator, owned, needle, repl);
    allocator.free(owned);
    return next;
}

fn isEnglish(language: []const u8) bool {
    return std.mem.eql(u8, language, "en-us") or std.mem.eql(u8, language, "en");
}

fn mapEnglish(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var index: usize = 0;
    while (index < input.len) {
        const rune = try nextRune(input, index);
        const raw = input[index .. index + rune.len];
        index += rune.len;
        switch (@as(u32, rune.codepoint)) {
            0x2018, 0x2019 => out.append(allocator, '\'') catch return error.OutOfMemory,
            0x00ab => out.appendSlice(allocator, "“") catch return error.OutOfMemory,
            0x00bb => out.appendSlice(allocator, "”") catch return error.OutOfMemory,
            0x201c, 0x201d => out.append(allocator, '"') catch return error.OutOfMemory,
            '(' => out.appendSlice(allocator, "«") catch return error.OutOfMemory,
            ')' => out.appendSlice(allocator, "»") catch return error.OutOfMemory,
            0x3001, 0xff0c => out.appendSlice(allocator, ", ") catch return error.OutOfMemory,
            0x3002 => out.appendSlice(allocator, ". ") catch return error.OutOfMemory,
            0xff01 => out.appendSlice(allocator, "! ") catch return error.OutOfMemory,
            0xff1a => out.appendSlice(allocator, ": ") catch return error.OutOfMemory,
            0xff1b => out.appendSlice(allocator, "; ") catch return error.OutOfMemory,
            0xff1f => out.appendSlice(allocator, "? ") catch return error.OutOfMemory,
            else => {
                if (isSpace(rune.codepoint) and rune.codepoint != ' ' and rune.codepoint != '\n') {
                    out.append(allocator, ' ') catch return error.OutOfMemory;
                } else {
                    out.appendSlice(allocator, raw) catch return error.OutOfMemory;
                }
            },
        }
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn squashSpaces(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var last = false;
    for (input) |byte| {
        if (byte == ' ') {
            if (!last) {
                out.append(allocator, byte) catch return error.OutOfMemory;
            }
            last = true;
        } else {
            out.append(allocator, byte) catch return error.OutOfMemory;
            last = false;
        }
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn titles(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        if (wordStart(input, i)) {
            if (starts(input, i, "DR.") or starts(input, i, "Dr.")) {
                if (upperAfter(input, i + 3)) {
                    out.appendSlice(allocator, "Doctor") catch return error.OutOfMemory;
                    i += 3;
                    continue;
                }
            }
            if (starts(input, i, "Mr.") or (starts(input, i, "MR.") and upperAfter(input, i + 3))) {
                out.appendSlice(allocator, "Mister") catch return error.OutOfMemory;
                i += 3;
                continue;
            }
            if (starts(input, i, "Ms.") or (starts(input, i, "MS.") and upperAfter(input, i + 3))) {
                out.appendSlice(allocator, "Miss") catch return error.OutOfMemory;
                i += 3;
                continue;
            }
            if (starts(input, i, "Mrs.") or (starts(input, i, "MRS.") and upperAfter(input, i + 4))) {
                out.appendSlice(allocator, "Mrs") catch return error.OutOfMemory;
                i += 4;
                continue;
            }
            if (startsIgnore(input, i, "etc.") and !upperAfter(input, i + 4)) {
                out.appendSlice(allocator, "etc") catch return error.OutOfMemory;
                i += 4;
                continue;
            }
            if (yeahLen(input, i)) |len| {
                out.append(allocator, input[i]) catch return error.OutOfMemory;
                out.appendSlice(allocator, "e'a") catch return error.OutOfMemory;
                i += len;
                continue;
            }
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn numbers(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (decimalLen(input, i)) |len| {
            out.appendSlice(allocator, input[i .. i + len]) catch return error.OutOfMemory;
            i += len;
            continue;
        }
        if (yearLen(input, i)) |len| {
            try appendYear(allocator, &out, input[i .. i + len]);
            i += len;
            continue;
        }
        if (timeLen(input, i)) |len| {
            try appendTime(allocator, &out, input[i .. i + len]);
            i += len;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn numericCommas(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (input, 0..) |byte, i| {
        if (byte == ',' and i > 0 and i + 1 < input.len and isDigit(input[i - 1]) and isDigit(input[i + 1])) {
            continue;
        }
        out.append(allocator, byte) catch return error.OutOfMemory;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn money(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (moneyLen(input, i)) |len| {
            try appendMoney(allocator, &out, input[i .. i + len]);
            i += len;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn pointNumbers(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (decimalLen(input, i)) |len| {
            try appendPoint(allocator, &out, input[i .. i + len]);
            i += len;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn digitDash(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (i + 2 < input.len and isDigit(input[i]) and input[i + 1] == '-' and isDigit(input[i + 2])) {
            out.append(allocator, input[i]) catch return error.OutOfMemory;
            out.appendSlice(allocator, " to ") catch return error.OutOfMemory;
            i += 2;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn digitS(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (i + 1 < input.len and isDigit(input[i]) and input[i + 1] == 'S' and wordEnd(input, i + 2)) {
            out.append(allocator, input[i]) catch return error.OutOfMemory;
            out.appendSlice(allocator, " S") catch return error.OutOfMemory;
            i += 2;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn consonantS(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (isUpperConsonant(input[i])) {
            var end = i + 1;
            if (end < input.len and input[end] == '\'') end += 1;
            if (end < input.len and input[end] == 's' and wordEnd(input, end + 1)) {
                out.append(allocator, input[i]) catch return error.OutOfMemory;
                out.appendSlice(allocator, "'S") catch return error.OutOfMemory;
                i = end + 1;
                continue;
            }
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn upperXS(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (starts(input, i, "X'S") and wordEnd(input, i + 3)) {
            out.appendSlice(allocator, "X's") catch return error.OutOfMemory;
            i += 3;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn dottedAbbr(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        const len = dottedAbbrLen(input, i) orelse {
            out.append(allocator, input[i]) catch return error.OutOfMemory;
            i += 1;
            continue;
        };
        for (input[i .. i + len]) |byte| {
            out.append(allocator, if (byte == '.') '-' else byte) catch return error.OutOfMemory;
        }
        i += len;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn letterDots(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (i + 2 < input.len and isAsciiLetter(input[i]) and input[i + 1] == '.' and isAsciiLetter(input[i + 2])) {
            out.append(allocator, input[i]) catch return error.OutOfMemory;
            out.append(allocator, '-') catch return error.OutOfMemory;
            i += 2;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn hundredSpace(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (hundredPrefixLen(input, i)) |len| {
            out.appendSlice(allocator, input[i .. i + len]) catch return error.OutOfMemory;
            out.append(allocator, ' ') catch return error.OutOfMemory;
            i += len;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn zSpace(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (i + 1 < input.len and input[i] == ' ' and input[i + 1] == 'z' and zBoundary(input, i + 2)) {
            out.append(allocator, 'z') catch return error.OutOfMemory;
            i += 2;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn ninetyUs(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    const needle = "nˈaɪnti";
    var i: usize = 0;
    while (i < input.len) {
        if (starts(input, i, needle) and !starts(input, i + needle.len, "ː")) {
            out.appendSlice(allocator, "nˈaɪndi") catch return error.OutOfMemory;
            i += needle.len;
            continue;
        }
        out.append(allocator, input[i]) catch return error.OutOfMemory;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn phonemeSpaces(
    allocator: std.mem.Allocator,
    input: []const u8,
    stress: bool,
    pinyin: bool,
) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var last_space = true;
    var index: usize = 0;
    while (index < input.len) {
        const rune = try nextRune(input, index);
        const raw = input[index .. index + rune.len];
        index += rune.len;
        if (isSpace(rune.codepoint)) {
            if (!last_space and out.items.len > 0) {
                out.append(allocator, ' ') catch return error.OutOfMemory;
            }
            last_space = true;
            continue;
        }
        if (stress and rune.codepoint == '\'') {
            out.appendSlice(allocator, "ˈ") catch return error.OutOfMemory;
        } else if (pinyin and rune.codepoint == 0x00dc) {
            out.append(allocator, 'V') catch return error.OutOfMemory;
        } else if (pinyin and rune.codepoint == 0x00fc) {
            out.append(allocator, 'v') catch return error.OutOfMemory;
        } else {
            out.appendSlice(allocator, raw) catch return error.OutOfMemory;
        }
        last_space = false;
    }
    if (out.items.len > 0 and out.items[out.items.len - 1] == ' ') {
        out.items.len -= 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn replaceLiteral(
    allocator: std.mem.Allocator,
    input: []const u8,
    needle: []const u8,
    repl: []const u8,
) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var start_at: usize = 0;
    while (std.mem.indexOfPos(u8, input, start_at, needle)) |at| {
        out.appendSlice(allocator, input[start_at..at]) catch return error.OutOfMemory;
        out.appendSlice(allocator, repl) catch return error.OutOfMemory;
        start_at = at + needle.len;
    }
    out.appendSlice(allocator, input[start_at..]) catch return error.OutOfMemory;
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn trimOwned(allocator: std.mem.Allocator, owned: []u8) Error![]u8 {
    const trimmed = trim(owned);
    if (trimmed.len == owned.len) {
        return owned;
    }
    const out = allocator.dupe(u8, trimmed) catch return error.OutOfMemory;
    allocator.free(owned);
    return out;
}

fn trim(value: []const u8) []const u8 {
    var start_at: usize = 0;
    var end_at = value.len;
    while (start_at < end_at and isAsciiTrim(value[start_at])) start_at += 1;
    while (end_at > start_at and isAsciiTrim(value[end_at - 1])) end_at -= 1;
    return value[start_at..end_at];
}

const XmlEntity = struct {
    len: usize,
    value: []const u8,
};

const SubTag = struct {
    end: usize,
    alias: ?[]const u8,
};

fn appendXmlDecoded(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    input: []const u8,
) Error!void {
    var index: usize = 0;
    while (index < input.len) {
        if (input[index] == '&') {
            if (xmlEntity(input, index)) |entity| {
                out.appendSlice(allocator, entity.value) catch return error.OutOfMemory;
                index += entity.len;
                continue;
            }
        }
        out.append(allocator, input[index]) catch return error.OutOfMemory;
        index += 1;
    }
}

fn xmlEntity(input: []const u8, index: usize) ?XmlEntity {
    if (starts(input, index, "&quot;")) return .{ .len = "&quot;".len, .value = "\"" };
    if (starts(input, index, "&lt;")) return .{ .len = "&lt;".len, .value = "<" };
    if (starts(input, index, "&gt;")) return .{ .len = "&gt;".len, .value = ">" };
    if (starts(input, index, "&apos;")) return .{ .len = "&apos;".len, .value = "'" };
    if (starts(input, index, "&amp;")) return .{ .len = "&amp;".len, .value = "&" };
    return null;
}

fn subTag(input: []const u8, index: usize) ?SubTag {
    if (index + 4 > input.len or input[index] != '<') return null;
    if (!eqlIgnore(input[index + 1 .. index + 4], "sub")) return null;
    const after_name = index + 4;
    if (after_name < input.len and isWord(input[after_name])) return null;
    const end = tagEnd(input, index) orelse return null;
    return .{ .end = end + 1, .alias = aliasAttr(input[after_name..end]) };
}

fn aliasAttr(attrs: []const u8) ?[]const u8 {
    var index: usize = 0;
    while (index + 5 <= attrs.len) : (index += 1) {
        if (!eqlIgnore(attrs[index .. index + 5], "alias")) continue;
        if (index > 0 and isWord(attrs[index - 1])) continue;
        var cursor = index + 5;
        if (cursor < attrs.len and isWord(attrs[cursor])) continue;
        while (cursor < attrs.len and isAsciiTrim(attrs[cursor])) cursor += 1;
        if (cursor >= attrs.len or attrs[cursor] != '=') continue;
        cursor += 1;
        while (cursor < attrs.len and isAsciiTrim(attrs[cursor])) cursor += 1;
        if (cursor >= attrs.len or attrs[cursor] != '"') continue;
        const start_at = cursor + 1;
        const end = std.mem.indexOfScalarPos(u8, attrs, start_at, '"') orelse return null;
        return attrs[start_at..end];
    }
    return null;
}

fn findSubClose(input: []const u8, start_at: usize) ?usize {
    var index = start_at;
    while (std.mem.indexOfScalarPos(u8, input, index, '<')) |tag_start| {
        if (tag_start + 6 <= input.len and eqlIgnore(input[tag_start .. tag_start + 6], "</sub>")) {
            return tag_start + 6;
        }
        index = tag_start + 1;
    }
    return null;
}

fn tagEnd(input: []const u8, start_at: usize) ?usize {
    return std.mem.indexOfScalarPos(u8, input, start_at + 1, '>');
}

fn eqlIgnore(left: []const u8, right: []const u8) bool {
    if (left.len != right.len) return false;
    for (left, right) |a, b| {
        if (lower(a) != lower(b)) return false;
    }
    return true;
}

fn appendYear(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: []const u8,
) Error!void {
    const year = parseU32(value[0..4]) orelse {
        out.appendSlice(allocator, value) catch return error.OutOfMemory;
        return;
    };
    if (year < 1100 or year % 1000 < 10) {
        out.appendSlice(allocator, value) catch return error.OutOfMemory;
        return;
    }
    const left = value[0..2];
    const right = parseU32(value[2..4]) orelse 0;
    const suffix = if (value.len == 5 and value[4] == 's') "s" else "";
    if (year % 1000 >= 100 and year % 1000 <= 999) {
        if (right == 0) {
            try appendFmt(allocator, out, "{s} hundred{s}", .{ left, suffix });
            return;
        }
        if (right < 10) {
            try appendFmt(allocator, out, "{s} oh {d}{s}", .{ left, right, suffix });
            return;
        }
    }
    try appendFmt(allocator, out, "{s} {d}{s}", .{ left, right, suffix });
}

fn appendTime(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: []const u8,
) Error!void {
    const colon = std.mem.indexOfScalar(u8, value, ':') orelse return error.InvalidInput;
    const hour = parseU32(value[0..colon]) orelse return error.InvalidInput;
    const minute = parseU32(value[colon + 1 ..]) orelse return error.InvalidInput;
    if (minute == 0) {
        try appendFmt(allocator, out, "{d} o'clock", .{hour});
    } else if (minute < 10) {
        try appendFmt(allocator, out, "{d} oh {d}", .{ hour, minute });
    } else {
        try appendFmt(allocator, out, "{d} {d}", .{ hour, minute });
    }
}

fn appendMoney(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: []const u8,
) Error!void {
    const pound = starts(value, 0, "£");
    const symbol_len: usize = if (pound) "£".len else 1;
    const bill = if (pound) "pound" else "dollar";
    const rest = value[symbol_len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.');
    if (dot == null) {
        try appendFmt(allocator, out, "{s} {s}{s}", .{ rest, bill, if (std.mem.eql(u8, rest, "1")) "" else "s" });
        return;
    }
    const at = dot.?;
    const whole = rest[0..at];
    const cents = centsValue(rest[at + 1 ..]);
    const coin = if (pound)
        if (cents == 1) "penny" else "pence"
    else if (cents == 1) "cent" else "cents";
    try appendFmt(
        allocator,
        out,
        "{s} {s}{s} and {d} {s}",
        .{ whole, bill, if (std.mem.eql(u8, whole, "1")) "" else "s", cents, coin },
    );
}

fn appendPoint(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    value: []const u8,
) Error!void {
    const dot = std.mem.indexOfScalar(u8, value, '.') orelse return error.InvalidInput;
    out.appendSlice(allocator, value[0..dot]) catch return error.OutOfMemory;
    out.appendSlice(allocator, " point") catch return error.OutOfMemory;
    for (value[dot + 1 ..]) |byte| {
        out.append(allocator, ' ') catch return error.OutOfMemory;
        out.append(allocator, byte) catch return error.OutOfMemory;
    }
}

fn appendFmt(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    comptime fmt: []const u8,
    args: anytype,
) Error!void {
    const text = std.fmt.allocPrint(allocator, fmt, args) catch return error.OutOfMemory;
    defer allocator.free(text);
    out.appendSlice(allocator, text) catch return error.OutOfMemory;
}

fn decimalLen(input: []const u8, index: usize) ?usize {
    var cursor = index;
    while (cursor < input.len and isDigit(input[cursor])) cursor += 1;
    if (cursor >= input.len or input[cursor] != '.') return null;
    cursor += 1;
    const start_digits = cursor;
    while (cursor < input.len and isDigit(input[cursor])) cursor += 1;
    return if (cursor > start_digits) cursor - index else null;
}

fn yearLen(input: []const u8, index: usize) ?usize {
    if (!wordStart(input, index) or index + 4 > input.len) return null;
    var cursor = index;
    var count: usize = 0;
    while (cursor < input.len and count < 4 and isDigit(input[cursor])) : ({
        cursor += 1;
        count += 1;
    }) {}
    if (count != 4) return null;
    if (cursor < input.len and input[cursor] == 's') cursor += 1;
    return if (wordEnd(input, cursor)) cursor - index else null;
}

fn timeLen(input: []const u8, index: usize) ?usize {
    if (!wordStart(input, index) or index >= input.len or !isDigit(input[index])) return null;
    var cursor = index + 1;
    if (input[index] == '1' and cursor < input.len and input[cursor] >= '0' and input[cursor] <= '2') {
        cursor += 1;
    }
    if (cursor >= input.len or input[cursor] != ':') return null;
    cursor += 1;
    if (cursor + 2 > input.len or input[cursor] > '5' or !isDigit(input[cursor + 1])) return null;
    cursor += 2;
    return if (wordEnd(input, cursor)) cursor - index else null;
}

fn moneyLen(input: []const u8, index: usize) ?usize {
    const symbol_len: usize = if (index < input.len and input[index] == '$')
        1
    else if (starts(input, index, "£"))
        "£".len
    else
        return null;
    var cursor = index + symbol_len;
    const digit_start = cursor;
    while (cursor < input.len and isDigit(input[cursor])) cursor += 1;
    if (cursor == digit_start) return null;
    if (cursor < input.len and input[cursor] == '.') {
        cursor += 1;
        const cents_start = cursor;
        while (cursor < input.len and isDigit(input[cursor])) cursor += 1;
        if (cursor == cents_start) return null;
    }
    while (true) {
        const suffix = moneySuffix(input, cursor) orelse break;
        cursor += suffix;
    }
    return if (wordEnd(input, cursor)) cursor - index else null;
}

fn moneySuffix(input: []const u8, index: usize) ?usize {
    const suffixes = [_][]const u8{ " hundred", " thousand", " billion", " million", " trillion" };
    for (suffixes) |suffix| {
        if (starts(input, index, suffix)) return suffix.len;
    }
    return null;
}

fn centsValue(value: []const u8) u32 {
    const first: u32 = if (value.len > 0 and isDigit(value[0])) value[0] - '0' else 0;
    const second: u32 = if (value.len > 1 and isDigit(value[1])) value[1] - '0' else 0;
    return first * 10 + second;
}

fn dottedAbbrLen(input: []const u8, index: usize) ?usize {
    var cursor = index;
    var pairs: usize = 0;
    while (cursor + 1 < input.len and isAsciiLetter(input[cursor]) and input[cursor + 1] == '.') {
        pairs += 1;
        cursor += 2;
    }
    if (pairs < 2 or cursor + 1 >= input.len or input[cursor] != ' ' or !isLower(input[cursor + 1])) {
        return null;
    }
    return cursor + 2 - index;
}

fn hundredPrefixLen(input: []const u8, index: usize) ?usize {
    const len: usize = if (index < input.len and input[index] >= 'a' and input[index] <= 'z')
        1
    else if (starts(input, index, "ɹ"))
        "ɹ".len
    else if (starts(input, index, "ː"))
        "ː".len
    else
        return null;
    return if (starts(input, index + len, "hˈʌndɹɪd")) len else null;
}

fn zBoundary(input: []const u8, index: usize) bool {
    if (index >= input.len) return true;
    return switch (input[index]) {
        ';', ':', ',', '.', '!', '?', ' ', '"' => true,
        else => starts(input, index, "¡") or
            starts(input, index, "¿") or
            starts(input, index, "—") or
            starts(input, index, "…") or
            starts(input, index, "«") or
            starts(input, index, "»") or
            starts(input, index, "“") or
            starts(input, index, "”"),
    };
}

fn yeahLen(input: []const u8, index: usize) ?usize {
    if (index >= input.len or (input[index] != 'y' and input[index] != 'Y')) return null;
    if (index + 3 <= input.len and lower(input[index + 1]) == 'e' and lower(input[index + 2]) == 'a' and wordEnd(input, index + 3)) {
        return 3;
    }
    if (index + 4 <= input.len and lower(input[index + 1]) == 'e' and lower(input[index + 2]) == 'a' and lower(input[index + 3]) == 'h' and wordEnd(input, index + 4)) {
        return 4;
    }
    return null;
}

fn upperAfter(input: []const u8, index: usize) bool {
    return index + 2 <= input.len and input[index] == ' ' and input[index + 1] >= 'A' and input[index + 1] <= 'Z';
}

fn parseU32(value: []const u8) ?u32 {
    if (value.len == 0) return null;
    var out: u32 = 0;
    for (value) |byte| {
        if (!isDigit(byte)) return null;
        out = std.math.mul(u32, out, 10) catch return null;
        out = std.math.add(u32, out, byte - '0') catch return null;
    }
    return out;
}

fn hasCjk(input: []const u8) bool {
    var index: usize = 0;
    while (index < input.len) {
        const rune = nextRune(input, index) catch return false;
        index += rune.len;
        if ((rune.codepoint >= 0x3400 and rune.codepoint <= 0x4dbf) or
            (rune.codepoint >= 0x4e00 and rune.codepoint <= 0x9fff) or
            (rune.codepoint >= 0xf900 and rune.codepoint <= 0xfaff))
        {
            return true;
        }
    }
    return false;
}

fn trimPinyinPunct(input: []const u8) []const u8 {
    var start_at: usize = 0;
    var end_at = input.len;
    while (start_at < end_at) {
        const len = punctStartLen(input[start_at..end_at]) orelse break;
        start_at += len;
    }
    while (end_at > start_at) {
        const len = punctEndLen(input[start_at..end_at]) orelse break;
        end_at -= len;
    }
    return input[start_at..end_at];
}

fn validPinyinToken(input: []const u8) bool {
    var index: usize = 0;
    var letters: usize = 0;
    while (index < input.len) {
        const rune = nextRune(input, index) catch return false;
        index += rune.len;
        if (index == input.len and rune.codepoint >= '1' and rune.codepoint <= '5') {
            return letters > 0;
        }
        if (isPinyinLetter(rune.codepoint)) {
            letters += 1;
            continue;
        }
        return false;
    }
    return false;
}

fn isPinyinLetter(codepoint: u21) bool {
    return (codepoint >= 'a' and codepoint <= 'z') or
        (codepoint >= 'A' and codepoint <= 'Z') or
        codepoint == 0x00fc or codepoint == 0x00dc or
        codepoint == 'v' or codepoint == 'V';
}

fn punctStartLen(input: []const u8) ?usize {
    if (input.len == 0) return null;
    return switch (input[0]) {
        '.', ',', '!', '?', ':', ';', '"', '(', ')', '[', ']', '{', '}' => 1,
        else => if (starts(input, 0, "“"))
            "“".len
        else if (starts(input, 0, "”"))
            "”".len
        else if (starts(input, 0, "«"))
            "«".len
        else if (starts(input, 0, "»"))
            "»".len
        else
            null,
    };
}

fn punctEndLen(input: []const u8) ?usize {
    if (input.len == 0) return null;
    const last = input[input.len - 1];
    if (last == '.' or last == ',' or last == '!' or last == '?' or last == ':' or
        last == ';' or last == '"' or last == '(' or last == ')' or last == '[' or
        last == ']' or last == '{' or last == '}')
    {
        return 1;
    }
    if (std.mem.endsWith(u8, input, "“")) return "“".len;
    if (std.mem.endsWith(u8, input, "”")) return "”".len;
    if (std.mem.endsWith(u8, input, "«")) return "«".len;
    if (std.mem.endsWith(u8, input, "»")) return "»".len;
    return null;
}

fn nextRune(input: []const u8, index: usize) Error!Rune {
    const first = input[index];
    if (first < 0x80) return .{ .codepoint = @intCast(first), .len = 1 };
    if ((first & 0xe0) == 0xc0) {
        if (index + 1 >= input.len or !isCont(input[index + 1])) return error.InvalidUtf8;
        const value: u32 = (@as(u32, first & 0x1f) << 6) | @as(u32, input[index + 1] & 0x3f);
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

fn isSpace(codepoint: u21) bool {
    return switch (@as(u32, codepoint)) {
        0x09...0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn wordStart(input: []const u8, index: usize) bool {
    return index < input.len and (index == 0 or !isWord(input[index - 1])) and isWord(input[index]);
}

fn wordEnd(input: []const u8, index: usize) bool {
    return index >= input.len or !isWord(input[index]);
}

fn starts(input: []const u8, index: usize, needle: []const u8) bool {
    return index <= input.len and needle.len <= input.len - index and std.mem.eql(u8, input[index .. index + needle.len], needle);
}

fn startsIgnore(input: []const u8, index: usize, needle: []const u8) bool {
    if (index > input.len or needle.len > input.len - index) return false;
    for (input[index .. index + needle.len], needle) |a, b| {
        if (lower(a) != lower(b)) return false;
    }
    return true;
}

fn isWord(byte: u8) bool {
    return isAsciiLetter(byte) or isDigit(byte) or byte == '_';
}

fn isAsciiLetter(byte: u8) bool {
    return (byte >= 'a' and byte <= 'z') or (byte >= 'A' and byte <= 'Z');
}

fn isLower(byte: u8) bool {
    return byte >= 'a' and byte <= 'z';
}

fn isUpperConsonant(byte: u8) bool {
    return switch (byte) {
        'B', 'C', 'D', 'F', 'G', 'H', 'J'...'N', 'P'...'T', 'V'...'Z' => true,
        else => false,
    };
}

fn isDigit(byte: u8) bool {
    return byte >= '0' and byte <= '9';
}

fn isSpaceByte(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r';
}

fn isCont(byte: u8) bool {
    return (byte & 0xc0) == 0x80;
}

fn lower(byte: u8) u8 {
    return if (byte >= 'A' and byte <= 'Z') byte + 32 else byte;
}

fn isAsciiTrim(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r';
}

test "phoneme normalizes english text" {
    const out = try normText(std.testing.allocator, "Dr. Smith paid $1.50 at 12:05 in 2026", "en-us");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("Doctor Smith paid 1 dollar and 50 cents at 12 oh 5 in 20 26", out);
}

test "phoneme post-processes english output" {
    const out = try post(std.testing.allocator, "kəkˈoːɹoʊ nˈaɪnti ahˈʌndɹɪd z! r x ɬ ʲ", "en-us");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("kˈoʊkəɹoʊ nˈaɪndi a hˈʌndɹɪdz! ɹ k l j", out);
}

test "phoneme extracts ssml plain text and explicit tags" {
    const plain = try plainSsml(std.testing.allocator, "<speak>A <sub alias=\"one &amp; two\">x</sub><break/> B</speak>");
    defer std.testing.allocator.free(plain);
    try std.testing.expectEqualStrings("A one & two B", plain);

    const explicit = try explicitPhonemes(std.testing.allocator, " 'liːd \n ɹə'kɔːɹd ");
    defer std.testing.allocator.free(explicit);
    try std.testing.expectEqualStrings("ˈliːd ɹəˈkɔːɹd", explicit);
}

test "phoneme detects and normalizes pinyin" {
    try std.testing.expect(looksPinyin("hang2 xing2"));
    try std.testing.expect(looksPinyin("“lü4|x”"));
    try std.testing.expect(!looksPinyin("hello"));
    try std.testing.expect(!looksPinyin("银 hang2"));

    const out = try pinyinNorm(std.testing.allocator, " lü4  Ü5 ");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("lv4 V5", out);
}
