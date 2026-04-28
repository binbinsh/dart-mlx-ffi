const std = @import("std");

pub const Error = error{
    InvalidUtf8,
    LengthOverflow,
    OutOfMemory,
};

pub const Item = struct {
    text: []u8,
    language: []const u8,
};

pub const List = struct {
    items: []Item = &.{},

    pub fn deinit(self: *List, allocator: std.mem.Allocator) void {
        for (self.items) |item| {
            allocator.free(item.text);
        }
        if (self.items.len > 0) allocator.free(self.items);
        self.* = .{};
    }
};

pub const Route = struct {
    language: []const u8,
    mixed: bool,
};

const Rune = struct {
    codepoint: u21,
    len: usize,
};

const Script = enum {
    cjk,
    latin,
};

const Builder = struct {
    allocator: std.mem.Allocator,
    default_language: []const u8,
    items: std.ArrayList(Item) = .empty,
    run: std.ArrayList(u8) = .empty,
    run_script: ?Script = null,
    last_flush_had_trailing_space: bool = false,

    fn deinit(self: *Builder) void {
        for (self.items.items) |item| {
            self.allocator.free(item.text);
        }
        self.items.deinit(self.allocator);
        self.run.deinit(self.allocator);
    }

    fn literal(self: *Builder, text: []const u8) Error!void {
        if (text.len == 0) return;
        const owned = self.allocator.dupe(u8, text) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned);
        self.items.append(self.allocator, .{ .text = owned, .language = "" }) catch return error.OutOfMemory;
    }

    fn flush(self: *Builder) Error!void {
        const script = self.run_script orelse {
            self.last_flush_had_trailing_space = false;
            return;
        };
        if (self.run.items.len == 0) {
            self.last_flush_had_trailing_space = false;
            self.run_script = null;
            return;
        }

        const run_text = self.run.items;
        const leading = isSpaceByte(run_text[0]);
        const trailing = isSpaceByte(run_text[run_text.len - 1]);
        if (leading and self.items.items.len > 0) {
            try self.literal(" ");
        }

        const owned = self.allocator.dupe(u8, run_text) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned);
        self.items.append(self.allocator, .{
            .text = owned,
            .language = switch (script) {
                .cjk => "cmn",
                .latin => if (std.mem.eql(u8, self.default_language, "en")) "en" else "en-us",
            },
        }) catch return error.OutOfMemory;

        if (trailing) {
            try self.literal(" ");
        }
        self.last_flush_had_trailing_space = trailing;
        self.run.clearRetainingCapacity();
        self.run_script = null;
    }

    fn finish(self: *Builder) Error!List {
        try self.flush();
        const owned = self.items.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
        self.items = .empty;
        self.run.deinit(self.allocator);
        self.run = .empty;
        return .{ .items = owned };
    }
};

pub fn split(
    allocator: std.mem.Allocator,
    input: []const u8,
    default_language: []const u8,
) Error!List {
    var builder = Builder{
        .allocator = allocator,
        .default_language = default_language,
    };
    errdefer builder.deinit();

    var index: usize = 0;
    while (index < input.len) {
        const rune = try nextRune(input, index);
        const bytes = input[index .. index + rune.len];
        index += rune.len;

        const script = scriptOf(rune.codepoint);
        if (script == null) {
            if (builder.run_script == null) {
                try builder.literal(bytes);
            } else {
                builder.run.appendSlice(allocator, bytes) catch return error.OutOfMemory;
            }
            continue;
        }

        if (builder.run_script != script.?) {
            const had_run = builder.run_script != null;
            try builder.flush();
            if (had_run and !builder.last_flush_had_trailing_space and builder.items.items.len > 0) {
                try builder.literal(" ");
            }
            builder.run_script = script;
        }
        builder.run.appendSlice(allocator, bytes) catch return error.OutOfMemory;
    }
    return builder.finish();
}

pub fn route(input: []const u8, requested: []const u8) Error!Route {
    var has_cjk = false;
    var has_latin = false;
    var index: usize = 0;
    while (index < input.len) {
        const rune = try nextRune(input, index);
        index += rune.len;
        switch (scriptOf(rune.codepoint) orelse continue) {
            .cjk => has_cjk = true,
            .latin => has_latin = true,
        }
    }

    if (has_latin and !has_cjk and std.mem.eql(u8, requested, "cmn")) {
        return .{ .language = "en-us", .mixed = false };
    }
    if (has_cjk and !has_latin and
        (std.mem.eql(u8, requested, "en-us") or std.mem.eql(u8, requested, "en")))
    {
        return .{ .language = "cmn", .mixed = false };
    }
    return .{ .language = requested, .mixed = has_cjk and has_latin };
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidUtf8 => "Zig Kokoro mixed-script splitter received invalid UTF-8.",
        error.LengthOverflow => "Zig Kokoro mixed-script splitter output length overflowed.",
        error.OutOfMemory => "Zig Kokoro mixed-script splitter ran out of memory.",
    };
}

fn scriptOf(codepoint: u21) ?Script {
    if ((codepoint >= 0x3400 and codepoint <= 0x4dbf) or
        (codepoint >= 0x4e00 and codepoint <= 0x9fff) or
        (codepoint >= 0xf900 and codepoint <= 0xfaff))
    {
        return .cjk;
    }
    if ((codepoint >= 'A' and codepoint <= 'Z') or
        (codepoint >= 'a' and codepoint <= 'z'))
    {
        return .latin;
    }
    return null;
}

fn nextRune(input: []const u8, index: usize) Error!Rune {
    const first = input[index];
    if (first < 0x80) return .{ .codepoint = @intCast(first), .len = 1 };
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

fn isSpaceByte(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r';
}

test "phoneme run splitter preserves mixed-script spacing" {
    var list = try split(std.testing.allocator, "Hello 你好 Moon", "cmn");
    defer list.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 5), list.items.len);
    try std.testing.expectEqualStrings("Hello ", list.items[0].text);
    try std.testing.expectEqualStrings("en-us", list.items[0].language);
    try std.testing.expectEqualStrings(" ", list.items[1].text);
    try std.testing.expectEqualStrings("", list.items[1].language);
    try std.testing.expectEqualStrings("你好 ", list.items[2].text);
    try std.testing.expectEqualStrings("cmn", list.items[2].language);
    try std.testing.expectEqualStrings("Moon", list.items[4].text);
    try std.testing.expectEqualStrings("en-us", list.items[4].language);
}

test "phoneme run splitter inserts script boundary space" {
    var list = try split(std.testing.allocator, "Pay $一", "cmn");
    defer list.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), list.items.len);
    try std.testing.expectEqualStrings("Pay $", list.items[0].text);
    try std.testing.expectEqualStrings("en-us", list.items[0].language);
    try std.testing.expectEqualStrings(" ", list.items[1].text);
    try std.testing.expectEqualStrings("", list.items[1].language);
    try std.testing.expectEqualStrings("一", list.items[2].text);
    try std.testing.expectEqualStrings("cmn", list.items[2].language);
}

test "phoneme route selects language and mixed mode" {
    const latin = try route("and", "cmn");
    try std.testing.expectEqualStrings("en-us", latin.language);
    try std.testing.expect(!latin.mixed);

    const cjk = try route("你好", "en-us");
    try std.testing.expectEqualStrings("cmn", cjk.language);
    try std.testing.expect(!cjk.mixed);

    const mixed = try route("Hello 你好", "cmn");
    try std.testing.expectEqualStrings("cmn", mixed.language);
    try std.testing.expect(mixed.mixed);
}
