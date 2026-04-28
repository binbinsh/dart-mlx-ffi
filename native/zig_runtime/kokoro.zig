const std = @import("std");
const fill = @import("fill.zig");

pub const Error = error{
    InvalidArchive,
    InvalidConfig,
    InvalidInput,
    InvalidNpy,
    InvalidUtf8,
    LengthOverflow,
    OutOfMemory,
    UnsupportedCompression,
};

pub const Plan = struct {
    tokens: []i64 = &.{},
    lengths: []isize = &.{},
    text: []u8 = &.{},
    starts: []isize = &.{},
    byte_lengths: []isize = &.{},

    pub fn deinit(self: *Plan, allocator: std.mem.Allocator) void {
        if (self.tokens.len > 0) allocator.free(self.tokens);
        if (self.lengths.len > 0) allocator.free(self.lengths);
        if (self.text.len > 0) allocator.free(self.text);
        if (self.starts.len > 0) allocator.free(self.starts);
        if (self.byte_lengths.len > 0) allocator.free(self.byte_lengths);
        self.* = .{};
    }
};

pub const Npy = struct {
    shape: []i64 = &.{},
    data: []u8 = &.{},

    pub fn deinit(self: *Npy, allocator: std.mem.Allocator) void {
        if (self.shape.len > 0) allocator.free(self.shape);
        if (self.data.len > 0) allocator.free(self.data);
        self.* = .{};
    }
};

pub const NpzItem = struct {
    name: []u8 = &.{},
    array: Npy = .{},

    pub fn deinit(self: *NpzItem, allocator: std.mem.Allocator) void {
        if (self.name.len > 0) allocator.free(self.name);
        self.array.deinit(allocator);
        self.* = .{};
    }
};

pub const Npz = struct {
    items: []NpzItem = &.{},

    pub fn deinit(self: *Npz, allocator: std.mem.Allocator) void {
        for (self.items) |*item| item.deinit(allocator);
        if (self.items.len > 0) allocator.free(self.items);
        self.* = .{};
    }
};

pub const Vocab = struct {
    codes: []i32 = &.{},
    ids: []i64 = &.{},

    pub fn deinit(self: *Vocab, allocator: std.mem.Allocator) void {
        if (self.codes.len > 0) allocator.free(self.codes);
        if (self.ids.len > 0) allocator.free(self.ids);
        self.* = .{};
    }
};

const Entry = struct {
    codepoint: u21,
    token: i64,
    bytes: [4]u8,
    byte_len: u8,
};

const Rune = struct {
    codepoint: u21,
    byte_len: usize,
};

const ZipData = struct {
    bytes: []const u8,
    owned: []u8 = &.{},

    fn deinit(self: *ZipData, allocator: std.mem.Allocator) void {
        if (self.owned.len > 0) allocator.free(self.owned);
        self.* = .{ .bytes = &.{} };
    }
};

const EndRecord = struct {
    count: usize,
    cd_offset: usize,
    cd_size: usize,
};

const VocabView = struct {
    codes: []const i32,
    ids: []const i64,

    fn id(self: VocabView, codepoint: u21) ?i64 {
        const target: i32 = @intCast(@as(u32, codepoint));
        for (self.codes, 0..) |code, index| {
            if (code == target) {
                return self.ids[index];
            }
        }
        return null;
    }

    fn has(self: VocabView, codepoint: u21) bool {
        return self.id(codepoint) != null;
    }
};

const CodeSet = struct {
    codes: []const i32,

    fn has(self: CodeSet, codepoint: u21) bool {
        const target: i32 = @intCast(@as(u32, codepoint));
        for (self.codes) |code| {
            if (code == target) {
                return true;
            }
        }
        return false;
    }
};

const Builder = struct {
    allocator: std.mem.Allocator,
    include_text: bool,
    current: std.ArrayList(Entry) = .empty,
    current_count: usize = 0,
    tokens: std.ArrayList(i64) = .empty,
    lengths: std.ArrayList(isize) = .empty,
    text: std.ArrayList(u8) = .empty,
    starts: std.ArrayList(isize) = .empty,
    byte_lengths: std.ArrayList(isize) = .empty,

    fn deinit(self: *Builder) void {
        self.current.deinit(self.allocator);
        self.tokens.deinit(self.allocator);
        self.lengths.deinit(self.allocator);
        self.text.deinit(self.allocator);
        self.starts.deinit(self.allocator);
        self.byte_lengths.deinit(self.allocator);
        self.* = .{ .allocator = self.allocator, .include_text = self.include_text };
    }

    fn append(self: *Builder, entry: Entry) Error!void {
        self.current.append(self.allocator, entry) catch return error.OutOfMemory;
        self.current_count = std.math.add(usize, self.current_count, 1) catch return error.LengthOverflow;
    }

    fn appendSlice(self: *Builder, entries: []const Entry) Error!void {
        for (entries) |entry| {
            try self.append(entry);
        }
    }

    fn flush(self: *Builder) Error!void {
        var start: usize = 0;
        var end = self.current.items.len;
        while (start < end and isSpace(self.current.items[start].codepoint)) {
            start += 1;
        }
        while (end > start and isSpace(self.current.items[end - 1].codepoint)) {
            end -= 1;
        }
        if (start == end) {
            self.current.clearRetainingCapacity();
            self.current_count = 0;
            return;
        }

        const count = end - start;
        self.lengths.append(self.allocator, try intSize(count)) catch return error.OutOfMemory;
        for (self.current.items[start..end]) |entry| {
            self.tokens.append(self.allocator, entry.token) catch return error.OutOfMemory;
        }

        if (self.include_text) {
            const text_start = self.text.items.len;
            self.starts.append(self.allocator, try intSize(text_start)) catch return error.OutOfMemory;
            var byte_count: usize = 0;
            for (self.current.items[start..end]) |entry| {
                const len: usize = entry.byte_len;
                self.text.appendSlice(self.allocator, entry.bytes[0..len]) catch return error.OutOfMemory;
                byte_count = std.math.add(usize, byte_count, len) catch return error.LengthOverflow;
            }
            self.byte_lengths.append(self.allocator, try intSize(byte_count)) catch return error.OutOfMemory;
        }

        self.current.clearRetainingCapacity();
        self.current_count = 0;
    }

    fn finish(self: *Builder) Error!Plan {
        try self.flush();

        const tokens = self.tokens.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
        self.tokens = .empty;
        errdefer if (tokens.len > 0) self.allocator.free(tokens);

        const lengths = self.lengths.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
        self.lengths = .empty;
        errdefer if (lengths.len > 0) self.allocator.free(lengths);

        var text: []u8 = &.{};
        var starts: []isize = &.{};
        var byte_lengths: []isize = &.{};
        if (self.include_text) {
            text = self.text.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
            self.text = .empty;
            errdefer if (text.len > 0) self.allocator.free(text);

            starts = self.starts.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
            self.starts = .empty;
            errdefer if (starts.len > 0) self.allocator.free(starts);

            byte_lengths = self.byte_lengths.toOwnedSlice(self.allocator) catch return error.OutOfMemory;
            self.byte_lengths = .empty;
            errdefer if (byte_lengths.len > 0) self.allocator.free(byte_lengths);
        }

        return .{
            .tokens = tokens,
            .lengths = lengths,
            .text = text,
            .starts = starts,
            .byte_lengths = byte_lengths,
        };
    }
};

pub fn filter(
    allocator: std.mem.Allocator,
    input: []const u8,
    codes: []const i32,
) Error![]u8 {
    const set = CodeSet{ .codes = codes };
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var index: usize = 0;
    var last_was_space = true;
    while (index < input.len) {
        const rune = try nextRune(input, index);
        index += rune.byte_len;
        if (!set.has(rune.codepoint)) {
            continue;
        }
        if (isSpace(rune.codepoint)) {
            if (!last_was_space and out.items.len > 0) {
                out.append(allocator, ' ') catch return error.OutOfMemory;
            }
            last_was_space = true;
            continue;
        }
        var bytes: [4]u8 = undefined;
        const len = writeUtf8(&bytes, rune.codepoint);
        out.appendSlice(allocator, bytes[0..len]) catch return error.OutOfMemory;
        last_was_space = false;
    }
    if (out.items.len > 0 and out.items[out.items.len - 1] == ' ') {
        out.items.len -= 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

pub fn clean(allocator: std.mem.Allocator, input: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var index: usize = 0;
    var last_was_space = true;
    while (index < input.len) {
        if (voiceMarkerLen(input, index)) |marker_len| {
            try appendSpace(allocator, &out, &last_was_space);
            index += marker_len;
            continue;
        }

        const rune = try nextRune(input, index);
        index += rune.byte_len;
        if (rune.codepoint == 0x200d or isAsciiDigit(rune.codepoint)) {
            continue;
        }
        if (isSpace(rune.codepoint)) {
            try appendSpace(allocator, &out, &last_was_space);
            continue;
        }

        var bytes: [4]u8 = undefined;
        const len = writeUtf8(&bytes, rune.codepoint);
        out.appendSlice(allocator, bytes[0..len]) catch return error.OutOfMemory;
        last_was_space = false;
    }
    if (out.items.len > 0 and out.items[out.items.len - 1] == ' ') {
        out.items.len -= 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

pub fn plan(
    allocator: std.mem.Allocator,
    input: []const u8,
    codes: []const i32,
    ids: []const i64,
    max_tokens: usize,
    include_text: bool,
) Error!Plan {
    if (codes.len != ids.len or max_tokens == 0) {
        return error.InvalidInput;
    }
    const vocab = VocabView{ .codes = codes, .ids = ids };
    const filtered = try filterEntries(allocator, input, vocab);
    defer if (filtered.len > 0) allocator.free(filtered);

    var builder = Builder{ .allocator = allocator, .include_text = include_text };
    defer builder.deinit();

    var segment_start: usize = 0;
    var index: usize = 0;
    while (index < filtered.len) : (index += 1) {
        const entry = filtered[index];
        if (isSpace(entry.codepoint) or isBreak(entry.codepoint)) {
            try appendSegment(&builder, filtered[segment_start .. index + 1], max_tokens);
            segment_start = index + 1;
        }
    }
    if (segment_start < filtered.len) {
        try appendSegment(&builder, filtered[segment_start..], max_tokens);
    }

    return builder.finish();
}

pub fn row(
    style: []f32,
    voice: []const f32,
    voice_rows: usize,
    voice_row_len: usize,
    index: usize,
) Error!void {
    if (voice_rows == 0 or voice_row_len == 0 or voice_row_len > style.len) {
        return error.InvalidInput;
    }
    const voice_total = std.math.mul(usize, voice_rows, voice_row_len) catch return error.LengthOverflow;
    if (voice_total > voice.len) {
        return error.InvalidInput;
    }
    const safe_index = @min(index, voice_rows - 1);
    const row_offset = std.math.mul(usize, safe_index, voice_row_len) catch return error.LengthOverflow;
    try fill.f32s(style, 0, voice, row_offset, voice_row_len);
}

pub fn inputs(
    input_ids: []i64,
    token_ids: []const i64,
    style: []f32,
    voice: []const f32,
    voice_rows: usize,
    voice_row_len: usize,
    speed_out: []f32,
    speed: f32,
) Error!void {
    if (speed_out.len == 0) {
        return error.InvalidInput;
    }
    try fill.frame(input_ids, token_ids, 0);
    try row(style, voice, voice_rows, voice_row_len, token_ids.len);
    speed_out[0] = speed;
}

pub fn loadNpz(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) Error!Npz {
    const max_npz_bytes = 2 * 1024 * 1024 * 1024;
    const bytes = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(max_npz_bytes),
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidArchive,
    };
    defer allocator.free(bytes);
    return parseNpz(allocator, bytes);
}

pub fn loadVocab(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) Error!Vocab {
    const max_config_bytes = 16 * 1024 * 1024;
    const bytes = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(max_config_bytes),
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidConfig,
    };
    defer allocator.free(bytes);
    return parseVocab(allocator, bytes);
}

pub fn parseVocab(allocator: std.mem.Allocator, bytes: []const u8) Error!Vocab {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        bytes,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return error.InvalidConfig;
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |object| object,
        else => return error.InvalidConfig,
    };
    const vocab = switch (root.get("vocab") orelse return error.InvalidConfig) {
        .object => |object| object,
        else => return error.InvalidConfig,
    };

    const codes = allocator.alloc(i32, vocab.count()) catch return error.OutOfMemory;
    errdefer allocator.free(codes);
    const ids = allocator.alloc(i64, vocab.count()) catch return error.OutOfMemory;
    errdefer allocator.free(ids);

    var iterator = vocab.iterator();
    var index: usize = 0;
    while (iterator.next()) |entry| {
        const key = entry.key_ptr.*;
        const rune = try singleRune(key);
        codes[index] = @intCast(@as(u32, rune.codepoint));
        ids[index] = switch (entry.value_ptr.*) {
            .integer => |value| value,
            else => return error.InvalidConfig,
        };
        index += 1;
    }
    return .{ .codes = codes, .ids = ids };
}

pub fn parseNpz(allocator: std.mem.Allocator, bytes: []const u8) Error!Npz {
    const end = try findEndRecord(bytes);
    if (end.cd_offset > bytes.len or end.cd_size > bytes.len - end.cd_offset) {
        return error.InvalidArchive;
    }

    var items: std.ArrayList(NpzItem) = .empty;
    errdefer {
        for (items.items) |*item| item.deinit(allocator);
        items.deinit(allocator);
    }

    var offset = end.cd_offset;
    var index: usize = 0;
    while (index < end.count) : (index += 1) {
        if (offset > bytes.len or 46 > bytes.len - offset) {
            return error.InvalidArchive;
        }
        const header = bytes[offset .. offset + 46];
        if (!std.mem.eql(u8, header[0..4], "PK\x01\x02")) {
            return error.InvalidArchive;
        }

        const flags = readU16(header[8..10]);
        const method = readU16(header[10..12]);
        const comp_size = try usizeFromU32(readU32(header[20..24]));
        const uncomp_size = try usizeFromU32(readU32(header[24..28]));
        const name_len = @as(usize, readU16(header[28..30]));
        const extra_len = @as(usize, readU16(header[30..32]));
        const comment_len = @as(usize, readU16(header[32..34]));
        const local_offset = try usizeFromU32(readU32(header[42..46]));
        const record_len = try checkedAdd(46, try checkedAdd(name_len, try checkedAdd(extra_len, comment_len)));
        if (record_len > bytes.len - offset) {
            return error.InvalidArchive;
        }

        const name = bytes[offset + 46 .. offset + 46 + name_len];
        offset += record_len;

        if ((flags & 0x1) != 0 or !std.mem.endsWith(u8, name, ".npy")) {
            continue;
        }
        var payload = try extractZipData(
            allocator,
            bytes,
            local_offset,
            comp_size,
            uncomp_size,
            method,
        );
        defer payload.deinit(allocator);

        var array = try parseNpy(allocator, payload.bytes);
        errdefer array.deinit(allocator);

        const base_name = name[0 .. name.len - 4];
        const owned_name = allocator.dupe(u8, base_name) catch return error.OutOfMemory;
        errdefer allocator.free(owned_name);

        items.append(allocator, .{ .name = owned_name, .array = array }) catch return error.OutOfMemory;
        array = .{};
    }

    if (items.items.len == 0) {
        return error.InvalidArchive;
    }
    const owned = items.toOwnedSlice(allocator) catch return error.OutOfMemory;
    items = .empty;
    return .{ .items = owned };
}

pub fn parseNpy(allocator: std.mem.Allocator, bytes: []const u8) Error!Npy {
    if (bytes.len < 10 or bytes[0] != 0x93 or !std.mem.eql(u8, bytes[1..6], "NUMPY")) {
        return error.InvalidNpy;
    }
    const major = bytes[6];
    const header_len: usize = switch (major) {
        1 => @as(usize, readU16(bytes[8..10])),
        2, 3 => blk: {
            if (bytes.len < 12) return error.InvalidNpy;
            break :blk try usizeFromU32(readU32(bytes[8..12]));
        },
        else => return error.InvalidNpy,
    };
    const header_start: usize = if (major == 1) 10 else 12;
    if (header_start > bytes.len or header_len > bytes.len - header_start) {
        return error.InvalidNpy;
    }
    const header = bytes[header_start .. header_start + header_len];
    if (!hasF32Descr(header)) {
        return error.InvalidNpy;
    }
    if (hasFortranOrder(header)) {
        return error.InvalidNpy;
    }

    const shape = try parseShape(allocator, header);
    errdefer if (shape.len > 0) allocator.free(shape);

    const values = try shapeSize(shape);
    const byte_count = std.math.mul(usize, values, @sizeOf(f32)) catch return error.LengthOverflow;
    const data_start = header_start + header_len;
    if (data_start > bytes.len or byte_count > bytes.len - data_start) {
        return error.InvalidNpy;
    }
    const data = allocator.alloc(u8, byte_count) catch return error.OutOfMemory;
    errdefer if (data.len > 0) allocator.free(data);
    @memcpy(data, bytes[data_start .. data_start + byte_count]);

    return .{ .shape = shape, .data = data };
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidArchive => "Zig Kokoro received an invalid or unreadable npz archive.",
        error.InvalidConfig => "Zig Kokoro received an invalid config.json vocab.",
        error.InvalidInput => "Zig Kokoro received invalid phoneme or vocab input.",
        error.InvalidNpy => "Zig Kokoro supports only little-endian C-order float32 npy arrays.",
        error.InvalidUtf8 => "Zig Kokoro received invalid UTF-8 phonemes.",
        error.LengthOverflow => "Zig Kokoro phoneme output length overflowed.",
        error.OutOfMemory => "Zig Kokoro ran out of memory.",
        error.UnsupportedCompression => "Zig Kokoro npz loader supports only stored and deflated entries.",
    };
}

fn findEndRecord(bytes: []const u8) Error!EndRecord {
    const sig = "PK\x05\x06";
    const pos = std.mem.lastIndexOf(u8, bytes, sig) orelse return error.InvalidArchive;
    if (22 > bytes.len - pos) {
        return error.InvalidArchive;
    }
    const record = bytes[pos .. pos + 22];
    const disk = readU16(record[4..6]);
    const cd_disk = readU16(record[6..8]);
    const count_disk = readU16(record[8..10]);
    const count = readU16(record[10..12]);
    const cd_size = try usizeFromU32(readU32(record[12..16]));
    const cd_offset = try usizeFromU32(readU32(record[16..20]));
    const comment_len = @as(usize, readU16(record[20..22]));
    if (disk != 0 or cd_disk != 0 or count_disk != count or comment_len > bytes.len - pos - 22) {
        return error.InvalidArchive;
    }
    return .{ .count = count, .cd_offset = cd_offset, .cd_size = cd_size };
}

fn singleRune(input: []const u8) Error!Rune {
    if (input.len == 0) {
        return error.InvalidConfig;
    }
    const rune = try nextRune(input, 0);
    if (rune.byte_len != input.len) {
        return error.InvalidConfig;
    }
    return rune;
}

fn extractZipData(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    local_offset: usize,
    comp_size: usize,
    uncomp_size: usize,
    method: u16,
) Error!ZipData {
    if (local_offset > bytes.len or 30 > bytes.len - local_offset) {
        return error.InvalidArchive;
    }
    const header = bytes[local_offset .. local_offset + 30];
    if (!std.mem.eql(u8, header[0..4], "PK\x03\x04")) {
        return error.InvalidArchive;
    }
    const name_len = @as(usize, readU16(header[26..28]));
    const extra_len = @as(usize, readU16(header[28..30]));
    const data_offset = try checkedAdd(local_offset, try checkedAdd(30, try checkedAdd(name_len, extra_len)));
    if (data_offset > bytes.len or comp_size > bytes.len - data_offset) {
        return error.InvalidArchive;
    }
    const compressed = bytes[data_offset .. data_offset + comp_size];
    return switch (method) {
        0 => if (compressed.len == uncomp_size)
            .{ .bytes = compressed }
        else
            error.InvalidArchive,
        8 => inflateRaw(allocator, compressed, uncomp_size),
        else => error.UnsupportedCompression,
    };
}

fn inflateRaw(
    allocator: std.mem.Allocator,
    compressed: []const u8,
    uncomp_size: usize,
) Error!ZipData {
    const out = allocator.alloc(u8, uncomp_size) catch return error.OutOfMemory;
    errdefer if (out.len > 0) allocator.free(out);

    var reader = std.Io.Reader.fixed(compressed);
    var flate_buffer: [std.compress.flate.max_window_len]u8 = undefined;
    var decompress: std.compress.flate.Decompress = .init(&reader, .raw, &flate_buffer);
    var writer = std.Io.Writer.fixed(out);
    decompress.reader.streamExact(&writer, out.len) catch return error.InvalidArchive;
    return .{ .bytes = out, .owned = out };
}

fn hasF32Descr(header: []const u8) bool {
    return std.mem.indexOf(u8, header, "'descr': '<f4'") != null or
        std.mem.indexOf(u8, header, "'descr':'<f4'") != null or
        std.mem.indexOf(u8, header, "\"descr\": \"<f4\"") != null or
        std.mem.indexOf(u8, header, "\"descr\":\"<f4\"") != null;
}

fn hasFortranOrder(header: []const u8) bool {
    return std.mem.indexOf(u8, header, "'fortran_order': True") != null or
        std.mem.indexOf(u8, header, "'fortran_order':True") != null or
        std.mem.indexOf(u8, header, "\"fortran_order\": true") != null or
        std.mem.indexOf(u8, header, "\"fortran_order\":true") != null;
}

fn parseShape(allocator: std.mem.Allocator, header: []const u8) Error![]i64 {
    const text = shapeText(header, "'shape'", '(', ')') orelse
        shapeText(header, "\"shape\"", '[', ']') orelse
        return error.InvalidNpy;

    var out: std.ArrayList(i64) = .empty;
    errdefer out.deinit(allocator);

    var iter = std.mem.splitScalar(u8, text, ',');
    while (iter.next()) |part_raw| {
        const part = trimAscii(part_raw);
        if (part.len == 0) {
            continue;
        }
        out.append(allocator, try parseDim(part)) catch return error.OutOfMemory;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn shapeText(header: []const u8, key: []const u8, open: u8, close: u8) ?[]const u8 {
    const key_at = std.mem.indexOf(u8, header, key) orelse return null;
    const rest = header[key_at + key.len ..];
    const open_at = std.mem.indexOfScalar(u8, rest, open) orelse return null;
    const after_open = rest[open_at + 1 ..];
    const close_at = std.mem.indexOfScalar(u8, after_open, close) orelse return null;
    return after_open[0..close_at];
}

fn parseDim(text: []const u8) Error!i64 {
    var value: u64 = 0;
    for (text) |byte| {
        if (byte < '0' or byte > '9') {
            return error.InvalidNpy;
        }
        value = std.math.mul(u64, value, 10) catch return error.LengthOverflow;
        value = std.math.add(u64, value, @as(u64, byte - '0')) catch return error.LengthOverflow;
        if (value > std.math.maxInt(i64)) {
            return error.LengthOverflow;
        }
    }
    return @intCast(value);
}

fn shapeSize(shape: []const i64) Error!usize {
    var size: usize = 1;
    for (shape) |dim| {
        if (dim < 0) return error.InvalidNpy;
        const value: usize = @intCast(dim);
        size = std.math.mul(usize, size, value) catch return error.LengthOverflow;
    }
    return size;
}

fn trimAscii(value: []const u8) []const u8 {
    var start: usize = 0;
    var end = value.len;
    while (start < end and isAsciiSpace(value[start])) start += 1;
    while (end > start and isAsciiSpace(value[end - 1])) end -= 1;
    return value[start..end];
}

fn checkedAdd(left: usize, right: usize) Error!usize {
    return std.math.add(usize, left, right) catch error.LengthOverflow;
}

fn usizeFromU32(value: u32) Error!usize {
    if (value > std.math.maxInt(usize)) {
        return error.LengthOverflow;
    }
    return @intCast(value);
}

fn isAsciiSpace(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r';
}

fn readU16(bytes: []const u8) u16 {
    return @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
}

fn readU32(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
}

fn appendSegment(builder: *Builder, segment: []const Entry, max_tokens: usize) Error!void {
    if (segment.len == 0) {
        return;
    }
    if (segment.len > max_tokens) {
        try builder.flush();
        for (segment) |entry| {
            if (builder.current_count == max_tokens) {
                try builder.flush();
            }
            try builder.append(entry);
        }
        return;
    }
    if (builder.current_count > 0 and builder.current_count + segment.len > max_tokens) {
        try builder.flush();
    }
    try builder.appendSlice(segment);
}

fn filterEntries(
    allocator: std.mem.Allocator,
    input: []const u8,
    vocab: VocabView,
) Error![]Entry {
    var out: std.ArrayList(Entry) = .empty;
    errdefer out.deinit(allocator);

    const space_id = vocab.id(' ');
    var index: usize = 0;
    var last_was_space = true;
    while (index < input.len) {
        const rune = try nextRune(input, index);
        index += rune.byte_len;
        const token = vocab.id(rune.codepoint) orelse continue;
        if (isSpace(rune.codepoint)) {
            if (!last_was_space and out.items.len > 0) {
                out.append(allocator, makeEntry(' ', space_id orelse token)) catch return error.OutOfMemory;
            }
            last_was_space = true;
            continue;
        }
        out.append(allocator, makeEntry(rune.codepoint, token)) catch return error.OutOfMemory;
        last_was_space = false;
    }
    if (out.items.len > 0 and isSpace(out.items[out.items.len - 1].codepoint)) {
        out.items.len -= 1;
    }
    return out.toOwnedSlice(allocator) catch error.OutOfMemory;
}

fn makeEntry(codepoint: u21, token: i64) Entry {
    var bytes: [4]u8 = undefined;
    const len = writeUtf8(&bytes, codepoint);
    return .{
        .codepoint = codepoint,
        .token = token,
        .bytes = bytes,
        .byte_len = len,
    };
}

fn nextRune(input: []const u8, index: usize) Error!Rune {
    const first = input[index];
    if (first < 0x80) {
        return .{ .codepoint = @intCast(first), .byte_len = 1 };
    }
    if ((first & 0xe0) == 0xc0) {
        if (index + 1 >= input.len or !isCont(input[index + 1])) return error.InvalidUtf8;
        const value: u32 = (@as(u32, first & 0x1f) << 6) |
            @as(u32, input[index + 1] & 0x3f);
        if (value < 0x80) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .byte_len = 2 };
    }
    if ((first & 0xf0) == 0xe0) {
        if (index + 2 >= input.len or !isCont(input[index + 1]) or !isCont(input[index + 2])) {
            return error.InvalidUtf8;
        }
        const value: u32 = (@as(u32, first & 0x0f) << 12) |
            (@as(u32, input[index + 1] & 0x3f) << 6) |
            @as(u32, input[index + 2] & 0x3f);
        if (value < 0x800 or (value >= 0xd800 and value <= 0xdfff)) return error.InvalidUtf8;
        return .{ .codepoint = @intCast(value), .byte_len = 3 };
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
        return .{ .codepoint = @intCast(value), .byte_len = 4 };
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

fn isSpace(codepoint: u21) bool {
    return switch (@as(u32, codepoint)) {
        0x09...0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

fn isBreak(codepoint: u21) bool {
    return switch (@as(u32, codepoint)) {
        '.', ',', '!', '?', ':', ';', 0x2014, 0x2026 => true,
        else => false,
    };
}

fn voiceMarkerLen(input: []const u8, index: usize) ?usize {
    if (input[index] != '(') {
        return null;
    }
    var cursor = index + 1;
    var letters: usize = 0;
    while (cursor < input.len and letters < 3 and isAsciiLetter(input[cursor])) {
        cursor += 1;
        letters += 1;
    }
    if (letters < 2 or cursor >= input.len or input[cursor] != ')') {
        return null;
    }
    return cursor - index + 1;
}

fn appendSpace(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    last_was_space: *bool,
) Error!void {
    if (!last_was_space.* and out.items.len > 0) {
        out.append(allocator, ' ') catch return error.OutOfMemory;
    }
    last_was_space.* = true;
}

fn isAsciiLetter(byte: u8) bool {
    return (byte >= 'a' and byte <= 'z') or (byte >= 'A' and byte <= 'Z');
}

fn isAsciiDigit(codepoint: u21) bool {
    return codepoint >= '0' and codepoint <= '9';
}

fn intSize(value: usize) Error!isize {
    if (value > std.math.maxInt(isize)) {
        return error.LengthOverflow;
    }
    return @intCast(value);
}

test "kokoro filters phonemes to vocab codepoints" {
    const codes = [_]i32{ ' ', 'a', 'b', 0x02c8, 0x0279 };
    const out = try filter(std.testing.allocator, " 'a ɹ2_b ", &codes);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqualStrings("a ɹb", out);
}

test "kokoro cleans eSpeak phoneme output" {
    const out = try clean(std.testing.allocator, "(en) h\u{200d}ə1\n(cm) lo");
    defer std.testing.allocator.free(out);

    try std.testing.expectEqualStrings("hə lo", out);
}

test "kokoro plans trimmed token chunks" {
    const codes = [_]i32{ ' ', 'a', 'b', '.' };
    const ids = [_]i64{ 16, 43, 44, 4 };
    var out = try plan(std.testing.allocator, "aaa bbb aaa", &codes, &ids, 6, true);
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(i64, &.{ 43, 43, 43, 44, 44, 44, 43, 43, 43 }, out.tokens);
    try std.testing.expectEqualSlices(isize, &.{ 3, 3, 3 }, out.lengths);
    try std.testing.expectEqualSlices(isize, &.{ 0, 3, 6 }, out.starts);
    try std.testing.expectEqualSlices(isize, &.{ 3, 3, 3 }, out.byte_lengths);
    try std.testing.expectEqualStrings("aaabbbaaa", out.text);
}

test "kokoro splits long segments by max tokens" {
    const codes = [_]i32{'a'};
    const ids = [_]i64{43};
    var out = try plan(std.testing.allocator, "aaaaaaaaa", &codes, &ids, 4, false);
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(isize, &.{ 4, 4, 1 }, out.lengths);
    try std.testing.expectEqual(@as(usize, 9), out.tokens.len);
}

test "kokoro prepares native input tensors" {
    var input_ids = [_]i64{9} ** 5;
    const token_ids = [_]i64{ 11, 12, 13 };
    var style = [_]f32{ 0, 0 };
    const voice = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var speed_out = [_]f32{0};

    try inputs(&input_ids, &token_ids, &style, &voice, 3, 2, &speed_out, 1.25);

    try std.testing.expectEqualSlices(i64, &.{ 0, 11, 12, 13, 0 }, &input_ids);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6 }, &style);
    try std.testing.expectEqual(@as(f32, 1.25), speed_out[0]);
}

test "kokoro parses config vocab into native buffers" {
    var out = try parseVocab(std.testing.allocator,
        \\{"vocab":{"a":43,"ɹ":91," ":16}}
    );
    defer out.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(i32, &.{ 'a', 0x0279, ' ' }, out.codes);
    try std.testing.expectEqualSlices(i64, &.{ 43, 91, 16 }, out.ids);
}
