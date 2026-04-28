const std = @import("std");
const abi = @import("abi.zig");
const kokoro = @import("kokoro.zig");
const phoneme = @import("phoneme.zig");
const phon_run = @import("phon_run.zig");
const phon_ssml = @import("phon_ssml.zig");

pub const PlanAbi = extern struct {
    tokens: [*c]i64,
    token_count: isize,
    lengths: [*c]isize,
    chunk_count: isize,
    text: [*c]u8,
    text_len: isize,
    starts: [*c]isize,
    byte_lengths: [*c]isize,
};

pub const NpyAbi = extern struct {
    name: [*c]u8,
    shape: [*c]i64,
    rank: isize,
    data: ?*anyopaque,
    byte_length: isize,
};

pub const VocabAbi = extern struct {
    codes: [*c]i32,
    ids: [*c]i64,
    count: isize,
};

pub const RunAbi = extern struct {
    text: [*c]u8,
    language: [*c]u8,
};

pub const SsmlAbi = extern struct {
    kind: i32,
    space_after: i32,
    text: [*c]u8,
};

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

export fn dinf_kok_filter(
    phonemes: [*c]const u8,
    codes: [*c]const i32,
    code_count: isize,
    error_out: ?*[*c]u8,
) [*c]u8 {
    if (phonemes == null or code_count < 0 or (code_count > 0 and codes == null)) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return null;
    }
    const input = phonemes[0..std.mem.len(phonemes)];
    const code_items: []const i32 = if (code_count == 0) &.{} else codes[0..@intCast(code_count)];
    const out = kokoro.filter(std.heap.c_allocator, input, code_items) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) std.heap.c_allocator.free(out);
    return abi.copyString(out);
}

export fn dinf_kok_clean(
    phonemes: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    if (phonemes == null) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return null;
    }
    const input = phonemes[0..std.mem.len(phonemes)];
    const out = kokoro.clean(std.heap.c_allocator, input) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) std.heap.c_allocator.free(out);
    return abi.copyString(out);
}

export fn dinf_kok_norm(
    text: [*c]const u8,
    language: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return phonemeCall(text, language, error_out, phoneme.normText);
}

export fn dinf_kok_post(
    text: [*c]const u8,
    language: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return phonemeCall(text, language, error_out, phoneme.post);
}

export fn dinf_kok_plain(
    text: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return phonemeUnary(text, error_out, phoneme.plainSsml);
}

export fn dinf_kok_exp(
    text: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return phonemeUnary(text, error_out, phoneme.explicitPhonemes);
}

export fn dinf_kok_pin_norm(
    text: [*c]const u8,
    error_out: ?*[*c]u8,
) [*c]u8 {
    return phonemeUnary(text, error_out, phoneme.pinyinNorm);
}

export fn dinf_kok_pin(text: [*c]const u8) i32 {
    if (text == null) {
        return 0;
    }
    return if (phoneme.looksPinyin(text[0..std.mem.len(text)])) 1 else 0;
}

export fn dinf_kok_runs(
    text: [*c]const u8,
    default_language: [*c]const u8,
    out: ?*[*c]RunAbi,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const items_out = out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    const count = count_out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    items_out.* = null;
    count.* = 0;
    if (text == null or default_language == null) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }

    var runs = phon_run.split(
        std.heap.c_allocator,
        text[0..std.mem.len(text)],
        default_language[0..std.mem.len(default_language)],
    ) catch |err| {
        setError(error_out, phon_run.errorMessage(err));
        return 1;
    };
    defer runs.deinit(std.heap.c_allocator);

    if (runs.items.len == 0) {
        return 0;
    }
    const raw = std.c.malloc(@sizeOf(RunAbi) * runs.items.len) orelse {
        setError(error_out, kokoro.errorMessage(error.OutOfMemory));
        return 1;
    };
    const items: [*]RunAbi = @ptrCast(@alignCast(raw));
    for (runs.items, 0..) |item, index| {
        items[index] = .{ .text = null, .language = null };
        items[index].text = abi.copyString(item.text);
        items[index].language = abi.copyString(item.language);
        if (items[index].text == null or items[index].language == null) {
            freeRunItems(items, index + 1);
            std.c.free(items);
            setError(error_out, kokoro.errorMessage(error.OutOfMemory));
            return 1;
        }
    }
    items_out.* = @ptrCast(items);
    count.* = @intCast(runs.items.len);
    return 0;
}

export fn dinf_kok_lang(
    text: [*c]const u8,
    requested: [*c]const u8,
    mixed_out: ?*i32,
    error_out: ?*[*c]u8,
) [*c]u8 {
    const mixed = mixed_out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return null;
    };
    mixed.* = 0;
    if (text == null or requested == null) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return null;
    }
    const value = phon_run.route(
        text[0..std.mem.len(text)],
        requested[0..std.mem.len(requested)],
    ) catch |err| {
        setError(error_out, phon_run.errorMessage(err));
        return null;
    };
    mixed.* = if (value.mixed) 1 else 0;
    return abi.copyString(value.language);
}

export fn dinf_kok_ssml(
    text: [*c]const u8,
    out: ?*[*c]SsmlAbi,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const items_out = out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    const count = count_out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    items_out.* = null;
    count.* = 0;
    if (text == null) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }

    var chunks = phon_ssml.parse(
        std.heap.c_allocator,
        text[0..std.mem.len(text)],
    ) catch |err| {
        setError(error_out, phoneme.errorMessage(err));
        return 1;
    };
    defer chunks.deinit(std.heap.c_allocator);

    if (chunks.chunks.len == 0) {
        return 0;
    }
    const raw = std.c.malloc(@sizeOf(SsmlAbi) * chunks.chunks.len) orelse {
        setError(error_out, kokoro.errorMessage(error.OutOfMemory));
        return 1;
    };
    const items: [*]SsmlAbi = @ptrCast(@alignCast(raw));
    for (chunks.chunks, 0..) |chunk, index| {
        items[index] = .{
            .kind = @intFromEnum(chunk.kind),
            .space_after = if (chunk.space_after) 1 else 0,
            .text = abi.copyString(chunk.text),
        };
        if (items[index].text == null) {
            freeSsmlItems(items, index + 1);
            std.c.free(items);
            setError(error_out, kokoro.errorMessage(error.OutOfMemory));
            return 1;
        }
    }
    items_out.* = @ptrCast(items);
    count.* = @intCast(chunks.chunks.len);
    return 0;
}

export fn dinf_kok_free_ssml(items: [*c]SsmlAbi, count: isize) void {
    if (items == null) {
        return;
    }
    if (count > 0) {
        freeSsmlItems(@ptrCast(items), @intCast(count));
    }
    std.c.free(items);
}

export fn dinf_kok_free_runs(items: [*c]RunAbi, count: isize) void {
    if (items == null) {
        return;
    }
    if (count > 0) {
        freeRunItems(@ptrCast(items), @intCast(count));
    }
    std.c.free(items);
}

export fn dinf_kok_row(
    style: [*c]f32,
    style_len: isize,
    voice: [*c]const f32,
    voice_len: isize,
    voice_rows: isize,
    voice_row_len: isize,
    index: isize,
    error_out: ?*[*c]u8,
) i32 {
    if (style_len < 0 or voice_len < 0 or voice_rows < 0 or voice_row_len < 0 or index < 0 or
        (style_len > 0 and style == null) or
        (voice_len > 0 and voice == null))
    {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }
    var empty_style: [0]f32 = .{};
    const style_items = if (style_len == 0) empty_style[0..] else style[0..@intCast(style_len)];
    const voice_items: []const f32 = if (voice_len == 0) &.{} else voice[0..@intCast(voice_len)];
    kokoro.row(
        style_items,
        voice_items,
        @intCast(voice_rows),
        @intCast(voice_row_len),
        @intCast(index),
    ) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return 1;
    };
    return 0;
}

export fn dinf_kok_inputs(
    input_ids: [*c]i64,
    input_len: isize,
    token_ids: [*c]const i64,
    token_count: isize,
    style: [*c]f32,
    style_len: isize,
    voice: [*c]const f32,
    voice_len: isize,
    voice_rows: isize,
    voice_row_len: isize,
    speed_out: [*c]f32,
    speed_len: isize,
    speed: f32,
    error_out: ?*[*c]u8,
) i32 {
    if (input_len < 0 or token_count < 0 or style_len < 0 or voice_len < 0 or
        voice_rows < 0 or voice_row_len < 0 or speed_len < 0 or
        (input_len > 0 and input_ids == null) or
        (token_count > 0 and token_ids == null) or
        (style_len > 0 and style == null) or
        (voice_len > 0 and voice == null) or
        (speed_len > 0 and speed_out == null))
    {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }

    var empty_ids: [0]i64 = .{};
    const input_items = if (input_len == 0) empty_ids[0..] else input_ids[0..@intCast(input_len)];
    const token_items: []const i64 = if (token_count == 0) &.{} else token_ids[0..@intCast(token_count)];
    var empty_style: [0]f32 = .{};
    const style_items = if (style_len == 0) empty_style[0..] else style[0..@intCast(style_len)];
    const voice_items: []const f32 = if (voice_len == 0) &.{} else voice[0..@intCast(voice_len)];
    var empty_speed: [0]f32 = .{};
    const speed_items = if (speed_len == 0) empty_speed[0..] else speed_out[0..@intCast(speed_len)];

    kokoro.inputs(
        input_items,
        token_items,
        style_items,
        voice_items,
        @intCast(voice_rows),
        @intCast(voice_row_len),
        speed_items,
        speed,
    ) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return 1;
    };
    return 0;
}

export fn dinf_kok_plan(
    phonemes: [*c]const u8,
    codes: [*c]const i32,
    ids: [*c]const i64,
    vocab_count: isize,
    max_tokens: isize,
    include_text: i32,
    out: ?*PlanAbi,
    error_out: ?*[*c]u8,
) i32 {
    const plan_out = out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    plan_out.* = emptyPlan();
    if (phonemes == null or vocab_count < 0 or max_tokens <= 0 or
        (vocab_count > 0 and (codes == null or ids == null)))
    {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }
    const input = phonemes[0..std.mem.len(phonemes)];
    const code_items: []const i32 = if (vocab_count == 0) &.{} else codes[0..@intCast(vocab_count)];
    const id_items: []const i64 = if (vocab_count == 0) &.{} else ids[0..@intCast(vocab_count)];
    const planned = kokoro.plan(
        std.heap.c_allocator,
        input,
        code_items,
        id_items,
        @intCast(max_tokens),
        include_text != 0,
    ) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return 1;
    };
    plan_out.* = toAbi(planned);
    return 0;
}

export fn dinf_kok_free_plan(plan: ?*PlanAbi) void {
    const value = plan orelse return;
    std.c.free(value.tokens);
    std.c.free(value.lengths);
    std.c.free(value.text);
    std.c.free(value.starts);
    std.c.free(value.byte_lengths);
    value.* = emptyPlan();
}

export fn dinf_kok_npy(
    bytes: [*c]const u8,
    byte_count: isize,
    out: ?*NpyAbi,
    error_out: ?*[*c]u8,
) i32 {
    const target = out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    target.* = emptyNpy();
    if (byte_count < 0 or (byte_count > 0 and bytes == null)) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }
    const input = if (byte_count == 0) &.{} else bytes[0..@intCast(byte_count)];
    const array = kokoro.parseNpy(std.heap.c_allocator, input) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return 1;
    };
    target.* = toNpyAbi(null, array);
    return 0;
}

export fn dinf_kok_npz(
    path: [*c]const u8,
    out: ?*[*c]NpyAbi,
    count_out: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const items_out = out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    const count = count_out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    items_out.* = null;
    count.* = 0;
    if (path == null) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }

    var archive = kokoro.loadNpz(
        std.heap.c_allocator,
        std.Io.Threaded.global_single_threaded.io(),
        path[0..std.mem.len(path)],
    ) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return 1;
    };
    defer archive.deinit(std.heap.c_allocator);

    const raw = std.c.malloc(@sizeOf(NpyAbi) * archive.items.len) orelse {
        setError(error_out, kokoro.errorMessage(error.OutOfMemory));
        return 1;
    };
    const items: [*]NpyAbi = @ptrCast(@alignCast(raw));
    for (items[0..archive.items.len]) |*item| {
        item.* = emptyNpy();
    }
    for (archive.items, 0..) |*item, index| {
        const name = abi.copyString(item.name);
        if (name == null) {
            for (items[0..index]) |*done| {
                freeNpy(done);
            }
            std.c.free(raw);
            setError(error_out, kokoro.errorMessage(error.OutOfMemory));
            return 1;
        }
        items[index] = toNpyAbi(name, item.array);
        item.array = .{};
    }
    items_out.* = @ptrCast(items);
    count.* = @intCast(archive.items.len);
    return 0;
}

export fn dinf_kok_vocab(
    path: [*c]const u8,
    out: ?*VocabAbi,
    error_out: ?*[*c]u8,
) i32 {
    const target = out orelse {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    };
    target.* = emptyVocab();
    if (path == null) {
        setError(error_out, kokoro.errorMessage(error.InvalidInput));
        return 1;
    }

    var vocab = kokoro.loadVocab(
        std.heap.c_allocator,
        std.Io.Threaded.global_single_threaded.io(),
        path[0..std.mem.len(path)],
    ) catch |err| {
        setError(error_out, kokoro.errorMessage(err));
        return 1;
    };
    target.* = toVocabAbi(vocab);
    vocab = .{};
    return 0;
}

export fn dinf_kok_free_vocab(value: ?*VocabAbi) void {
    const item = value orelse return;
    std.c.free(item.codes);
    std.c.free(item.ids);
    item.* = emptyVocab();
}

export fn dinf_kok_free_npy(value: ?*NpyAbi) void {
    const item = value orelse return;
    freeNpy(item);
}

export fn dinf_kok_free_npz(items: [*c]NpyAbi, count: isize) void {
    if (items == null) {
        return;
    }
    if (count > 0) {
        const item_count: usize = @intCast(count);
        for (items[0..item_count]) |*item| {
            freeNpy(item);
        }
    }
    std.c.free(items);
}

fn phonemeCall(
    text: [*c]const u8,
    language: [*c]const u8,
    error_out: ?*[*c]u8,
    comptime func: fn (std.mem.Allocator, []const u8, []const u8) phoneme.Error![]u8,
) [*c]u8 {
    if (text == null or language == null) {
        setError(error_out, phoneme.errorMessage(error.InvalidInput));
        return null;
    }
    const out = func(
        std.heap.c_allocator,
        text[0..std.mem.len(text)],
        language[0..std.mem.len(language)],
    ) catch |err| {
        setError(error_out, phoneme.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) std.heap.c_allocator.free(out);
    return abi.copyString(out);
}

fn phonemeUnary(
    text: [*c]const u8,
    error_out: ?*[*c]u8,
    comptime func: fn (std.mem.Allocator, []const u8) phoneme.Error![]u8,
) [*c]u8 {
    if (text == null) {
        setError(error_out, phoneme.errorMessage(error.InvalidInput));
        return null;
    }
    const out = func(
        std.heap.c_allocator,
        text[0..std.mem.len(text)],
    ) catch |err| {
        setError(error_out, phoneme.errorMessage(err));
        return null;
    };
    defer if (out.len > 0) std.heap.c_allocator.free(out);
    return abi.copyString(out);
}

fn freeRunItems(items: [*]RunAbi, count: usize) void {
    for (items[0..count]) |item| {
        abi.freeString(item.text);
        abi.freeString(item.language);
    }
}

fn freeSsmlItems(items: [*]SsmlAbi, count: usize) void {
    for (items[0..count]) |item| {
        abi.freeString(item.text);
    }
}

fn toAbi(plan: kokoro.Plan) PlanAbi {
    return .{
        .tokens = if (plan.tokens.len == 0) null else plan.tokens.ptr,
        .token_count = @intCast(plan.tokens.len),
        .lengths = if (plan.lengths.len == 0) null else plan.lengths.ptr,
        .chunk_count = @intCast(plan.lengths.len),
        .text = if (plan.text.len == 0) null else plan.text.ptr,
        .text_len = @intCast(plan.text.len),
        .starts = if (plan.starts.len == 0) null else plan.starts.ptr,
        .byte_lengths = if (plan.byte_lengths.len == 0) null else plan.byte_lengths.ptr,
    };
}

fn toNpyAbi(name: [*c]u8, array: kokoro.Npy) NpyAbi {
    return .{
        .name = name,
        .shape = if (array.shape.len == 0) null else array.shape.ptr,
        .rank = @intCast(array.shape.len),
        .data = if (array.data.len == 0) null else array.data.ptr,
        .byte_length = @intCast(array.data.len),
    };
}

fn toVocabAbi(vocab: kokoro.Vocab) VocabAbi {
    return .{
        .codes = if (vocab.codes.len == 0) null else vocab.codes.ptr,
        .ids = if (vocab.ids.len == 0) null else vocab.ids.ptr,
        .count = @intCast(vocab.codes.len),
    };
}

fn freeNpy(item: *NpyAbi) void {
    std.c.free(item.name);
    std.c.free(item.shape);
    std.c.free(item.data);
    item.* = emptyNpy();
}

fn emptyPlan() PlanAbi {
    return .{
        .tokens = null,
        .token_count = 0,
        .lengths = null,
        .chunk_count = 0,
        .text = null,
        .text_len = 0,
        .starts = null,
        .byte_lengths = null,
    };
}

fn emptyNpy() NpyAbi {
    return .{
        .name = null,
        .shape = null,
        .rank = 0,
        .data = null,
        .byte_length = 0,
    };
}

fn emptyVocab() VocabAbi {
    return .{
        .codes = null,
        .ids = null,
        .count = 0,
    };
}
