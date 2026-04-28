const std = @import("std");
const abi = @import("abi.zig");
const fill = @import("fill.zig");
const front_config = @import("front_config.zig");
const structured_input = @import("structured_input.zig");
const target = @import("target.zig");

pub const ConfigAbi = extern struct {
    batch_size: isize,
    token_length: isize,
    char_length: isize,
    homograph_targets: isize,
    polyphone_targets: isize,
    homograph_classes: isize,
    polyphone_classes: isize,
    emphasis_threshold: f64,
};

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

export fn dinf_struct_config(
    export_path: [*c]const u8,
    structured_path: [*c]const u8,
    out: ?*ConfigAbi,
    error_out: ?*[*c]u8,
) i32 {
    const target_out = out orelse {
        setError(error_out, front_config.errorMessage(error.InvalidInput));
        return 1;
    };
    target_out.* = emptyConfig();
    if (export_path == null or structured_path == null) {
        setError(error_out, front_config.errorMessage(error.InvalidInput));
        return 1;
    }
    const config = front_config.load(
        std.heap.c_allocator,
        std.Io.Threaded.global_single_threaded.io(),
        export_path[0..std.mem.len(export_path)],
        structured_path[0..std.mem.len(structured_path)],
    ) catch |err| {
        setError(error_out, front_config.errorMessage(err));
        return 1;
    };
    target_out.* = toConfigAbi(config);
    return 0;
}

export fn dinf_struct_reset(
    input_ids: [*c]i64,
    input_len: isize,
    token_pad_id: i64,
    attention: [*c]i64,
    attention_len: isize,
    char_ids: [*c]i64,
    char_len: isize,
    char_pad_id: i64,
    char_mask: [*c]i64,
    char_mask_len: isize,
    homograph_targets: [*c]u8,
    homograph_target_len: isize,
    homograph_candidates: [*c]u8,
    homograph_candidate_len: isize,
    polyphone_targets: [*c]u8,
    polyphone_target_len: isize,
    polyphone_candidates: [*c]u8,
    polyphone_candidate_len: isize,
    error_out: ?*[*c]u8,
) i32 {
    if (input_len < 0 or attention_len < 0 or char_len < 0 or char_mask_len < 0 or
        homograph_target_len < 0 or homograph_candidate_len < 0 or
        polyphone_target_len < 0 or polyphone_candidate_len < 0 or
        (input_len > 0 and input_ids == null) or
        (attention_len > 0 and attention == null) or
        (char_len > 0 and char_ids == null) or
        (char_mask_len > 0 and char_mask == null) or
        (homograph_target_len > 0 and homograph_targets == null) or
        (homograph_candidate_len > 0 and homograph_candidates == null) or
        (polyphone_target_len > 0 and polyphone_targets == null) or
        (polyphone_candidate_len > 0 and polyphone_candidates == null))
    {
        setError(error_out, fill.errorMessage(error.InvalidInput));
        return 1;
    }

    var empty_i64: [0]i64 = .{};
    var empty_u8: [0]u8 = .{};
    structured_input.reset(
        if (input_len == 0) empty_i64[0..] else input_ids[0..@intCast(input_len)],
        token_pad_id,
        if (attention_len == 0) empty_i64[0..] else attention[0..@intCast(attention_len)],
        if (char_len == 0) empty_i64[0..] else char_ids[0..@intCast(char_len)],
        char_pad_id,
        if (char_mask_len == 0) empty_i64[0..] else char_mask[0..@intCast(char_mask_len)],
        if (homograph_target_len == 0) empty_u8[0..] else homograph_targets[0..@intCast(homograph_target_len)],
        if (homograph_candidate_len == 0) empty_u8[0..] else homograph_candidates[0..@intCast(homograph_candidate_len)],
        if (polyphone_target_len == 0) empty_u8[0..] else polyphone_targets[0..@intCast(polyphone_target_len)],
        if (polyphone_candidate_len == 0) empty_u8[0..] else polyphone_candidates[0..@intCast(polyphone_candidate_len)],
    );
    return 0;
}

fn toConfigAbi(config: front_config.Config) ConfigAbi {
    return .{
        .batch_size = config.batch_size,
        .token_length = config.token_length,
        .char_length = config.char_length,
        .homograph_targets = config.homograph_targets,
        .polyphone_targets = config.polyphone_targets,
        .homograph_classes = config.homograph_classes,
        .polyphone_classes = config.polyphone_classes,
        .emphasis_threshold = config.emphasis_threshold,
    };
}

fn emptyConfig() ConfigAbi {
    return toConfigAbi(.{});
}

export fn dinf_struct_matches(
    target_values: [*c]u8,
    target_len: isize,
    target_offset: isize,
    target_width: isize,
    candidate_values: [*c]u8,
    candidate_len: isize,
    candidate_offset: isize,
    candidate_width: isize,
    matches: [*c]const target.Match,
    match_count: isize,
    token_starts: [*c]const i32,
    token_ends: [*c]const i32,
    token_count: isize,
    error_out: ?*[*c]u8,
) i32 {
    if (target_len < 0 or target_offset < 0 or target_width < 0 or
        candidate_len < 0 or candidate_offset < 0 or candidate_width < 0 or
        match_count < 0 or token_count < 0 or
        (target_len > 0 and target_values == null) or
        (candidate_len > 0 and candidate_values == null) or
        (match_count > 0 and matches == null) or
        (token_count > 0 and (token_starts == null or token_ends == null)))
    {
        setError(error_out, fill.errorMessage(error.InvalidInput));
        return 1;
    }
    if (match_count == 0) {
        return 0;
    }

    var empty_target: [0]u8 = .{};
    var empty_candidate: [0]u8 = .{};
    const target_items = if (target_len == 0) empty_target[0..] else target_values[0..@intCast(target_len)];
    const candidate_items = if (candidate_len == 0) empty_candidate[0..] else candidate_values[0..@intCast(candidate_len)];
    const match_items = matches[0..@intCast(match_count)];
    const token_start_items: ?[]const i32 = if (token_count == 0) null else token_starts[0..@intCast(token_count)];
    const token_end_items: ?[]const i32 = if (token_count == 0) null else token_ends[0..@intCast(token_count)];

    structured_input.matchTargets(
        target_items,
        @intCast(target_offset),
        @intCast(target_width),
        candidate_items,
        @intCast(candidate_offset),
        @intCast(candidate_width),
        match_items,
        token_start_items,
        token_end_items,
    ) catch |err| {
        setError(error_out, fill.errorMessage(err));
        return 1;
    };
    return 0;
}
