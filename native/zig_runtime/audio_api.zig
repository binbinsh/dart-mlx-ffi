const std = @import("std");
const abi = @import("abi.zig");
const audio = @import("audio.zig");

const F32Chunks = struct {
    items: []const []const f32,
    allocated: ?[][]const f32 = null,

    fn deinit(self: F32Chunks, allocator: std.mem.Allocator) void {
        if (self.allocated) |items| {
            allocator.free(items);
        }
    }
};

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = abi.copyString(message);
    }
}

fn readChunks(
    allocator: std.mem.Allocator,
    sample_chunks: [*c]const [*c]const f32,
    sample_counts: [*c]const isize,
    chunk_count: isize,
    error_out: ?*[*c]u8,
) ?F32Chunks {
    if (chunk_count < 0 or (chunk_count > 0 and sample_chunks == null) or (chunk_count > 0 and sample_counts == null)) {
        setError(error_out, audio.errorMessage(error.InvalidInput));
        return null;
    }
    const count: usize = @intCast(chunk_count);
    if (count == 0) {
        return .{ .items = &.{} };
    }

    const chunks = allocator.alloc([]const f32, count) catch {
        setError(error_out, audio.errorMessage(error.OutOfMemory));
        return null;
    };
    var index: usize = 0;
    while (index < count) : (index += 1) {
        const len = sample_counts[index];
        if (len < 0) {
            allocator.free(chunks);
            setError(error_out, audio.errorMessage(error.InvalidInput));
            return null;
        }
        const samples = sample_chunks[index];
        if (len > 0 and samples == null) {
            allocator.free(chunks);
            setError(error_out, audio.errorMessage(error.InvalidInput));
            return null;
        }
        chunks[index] = if (len == 0) &.{} else samples[0..@intCast(len)];
    }
    return .{ .items = chunks, .allocated = chunks };
}

export fn dinf_audio_wav_pcm16(
    samples: [*c]const f32,
    sample_count: isize,
    sample_rate: i32,
    byte_length_out: ?*isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    const byte_length = byte_length_out orelse {
        setError(error_out, "Zig audio byte_length_out is null.");
        return null;
    };
    byte_length.* = 0;
    if (sample_count < 0 or (sample_count > 0 and samples == null)) {
        setError(error_out, audio.errorMessage(error.InvalidInput));
        return null;
    }
    const items = if (sample_count == 0) &.{} else samples[0..@intCast(sample_count)];
    const wav = audio.wavPcm16(std.heap.c_allocator, items, sample_rate) catch |err| {
        setError(error_out, audio.errorMessage(err));
        return null;
    };
    byte_length.* = @intCast(wav.len);
    return @ptrCast(wav.ptr);
}

export fn dinf_audio_wav_pcm16_chunks(
    sample_chunks: [*c]const [*c]const f32,
    sample_counts: [*c]const isize,
    chunk_count: isize,
    sample_rate: i32,
    byte_length_out: ?*isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    const byte_length = byte_length_out orelse {
        setError(error_out, "Zig audio byte_length_out is null.");
        return null;
    };
    byte_length.* = 0;

    const allocator = std.heap.c_allocator;
    const chunks = readChunks(allocator, sample_chunks, sample_counts, chunk_count, error_out) orelse return null;
    defer chunks.deinit(allocator);

    const wav = audio.wavPcm16Chunks(allocator, chunks.items, sample_rate) catch |err| {
        setError(error_out, audio.errorMessage(err));
        return null;
    };
    byte_length.* = @intCast(wav.len);
    return @ptrCast(wav.ptr);
}

export fn dinf_audio_concat_f32(
    sample_chunks: [*c]const [*c]const f32,
    sample_counts: [*c]const isize,
    chunk_count: isize,
    sample_count_out: ?*isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    const sample_count = sample_count_out orelse {
        setError(error_out, "Zig audio sample_count_out is null.");
        return null;
    };
    sample_count.* = 0;

    const allocator = std.heap.c_allocator;
    const chunks = readChunks(allocator, sample_chunks, sample_counts, chunk_count, error_out) orelse return null;
    defer chunks.deinit(allocator);

    const out = audio.concatF32(allocator, chunks.items) catch |err| {
        setError(error_out, audio.errorMessage(err));
        return null;
    };
    sample_count.* = @intCast(out.len);
    if (out.len == 0) {
        allocator.free(out);
        return null;
    }
    return @ptrCast(out.ptr);
}
