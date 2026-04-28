const std = @import("std");

pub const Error = error{
    InvalidInput,
    InvalidSampleRate,
    LengthOverflow,
    OutOfMemory,
};

const wav_header_bytes = 44;

pub fn wavPcm16(allocator: std.mem.Allocator, samples: []const f32, sample_rate: i32) Error![]u8 {
    return wavPcm16Chunks(allocator, &.{samples}, sample_rate);
}

pub fn wavPcm16Chunks(allocator: std.mem.Allocator, chunks: []const []const f32, sample_rate: i32) Error![]u8 {
    if (sample_rate <= 0) {
        return error.InvalidSampleRate;
    }
    var total_samples: usize = 0;
    for (chunks) |chunk| {
        total_samples = std.math.add(usize, total_samples, chunk.len) catch return error.LengthOverflow;
    }
    const data_bytes = std.math.mul(usize, total_samples, 2) catch return error.LengthOverflow;
    const total_bytes = std.math.add(usize, wav_header_bytes, data_bytes) catch return error.LengthOverflow;
    if (total_bytes > std.math.maxInt(u32)) {
        return error.LengthOverflow;
    }
    const rate: u32 = @intCast(sample_rate);
    const byte_rate = std.math.mul(u32, rate, 2) catch return error.LengthOverflow;
    const out = allocator.alloc(u8, total_bytes) catch return error.OutOfMemory;
    errdefer allocator.free(out);

    writeAscii(out, 0, "RIFF");
    writeU32(out, 4, std.math.add(u32, 36, @intCast(data_bytes)) catch return error.LengthOverflow);
    writeAscii(out, 8, "WAVE");
    writeAscii(out, 12, "fmt ");
    writeU32(out, 16, 16);
    writeU16(out, 20, 1);
    writeU16(out, 22, 1);
    writeU32(out, 24, rate);
    writeU32(out, 28, byte_rate);
    writeU16(out, 32, 2);
    writeU16(out, 34, 16);
    writeAscii(out, 36, "data");
    writeU32(out, 40, @intCast(data_bytes));

    var offset: usize = wav_header_bytes;
    for (chunks) |chunk| {
        for (chunk) |sample| {
            writeI16(out, offset, pcm16(sample));
            offset += 2;
        }
    }
    return out;
}

pub fn concatF32(allocator: std.mem.Allocator, chunks: []const []const f32) Error![]f32 {
    var total_samples: usize = 0;
    for (chunks) |chunk| {
        total_samples = std.math.add(usize, total_samples, chunk.len) catch return error.LengthOverflow;
    }

    const out = allocator.alloc(f32, total_samples) catch return error.OutOfMemory;
    errdefer allocator.free(out);

    var offset: usize = 0;
    for (chunks) |chunk| {
        @memcpy(out[offset .. offset + chunk.len], chunk);
        offset += chunk.len;
    }
    return out;
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidInput => "Zig audio received an invalid input buffer.",
        error.InvalidSampleRate => "Zig audio requires a positive sample rate.",
        error.LengthOverflow => "Zig audio output byte length overflowed.",
        error.OutOfMemory => "Zig audio ran out of memory.",
    };
}

fn pcm16(sample: f32) i16 {
    const clipped = @min(@max(sample, -1.0), 1.0);
    const scaled = @as(f64, clipped) * 32767.0;
    return @intFromFloat(std.math.round(scaled));
}

fn writeAscii(out: []u8, offset: usize, value: []const u8) void {
    @memcpy(out[offset .. offset + value.len], value);
}

fn writeU16(out: []u8, offset: usize, value: u16) void {
    out[offset] = @intCast(value & 0xff);
    out[offset + 1] = @intCast((value >> 8) & 0xff);
}

fn writeI16(out: []u8, offset: usize, value: i16) void {
    writeU16(out, offset, @bitCast(value));
}

fn writeU32(out: []u8, offset: usize, value: u32) void {
    out[offset] = @intCast(value & 0xff);
    out[offset + 1] = @intCast((value >> 8) & 0xff);
    out[offset + 2] = @intCast((value >> 16) & 0xff);
    out[offset + 3] = @intCast((value >> 24) & 0xff);
}

test "audio wav encoder writes pcm16 header and samples" {
    const wav = try wavPcm16(std.testing.allocator, &.{ -1.0, 0.0, 1.0 }, 24000);
    defer std.testing.allocator.free(wav);

    try std.testing.expectEqual(@as(usize, 50), wav.len);
    try std.testing.expectEqualStrings("RIFF", wav[0..4]);
    try std.testing.expectEqualStrings("WAVE", wav[8..12]);
    try std.testing.expectEqualStrings("fmt ", wav[12..16]);
    try std.testing.expectEqualStrings("data", wav[36..40]);
    try std.testing.expectEqual(@as(u8, 0x01), wav[44]);
    try std.testing.expectEqual(@as(u8, 0x80), wav[45]);
    try std.testing.expectEqual(@as(u8, 0x00), wav[46]);
    try std.testing.expectEqual(@as(u8, 0x00), wav[47]);
    try std.testing.expectEqual(@as(u8, 0xff), wav[48]);
    try std.testing.expectEqual(@as(u8, 0x7f), wav[49]);
}

test "audio wav encoder rejects invalid sample rate" {
    try std.testing.expectError(error.InvalidSampleRate, wavPcm16(std.testing.allocator, &.{0.0}, 0));
}

test "audio wav encoder writes chunked samples without pre-concat" {
    const left = [_]f32{ -1.0, 0.0 };
    const right = [_]f32{1.0};
    const wav = try wavPcm16Chunks(std.testing.allocator, &.{ &left, &right }, 24000);
    defer std.testing.allocator.free(wav);

    try std.testing.expectEqual(@as(usize, 50), wav.len);
    try std.testing.expectEqual(@as(u8, 0x01), wav[44]);
    try std.testing.expectEqual(@as(u8, 0x80), wav[45]);
    try std.testing.expectEqual(@as(u8, 0x00), wav[46]);
    try std.testing.expectEqual(@as(u8, 0x00), wav[47]);
    try std.testing.expectEqual(@as(u8, 0xff), wav[48]);
    try std.testing.expectEqual(@as(u8, 0x7f), wav[49]);
}

test "audio concat joins float32 chunks" {
    const left = [_]f32{ 0.1, 0.2 };
    const right = [_]f32{0.3};
    const out = try concatF32(std.testing.allocator, &.{ &left, &right });
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(@as(usize, 3), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), out[0], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), out[1], 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), out[2], 0.000001);
}
