const std = @import("std");

pub const Error = error{
    InvalidConfig,
    InvalidInput,
    OutOfMemory,
};

pub const Config = struct {
    batch_size: isize = 1,
    token_length: isize = 512,
    char_length: isize = 1024,
    homograph_targets: isize = 16,
    polyphone_targets: isize = 16,
    homograph_classes: isize = 1,
    polyphone_classes: isize = 1,
    emphasis_threshold: f64 = 0.75,
};

pub fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    export_path: []const u8,
    structured_path: []const u8,
) Error!Config {
    if (export_path.len == 0 or structured_path.len == 0) {
        return error.InvalidInput;
    }
    const export_bytes = try readFile(allocator, io, export_path);
    defer allocator.free(export_bytes);
    const structured_bytes = try readFile(allocator, io, structured_path);
    defer allocator.free(structured_bytes);
    return parse(export_bytes, structured_bytes);
}

pub fn parse(export_bytes: []const u8, structured_bytes: []const u8) Error!Config {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const export_parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        export_bytes,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return error.InvalidConfig;
    defer export_parsed.deinit();
    const structured_parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        structured_bytes,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return error.InvalidConfig;
    defer structured_parsed.deinit();

    const export_object = switch (export_parsed.value) {
        .object => |object| object,
        else => return error.InvalidConfig,
    };
    const structured_object = switch (structured_parsed.value) {
        .object => |object| object,
        else => return error.InvalidConfig,
    };

    return .{
        .batch_size = intField(export_object, "export_batch_size", 1),
        .token_length = intField(export_object, "export_token_length", 512),
        .char_length = intField(export_object, "export_char_length", 1024),
        .homograph_targets = intField(export_object, "export_homograph_targets", 16),
        .polyphone_targets = intField(export_object, "export_polyphone_targets", 16),
        .homograph_classes = intField(export_object, "num_homograph_classes", 1),
        .polyphone_classes = intField(export_object, "num_polyphone_classes", 1),
        .emphasis_threshold = floatField(structured_object, "emphasis_decoding_threshold", 0.75),
    };
}

pub fn errorMessage(err: Error) []const u8 {
    return switch (err) {
        error.InvalidConfig => "Zig UniFrontend received invalid structured config JSON.",
        error.InvalidInput => "Zig UniFrontend received invalid config paths.",
        error.OutOfMemory => "Zig UniFrontend config parser ran out of memory.",
    };
}

fn readFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    file_path: []const u8,
) Error![]u8 {
    const max_config_bytes = 16 * 1024 * 1024;
    return std.Io.Dir.cwd().readFileAlloc(
        io,
        file_path,
        allocator,
        .limited(max_config_bytes),
    ) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidConfig,
    };
}

fn intField(object: std.json.ObjectMap, key: []const u8, default_value: isize) isize {
    const value = object.get(key) orelse return default_value;
    return switch (value) {
        .integer => |integer| if (integer >= 0 and integer <= std.math.maxInt(isize))
            @intCast(integer)
        else
            default_value,
        .float => |float| if (float >= 0 and float <= @as(f64, @floatFromInt(std.math.maxInt(isize))))
            @intFromFloat(float)
        else
            default_value,
        else => default_value,
    };
}

fn floatField(object: std.json.ObjectMap, key: []const u8, default_value: f64) f64 {
    const value = object.get(key) orelse return default_value;
    return switch (value) {
        .integer => |integer| @floatFromInt(integer),
        .float => |float| float,
        else => default_value,
    };
}

test "structured config parses export and threshold fields" {
    const config = try parse(
        \\{
        \\  "export_batch_size": 2,
        \\  "export_token_length": 128,
        \\  "export_char_length": 256,
        \\  "export_homograph_targets": 4,
        \\  "export_polyphone_targets": 5,
        \\  "num_homograph_classes": 6,
        \\  "num_polyphone_classes": 7
        \\}
    ,
        \\{"emphasis_decoding_threshold": 0.6}
    );

    try std.testing.expectEqual(@as(isize, 2), config.batch_size);
    try std.testing.expectEqual(@as(isize, 128), config.token_length);
    try std.testing.expectEqual(@as(isize, 256), config.char_length);
    try std.testing.expectEqual(@as(isize, 4), config.homograph_targets);
    try std.testing.expectEqual(@as(isize, 5), config.polyphone_targets);
    try std.testing.expectEqual(@as(isize, 6), config.homograph_classes);
    try std.testing.expectEqual(@as(isize, 7), config.polyphone_classes);
    try std.testing.expectEqual(@as(f64, 0.6), config.emphasis_threshold);
}
