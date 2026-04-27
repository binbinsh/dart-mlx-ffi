const std = @import("std");

pub const Error = error{
    FileOpen,
    FileTooLarge,
    OutOfMemory,
};

pub const Abi = extern struct {
    native_backend: [*c]const u8,
    peak_memory_bytes: u64,
    vm_hwm: u64,
    vm_rss: u64,
    phys_footprint: u64,
    resident_size: u64,
    virtual_size: u64,
    peak_working_set: u64,
    working_set: u64,
    android_peak_pss: u64,
    android_pss: u64,
    android_rss: u64,
    android_native_heap_pss: u64,
    android_java_heap_pss: u64,
    android_native_heap_private_dirty: u64,
    android_java_heap_private_dirty: u64,
};

pub fn empty() Abi {
    return .{
        .native_backend = null,
        .peak_memory_bytes = 0,
        .vm_hwm = 0,
        .vm_rss = 0,
        .phys_footprint = 0,
        .resident_size = 0,
        .virtual_size = 0,
        .peak_working_set = 0,
        .working_set = 0,
        .android_peak_pss = 0,
        .android_pss = 0,
        .android_rss = 0,
        .android_native_heap_pss = 0,
        .android_java_heap_pss = 0,
        .android_native_heap_private_dirty = 0,
        .android_java_heap_private_dirty = 0,
    };
}

pub fn linux(allocator: std.mem.Allocator) Error!Abi {
    const status = try readProcStatus(allocator);
    defer allocator.free(status);
    const peak = procStatusKb(status, "VmHWM:");
    const rss = procStatusKb(status, "VmRSS:");
    var info = empty();
    info.native_backend = "zig".ptr;
    info.peak_memory_bytes = peak;
    info.vm_hwm = peak;
    info.vm_rss = rss;
    return info;
}

fn readProcStatus(allocator: std.mem.Allocator) Error![]u8 {
    const file = std.c.fopen("/proc/self/status", "rb") orelse return error.FileOpen;
    defer _ = std.c.fclose(file);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var buffer: [4096]u8 = undefined;
    while (true) {
        const read = std.c.fread(buffer[0..].ptr, 1, buffer.len, file);
        if (read == 0) {
            break;
        }
        if (out.items.len + read > 128 * 1024) {
            return error.FileTooLarge;
        }
        out.appendSlice(allocator, buffer[0..read]) catch return error.OutOfMemory;
    }
    return out.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

pub fn procStatusKb(status: []const u8, key: []const u8) u64 {
    var lines = std.mem.splitScalar(u8, status, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) {
            continue;
        }
        var fields = std.mem.tokenizeAny(u8, line[key.len..], " \t");
        const value = fields.next() orelse return 0;
        const kb = std.fmt.parseInt(u64, value, 10) catch return 0;
        return kb * 1024;
    }
    return 0;
}

test "Linux proc status memory fields parse as bytes" {
    const status =
        \\Name: dart
        \\VmHWM:    12 kB
        \\VmRSS:     9 kB
    ;
    try std.testing.expectEqual(@as(u64, 12 * 1024), procStatusKb(status, "VmHWM:"));
    try std.testing.expectEqual(@as(u64, 9 * 1024), procStatusKb(status, "VmRSS:"));
    try std.testing.expectEqual(@as(u64, 0), procStatusKb(status, "VmSize:"));
}
