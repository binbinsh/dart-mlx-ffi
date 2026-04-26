pub const api = "mlx-c";
pub const owner = "zig";
pub const enabled = false;

pub fn unavailableMessage() []const u8 {
    return "Zig-owned MLX backend is not implemented yet; mlx-c must be called from native/zig_runtime, not Dart.";
}

pub fn statusJson() []const u8 {
    return "{\"owner\":\"zig\",\"api\":\"mlx-c\",\"enabled\":false}";
}
