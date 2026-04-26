const std = @import("std");
const builtin = @import("builtin");
const mlx_backend = @import("mlx_backend.zig");

const pinned_zig_version = "0.16.0";

const Engine = enum(i32) {
    mlx = 0,
    coreml = 1,
    onnx = 2,
    litert = 3,
};

const Dtype = enum(i32) {
    float32 = 1,
    int32 = 2,
    int64 = 3,
    uint8 = 4,
    float64 = 5,
    float16 = 6,
    boolean = 7,
};

const NativeTensor = extern struct {
    dtype: i32,
    rank: i32,
    shape: [*c]i64,
    byte_length: isize,
    data: ?*anyopaque,
};

const NamedTensor = extern struct {
    name: [*c]u8,
    tensor: NativeTensor,
};

const SessionMode = enum {
    echo,
    adapter,
    mlx,
};

const Session = struct {
    engine: i32,
    mode: SessionMode,
    model_path: [*c]u8,
    options_json: [*c]u8,
    adapter_handle: ?*anyopaque,
    mlx_handle: ?*mlx_backend.Session,
};

const cpp = if (builtin.is_test) struct {
    pub fn dinf_cpp_runtime_create(
        engine: i32,
        model_path: [*c]const u8,
        options_json: [*c]const u8,
        error_out: ?*[*c]u8,
    ) ?*anyopaque {
        _ = engine;
        _ = model_path;
        _ = options_json;
        setError(error_out, "test C++ adapter is unavailable");
        return null;
    }

    pub fn dinf_cpp_runtime_free(session: ?*anyopaque) void {
        _ = session;
    }

    pub fn dinf_cpp_runtime_run(
        session: ?*anyopaque,
        inputs: [*c]const NamedTensor,
        input_count: isize,
        outputs: ?*[*c]NamedTensor,
        output_count: ?*isize,
        error_out: ?*[*c]u8,
    ) i32 {
        _ = session;
        _ = inputs;
        _ = input_count;
        _ = outputs;
        _ = output_count;
        setError(error_out, "test C++ adapter is unavailable");
        return 1;
    }

    pub fn dinf_cpp_runtime_memory_info_json() [*c]u8 {
        return copyString("{\"peak_memory_bytes\":0,\"native_backend\":\"zig-test\"}");
    }

    pub fn dinf_cpp_runtime_diagnostics_json(session: ?*anyopaque) [*c]u8 {
        _ = session;
        return copyString("{}");
    }
} else struct {
    extern fn dinf_cpp_runtime_create(
        engine: i32,
        model_path: [*c]const u8,
        options_json: [*c]const u8,
        error_out: ?*[*c]u8,
    ) ?*anyopaque;
    extern fn dinf_cpp_runtime_free(session: ?*anyopaque) void;
    extern fn dinf_cpp_runtime_run(
        session: ?*anyopaque,
        inputs: [*c]const NamedTensor,
        input_count: isize,
        outputs: ?*[*c]NamedTensor,
        output_count: ?*isize,
        error_out: ?*[*c]u8,
    ) i32;
    extern fn dinf_cpp_runtime_memory_info_json() [*c]u8;
    extern fn dinf_cpp_runtime_diagnostics_json(session: ?*anyopaque) [*c]u8;
};

fn copyString(value: []const u8) [*c]u8 {
    const raw = std.c.malloc(value.len + 1) orelse return null;
    const bytes: [*]u8 = @ptrCast(raw);
    @memcpy(bytes[0..value.len], value);
    bytes[value.len] = 0;
    return @ptrCast(bytes);
}

fn freeString(value: [*c]u8) void {
    if (value != null) {
        std.c.free(value);
    }
}

fn copyCString(value: [*c]const u8) [*c]u8 {
    if (value == null) {
        return copyString("");
    }
    const len = std.mem.len(value);
    return copyString(value[0..len]);
}

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = copyString(message);
    }
}

fn backendJson() []const u8 {
    return "{\"native_backend\":\"zig\",\"zig_version\":\"0.16.0\",\"async_model\":\"std.Io-ready\",\"abi\":\"dart_inference_runtime_v1\",\"mlx_backend\":" ++ mlx_backend.status_json ++ "}";
}

fn copyBackendJson() [*c]u8 {
    return copyString(backendJson());
}

fn dtypeSize(dtype: i32) usize {
    return switch (dtype) {
        @intFromEnum(Dtype.float32), @intFromEnum(Dtype.int32) => 4,
        @intFromEnum(Dtype.int64), @intFromEnum(Dtype.float64) => 8,
        @intFromEnum(Dtype.uint8), @intFromEnum(Dtype.boolean) => 1,
        @intFromEnum(Dtype.float16) => 2,
        else => 0,
    };
}

fn validTensor(tensor: NativeTensor) bool {
    if (tensor.rank < 0 or tensor.byte_length < 0) {
        return false;
    }
    if (tensor.rank > 0 and tensor.shape == null) {
        return false;
    }
    if (tensor.byte_length > 0 and tensor.data == null) {
        return false;
    }
    return dtypeSize(tensor.dtype) != 0;
}

fn copyTensor(input: NamedTensor) ?NamedTensor {
    if (!validTensor(input.tensor)) {
        return null;
    }

    const name = copyCString(@ptrCast(input.name));
    if (name == null) {
        return null;
    }

    var output = NamedTensor{
        .name = name,
        .tensor = .{
            .dtype = input.tensor.dtype,
            .rank = input.tensor.rank,
            .shape = null,
            .byte_length = input.tensor.byte_length,
            .data = null,
        },
    };

    if (input.tensor.rank > 0) {
        const rank: usize = @intCast(input.tensor.rank);
        const raw_shape = std.c.malloc(@sizeOf(i64) * rank) orelse {
            freeString(output.name);
            return null;
        };
        const shape: [*]i64 = @ptrCast(@alignCast(raw_shape));
        @memcpy(shape[0..rank], input.tensor.shape[0..rank]);
        output.tensor.shape = @ptrCast(shape);
    }

    if (input.tensor.byte_length > 0) {
        const byte_len: usize = @intCast(input.tensor.byte_length);
        const raw_data = std.c.malloc(byte_len) orelse {
            freeString(output.name);
            std.c.free(output.tensor.shape);
            return null;
        };
        const data: [*]u8 = @ptrCast(raw_data);
        const source: [*]const u8 = @ptrCast(input.tensor.data.?);
        @memcpy(data[0..byte_len], source[0..byte_len]);
        output.tensor.data = @ptrCast(data);
    }

    return output;
}

fn engineName(engine: i32) []const u8 {
    return switch (engine) {
        @intFromEnum(Engine.mlx) => "mlx",
        @intFromEnum(Engine.coreml) => "coreml",
        @intFromEnum(Engine.onnx) => "onnx",
        @intFromEnum(Engine.litert) => "litert",
        else => "unknown",
    };
}

fn containsEchoMode(options_json: [*c]const u8) bool {
    if (options_json == null) {
        return false;
    }
    const value = options_json[0..std.mem.len(options_json)];
    return std.mem.indexOf(u8, value, "\"zigRuntimeMode\":\"echo\"") != null or
        std.mem.indexOf(u8, value, "\"zigRuntimeMode\": \"echo\"") != null;
}

fn isEchoPath(model_path: [*c]const u8) bool {
    if (model_path == null) {
        return false;
    }
    const value = model_path[0..std.mem.len(model_path)];
    return std.mem.eql(u8, value, "zig://echo");
}

fn createEchoSession(
    engine: i32,
    model_path: [*c]const u8,
    options_json: [*c]const u8,
) ?*Session {
    const raw = std.c.malloc(@sizeOf(Session)) orelse return null;
    const session: *Session = @ptrCast(@alignCast(raw));
    session.* = .{
        .engine = engine,
        .mode = .echo,
        .model_path = copyCString(model_path),
        .options_json = copyCString(options_json),
        .adapter_handle = null,
        .mlx_handle = null,
    };
    if (session.model_path == null or session.options_json == null) {
        freeString(session.model_path);
        freeString(session.options_json);
        std.c.free(session);
        return null;
    }
    return session;
}

fn createAdapterSession(
    engine: i32,
    model_path: [*c]const u8,
    options_json: [*c]const u8,
    error_out: ?*[*c]u8,
) ?*Session {
    const adapter_handle = cpp.dinf_cpp_runtime_create(
        engine,
        model_path,
        options_json,
        error_out,
    ) orelse return null;

    const raw = std.c.malloc(@sizeOf(Session)) orelse {
        cpp.dinf_cpp_runtime_free(adapter_handle);
        setError(error_out, "failed to allocate Zig adapter session");
        return null;
    };
    const session: *Session = @ptrCast(@alignCast(raw));
    session.* = .{
        .engine = engine,
        .mode = .adapter,
        .model_path = copyCString(model_path),
        .options_json = copyCString(options_json),
        .adapter_handle = adapter_handle,
        .mlx_handle = null,
    };
    if (session.model_path == null or session.options_json == null) {
        freeString(session.model_path);
        freeString(session.options_json);
        std.c.free(session);
        cpp.dinf_cpp_runtime_free(adapter_handle);
        setError(error_out, "failed to allocate Zig adapter session strings");
        return null;
    }
    return session;
}

fn createMlxSession(
    engine: i32,
    model_path: [*c]const u8,
    options_json: [*c]const u8,
    error_out: ?*[*c]u8,
) ?*Session {
    const path_value = model_path[0..std.mem.len(model_path)];
    const mlx_handle = mlx_backend.createSession(
        std.heap.c_allocator,
        std.Io.Threaded.global_single_threaded.io(),
        path_value,
    ) catch |err| {
        setError(error_out, mlx_backend.sessionErrorMessage(err));
        return null;
    };
    const raw = std.c.malloc(@sizeOf(Session)) orelse {
        mlx_handle.deinit();
        return null;
    };
    const session: *Session = @ptrCast(@alignCast(raw));
    session.* = .{
        .engine = engine,
        .mode = .mlx,
        .model_path = copyCString(model_path),
        .options_json = copyCString(options_json),
        .adapter_handle = null,
        .mlx_handle = mlx_handle,
    };
    if (session.model_path == null or session.options_json == null) {
        freeString(session.model_path);
        freeString(session.options_json);
        mlx_handle.deinit();
        std.c.free(session);
        return null;
    }
    return session;
}

export fn dart_inference_runtime_backend_json() [*c]u8 {
    return copyBackendJson();
}

export fn dart_inference_runtime_create(
    engine: i32,
    model_path: [*c]const u8,
    options_json: [*c]const u8,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    if (model_path == null) {
        setError(error_out, "model_path is null");
        return null;
    }
    const session = if (isEchoPath(model_path) or containsEchoMode(options_json))
        createEchoSession(engine, model_path, options_json)
    else if (engine == @intFromEnum(Engine.mlx))
        createMlxSession(engine, model_path, options_json, error_out)
    else
        createAdapterSession(engine, model_path, options_json, error_out);
    const resolved = session orelse {
        if (error_out) |out| {
            if (out.* == null) {
                setError(error_out, "failed to allocate Zig runtime session");
            }
        }
        return null;
    };
    return @ptrCast(resolved);
}

export fn dart_inference_runtime_free(handle: ?*anyopaque) void {
    const raw = handle orelse return;
    const session: *Session = @ptrCast(@alignCast(raw));
    switch (session.mode) {
        .echo => {},
        .adapter => cpp.dinf_cpp_runtime_free(session.adapter_handle),
        .mlx => if (session.mlx_handle) |mlx_handle| mlx_handle.deinit(),
    }
    freeString(session.model_path);
    freeString(session.options_json);
    std.c.free(session);
}

export fn dart_inference_runtime_run(
    handle: ?*anyopaque,
    inputs: [*c]const NamedTensor,
    input_count: isize,
    outputs: ?*[*c]NamedTensor,
    output_count: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    if (handle == null) {
        setError(error_out, "runtime session is null");
        return 1;
    }
    if (outputs == null or output_count == null) {
        setError(error_out, "output pointers are null");
        return 1;
    }
    outputs.?.* = null;
    output_count.?.* = 0;
    if (input_count < 0) {
        setError(error_out, "input_count is negative");
        return 1;
    }
    if (input_count > 0 and inputs == null) {
        setError(error_out, "input pointer is null");
        return 1;
    }

    const session: *Session = @ptrCast(@alignCast(handle.?));
    switch (session.mode) {
        .echo => {},
        .mlx => return runMlxSession(session.mlx_handle.?, inputs, input_count, error_out),
        .adapter => return cpp.dinf_cpp_runtime_run(
            session.adapter_handle,
            inputs,
            input_count,
            outputs,
            output_count,
            error_out,
        ),
    }

    const count: usize = @intCast(input_count);
    if (count == 0) {
        return 0;
    }
    const raw = std.c.malloc(@sizeOf(NamedTensor) * count) orelse {
        setError(error_out, "failed to allocate output tensor array");
        return 1;
    };
    const out_items: [*]NamedTensor = @ptrCast(@alignCast(raw));
    var produced: usize = 0;
    while (produced < count) : (produced += 1) {
        const copied = copyTensor(inputs[produced]) orelse {
            dart_inference_runtime_free_tensors(@ptrCast(out_items), @intCast(produced));
            setError(error_out, "failed to copy output tensor");
            return 1;
        };
        out_items[produced] = copied;
    }

    outputs.?.* = @ptrCast(out_items);
    output_count.?.* = @intCast(count);
    return 0;
}

fn runMlxSession(
    mlx_session: *const mlx_backend.Session,
    inputs: [*c]const NamedTensor,
    input_count: isize,
    error_out: ?*[*c]u8,
) i32 {
    const count: usize = @intCast(input_count);
    const allocator = std.heap.c_allocator;
    const views = allocator.alloc(mlx_backend.InputTensor, count) catch {
        setError(error_out, "failed to allocate Zig MLX input view array");
        return 1;
    };
    defer allocator.free(views);
    var index: usize = 0;
    while (index < count) : (index += 1) {
        const tensor = inputs[index].tensor;
        if (!validTensor(tensor)) {
            setError(error_out, "Zig-owned MLX backend received an invalid tensor.");
            return 1;
        }
        views[index] = .{
            .dtype = tensor.dtype,
            .rank = tensor.rank,
            .shape = @ptrCast(tensor.shape),
            .byte_length = tensor.byte_length,
            .data = tensor.data,
        };
    }

    const batch = mlx_backend.prepareInputBatch(
        allocator,
        if (count == 0) null else views.ptr,
        count,
    ) catch |err| {
        setError(error_out, mlx_backend.tensorErrorMessage(err));
        return 1;
    };
    defer batch.deinit();

    if (mlx_backend.versionString(allocator)) |version| {
        defer allocator.free(version);
        const message = std.fmt.allocPrint(
            allocator,
            "Zig-owned MLX backend converted {d} input tensor(s) to mlx-c arrays for {s} with mlx-c {s}, but model execution is not implemented yet.",
            .{ count, mlx_backend.artifactKindName(mlx_session.artifact_kind), version },
        ) catch {
            setError(error_out, mlx_backend.unavailableMessage());
            return 1;
        };
        defer allocator.free(message);
        setError(error_out, message);
        return 1;
    } else |_| {}
    setError(error_out, mlx_backend.unavailableMessage());
    return 1;
}

export fn dart_inference_runtime_free_tensors(tensors: [*c]NamedTensor, count: isize) void {
    if (tensors == null) {
        return;
    }
    if (count <= 0) {
        std.c.free(tensors);
        return;
    }
    const item_count: usize = @intCast(count);
    var index: usize = 0;
    while (index < item_count) : (index += 1) {
        freeString(tensors[index].name);
        std.c.free(tensors[index].tensor.shape);
        std.c.free(tensors[index].tensor.data);
    }
    std.c.free(tensors);
}

export fn dart_inference_runtime_free_string(value: [*c]u8) void {
    freeString(value);
}

export fn dart_inference_runtime_alloc(byte_length: isize) ?*anyopaque {
    if (byte_length <= 0) {
        return null;
    }
    const len: usize = @intCast(byte_length);
    return std.c.malloc(len);
}

export fn dart_inference_runtime_free_buffer(value: ?*anyopaque) void {
    if (value) |ptr| {
        std.c.free(ptr);
    }
}

export fn dart_inference_runtime_memory_info_json() [*c]u8 {
    return cpp.dinf_cpp_runtime_memory_info_json();
}

export fn dart_inference_runtime_diagnostics_json(handle: ?*anyopaque) [*c]u8 {
    const raw = handle orelse return copyString("{}");
    const session: *Session = @ptrCast(@alignCast(raw));
    if (session.mode == .adapter) {
        return cpp.dinf_cpp_runtime_diagnostics_json(session.adapter_handle);
    }
    if (session.mode == .mlx) {
        const mlx_session_json = mlx_backend.sessionDiagnosticsJson(
            session.mlx_handle.?,
            std.heap.c_allocator,
        ) catch return copyString("{}");
        defer std.heap.c_allocator.free(mlx_session_json);
        const text = std.fmt.allocPrintSentinel(
            std.heap.c_allocator,
            "{{\"native_backend\":\"zig\",\"engine\":\"{s}\",\"mode\":\"mlx\",\"zig_version\":\"{s}\",\"mlx_backend\":{s},\"mlx_session\":{s}}}",
            .{ engineName(session.engine), pinned_zig_version, mlx_backend.status_json, mlx_session_json },
            0,
        ) catch return copyString("{}");
        return @ptrCast(text.ptr);
    }
    const mode = switch (session.mode) {
        .echo => "echo",
        .mlx => unreachable,
        .adapter => unreachable,
    };
    const text = std.fmt.allocPrintSentinel(
        std.heap.c_allocator,
        "{{\"native_backend\":\"zig\",\"engine\":\"{s}\",\"mode\":\"{s}\",\"zig_version\":\"{s}\",\"mlx_backend\":{s}}}",
        .{ engineName(session.engine), mode, pinned_zig_version, mlx_backend.status_json },
        0,
    ) catch return copyString("{}");
    return @ptrCast(text.ptr);
}

test "backend json is stable" {
    try std.testing.expect(std.mem.indexOf(u8, backendJson(), "\"native_backend\":\"zig\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, backendJson(), "\"mlx_backend\":{\"owner\":\"zig\",\"api\":\"mlx-c\"") != null);
    try std.testing.expectEqualStrings(pinned_zig_version, builtin.zig_version_string);
}
