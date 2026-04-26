const std = @import("std");
const builtin = @import("builtin");
const coreml = @import("coreml.zig");
const rt_env = @import("env.zig");
const mlx_backend = @import("mlx_backend.zig");
const policy = @import("policy.zig");

const pinned_zig_version = "0.16.0";
const Engine = policy.Engine;

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

const MemoryError = error{
    FileOpen,
    FileTooLarge,
    OutOfMemory,
};

const TensorAllocError = error{
    UnsupportedDtype,
    InvalidShape,
    ByteLengthOverflow,
    OutOfMemory,
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
    pub fn dinf_cpp_open(
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

    pub fn dinf_cpp_close(session: ?*anyopaque) void {
        _ = session;
    }

    pub fn dinf_cpp_run(
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

    pub fn dinf_cpp_mem_json() [*c]u8 {
        return copyString("{\"peak_memory_bytes\":0,\"native_backend\":\"zig-test\"}");
    }

    pub fn dinf_cpp_diag_json(session: ?*anyopaque) [*c]u8 {
        _ = session;
        return copyString("{}");
    }
} else struct {
    extern fn dinf_cpp_open(
        engine: i32,
        model_path: [*c]const u8,
        options_json: [*c]const u8,
        error_out: ?*[*c]u8,
    ) ?*anyopaque;
    extern fn dinf_cpp_close(session: ?*anyopaque) void;
    extern fn dinf_cpp_run(
        session: ?*anyopaque,
        inputs: [*c]const NamedTensor,
        input_count: isize,
        outputs: ?*[*c]NamedTensor,
        output_count: ?*isize,
        error_out: ?*[*c]u8,
    ) i32;
    extern fn dinf_cpp_mem_json() [*c]u8;
    extern fn dinf_cpp_diag_json(session: ?*anyopaque) [*c]u8;
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
    return "{\"native_backend\":\"zig\",\"zig_version\":\"0.16.0\",\"async_model\":\"std.Io-ready\",\"abi\":\"dinf_v1\",\"mlx_backend\":" ++ mlx_backend.status_json ++ "}";
}

fn copyBackendJson() [*c]u8 {
    return copyString(backendJson());
}

fn optionalCString(value: [*c]const u8) ?[]const u8 {
    if (value == null) {
        return null;
    }
    const len = std.mem.len(value);
    if (len == 0) {
        return null;
    }
    return value[0..len];
}

fn cStringOrEmpty(value: [*c]const u8) []const u8 {
    return optionalCString(value) orelse "";
}

fn linuxMemoryInfoJson(allocator: std.mem.Allocator) MemoryError![*c]u8 {
    const status = try readProcStatus(allocator);
    defer allocator.free(status);
    const peak = procStatusKb(status, "VmHWM:");
    const rss = procStatusKb(status, "VmRSS:");
    const text = std.fmt.allocPrintSentinel(
        allocator,
        "{{\"native_backend\":\"zig\",\"peak_memory_bytes\":{d},\"vm_hwm\":{d},\"vm_rss\":{d}}}",
        .{ peak, peak, rss },
        0,
    ) catch return error.OutOfMemory;
    return @ptrCast(text.ptr);
}

fn readProcStatus(allocator: std.mem.Allocator) MemoryError![]u8 {
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

fn procStatusKb(status: []const u8, key: []const u8) u64 {
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

fn dtypeSize(dtype: i32) usize {
    return switch (dtype) {
        @intFromEnum(Dtype.float32), @intFromEnum(Dtype.int32) => 4,
        @intFromEnum(Dtype.int64), @intFromEnum(Dtype.float64) => 8,
        @intFromEnum(Dtype.uint8), @intFromEnum(Dtype.boolean) => 1,
        @intFromEnum(Dtype.float16) => 2,
        else => 0,
    };
}

fn tensorByteLength(
    dtype: i32,
    shape: [*c]const i64,
    rank: i32,
) TensorAllocError!usize {
    const item_size = dtypeSize(dtype);
    if (item_size == 0) {
        return error.UnsupportedDtype;
    }
    if (rank < 0) {
        return error.InvalidShape;
    }
    if (rank > 0 and shape == null) {
        return error.InvalidShape;
    }

    var elements: usize = 1;
    const count: usize = @intCast(rank);
    var index: usize = 0;
    while (index < count) : (index += 1) {
        const dim = shape[index];
        if (dim < 0) {
            return error.InvalidShape;
        }
        const dim_usize: usize = @intCast(dim);
        elements = std.math.mul(usize, elements, dim_usize) catch
            return error.ByteLengthOverflow;
    }
    return std.math.mul(usize, elements, item_size) catch
        error.ByteLengthOverflow;
}

fn tensorAllocErrorMessage(err: TensorAllocError) []const u8 {
    return switch (err) {
        error.UnsupportedDtype => "Zig runtime tensor allocation received an unsupported dtype.",
        error.InvalidShape => "Zig runtime tensor allocation received an invalid shape.",
        error.ByteLengthOverflow => "Zig runtime tensor allocation byte length overflowed.",
        error.OutOfMemory => "Zig runtime tensor allocation ran out of memory.",
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
    const expected = tensorByteLength(
        tensor.dtype,
        @ptrCast(tensor.shape),
        tensor.rank,
    ) catch return false;
    return @as(usize, @intCast(tensor.byte_length)) == expected;
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

fn runtimeModeIsEcho(options_json: [*c]const u8) bool {
    if (options_json == null) {
        return false;
    }
    const value = options_json[0..std.mem.len(options_json)];
    if (value.len == 0) {
        return false;
    }
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        std.heap.c_allocator,
        value,
        .{ .duplicate_field_behavior = .use_last },
    ) catch return false;
    defer parsed.deinit();
    const object = switch (parsed.value) {
        .object => |object| object,
        else => return false,
    };
    const mode = object.get("zigRuntimeMode") orelse return false;
    return switch (mode) {
        .string => |text| std.mem.eql(u8, text, "echo"),
        else => false,
    };
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
    const adapter_handle = cpp.dinf_cpp_open(
        engine,
        model_path,
        options_json,
        error_out,
    ) orelse return null;

    const raw = std.c.malloc(@sizeOf(Session)) orelse {
        cpp.dinf_cpp_close(adapter_handle);
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
        cpp.dinf_cpp_close(adapter_handle);
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

export fn dinf_info_json() [*c]u8 {
    return copyBackendJson();
}

export fn dinf_platform_id() i32 {
    return policy.platformId();
}

export fn dinf_caps_json(engine: i32) [*c]u8 {
    const text = std.fmt.allocPrintSentinel(
        std.heap.c_allocator,
        "{{\"native_backend\":\"zig\",\"engine\":\"{s}\",\"platform\":\"{s}\",\"accelerators\":{s}}}",
        .{
            policy.engineName(engine),
            policy.platformName(policy.platformId()),
            policy.acceleratorsJson(engine),
        },
        0,
    ) catch return copyString("{}");
    return @ptrCast(text.ptr);
}

export fn dinf_engine_accels_json(engine: i32) [*c]u8 {
    return copyString(policy.acceleratorsJson(engine));
}

export fn dinf_engine_order_json(platform: i32) [*c]u8 {
    return copyString(policy.engineOrderJson(platform));
}

export fn dinf_mlx_artifact_registered(
    format: [*c]const u8,
    artifact_path: [*c]const u8,
) i32 {
    return if (policy.mlxArtifactRegistered(
        optionalCString(format),
        optionalCString(artifact_path),
    )) 1 else 0;
}

export fn dinf_artifact_matches(
    engine: i32,
    platform: i32,
    target_platforms: [*c]const u8,
    format: [*c]const u8,
    artifact_path: [*c]const u8,
    allow_preview_mlx: i32,
) i32 {
    return if (policy.artifactMatches(
        engine,
        platform,
        optionalCString(target_platforms),
        optionalCString(format),
        optionalCString(artifact_path),
        allow_preview_mlx != 0,
    )) 1 else 0;
}

export fn dinf_open(
    engine: i32,
    model_path: [*c]const u8,
    options_json: [*c]const u8,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    if (model_path == null) {
        setError(error_out, "model_path is null");
        return null;
    }
    const session = if (isEchoPath(model_path) or runtimeModeIsEcho(options_json))
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

export fn dinf_close(handle: ?*anyopaque) void {
    const raw = handle orelse return;
    const session: *Session = @ptrCast(@alignCast(raw));
    switch (session.mode) {
        .echo => {},
        .adapter => cpp.dinf_cpp_close(session.adapter_handle),
        .mlx => if (session.mlx_handle) |mlx_handle| mlx_handle.deinit(),
    }
    freeString(session.model_path);
    freeString(session.options_json);
    std.c.free(session);
}

export fn dinf_run(
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
        .mlx => return runMlxSession(
            session.mlx_handle.?,
            inputs,
            input_count,
            outputs,
            output_count,
            error_out,
        ),
        .adapter => return cpp.dinf_cpp_run(
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
        if (!validTensor(inputs[produced].tensor)) {
            dinf_free_tensors(@ptrCast(out_items), @intCast(produced));
            setError(error_out, "Zig runtime received an invalid tensor.");
            return 1;
        }
        const copied = copyTensor(inputs[produced]) orelse {
            dinf_free_tensors(@ptrCast(out_items), @intCast(produced));
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
    outputs: ?*[*c]NamedTensor,
    output_count: ?*isize,
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
            .name = inputs[index].name,
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

    var output_batch = mlx_backend.executeSession(allocator, mlx_session, batch) catch |err| {
        setMlxExecutionError(allocator, mlx_session, err, error_out);
        return 1;
    };
    defer output_batch.deinit();
    return copyMlxOutputs(&output_batch, outputs, output_count, error_out);
}

fn setMlxExecutionError(
    allocator: std.mem.Allocator,
    mlx_session: *const mlx_backend.Session,
    err: mlx_backend.ExecutionError,
    error_out: ?*[*c]u8,
) void {
    if (mlx_backend.versionString(allocator)) |version| {
        defer allocator.free(version);
        const message = std.fmt.allocPrint(
            allocator,
            "Zig-owned MLX backend prepared {s} with mlx-c {s}: {s}",
            .{
                mlx_backend.artifactKindName(mlx_session.artifact_kind),
                version,
                mlx_backend.executionErrorMessage(err),
            },
        ) catch {
            setError(error_out, mlx_backend.executionErrorMessage(err));
            return;
        };
        defer allocator.free(message);
        setError(error_out, message);
    } else |_| {
        setError(error_out, mlx_backend.executionErrorMessage(err));
    }
}

fn copyMlxOutputs(
    output_batch: *mlx_backend.OutputBatch,
    outputs: ?*[*c]NamedTensor,
    output_count: ?*isize,
    error_out: ?*[*c]u8,
) i32 {
    const count = output_batch.tensors.len;
    if (count > std.math.maxInt(isize)) {
        setError(error_out, "Zig-owned MLX backend produced too many output tensors.");
        return 1;
    }
    if (count == 0) {
        outputs.?.* = null;
        output_count.?.* = 0;
        return 0;
    }
    const raw = std.c.malloc(@sizeOf(NamedTensor) * count) orelse {
        setError(error_out, "failed to allocate Zig MLX output tensor array");
        return 1;
    };
    const out_items: [*]NamedTensor = @ptrCast(@alignCast(raw));
    var produced: usize = 0;
    while (produced < count) : (produced += 1) {
        out_items[produced] = moveMlxOutputTensor(&output_batch.tensors[produced]) orelse {
            dinf_free_tensors(@ptrCast(out_items), @intCast(produced));
            setError(error_out, "failed to move Zig MLX output tensor");
            return 1;
        };
    }
    outputs.?.* = @ptrCast(out_items);
    output_count.?.* = @intCast(count);
    return 0;
}

fn moveMlxOutputTensor(output: *mlx_backend.OutputTensor) ?NamedTensor {
    if (output.shape.len > std.math.maxInt(i32) or output.bytes.len > std.math.maxInt(isize)) {
        return null;
    }
    // MLX run paths materialize with c_allocator, so these buffers can be
    // released by dinf_free_tensors after moving ownership.
    const name = copyString(output.name);
    if (name == null) {
        return null;
    }

    var tensor = NativeTensor{
        .dtype = output.dtype,
        .rank = @intCast(output.shape.len),
        .shape = null,
        .byte_length = @intCast(output.bytes.len),
        .data = null,
    };
    if (output.shape.len > 0) {
        tensor.shape = @ptrCast(output.shape.ptr);
        output.shape = &.{};
    }
    if (output.bytes.len > 0) {
        tensor.data = @ptrCast(output.bytes.ptr);
        output.bytes = &.{};
    }
    return .{ .name = name, .tensor = tensor };
}

export fn dinf_free_tensors(tensors: [*c]NamedTensor, count: isize) void {
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

export fn dinf_free_str(value: [*c]u8) void {
    freeString(value);
}

export fn dinf_alloc(byte_length: isize) ?*anyopaque {
    if (byte_length <= 0) {
        return null;
    }
    const len: usize = @intCast(byte_length);
    return std.c.malloc(len);
}

export fn dinf_alloc_tensor(
    dtype: i32,
    shape: [*c]const i64,
    rank: i32,
    byte_length_out: ?*isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    const out = byte_length_out orelse {
        setError(error_out, "byte_length_out is null");
        return null;
    };
    out.* = 0;
    const byte_length = tensorByteLength(dtype, shape, rank) catch |err| {
        setError(error_out, tensorAllocErrorMessage(err));
        return null;
    };
    if (byte_length > std.math.maxInt(isize)) {
        setError(error_out, tensorAllocErrorMessage(error.ByteLengthOverflow));
        return null;
    }
    out.* = @intCast(byte_length);
    if (byte_length == 0) {
        return null;
    }
    return std.c.malloc(byte_length) orelse {
        out.* = 0;
        setError(error_out, tensorAllocErrorMessage(error.OutOfMemory));
        return null;
    };
}

export fn dinf_free_buf(value: ?*anyopaque) void {
    if (value) |ptr| {
        std.c.free(ptr);
    }
}

export fn dinf_mem_json() [*c]u8 {
    if (builtin.os.tag == .linux and builtin.abi != .android) {
        return linuxMemoryInfoJson(std.heap.c_allocator) catch cpp.dinf_cpp_mem_json();
    }
    return cpp.dinf_cpp_mem_json();
}

export fn dinf_ort_libs_json(
    runtime_env_file: [*c]const u8,
    search_roots: [*c]const u8,
    explicit_libraries: [*c]const u8,
    library_dirs: [*c]const u8,
    library_names: [*c]const u8,
) [*c]u8 {
    const allocator = std.heap.c_allocator;
    const json = rt_env.ortLibsJson(
        allocator,
        std.Io.Threaded.global_single_threaded.io(),
        optionalCString(runtime_env_file),
        cStringOrEmpty(search_roots),
        cStringOrEmpty(explicit_libraries),
        cStringOrEmpty(library_dirs),
        cStringOrEmpty(library_names),
    ) catch return copyString("[]");
    defer allocator.free(json);
    return copyString(json);
}

export fn dinf_coreml_layout_json(root_path: [*c]const u8) [*c]u8 {
    const allocator = std.heap.c_allocator;
    const json = coreml.layoutJson(
        allocator,
        std.Io.Threaded.global_single_threaded.io(),
        cStringOrEmpty(root_path),
    ) catch return copyString("{}");
    defer allocator.free(json);
    return copyString(json);
}

export fn dinf_diag_json(handle: ?*anyopaque) [*c]u8 {
    const raw = handle orelse return copyString("{}");
    const session: *Session = @ptrCast(@alignCast(raw));
    if (session.mode == .adapter) {
        return cpp.dinf_cpp_diag_json(session.adapter_handle);
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
            .{ policy.engineName(session.engine), pinned_zig_version, mlx_backend.status_json, mlx_session_json },
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
        .{ policy.engineName(session.engine), mode, pinned_zig_version, mlx_backend.status_json },
        0,
    ) catch return copyString("{}");
    return @ptrCast(text.ptr);
}

test "backend json is stable" {
    try std.testing.expect(std.mem.indexOf(u8, backendJson(), "\"native_backend\":\"zig\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, backendJson(), "\"mlx_backend\":{\"owner\":\"zig\",\"api\":\"mlx-c\"") != null);
    try std.testing.expectEqualStrings(pinned_zig_version, builtin.zig_version_string);
}

test "runtime capabilities are reported from Zig" {
    try std.testing.expectEqual(policy.platformId(), dinf_platform_id());
    const json = dinf_caps_json(@intFromEnum(Engine.coreml));
    defer dinf_free_str(json);
    const text = std.mem.span(json);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"native_backend\":\"zig\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"engine\":\"coreml\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"platform\":\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"accelerators\":[\"ane\",\"gpu\",\"cpu\"]") != null);
}

test "runtime resolver policy is reported from Zig" {
    const macos_order = dinf_engine_order_json(@intFromEnum(policy.Platform.macos));
    defer dinf_free_str(macos_order);
    try std.testing.expectEqualStrings("[\"coreml\",\"mlx\",\"onnx\"]", std.mem.span(macos_order));

    const linux_order = dinf_engine_order_json(@intFromEnum(policy.Platform.linux));
    defer dinf_free_str(linux_order);
    try std.testing.expectEqualStrings("[\"onnx\"]", std.mem.span(linux_order));

    const litert_accels = dinf_engine_accels_json(@intFromEnum(Engine.litert));
    defer dinf_free_str(litert_accels);
    try std.testing.expectEqualStrings("[\"gpu\",\"npu\",\"cpu\"]", std.mem.span(litert_accels));
    try std.testing.expectEqual(@as(i32, 1), dinf_mlx_artifact_registered(null, "bundle/function.mlxfn"));
    try std.testing.expectEqual(@as(i32, 0), dinf_artifact_matches(
        @intFromEnum(Engine.mlx),
        @intFromEnum(policy.Platform.macos),
        "macos",
        "mlx-safetensors",
        "model.safetensors",
        0,
    ));
}

test "runtime mode is parsed from Zig-owned options JSON" {
    try std.testing.expect(runtimeModeIsEcho("{\"zigRuntimeMode\":\"echo\"}"));
    try std.testing.expect(runtimeModeIsEcho("{\"diagnostics\":true,\"zigRuntimeMode\" : \"echo\"}"));
    try std.testing.expect(!runtimeModeIsEcho("{\"zigRuntimeMode\":\"adapter\"}"));
    try std.testing.expect(!runtimeModeIsEcho("{\"message\":\"\\\"zigRuntimeMode\\\":\\\"echo\\\"\"}"));
    try std.testing.expect(!runtimeModeIsEcho("{\"zigRuntimeMode\":true}"));
    try std.testing.expect(!runtimeModeIsEcho("{"));
}

test "Linux proc status memory fields parse as bytes" {
    const status =
        \\Name:   dart_inference
        \\VmHWM:       1234 kB
        \\VmRSS:         42 kB
        \\
    ;
    try std.testing.expectEqual(@as(u64, 1234 * 1024), procStatusKb(status, "VmHWM:"));
    try std.testing.expectEqual(@as(u64, 42 * 1024), procStatusKb(status, "VmRSS:"));
    try std.testing.expectEqual(@as(u64, 0), procStatusKb(status, "VmPeak:"));
}

test "memory info is owned by Zig on Linux" {
    const json = dinf_mem_json();
    defer dinf_free_str(json);
    if (builtin.os.tag == .linux and builtin.abi != .android) {
        const text = json[0..std.mem.len(json)];
        try std.testing.expect(std.mem.indexOf(u8, text, "\"native_backend\":\"zig\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, text, "\"peak_memory_bytes\"") != null);
    }
}

test "runtime tensor buffer allocation computes byte length in Zig" {
    const shape = [_]i64{ 2, 3 };
    var byte_length: isize = 0;
    var error_value: [*c]u8 = null;
    const pointer = dinf_alloc_tensor(
        @intFromEnum(Dtype.float32),
        shape[0..].ptr,
        @intCast(shape.len),
        &byte_length,
        &error_value,
    );
    defer dinf_free_buf(pointer);
    defer dinf_free_str(error_value);
    try std.testing.expect(pointer != null);
    try std.testing.expectEqual(@as(isize, 24), byte_length);
    try std.testing.expect(error_value == null);
}

test "runtime tensor buffer allocation keeps zero-sized tensors allocation-free" {
    const shape = [_]i64{ 0, 3 };
    var byte_length: isize = -1;
    var error_value: [*c]u8 = null;
    const pointer = dinf_alloc_tensor(
        @intFromEnum(Dtype.float32),
        shape[0..].ptr,
        @intCast(shape.len),
        &byte_length,
        &error_value,
    );
    defer dinf_free_str(error_value);
    try std.testing.expect(pointer == null);
    try std.testing.expectEqual(@as(isize, 0), byte_length);
    try std.testing.expect(error_value == null);
}

test "runtime tensor buffer allocation rejects invalid shapes in Zig" {
    const shape = [_]i64{-1};
    var byte_length: isize = -1;
    var error_value: [*c]u8 = null;
    const pointer = dinf_alloc_tensor(
        @intFromEnum(Dtype.float32),
        shape[0..].ptr,
        @intCast(shape.len),
        &byte_length,
        &error_value,
    );
    defer dinf_free_str(error_value);
    try std.testing.expect(pointer == null);
    try std.testing.expectEqual(@as(isize, 0), byte_length);
    try std.testing.expect(error_value != null);
    try std.testing.expect(std.mem.indexOf(u8, std.mem.span(error_value), "invalid shape") != null);
}

test "runtime tensor validation rejects dtype shape byte mismatches" {
    var shape = [_]i64{2};
    const data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const invalid = NativeTensor{
        .dtype = @intFromEnum(Dtype.float32),
        .rank = @intCast(shape.len),
        .shape = shape[0..].ptr,
        .byte_length = 4,
        .data = @ptrCast(@constCast(data[0..].ptr)),
    };
    try std.testing.expect(!validTensor(invalid));

    var valid = invalid;
    valid.byte_length = 8;
    try std.testing.expect(validTensor(valid));
}

test "MLX output batch moves into runtime ABI tensors" {
    const allocator = std.heap.c_allocator;
    const tensor_items = try allocator.alloc(mlx_backend.OutputTensor, 1);
    const name = try allocator.dupe(u8, "logits");
    errdefer allocator.free(name);
    const shape = try allocator.dupe(i64, &.{ 1, 4 });
    errdefer allocator.free(shape);
    const bytes = try allocator.dupe(u8, &.{ 1, 2, 3, 4 });
    errdefer allocator.free(bytes);
    tensor_items[0] = .{
        .allocator = allocator,
        .name = name,
        .dtype = @intFromEnum(Dtype.float32),
        .shape = shape,
        .bytes = bytes,
    };
    var batch = mlx_backend.OutputBatch{ .allocator = allocator, .tensors = tensor_items };
    defer batch.deinit();

    var outputs: [*c]NamedTensor = null;
    var output_count: isize = 0;
    var error_value: [*c]u8 = null;
    try std.testing.expectEqual(
        @as(i32, 0),
        copyMlxOutputs(&batch, &outputs, &output_count, &error_value),
    );
    defer dinf_free_tensors(outputs, output_count);
    try std.testing.expectEqual(@as(isize, 1), output_count);
    try std.testing.expectEqualStrings("logits", std.mem.span(outputs[0].name));
    try std.testing.expectEqual(@intFromEnum(Dtype.float32), outputs[0].tensor.dtype);
    try std.testing.expectEqual(@as(i32, 2), outputs[0].tensor.rank);
    try std.testing.expectEqual(@as(i64, 1), outputs[0].tensor.shape[0]);
    try std.testing.expectEqual(@as(i64, 4), outputs[0].tensor.shape[1]);
    try std.testing.expectEqual(@as(isize, 4), outputs[0].tensor.byte_length);
    const output_bytes: [*]const u8 = @ptrCast(outputs[0].tensor.data.?);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, output_bytes[0..4]);
}
