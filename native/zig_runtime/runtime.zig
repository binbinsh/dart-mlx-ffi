const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi.zig");
const coreml = @import("coreml.zig");
const diag = @import("diag.zig");
const hf = @import("hf.zig");
const rt_env = @import("env.zig");
const mlx_backend = @import("mlx_backend.zig");
const open_opts = @import("open_opts.zig");
const policy = @import("policy.zig");
const resolve = @import("resolve.zig");

const pinned_zig_version = "0.16.0";
const Engine = policy.Engine;
const Dtype = abi.Dtype;
const NativeTensor = abi.NativeTensor;
const NamedTensor = abi.NamedTensor;
const ResolveArtifact = resolve.Artifact;
const ResolveResult = resolve.Result;
const copyString = abi.copyString;
const freeString = abi.freeString;
const copyCString = abi.copyCString;
const validTensor = abi.validTensor;
const copyTensor = abi.copyTensor;
const mlx_artifacts = "mlxfn\x1edart_inference_linear";

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

const InfoAbi = extern struct {
    native_backend: [*c]const u8,
    zig_version: [*c]const u8,
    async_model: [*c]const u8,
    abi: [*c]const u8,
    mlx_owner: [*c]const u8,
    mlx_api: [*c]const u8,
    mlx_linked: i32,
    mlx_enabled: i32,
    mlx_artifacts: [*c]const u8,
};

const MemoryAbi = extern struct {
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

    pub fn dinf_cpp_mem(out: *MemoryAbi) void {
        out.* = emptyMemoryInfo();
        out.native_backend = "zig-test".ptr;
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
    extern fn dinf_cpp_mem(out: *MemoryAbi) void;
    extern fn dinf_cpp_diag_json(session: ?*anyopaque) [*c]u8;
};

fn setError(error_out: ?*[*c]u8, message: []const u8) void {
    if (error_out) |out| {
        out.* = copyString(message);
    }
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

fn entrySlice(entries: [*c]const open_opts.Entry, count: isize) []const open_opts.Entry {
    if (entries == null or count <= 0) {
        return &.{};
    }
    return entries[0..@intCast(count)];
}

fn emptyMemoryInfo() MemoryAbi {
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

fn linuxMemoryInfo(allocator: std.mem.Allocator) MemoryError!MemoryAbi {
    const status = try readProcStatus(allocator);
    defer allocator.free(status);
    const peak = procStatusKb(status, "VmHWM:");
    const rss = procStatusKb(status, "VmRSS:");
    var info = emptyMemoryInfo();
    info.native_backend = "zig".ptr;
    info.peak_memory_bytes = peak;
    info.vm_hwm = peak;
    info.vm_rss = rss;
    return info;
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

export fn dinf_info(out: ?*InfoAbi) i32 {
    const info = out orelse return 1;
    info.* = .{
        .native_backend = "zig".ptr,
        .zig_version = pinned_zig_version.ptr,
        .async_model = "std.Io-ready".ptr,
        .abi = "dinf_v1".ptr,
        .mlx_owner = mlx_backend.owner.ptr,
        .mlx_api = mlx_backend.api.ptr,
        .mlx_linked = if (mlx_backend.linked) 1 else 0,
        .mlx_enabled = if (mlx_backend.enabled) 1 else 0,
        .mlx_artifacts = mlx_artifacts.ptr,
    };
    return 0;
}

export fn dinf_platform_id() i32 {
    return policy.platformId();
}

export fn dinf_accel_mask(engine: i32) i32 {
    return policy.acceleratorMask(engine);
}

export fn dinf_resolve(
    model_id: [*c]const u8,
    platform: i32,
    requested: i32,
    allow_fallback: i32,
    prefer_mask: i32,
    artifacts: [*c]const ResolveArtifact,
    artifact_count: isize,
    result_out: ?*ResolveResult,
    error_out: ?*[*c]u8,
) i32 {
    const out = result_out orelse {
        setError(error_out, "runtime resolver result_out is null");
        return 1;
    };
    out.* = .{ .engine = -1, .accel_mask = 0, .fallback_engine = -1 };
    const artifact_slice = artifactSlice(artifacts, artifact_count) orelse {
        setError(error_out, "runtime resolver received invalid artifacts");
        return 1;
    };
    const result = resolve.select(
        platform,
        requested,
        allow_fallback != 0,
        prefer_mask,
        artifact_slice,
    ) catch |err| {
        setResolveError(error_out, err, cStringOrEmpty(model_id), requested, platform);
        return 1;
    };
    out.* = result;
    return 0;
}

export fn dinf_fallback(
    platform: i32,
    registered_engines: [*c]const i32,
    registered_count: isize,
    artifacts: [*c]const ResolveArtifact,
    artifact_count: isize,
) i32 {
    const artifact_slice = artifactSlice(artifacts, artifact_count) orelse return -1;
    const registered_slice = intSlice(registered_engines, registered_count) orelse return -1;
    return resolve.fallback(platform, registered_slice, artifact_slice) orelse -1;
}

export fn dinf_artifact_path(
    root_path: [*c]const u8,
    artifact_path: [*c]const u8,
) [*c]u8 {
    const allocator = std.heap.c_allocator;
    const resolved = policy.artifactPath(
        allocator,
        cStringOrEmpty(root_path),
        cStringOrEmpty(artifact_path),
    ) catch return copyString("");
    defer allocator.free(resolved);
    return copyString(resolved);
}

fn artifactSlice(pointer: [*c]const ResolveArtifact, count: isize) ?[]const ResolveArtifact {
    if (count < 0) {
        return null;
    }
    if (count == 0) {
        return &.{};
    }
    if (pointer == null) {
        return null;
    }
    const len: usize = @intCast(count);
    return pointer[0..len];
}

fn intSlice(pointer: [*c]const i32, count: isize) ?[]const i32 {
    if (count < 0) {
        return null;
    }
    if (count == 0) {
        return &.{};
    }
    if (pointer == null) {
        return null;
    }
    const len: usize = @intCast(count);
    return pointer[0..len];
}

fn setResolveError(
    error_out: ?*[*c]u8,
    err: resolve.SelectError,
    model_id: []const u8,
    requested: i32,
    platform: i32,
) void {
    const allocator = std.heap.c_allocator;
    const id = if (model_id.len == 0) "model" else model_id;
    const message = switch (err) {
        error.RequestedUnavailable => std.fmt.allocPrint(
            allocator,
            "Model {s} has no {s} artifact for {s}.",
            .{ id, policy.engineName(requested), policy.platformName(platform) },
        ) catch {
            setError(error_out, "runtime resolver failed");
            return;
        },
        error.NoArtifact => std.fmt.allocPrint(
            allocator,
            "Model {s} has no runtime artifact for {s}.",
            .{ id, policy.platformName(platform) },
        ) catch {
            setError(error_out, "runtime resolver failed");
            return;
        },
    };
    defer allocator.free(message);
    setError(error_out, message);
}

fn openSession(
    engine: i32,
    model_path: [*c]const u8,
    options_json: [*c]const u8,
    force_echo: bool,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    if (model_path == null) {
        setError(error_out, "model_path is null");
        return null;
    }
    if (policy.artifactRemote(optionalCString(model_path))) {
        setError(error_out, "Runtime artifact must be resolved to a local path before native execution.");
        return null;
    }
    const session = if (isEchoPath(model_path) or force_echo)
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

export fn dinf_open(
    engine: i32,
    model_path: [*c]const u8,
    prefer_mask: i32,
    diagnostics: i32,
    num_threads: i32,
    metadata_entries: [*c]const open_opts.Entry,
    metadata_count: isize,
    backend_entries: [*c]const open_opts.Entry,
    backend_count: isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    const allocator = std.heap.c_allocator;
    const metadata = entrySlice(metadata_entries, metadata_count);
    const backend = entrySlice(backend_entries, backend_count);
    const force_echo = open_opts.textEquals(metadata, backend, "zigRuntimeMode", "echo");
    const options = open_opts.build(
        allocator,
        engine,
        prefer_mask,
        diagnostics != 0,
        num_threads,
        metadata,
        backend,
    ) catch {
        setError(error_out, "invalid runtime options");
        return null;
    };
    defer allocator.free(options);
    return openSession(engine, model_path, options.ptr, force_echo, error_out);
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
    abi.freeTensors(tensors, count);
}

export fn dinf_free_str(value: [*c]u8) void {
    freeString(value);
}

export fn dinf_alloc(byte_length: isize) ?*anyopaque {
    return abi.alloc(byte_length);
}

export fn dinf_alloc_tensor(
    dtype: i32,
    shape: [*c]const i64,
    rank: i32,
    byte_length_out: ?*isize,
    error_out: ?*[*c]u8,
) ?*anyopaque {
    return abi.allocTensor(
        dtype,
        shape,
        rank,
        byte_length_out,
        error_out,
    );
}

export fn dinf_free_buf(value: ?*anyopaque) void {
    abi.freeBuf(value);
}

export fn dinf_mem(out: ?*MemoryAbi) i32 {
    const info = out orelse return 1;
    info.* = if (builtin.os.tag == .linux and builtin.abi != .android)
        linuxMemoryInfo(std.heap.c_allocator) catch blk: {
            var fallback = emptyMemoryInfo();
            cpp.dinf_cpp_mem(&fallback);
            fallback.native_backend = "zig".ptr;
            break :blk fallback;
        }
    else blk: {
        var fallback = emptyMemoryInfo();
        cpp.dinf_cpp_mem(&fallback);
        fallback.native_backend = "zig".ptr;
        break :blk fallback;
    };
    return 0;
}

export fn dinf_ort_libs(
    runtime_env_file: [*c]const u8,
    search_roots: [*c]const u8,
    explicit_libraries: [*c]const u8,
    library_dirs: [*c]const u8,
    library_names: [*c]const u8,
) [*c]u8 {
    const allocator = std.heap.c_allocator;
    const text = rt_env.ortLibsText(
        allocator,
        std.Io.Threaded.global_single_threaded.io(),
        optionalCString(runtime_env_file),
        cStringOrEmpty(search_roots),
        cStringOrEmpty(explicit_libraries),
        cStringOrEmpty(library_dirs),
        cStringOrEmpty(library_names),
    ) catch return copyString("");
    defer allocator.free(text);
    return copyString(text);
}

export fn dinf_coreml_layout(root_path: [*c]const u8) [*c]u8 {
    const allocator = std.heap.c_allocator;
    const text = coreml.layoutText(
        allocator,
        std.Io.Threaded.global_single_threaded.io(),
        cStringOrEmpty(root_path),
    ) catch return copyString("");
    defer allocator.free(text);
    return copyString(text);
}

export fn dinf_hf_ref(
    source_uri: [*c]const u8,
    artifact_path: [*c]const u8,
    repo: [*c]const u8,
    artifact: [*c]const u8,
    revision: [*c]const u8,
) [*c]u8 {
    const allocator = std.heap.c_allocator;
    const text = hf.refText(
        allocator,
        optionalCString(source_uri),
        optionalCString(artifact_path),
        optionalCString(repo),
        optionalCString(artifact),
        optionalCString(revision),
    ) catch return copyString("");
    defer allocator.free(text);
    return copyString(text);
}

export fn dinf_hf_cache_root() [*c]u8 {
    const allocator = std.heap.c_allocator;
    const value = hf.defaultCacheRoot(allocator) catch return copyString("");
    defer allocator.free(value);
    return copyString(value);
}

export fn dinf_hf_token() [*c]u8 {
    const allocator = std.heap.c_allocator;
    const value = hf.authToken(allocator) catch return copyString("");
    const token = value orelse return copyString("");
    defer allocator.free(token);
    return copyString(token);
}

export fn dinf_hf_cache_path(
    cache_root: [*c]const u8,
    repo: [*c]const u8,
    revision: [*c]const u8,
    artifact_path: [*c]const u8,
) [*c]u8 {
    const allocator = std.heap.c_allocator;
    const value = hf.cachePath(
        allocator,
        cStringOrEmpty(cache_root),
        cStringOrEmpty(repo),
        cStringOrEmpty(revision),
        cStringOrEmpty(artifact_path),
    ) catch return copyString("");
    defer allocator.free(value);
    return copyString(value);
}

export fn dinf_hf_dir_artifact(artifact_path: [*c]const u8) i32 {
    return if (hf.isDirectoryArtifact(cStringOrEmpty(artifact_path))) 1 else 0;
}

export fn dinf_diag(handle: ?*anyopaque, count_out: ?*isize) [*c]diag.Entry {
    const count = count_out orelse return null;
    count.* = 0;
    const raw = handle orelse return null;
    const session: *Session = @ptrCast(@alignCast(raw));
    const allocator = std.heap.c_allocator;
    const entries = if (session.mode == .adapter) blk: {
        const json = cpp.dinf_cpp_diag_json(session.adapter_handle);
        if (json == null) return null;
        defer freeString(json);
        const text = json[0..std.mem.len(json)];
        break :blk diag.fromJson(allocator, text) catch return null;
    } else blk: {
        const mode = switch (session.mode) {
            .echo => "echo",
            .mlx => "mlx",
            .adapter => unreachable,
        };
        const mlx_session = if (session.mode == .mlx) session.mlx_handle.? else null;
        break :blk diag.zigSession(
            allocator,
            policy.engineName(session.engine),
            mode,
            pinned_zig_version,
            mlx_session,
        ) catch return null;
    };
    count.* = @intCast(entries.len);
    return entries.ptr;
}

export fn dinf_free_diag(entries: [*c]diag.Entry, count: isize) void {
    diag.freeEntries(entries, count);
}

test "runtime info ABI uses static fields" {
    var info: InfoAbi = undefined;
    try std.testing.expectEqual(@as(i32, 0), dinf_info(&info));
    try std.testing.expectEqualStrings("zig", info.native_backend[0..std.mem.len(info.native_backend)]);
    try std.testing.expectEqualStrings(pinned_zig_version, info.zig_version[0..std.mem.len(info.zig_version)]);
    try std.testing.expectEqualStrings(pinned_zig_version, builtin.zig_version_string);
    try std.testing.expectEqualStrings("mlx-c", info.mlx_api[0..std.mem.len(info.mlx_api)]);
    try std.testing.expectEqual(if (mlx_backend.enabled) @as(i32, 1) else @as(i32, 0), info.mlx_enabled);
    try std.testing.expect(std.mem.indexOf(u8, info.mlx_artifacts[0..std.mem.len(info.mlx_artifacts)], "mlxfn") != null);
    try std.testing.expectEqual(@as(i32, 1), dinf_info(null));
}

test "runtime capabilities are reported from Zig" {
    try std.testing.expectEqual(policy.platformId(), dinf_platform_id());
    try std.testing.expectEqual(
        policy.accel_ane | policy.accel_gpu | policy.accel_cpu,
        dinf_accel_mask(@intFromEnum(Engine.coreml)),
    );
    try std.testing.expectEqual(
        policy.accel_gpu | policy.accel_npu | policy.accel_cpu,
        dinf_accel_mask(@intFromEnum(Engine.litert)),
    );
}

test "runtime resolver policy is reported from Zig" {
    const artifacts = [_]ResolveArtifact{
        .{
            .engine = @intFromEnum(Engine.coreml),
            .path = "coreml",
            .format = "",
            .target_platforms = "macos",
        },
        .{
            .engine = @intFromEnum(Engine.onnx),
            .path = "model.onnx",
            .format = "",
            .target_platforms = "macos",
        },
    };
    var result: ResolveResult = undefined;
    var error_value: [*c]u8 = null;
    const status = dinf_resolve(
        "demo",
        @intFromEnum(policy.Platform.macos),
        -1,
        1,
        0,
        artifacts[0..].ptr,
        @intCast(artifacts.len),
        &result,
        &error_value,
    );
    defer dinf_free_str(error_value);
    try std.testing.expectEqual(@as(i32, 0), status);
    try std.testing.expectEqual(@as(i32, @intFromEnum(Engine.coreml)), result.engine);
    try std.testing.expectEqual(
        @as(i32, policy.accel_ane | policy.accel_gpu | policy.accel_cpu),
        result.accel_mask,
    );

    const registered = [_]i32{@intFromEnum(Engine.onnx)};
    const fallback = dinf_fallback(
        @intFromEnum(policy.Platform.macos),
        registered[0..].ptr,
        @intCast(registered.len),
        artifacts[0..].ptr,
        @intCast(artifacts.len),
    );
    try std.testing.expectEqual(@as(i32, @intFromEnum(Engine.onnx)), fallback);

    const path = dinf_artifact_path("/models", "model.onnx");
    defer dinf_free_str(path);
    try std.testing.expectEqualStrings("/models/model.onnx", std.mem.span(path));
}

test "runtime mode is selected from typed open entries" {
    const backend = [_]open_opts.Entry{
        .{ .path = "zigRuntimeMode", .kind = 1, .text = "echo", .int_value = 0, .double_value = 0, .bool_value = 0 },
    };
    var error_value: [*c]u8 = null;
    const handle = dinf_open(
        @intFromEnum(Engine.onnx),
        "model.onnx",
        0,
        0,
        0,
        null,
        0,
        backend[0..].ptr,
        @intCast(backend.len),
        &error_value,
    );
    defer dinf_free_str(error_value);
    try std.testing.expect(handle != null);
    defer dinf_close(handle);
    const session: *Session = @ptrCast(@alignCast(handle.?));
    try std.testing.expectEqual(SessionMode.echo, session.mode);
}

test "runtime open rejects unresolved remote artifacts in Zig" {
    var error_value: [*c]u8 = null;
    const handle = openSession(
        @intFromEnum(Engine.onnx),
        "hf://acme/demo/model.onnx",
        "{}",
        false,
        &error_value,
    );
    defer dinf_free_str(error_value);
    try std.testing.expect(handle == null);
    try std.testing.expect(error_value != null);
    try std.testing.expect(std.mem.indexOf(u8, std.mem.span(error_value), "local path") != null);
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
    var info: MemoryAbi = undefined;
    try std.testing.expectEqual(@as(i32, 0), dinf_mem(&info));
    if (builtin.os.tag == .linux and builtin.abi != .android) {
        try std.testing.expectEqualStrings("zig", info.native_backend[0..std.mem.len(info.native_backend)]);
        try std.testing.expect(info.peak_memory_bytes >= info.vm_rss);
        try std.testing.expect(info.vm_hwm >= info.vm_rss);
    }
    try std.testing.expectEqual(@as(i32, 1), dinf_mem(null));
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
