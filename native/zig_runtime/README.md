# Zig Runtime

This directory owns the Dart-facing model runtime ABI.

Current shape:

- Dart binds only to `dart_inference_runtime_*` symbols exported by `runtime.zig`.
- C/C++/Objective-C++ backends are private adapters behind Zig and export
  `dinf_cpp_runtime_*` symbols.
- Apple builds also link Zig to a private `dart_inference_mlx_c` library that
  exposes `mlx-c`; explicit MLX sessions are created by Zig, discover local
  safetensors and `.mlxfn` artifact layouts, parse local MLX metadata and
  quantization fields, load safetensors weight maps or imported functions on
  Apple, and `run()` enters Zig-side managed tensor-to-`mlx_array` conversion
  plus the Zig-owned output materialization path. Exported `.mlxfn` bundles run
  through `mlx_imported_function_apply`; a minimal `dart_inference_linear`
  executor template is wired through `mlx_matmul`/`mlx_add`; other architectures
  return an executor-not-implemented error from Zig.
- The Dart runtime registry includes MLX, but the default resolver only selects
  MLX automatically for implemented `.mlxfn`/`mlx-function` artifacts. Explicit
  MLX safetensors loads are still allowed for diagnostics and executor work.
- Hot input paths can allocate native memory through Zig and pass
  `NativeTensorBuffer` views directly into `dart_inference_runtime_run`.
- Zig is pinned by the repository `.zigversion` file and
  `native/zig_runtime/toolchain.json`.

The migration goal is coarse-grained Dart calls into Zig. Hot loops, tensor
packing, streaming, scheduling, and backend dispatch should live on the Zig side
instead of crossing Dart FFI per operation.
