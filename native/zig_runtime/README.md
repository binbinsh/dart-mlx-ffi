# Zig Runtime

This directory owns the Dart-facing model runtime ABI.

Current shape:

- Dart binds only to `dart_inference_runtime_*` symbols exported by `runtime.zig`.
- C/C++/Objective-C++ backends are private adapters behind Zig and export
  `dinf_cpp_runtime_*` symbols.
- Apple builds also link Zig to a private `dart_inference_mlx_c` library that
  exposes `mlx-c`; explicit MLX loads are handled by Zig and currently fail
  before model execution because the executor is not implemented yet.
- Hot input paths can allocate native memory through Zig and pass
  `NativeTensorBuffer` views directly into `dart_inference_runtime_run`.
- Zig is pinned by the repository `.zigversion` file and
  `native/zig_runtime/toolchain.json`.

The migration goal is coarse-grained Dart calls into Zig. Hot loops, tensor
packing, streaming, scheduling, and backend dispatch should live on the Zig side
instead of crossing Dart FFI per operation.
