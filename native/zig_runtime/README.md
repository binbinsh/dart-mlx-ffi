# Zig Runtime

This directory owns the Dart-facing model runtime ABI.

Current shape:

- Dart binds only to `dart_inference_runtime_*` symbols exported by `runtime.zig`.
- C/C++/Objective-C++ backends are private adapters behind Zig and export
  `dinf_cpp_runtime_*` symbols.
- Hot input paths can allocate native memory through Zig and pass
  `NativeTensorBuffer` views directly into `dart_inference_runtime_run`.
- Zig is pinned by the repository `.zigversion` file and
  `native/zig_runtime/toolchain.json`.

The migration goal is coarse-grained Dart calls into Zig. Hot loops, tensor
packing, streaming, scheduling, and backend dispatch should live on the Zig side
instead of crossing Dart FFI per operation.
