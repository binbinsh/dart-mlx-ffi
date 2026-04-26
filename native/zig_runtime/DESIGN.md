# Dart -> Zig -> Native Runtime Design

## Target Shape

`dart_inference` uses Zig as the only Dart-facing native boundary.

```text
Dart API -> dart_inference_runtime_* C ABI -> Zig runtime -> C/C++/ObjC++ libs
```

The Dart layer stays shallow:

- create/load sessions
- pass typed tensors or native tensor buffers
- receive typed tensor views
- close sessions/output batches

Zig owns:

- ABI validation
- native memory ownership
- request scratch arenas
- backend dispatch
- async/concurrent scheduling
- MLX calls through `mlx-c`

C/C++/Objective-C++ libraries are private adapters behind Zig. They must not be
bound directly from Dart.

## Hot Path ABI

The hot path is `dart_inference_runtime_run`.

It does not use JSON. It receives an array of fixed-layout structs:

- tensor name pointer
- dtype
- rank
- shape pointer
- byte length
- data pointer

JSON is allowed only for cold paths:

- session creation options
- diagnostics
- backend metadata

`RuntimeOptions.diagnostics` must stay opt-in. `run()` should not request or
parse diagnostics JSON on the hot path unless diagnostics were enabled for the
session.

## Memory Ownership

Input memory:

- Dart heap typed data is copied once into reusable native scratch buffers.
- The scratch buffers are owned by the session and reused across runs.
- Names and shapes are cached per session to avoid per-run allocation.
- True zero-copy input uses `NativeTensorBuffer`, which allocates through Zig
  and exposes native memory as Dart typed-data views.

Output memory:

- Native backends return native-owned tensor arrays.
- Dart wraps output bytes with external typed-data views instead of copying.
- `ModelOutputs.close()` releases the native output batch immediately.
- A finalizer releases the native output batch if the caller does not close it.

Backend memory:

- Model weights and provider sessions stay native.
- Recurrent caches and temporary tensors should be allocated in Zig/backend
  arenas, not Dart heap objects.
- Shape/name metadata should be interned per session.

## Copy Budget

Default safe path:

- Dart heap input -> native scratch: one copy
- native output -> Dart external view: zero copies

Performance path:

- `NativeTensorBuffer` -> Zig runtime -> native backend: zero copies
- native output -> Dart external view: zero copies

Avoid:

- per-op Dart FFI calls
- per-run JSON parsing
- per-run name/string allocation
- output `Uint8List.fromList`
- Dart-side tensor algebra hot loops

## Async Model

The ABI is synchronous today because every target can support it. Zig remains
`std.Io` ready:

- session owns an execution context
- long-running inference can move to `io.async`
- streams should expose coarse-grained chunk callbacks or pull APIs
- cancellation should be represented as an explicit session/request operation,
  not by abandoning Dart futures while native work keeps running

## Migration Rules

- Public Dart imports expose `runtime.dart`, `models.dart`, and a shallow
  `dart_inference.dart`.
- No public raw `mlx-c` binding surface.
- MLX calls move behind Zig. Default runtime resolution may select MLX only
  for artifact kinds with implemented Zig executors; explicit MLX sessions are
  created by Zig, validate local safetensors and `.mlxfn` artifact layouts,
  parse MLX metadata and quantization fields, load Apple `mlx-c` weight maps or
  imported functions into Zig-owned session state, materialize `mlx_array`
  outputs into the runtime tensor ABI, and execute registered Zig MLX
  architectures directly. Exported `.mlxfn` bundles are the generic
  imported-function path; `dart_inference_linear` is the first safetensors
  executor template; unregistered architectures must fail from the Zig-owned
  `mlx-c` path rather than the C/C++ adapter.
- The Apple build hook produces `dart_inference_mlx_c` as a private `mlx-c`
  dependency for Zig. It is not a Dart-facing code asset API.
- The former Dart-facing MLX raw/shim/stable APIs and C++ bridge are removed
  from the package source.
- New native entry points use `dart_inference_runtime_*`.
