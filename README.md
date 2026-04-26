# dart_inference

`dart_inference` is a Dart and Flutter FFI package for cross-platform model
inference through a pinned Zig runtime.

The native call stack is intentionally narrow:

```text
Dart API -> dart_inference_runtime_* ABI -> Zig -> private C/C++/ObjC++ libs
```

Dart no longer exposes raw MLX C bindings or a per-op tensor algebra layer.
Model sessions, tensor packing, backend dispatch, native memory ownership, and
future MLX calls belong behind the Zig runtime boundary.

## Status

- Package name: `dart_inference`
- Version format: `1.yyyy.commit-count`
- Pinned Zig release: `0.16.0`
- Dart-facing native symbols: `dart_inference_runtime_*`
- Private adapter symbols: `dinf_cpp_runtime_*`
- Current native backends: Core ML, ONNX Runtime, LiteRT
- MLX: reserved for a Zig-owned backend; it is not selected by default until
  that backend is implemented and registered

## Public Entry Points

- `package:dart_inference/dart_inference.dart`
- `package:dart_inference/runtime.dart`
- `package:dart_inference/models.dart`

`package:dart_inference/raw.dart` is intentionally empty.

## Runtime Example

```dart
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

void main() {
  final backend = NativeRuntimeBackend.info();
  print(backend);

  final input = NativeTensorBuffer.float32([1, 4]);
  input.asFloat32List().setAll(0, <double>[1, 2, 3, 4]);

  // Pass input.tensor to a ModelSession.run(...) call. NativeTensorBuffer keeps
  // the input outside the Dart heap and avoids the per-run scratch copy.
  input.close();
}
```

For ordinary Dart typed-data inputs, the runtime copies into reusable native
scratch buffers owned by the session. Outputs are exposed as typed-data views
over native memory and released by `ModelOutputs.close()`.

## Native Build

The package build hook builds two dynamic libraries:

- `dart_inference_runtime_adapter`: private C/C++/ObjC++ backend adapter
- `dart_inference_runtime`: Zig ABI library loaded by Dart FFI

Zig is pinned by:

- `.zigversion`
- `native/zig_runtime/toolchain.json`

The hook accepts a pinned Zig executable through `DART_INFERENCE_ZIG`. If that
is not set, it checks `ZIG`, then `.dart_tool/zig/zig-<arch>-<host>-0.16.0/zig`,
then `zig` on `PATH`.

Useful runtime build environment variables:

- `DART_INFERENCE_ENABLE_ORT=1`
- `DART_INFERENCE_ORT_INCLUDE_DIR=/path/to/onnxruntime/include`
- `DART_INFERENCE_ORT_LIBRARY=/path/to/libonnxruntime.so`
- `DART_INFERENCE_ORT_RUNTIME_LIBRARY=/path/to/libonnxruntime.so`
- `DART_INFERENCE_LITERT_LIBRARY=/path/to/libtensorflowlite.so`
- `DART_INFERENCE_LITERT_EXTRA_LIBRARIES=/path/a.so:/path/b.so`

The Dart helpers also read `.dart_inference_runtime_env.json`, or the file named
by `DART_INFERENCE_RUNTIME_ENV_FILE`, for staged native runtime paths.

## Model Helpers

`models.dart` exports runtime-first helpers:

- shared model specs, manifests, tuning, metadata, and small utilities
- Kokoro ONNX TTS helpers
- UniFrontend structured text helpers
- TTS backend catalog and registry helpers

The former raw/shim/stable APIs and old Dart-facing MLX C++ bridge have been
removed from the package source. New provider work should add coarse model
runtime calls behind Zig instead of reintroducing Dart-side per-op FFI.

## Local Validation

```sh
dart analyze
dart test
dart pub publish --dry-run
.dart_tool/zig/zig-x86_64-linux-0.16.0/zig test native/zig_runtime/runtime.zig -lc
```

The exact Zig path varies by host architecture and platform.
