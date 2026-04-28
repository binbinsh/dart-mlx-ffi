# dart_inference

`dart_inference` is a Dart and Flutter FFI package for cross-platform model
inference through a pinned Zig runtime.

The native call stack is intentionally narrow:

```text
Dart API -> dinf_* ABI -> Zig -> private C/C++/ObjC++ libs
```

Dart no longer exposes raw MLX C bindings or a per-op tensor algebra layer.
Model sessions, tensor packing, backend dispatch, native memory ownership, and
MLX calls belong behind the Zig runtime boundary.

## Status

- Package name: `dart_inference`
- Version format: `1.yyyy.commit-count`
- Pinned Zig release: `0.16.0`
- Dart-facing native symbols: `dinf_*`
- Private adapter symbols: `dinf_cpp_*`
- Current native backends: Core ML, ONNX Runtime, LiteRT
- MLX: Zig-owned `mlx-c` link layer exists for Apple targets; model execution
  is still disabled until the Zig executor is implemented and registered

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

The package build hook builds two required dynamic libraries:

- `dart_inference_runtime_adapter`: private C/C++/ObjC++ backend adapter
- `dart_inference_runtime`: Zig ABI library loaded by Dart FFI

On iOS and macOS it also builds `dinf_zig_mlx_c`, a private `mlx-c`
dependency linked only by the Zig runtime. Dart never binds that library
directly.

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

### UniFrontend TTS Smoke

`dart_inference:tts_backends_status` can audit UniFrontend provider assets and
smoke local ONNX TTS components through the Dart -> Zig -> ONNX Runtime path:

```sh
dart run dart_inference:tts_backends_status \
  --root /path/to/unifrontend \
  --provider cuda \
  --smoke-onnx
```

CosyVoice2 status is ONNX-target-first: existing `.pt`, `.zip`, and
`.safetensors` files are reported as export sources, while runtime readiness is
based on the required ONNX targets (`flow.encoder.fp32.onnx`, `llm.onnx`, and
`hift.onnx` in addition to the already loadable components).

The provider asset audit is also ONNX-target-first for the remaining local TTS
backends. It reports each target graph path, the source weight file it should be
exported from, missing required ONNX assets, and whether that provider can run
through the pure `Dart -> Zig -> ONNX Runtime` path today.

Catalog-declared TTS ONNX targets also get a generic component bundle in the
status output. When an exported target graph appears on disk, the same bundle
can load it, run it by component name, and build synthetic smoke inputs from
ONNX Runtime metadata without adding provider-specific Dart glue.

Use `--provider tensorrt --trt-cache-dir <dir> --trt-workspace-mb <mb>` to
benchmark TensorRT EP where ONNX Runtime exposes it. Add `--trt-fp16` only when
the graph has been checked for acceptable FP16 parity. The status output also
includes a TensorRT dependency audit, so missing `libnvinfer*` libraries are
reported separately from model-load failures and TensorRT loads are skipped
until the runtime dependencies are visible. `tts_infer`, `tts_server`,
`structured_frontend_infer`, `structured_smoke_infer`, and `onnx_server` use
the same preflight check for strict TensorRT requests.

## Local Validation

```sh
dart analyze
dart test
dart pub publish --dry-run
.dart_tool/zig/zig-x86_64-linux-0.16.0/zig test native/zig_runtime/runtime.zig -lc
```

The exact Zig path varies by host architecture and platform.
