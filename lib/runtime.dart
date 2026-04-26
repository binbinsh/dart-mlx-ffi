/// Cross-platform model runtime API.
///
/// This API is the Dart side of the Dart -> Zig -> native provider runtime.
/// It is intended for coarse model-level inference across Core ML, ONNX
/// Runtime, LiteRT, and Zig-owned MLX backends.
library;

export 'src/models/shared/model_spec.dart'
    show ModelModality, ModelSpec, QuantScheme;
export 'src/models/shared/manifest.dart';
export 'src/models/shared/runtime_metadata.dart';
export 'src/runtime/artifact_resolver.dart';
export 'src/runtime/coreml_layout.dart';
export 'src/runtime/native_runtime.dart'
    show
        NativeModelRuntime,
        NativeRuntimeBackend,
        NativeRuntimeMemory,
        NativeTensorBuffer;
export 'src/runtime/onnx.dart';
export 'src/runtime/runtime.dart';
