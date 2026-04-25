/// Cross-platform model runtime API.
///
/// This API sits above the MLX tensor API and is intended for model-level
/// inference across MLX, Core ML, ONNX Runtime, and LiteRT artifacts.
library;

export 'src/models/shared/model_spec.dart'
    show ModelModality, ModelSpec, QuantScheme;
export 'src/models/shared/manifest.dart';
export 'src/models/shared/runtime_metadata.dart';
export 'src/runtime/artifact_resolver.dart';
export 'src/runtime/coreml_layout.dart';
export 'src/runtime/native_runtime.dart'
    show NativeModelRuntime, NativeRuntimeMemory;
export 'src/runtime/runtime.dart';
