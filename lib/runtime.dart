/// Cross-platform model runtime API.
///
/// This API is the Dart side of the Dart -> native FFI -> native provider runtime.
/// It is intended for coarse model-level inference across Core ML, ONNX
/// Runtime, LiteRT, and native-backed MLX backends.
library;

export 'src/runtime/artifact_resolver.dart';
export 'src/runtime/coreml_layout.dart';
export 'src/runtime/input_json.dart';
export 'src/runtime/native_byte_buffer.dart' show NativeByteBuffer;
export 'src/runtime/native_float32_source.dart'
    show
        NativeFloat32SourceCallback,
        NativeFloat32SourcesCallback,
        nativeFloat32SourceLength,
        withNativeFloat32Source,
        withNativeFloat32Sources;
export 'src/runtime/native_float64_source.dart'
    show
        NativeFloat64SourceCallback,
        NativeFloat64SourcesCallback,
        nativeFloat64SourceLength,
        withNativeFloat64Source,
        withNativeFloat64Sources;
export 'src/runtime/native_int32_source.dart'
    show
        NativeInt32SourceCallback,
        NativeInt32SourcesCallback,
        nativeInt32SourceLength,
        withNativeInt32Source,
        withNativeInt32Sources;
export 'src/runtime/native_int64_source.dart'
    show
        NativeInt64SourceCallback,
        NativeInt64SourcesCallback,
        nativeInt64SourceLength,
        withNativeInt64Source,
        withNativeInt64Sources;
export 'src/runtime/native_tensor_buffers.dart'
    show
        nativeBooleanBuffer,
        nativeFloat32Buffer,
        nativeFloat64Buffer,
        nativeInt32Buffer,
        nativeInt64Buffer,
        nativeTensorBufferFromTypedData,
        nativeUint8Buffer;
export 'src/runtime/native_runtime.dart'
    show
        NativeModelRuntime,
        NativeRuntimeBackend,
        NativeRuntimeMemory,
        NativeTensorBuffer;
export 'src/runtime/onnx.dart';
export 'src/runtime/runtime.dart';
export 'src/runtime/runtime_deps.dart';
