/// Native Core ML / ONNX Runtime / LiteRT runtime adapters.
library;

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../models/shared/runtime_metadata.dart';
import 'native_bindings.dart' as native;
import 'runtime.dart';

/// Cross-platform memory snapshot from the native runtime bridge.
abstract final class NativeRuntimeMemory {
  static Map<String, Object?> snapshot() {
    final ptr = native.dmf_runtime_memory_info_json();
    if (ptr == ffi.nullptr) {
      return const <String, Object?>{};
    }
    try {
      final decoded = jsonDecode(ptr.cast<Utf8>().toDartString());
      if (decoded is Map) {
        return Map<String, Object?>.from(decoded);
      }
      return const <String, Object?>{};
    } finally {
      native.dmf_runtime_free_string(ptr);
    }
  }
}

/// ModelRuntime implementation backed by the bundled native runtime bridge.
final class NativeModelRuntime implements ModelRuntime {
  NativeModelRuntime(this.engine);

  @override
  RuntimeCapabilities get capabilities => RuntimeCapabilities(
    engine: engine,
    platform: RuntimePlatformCurrent.current(),
    accelerators: switch (engine) {
      RuntimeEngine.coreml => const [
        Accelerator.ane,
        Accelerator.gpu,
        Accelerator.cpu,
      ],
      RuntimeEngine.onnx => const [Accelerator.gpu, Accelerator.cpu],
      RuntimeEngine.litert => const [
        Accelerator.gpu,
        Accelerator.npu,
        Accelerator.cpu,
      ],
      RuntimeEngine.mlx => const [Accelerator.gpu, Accelerator.cpu],
    },
  );

  final RuntimeEngine engine;

  @override
  ModelSession load(ModelBundle bundle, RuntimeOptions options) {
    if (bundle.artifact.engine != engine) {
      throw ArgumentError(
        'Artifact engine ${bundle.artifact.engine.name} does not match '
        'runtime ${engine.name}.',
      );
    }
    if (bundle.artifact.path.startsWith('hf://')) {
      throw StateError(
        'Runtime artifact ${bundle.artifact.path} must be resolved to a local '
        'path before native execution. Use RuntimeRegistry.loadAsync(), '
        'HuggingFaceArtifactCache.resolve(), benchmark/runtime/resolve_hf_artifacts.py, '
        'or provide a local RuntimeArtifact path.',
      );
    }
    final error = calloc<ffi.Pointer<ffi.Char>>();
    final path = bundle.artifactPath.toNativeUtf8().cast<ffi.Char>();
    final optionsJson = jsonEncode({
      'accelerators': options.prefer.map((value) => value.name).toList(),
      'diagnostics': options.diagnostics,
      if (options.numThreads != null) 'numThreads': options.numThreads,
      ...bundle.artifact.metadata,
      ...options.backendOptions,
    }).toNativeUtf8().cast<ffi.Char>();
    try {
      final handle = native.dmf_runtime_create(
        _engineId(engine),
        path,
        optionsJson,
        error,
      );
      if (handle == ffi.nullptr) {
        throw StateError(_takeError(error));
      }
      return _NativeModelSession(handle);
    } finally {
      calloc.free(path);
      calloc.free(optionsJson);
      calloc.free(error);
    }
  }
}

final class _NativeModelSession implements ModelSession {
  _NativeModelSession(this._handle);

  ffi.Pointer<ffi.Void> _handle;

  @override
  Map<String, Object?> get diagnostics {
    _checkOpen();
    return _diagnostics();
  }

  @override
  ModelOutputs run(ModelInputs inputs) {
    _checkOpen();
    final tensors = _encodeInputs(inputs.values);
    final outputPtr = calloc<ffi.Pointer<native.DmfNamedTensor>>();
    final outputCount = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.dmf_runtime_run(
        _handle,
        tensors.pointer,
        tensors.count,
        outputPtr,
        outputCount,
        error,
      );
      if (status != 0) {
        throw StateError(_takeError(error));
      }
      final outputs = _decodeOutputs(outputPtr.value, outputCount.value);
      return ModelOutputs(outputs, diagnostics: _diagnostics());
    } finally {
      if (outputPtr.value != ffi.nullptr) {
        native.dmf_runtime_free_tensors(outputPtr.value, outputCount.value);
      }
      tensors.free();
      calloc.free(outputPtr);
      calloc.free(outputCount);
      calloc.free(error);
    }
  }

  @override
  Stream<ModelOutputs> stream(ModelInputs inputs) async* {
    yield run(inputs);
  }

  @override
  void close() {
    if (_handle == ffi.nullptr) return;
    native.dmf_runtime_free(_handle);
    _handle = ffi.nullptr;
  }

  void _checkOpen() {
    if (_handle == ffi.nullptr) {
      throw StateError('Model session is closed.');
    }
  }

  Map<String, Object?> _diagnostics() {
    final ptr = native.dmf_runtime_diagnostics_json(_handle);
    if (ptr == ffi.nullptr) {
      return const <String, Object?>{};
    }
    try {
      final decoded = jsonDecode(ptr.cast<Utf8>().toDartString());
      if (decoded is Map) {
        return Map<String, Object?>.from(decoded);
      }
      return const <String, Object?>{};
    } finally {
      native.dmf_runtime_free_string(ptr);
    }
  }
}

final class _EncodedInputs {
  _EncodedInputs(this.pointer, this.count, this._allocations);

  final ffi.Pointer<native.DmfNamedTensor> pointer;
  final int count;
  final List<ffi.Pointer<ffi.NativeType>> _allocations;

  void free() {
    for (final allocation in _allocations.reversed) {
      calloc.free(allocation);
    }
  }
}

_EncodedInputs _encodeInputs(Map<String, Object?> values) {
  final entries = values.entries.toList(growable: false);
  final pointer = calloc<native.DmfNamedTensor>(entries.length);
  final allocations = <ffi.Pointer<ffi.NativeType>>[pointer];
  for (var index = 0; index < entries.length; index++) {
    final entry = entries[index];
    final tensor = _asRuntimeTensor(entry.key, entry.value);
    final name = entry.key.toNativeUtf8().cast<ffi.Char>();
    final shape = calloc<ffi.Int64>(tensor.shape.length);
    final data = calloc<ffi.Uint8>(tensor.bytes.lengthInBytes);
    shape.asTypedList(tensor.shape.length).setAll(0, tensor.shape);
    data.asTypedList(tensor.bytes.lengthInBytes).setAll(0, tensor.bytes);
    pointer[index]
      ..name = name
      ..tensor.dtype = _dtypeId(tensor.dtype)
      ..tensor.rank = tensor.shape.length
      ..tensor.shape = shape
      ..tensor.byteLength = tensor.bytes.lengthInBytes
      ..tensor.data = data.cast<ffi.Void>();
    allocations
      ..add(name)
      ..add(shape)
      ..add(data);
  }
  return _EncodedInputs(pointer, entries.length, allocations);
}

RuntimeTensor _asRuntimeTensor(String name, Object? value) {
  if (value is RuntimeTensor) return value;
  if (value is Float32List) return RuntimeTensor.float32([value.length], value);
  if (value is Int32List) return RuntimeTensor.int32([value.length], value);
  if (value is Int64List) return RuntimeTensor.int64([value.length], value);
  if (value is Float64List) return RuntimeTensor.float64([value.length], value);
  if (value is Uint8List) return RuntimeTensor.uint8([value.length], value);
  throw ArgumentError.value(value, name, 'Expected RuntimeTensor or TypedData');
}

Map<String, Object?> _decodeOutputs(
  ffi.Pointer<native.DmfNamedTensor> pointer,
  int count,
) {
  final outputs = <String, Object?>{};
  for (var index = 0; index < count; index++) {
    final named = pointer[index];
    final name = named.name.cast<Utf8>().toDartString();
    final tensor = named.tensor;
    final shape = tensor.shape.asTypedList(tensor.rank).toList();
    final bytes = Uint8List.fromList(
      tensor.data.cast<ffi.Uint8>().asTypedList(tensor.byteLength),
    );
    outputs[name] = RuntimeTensor(
      dtype: _dtypeFromId(tensor.dtype),
      shape: shape,
      bytes: bytes,
    );
  }
  return outputs;
}

String _takeError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native runtime call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.dmf_runtime_free_string(value);
    error.value = ffi.nullptr;
  }
}

int _engineId(RuntimeEngine engine) => switch (engine) {
  RuntimeEngine.mlx => 0,
  RuntimeEngine.coreml => 1,
  RuntimeEngine.onnx => 2,
  RuntimeEngine.litert => 3,
};

int _dtypeId(RuntimeTensorDataType dtype) => switch (dtype) {
  RuntimeTensorDataType.float32 => 1,
  RuntimeTensorDataType.int32 => 2,
  RuntimeTensorDataType.int64 => 3,
  RuntimeTensorDataType.uint8 => 4,
  RuntimeTensorDataType.float64 => 5,
  RuntimeTensorDataType.float16 => 6,
  RuntimeTensorDataType.boolean => 7,
};

RuntimeTensorDataType _dtypeFromId(int id) => switch (id) {
  1 => RuntimeTensorDataType.float32,
  2 => RuntimeTensorDataType.int32,
  3 => RuntimeTensorDataType.int64,
  4 => RuntimeTensorDataType.uint8,
  5 => RuntimeTensorDataType.float64,
  6 => RuntimeTensorDataType.float16,
  7 => RuntimeTensorDataType.boolean,
  _ => throw StateError('Unsupported native tensor dtype: $id'),
};
