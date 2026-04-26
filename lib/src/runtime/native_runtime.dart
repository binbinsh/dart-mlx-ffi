/// Native Core ML / ONNX Runtime / LiteRT runtime adapters.
library;

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../models/shared/runtime_metadata.dart';
import 'native_bindings.dart' as native;
import 'runtime.dart';

final _nativeRuntimeTensorBuffers = Expando<NativeTensorBuffer>(
  'NativeTensorBuffer',
);
final _nativeInputFinalizer = Finalizer<ffi.Pointer<ffi.Void>>((pointer) {
  if (pointer != ffi.nullptr) {
    native.freeBuf(pointer);
  }
});

/// Native runtime implementation metadata.
abstract final class NativeRuntimeBackend {
  static Map<String, Object?> info() {
    final ptr = native.infoJson();
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
      native.freeStr(ptr);
    }
  }
}

/// Cross-platform memory snapshot from the native runtime bridge.
abstract final class NativeRuntimeMemory {
  static Map<String, Object?> snapshot() {
    final ptr = native.memJson();
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
      native.freeStr(ptr);
    }
  }
}

/// Zig-owned native tensor input buffer.
///
/// Use this for hot paths that should avoid copying Dart heap typed data into
/// native scratch memory before each inference call. Call [close] when the
/// buffer is no longer needed.
final class NativeTensorBuffer {
  NativeTensorBuffer._(
    this.dtype,
    List<int> shape,
    this.byteLength,
    ffi.Pointer<ffi.Void> pointer,
  ) : shape = List<int>.unmodifiable(shape),
      _pointer = pointer,
      _bytes = _nativeBytes(pointer, byteLength) {
    if (_pointer != ffi.nullptr) {
      _nativeInputFinalizer.attach(this, _pointer, detach: this);
    }
  }

  factory NativeTensorBuffer.allocate({
    required RuntimeTensorDataType dtype,
    required List<int> shape,
  }) {
    if (shape.length > 0x7fffffff) {
      throw RangeError.value(shape.length, 'shape', 'Rank must fit int32');
    }
    final rank = shape.length;
    final shapePointer = rank == 0 ? ffi.nullptr : calloc<ffi.Int64>(rank);
    final byteLength = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      if (rank > 0) {
        shapePointer.asTypedList(rank).setAll(0, shape);
      }
      final pointer = native.allocTensor(
        _dtypeId(dtype),
        shapePointer,
        rank,
        byteLength,
        error,
      );
      final resolvedByteLength = byteLength.value;
      if (pointer == ffi.nullptr) {
        if (error.value != ffi.nullptr) {
          throw StateError(_takeError(error));
        }
        if (resolvedByteLength == 0) {
          return NativeTensorBuffer._(dtype, shape, 0, ffi.nullptr);
        }
      }
      if (pointer == ffi.nullptr) {
        throw StateError('Failed to allocate native tensor buffer.');
      }
      return NativeTensorBuffer._(dtype, shape, resolvedByteLength, pointer);
    } finally {
      if (shapePointer != ffi.nullptr) {
        calloc.free(shapePointer);
      }
      calloc.free(byteLength);
      calloc.free(error);
    }
  }

  factory NativeTensorBuffer.float32(List<int> shape) =>
      NativeTensorBuffer.allocate(
        dtype: RuntimeTensorDataType.float32,
        shape: shape,
      );

  factory NativeTensorBuffer.int32(List<int> shape) =>
      NativeTensorBuffer.allocate(
        dtype: RuntimeTensorDataType.int32,
        shape: shape,
      );

  factory NativeTensorBuffer.int64(List<int> shape) =>
      NativeTensorBuffer.allocate(
        dtype: RuntimeTensorDataType.int64,
        shape: shape,
      );

  factory NativeTensorBuffer.float64(List<int> shape) =>
      NativeTensorBuffer.allocate(
        dtype: RuntimeTensorDataType.float64,
        shape: shape,
      );

  factory NativeTensorBuffer.uint8(List<int> shape) =>
      NativeTensorBuffer.allocate(
        dtype: RuntimeTensorDataType.uint8,
        shape: shape,
      );

  factory NativeTensorBuffer.boolean(List<int> shape) =>
      NativeTensorBuffer.allocate(
        dtype: RuntimeTensorDataType.boolean,
        shape: shape,
      );

  factory NativeTensorBuffer.float16(List<int> shape) =>
      NativeTensorBuffer.allocate(
        dtype: RuntimeTensorDataType.float16,
        shape: shape,
      );

  final RuntimeTensorDataType dtype;
  final List<int> shape;
  final int byteLength;
  ffi.Pointer<ffi.Void> _pointer;
  final Uint8List _bytes;

  bool get isClosed => _pointer == ffi.nullptr && byteLength > 0;

  Uint8List get bytes {
    _checkOpen();
    return _bytes;
  }

  RuntimeTensor get tensor {
    _checkOpen();
    final value = RuntimeTensor(
      dtype: dtype,
      shape: shape,
      bytes: _bytes,
      owner: this,
    );
    _nativeRuntimeTensorBuffers[value] = this;
    return value;
  }

  void copyFrom(TypedData data) {
    final source = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    if (source.lengthInBytes != byteLength) {
      throw ArgumentError.value(
        source.lengthInBytes,
        'data',
        'Expected $byteLength bytes',
      );
    }
    bytes.setAll(0, source);
  }

  Float32List asFloat32List() {
    _checkDtype(RuntimeTensorDataType.float32);
    return bytes.buffer.asFloat32List(bytes.offsetInBytes, byteLength ~/ 4);
  }

  Int32List asInt32List() {
    _checkDtype(RuntimeTensorDataType.int32);
    return bytes.buffer.asInt32List(bytes.offsetInBytes, byteLength ~/ 4);
  }

  Int64List asInt64List() {
    _checkDtype(RuntimeTensorDataType.int64);
    return bytes.buffer.asInt64List(bytes.offsetInBytes, byteLength ~/ 8);
  }

  Float64List asFloat64List() {
    _checkDtype(RuntimeTensorDataType.float64);
    return bytes.buffer.asFloat64List(bytes.offsetInBytes, byteLength ~/ 8);
  }

  Uint8List asUint8List() => bytes;

  ffi.Pointer<ffi.Void> _pointerForRun(int tensorByteLength) {
    _checkOpen();
    if (tensorByteLength != byteLength) {
      throw StateError(
        'Native tensor buffer byte length changed from $byteLength to '
        '$tensorByteLength.',
      );
    }
    return _pointer;
  }

  void close() {
    final pointer = _pointer;
    if (pointer == ffi.nullptr) {
      return;
    }
    _pointer = ffi.nullptr;
    _nativeInputFinalizer.detach(this);
    native.freeBuf(pointer);
  }

  void _checkOpen() {
    if (isClosed) {
      throw StateError('Native tensor buffer is closed.');
    }
  }

  void _checkDtype(RuntimeTensorDataType expected) {
    _checkOpen();
    if (dtype != expected) {
      throw StateError('Native tensor buffer dtype is ${dtype.name}.');
    }
  }
}

/// ModelRuntime implementation backed by the bundled native runtime bridge.
final class NativeModelRuntime implements ModelRuntime {
  NativeModelRuntime(this.engine);

  @override
  RuntimeCapabilities get capabilities => _caps(engine);

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
      final handle = native.open(_engineId(engine), path, optionsJson, error);
      if (handle == ffi.nullptr) {
        throw StateError(_takeError(error));
      }
      return _NativeModelSession(
        handle,
        diagnosticsEnabled: options.diagnostics,
      );
    } finally {
      calloc.free(path);
      calloc.free(optionsJson);
      calloc.free(error);
    }
  }
}

final class _NativeModelSession implements ModelSession {
  _NativeModelSession(this._handle, {required bool diagnosticsEnabled})
    : _diagnosticsEnabled = diagnosticsEnabled;

  ffi.Pointer<ffi.Void> _handle;
  final bool _diagnosticsEnabled;
  final Map<String, ffi.Pointer<ffi.Char>> _namePointers = {};
  final Map<String, _ShapeBuffer> _shapePointers = {};
  final Map<String, _NativeByteBuffer> _inputBuffers = {};
  final _InputTensorArena _inputTensorArena = _InputTensorArena();

  @override
  Map<String, Object?> get diagnostics {
    _checkOpen();
    return _diagnostics();
  }

  @override
  ModelOutputs run(ModelInputs inputs) {
    _checkOpen();
    final tensors = _encodeInputs(inputs.values);
    final outputPtr = calloc<ffi.Pointer<native.NamedTensorAbi>>();
    final outputCount = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    var outputTransferred = false;
    try {
      final status = native.run(
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
      final owner = outputPtr.value == ffi.nullptr
          ? null
          : _NativeOutputOwner(outputPtr.value, outputCount.value);
      outputTransferred = owner != null;
      final outputs = _decodeOutputs(outputPtr.value, outputCount.value, owner);
      return ModelOutputs(
        outputs,
        diagnostics: _diagnosticsEnabled
            ? _diagnostics()
            : const <String, Object?>{},
        release: owner?.close,
      );
    } finally {
      if (!outputTransferred && outputPtr.value != ffi.nullptr) {
        native.freeTensors(outputPtr.value, outputCount.value);
      }
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
    native.close(_handle);
    _handle = ffi.nullptr;
    for (final buffer in _inputBuffers.values) {
      buffer.close();
    }
    _inputBuffers.clear();
    _inputTensorArena.close();
    for (final shape in _shapePointers.values) {
      shape.close();
    }
    _shapePointers.clear();
    for (final name in _namePointers.values) {
      calloc.free(name);
    }
    _namePointers.clear();
  }

  void _checkOpen() {
    if (_handle == ffi.nullptr) {
      throw StateError('Model session is closed.');
    }
  }

  Map<String, Object?> _diagnostics() {
    final ptr = native.diagJson(_handle);
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
      native.freeStr(ptr);
    }
  }

  ffi.Pointer<ffi.Char> _namePointer(String name) {
    return _namePointers.putIfAbsent(
      name,
      () => name.toNativeUtf8().cast<ffi.Char>(),
    );
  }

  ffi.Pointer<ffi.Int64> _shapePointer(List<int> shape) {
    final key = shape.join(',');
    return _shapePointers.putIfAbsent(key, () => _ShapeBuffer(shape)).pointer;
  }

  ffi.Pointer<ffi.Void> _inputDataPointer(String name, RuntimeTensor tensor) {
    final nativeBuffer = _nativeRuntimeTensorBuffers[tensor];
    if (nativeBuffer != null) {
      return nativeBuffer._pointerForRun(tensor.bytes.lengthInBytes);
    }
    final bytes = tensor.bytes;
    if (bytes.isEmpty) {
      return ffi.nullptr;
    }
    return _inputBuffers
        .putIfAbsent(name, _NativeByteBuffer.new)
        .copyFrom(bytes);
  }

  _EncodedInputs _encodeInputs(Map<String, Object?> values) {
    final entries = values.entries.toList(growable: false);
    final pointer = _inputTensorArena.pointerFor(entries.length);
    for (var index = 0; index < entries.length; index++) {
      final entry = entries[index];
      final tensor = _asRuntimeTensor(entry.key, entry.value);
      pointer[index]
        ..name = _namePointer(entry.key)
        ..tensor.dtype = _dtypeId(tensor.dtype)
        ..tensor.rank = tensor.shape.length
        ..tensor.shape = _shapePointer(tensor.shape)
        ..tensor.byteLength = tensor.bytes.lengthInBytes
        ..tensor.data = _inputDataPointer(entry.key, tensor);
    }
    return _EncodedInputs(pointer, entries.length);
  }
}

final class _EncodedInputs {
  _EncodedInputs(this.pointer, this.count);

  final ffi.Pointer<native.NamedTensorAbi> pointer;
  final int count;
}

final class _InputTensorArena {
  ffi.Pointer<native.NamedTensorAbi> pointer = ffi.nullptr;
  int capacity = 0;

  ffi.Pointer<native.NamedTensorAbi> pointerFor(int count) {
    if (count == 0) {
      return ffi.nullptr;
    }
    if (capacity >= count) {
      return pointer;
    }
    close();
    pointer = calloc<native.NamedTensorAbi>(count);
    capacity = count;
    return pointer;
  }

  void close() {
    if (pointer == ffi.nullptr) {
      return;
    }
    calloc.free(pointer);
    pointer = ffi.nullptr;
    capacity = 0;
  }
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
  ffi.Pointer<native.NamedTensorAbi> pointer,
  int count,
  Object? owner,
) {
  final outputs = <String, Object?>{};
  for (var index = 0; index < count; index++) {
    final named = pointer[index];
    final name = named.name.cast<Utf8>().toDartString();
    final tensor = named.tensor;
    final shape = tensor.shape.asTypedList(tensor.rank).toList();
    final bytes = tensor.byteLength == 0 || tensor.data == ffi.nullptr
        ? Uint8List(0)
        : tensor.data.cast<ffi.Uint8>().asTypedList(tensor.byteLength);
    outputs[name] = RuntimeTensor(
      dtype: _dtypeFromId(tensor.dtype),
      shape: shape,
      bytes: bytes,
      owner: owner,
    );
  }
  return outputs;
}

final class _ShapeBuffer {
  _ShapeBuffer(List<int> shape)
    : pointer = calloc<ffi.Int64>(shape.length),
      length = shape.length {
    pointer.asTypedList(length).setAll(0, shape);
  }

  final ffi.Pointer<ffi.Int64> pointer;
  final int length;

  void close() {
    calloc.free(pointer);
  }
}

final class _NativeByteBuffer {
  ffi.Pointer<ffi.Uint8> pointer = ffi.nullptr;
  int capacity = 0;

  ffi.Pointer<ffi.Void> copyFrom(Uint8List bytes) {
    _ensureCapacity(bytes.lengthInBytes);
    pointer.asTypedList(bytes.lengthInBytes).setAll(0, bytes);
    return pointer.cast<ffi.Void>();
  }

  void _ensureCapacity(int byteLength) {
    if (capacity >= byteLength) {
      return;
    }
    close();
    final allocated = native.alloc(byteLength);
    if (allocated == ffi.nullptr) {
      throw StateError('Failed to allocate native input buffer.');
    }
    pointer = allocated.cast<ffi.Uint8>();
    capacity = byteLength;
  }

  void close() {
    if (pointer == ffi.nullptr) {
      return;
    }
    native.freeBuf(pointer.cast<ffi.Void>());
    pointer = ffi.nullptr;
    capacity = 0;
  }
}

final _outputFinalizer = Finalizer<_NativeOutputLease>((lease) {
  lease.release();
});

final class _NativeOutputOwner {
  _NativeOutputOwner(ffi.Pointer<native.NamedTensorAbi> pointer, int count) {
    final lease = _NativeOutputLease(pointer, count);
    _lease = lease;
    _outputFinalizer.attach(this, lease, detach: this);
  }

  _NativeOutputLease? _lease;

  void close() {
    final lease = _lease;
    if (lease == null) {
      return;
    }
    _lease = null;
    _outputFinalizer.detach(this);
    lease.release();
  }
}

final class _NativeOutputLease {
  _NativeOutputLease(this.pointer, this.count);

  final ffi.Pointer<native.NamedTensorAbi> pointer;
  final int count;
  bool _released = false;

  void release() {
    if (_released || pointer == ffi.nullptr) {
      return;
    }
    _released = true;
    native.freeTensors(pointer, count);
  }
}

String _takeError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native runtime call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
  }
}

int _engineId(RuntimeEngine engine) => switch (engine) {
  RuntimeEngine.mlx => 0,
  RuntimeEngine.coreml => 1,
  RuntimeEngine.onnx => 2,
  RuntimeEngine.litert => 3,
};

RuntimeCapabilities _caps(RuntimeEngine engine) {
  final ptr = native.capsJson(_engineId(engine));
  if (ptr == ffi.nullptr) {
    return _fallbackCaps(engine);
  }
  try {
    final decoded = jsonDecode(ptr.cast<Utf8>().toDartString());
    if (decoded is! Map) {
      return _fallbackCaps(engine);
    }
    return RuntimeCapabilities(
      engine: engine,
      platform: _platformByName('${decoded['platform']}'),
      accelerators: _accelerators(decoded['accelerators']),
      details: {'nativeBackend': decoded['native_backend'] ?? 'zig'},
    );
  } finally {
    native.freeStr(ptr);
  }
}

RuntimeCapabilities _fallbackCaps(RuntimeEngine engine) => RuntimeCapabilities(
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

RuntimePlatform _platformByName(String name) => switch (name) {
  'ios' => RuntimePlatform.ios,
  'macos' => RuntimePlatform.macos,
  'windows' => RuntimePlatform.windows,
  'linux' => RuntimePlatform.linux,
  'android' => RuntimePlatform.android,
  _ => RuntimePlatform.unknown,
};

List<Accelerator> _accelerators(Object? value) {
  if (value is! List) {
    return const [];
  }
  return [
    for (final item in value)
      switch ('$item') {
        'ane' => Accelerator.ane,
        'gpu' => Accelerator.gpu,
        'npu' => Accelerator.npu,
        'cpu' => Accelerator.cpu,
        _ => null,
      },
  ].whereType<Accelerator>().toList(growable: false);
}

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

Uint8List _nativeBytes(ffi.Pointer<ffi.Void> pointer, int byteLength) {
  if (byteLength == 0) {
    return Uint8List(0);
  }
  return pointer.cast<ffi.Uint8>().asTypedList(byteLength);
}
