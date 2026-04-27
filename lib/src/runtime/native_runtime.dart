/// Native Core ML / ONNX Runtime / LiteRT runtime adapters.
library;

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
const _infoListSep = '\x1e';
const _entryPathSep = '\x1f';
const _entryString = 1;
const _entryInt = 2;
const _entryBool = 3;
const _entryMap = 4;
const _entryList = 5;
const _entryDouble = 6;
const _entryNull = 7;

/// Native runtime implementation metadata.
abstract final class NativeRuntimeBackend {
  static Map<String, Object?> info() {
    final out = calloc<native.InfoAbi>();
    try {
      if (native.info(out) != 0) {
        return const <String, Object?>{};
      }
      final value = out.ref;
      return <String, Object?>{
        'native_backend': _staticText(value.nativeBackend),
        'zig_version': _staticText(value.zigVersion),
        'async_model': _staticText(value.asyncModel),
        'abi': _staticText(value.abi),
        'mlx_backend': <String, Object?>{
          'owner': _staticText(value.mlxOwner),
          'api': _staticText(value.mlxApi),
          'linked': value.mlxLinked != 0,
          'enabled': value.mlxEnabled != 0,
          'registered_artifacts': _staticList(value.mlxArtifacts),
        },
      };
    } finally {
      calloc.free(out);
    }
  }
}

String _staticText(ffi.Pointer<ffi.Char> value) {
  if (value == ffi.nullptr) return '';
  return value.cast<Utf8>().toDartString();
}

List<String> _staticList(ffi.Pointer<ffi.Char> value) {
  final text = _staticText(value);
  if (text.isEmpty) return const <String>[];
  return text.split(_infoListSep);
}

/// Cross-platform memory snapshot from the native runtime bridge.
abstract final class NativeRuntimeMemory {
  static Map<String, Object?> snapshot() {
    final out = calloc<native.MemAbi>();
    try {
      if (native.mem(out) != 0) {
        return const <String, Object?>{};
      }
      final value = out.ref;
      final snapshot = <String, Object?>{
        'peak_memory_bytes': value.peakMemoryBytes,
      };
      _putText(snapshot, 'native_backend', value.nativeBackend);
      _putNonZero(snapshot, 'vm_hwm', value.vmHwm);
      _putNonZero(snapshot, 'vm_rss', value.vmRss);
      _putNonZero(snapshot, 'phys_footprint', value.physFootprint);
      _putNonZero(snapshot, 'resident_size', value.residentSize);
      _putNonZero(snapshot, 'virtual_size', value.virtualSize);
      _putNonZero(snapshot, 'peak_working_set', value.peakWorkingSet);
      _putNonZero(snapshot, 'working_set', value.workingSet);
      _putNonZero(snapshot, 'android_peak_pss', value.androidPeakPss);
      _putNonZero(snapshot, 'android_pss', value.androidPss);
      _putNonZero(snapshot, 'android_rss', value.androidRss);
      _putNonZero(
        snapshot,
        'android_native_heap_pss',
        value.androidNativeHeapPss,
      );
      _putNonZero(snapshot, 'android_java_heap_pss', value.androidJavaHeapPss);
      _putNonZero(
        snapshot,
        'android_native_heap_private_dirty',
        value.androidNativeHeapPrivateDirty,
      );
      _putNonZero(
        snapshot,
        'android_java_heap_private_dirty',
        value.androidJavaHeapPrivateDirty,
      );
      return snapshot;
    } finally {
      calloc.free(out);
    }
  }
}

void _putText(
  Map<String, Object?> map,
  String key,
  ffi.Pointer<ffi.Char> value,
) {
  final text = _staticText(value);
  if (text.isNotEmpty) map[key] = text;
}

void _putNonZero(Map<String, Object?> map, String key, int value) {
  if (value != 0) map[key] = value;
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

final class _ValueEntryArena {
  _ValueEntryArena(Map<String, Object?> values) {
    count = _countMap(values);
    pointer = count == 0 ? ffi.nullptr : calloc<native.ValueEntryAbi>(count);
    try {
      var index = 0;
      for (final entry in values.entries) {
        index = _write(pointer, index, entry.key, entry.value);
      }
    } catch (_) {
      close();
      rethrow;
    }
  }

  late final ffi.Pointer<native.ValueEntryAbi> pointer;
  late final int count;
  final List<ffi.Pointer<ffi.Char>> _strings = [];

  int _write(
    ffi.Pointer<native.ValueEntryAbi> target,
    int index,
    String path,
    Object? value,
  ) {
    final entry = target[index];
    entry
      ..path = _own(path)
      ..kind = _entryKind(value)
      ..text = value is String ? _own(value) : ffi.nullptr
      ..intValue = value is int ? value : 0
      ..doubleValue = value is double ? value : 0
      ..boolValue = value == true ? 1 : 0;
    var next = index + 1;
    if (value is Map) {
      value.forEach((key, child) {
        if (key is! String) {
          throw ArgumentError.value(key, 'runtime option key');
        }
        next = _write(target, next, '$path$_entryPathSep$key', child);
      });
    } else if (value is List) {
      for (var i = 0; i < value.length; i += 1) {
        next = _write(target, next, '$path$_entryPathSep$i', value[i]);
      }
    }
    return next;
  }

  ffi.Pointer<ffi.Char> _own(String value) {
    final pointer = value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    _strings.add(pointer);
    return pointer;
  }

  void close() {
    for (final value in _strings) {
      calloc.free(value);
    }
    _strings.clear();
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

int _countMap(Map<String, Object?> values) {
  var count = 0;
  for (final value in values.values) {
    count += _countValue(value);
  }
  return count;
}

int _countValue(Object? value) {
  if (value is Map) {
    var count = 1;
    for (final child in value.values) {
      count += _countValue(child);
    }
    return count;
  }
  if (value is List) {
    var count = 1;
    for (final child in value) {
      count += _countValue(child);
    }
    return count;
  }
  return 1;
}

int _entryKind(Object? value) {
  if (value == null) return _entryNull;
  if (value is String) return _entryString;
  if (value is int) return _entryInt;
  if (value is double) return _entryDouble;
  if (value is bool) return _entryBool;
  if (value is Map) return _entryMap;
  if (value is List) return _entryList;
  throw ArgumentError.value(value, 'runtime option value');
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
    final error = calloc<ffi.Pointer<ffi.Char>>();
    final path = bundle.artifactPath.toNativeUtf8().cast<ffi.Char>();
    final metadata = _ValueEntryArena(bundle.artifact.metadata);
    final backend = _ValueEntryArena(options.backendOptions);
    try {
      final handle = native.open(
        _engineId(engine),
        path,
        _preferMask(options.prefer),
        options.diagnostics ? 1 : 0,
        options.numThreads ?? 0,
        metadata.pointer,
        metadata.count,
        backend.pointer,
        backend.count,
        error,
      );
      if (handle == ffi.nullptr) {
        throw StateError(_takeError(error));
      }
      return _NativeModelSession(
        handle,
        diagnosticsEnabled: options.diagnostics,
      );
    } finally {
      calloc.free(path);
      metadata.close();
      backend.close();
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
  final Map<int, _ShapeBuffer> _vectorShapePointers = {};
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
    for (final shape in _vectorShapePointers.values) {
      shape.close();
    }
    _vectorShapePointers.clear();
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
    final count = calloc<ffi.IntPtr>();
    ffi.Pointer<native.ValueEntryAbi> entries = ffi.nullptr;
    try {
      entries = native.diag(_handle, count);
      final length = count.value;
      if (entries == ffi.nullptr || length <= 0) {
        return const <String, Object?>{};
      }
      final diagnostics = <String, Object?>{};
      for (var i = 0; i < length; i += 1) {
        final entry = (entries + i).ref;
        _putDiag(diagnostics, _diagPath(entry), _diagValue(entry));
      }
      return diagnostics;
    } finally {
      if (entries != ffi.nullptr) {
        native.freeDiag(entries, count.value);
      }
      calloc.free(count);
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

  ffi.Pointer<ffi.Int64> _vectorShapePointer(int length) {
    return _vectorShapePointers
        .putIfAbsent(length, () => _ShapeBuffer([length]))
        .pointer;
  }

  ffi.Pointer<ffi.Void> _inputDataPointer(String name, RuntimeTensor tensor) {
    final nativeBuffer = _nativeRuntimeTensorBuffers[tensor];
    if (nativeBuffer != null) {
      return nativeBuffer._pointerForRun(tensor.bytes.lengthInBytes);
    }
    final bytes = tensor.bytes;
    return _inputBytesPointer(name, bytes);
  }

  ffi.Pointer<ffi.Void> _inputBytesPointer(String name, Uint8List bytes) {
    if (bytes.isEmpty) {
      return ffi.nullptr;
    }
    return _inputBuffers
        .putIfAbsent(name, _NativeByteBuffer.new)
        .copyFrom(bytes);
  }

  _EncodedInputs _encodeInputs(Map<String, Object?> values) {
    final pointer = _inputTensorArena.pointerFor(values.length);
    var index = 0;
    for (final entry in values.entries) {
      _writeInput(pointer, index, entry.key, entry.value);
      index += 1;
    }
    return _EncodedInputs(pointer, values.length);
  }

  void _writeInput(
    ffi.Pointer<native.NamedTensorAbi> pointer,
    int index,
    String name,
    Object? value,
  ) {
    if (value is RuntimeTensor) {
      _writeRuntimeTensor(pointer, index, name, value);
      return;
    }
    if (value is Float32List) {
      _writeTypedList(
        pointer,
        index,
        name,
        RuntimeTensorDataType.float32,
        value.length,
        value,
      );
      return;
    }
    if (value is Int32List) {
      _writeTypedList(
        pointer,
        index,
        name,
        RuntimeTensorDataType.int32,
        value.length,
        value,
      );
      return;
    }
    if (value is Int64List) {
      _writeTypedList(
        pointer,
        index,
        name,
        RuntimeTensorDataType.int64,
        value.length,
        value,
      );
      return;
    }
    if (value is Float64List) {
      _writeTypedList(
        pointer,
        index,
        name,
        RuntimeTensorDataType.float64,
        value.length,
        value,
      );
      return;
    }
    if (value is Uint8List) {
      _writeTypedList(
        pointer,
        index,
        name,
        RuntimeTensorDataType.uint8,
        value.length,
        value,
      );
      return;
    }
    throw ArgumentError.value(
      value,
      name,
      'Expected RuntimeTensor or TypedData',
    );
  }

  void _writeRuntimeTensor(
    ffi.Pointer<native.NamedTensorAbi> pointer,
    int index,
    String name,
    RuntimeTensor value,
  ) {
    pointer[index]
      ..name = _namePointer(name)
      ..tensor.dtype = _dtypeId(value.dtype)
      ..tensor.rank = value.shape.length
      ..tensor.shape = _shapePointer(value.shape)
      ..tensor.byteLength = value.bytes.lengthInBytes
      ..tensor.data = _inputDataPointer(name, value);
  }

  void _writeTypedList(
    ffi.Pointer<native.NamedTensorAbi> pointer,
    int index,
    String name,
    RuntimeTensorDataType dtype,
    int length,
    TypedData value,
  ) {
    final bytes = _typedBytes(value);
    pointer[index]
      ..name = _namePointer(name)
      ..tensor.dtype = _dtypeId(dtype)
      ..tensor.rank = 1
      ..tensor.shape = _vectorShapePointer(length)
      ..tensor.byteLength = bytes.lengthInBytes
      ..tensor.data = _inputBytesPointer(name, bytes);
  }
}

List<String> _diagPath(native.ValueEntryAbi entry) {
  if (entry.path == ffi.nullptr) return const <String>[];
  final text = entry.path.cast<Utf8>().toDartString();
  return text.isEmpty ? const <String>[] : text.split(_entryPathSep);
}

Object? _diagValue(native.ValueEntryAbi entry) {
  switch (entry.kind) {
    case _entryString:
      return _staticText(entry.text);
    case _entryInt:
      return entry.intValue;
    case _entryBool:
      return entry.boolValue != 0;
    case _entryMap:
      return <String, Object?>{};
    case _entryList:
      return <Object?>[];
    case _entryDouble:
      return entry.doubleValue;
    case _entryNull:
      return null;
    default:
      return null;
  }
}

void _putDiag(Map<String, Object?> root, List<String> path, Object? value) {
  if (path.isEmpty) return;
  Object? current = root;
  for (var i = 0; i < path.length - 1; i += 1) {
    final segment = path[i];
    final nextIsList = int.tryParse(path[i + 1]) != null;
    current = _diagChild(current, segment, nextIsList);
  }
  _diagSet(current, path.last, value);
}

Object? _diagChild(Object? current, String segment, bool list) {
  final next = list ? <Object?>[] : <String, Object?>{};
  if (current is Map<String, Object?>) {
    return current.putIfAbsent(segment, () => next);
  }
  if (current is List<Object?>) {
    final index = int.tryParse(segment);
    if (index == null) return null;
    while (current.length <= index) {
      current.add(null);
    }
    current[index] ??= next;
    return current[index];
  }
  return null;
}

void _diagSet(Object? current, String segment, Object? value) {
  if (current is Map<String, Object?>) {
    current[segment] = value;
    return;
  }
  if (current is List<Object?>) {
    final index = int.tryParse(segment);
    if (index == null) return;
    while (current.length <= index) {
      current.add(null);
    }
    current[index] = value;
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

int _preferMask(List<Accelerator> values) {
  var mask = 0;
  for (final value in values) {
    mask |= switch (value) {
      Accelerator.cpu => 1,
      Accelerator.gpu => 2,
      Accelerator.ane => 4,
      Accelerator.npu => 8,
    };
  }
  return mask;
}

RuntimeCapabilities _caps(RuntimeEngine engine) {
  final mask = native.accelMask(_engineId(engine));
  return RuntimeCapabilities(
    engine: engine,
    platform: RuntimePlatformCurrent.current(),
    accelerators: _accelerators(mask),
    details: const {'nativeBackend': 'zig'},
  );
}

List<Accelerator> _accelerators(int mask) => [
  if ((mask & 4) != 0) Accelerator.ane,
  if ((mask & 2) != 0) Accelerator.gpu,
  if ((mask & 8) != 0) Accelerator.npu,
  if ((mask & 1) != 0) Accelerator.cpu,
];

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

Uint8List _typedBytes(TypedData data) {
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}
