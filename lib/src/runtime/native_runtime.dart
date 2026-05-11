/// Native Core ML / ONNX Runtime / LiteRT runtime adapters.
library;

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'native_ffi.dart' as dz;
import 'package:ffi/ffi.dart';

import '../models/shared/runtime_metadata.dart';
import 'coreml_runtime.dart' as coreml;
import 'native_ffi_types.dart';
import 'native_bindings.dart' as native;
import 'native_byte_buffer.dart';
import 'native_tensor_allocation.dart';
import 'runtime.dart';

final _nativeRuntimeTensorBuffers = Expando<NativeTensorBuffer>(
  'NativeTensorBuffer',
);
const _infoListSep = '\x1e';
const _entryPathSep = '\x1f';
const _entryString = 1;
const _entryInt = 2;
const _entryBool = 3;
const _entryMap = 4;
const _entryList = 5;
const _entryDouble = 6;
const _entryNull = 7;
const _tensorMemoryCpu = 0;
const _tensorMemoryNativeHandle = 1;
const _tensorMemoryCpuView = 2;

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
        'runtime_version': _staticText(value.runtimeVersion),
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

/// native-backed native tensor input buffer.
///
/// Use this for hot paths that should avoid copying Dart heap typed data into
/// native scratch memory before each inference call. Call [close] when the
/// buffer is no longer needed.
final class NativeTensorBuffer {
  NativeTensorBuffer._(
    this.dtype,
    List<int> shape,
    this.byteLength,
    NativeTensorAllocation allocation,
  ) : shape = List<int>.unmodifiable(shape),
      _allocation = allocation,
      _pointer = allocation.pointer,
      _bytes = _nativeBytes(allocation.pointer, byteLength) {
    if (_pointer != ffi.nullptr) {
      nativeTensorAllocationFinalizer.attach(this, _allocation, detach: this);
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
    final shapeBuffer = rank == 0
        ? null
        : dz.NativeInt64Array.fromValues(shape);
    final byteLength = dz.NativeIntPtrArray.allocate(1);
    final error = _nativeErrorSlot();
    try {
      final pointer = native.allocTensor(
        _dtypeId(dtype),
        shapeBuffer?.pointer ?? ffi.nullptr,
        rank,
        byteLength.pointer,
        error.pointer,
      );
      final resolvedByteLength = byteLength[0];
      if (pointer == ffi.nullptr) {
        if (error.value != ffi.nullptr) {
          throw StateError(error.take());
        }
        if (resolvedByteLength == 0) {
          return NativeTensorBuffer._(
            dtype,
            shape,
            0,
            NativeTensorAllocation.runtime(ffi.nullptr),
          );
        }
      }
      if (pointer == ffi.nullptr) {
        throw StateError('Failed to allocate native tensor buffer.');
      }
      return NativeTensorBuffer._(
        dtype,
        shape,
        resolvedByteLength,
        NativeTensorAllocation.runtime(pointer),
      );
    } finally {
      shapeBuffer?.close();
      byteLength.close();
      error.close();
    }
  }

  /// Adopts memory returned by the native runtime and releases it with
  /// `dinf_free_buf` when this buffer closes.
  factory NativeTensorBuffer.adopt({
    required RuntimeTensorDataType dtype,
    required List<int> shape,
    required int byteLength,
    required ffi.Pointer<ffi.Void> pointer,
  }) {
    if (byteLength < 0) {
      throw RangeError.value(byteLength, 'byteLength');
    }
    if (byteLength == 0) {
      return NativeTensorBuffer._(
        dtype,
        shape,
        0,
        NativeTensorAllocation.runtime(ffi.nullptr),
      );
    }
    if (pointer == ffi.nullptr) {
      throw ArgumentError.value(pointer, 'pointer', 'must not be null');
    }
    return NativeTensorBuffer._(
      dtype,
      shape,
      byteLength,
      NativeTensorAllocation.runtime(pointer),
    );
  }

  factory NativeTensorBuffer.nativeFfi({
    required RuntimeTensorDataType dtype,
    required List<int> shape,
    dz.NativeFfi? ffiRuntime,
  }) {
    final runtime = ffiRuntime ?? dz.NativeFfi.shared;
    final allocated = allocateNativeFfiTensor(
      runtime: runtime,
      dtype: nativeFfiDtype(dtype),
      shape: shape,
    );
    return NativeTensorBuffer._(
      dtype,
      shape,
      allocated.byteLength,
      allocated.allocation,
    );
  }

  factory NativeTensorBuffer.float32(
    List<int> shape, {
    dz.NativeFfi? ffiRuntime,
  }) => _nativeFfiBuffer(RuntimeTensorDataType.float32, shape, ffiRuntime);

  factory NativeTensorBuffer.int32(
    List<int> shape, {
    dz.NativeFfi? ffiRuntime,
  }) => _nativeFfiBuffer(RuntimeTensorDataType.int32, shape, ffiRuntime);

  factory NativeTensorBuffer.int64(
    List<int> shape, {
    dz.NativeFfi? ffiRuntime,
  }) => _nativeFfiBuffer(RuntimeTensorDataType.int64, shape, ffiRuntime);

  factory NativeTensorBuffer.float64(
    List<int> shape, {
    dz.NativeFfi? ffiRuntime,
  }) => _nativeFfiBuffer(RuntimeTensorDataType.float64, shape, ffiRuntime);

  factory NativeTensorBuffer.uint8(
    List<int> shape, {
    dz.NativeFfi? ffiRuntime,
  }) => _nativeFfiBuffer(RuntimeTensorDataType.uint8, shape, ffiRuntime);

  factory NativeTensorBuffer.boolean(
    List<int> shape, {
    dz.NativeFfi? ffiRuntime,
  }) => _nativeFfiBuffer(RuntimeTensorDataType.boolean, shape, ffiRuntime);

  factory NativeTensorBuffer.float16(
    List<int> shape, {
    dz.NativeFfi? ffiRuntime,
  }) => _nativeFfiBuffer(RuntimeTensorDataType.float16, shape, ffiRuntime);

  final RuntimeTensorDataType dtype;
  final List<int> shape;
  final int byteLength;
  final NativeTensorAllocation _allocation;
  ffi.Pointer<ffi.Void> _pointer;
  final Uint8List _bytes;

  bool get isClosed => _pointer == ffi.nullptr && byteLength > 0;

  ffi.Pointer<ffi.Void> get nativeData {
    _checkOpen();
    return _pointer;
  }

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
      nativeData: _pointer,
      owner: this,
    );
    _nativeRuntimeTensorBuffers[value] = this;
    return value;
  }

  RuntimeTensor tensorView({
    required List<int> shape,
    required int byteLength,
  }) {
    _checkOpen();
    if (byteLength < 0 || byteLength > this.byteLength) {
      throw RangeError.range(byteLength, 0, this.byteLength, 'byteLength');
    }
    final bytes = byteLength == this.byteLength
        ? _bytes
        : Uint8List.sublistView(_bytes, 0, byteLength);
    final value = RuntimeTensor(
      dtype: dtype,
      shape: shape,
      bytes: bytes,
      nativeData: _pointer,
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
    if (tensorByteLength > byteLength) {
      throw StateError(
        'Native tensor buffer byte length exceeded $byteLength with '
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
    nativeTensorAllocationFinalizer.detach(this);
    _allocation.release();
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

NativeTensorBuffer _nativeFfiBuffer(
  RuntimeTensorDataType dtype,
  List<int> shape,
  dz.NativeFfi? ffiRuntime,
) => NativeTensorBuffer.nativeFfi(
  dtype: dtype,
  shape: shape,
  ffiRuntime: ffiRuntime,
);

final class _ValueEntryArena {
  _ValueEntryArena(Map<String, Object?> values) {
    count = _countMap(values);
    _buffer = NativeByteBuffer.allocate(
      ffi.sizeOf<native.ValueEntryAbi>() * count,
    );
    pointer = _buffer.pointer.cast<native.ValueEntryAbi>();
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
  late final NativeByteBuffer _buffer;
  final List<dz.NativeUtf8CString> _strings = [];

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
    final string = dz.NativeUtf8CString.utf8(value);
    _strings.add(string);
    return string.pointer;
  }

  void close() {
    for (final value in _strings) {
      value.close();
    }
    _strings.clear();
    _buffer.close();
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
    final error = _nativeErrorSlot();
    final path = dz.NativeUtf8CString.utf8(bundle.artifactPath);
    final metadata = _ValueEntryArena(bundle.artifact.metadata);
    final backend = _ValueEntryArena(options.backendOptions);
    try {
      final handle = native.open(
        _engineId(engine),
        path.pointer,
        _preferMask(options.prefer),
        options.diagnostics ? 1 : 0,
        options.numThreads ?? 0,
        metadata.pointer,
        metadata.count,
        backend.pointer,
        backend.count,
        error.pointer,
      );
      if (handle == ffi.nullptr) {
        throw StateError(error.take());
      }
      return _NativeModelSession(
        handle,
        diagnosticsEnabled: options.diagnostics,
      );
    } finally {
      path.close();
      metadata.close();
      backend.close();
      error.close();
    }
  }
}

final class _NativeModelSession
    implements ModelSession, coreml.CoremlStateResettable {
  _NativeModelSession(this._handle, {required bool diagnosticsEnabled})
    : _diagnosticsEnabled = diagnosticsEnabled;

  ffi.Pointer<ffi.Void> _handle;
  final bool _diagnosticsEnabled;
  final Map<String, dz.NativeUtf8CString> _namePointers = {};
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
    final outputPtr = dz.NativePointerArray<native.NamedTensorAbi>.allocate(1);
    outputPtr[0] = ffi.nullptr;
    final outputCount = dz.NativeIntPtrArray.allocate(1);
    final error = _nativeErrorSlot();
    var outputTransferred = false;
    try {
      final status = native.run(
        _handle,
        tensors.pointer,
        tensors.count,
        outputPtr.pointer,
        outputCount.pointer,
        error.pointer,
      );
      if (status != 0) {
        throw StateError(error.take());
      }
      final count = outputCount[0];
      final pointer = outputPtr[0];
      final owner = pointer == ffi.nullptr
          ? null
          : _NativeOutputOwner(pointer, count);
      outputTransferred = owner != null;
      final outputs = _decodeOutputs(pointer, count, owner);
      return ModelOutputs(
        outputs,
        diagnostics: _diagnosticsEnabled
            ? _diagnostics()
            : const <String, Object?>{},
        release: owner?.close,
      );
    } finally {
      if (!outputTransferred && outputPtr[0] != ffi.nullptr) {
        native.freeTensors(outputPtr[0], outputCount[0]);
      }
      outputPtr.close();
      outputCount.close();
      error.close();
    }
  }

  @override
  Stream<ModelOutputs> stream(ModelInputs inputs) async* {
    yield run(inputs);
  }

  @override
  void resetCoremlState() {
    _checkOpen();
    coreml.resetCoremlState(_handle);
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
      name.close();
    }
    _namePointers.clear();
  }

  void _checkOpen() {
    if (_handle == ffi.nullptr) {
      throw StateError('Model session is closed.');
    }
  }

  Map<String, Object?> _diagnostics() {
    final count = dz.NativeIntPtrArray.allocate(1);
    ffi.Pointer<native.ValueEntryAbi> entries = ffi.nullptr;
    try {
      entries = native.diag(_handle, count.pointer);
      final length = count[0];
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
        native.freeDiag(entries, count[0]);
      }
      count.close();
    }
  }

  ffi.Pointer<ffi.Char> _namePointer(String name) {
    return _namePointers
        .putIfAbsent(name, () => dz.NativeUtf8CString.utf8(name))
        .pointer;
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
    if (tensor.isNativeHandle) {
      throw StateError(
        'Native runtime tensor "$name" has no CPU data pointer.',
      );
    }
    final nativeBuffer = _nativeRuntimeTensorBuffers[tensor];
    if (nativeBuffer != null) {
      return nativeBuffer._pointerForRun(tensor.bytes.lengthInBytes);
    }
    final nativeData = tensor.nativeData;
    if (nativeData != null && nativeData != ffi.nullptr) {
      return nativeData;
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
      ..tensor.byteLength = value.byteLength
      ..tensor.data = value.isNativeHandle
          ? ffi.nullptr
          : _inputDataPointer(name, value)
      ..tensor.handle = _nativeTensorHandle(name, value)
      ..tensor.memoryKind = value.isNativeHandle
          ? _tensorMemoryNativeHandle
          : _tensorMemoryCpu
      ..tensor.reserved = 0;
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
      ..tensor.data = _inputBytesPointer(name, bytes)
      ..tensor.handle = ffi.nullptr
      ..tensor.memoryKind = _tensorMemoryCpu
      ..tensor.reserved = 0;
  }
}

ffi.Pointer<ffi.Void> _nativeTensorHandle(String name, RuntimeTensor value) {
  if (!value.isNativeHandle) {
    return ffi.nullptr;
  }
  final handle = value.nativeHandle;
  if (handle == null || handle == ffi.nullptr) {
    throw StateError('Native runtime tensor "$name" has a null handle.');
  }
  return handle;
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
  NativeByteBuffer? _buffer;

  ffi.Pointer<native.NamedTensorAbi> pointerFor(int count) {
    if (count == 0) {
      return ffi.nullptr;
    }
    if (capacity >= count) {
      return pointer;
    }
    close();
    final buffer = NativeByteBuffer.allocate(
      ffi.sizeOf<native.NamedTensorAbi>() * count,
    );
    _buffer = buffer;
    pointer = buffer.pointer.cast<native.NamedTensorAbi>();
    capacity = count;
    return pointer;
  }

  void close() {
    if (pointer == ffi.nullptr) {
      return;
    }
    _buffer?.close();
    _buffer = null;
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
    final memoryKind = _tensorMemoryKind(tensor.memoryKind);
    final isHandle = memoryKind == RuntimeTensorMemoryKind.nativeHandle;
    final bytes =
        isHandle || tensor.byteLength == 0 || tensor.data == ffi.nullptr
        ? Uint8List(0)
        : tensor.data.cast<ffi.Uint8>().asTypedList(tensor.byteLength);
    outputs[name] = RuntimeTensor(
      dtype: _dtypeFromId(tensor.dtype),
      shape: shape,
      bytes: bytes,
      byteLength: tensor.byteLength,
      nativeData: isHandle ? null : tensor.data,
      nativeHandle: isHandle ? tensor.handle : null,
      memoryKind: memoryKind,
      owner: owner,
    );
  }
  return outputs;
}

final class _ShapeBuffer {
  _ShapeBuffer(List<int> shape)
    : _buffer = dz.NativeInt64Array.fromValues(shape),
      length = shape.length;

  final dz.NativeInt64Array _buffer;
  final int length;

  ffi.Pointer<ffi.Int64> get pointer => _buffer.pointer;

  void close() {
    _buffer.close();
  }
}

final class _NativeByteBuffer {
  ffi.Pointer<ffi.Uint8> pointer = ffi.nullptr;
  int capacity = 0;
  NativeByteBuffer? _buffer;

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
    final buffer = NativeByteBuffer.allocate(byteLength);
    _buffer = buffer;
    pointer = buffer.pointer;
    capacity = byteLength;
  }

  void close() {
    if (pointer == ffi.nullptr) {
      return;
    }
    _buffer?.close();
    _buffer = null;
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

dz.NativeUtf8ErrorSlot _nativeErrorSlot() {
  return dz.NativeUtf8ErrorSlot(
    free: native.freeStr,
    fallbackMessage: 'Native runtime call failed.',
  );
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
    details: const {'nativeBackend': 'cpp'},
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

RuntimeTensorMemoryKind _tensorMemoryKind(int id) => switch (id) {
  _tensorMemoryCpu => RuntimeTensorMemoryKind.cpu,
  _tensorMemoryNativeHandle => RuntimeTensorMemoryKind.nativeHandle,
  _tensorMemoryCpuView => RuntimeTensorMemoryKind.cpu,
  _ => throw StateError('Unsupported native tensor memory kind: $id'),
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
