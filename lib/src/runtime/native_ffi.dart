import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'runtime.dart';

typedef NativeUtf8Free = void Function(ffi.Pointer<ffi.Char>);

final class NativeUtf8CString {
  NativeUtf8CString.utf8(String value, {NativeFfi? ffiRuntime})
    : pointer = value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();

  final ffi.Pointer<ffi.Char> pointer;

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class NativeUtf8CStringArray {
  NativeUtf8CStringArray(List<String> values, {NativeFfi? ffiRuntime})
    : length = values.length,
      pointer = values.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Pointer<ffi.Char>>(values.length) {
    try {
      for (var i = 0; i < values.length; i += 1) {
        pointer[i] = values[i].toNativeUtf8(allocator: calloc).cast<ffi.Char>();
      }
    } catch (_) {
      close();
      rethrow;
    }
  }

  final int length;
  final ffi.Pointer<ffi.Pointer<ffi.Char>> pointer;

  void close() {
    if (pointer == ffi.nullptr) return;
    for (var i = 0; i < length; i += 1) {
      final value = pointer[i];
      if (value != ffi.nullptr) {
        calloc.free(value);
      }
    }
    calloc.free(pointer);
  }
}

final class NativeByteArray {
  NativeByteArray.allocate(this.length, {NativeFfi? ffiRuntime})
    : pointer = length <= 0 ? ffi.nullptr : calloc<ffi.Uint8>(length);

  NativeByteArray.bytes(List<int> values, {NativeFfi? ffiRuntime})
    : length = values.length,
      pointer = values.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Uint8>(values.length) {
    if (values.isNotEmpty) {
      pointer.asTypedList(values.length).setAll(0, values);
    }
  }

  final int length;
  final ffi.Pointer<ffi.Uint8> pointer;

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class NativeIntPtrArray {
  NativeIntPtrArray.allocate(this.length, {NativeFfi? ffiRuntime})
    : pointer = length <= 0 ? ffi.nullptr : calloc<ffi.IntPtr>(length);

  final int length;
  final ffi.Pointer<ffi.IntPtr> pointer;

  int operator [](int index) => pointer[index];

  ffi.Pointer<ffi.IntPtr> elementAt(int index) => pointer + index;

  void operator []=(int index, int value) {
    pointer[index] = value;
  }

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class NativeInt32Array {
  NativeInt32Array.allocate(this.length, {NativeFfi? ffiRuntime})
    : pointer = length <= 0 ? ffi.nullptr : calloc<ffi.Int32>(length);

  NativeInt32Array.fromValues(List<int> values, {NativeFfi? ffiRuntime})
    : length = values.length,
      pointer = values.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Int32>(values.length) {
    if (values.isNotEmpty) {
      pointer.asTypedList(values.length).setAll(0, values);
    }
  }

  final int length;
  final ffi.Pointer<ffi.Int32> pointer;

  int operator [](int index) => pointer[index];

  ffi.Pointer<ffi.Int32> elementAt(int index) => pointer + index;

  void operator []=(int index, int value) {
    pointer[index] = value;
  }

  Int32List asTypedList() => pointer.asTypedList(length);

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class NativeInt64Array {
  NativeInt64Array.allocate(this.length, {NativeFfi? ffiRuntime})
    : pointer = length <= 0 ? ffi.nullptr : calloc<ffi.Int64>(length);

  NativeInt64Array.fromValues(List<int> values, {NativeFfi? ffiRuntime})
    : length = values.length,
      pointer = values.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Int64>(values.length) {
    if (values.isNotEmpty) {
      pointer.asTypedList(values.length).setAll(0, values);
    }
  }

  final int length;
  final ffi.Pointer<ffi.Int64> pointer;

  int operator [](int index) => pointer[index];

  ffi.Pointer<ffi.Int64> elementAt(int index) => pointer + index;

  void operator []=(int index, int value) {
    pointer[index] = value;
  }

  Int64List asTypedList() => pointer.asTypedList(length);

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class NativeDoubleArray {
  NativeDoubleArray.allocate(this.length, {NativeFfi? ffiRuntime})
    : pointer = length <= 0 ? ffi.nullptr : calloc<ffi.Double>(length);

  final int length;
  final ffi.Pointer<ffi.Double> pointer;

  double operator [](int index) => pointer[index];

  ffi.Pointer<ffi.Double> elementAt(int index) => pointer + index;

  void operator []=(int index, double value) {
    pointer[index] = value;
  }

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class NativePointerArray<T extends ffi.NativeType> {
  NativePointerArray.allocate(this.length, {NativeFfi? ffiRuntime})
    : pointer = length <= 0
          ? ffi.nullptr
          : calloc<ffi.Pointer<ffi.Void>>(length).cast<ffi.Pointer<T>>();

  final int length;
  final ffi.Pointer<ffi.Pointer<T>> pointer;

  ffi.Pointer<T> operator [](int index) => pointer[index];

  void operator []=(int index, ffi.Pointer<T> value) {
    pointer[index] = value;
  }

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class NativeUtf8ErrorSlot {
  NativeUtf8ErrorSlot({
    required this.free,
    required this.fallbackMessage,
    NativeFfi? ffiRuntime,
  }) : pointer = calloc<ffi.Pointer<ffi.Char>>() {
    pointer.value = ffi.nullptr;
  }

  final NativeUtf8Free free;
  final String fallbackMessage;
  final ffi.Pointer<ffi.Pointer<ffi.Char>> pointer;

  ffi.Pointer<ffi.Char> get value => pointer.value;

  void clear() {
    final current = pointer.value;
    if (current != ffi.nullptr) {
      free(current);
    }
    pointer.value = ffi.nullptr;
  }

  String take() {
    final current = pointer.value;
    if (current == ffi.nullptr) {
      return fallbackMessage;
    }
    try {
      return current.cast<Utf8>().toDartString();
    } finally {
      free(current);
      pointer.value = ffi.nullptr;
    }
  }

  void close() {
    clear();
    calloc.free(pointer);
  }
}

enum NativeTensorMemoryKind { cpu, nativeHandle }

final class NativeTensorView {
  const NativeTensorView({
    required this.dtype,
    required this.shape,
    required this.data,
    required this.byteLength,
    this.handle,
    this.memoryKind = NativeTensorMemoryKind.cpu,
  });

  final RuntimeTensorDataType dtype;
  final List<int> shape;
  final ffi.Pointer<ffi.Void> data;
  final int byteLength;
  final ffi.Pointer<ffi.Void>? handle;
  final NativeTensorMemoryKind memoryKind;
}

final class NativeFfi {
  const NativeFfi();

  static const shared = NativeFfi();

  ffi.Pointer<ffi.Void> alloc(int byteLength) {
    if (byteLength <= 0) {
      return ffi.nullptr;
    }
    return calloc<ffi.Uint8>(byteLength).cast<ffi.Void>();
  }

  void free(ffi.Pointer<ffi.Void> pointer, int byteLength) {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }

  int tensorByteLength(RuntimeTensorDataType dtype, List<int> shape) {
    var count = 1;
    for (final dim in shape) {
      if (dim < 0) {
        throw StateError('invalid tensor shape: negative dimension $dim');
      }
      if (dim == 0) {
        count = 0;
        continue;
      }
      final next = count * dim;
      if (count != 0 && next ~/ dim != count) {
        throw StateError('invalid tensor shape: element count overflow');
      }
      count = next;
    }
    final byteLength = count * _dtypeByteLength(dtype);
    if (count != 0 && byteLength ~/ count != _dtypeByteLength(dtype)) {
      throw StateError('invalid tensor shape: byte length overflow');
    }
    return byteLength;
  }
}

int _dtypeByteLength(RuntimeTensorDataType dtype) => switch (dtype) {
  RuntimeTensorDataType.float32 => 4,
  RuntimeTensorDataType.int32 => 4,
  RuntimeTensorDataType.int64 => 8,
  RuntimeTensorDataType.uint8 => 1,
  RuntimeTensorDataType.float64 => 8,
  RuntimeTensorDataType.float16 => 2,
  RuntimeTensorDataType.boolean => 1,
};

T withInt32Sources<T>(
  NativeFfi runtime,
  List<Object> sources,
  T Function(
    ffi.Pointer<ffi.Pointer<ffi.Int32>> pointers,
    ffi.Pointer<ffi.IntPtr> lengths,
    List<Object> owners,
  )
  call,
) {
  final pointers = NativePointerArray<ffi.Int32>.allocate(sources.length);
  final lengths = NativeIntPtrArray.allocate(sources.length);
  final owners = <Object>[];
  try {
    for (var i = 0; i < sources.length; i += 1) {
      final source = _int32List(sources[i]);
      lengths[i] = source.length;
      if (source.isEmpty) {
        pointers[i] = ffi.nullptr;
        continue;
      }
      final buffer = NativeInt32Array.fromValues(source);
      owners.add(buffer);
      pointers[i] = buffer.pointer;
    }
    return call(pointers.pointer, lengths.pointer, owners);
  } finally {
    for (final owner in owners.reversed) {
      if (owner is NativeInt32Array) owner.close();
    }
    lengths.close();
    pointers.close();
  }
}

T withInt64Sources<T>(
  NativeFfi runtime,
  List<Object> sources,
  T Function(
    ffi.Pointer<ffi.Pointer<ffi.Int64>> pointers,
    ffi.Pointer<ffi.IntPtr> lengths,
    List<Object> owners,
  )
  call,
) {
  final pointers = NativePointerArray<ffi.Int64>.allocate(sources.length);
  final lengths = NativeIntPtrArray.allocate(sources.length);
  final owners = <Object>[];
  try {
    for (var i = 0; i < sources.length; i += 1) {
      final source = _int64List(sources[i]);
      lengths[i] = source.length;
      if (source.isEmpty) {
        pointers[i] = ffi.nullptr;
        continue;
      }
      final buffer = NativeInt64Array.fromValues(source);
      owners.add(buffer);
      pointers[i] = buffer.pointer;
    }
    return call(pointers.pointer, lengths.pointer, owners);
  } finally {
    for (final owner in owners.reversed) {
      if (owner is NativeInt64Array) owner.close();
    }
    lengths.close();
    pointers.close();
  }
}

T withFloat32Sources<T>(
  NativeFfi runtime,
  List<Object> sources,
  T Function(
    ffi.Pointer<ffi.Pointer<ffi.Float>> pointers,
    ffi.Pointer<ffi.IntPtr> lengths,
    List<Object> owners,
  )
  call,
) {
  final pointers = NativePointerArray<ffi.Float>.allocate(sources.length);
  final lengths = NativeIntPtrArray.allocate(sources.length);
  final owners = <Object>[];
  try {
    for (var i = 0; i < sources.length; i += 1) {
      final source = _float32List(sources[i]);
      lengths[i] = source.length;
      if (source.isEmpty) {
        pointers[i] = ffi.nullptr;
        continue;
      }
      final buffer = calloc<ffi.Float>(source.length);
      buffer.asTypedList(source.length).setAll(0, source);
      owners.add(buffer);
      pointers[i] = buffer;
    }
    return call(pointers.pointer, lengths.pointer, owners);
  } finally {
    for (final owner in owners.reversed) {
      if (owner is ffi.Pointer<ffi.Float>) calloc.free(owner);
    }
    lengths.close();
    pointers.close();
  }
}

T withFloat64Sources<T>(
  NativeFfi runtime,
  List<Object> sources,
  T Function(
    ffi.Pointer<ffi.Pointer<ffi.Double>> pointers,
    ffi.Pointer<ffi.IntPtr> lengths,
    List<Object> owners,
  )
  call,
) {
  final pointers = NativePointerArray<ffi.Double>.allocate(sources.length);
  final lengths = NativeIntPtrArray.allocate(sources.length);
  final owners = <Object>[];
  try {
    for (var i = 0; i < sources.length; i += 1) {
      final source = _float64List(sources[i]);
      lengths[i] = source.length;
      if (source.isEmpty) {
        pointers[i] = ffi.nullptr;
        continue;
      }
      final buffer = calloc<ffi.Double>(source.length);
      buffer.asTypedList(source.length).setAll(0, source);
      owners.add(buffer);
      pointers[i] = buffer;
    }
    return call(pointers.pointer, lengths.pointer, owners);
  } finally {
    for (final owner in owners.reversed) {
      if (owner is ffi.Pointer<ffi.Double>) calloc.free(owner);
    }
    lengths.close();
    pointers.close();
  }
}

List<int> _int32List(Object source) {
  if (source is Int32List) return source;
  if (source is List<int>) return source;
  if (source is NativeTensorView) {
    return source.data.cast<ffi.Int32>().asTypedList(source.byteLength ~/ 4);
  }
  throw ArgumentError.value(source, 'source', 'expected int32 source');
}

List<int> _int64List(Object source) {
  if (source is Int64List) return source;
  if (source is List<int>) return source;
  if (source is NativeTensorView) {
    return source.data.cast<ffi.Int64>().asTypedList(source.byteLength ~/ 8);
  }
  throw ArgumentError.value(source, 'source', 'expected int64 source');
}

Float32List _float32List(Object source) {
  if (source is Float32List) return source;
  if (source is NativeTensorView) {
    return source.data.cast<ffi.Float>().asTypedList(source.byteLength ~/ 4);
  }
  throw ArgumentError.value(source, 'source', 'expected float32 source');
}

Float64List _float64List(Object source) {
  if (source is Float64List) return source;
  if (source is NativeTensorView) {
    return source.data.cast<ffi.Double>().asTypedList(source.byteLength ~/ 8);
  }
  throw ArgumentError.value(source, 'source', 'expected float64 source');
}

final class NativeScratchTensorBuffer {
  NativeScratchTensorBuffer._(
    this.dtype,
    this.shape,
    this.data,
    this.byteLength,
  );

  factory NativeScratchTensorBuffer.int64(List<int> shape) {
    final byteLength =
        shape.fold<int>(1, (value, dim) => value * dim) *
        ffi.sizeOf<ffi.Int64>();
    final data = byteLength == 0
        ? ffi.nullptr
        : calloc<ffi.Uint8>(byteLength).cast<ffi.Void>();
    return NativeScratchTensorBuffer._(
      RuntimeTensorDataType.int64,
      List<int>.unmodifiable(shape),
      data,
      byteLength,
    );
  }

  final RuntimeTensorDataType dtype;
  final List<int> shape;
  final ffi.Pointer<ffi.Void> data;
  final int byteLength;

  void close() {
    if (data != ffi.nullptr) {
      calloc.free(data);
    }
  }
}
