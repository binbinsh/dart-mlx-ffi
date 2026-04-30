import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'native_ffi.dart' as dz;

import 'native_tensor_view.dart';
import 'native_runtime.dart' show NativeTensorBuffer;
import 'native_tensor_buffers.dart';
import 'runtime.dart';

typedef NativeInt32SourceCallback<T> =
    T Function(ffi.Pointer<ffi.Int32> pointer, int length);
typedef NativeInt32SourcesCallback<T> =
    T Function(
      ffi.Pointer<ffi.Pointer<ffi.Int32>> pointers,
      ffi.Pointer<ffi.IntPtr> lengths,
    );

int nativeInt32SourceLength(Object source) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    return source.byteLength ~/ 4;
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    return source.byteLength ~/ 4;
  }
  if (source is Int32List) {
    return source.length;
  }
  if (source is List<int>) {
    return source.length;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Int32List/List<int>',
  );
}

T withNativeInt32Source<T>(
  Object source,
  NativeInt32SourceCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    if (source.byteLength == 0) {
      return call(ffi.nullptr, 0);
    }
    return call(source.nativeData.cast<ffi.Int32>(), source.byteLength ~/ 4);
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    final nativeData = source.nativeData;
    if (nativeData != null && nativeData != ffi.nullptr) {
      return call(nativeData.cast<ffi.Int32>(), source.byteLength ~/ 4);
    }
    if (source.isNativeHandle) {
      throw StateError('RuntimeTensor is backed by a native runtime handle.');
    }
    return _withCopiedInt32Source(source.asInt32List(), call, ffiRuntime);
  }
  if (source is Int32List) {
    return _withCopiedInt32Source(source, call, ffiRuntime);
  }
  if (source is List<int>) {
    if (source.isEmpty) {
      return call(ffi.nullptr, 0);
    }
    return _withCopiedInt32Source(Int32List.fromList(source), call, ffiRuntime);
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Int32List/List<int>',
  );
}

T withNativeInt32Sources<T>(
  List<Object> sources,
  NativeInt32SourcesCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  return dz.withInt32Sources(ffiRuntime ?? dz.NativeFfi.shared, [
    for (final source in sources) _dzInt32Source(source),
  ], (pointers, lengths, _) => call(pointers, lengths));
}

T _withCopiedInt32Source<T>(
  Int32List source,
  NativeInt32SourceCallback<T> call,
  dz.NativeFfi? ffiRuntime,
) {
  if (source.isEmpty) {
    return call(ffi.nullptr, 0);
  }
  final buffer = nativeInt32Buffer(source, ffiRuntime: ffiRuntime);
  try {
    return call(buffer.nativeData.cast<ffi.Int32>(), source.length);
  } finally {
    buffer.close();
  }
}

void _checkNativeBuffer(NativeTensorBuffer buffer) {
  if (buffer.dtype != RuntimeTensorDataType.int32) {
    throw StateError('Expected int32 buffer, got ${buffer.dtype.name}.');
  }
}

void _checkRuntimeTensor(RuntimeTensor tensor) {
  if (tensor.dtype != RuntimeTensorDataType.int32) {
    throw StateError('Expected int32 tensor, got ${tensor.dtype.name}.');
  }
}

Object _dzInt32Source(Object source) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    return nativeTensorView(source);
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    final nativeData = source.nativeData;
    if (nativeData != null && nativeData != ffi.nullptr) {
      return nativeTensorView(source);
    }
    if (source.isNativeHandle) {
      throw StateError('RuntimeTensor is backed by a native runtime handle.');
    }
    return source.asInt32List();
  }
  if (source is Int32List || source is List<int>) {
    return source;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Int32List/List<int>',
  );
}
