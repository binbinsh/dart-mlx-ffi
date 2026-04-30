import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'native_ffi.dart' as dz;

import 'native_tensor_view.dart';
import 'native_runtime.dart' show NativeTensorBuffer;
import 'native_tensor_buffers.dart';
import 'runtime.dart';

typedef NativeInt64SourceCallback<T> =
    T Function(ffi.Pointer<ffi.Int64> pointer, int length);
typedef NativeInt64SourcesCallback<T> =
    T Function(
      ffi.Pointer<ffi.Pointer<ffi.Int64>> pointers,
      ffi.Pointer<ffi.IntPtr> lengths,
    );

int nativeInt64SourceLength(Object source) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    return source.byteLength ~/ 8;
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    return source.byteLength ~/ 8;
  }
  if (source is Int64List) {
    return source.length;
  }
  if (source is List<int>) {
    return source.length;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Int64List/List<int>',
  );
}

T withNativeInt64Source<T>(
  Object source,
  NativeInt64SourceCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    if (source.byteLength == 0) {
      return call(ffi.nullptr, 0);
    }
    return call(source.nativeData.cast<ffi.Int64>(), source.byteLength ~/ 8);
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    final nativeData = source.nativeData;
    if (nativeData != null && nativeData != ffi.nullptr) {
      return call(nativeData.cast<ffi.Int64>(), source.byteLength ~/ 8);
    }
    if (source.isNativeHandle) {
      throw StateError('RuntimeTensor is backed by a native runtime handle.');
    }
    return _withCopiedInt64Source(source.asInt64List(), call, ffiRuntime);
  }
  if (source is Int64List) {
    return _withCopiedInt64Source(source, call, ffiRuntime);
  }
  if (source is List<int>) {
    if (source.isEmpty) {
      return call(ffi.nullptr, 0);
    }
    return _withCopiedInt64Source(Int64List.fromList(source), call, ffiRuntime);
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Int64List/List<int>',
  );
}

T withNativeInt64Sources<T>(
  List<Object> sources,
  NativeInt64SourcesCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  return dz.withInt64Sources(ffiRuntime ?? dz.NativeFfi.shared, [
    for (final source in sources) _dzInt64Source(source),
  ], (pointers, lengths, _) => call(pointers, lengths));
}

T _withCopiedInt64Source<T>(
  Int64List source,
  NativeInt64SourceCallback<T> call,
  dz.NativeFfi? ffiRuntime,
) {
  if (source.isEmpty) {
    return call(ffi.nullptr, 0);
  }
  final buffer = nativeInt64Buffer(source, ffiRuntime: ffiRuntime);
  try {
    return call(buffer.nativeData.cast<ffi.Int64>(), source.length);
  } finally {
    buffer.close();
  }
}

void _checkNativeBuffer(NativeTensorBuffer buffer) {
  if (buffer.dtype != RuntimeTensorDataType.int64) {
    throw StateError('Expected int64 buffer, got ${buffer.dtype.name}.');
  }
}

void _checkRuntimeTensor(RuntimeTensor tensor) {
  if (tensor.dtype != RuntimeTensorDataType.int64) {
    throw StateError('Expected int64 tensor, got ${tensor.dtype.name}.');
  }
}

Object _dzInt64Source(Object source) {
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
    return source.asInt64List();
  }
  if (source is Int64List || source is List<int>) {
    return source;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Int64List/List<int>',
  );
}
