import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'native_ffi.dart' as dz;

import 'native_tensor_view.dart';
import 'native_runtime.dart' show NativeTensorBuffer;
import 'native_tensor_buffers.dart';
import 'runtime.dart';

typedef NativeFloat64SourceCallback<T> =
    T Function(ffi.Pointer<ffi.Double> pointer, int length);
typedef NativeFloat64SourcesCallback<T> =
    T Function(
      ffi.Pointer<ffi.Pointer<ffi.Double>> pointers,
      ffi.Pointer<ffi.IntPtr> lengths,
    );

int nativeFloat64SourceLength(Object source) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    return source.byteLength ~/ 8;
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    return source.byteLength ~/ 8;
  }
  if (source is Float64List) {
    return source.length;
  }
  if (source is List<double>) {
    return source.length;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Float64List/List<double>',
  );
}

T withNativeFloat64Source<T>(
  Object source,
  NativeFloat64SourceCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    if (source.byteLength == 0) {
      return call(ffi.nullptr, 0);
    }
    return call(source.nativeData.cast<ffi.Double>(), source.byteLength ~/ 8);
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    final nativeData = source.nativeData;
    if (nativeData != null && nativeData != ffi.nullptr) {
      return call(nativeData.cast<ffi.Double>(), source.byteLength ~/ 8);
    }
    if (source.isNativeHandle) {
      throw StateError('RuntimeTensor is backed by a native runtime handle.');
    }
    return _withCopiedFloat64Source(source.asFloat64List(), call, ffiRuntime);
  }
  if (source is Float64List) {
    return _withCopiedFloat64Source(source, call, ffiRuntime);
  }
  if (source is List<double>) {
    if (source.isEmpty) {
      return call(ffi.nullptr, 0);
    }
    return _withCopiedFloat64Source(
      Float64List.fromList(source),
      call,
      ffiRuntime,
    );
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Float64List/List<double>',
  );
}

T withNativeFloat64Sources<T>(
  List<Object> sources,
  NativeFloat64SourcesCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  return dz.withFloat64Sources(ffiRuntime ?? dz.NativeFfi.shared, [
    for (final source in sources) _dzFloat64Source(source),
  ], (pointers, lengths, _) => call(pointers, lengths));
}

T _withCopiedFloat64Source<T>(
  Float64List source,
  NativeFloat64SourceCallback<T> call,
  dz.NativeFfi? ffiRuntime,
) {
  if (source.isEmpty) {
    return call(ffi.nullptr, 0);
  }
  final buffer = nativeFloat64Buffer(source, ffiRuntime: ffiRuntime);
  try {
    return call(buffer.nativeData.cast<ffi.Double>(), source.length);
  } finally {
    buffer.close();
  }
}

void _checkNativeBuffer(NativeTensorBuffer buffer) {
  if (buffer.dtype != RuntimeTensorDataType.float64) {
    throw StateError('Expected float64 buffer, got ${buffer.dtype.name}.');
  }
}

void _checkRuntimeTensor(RuntimeTensor tensor) {
  if (tensor.dtype != RuntimeTensorDataType.float64) {
    throw StateError('Expected float64 tensor, got ${tensor.dtype.name}.');
  }
}

Object _dzFloat64Source(Object source) {
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
    return source.asFloat64List();
  }
  if (source is Float64List) {
    return source;
  }
  if (source is List<double>) {
    return Float64List.fromList(source);
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Float64List/List<double>',
  );
}
