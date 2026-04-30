import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'native_ffi.dart' as dz;

import 'native_tensor_view.dart';
import 'native_runtime.dart' show NativeTensorBuffer;
import 'native_tensor_buffers.dart';
import 'runtime.dart';

typedef NativeFloat32SourceCallback<T> =
    T Function(ffi.Pointer<ffi.Float> pointer, int length);
typedef NativeFloat32SourcesCallback<T> =
    T Function(
      ffi.Pointer<ffi.Pointer<ffi.Float>> pointers,
      ffi.Pointer<ffi.IntPtr> lengths,
    );

int nativeFloat32SourceLength(Object source) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    return source.byteLength ~/ 4;
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    return source.byteLength ~/ 4;
  }
  if (source is Float32List) {
    return source.length;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Float32List',
  );
}

T withNativeFloat32Source<T>(
  Object source,
  NativeFloat32SourceCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  if (source is NativeTensorBuffer) {
    _checkNativeBuffer(source);
    if (source.byteLength == 0) {
      return call(ffi.nullptr, 0);
    }
    return call(source.nativeData.cast<ffi.Float>(), source.byteLength ~/ 4);
  }
  if (source is RuntimeTensor) {
    _checkRuntimeTensor(source);
    final nativeData = source.nativeData;
    if (nativeData != null && nativeData != ffi.nullptr) {
      return call(nativeData.cast<ffi.Float>(), source.byteLength ~/ 4);
    }
    if (source.isNativeHandle) {
      throw StateError('RuntimeTensor is backed by a native runtime handle.');
    }
    return _withCopiedFloat32Source(source.asFloat32List(), call, ffiRuntime);
  }
  if (source is Float32List) {
    return _withCopiedFloat32Source(source, call, ffiRuntime);
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Float32List',
  );
}

T withNativeFloat32Sources<T>(
  List<Object> sources,
  NativeFloat32SourcesCallback<T> call, {
  dz.NativeFfi? ffiRuntime,
}) {
  return dz.withFloat32Sources(ffiRuntime ?? dz.NativeFfi.shared, [
    for (final source in sources) _dzFloat32Source(source),
  ], (pointers, lengths, _) => call(pointers, lengths));
}

T _withCopiedFloat32Source<T>(
  Float32List source,
  NativeFloat32SourceCallback<T> call,
  dz.NativeFfi? ffiRuntime,
) {
  if (source.isEmpty) {
    return call(ffi.nullptr, 0);
  }
  final buffer = nativeFloat32Buffer(source, ffiRuntime: ffiRuntime);
  try {
    return call(buffer.nativeData.cast<ffi.Float>(), source.length);
  } finally {
    buffer.close();
  }
}

void _checkNativeBuffer(NativeTensorBuffer buffer) {
  if (buffer.dtype != RuntimeTensorDataType.float32) {
    throw StateError('Expected float32 buffer, got ${buffer.dtype.name}.');
  }
}

void _checkRuntimeTensor(RuntimeTensor tensor) {
  if (tensor.dtype != RuntimeTensorDataType.float32) {
    throw StateError('Expected float32 tensor, got ${tensor.dtype.name}.');
  }
}

Object _dzFloat32Source(Object source) {
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
    return source.asFloat32List();
  }
  if (source is Float32List) {
    return source;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/RuntimeTensor/Float32List',
  );
}
