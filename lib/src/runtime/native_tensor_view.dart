import 'dart:ffi' as ffi;

import 'native_ffi.dart' as nf;

import 'native_ffi_types.dart';
import 'native_runtime.dart' show NativeTensorBuffer;
import 'runtime.dart';

nf.NativeTensorView nativeTensorView(Object tensor) {
  if (tensor is NativeTensorBuffer) {
    return nativeBufferView(tensor);
  }
  if (tensor is RuntimeTensor) {
    return runtimeTensorView(tensor);
  }
  throw ArgumentError.value(
    tensor,
    'tensor',
    'expected NativeTensorBuffer or RuntimeTensor',
  );
}

nf.NativeTensorView nativeBufferView(NativeTensorBuffer buffer) {
  return nf.NativeTensorView(
    dtype: nativeFfiDtype(buffer.dtype),
    shape: buffer.shape,
    data: buffer.nativeData,
    byteLength: buffer.byteLength,
  );
}

nf.NativeTensorView runtimeTensorView(RuntimeTensor tensor) {
  if (tensor.isNativeHandle) {
    final handle = tensor.nativeHandle;
    if (handle == null || handle == ffi.nullptr) {
      throw StateError('RuntimeTensor native handle is null.');
    }
    return nf.NativeTensorView(
      dtype: nativeFfiDtype(tensor.dtype),
      shape: tensor.shape,
      data: ffi.nullptr,
      byteLength: tensor.byteLength,
      handle: handle,
      memoryKind: nf.NativeTensorMemoryKind.nativeHandle,
    );
  }
  final data = tensor.nativeData;
  if (data == null || data == ffi.nullptr) {
    throw StateError(
      'RuntimeTensor has no native data pointer; use NativeTensorBuffer for '
      'zero-copy Dart FFI calls.',
    );
  }
  return nf.NativeTensorView(
    dtype: nativeFfiDtype(tensor.dtype),
    shape: tensor.shape,
    data: data,
    byteLength: tensor.byteLength,
    memoryKind: nativeFfiMemoryKind(tensor.memoryKind),
  );
}
