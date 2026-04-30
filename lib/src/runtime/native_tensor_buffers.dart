import 'dart:typed_data';

import 'native_ffi.dart' as dz;

import 'native_runtime.dart' show NativeTensorBuffer;
import 'runtime.dart' show RuntimeTensorDataType;

NativeTensorBuffer nativeTensorBufferFromTypedData({
  required RuntimeTensorDataType dtype,
  required List<int> shape,
  required TypedData data,
  dz.NativeFfi? ffiRuntime,
}) {
  final buffer = NativeTensorBuffer.nativeFfi(
    dtype: dtype,
    shape: shape,
    ffiRuntime: ffiRuntime,
  );
  try {
    buffer.copyFrom(data);
    return buffer;
  } catch (_) {
    buffer.close();
    rethrow;
  }
}

NativeTensorBuffer nativeFloat32Buffer(
  Float32List values, {
  List<int>? shape,
  dz.NativeFfi? ffiRuntime,
}) {
  return nativeTensorBufferFromTypedData(
    dtype: RuntimeTensorDataType.float32,
    shape: shape ?? [values.length],
    data: values,
    ffiRuntime: ffiRuntime,
  );
}

NativeTensorBuffer nativeInt32Buffer(
  Int32List values, {
  List<int>? shape,
  dz.NativeFfi? ffiRuntime,
}) {
  return nativeTensorBufferFromTypedData(
    dtype: RuntimeTensorDataType.int32,
    shape: shape ?? [values.length],
    data: values,
    ffiRuntime: ffiRuntime,
  );
}

NativeTensorBuffer nativeInt64Buffer(
  Int64List values, {
  List<int>? shape,
  dz.NativeFfi? ffiRuntime,
}) {
  return nativeTensorBufferFromTypedData(
    dtype: RuntimeTensorDataType.int64,
    shape: shape ?? [values.length],
    data: values,
    ffiRuntime: ffiRuntime,
  );
}

NativeTensorBuffer nativeFloat64Buffer(
  Float64List values, {
  List<int>? shape,
  dz.NativeFfi? ffiRuntime,
}) {
  return nativeTensorBufferFromTypedData(
    dtype: RuntimeTensorDataType.float64,
    shape: shape ?? [values.length],
    data: values,
    ffiRuntime: ffiRuntime,
  );
}

NativeTensorBuffer nativeUint8Buffer(
  Uint8List values, {
  List<int>? shape,
  dz.NativeFfi? ffiRuntime,
}) {
  return nativeTensorBufferFromTypedData(
    dtype: RuntimeTensorDataType.uint8,
    shape: shape ?? [values.length],
    data: values,
    ffiRuntime: ffiRuntime,
  );
}

NativeTensorBuffer nativeBooleanBuffer(
  Uint8List values, {
  List<int>? shape,
  dz.NativeFfi? ffiRuntime,
}) {
  return nativeTensorBufferFromTypedData(
    dtype: RuntimeTensorDataType.boolean,
    shape: shape ?? [values.length],
    data: values,
    ffiRuntime: ffiRuntime,
  );
}
