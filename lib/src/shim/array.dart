// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<ffi.Pointer<ffi.Char> Function(DartMlxDeviceHandle)>()
external ffi.Pointer<ffi.Char> dart_mlx_device_tostring_copy(
  DartMlxDeviceHandle handle,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_array_from_bool(
  ffi.Pointer<ffi.Void> data,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_array_from_int32(
  ffi.Pointer<ffi.Void> data,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_array_from_float32(
  ffi.Pointer<ffi.Void> data,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_array_from_float64(
  ffi.Pointer<ffi.Void> data,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_array_from_int64(
  ffi.Pointer<ffi.Void> data,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_array_from_uint64(
  ffi.Pointer<ffi.Void> data,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<ffi.Void Function(DartMlxArrayHandle)>()
external void dart_mlx_array_free(DartMlxArrayHandle handle);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle)>()
external int dart_mlx_array_eval(DartMlxArrayHandle handle);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle)>()
external int dart_mlx_array_ndim(DartMlxArrayHandle handle);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle)>()
external int dart_mlx_array_size(DartMlxArrayHandle handle);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle)>()
external int dart_mlx_array_dtype(DartMlxArrayHandle handle);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Int>, ffi.Int)>()
external int dart_mlx_array_copy_shape(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Int> outShape,
  int outShapeLen,
);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Uint8>, ffi.Int)>()
external int dart_mlx_array_copy_bool(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Uint8> out,
  int len,
);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Int32>, ffi.Int)>()
external int dart_mlx_array_copy_int32(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Int32> out,
  int len,
);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Uint32>, ffi.Int)>()
external int dart_mlx_array_copy_uint32(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Uint32> out,
  int len,
);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Int64>, ffi.Int)>()
external int dart_mlx_array_copy_int64(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Int64> out,
  int len,
);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Uint64>, ffi.Int)>()
external int dart_mlx_array_copy_uint64(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Uint64> out,
  int len,
);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Float>, ffi.Int)>()
external int dart_mlx_array_copy_float32(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Float> out,
  int len,
);

@ffi.Native<ffi.Int Function(DartMlxArrayHandle, ffi.Pointer<ffi.Double>, ffi.Int)>()
external int dart_mlx_array_copy_float64(
  DartMlxArrayHandle handle,
  ffi.Pointer<ffi.Double> out,
  int len,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function(DartMlxArrayHandle)>()
external ffi.Pointer<ffi.Char> dart_mlx_array_tostring_copy(
  DartMlxArrayHandle handle,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_add(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_subtract(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_multiply(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_divide(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_matmul(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_equal(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_where(
  DartMlxArrayHandle condition,
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_abs(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_negative(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_exp(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_log(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_sin(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_cos(DartMlxArrayHandle input);
