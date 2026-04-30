// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_broadcast_arrays(
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputsLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Int,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_split_sections(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> indices,
  int indicesLen,
  int axis,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_segmented_mm(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  DartMlxArrayHandle segments,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_block_masked_mm(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  int blockSize,
  DartMlxArrayHandle maskOut,
  DartMlxArrayHandle maskLhs,
  DartMlxArrayHandle maskRhs,
);
