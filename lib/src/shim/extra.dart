// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_diag(
  DartMlxArrayHandle input,
  int k,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_diagonal(
  DartMlxArrayHandle input,
  int offset,
  int axis1,
  int axis2,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_kron(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Bool,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_meshgrid(
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputsLen,
  bool sparse,
  ffi.Pointer<ffi.Char> indexing,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_partition(
  DartMlxArrayHandle input,
  int kth,
  int axis,
  bool hasAxis,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_scatter(
  DartMlxArrayHandle input,
  ffi.Pointer<DartMlxArrayHandle> indices,
  int indicesLen,
  DartMlxArrayHandle updates,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  int op,
  int singleAxis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_scatter_single(
  DartMlxArrayHandle input,
  DartMlxArrayHandle indices,
  DartMlxArrayHandle updates,
  int axis,
  int op,
);
