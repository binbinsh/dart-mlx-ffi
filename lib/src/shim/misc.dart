// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<DartMlxArrayHandle Function(ffi.Double, ffi.Double, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_linspace(
  double start,
  double stop,
  int num,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_outer(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Double,
    ffi.Double,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_isclose(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  double rtol,
  double atol,
  bool equalNan,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_logical_and(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_logical_or(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_logical_not(
  DartMlxArrayHandle input,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_repeat(
  DartMlxArrayHandle input,
  int repeats,
  int axis,
  bool hasAxis,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_roll(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> shift,
  int shiftLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  int axis,
  int mode,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_median(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  bool keepDims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_masked_scatter(
  DartMlxArrayHandle input,
  DartMlxArrayHandle mask,
  DartMlxArrayHandle values,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Float,
    ffi.Bool,
    ffi.Float,
    ffi.Bool,
    ffi.Float,
  )
>()
external DartMlxArrayHandle dart_mlx_nan_to_num(
  DartMlxArrayHandle input,
  double nan,
  bool hasPosInf,
  double posInf,
  bool hasNegInf,
  double negInf,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_divmod(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);
