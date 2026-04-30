// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_greater(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_greater_equal(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_less(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_less_equal(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_floor_divide(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_logaddexp(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_inner(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Bool, ffi.Float)>()
external DartMlxArrayHandle dart_mlx_hadamard_transform(
  DartMlxArrayHandle input,
  bool hasScale,
  double scale,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_floor(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_sqrt(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_rsqrt(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_square(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_reciprocal(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_sigmoid(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_degrees(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_radians(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_expm1(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_erf(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_erfinv(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_log1p(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_log2(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_log10(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_round(
  DartMlxArrayHandle input,
  int decimals,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_stop_gradient(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_isfinite(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_isinf(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_isnan(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_isneginf(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_isposinf(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_zeros_like(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_ones_like(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Double, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_full_like(
  DartMlxArrayHandle input,
  double value,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_to_fp8(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_from_fp8(
  DartMlxArrayHandle input,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_put_along_axis(
  DartMlxArrayHandle input,
  DartMlxArrayHandle indices,
  DartMlxArrayHandle values,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_scatter_add_axis(
  DartMlxArrayHandle input,
  DartMlxArrayHandle indices,
  DartMlxArrayHandle values,
  int axis,
);
