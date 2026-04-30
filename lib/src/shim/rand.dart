// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<
  DartMlxArrayHandle Function(
    ffi.Double,
    ffi.Double,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_random_uniform(
  double low,
  double high,
  ffi.Pointer<ffi.Int> shape,
  int dim,
  int dtype,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
    ffi.Double,
    ffi.Double,
  )
>()
external DartMlxArrayHandle dart_mlx_random_normal(
  ffi.Pointer<ffi.Int> shape,
  int dim,
  int dtype,
  double loc,
  double scale,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_random_bernoulli(
  DartMlxArrayHandle probability,
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  DartMlxArrayHandle key,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Uint64)>()
external DartMlxArrayHandle dart_mlx_random_key(int seed);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<DartMlxArrayHandle>,
  )
>()
external int dart_mlx_random_split(
  DartMlxArrayHandle key,
  ffi.Pointer<DartMlxArrayHandle> first,
  ffi.Pointer<DartMlxArrayHandle> second,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_random_categorical(
  DartMlxArrayHandle logits,
  int axis,
  int mode,
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  int numSamples,
  DartMlxArrayHandle key,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_random_permutation(
  DartMlxArrayHandle input,
  int axis,
  DartMlxArrayHandle key,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Int, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_random_permutation_arange(
  int x,
  DartMlxArrayHandle key,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_random_gumbel(
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  int dtype,
  DartMlxArrayHandle key,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
    ffi.Double,
    ffi.Double,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_random_laplace(
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  int dtype,
  double loc,
  double scale,
  DartMlxArrayHandle key,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_random_multivariate_normal(
  DartMlxArrayHandle mean,
  DartMlxArrayHandle cov,
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  int dtype,
  DartMlxArrayHandle key,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Int,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_random_randint(
  int low,
  int high,
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  int dtype,
  DartMlxArrayHandle key,
);
