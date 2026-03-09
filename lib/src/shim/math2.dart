// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_fft2(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_fftn(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_ifft2(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_ifftn(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_rfft2(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_rfftn(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_irfft2(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_irfftn(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> n,
  int nLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_fftshift(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_fft_ifftshift(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_linalg_cholesky(
  DartMlxArrayHandle input,
  bool upper,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_linalg_cross(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  int axis,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<DartMlxArrayHandle>,
  )
>()
external int dart_mlx_linalg_eigh(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Char> uplo,
  ffi.Pointer<DartMlxArrayHandle> values,
  ffi.Pointer<DartMlxArrayHandle> vectors,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_linalg_eigvals(
  DartMlxArrayHandle input,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Pointer<ffi.Char>)>()
external DartMlxArrayHandle dart_mlx_linalg_eigvalsh(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Char> uplo,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_linalg_lu(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<DartMlxArrayHandle>,
  )
>()
external int dart_mlx_linalg_lu_factor(
  DartMlxArrayHandle input,
  ffi.Pointer<DartMlxArrayHandle> lu,
  ffi.Pointer<DartMlxArrayHandle> pivots,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Double,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_linalg_norm(
  DartMlxArrayHandle input,
  double ord,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  bool keepDims,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_linalg_norm_matrix(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Char> ord,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  bool keepDims,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_linalg_norm_l2(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  bool keepDims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_linalg_pinv(
  DartMlxArrayHandle input,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_linalg_solve_triangular(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  bool upper,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_linalg_svd(
  DartMlxArrayHandle input,
  bool computeUv,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);
