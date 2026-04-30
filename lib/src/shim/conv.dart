// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_conv1d(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  int stride,
  int padding,
  int dilation,
  int groups,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_conv2d(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  int stride0,
  int stride1,
  int padding0,
  int padding1,
  int dilation0,
  int dilation1,
  int groups,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_conv3d(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  int stride0,
  int stride1,
  int stride2,
  int padding0,
  int padding1,
  int padding2,
  int dilation0,
  int dilation1,
  int dilation2,
  int groups,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Int,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_conv_general(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  ffi.Pointer<ffi.Int> stride,
  int strideLen,
  ffi.Pointer<ffi.Int> paddingLo,
  int paddingLoLen,
  ffi.Pointer<ffi.Int> paddingHi,
  int paddingHiLen,
  ffi.Pointer<ffi.Int> kernelDilation,
  int kernelDilationLen,
  ffi.Pointer<ffi.Int> inputDilation,
  int inputDilationLen,
  int groups,
  bool flip,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_conv_transpose1d(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  int stride,
  int padding,
  int dilation,
  int outputPadding,
  int groups,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_conv_transpose2d(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  int stride0,
  int stride1,
  int padding0,
  int padding1,
  int dilation0,
  int dilation1,
  int outputPadding0,
  int outputPadding1,
  int groups,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_conv_transpose3d(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  int stride0,
  int stride1,
  int stride2,
  int padding0,
  int padding1,
  int padding2,
  int dilation0,
  int dilation1,
  int dilation2,
  int outputPadding0,
  int outputPadding1,
  int outputPadding2,
  int groups,
);
