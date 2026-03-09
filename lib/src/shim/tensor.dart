// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_take(
  DartMlxArrayHandle input,
  DartMlxArrayHandle indices,
  int axis,
  bool hasAxis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_take_along_axis(
  DartMlxArrayHandle input,
  DartMlxArrayHandle indices,
  int axis,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_gather(
  DartMlxArrayHandle input,
  ffi.Pointer<DartMlxArrayHandle> indices,
  int indicesLen,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  ffi.Pointer<ffi.Int> sliceSizes,
  int sliceSizesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_gather_single(
  DartMlxArrayHandle input,
  DartMlxArrayHandle indices,
  int axis,
  ffi.Pointer<ffi.Int> sliceSizes,
  int sliceSizesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_gather_mm(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  DartMlxArrayHandle lhsIndices,
  DartMlxArrayHandle rhsIndices,
  bool sortedIndices,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_slice(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> start,
  int startLen,
  ffi.Pointer<ffi.Int> stop,
  int stopLen,
  ffi.Pointer<ffi.Int> strides,
  int stridesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_slice_dynamic(
  DartMlxArrayHandle input,
  DartMlxArrayHandle start,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  ffi.Pointer<ffi.Int> sliceSize,
  int sliceSizeLen,
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
  )
>()
external DartMlxArrayHandle dart_mlx_slice_update(
  DartMlxArrayHandle source,
  DartMlxArrayHandle update,
  ffi.Pointer<ffi.Int> start,
  int startLen,
  ffi.Pointer<ffi.Int> stop,
  int stopLen,
  ffi.Pointer<ffi.Int> strides,
  int stridesLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_slice_update_dynamic(
  DartMlxArrayHandle source,
  DartMlxArrayHandle update,
  DartMlxArrayHandle start,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_flatten(
  DartMlxArrayHandle input,
  int startAxis,
  int endAxis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_moveaxis(
  DartMlxArrayHandle input,
  int source,
  int destination,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_swapaxes(
  DartMlxArrayHandle input,
  int axis1,
  int axis2,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_transpose_axes(
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
external DartMlxArrayHandle dart_mlx_tile(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> reps,
  int repsLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Char>,
  )
>()
external DartMlxArrayHandle dart_mlx_pad(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> axes,
  int axesLen,
  ffi.Pointer<ffi.Int> lowPads,
  int lowPadsLen,
  ffi.Pointer<ffi.Int> highPads,
  int highPadsLen,
  DartMlxArrayHandle padValue,
  ffi.Pointer<ffi.Char> mode,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Int,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Char>,
  )
>()
external DartMlxArrayHandle dart_mlx_pad_symmetric(
  DartMlxArrayHandle input,
  int padWidth,
  DartMlxArrayHandle padValue,
  ffi.Pointer<ffi.Char> mode,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_unflatten(
  DartMlxArrayHandle input,
  int axis,
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_einsum(
  ffi.Pointer<ffi.Char> subscripts,
  ffi.Pointer<DartMlxArrayHandle> operands,
  int operandsLen,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_tensordot(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  ffi.Pointer<ffi.Int> axesA,
  int axesALen,
  ffi.Pointer<ffi.Int> axesB,
  int axesBLen,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_tensordot_axis(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
  int axis,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Int,
    ffi.Bool,
    ffi.Int,
    ffi.Pointer<ffi.Char>,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_qqmm(
  DartMlxArrayHandle x,
  DartMlxArrayHandle w,
  DartMlxArrayHandle weightScales,
  bool hasGroupSize,
  int groupSize,
  bool hasBits,
  int bits,
  ffi.Pointer<ffi.Char> mode,
  DartMlxArrayHandle globalScaleX,
  DartMlxArrayHandle globalScaleW,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Bool,
    ffi.Int,
    ffi.Bool,
    ffi.Int,
    ffi.Pointer<ffi.Char>,
    ffi.Bool,
  )
>()
external DartMlxArrayHandle dart_mlx_gather_qmm(
  DartMlxArrayHandle x,
  DartMlxArrayHandle weights,
  DartMlxArrayHandle scales,
  DartMlxArrayHandle biases,
  DartMlxArrayHandle lhsIndices,
  DartMlxArrayHandle rhsIndices,
  bool transpose,
  bool hasGroupSize,
  int groupSize,
  bool hasBits,
  int bits,
  ffi.Pointer<ffi.Char> mode,
  bool sortedIndices,
);
