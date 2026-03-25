// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

typedef DartMlxAneInteropHandle = ffi.Pointer<ffi.Void>;

@ffi.Native<ffi.Int Function(ffi.Pointer<ffi.Char>)>()
external int dart_mlx_ane_interop_set_eval_path(
  ffi.Pointer<ffi.Char> value,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function()>()
external ffi.Pointer<ffi.Char> dart_mlx_ane_interop_eval_path_copy();

@ffi.Native<
  DartMlxAneInteropHandle Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Uint8>>,
    ffi.Pointer<ffi.Size>,
    ffi.Size,
    ffi.Size,
    ffi.Size,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxAneInteropHandle dart_mlx_ane_interop_new_single_io(
  ffi.Pointer<ffi.Char> milText,
  ffi.Pointer<ffi.Pointer<ffi.Char>> weightPaths,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> weightData,
  ffi.Pointer<ffi.Size> weightLens,
  int weightCount,
  int inputBytes,
  int outputBytes,
  int inputChannels,
  int inputSpatial,
  int outputChannels,
  int outputSpatial,
);

@ffi.Native<ffi.Void Function(DartMlxAneInteropHandle)>()
external void dart_mlx_ane_interop_free(DartMlxAneInteropHandle handle);

@ffi.Native<ffi.Int Function(DartMlxAneInteropHandle)>()
external int dart_mlx_ane_interop_eval(DartMlxAneInteropHandle handle);

@ffi.Native<
  ffi.Int Function(DartMlxAneInteropHandle, ffi.Pointer<ffi.Float>, ffi.Size)
>()
external int dart_mlx_ane_interop_write_input_f32(
  DartMlxAneInteropHandle handle,
  ffi.Pointer<ffi.Float> values,
  int count,
);

@ffi.Native<
  ffi.Pointer<ffi.Float> Function(
    DartMlxAneInteropHandle,
    ffi.Pointer<ffi.Size>,
  )
>()
external ffi.Pointer<ffi.Float> dart_mlx_ane_interop_read_output_f32_copy(
  DartMlxAneInteropHandle handle,
  ffi.Pointer<ffi.Size> countOut,
);

@ffi.Native<ffi.Int64 Function(DartMlxAneInteropHandle)>()
external int dart_mlx_ane_interop_last_hw_execution_time_ns(
  DartMlxAneInteropHandle handle,
);
