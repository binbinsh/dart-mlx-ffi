// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

typedef DartMlxCoreMlHandle = ffi.Pointer<ffi.Void>;

@ffi.Native<
  DartMlxCoreMlHandle Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Int,
    ffi.Int,
  )
>()
external DartMlxCoreMlHandle dart_mlx_coreml_model_load(
  ffi.Pointer<ffi.Char> path,
  ffi.Pointer<ffi.Char> inputName,
  ffi.Pointer<ffi.Char> outputName,
  ffi.Pointer<ffi.Int> inputShape,
  int inputRank,
  int outputCount,
  int computeUnits,
);

@ffi.Native<ffi.Void Function(DartMlxCoreMlHandle)>()
external void dart_mlx_coreml_model_free(DartMlxCoreMlHandle handle);

@ffi.Native<
  ffi.Pointer<ffi.Float> Function(
    DartMlxCoreMlHandle,
    ffi.Pointer<ffi.Float>,
    ffi.Size,
    ffi.Pointer<ffi.Size>,
  )
>()
external ffi.Pointer<ffi.Float> dart_mlx_coreml_predict_f32_copy(
  DartMlxCoreMlHandle handle,
  ffi.Pointer<ffi.Float> inputValues,
  int inputCount,
  ffi.Pointer<ffi.Size> outputCount,
);

@ffi.Native<ffi.Int64 Function(DartMlxCoreMlHandle)>()
external int dart_mlx_coreml_last_predict_time_ns(DartMlxCoreMlHandle handle);
