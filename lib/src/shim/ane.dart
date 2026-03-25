// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

typedef DartMlxAnePrivateModelHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxAnePrivateSessionHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxAnePrivateChainHandle = ffi.Pointer<ffi.Void>;

@ffi.Native<ffi.Bool Function()>()
external bool dart_mlx_ane_private_is_compiled();

@ffi.Native<ffi.Bool Function()>()
external bool dart_mlx_ane_private_is_enabled();

@ffi.Native<ffi.Void Function()>()
external void dart_mlx_ane_private_clear_error();

@ffi.Native<ffi.Pointer<ffi.Char> Function()>()
external ffi.Pointer<ffi.Char> dart_mlx_ane_private_last_error_copy();

@ffi.Native<ffi.Pointer<ffi.Char> Function()>()
external ffi.Pointer<ffi.Char> dart_mlx_ane_private_probe_json_copy();

@ffi.Native<
  DartMlxAnePrivateModelHandle Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Uint8>>,
    ffi.Pointer<ffi.Size>,
    ffi.Size,
  )
>()
external DartMlxAnePrivateModelHandle dart_mlx_ane_private_model_new_mil(
  ffi.Pointer<ffi.Char> milText,
  ffi.Pointer<ffi.Pointer<ffi.Char>> weightPaths,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> weightData,
  ffi.Pointer<ffi.Size> weightLens,
  int weightCount,
);

@ffi.Native<
  DartMlxAnePrivateModelHandle Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Uint8>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Size>,
    ffi.Size,
  )
>()
external DartMlxAnePrivateModelHandle dart_mlx_ane_private_model_new_mil_ex(
  ffi.Pointer<ffi.Char> milText,
  ffi.Pointer<ffi.Pointer<ffi.Char>> weightPaths,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> weightData,
  ffi.Pointer<ffi.Size> weightLens,
  ffi.Pointer<ffi.Size> weightOffsets,
  int weightCount,
);

@ffi.Native<ffi.Void Function(DartMlxAnePrivateModelHandle)>()
external void dart_mlx_ane_private_model_free(
  DartMlxAnePrivateModelHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateModelHandle)>()
external int dart_mlx_ane_private_model_compile(
  DartMlxAnePrivateModelHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateModelHandle)>()
external int dart_mlx_ane_private_model_load(
  DartMlxAnePrivateModelHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateModelHandle)>()
external int dart_mlx_ane_private_model_unload(
  DartMlxAnePrivateModelHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateModelHandle)>()
external int dart_mlx_ane_private_model_is_loaded(
  DartMlxAnePrivateModelHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateModelHandle)>()
external int dart_mlx_ane_private_model_compiled_exists(
  DartMlxAnePrivateModelHandle handle,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function(DartMlxAnePrivateModelHandle)>()
external ffi.Pointer<ffi.Char> dart_mlx_ane_private_model_hex_identifier_copy(
  DartMlxAnePrivateModelHandle handle,
);

@ffi.Native<
  DartMlxAnePrivateSessionHandle Function(
    DartMlxAnePrivateModelHandle,
    ffi.Pointer<ffi.Size>,
    ffi.Size,
    ffi.Pointer<ffi.Size>,
    ffi.Size,
  )
>()
external DartMlxAnePrivateSessionHandle dart_mlx_ane_private_session_new(
  DartMlxAnePrivateModelHandle model,
  ffi.Pointer<ffi.Size> inputBytes,
  int inputCount,
  ffi.Pointer<ffi.Size> outputBytes,
  int outputCount,
);

@ffi.Native<ffi.Void Function(DartMlxAnePrivateSessionHandle)>()
external void dart_mlx_ane_private_session_free(
  DartMlxAnePrivateSessionHandle handle,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxAnePrivateSessionHandle,
    ffi.Size,
    ffi.Pointer<ffi.Uint8>,
    ffi.Size,
  )
>()
external int dart_mlx_ane_private_session_write_input_bytes(
  DartMlxAnePrivateSessionHandle handle,
  int index,
  ffi.Pointer<ffi.Uint8> data,
  int len,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxAnePrivateSessionHandle,
    ffi.Size,
    ffi.Pointer<ffi.Void>,
    ffi.Size,
    ffi.Size,
    ffi.Size,
  )
>()
external int dart_mlx_ane_private_session_write_input_array_packed_f32(
  DartMlxAnePrivateSessionHandle handle,
  int index,
  ffi.Pointer<ffi.Void> input,
  int seqLen,
  int dim,
  int lane,
);

@ffi.Native<
  ffi.Pointer<ffi.Uint8> Function(
    DartMlxAnePrivateSessionHandle,
    ffi.Size,
    ffi.Pointer<ffi.Size>,
  )
>()
external ffi.Pointer<ffi.Uint8>
dart_mlx_ane_private_session_read_output_bytes_copy(
  DartMlxAnePrivateSessionHandle handle,
  int index,
  ffi.Pointer<ffi.Size> lenOut,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxAnePrivateSessionHandle,
    ffi.Size,
    ffi.Pointer<ffi.Uint8>,
    ffi.Size,
  )
>()
external int dart_mlx_ane_private_session_read_output_bytes(
  DartMlxAnePrivateSessionHandle handle,
  int index,
  ffi.Pointer<ffi.Uint8> out,
  int len,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateSessionHandle)>()
external int dart_mlx_ane_private_session_evaluate(
  DartMlxAnePrivateSessionHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateSessionHandle)>()
external int dart_mlx_ane_private_session_prepare_realtime(
  DartMlxAnePrivateSessionHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateSessionHandle)>()
external int dart_mlx_ane_private_session_teardown_realtime(
  DartMlxAnePrivateSessionHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateSessionHandle)>()
external int dart_mlx_ane_private_session_evaluate_realtime(
  DartMlxAnePrivateSessionHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateSessionHandle)>()
external int dart_mlx_ane_private_session_realtime_is_loaded(
  DartMlxAnePrivateSessionHandle handle,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    DartMlxAnePrivateSessionHandle,
    ffi.Bool,
    ffi.Bool,
    ffi.Bool,
    ffi.Bool,
    ffi.Bool,
  )
>()
external ffi.Pointer<ffi.Char>
dart_mlx_ane_private_session_probe_chaining_json_copy(
  DartMlxAnePrivateSessionHandle handle,
  bool validateRequest,
  bool useSharedSignalEvent,
  bool attemptPrepare,
  bool callEnqueueSets,
  bool callBuffersReady,
);

@ffi.Native<
  DartMlxAnePrivateChainHandle Function(
    DartMlxAnePrivateSessionHandle,
    ffi.Bool,
    ffi.Bool,
    ffi.Bool,
  )
>()
external DartMlxAnePrivateChainHandle dart_mlx_ane_private_chain_new(
  DartMlxAnePrivateSessionHandle session,
  bool validateRequest,
  bool useSharedSignalEvent,
  bool attemptPrepare,
);

@ffi.Native<ffi.Void Function(DartMlxAnePrivateChainHandle)>()
external void dart_mlx_ane_private_chain_free(
  DartMlxAnePrivateChainHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateChainHandle)>()
external int dart_mlx_ane_private_chain_is_prepared(
  DartMlxAnePrivateChainHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateChainHandle)>()
external int dart_mlx_ane_private_chain_has_enqueue_sets(
  DartMlxAnePrivateChainHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateChainHandle)>()
external int dart_mlx_ane_private_chain_has_buffers_ready(
  DartMlxAnePrivateChainHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateChainHandle)>()
external int dart_mlx_ane_private_chain_enqueue_sets(
  DartMlxAnePrivateChainHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxAnePrivateChainHandle)>()
external int dart_mlx_ane_private_chain_buffers_ready(
  DartMlxAnePrivateChainHandle handle,
);

@ffi.Native<ffi.Pointer<ffi.Uint8> Function(ffi.Pointer<ffi.Float>, ffi.Size)>()
external ffi.Pointer<ffi.Uint8>
dart_mlx_ane_private_encode_fp32_to_fp16_bytes_copy(
  ffi.Pointer<ffi.Float> values,
  int count,
);

@ffi.Native<
  ffi.Pointer<ffi.Float> Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Size,
    ffi.Pointer<ffi.Size>,
  )
>()
external ffi.Pointer<ffi.Float>
dart_mlx_ane_private_decode_fp16_bytes_to_fp32_copy(
  ffi.Pointer<ffi.Uint8> bytes,
  int byteLen,
  ffi.Pointer<ffi.Size> countOut,
);
