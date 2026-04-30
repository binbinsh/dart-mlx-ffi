// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Char>, DartMlxStreamHandle)>()
external DartMlxArrayHandle dart_mlx_load_with_stream(
  ffi.Pointer<ffi.Char> file,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Char>,
    DartMlxStreamHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_load_safetensors_with_stream(
  ffi.Pointer<ffi.Char> file,
  DartMlxStreamHandle stream,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> arraysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> keysOut,
  ffi.Pointer<ffi.Size> arraysLen,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataKeysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataValuesOut,
  ffi.Pointer<ffi.Size> metadataLen,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Uint8>, ffi.Size, DartMlxStreamHandle)>()
external DartMlxArrayHandle dart_mlx_load_bytes(
  ffi.Pointer<ffi.Uint8> bytes,
  int len,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Size,
    DartMlxStreamHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_load_safetensors_bytes(
  ffi.Pointer<ffi.Uint8> bytes,
  int len,
  DartMlxStreamHandle stream,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> arraysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> keysOut,
  ffi.Pointer<ffi.Size> arraysLen,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataKeysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataValuesOut,
  ffi.Pointer<ffi.Size> metadataLen,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Pointer<ffi.Uint8>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_save_bytes(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> bytesOut,
  ffi.Pointer<ffi.Size> lenOut,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Uint8>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_save_safetensors_bytes(
  ffi.Pointer<DartMlxArrayHandle> arrays,
  ffi.Pointer<ffi.Pointer<ffi.Char>> keys,
  int arraysLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> metadataKeys,
  ffi.Pointer<ffi.Pointer<ffi.Char>> metadataValues,
  int metadataLen,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> bytesOut,
  ffi.Pointer<ffi.Size> lenOut,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void dart_mlx_free_buffer(ffi.Pointer<ffi.Void> pointer);
