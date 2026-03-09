// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

typedef DartMlxReaderHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxWriterHandle = ffi.Pointer<ffi.Void>;

@ffi.Native<DartMlxReaderHandle Function(ffi.Pointer<ffi.Uint8>, ffi.Size, ffi.Pointer<ffi.Char>)>()
external DartMlxReaderHandle dart_mlx_bytes_reader_new(
  ffi.Pointer<ffi.Uint8> bytes,
  int len,
  ffi.Pointer<ffi.Char> label,
);

@ffi.Native<ffi.Void Function(DartMlxReaderHandle)>()
external void dart_mlx_io_reader_free(DartMlxReaderHandle handle);

@ffi.Native<ffi.Pointer<ffi.Char> Function(DartMlxReaderHandle)>()
external ffi.Pointer<ffi.Char> dart_mlx_io_reader_tostring_copy(
  DartMlxReaderHandle handle,
);

@ffi.Native<ffi.Void Function(DartMlxReaderHandle)>()
external void dart_mlx_io_reader_rewind(DartMlxReaderHandle handle);

@ffi.Native<DartMlxArrayHandle Function(DartMlxReaderHandle, DartMlxStreamHandle)>()
external DartMlxArrayHandle dart_mlx_load_reader_handle(
  DartMlxReaderHandle reader,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxReaderHandle,
    DartMlxStreamHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_load_safetensors_reader_handle(
  DartMlxReaderHandle reader,
  DartMlxStreamHandle stream,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> arraysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> keysOut,
  ffi.Pointer<ffi.Size> arraysLen,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataKeysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataValuesOut,
  ffi.Pointer<ffi.Size> metadataLen,
);

@ffi.Native<DartMlxWriterHandle Function(ffi.Pointer<ffi.Char>)>()
external DartMlxWriterHandle dart_mlx_bytes_writer_new(
  ffi.Pointer<ffi.Char> label,
);

@ffi.Native<ffi.Void Function(DartMlxWriterHandle)>()
external void dart_mlx_io_writer_free(DartMlxWriterHandle handle);

@ffi.Native<ffi.Pointer<ffi.Char> Function(DartMlxWriterHandle)>()
external ffi.Pointer<ffi.Char> dart_mlx_io_writer_tostring_copy(
  DartMlxWriterHandle handle,
);

@ffi.Native<ffi.Void Function(DartMlxWriterHandle)>()
external void dart_mlx_io_writer_rewind(DartMlxWriterHandle handle);

@ffi.Native<
  ffi.Int Function(
    DartMlxWriterHandle,
    ffi.Pointer<ffi.Pointer<ffi.Uint8>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_io_writer_bytes_copy(
  DartMlxWriterHandle writer,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> bytesOut,
  ffi.Pointer<ffi.Size> lenOut,
);

@ffi.Native<ffi.Int Function(DartMlxWriterHandle, DartMlxArrayHandle)>()
external int dart_mlx_save_writer_handle(
  DartMlxWriterHandle writer,
  DartMlxArrayHandle input,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxWriterHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
  )
>()
external int dart_mlx_save_safetensors_writer_handle(
  DartMlxWriterHandle writer,
  ffi.Pointer<DartMlxArrayHandle> arrays,
  ffi.Pointer<ffi.Pointer<ffi.Char>> keys,
  int arraysLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> metadataKeys,
  ffi.Pointer<ffi.Pointer<ffi.Char>> metadataValues,
  int metadataLen,
);
