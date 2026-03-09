part of '../stable_api.dart';

abstract final class MlxIo {
  /// Loads a single MLX array file.
  static MlxArray load(String path, {MlxStream? stream}) =>
      _withNativePath(path, (nativePath) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            stream == null ? 'dart_mlx_load' : 'dart_mlx_load_with_stream',
            stream == null
                ? shim.dart_mlx_load(nativePath)
                : shim.dart_mlx_load_with_stream(
                    nativePath,
                    stream._handle,
                  ),
          ),
        );
      });

  /// Saves a single MLX array file.
  static void save(String path, MlxArray array) {
    _withNativePath(path, (nativePath) {
      _clearError();
      _checkStatus('dart_mlx_save', shim.dart_mlx_save(nativePath, array._handle));
    });
  }

  /// Loads a `.safetensors` file and returns tensors plus metadata.
  static MlxSafetensorsData loadSafetensors(
    String path, {
    MlxStream? stream,
  }) =>
      _withNativePath(path, (nativePath) {
        final arraysOut = calloc<ffi.Pointer<shim.DartMlxArrayHandle>>();
        final keysOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
        final arraysLen = calloc<ffi.Size>();
        final metadataKeysOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
        final metadataValuesOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
        final metadataLen = calloc<ffi.Size>();
        try {
          _clearError();
          _checkStatus(
            stream == null
                ? 'dart_mlx_load_safetensors'
                : 'dart_mlx_load_safetensors_with_stream',
            stream == null
                ? shim.dart_mlx_load_safetensors(
                    nativePath,
                    arraysOut,
                    keysOut,
                    arraysLen,
                    metadataKeysOut,
                    metadataValuesOut,
                    metadataLen,
                  )
                : shim.dart_mlx_load_safetensors_with_stream(
                    nativePath,
                    stream._handle,
                    arraysOut,
                    keysOut,
                    arraysLen,
                    metadataKeysOut,
                    metadataValuesOut,
                    metadataLen,
                  ),
          );
          final tensors = <String, MlxArray>{};
          for (var index = 0; index < arraysLen.value; index++) {
            final key = keysOut.value[index].cast<Utf8>().toDartString();
            tensors[key] = MlxArray._(
              _checkHandle(
                'dart_mlx_load_safetensors.tensor',
                arraysOut.value[index],
              ),
            );
          }
          final metadata = <String, String>{};
          for (var index = 0; index < metadataLen.value; index++) {
            final key = metadataKeysOut.value[index].cast<Utf8>().toDartString();
            final value =
                metadataValuesOut.value[index].cast<Utf8>().toDartString();
            metadata[key] = value;
          }
          return (tensors: tensors, metadata: metadata);
        } finally {
          if (keysOut.value != ffi.nullptr) {
            shim.dart_mlx_free_string_array(keysOut.value, arraysLen.value);
          }
          if (metadataKeysOut.value != ffi.nullptr) {
            shim.dart_mlx_free_string_array(
              metadataKeysOut.value,
              metadataLen.value,
            );
          }
          if (metadataValuesOut.value != ffi.nullptr) {
            shim.dart_mlx_free_string_array(
              metadataValuesOut.value,
              metadataLen.value,
            );
          }
          calloc.free(arraysOut);
          calloc.free(keysOut);
          calloc.free(arraysLen);
          calloc.free(metadataKeysOut);
          calloc.free(metadataValuesOut);
          calloc.free(metadataLen);
        }
      });

  /// Loads a single MLX array from in-memory bytes.
  static MlxArray loadBytes(Uint8List bytes, {MlxStream? stream}) {
    final nativeBytes = calloc<ffi.Uint8>(bytes.length);
    try {
      nativeBytes.asTypedList(bytes.length).setAll(0, bytes);
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_load_bytes',
          shim.dart_mlx_load_bytes(
            nativeBytes,
            bytes.length,
            stream?._handle ?? ffi.nullptr,
          ),
        ),
      );
    } finally {
      calloc.free(nativeBytes);
    }
  }

  /// Serializes a single MLX array to in-memory bytes.
  static Uint8List saveBytes(MlxArray array) {
    final bytesOut = calloc<ffi.Pointer<ffi.Uint8>>();
    final lenOut = calloc<ffi.Size>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_save_bytes',
        shim.dart_mlx_save_bytes(array._handle, bytesOut, lenOut),
      );
      return _copyOwnedBytes(bytesOut.value, lenOut.value);
    } finally {
      calloc.free(bytesOut);
      calloc.free(lenOut);
    }
  }

  /// Loads a single MLX array from a reusable in-memory reader.
  static MlxArray loadReader(MlxBytesReader reader, {MlxStream? stream}) {
    reader._ensureOpen();
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_load_reader_handle',
        shim.dart_mlx_load_reader_handle(
          reader._handle,
          stream?._handle ?? ffi.nullptr,
        ),
      ),
    );
  }

  /// Writes a single MLX array into an in-memory writer.
  static void saveWriter(MlxBytesWriter writer, MlxArray array) {
    writer._ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_save_writer_handle',
      shim.dart_mlx_save_writer_handle(writer._handle, array._handle),
    );
  }

  /// Loads tensors and metadata from in-memory safetensors bytes.
  static MlxSafetensorsData loadSafetensorsBytes(
    Uint8List bytes, {
    MlxStream? stream,
  }) {
    final nativeBytes = calloc<ffi.Uint8>(bytes.length);
    final arraysOut = calloc<ffi.Pointer<shim.DartMlxArrayHandle>>();
    final keysOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final arraysLen = calloc<ffi.Size>();
    final metadataKeysOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final metadataValuesOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final metadataLen = calloc<ffi.Size>();
    try {
      nativeBytes.asTypedList(bytes.length).setAll(0, bytes);
      _clearError();
      _checkStatus(
        'dart_mlx_load_safetensors_bytes',
        shim.dart_mlx_load_safetensors_bytes(
          nativeBytes,
          bytes.length,
          stream?._handle ?? ffi.nullptr,
          arraysOut,
          keysOut,
          arraysLen,
          metadataKeysOut,
          metadataValuesOut,
          metadataLen,
        ),
      );
      final tensors = <String, MlxArray>{};
      for (var index = 0; index < arraysLen.value; index++) {
        final key = keysOut.value[index].cast<Utf8>().toDartString();
        tensors[key] = MlxArray._(
          _checkHandle('dart_mlx_load_safetensors_bytes.tensor', arraysOut.value[index]),
        );
      }
      final metadata = <String, String>{};
      for (var index = 0; index < metadataLen.value; index++) {
        final key = metadataKeysOut.value[index].cast<Utf8>().toDartString();
        final value = metadataValuesOut.value[index].cast<Utf8>().toDartString();
        metadata[key] = value;
      }
      return (tensors: tensors, metadata: metadata);
    } finally {
      calloc.free(nativeBytes);
      if (keysOut.value != ffi.nullptr) {
        shim.dart_mlx_free_string_array(keysOut.value, arraysLen.value);
      }
      if (metadataKeysOut.value != ffi.nullptr) {
        shim.dart_mlx_free_string_array(metadataKeysOut.value, metadataLen.value);
      }
      if (metadataValuesOut.value != ffi.nullptr) {
        shim.dart_mlx_free_string_array(metadataValuesOut.value, metadataLen.value);
      }
      calloc.free(arraysOut);
      calloc.free(keysOut);
      calloc.free(arraysLen);
      calloc.free(metadataKeysOut);
      calloc.free(metadataValuesOut);
      calloc.free(metadataLen);
    }
  }

  /// Loads tensors and metadata from a reusable in-memory reader.
  static MlxSafetensorsData loadSafetensorsReader(
    MlxBytesReader reader, {
    MlxStream? stream,
  }) {
    reader._ensureOpen();
    final arraysOut = calloc<ffi.Pointer<shim.DartMlxArrayHandle>>();
    final keysOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final arraysLen = calloc<ffi.Size>();
    final metadataKeysOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final metadataValuesOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final metadataLen = calloc<ffi.Size>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_load_safetensors_reader_handle',
        shim.dart_mlx_load_safetensors_reader_handle(
          reader._handle,
          stream?._handle ?? ffi.nullptr,
          arraysOut,
          keysOut,
          arraysLen,
          metadataKeysOut,
          metadataValuesOut,
          metadataLen,
        ),
      );
      final tensors = <String, MlxArray>{};
      for (var index = 0; index < arraysLen.value; index++) {
        final key = keysOut.value[index].cast<Utf8>().toDartString();
        tensors[key] = MlxArray._(
          _checkHandle(
            'dart_mlx_load_safetensors_reader_handle.tensor',
            arraysOut.value[index],
          ),
        );
      }
      final metadata = <String, String>{};
      for (var index = 0; index < metadataLen.value; index++) {
        final key = metadataKeysOut.value[index].cast<Utf8>().toDartString();
        final value = metadataValuesOut.value[index].cast<Utf8>().toDartString();
        metadata[key] = value;
      }
      return (tensors: tensors, metadata: metadata);
    } finally {
      if (keysOut.value != ffi.nullptr) {
        shim.dart_mlx_free_string_array(keysOut.value, arraysLen.value);
      }
      if (metadataKeysOut.value != ffi.nullptr) {
        shim.dart_mlx_free_string_array(metadataKeysOut.value, metadataLen.value);
      }
      if (metadataValuesOut.value != ffi.nullptr) {
        shim.dart_mlx_free_string_array(metadataValuesOut.value, metadataLen.value);
      }
      calloc.free(arraysOut);
      calloc.free(keysOut);
      calloc.free(arraysLen);
      calloc.free(metadataKeysOut);
      calloc.free(metadataValuesOut);
      calloc.free(metadataLen);
    }
  }

  /// Saves tensors plus optional metadata as `.safetensors`.
  static void saveSafetensors(
    String path,
    Map<String, MlxArray> tensors, {
    Map<String, String> metadata = const {},
  }) {
    _withNativePath(path, (nativePath) {
      final arrayHandles = calloc<ffi.Pointer<ffi.Void>>(tensors.length);
      final keyPointers = calloc<ffi.Pointer<ffi.Char>>(tensors.length);
      final metadataKeyPointers = calloc<ffi.Pointer<ffi.Char>>(metadata.length);
      final metadataValuePointers =
          calloc<ffi.Pointer<ffi.Char>>(metadata.length);
      final allocatedStrings = <ffi.Pointer<ffi.Char>>[];
      try {
        var index = 0;
        for (final entry in tensors.entries) {
          final key = entry.key.toNativeUtf8().cast<ffi.Char>();
          allocatedStrings.add(key);
          keyPointers[index] = key;
          arrayHandles[index] = entry.value._handle;
          index++;
        }

        index = 0;
        for (final entry in metadata.entries) {
          final key = entry.key.toNativeUtf8().cast<ffi.Char>();
          final value = entry.value.toNativeUtf8().cast<ffi.Char>();
          allocatedStrings..add(key)..add(value);
          metadataKeyPointers[index] = key;
          metadataValuePointers[index] = value;
          index++;
        }

        _clearError();
        _checkStatus(
          'dart_mlx_save_safetensors',
          shim.dart_mlx_save_safetensors(
            nativePath,
            arrayHandles,
            keyPointers,
            tensors.length,
            metadataKeyPointers,
            metadataValuePointers,
            metadata.length,
          ),
        );
      } finally {
        for (final pointer in allocatedStrings) {
          calloc.free(pointer);
        }
        calloc.free(arrayHandles);
        calloc.free(keyPointers);
        calloc.free(metadataKeyPointers);
        calloc.free(metadataValuePointers);
      }
    });
  }

  /// Serializes tensors plus metadata to in-memory safetensors bytes.
  static Uint8List saveSafetensorsBytes(
    Map<String, MlxArray> tensors, {
    Map<String, String> metadata = const {},
  }) {
    final arrayHandles = calloc<ffi.Pointer<ffi.Void>>(tensors.length);
    final keyPointers = calloc<ffi.Pointer<ffi.Char>>(tensors.length);
    final metadataKeyPointers = calloc<ffi.Pointer<ffi.Char>>(metadata.length);
    final metadataValuePointers = calloc<ffi.Pointer<ffi.Char>>(metadata.length);
    final bytesOut = calloc<ffi.Pointer<ffi.Uint8>>();
    final lenOut = calloc<ffi.Size>();
    final allocatedStrings = <ffi.Pointer<ffi.Char>>[];
    try {
      var index = 0;
      for (final entry in tensors.entries) {
        final key = entry.key.toNativeUtf8().cast<ffi.Char>();
        allocatedStrings.add(key);
        keyPointers[index] = key;
        arrayHandles[index] = entry.value._handle;
        index++;
      }
      index = 0;
      for (final entry in metadata.entries) {
        final key = entry.key.toNativeUtf8().cast<ffi.Char>();
        final value = entry.value.toNativeUtf8().cast<ffi.Char>();
        allocatedStrings..add(key)..add(value);
        metadataKeyPointers[index] = key;
        metadataValuePointers[index] = value;
        index++;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_save_safetensors_bytes',
        shim.dart_mlx_save_safetensors_bytes(
          arrayHandles.cast(),
          keyPointers,
          tensors.length,
          metadataKeyPointers,
          metadataValuePointers,
          metadata.length,
          bytesOut,
          lenOut,
        ),
      );
      return _copyOwnedBytes(bytesOut.value, lenOut.value);
    } finally {
      for (final pointer in allocatedStrings) {
        calloc.free(pointer);
      }
      calloc.free(arrayHandles);
      calloc.free(keyPointers);
      calloc.free(metadataKeyPointers);
      calloc.free(metadataValuePointers);
      calloc.free(bytesOut);
      calloc.free(lenOut);
    }
  }

  /// Writes tensors and metadata into an in-memory writer.
  static void saveSafetensorsWriter(
    MlxBytesWriter writer,
    Map<String, MlxArray> tensors, {
    Map<String, String> metadata = const {},
  }) {
    writer._ensureOpen();
    final arrayHandles = calloc<ffi.Pointer<ffi.Void>>(tensors.length);
    final keyPointers = calloc<ffi.Pointer<ffi.Char>>(tensors.length);
    final metadataKeyPointers = calloc<ffi.Pointer<ffi.Char>>(metadata.length);
    final metadataValuePointers = calloc<ffi.Pointer<ffi.Char>>(metadata.length);
    final allocatedStrings = <ffi.Pointer<ffi.Char>>[];
    try {
      var index = 0;
      for (final entry in tensors.entries) {
        final key = entry.key.toNativeUtf8().cast<ffi.Char>();
        allocatedStrings.add(key);
        keyPointers[index] = key;
        arrayHandles[index] = entry.value._handle;
        index++;
      }
      index = 0;
      for (final entry in metadata.entries) {
        final key = entry.key.toNativeUtf8().cast<ffi.Char>();
        final value = entry.value.toNativeUtf8().cast<ffi.Char>();
        allocatedStrings..add(key)..add(value);
        metadataKeyPointers[index] = key;
        metadataValuePointers[index] = value;
        index++;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_save_safetensors_writer_handle',
        shim.dart_mlx_save_safetensors_writer_handle(
          writer._handle,
          arrayHandles.cast(),
          keyPointers,
          tensors.length,
          metadataKeyPointers,
          metadataValuePointers,
          metadata.length,
        ),
      );
    } finally {
      for (final pointer in allocatedStrings) {
        calloc.free(pointer);
      }
      calloc.free(arrayHandles);
      calloc.free(keyPointers);
      calloc.free(metadataKeyPointers);
      calloc.free(metadataValuePointers);
    }
  }
}

/// Module-style IO namespace.
final class MlxIoModule {
  const MlxIoModule._();

  /// Loads a single MLX array file.
  MlxArray load(String path, {MlxStream? stream}) => MlxIo.load(path, stream: stream);

  /// Saves a single MLX array file.
  void save(String path, MlxArray array) => MlxIo.save(path, array);

  /// Loads tensors and metadata from `.safetensors`.
  MlxSafetensorsData loadSafetensors(String path, {MlxStream? stream}) =>
      MlxIo.loadSafetensors(path, stream: stream);

  /// Loads a single MLX array from in-memory bytes.
  MlxArray loadBytes(Uint8List bytes, {MlxStream? stream}) =>
      MlxIo.loadBytes(bytes, stream: stream);

  /// Serializes a single MLX array to bytes.
  Uint8List saveBytes(MlxArray array) => MlxIo.saveBytes(array);

  /// Loads a single MLX array from a reusable in-memory reader.
  MlxArray loadReader(MlxBytesReader reader, {MlxStream? stream}) =>
      MlxIo.loadReader(reader, stream: stream);

  /// Writes a single MLX array to a reusable in-memory writer.
  void saveWriter(MlxBytesWriter writer, MlxArray array) =>
      MlxIo.saveWriter(writer, array);

  /// Loads safetensors data from in-memory bytes.
  MlxSafetensorsData loadSafetensorsBytes(Uint8List bytes, {MlxStream? stream}) =>
      MlxIo.loadSafetensorsBytes(bytes, stream: stream);

  /// Loads safetensors data from a reusable in-memory reader.
  MlxSafetensorsData loadSafetensorsReader(MlxBytesReader reader, {MlxStream? stream}) =>
      MlxIo.loadSafetensorsReader(reader, stream: stream);

  /// Saves tensors and metadata as `.safetensors`.
  void saveSafetensors(
    String path,
    Map<String, MlxArray> tensors, {
    Map<String, String> metadata = const {},
  }) => MlxIo.saveSafetensors(path, tensors, metadata: metadata);

  /// Serializes tensors and metadata to safetensors bytes.
  Uint8List saveSafetensorsBytes(
    Map<String, MlxArray> tensors, {
    Map<String, String> metadata = const {},
  }) => MlxIo.saveSafetensorsBytes(tensors, metadata: metadata);

  /// Writes safetensors data into a reusable in-memory writer.
  void saveSafetensorsWriter(
    MlxBytesWriter writer,
    Map<String, MlxArray> tensors, {
    Map<String, String> metadata = const {},
  }) => MlxIo.saveSafetensorsWriter(writer, tensors, metadata: metadata);
}

/// High-level FFT namespace.
