part of '../stable_api.dart';

final class MlxBytesReader {
  MlxBytesReader(
    Uint8List bytes, {
    String label = 'dart-bytes-reader',
  }) {
    final nativeBytes = calloc<ffi.Uint8>(bytes.length);
    try {
      nativeBytes.asTypedList(bytes.length).setAll(0, bytes);
      _handle = _withCString(label, (labelPtr) {
        _clearError();
        return _checkHandle(
          'dart_mlx_bytes_reader_new',
          shim.dart_mlx_bytes_reader_new(nativeBytes, bytes.length, labelPtr),
        );
      });
    } finally {
      calloc.free(nativeBytes);
    }
    _finalizer.attach(this, _handle, detach: this);
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_io_reader_free);

  late final shim.DartMlxReaderHandle _handle;
  bool _closed = false;

  void rewind() {
    _ensureOpen();
    shim.dart_mlx_io_reader_rewind(_handle);
  }

  @override
  String toString() {
    _ensureOpen();
    _clearError();
    return _copyOwnedString(shim.dart_mlx_io_reader_tostring_copy(_handle));
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_io_reader_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxBytesReader has been closed.');
    }
  }
}

final class MlxBytesWriter {
  MlxBytesWriter({String label = 'dart-bytes-writer'}) {
    _handle = _withCString(label, (labelPtr) {
      _clearError();
      return _checkHandle(
        'dart_mlx_bytes_writer_new',
        shim.dart_mlx_bytes_writer_new(labelPtr),
      );
    });
    _finalizer.attach(this, _handle, detach: this);
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_io_writer_free);

  late final shim.DartMlxWriterHandle _handle;
  bool _closed = false;

  Uint8List get bytes {
    _ensureOpen();
    final bytesOut = calloc<ffi.Pointer<ffi.Uint8>>();
    final lenOut = calloc<ffi.Size>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_io_writer_bytes_copy',
        shim.dart_mlx_io_writer_bytes_copy(_handle, bytesOut, lenOut),
      );
      return _copyOwnedBytes(bytesOut.value, lenOut.value);
    } finally {
      calloc.free(bytesOut);
      calloc.free(lenOut);
    }
  }

  void rewind() {
    _ensureOpen();
    shim.dart_mlx_io_writer_rewind(_handle);
  }

  @override
  String toString() {
    _ensureOpen();
    _clearError();
    return _copyOwnedString(shim.dart_mlx_io_writer_tostring_copy(_handle));
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_io_writer_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxBytesWriter has been closed.');
    }
  }
}
