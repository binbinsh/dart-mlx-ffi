part of '../stable_api.dart';

typedef _ErrorHandlerNative = ffi.Void Function(
  ffi.Pointer<ffi.Char>,
  ffi.Pointer<ffi.Void>,
);

void _onError(ffi.Pointer<ffi.Char> message, ffi.Pointer<ffi.Void> _) {
  final bytes = <int>[];
  var index = 0;
  final rawBytes = message.cast<ffi.Uint8>();
  while (true) {
    final value = rawBytes[index];
    if (value == 0 || index >= 4096) {
      break;
    }
    bytes.add(value);
    index++;
  }
  _lastErrorMessage = String.fromCharCodes(bytes);
}

String? _lastErrorMessage;
bool _runtimeInitialized = false;
final ffi.NativeCallable<_ErrorHandlerNative> _errorHandler =
    ffi.NativeCallable<_ErrorHandlerNative>.listener(_onError);

void _ensureRuntimeInitialized() {
  if (_runtimeInitialized) {
    return;
  }
  raw.mlx_set_error_handler(
    _errorHandler.nativeFunction,
    ffi.nullptr,
    ffi.nullptr,
  );
  hooks.debugDispatchError = (message) => _onError(message, ffi.nullptr);
  _runtimeInitialized = true;
}

void _clearError() {
  _ensureRuntimeInitialized();
  _lastErrorMessage = null;
}

Never _throwError(String operation, [int? code]) {
  throw MlxException(_lastErrorMessage ?? '$operation failed.', code: code);
}

void _checkStatus(String operation, int status) {
  if (status != 0) {
    _throwError(operation, status);
  }
}

ffi.Pointer<ffi.Void> _checkHandle(
  String operation,
  ffi.Pointer<ffi.Void> handle,
) {
  if (handle == ffi.nullptr) {
    _throwError(operation);
  }
  return handle;
}

String _copyOwnedString(ffi.Pointer<ffi.Char> value) {
  if (value == ffi.nullptr) {
    _throwError('string conversion');
  }
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    shim.dart_mlx_string_free_copy(value);
  }
}

/// Exception thrown when an MLX operation fails.
final class MlxException implements Exception {
  const MlxException(this.message, {this.code});

  final String message;
  final int? code;

  @override
  String toString() => code == null
      ? 'MlxException: $message'
      : 'MlxException(code: $code, message: $message)';
}

/// Managed MLX array backed by a native MLX handle.
///
/// This high-level wrapper intentionally covers a stable core subset of MLX.
/// For the full `mlx-c` API surface, import `package:dart_mlx_ffi/raw.dart`.
