part of '../stable_api.dart';

final class MlxDevice {
  MlxDevice._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  /// Returns the current MLX default device.
  factory MlxDevice.defaultDevice() {
    _clearError();
    return MlxDevice._(
      _checkHandle('dart_mlx_default_device', shim.dart_mlx_default_device()),
    );
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_device_free);

  final ffi.Pointer<ffi.Void> _handle;
  bool _closed = false;

  /// Returns `true` once [close] has been called.
  bool get isClosed => _closed;

  /// Whether the native device is currently available.
  bool get isAvailable {
    _ensureOpen();
    final result =
        hooks.debugDeviceIsAvailableOverride?.call(_handle) ??
        shim.dart_mlx_device_is_available(_handle);
    if (result < 0) {
      _throwError('dart_mlx_device_is_available');
    }
    return result == 1;
  }

  /// Native device index.
  int get index {
    _ensureOpen();
    final result = shim.dart_mlx_device_get_index(_handle);
    if (result < 0) {
      _throwError('dart_mlx_device_get_index');
    }
    return result;
  }

  /// Native device type.
  raw.mlx_device_type_ get type {
    _ensureOpen();
    final result = shim.dart_mlx_device_get_type(_handle);
    if (result < 0) {
      _throwError('dart_mlx_device_get_type');
    }
    return raw.mlx_device_type_.fromValue(result);
  }

  /// Compares two device handles using MLX native equality.
  bool equals(MlxDevice other) {
    _ensureOpen();
    other._ensureOpen();
    return shim.dart_mlx_device_equal(_handle, other._handle);
  }

  @override
  String toString() {
    _ensureOpen();
    _clearError();
    return _copyOwnedString(shim.dart_mlx_device_tostring_copy(_handle));
  }

  /// Releases the native device handle.
  ///
  /// This method is idempotent.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_device_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxDevice has been closed.');
    }
  }

  /// Number of devices available for [type].
  static int count(raw.mlx_device_type_ type) {
    final result =
        hooks.debugDeviceCountOverride?.call(type.value) ??
        shim.dart_mlx_device_count(type.value);
    if (result < 0) {
      _throwError('dart_mlx_device_count');
    }
    return result;
  }

  /// Sets the MLX default device.
  static void setDefault(MlxDevice device) {
    device._ensureOpen();
    _clearError();
    _checkStatus('dart_mlx_set_default_device', shim.dart_mlx_set_default_device(device._handle));
  }

  /// Snapshot of backend-specific device information.
  MlxDeviceInfo get info {
    _ensureOpen();
    _clearError();
    final json = _copyOwnedString(shim.dart_mlx_device_info_json_copy(_handle));
    return MlxDeviceInfo._(Map<String, Object>.from(jsonDecode(json) as Map));
  }
}
