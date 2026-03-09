part of '../stable_api.dart';

abstract final class MlxRuntime {
  /// Seeds the global random generator.
  static void seed(int seed) {
    _clearError();
    _checkStatus('mlx_random_seed', raw.mlx_random_seed(seed));
  }

  /// Batch-evaluates a list of arrays.
  static void evalAll(List<MlxArray> arrays) {
    final handles = calloc<ffi.Pointer<ffi.Void>>(arrays.length);
    try {
      for (var index = 0; index < arrays.length; index++) {
        handles[index] = arrays[index]._handle;
      }
      _clearError();
      _checkStatus('dart_mlx_eval_many', shim.dart_mlx_eval_many(handles, arrays.length));
    } finally {
      calloc.free(handles);
    }
  }

  /// Schedules asynchronous evaluation for a list of arrays.
  static void asyncEvalAll(List<MlxArray> arrays) {
    final handles = calloc<ffi.Pointer<ffi.Void>>(arrays.length);
    try {
      for (var index = 0; index < arrays.length; index++) {
        handles[index] = arrays[index]._handle;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_async_eval_many',
        shim.dart_mlx_async_eval_many(handles, arrays.length),
      );
    } finally {
      calloc.free(handles);
    }
  }
}

/// High-level memory statistics and tuning.
abstract final class MlxMemory {
  /// Currently active memory in bytes.
  static int activeBytes() => _readSizeValue('mlx_get_active_memory', raw.mlx_get_active_memory);

  /// Cached memory in bytes.
  static int cacheBytes() => _readSizeValue('mlx_get_cache_memory', raw.mlx_get_cache_memory);

  /// Peak memory in bytes.
  static int peakBytes() => _readSizeValue('mlx_get_peak_memory', raw.mlx_get_peak_memory);

  /// Configured memory limit in bytes.
  static int memoryLimitBytes() =>
      _readSizeValue('mlx_get_memory_limit', raw.mlx_get_memory_limit);

  /// Sets the cache limit and returns the applied value.
  static int setCacheLimitBytes(int bytes) =>
      _writeSizeValue('mlx_set_cache_limit', raw.mlx_set_cache_limit, bytes);

  /// Sets the memory limit and returns the applied value.
  static int setMemoryLimitBytes(int bytes) =>
      _writeSizeValue('mlx_set_memory_limit', raw.mlx_set_memory_limit, bytes);

  /// Sets the wired memory limit and returns the applied value.
  static int setWiredLimitBytes(int bytes) =>
      _writeSizeValue('mlx_set_wired_limit', raw.mlx_set_wired_limit, bytes);

  /// Clears cached allocator memory.
  static void clearCache() {
    _clearError();
    _checkStatus('mlx_clear_cache', raw.mlx_clear_cache());
  }

  /// Resets the peak memory counter.
  static void resetPeak() {
    _clearError();
    _checkStatus('mlx_reset_peak_memory', raw.mlx_reset_peak_memory());
  }
}

/// High-level Metal runtime helpers.
abstract final class MlxMetal {
  /// Whether the Metal backend is available in the current runtime.
  static bool isAvailable() {
    final result = calloc<ffi.Bool>();
    try {
      _clearError();
      _checkStatus('mlx_metal_is_available', raw.mlx_metal_is_available(result));
      return result.value;
    } finally {
      calloc.free(result);
    }
  }

  /// Starts a Metal capture to the given output path.
  static void startCapture(String path) {
    _withNativePath(path, (nativePath) {
      _clearError();
      _checkStatus('mlx_metal_start_capture', raw.mlx_metal_start_capture(nativePath));
    });
  }

  /// Stops the active Metal capture.
  static void stopCapture() {
    _clearError();
    _checkStatus('mlx_metal_stop_capture', raw.mlx_metal_stop_capture());
  }
}

/// Module-style memory/runtime namespace.
final class MlxMemoryModule {
  const MlxMemoryModule._();

  /// Currently active memory in bytes.
  int activeBytes() => MlxMemory.activeBytes();

  /// Cached memory in bytes.
  int cacheBytes() => MlxMemory.cacheBytes();

  /// Peak memory in bytes.
  int peakBytes() => MlxMemory.peakBytes();

  /// Memory limit in bytes.
  int memoryLimitBytes() => MlxMemory.memoryLimitBytes();

  /// Sets the cache limit and returns the applied value.
  int setCacheLimitBytes(int bytes) => MlxMemory.setCacheLimitBytes(bytes);

  /// Sets the memory limit and returns the applied value.
  int setMemoryLimitBytes(int bytes) => MlxMemory.setMemoryLimitBytes(bytes);

  /// Sets the wired memory limit and returns the applied value.
  int setWiredLimitBytes(int bytes) => MlxMemory.setWiredLimitBytes(bytes);

  /// Clears cached allocator memory.
  void clearCache() => MlxMemory.clearCache();

  /// Resets the peak memory counter.
  void resetPeak() => MlxMemory.resetPeak();
}

/// Module-style Metal namespace.
final class MlxMetalModule {
  const MlxMetalModule._();

  /// Whether the Metal backend is available.
  bool isAvailable() => MlxMetal.isAvailable();

  /// Starts a Metal capture.
  void startCapture(String path) => MlxMetal.startCapture(path);

  /// Stops the current Metal capture.
  void stopCapture() => MlxMetal.stopCapture();
}

/// MLX runtime version helpers.
abstract final class MlxVersion {
  /// Returns the MLX runtime version string.
  static String current() {
    _clearError();
    final copy = hooks.debugVersionCopyOverride?.call() ?? shim.dart_mlx_version_copy();
    return _copyOwnedString(copy);
  }
}

