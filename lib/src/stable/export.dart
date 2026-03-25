part of '../stable_api.dart';

/// Imported MLX function loaded from disk.
final class MlxImportedFunction {
  MlxImportedFunction._(this._handle);

  final shim.DartMlxImportedHandle _handle;
  bool _closed = false;

  /// Invokes the imported function.
  List<MlxArray> call(
    List<MlxArray> inputs, {
    Map<String, MlxArray> kwargs = const {},
  }) {
    _ensureOpen();
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _withArrayHandles(inputs, (inputHandles, inputLen) {
        if (kwargs.isEmpty) {
          _clearError();
          _checkStatus(
            'dart_mlx_imported_function_apply',
            shim.dart_mlx_imported_function_apply(
              _handle,
              inputHandles.cast(),
              inputLen,
              outputsOut,
              outputsLen,
            ),
          );
          return;
        }
        _withKwargHandles(kwargs, (keys, values, valueLen) {
          _clearError();
          _checkStatus(
            'dart_mlx_imported_function_apply_kwargs',
            shim.dart_mlx_imported_function_apply_kwargs(
              _handle,
              inputHandles.cast(),
              inputLen,
              keys,
              values.cast(),
              valueLen,
              outputsOut,
              outputsLen,
            ),
          );
        });
      });
      return _readOutputArrayList(outputsOut.value, outputsLen.value);
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
    }
  }

  /// Invokes the imported function when exactly one MLX output is expected.
  MlxArray callOne(List<MlxArray> inputs) {
    _ensureOpen();
    return _withArrayHandles(inputs, (inputHandles, inputLen) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_imported_function_apply_one',
          shim.dart_mlx_imported_function_apply_one(
            _handle,
            inputHandles.cast(),
            inputLen,
          ),
        ),
      );
    });
  }

  /// Releases imported function resources.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    shim.dart_mlx_imported_function_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxImportedFunction has been closed.');
    }
  }
}

/// Incremental MLX exporter that records sample signatures before finalization.
final class MlxFunctionExporter {
  MlxFunctionExporter._(
    this._handle, {
    List<_Lease> retainedLeases = const [],
  }) : _retainedLeases = retainedLeases;

  final shim.DartMlxExporterHandle _handle;
  final List<_Lease> _retainedLeases;
  bool _closed = false;

  /// Adds one sample invocation signature to the export artifact.
  void addSample(
    List<MlxArray> inputs, {
    Map<String, MlxArray> kwargs = const {},
  }) {
    _ensureOpen();
    _withArrayHandles(inputs, (inputHandles, inputLen) {
      return _withKwargHandles(kwargs, (keys, values, valueLen) {
        _clearError();
        _checkStatus(
          'dart_mlx_function_exporter_apply',
          shim.dart_mlx_function_exporter_apply(
            _handle,
            inputHandles.cast(),
            inputLen,
            keys,
            values.cast(),
            valueLen,
          ),
        );
      });
    });
  }

  /// Finalizes and releases exporter resources.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    shim.dart_mlx_function_exporter_free(_handle);
    for (final lease in _retainedLeases) {
      lease.release();
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxFunctionExporter has been closed.');
    }
  }
}

/// Function export/import helpers.
abstract final class MlxExport {
  /// Exports [function] with sample [args] to [path].
  static void exportFunction(
    String path,
    MlxFunction function,
    List<MlxArray> args, {
    bool shapeless = false,
  }) {
    function._ensureOpen();
    _withArrayHandles(args, (argHandles, argLen) {
      _withNativePath(path, (nativePath) {
        _clearError();
        _checkStatus(
          'dart_mlx_export_function',
          shim.dart_mlx_export_function(
            nativePath,
            function._handle,
            argHandles.cast(),
            argLen,
            shapeless,
          ),
        );
      });
    });
  }

  /// Exports [function] with positional and keyword sample inputs to [path].
  static void exportKwFunction(
    String path,
    MlxKwFunction function,
    List<MlxArray> args, {
    Map<String, MlxArray> kwargs = const {},
    bool shapeless = false,
  }) {
    function._ensureOpen();
    _withArrayHandles(args, (argHandles, argLen) {
      return _withKwargHandles(kwargs, (keys, values, valueLen) {
        _withNativePath(path, (nativePath) {
          _clearError();
          _checkStatus(
            'dart_mlx_export_kw_function',
            shim.dart_mlx_export_kw_function(
              nativePath,
              function._handle,
              argHandles.cast(),
              argLen,
              keys,
              values.cast(),
              valueLen,
              shapeless,
            ),
          );
        });
      });
    });
  }

  /// Loads an exported function from [path].
  static MlxImportedFunction importFunction(String path) =>
      _withNativePath(path, (nativePath) {
        _clearError();
        return MlxImportedFunction._(
          _checkHandle(
            'dart_mlx_imported_function_new',
            shim.dart_mlx_imported_function_new(nativePath),
          ),
        );
      });

  /// Starts an incremental exporter for [function].
  static MlxFunctionExporter exporter(
    String path,
    MlxFunction function, {
    bool shapeless = false,
  }) => _withNativePath(path, (nativePath) {
    function._ensureOpen();
    _clearError();
    return MlxFunctionExporter._(
      _checkHandle(
        'dart_mlx_function_exporter_new',
        shim.dart_mlx_function_exporter_new(
          nativePath,
          function._handle,
          shapeless,
        ),
      ),
      retainedLeases: function._retainForChild(),
    );
  });
}
