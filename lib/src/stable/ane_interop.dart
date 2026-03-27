part of '../stable_api.dart';

enum MlxAneInteropEvalPath {
  inMemory(''),
  client('client'),
  clientDirect('clientDirect'),
  realtime('realtime');

  const MlxAneInteropEvalPath(this.envValue);

  final String envValue;
}

/// Thin wrapper over vendored Espresso ANEInterop for single-input/single-output models.
final class MlxAneInteropKernel {
  MlxAneInteropKernel._(this._handle, this.outputElementCount) {
    _finalizer.attach(this, _handle, detach: this);
  }

  factory MlxAneInteropKernel.singleIo({
    required String milText,
    required List<MlxAneWeight> weights,
    required int inputBytes,
    required int outputBytes,
    required int inputChannels,
    required int inputSpatial,
    required int outputChannels,
    required int outputSpatial,
  }) {
    final weightPaths = weights.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Pointer<ffi.Char>>(weights.length);
    final weightData = weights.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Pointer<ffi.Uint8>>(weights.length);
    final weightLens = weights.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Size>(weights.length);
    final allocatedPaths = <ffi.Pointer<ffi.Char>>[];
    final allocatedBuffers = <ffi.Pointer<ffi.Uint8>>[];
    try {
      for (var index = 0; index < weights.length; index++) {
        final path = weights[index].path.toNativeUtf8().cast<ffi.Char>();
        allocatedPaths.add(path);
        weightPaths[index] = path;
        final bytes = weights[index].data;
        final buffer = calloc<ffi.Uint8>(bytes.length);
        allocatedBuffers.add(buffer);
        buffer.asTypedList(bytes.length).setAll(0, bytes);
        weightData[index] = buffer;
        weightLens[index] = bytes.length;
      }
      return _withCString(milText, (nativeMil) {
        _clearAnePrivateError();
        final handle = _checkAnePrivateHandle(
          'aneInterop kernel creation',
          shim.dart_mlx_ane_interop_new_single_io(
            nativeMil,
            weightPaths,
            weightData,
            weightLens,
            weights.length,
            inputBytes,
            outputBytes,
            inputChannels,
            inputSpatial,
            outputChannels,
            outputSpatial,
          ),
        );
        return MlxAneInteropKernel._(handle, outputChannels * outputSpatial);
      });
    } finally {
      for (final path in allocatedPaths) {
        calloc.free(path);
      }
      for (final buffer in allocatedBuffers) {
        calloc.free(buffer);
      }
      if (weightPaths != ffi.nullptr) calloc.free(weightPaths);
      if (weightData != ffi.nullptr) calloc.free(weightData);
      if (weightLens != ffi.nullptr) calloc.free(weightLens);
    }
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_ane_interop_free);

  final shim.DartMlxAneInteropHandle _handle;
  final int outputElementCount;
  ffi.Pointer<ffi.Float>? _inputScratch;
  int _inputScratchCount = 0;
  ffi.Pointer<ffi.Float>? _outputScratch;
  bool _closed = false;

  bool get isClosed => _closed;

  Float32List runFloat32(Float32List input) {
    _ensureOpen();
    final nativeInput = calloc<ffi.Float>(input.length);
    final countOut = calloc<ffi.Size>();
    try {
      nativeInput.asTypedList(input.length).setAll(0, input);
      _clearAnePrivateError();
      _checkAnePrivateStatus(
        'aneInterop writeInput',
        shim.dart_mlx_ane_interop_write_input_f32(
          _handle,
          nativeInput,
          input.length,
        ),
      );
      _checkAnePrivateStatus(
        'aneInterop eval',
        shim.dart_mlx_ane_interop_eval(_handle),
      );
      final pointer = shim.dart_mlx_ane_interop_read_output_f32_copy(
        _handle,
        countOut,
      );
      return _copyOwnedFloat32List(pointer, countOut.value);
    } finally {
      calloc.free(nativeInput);
      calloc.free(countOut);
    }
  }

  Float32List runRawFloat32(Float32List input) {
    _ensureOpen();
    final nativeInput = _ensureInputScratch(input.length);
    final output = _ensureOutputScratch();
    try {
      nativeInput.asTypedList(input.length).setAll(0, input);
      _clearAnePrivateError();
      _checkAnePrivateStatus(
        'aneInterop writeInputRaw',
        shim.dart_mlx_ane_interop_write_input_raw_f32(
          _handle,
          nativeInput,
          input.length,
        ),
      );
      _checkAnePrivateStatus(
        'aneInterop eval',
        shim.dart_mlx_ane_interop_eval(_handle),
      );
      _checkAnePrivateStatus(
        'aneInterop readOutputRaw',
        shim.dart_mlx_ane_interop_read_output_raw_f32(
          _handle,
          output,
          outputElementCount,
        ),
      );
      return Float32List.fromList(output.asTypedList(outputElementCount));
    } finally {
    }
  }

  int lastHardwareExecutionTimeNs() {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_ane_interop_last_hw_execution_time_ns(_handle);
    if (result < 0) {
      _throwAnePrivateError('aneInterop lastHardwareExecutionTimeNs');
    }
    return result;
  }

  void close() {
    if (_closed) return;
    _closed = true;
    if (_inputScratch != null) {
      calloc.free(_inputScratch!);
      _inputScratch = null;
      _inputScratchCount = 0;
    }
    if (_outputScratch != null) {
      calloc.free(_outputScratch!);
      _outputScratch = null;
    }
    _finalizer.detach(this);
    shim.dart_mlx_ane_interop_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxAneInteropKernel has been closed.');
    }
  }

  ffi.Pointer<ffi.Float> _ensureInputScratch(int count) {
    if (_inputScratch == null || _inputScratchCount < count) {
      if (_inputScratch != null) {
        calloc.free(_inputScratch!);
      }
      _inputScratch = calloc<ffi.Float>(count);
      _inputScratchCount = count;
    }
    return _inputScratch!;
  }

  ffi.Pointer<ffi.Float> _ensureOutputScratch() {
    if (_outputScratch == null) {
      _outputScratch = calloc<ffi.Float>(outputElementCount);
    }
    return _outputScratch!;
  }
}

abstract final class MlxAneInterop {
  static void setEvalPath(MlxAneInteropEvalPath path) {
    if (path.envValue.isEmpty) {
      _checkAnePrivateStatus(
        'aneInterop setEvalPath',
        shim.dart_mlx_ane_interop_set_eval_path(ffi.nullptr),
      );
      return;
    }
    _withCString(path.envValue, (nativeValue) {
      _checkAnePrivateStatus(
        'aneInterop setEvalPath',
        shim.dart_mlx_ane_interop_set_eval_path(nativeValue),
      );
    });
  }

  static String? evalPath() =>
      _copyAnePrivateOwnedStringOrNull(shim.dart_mlx_ane_interop_eval_path_copy());
}
