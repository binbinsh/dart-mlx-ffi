part of '../stable_api.dart';

/// Official Core ML compute units.
enum MlxCoreMlComputeUnits {
  cpuOnly(0),
  cpuAndGpu(1),
  all(2),
  cpuAndNeuralEngine(3);

  const MlxCoreMlComputeUnits(this.value);
  final int value;
}

/// Thin wrapper over Core ML single-input/single-output models.
final class MlxCoreMlModel {
  MlxCoreMlModel._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  factory MlxCoreMlModel.loadSingleIo({
    required String path,
    required String inputName,
    required String outputName,
    required List<int> inputShape,
    required int outputCount,
    MlxCoreMlComputeUnits computeUnits =
        MlxCoreMlComputeUnits.cpuAndNeuralEngine,
  }) {
    if (inputShape.isEmpty) {
      throw ArgumentError.value(inputShape, 'inputShape', 'Must not be empty.');
    }
    return _withNativePath(path, (nativePath) {
      return _withCString(inputName, (nativeInput) {
        return _withCString(outputName, (nativeOutput) {
          return _withInts(inputShape, (shapePtr, _) {
            _clearAnePrivateError();
            final handle = _checkAnePrivateHandle(
              'coreml model load',
              shim.dart_mlx_coreml_model_load(
                nativePath,
                nativeInput,
                nativeOutput,
                shapePtr,
                inputShape.length,
                outputCount,
                computeUnits.value,
              ),
            );
            return MlxCoreMlModel._(handle);
          });
        });
      });
    });
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_coreml_model_free);

  final shim.DartMlxCoreMlHandle _handle;
  bool _closed = false;

  bool get isClosed => _closed;

  Float32List predict(Float32List input) {
    _ensureOpen();
    final nativeInput = calloc<ffi.Float>(input.length);
    final outputCount = calloc<ffi.Size>();
    try {
      nativeInput.asTypedList(input.length).setAll(0, input);
      _clearAnePrivateError();
      final pointer = shim.dart_mlx_coreml_predict_f32_copy(
        _handle,
        nativeInput,
        input.length,
        outputCount,
      );
      if (pointer == ffi.nullptr) {
        _throwAnePrivateError('coreml predict');
      }
      return _copyOwnedFloat32List(pointer, outputCount.value);
    } finally {
      calloc.free(nativeInput);
      calloc.free(outputCount);
    }
  }

  int lastPredictTimeNs() {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_coreml_last_predict_time_ns(_handle);
    if (result < 0) {
      _throwAnePrivateError('coreml lastPredictTimeNs');
    }
    return result;
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_coreml_model_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxCoreMlModel has been closed.');
    }
  }
}

/// Module-style official Core ML namespace.
final class MlxCoreMlModule {
  const MlxCoreMlModule._();

  MlxCoreMlModel loadSingleIo({
    required String path,
    required String inputName,
    required String outputName,
    required List<int> inputShape,
    required int outputCount,
    MlxCoreMlComputeUnits computeUnits =
        MlxCoreMlComputeUnits.cpuAndNeuralEngine,
  }) => MlxCoreMlModel.loadSingleIo(
    path: path,
    inputName: inputName,
    outputName: outputName,
    inputShape: inputShape,
    outputCount: outputCount,
    computeUnits: computeUnits,
  );
}
