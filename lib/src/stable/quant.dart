part of '../stable_api.dart';

/// Quantized weight pack returned by [MlxQuant.quantize].
final class MlxQuantizedMatrix {
  MlxQuantizedMatrix(this.weights, this.scales, [this.biases]);

  final MlxArray weights;
  final MlxArray scales;
  final MlxArray? biases;

  void close() {
    biases?.close();
    scales.close();
    weights.close();
  }
}

/// High-level quantization helpers.
abstract final class MlxQuant {
  static MlxQuantizedMatrix quantize(
    MlxArray weights, {
    int? groupSize,
    int? bits,
    String mode = 'affine',
    MlxArray? globalScale,
  }) {
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _withCString(mode, (modePtr) {
        _clearError();
        _checkStatus(
          'dart_mlx_quantize',
          shim.dart_mlx_quantize(
            weights._handle,
            groupSize != null,
            groupSize ?? 0,
            bits != null,
            bits ?? 0,
            modePtr,
            globalScale?._handle ?? ffi.nullptr,
            outputsOut,
            outputsLen,
          ),
        );
      });
      final outputs = _readOutputArrayList(outputsOut.value, outputsLen.value);
      if (outputs.length case 2) {
        return MlxQuantizedMatrix(outputs[0], outputs[1]);
      }
      if (outputs.length case 3) {
        return MlxQuantizedMatrix(outputs[0], outputs[1], outputs[2]);
      }
      for (final output in outputs) {
        output.close();
      }
      throw StateError('Unexpected quantize() output count: ${outputs.length}.');
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
    }
  }

  static MlxArray dequantize(
    MlxQuantizedMatrix matrix, {
    int? groupSize,
    int? bits,
    String mode = 'affine',
    MlxArray? globalScale,
    MlxDType? dtype,
  }) {
    if (mode == 'affine' && matrix.biases == null) {
      throw ArgumentError('Affine dequantize requires biases.');
    }
    return _withCString(mode, (modePtr) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_dequantize',
          shim.dart_mlx_dequantize(
            matrix.weights._handle,
            matrix.scales._handle,
            matrix.biases?._handle ?? ffi.nullptr,
            groupSize != null,
            groupSize ?? 0,
            bits != null,
            bits ?? 0,
            modePtr,
            globalScale?._handle ?? ffi.nullptr,
            dtype != null,
            dtype?.value ?? 0,
          ),
        ),
      );
    });
  }

  static MlxArray matmul(
    MlxArray x,
    MlxQuantizedMatrix matrix, {
    bool transpose = true,
    int? groupSize,
    int? bits,
    String mode = 'affine',
  }) {
    if (mode == 'affine' && matrix.biases == null) {
      throw ArgumentError('Affine quantizedMatmul requires biases.');
    }
    return _withCString(mode, (modePtr) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_quantized_matmul',
          shim.dart_mlx_quantized_matmul(
            x._handle,
            matrix.weights._handle,
            matrix.scales._handle,
            matrix.biases?._handle ?? ffi.nullptr,
            transpose,
            groupSize != null,
            groupSize ?? 0,
            bits != null,
            bits ?? 0,
            modePtr,
          ),
        ),
      );
    });
  }

  static MlxArray qqmm(
    MlxArray x,
    MlxArray weights, {
    MlxArray? weightScales,
    int? groupSize,
    int? bits,
    String mode = 'nvfp4',
    MlxArray? globalScaleX,
    MlxArray? globalScaleW,
  }) => _withCString(mode, (modePtr) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_qqmm',
        shim.dart_mlx_qqmm(
          x._handle,
          weights._handle,
          weightScales?._handle ?? ffi.nullptr,
          groupSize != null,
          groupSize ?? 0,
          bits != null,
          bits ?? 0,
          modePtr,
          globalScaleX?._handle ?? ffi.nullptr,
          globalScaleW?._handle ?? ffi.nullptr,
        ),
      ),
    );
  });

  static MlxArray gatherQmm(
    MlxArray x,
    MlxQuantizedMatrix matrix, {
    MlxArray? lhsIndices,
    MlxArray? rhsIndices,
    bool transpose = true,
    int? groupSize,
    int? bits,
    String mode = 'affine',
    bool sortedIndices = false,
  }) {
    if (mode == 'affine' && matrix.biases == null) {
      throw ArgumentError('Affine gatherQmm requires biases.');
    }
    return _withCString(mode, (modePtr) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_gather_qmm',
          shim.dart_mlx_gather_qmm(
            x._handle,
            matrix.weights._handle,
            matrix.scales._handle,
            matrix.biases?._handle ?? ffi.nullptr,
            lhsIndices?._handle ?? ffi.nullptr,
            rhsIndices?._handle ?? ffi.nullptr,
            transpose,
            groupSize != null,
            groupSize ?? 0,
            bits != null,
            bits ?? 0,
            modePtr,
            sortedIndices,
          ),
        ),
      );
    });
  }
}

/// Module-style quantization namespace.
final class MlxQuantModule {
  const MlxQuantModule._();

  MlxQuantizedMatrix quantize(
    MlxArray weights, {
    int? groupSize,
    int? bits,
    String mode = 'affine',
    MlxArray? globalScale,
  }) => MlxQuant.quantize(
    weights,
    groupSize: groupSize,
    bits: bits,
    mode: mode,
    globalScale: globalScale,
  );

  MlxArray dequantize(
    MlxQuantizedMatrix matrix, {
    int? groupSize,
    int? bits,
    String mode = 'affine',
    MlxArray? globalScale,
    MlxDType? dtype,
  }) => MlxQuant.dequantize(
    matrix,
    groupSize: groupSize,
    bits: bits,
    mode: mode,
    globalScale: globalScale,
    dtype: dtype,
  );

  MlxArray matmul(
    MlxArray x,
    MlxQuantizedMatrix matrix, {
    bool transpose = true,
    int? groupSize,
    int? bits,
    String mode = 'affine',
  }) => MlxQuant.matmul(
    x,
    matrix,
    transpose: transpose,
    groupSize: groupSize,
    bits: bits,
    mode: mode,
  );

  MlxArray qqmm(
    MlxArray x,
    MlxArray weights, {
    MlxArray? weightScales,
    int? groupSize,
    int? bits,
    String mode = 'nvfp4',
    MlxArray? globalScaleX,
    MlxArray? globalScaleW,
  }) => MlxQuant.qqmm(
    x,
    weights,
    weightScales: weightScales,
    groupSize: groupSize,
    bits: bits,
    mode: mode,
    globalScaleX: globalScaleX,
    globalScaleW: globalScaleW,
  );

  MlxArray gatherQmm(
    MlxArray x,
    MlxQuantizedMatrix matrix, {
    MlxArray? lhsIndices,
    MlxArray? rhsIndices,
    bool transpose = true,
    int? groupSize,
    int? bits,
    String mode = 'affine',
    bool sortedIndices = false,
  }) => MlxQuant.gatherQmm(
    x,
    matrix,
    lhsIndices: lhsIndices,
    rhsIndices: rhsIndices,
    transpose: transpose,
    groupSize: groupSize,
    bits: bits,
    mode: mode,
    sortedIndices: sortedIndices,
  );
}
