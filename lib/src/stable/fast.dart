part of '../stable_api.dart';

/// High-level fast path helpers.
abstract final class MlxFast {
  /// Fast layer normalization.
  static MlxArray layerNorm(
    MlxArray input, {
    MlxArray? weight,
    MlxArray? bias,
    double eps = 1e-5,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_fast_layer_norm',
        shim.dart_mlx_fast_layer_norm(
          input._handle,
          weight?._handle ?? ffi.nullptr,
          bias?._handle ?? ffi.nullptr,
          eps,
        ),
      ),
    );
  }

  /// Fast RMS normalization.
  static MlxArray rmsNorm(
    MlxArray input, {
    MlxArray? weight,
    double eps = 1e-5,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_fast_rms_norm',
        shim.dart_mlx_fast_rms_norm(
          input._handle,
          weight?._handle ?? ffi.nullptr,
          eps,
        ),
      ),
    );
  }

  /// Fast rotary positional embedding.
  static MlxArray rope(
    MlxArray input, {
    required int dims,
    bool traditional = false,
    double? base,
    double scale = 1,
    int offset = 0,
    MlxArray? freqs,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_fast_rope',
        shim.dart_mlx_fast_rope(
          input._handle,
          dims,
          traditional,
          base != null,
          base ?? 0,
          scale,
          offset,
          freqs?._handle ?? ffi.nullptr,
        ),
      ),
    );
  }

  /// Fast rotary positional embedding with tensor offsets.
  static MlxArray ropeDynamic(
    MlxArray input, {
    required int dims,
    required MlxArray offset,
    bool traditional = false,
    double? base,
    double scale = 1,
    MlxArray? freqs,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_fast_rope_dynamic',
        shim.dart_mlx_fast_rope_dynamic(
          input._handle,
          dims,
          traditional,
          base != null,
          base ?? 0,
          scale,
          offset._handle,
          freqs?._handle ?? ffi.nullptr,
        ),
      ),
    );
  }

  /// Fast scaled dot product attention.
  static MlxArray scaledDotProductAttention(
    MlxArray queries,
    MlxArray keys,
    MlxArray values, {
    double scale = 1,
    String maskMode = '',
    MlxArray? mask,
    MlxArray? sinks,
  }) => _withNativePath(maskMode, (maskModePtr) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_fast_sdpa',
        shim.dart_mlx_fast_sdpa(
          queries._handle,
          keys._handle,
          values._handle,
          scale,
          maskModePtr,
          mask?._handle ?? ffi.nullptr,
          sinks?._handle ?? ffi.nullptr,
        ),
      ),
    );
  });
}

/// Module-style fast namespace.
final class MlxFastModule {
  const MlxFastModule._();

  /// Creates a Metal kernel config.
  MlxMetalConfig metalConfig() => MlxMetalConfig();

  /// Creates a CUDA kernel config.
  MlxCudaConfig cudaConfig() => MlxCudaConfig();

  /// Creates a Metal custom kernel.
  MlxMetalKernel metalKernel(
    String name,
    List<String> inputNames,
    List<String> outputNames,
    String source, {
    String header = '',
    bool ensureRowContiguous = true,
    bool atomicOutputs = false,
  }) => MlxMetalKernel(
    name,
    inputNames,
    outputNames,
    source,
    header: header,
    ensureRowContiguous: ensureRowContiguous,
    atomicOutputs: atomicOutputs,
  );

  /// Creates a CUDA custom kernel.
  MlxCudaKernel cudaKernel(
    String name,
    List<String> inputNames,
    List<String> outputNames,
    String source, {
    String header = '',
    bool ensureRowContiguous = true,
    int sharedMemory = 0,
  }) => MlxCudaKernel(
    name,
    inputNames,
    outputNames,
    source,
    header: header,
    ensureRowContiguous: ensureRowContiguous,
    sharedMemory: sharedMemory,
  );

  /// Fast layer normalization.
  MlxArray layerNorm(
    MlxArray input, {
    MlxArray? weight,
    MlxArray? bias,
    double eps = 1e-5,
  }) => MlxFast.layerNorm(input, weight: weight, bias: bias, eps: eps);

  /// Fast RMS normalization.
  MlxArray rmsNorm(MlxArray input, {MlxArray? weight, double eps = 1e-5}) =>
      MlxFast.rmsNorm(input, weight: weight, eps: eps);

  /// Fast rotary positional embedding.
  MlxArray rope(
    MlxArray input, {
    required int dims,
    bool traditional = false,
    double? base,
    double scale = 1,
    int offset = 0,
    MlxArray? freqs,
  }) => MlxFast.rope(
    input,
    dims: dims,
    traditional: traditional,
    base: base,
    scale: scale,
    offset: offset,
    freqs: freqs,
  );

  /// Fast rotary positional embedding with tensor offsets.
  MlxArray ropeDynamic(
    MlxArray input, {
    required int dims,
    required MlxArray offset,
    bool traditional = false,
    double? base,
    double scale = 1,
    MlxArray? freqs,
  }) => MlxFast.ropeDynamic(
    input,
    dims: dims,
    offset: offset,
    traditional: traditional,
    base: base,
    scale: scale,
    freqs: freqs,
  );

  /// Fast scaled dot product attention.
  MlxArray scaledDotProductAttention(
    MlxArray queries,
    MlxArray keys,
    MlxArray values, {
    double scale = 1,
    String maskMode = '',
    MlxArray? mask,
    MlxArray? sinks,
  }) => MlxFast.scaledDotProductAttention(
    queries,
    keys,
    values,
    scale: scale,
    maskMode: maskMode,
    mask: mask,
    sinks: sinks,
  );
}
