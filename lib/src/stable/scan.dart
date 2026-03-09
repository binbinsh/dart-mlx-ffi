part of '../stable_api.dart';

abstract final class MlxScan {
  static MlxArray cumsum(
    MlxArray input, {
    int axis = 0,
    bool reverse = false,
    bool inclusive = true,
  }) => _scan(input, axis, reverse, inclusive, 0, 'dart_mlx_scan.cumsum');

  static MlxArray cumprod(
    MlxArray input, {
    int axis = 0,
    bool reverse = false,
    bool inclusive = true,
  }) => _scan(input, axis, reverse, inclusive, 1, 'dart_mlx_scan.cumprod');

  static MlxArray cummax(
    MlxArray input, {
    int axis = 0,
    bool reverse = false,
    bool inclusive = true,
  }) => _scan(input, axis, reverse, inclusive, 2, 'dart_mlx_scan.cummax');

  static MlxArray cummin(
    MlxArray input, {
    int axis = 0,
    bool reverse = false,
    bool inclusive = true,
  }) => _scan(input, axis, reverse, inclusive, 3, 'dart_mlx_scan.cummin');

  static MlxArray logcumsumexp(
    MlxArray input, {
    int axis = 0,
    bool reverse = false,
    bool inclusive = true,
  }) => _scan(input, axis, reverse, inclusive, 4, 'dart_mlx_scan.logcumsumexp');

  static MlxArray eye(
    int n, {
    int m = -1,
    int k = 0,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) {
    final resolvedM = m < 0 ? n : m;
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_eye', shim.dart_mlx_eye(n, resolvedM, k, dtype.value)),
    );
  }

  static MlxArray identity(
    int n, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_identity', shim.dart_mlx_identity(n, dtype.value)),
    );
  }

  static MlxArray hamming(int m) =>
      MlxArray._(_checkHandle('dart_mlx_hamming', shim.dart_mlx_hamming(m)));

  static MlxArray hanning(int m) =>
      MlxArray._(_checkHandle('dart_mlx_hanning', shim.dart_mlx_hanning(m)));

  static MlxArray tri(
    int n, {
    int m = -1,
    int k = 0,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) {
    final resolvedM = m < 0 ? n : m;
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_tri', shim.dart_mlx_tri(n, resolvedM, k, dtype.value)),
    );
  }

  static MlxArray tril(MlxArray input, {int k = 0}) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_tril', shim.dart_mlx_tril(input._handle, k)),
    );
  }

  static MlxArray triu(MlxArray input, {int k = 0}) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_triu', shim.dart_mlx_triu(input._handle, k)),
    );
  }

  static MlxArray trace(
    MlxArray input, {
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_trace',
        shim.dart_mlx_trace(input._handle, offset, axis1, axis2, dtype.value),
      ),
    );
  }

  static MlxArray _scan(
    MlxArray input,
    int axis,
    bool reverse,
    bool inclusive,
    int op,
    String name,
  ) {
    _clearError();
    return MlxArray._(
      _checkHandle(name, shim.dart_mlx_scan(input._handle, axis, reverse, inclusive, op)),
    );
  }
}

extension MlxModuleScanExt on MlxModule {
  MlxArray cumsum(MlxArray input, {int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cumsum(input, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray cumprod(MlxArray input, {int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cumprod(input, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray cummax(MlxArray input, {int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cummax(input, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray cummin(MlxArray input, {int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cummin(input, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray logcumsumexp(
    MlxArray input, {
    int axis = 0,
    bool reverse = false,
    bool inclusive = true,
  }) => MlxScan.logcumsumexp(input, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray eye(int n, {int m = -1, int k = 0, MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32}) =>
      MlxScan.eye(n, m: m, k: k, dtype: dtype);

  MlxArray identity(int n, {MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32}) =>
      MlxScan.identity(n, dtype: dtype);

  MlxArray hamming(int m) => MlxScan.hamming(m);

  MlxArray hanning(int m) => MlxScan.hanning(m);

  MlxArray tri(int n, {int m = -1, int k = 0, MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32}) =>
      MlxScan.tri(n, m: m, k: k, dtype: dtype);

  MlxArray tril(MlxArray input, {int k = 0}) => MlxScan.tril(input, k: k);

  MlxArray triu(MlxArray input, {int k = 0}) => MlxScan.triu(input, k: k);

  MlxArray trace(
    MlxArray input, {
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxScan.trace(input, offset: offset, axis1: axis1, axis2: axis2, dtype: dtype);
}

extension MlxArrayScanExt on MlxArray {
  MlxArray cumsum({int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cumsum(this, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray cumprod({int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cumprod(this, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray cummax({int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cummax(this, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray cummin({int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.cummin(this, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray logcumsumexp({int axis = 0, bool reverse = false, bool inclusive = true}) =>
      MlxScan.logcumsumexp(this, axis: axis, reverse: reverse, inclusive: inclusive);

  MlxArray tril({int k = 0}) => MlxScan.tril(this, k: k);

  MlxArray triu({int k = 0}) => MlxScan.triu(this, k: k);

  MlxArray trace({
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxScan.trace(this, offset: offset, axis1: axis1, axis2: axis2, dtype: dtype);
}
