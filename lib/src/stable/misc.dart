part of '../stable_api.dart';

abstract final class MlxMisc {
  static MlxArray linspace(
    double start,
    double stop,
    int num, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_linspace',
        shim.dart_mlx_linspace(start, stop, num, dtype.value),
      ),
    );
  }

  static MlxArray outer(MlxArray a, MlxArray b) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_outer', shim.dart_mlx_outer(a._handle, b._handle)),
    );
  }

  static MlxArray isClose(
    MlxArray a,
    MlxArray b, {
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equalNan = false,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_isclose',
        shim.dart_mlx_isclose(a._handle, b._handle, rtol, atol, equalNan),
      ),
    );
  }

  static MlxArray logicalAnd(MlxArray a, MlxArray b) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_logical_and',
        shim.dart_mlx_logical_and(a._handle, b._handle),
      ),
    );
  }

  static MlxArray logicalOr(MlxArray a, MlxArray b) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_logical_or',
        shim.dart_mlx_logical_or(a._handle, b._handle),
      ),
    );
  }

  static MlxArray logicalNot(MlxArray input) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_logical_not', shim.dart_mlx_logical_not(input._handle)),
    );
  }

  static MlxArray repeat(MlxArray input, int repeats, {int? axis}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_repeat',
        shim.dart_mlx_repeat(input._handle, repeats, axis ?? 0, axis != null),
      ),
    );
  }

  static MlxArray roll(
    MlxArray input,
    List<int> shift, {
    int? axis,
    List<int>? axes,
  }) {
    if (axis != null && axes != null) {
      throw ArgumentError('roll() accepts either axis or axes, not both.');
    }
    return _withInts(shift, (shiftPtr, shiftLen) {
      return _withInts(axes ?? const [], (axesPtr, axesLen) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_roll',
            shim.dart_mlx_roll(
              input._handle,
              shiftPtr,
              shiftLen,
              axesPtr,
              axesLen,
              axis ?? 0,
              axis != null ? 1 : (axes != null ? 2 : 0),
            ),
          ),
        );
      });
    });
  }

  static MlxArray median(
    MlxArray input, {
    List<int>? axes,
    bool keepDims = false,
  }) {
    final resolvedAxes = axes ?? List<int>.generate(input.ndim, (index) => index);
    return _withInts(resolvedAxes, (axesPtr, axesLen) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_median',
        shim.dart_mlx_median(input._handle, axesPtr, axesLen, keepDims),
      ),
    );
  });
  }

  static MlxArray maskedScatter(
    MlxArray input,
    MlxArray mask,
    MlxArray values,
  ) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_masked_scatter',
        shim.dart_mlx_masked_scatter(input._handle, mask._handle, values._handle),
      ),
    );
  }

  static MlxArray nanToNum(
    MlxArray input, {
    double nan = 0,
    double? posInf,
    double? negInf,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_nan_to_num',
        shim.dart_mlx_nan_to_num(
          input._handle,
          nan,
          posInf != null,
          posInf ?? 0,
          negInf != null,
          negInf ?? 0,
        ),
      ),
    );
  }

  static MlxDivModResult divmod(MlxArray a, MlxArray b) {
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_divmod',
        shim.dart_mlx_divmod(a._handle, b._handle, outputsOut, outputsLen),
      );
      final outputs = _readOutputArrayList(outputsOut.value, outputsLen.value);
      return (quotient: outputs[0], remainder: outputs[1]);
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
    }
  }
}

extension MlxModuleMiscExt on MlxModule {
  MlxArray linspace(
    double start,
    double stop,
    int num, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxMisc.linspace(start, stop, num, dtype: dtype);

  MlxArray outer(MlxArray a, MlxArray b) => MlxMisc.outer(a, b);

  MlxArray isClose(
    MlxArray a,
    MlxArray b, {
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equalNan = false,
  }) => MlxMisc.isClose(a, b, rtol: rtol, atol: atol, equalNan: equalNan);

  MlxArray logicalAnd(MlxArray a, MlxArray b) => MlxMisc.logicalAnd(a, b);

  MlxArray logicalOr(MlxArray a, MlxArray b) => MlxMisc.logicalOr(a, b);

  MlxArray logicalNot(MlxArray input) => MlxMisc.logicalNot(input);

  MlxArray repeat(MlxArray input, int repeats, {int? axis}) =>
      MlxMisc.repeat(input, repeats, axis: axis);

  MlxArray roll(MlxArray input, List<int> shift, {int? axis, List<int>? axes}) =>
      MlxMisc.roll(input, shift, axis: axis, axes: axes);

  MlxArray median(MlxArray input, {List<int>? axes, bool keepDims = false}) =>
      MlxMisc.median(input, axes: axes, keepDims: keepDims);

  MlxArray maskedScatter(MlxArray input, MlxArray mask, MlxArray values) =>
      MlxMisc.maskedScatter(input, mask, values);

  MlxArray nanToNum(
    MlxArray input, {
    double nan = 0,
    double? posInf,
    double? negInf,
  }) => MlxMisc.nanToNum(input, nan: nan, posInf: posInf, negInf: negInf);

  MlxDivModResult divmod(MlxArray a, MlxArray b) => MlxMisc.divmod(a, b);
}

extension MlxArrayMiscExt on MlxArray {
  MlxArray logicalNot() => MlxMisc.logicalNot(this);

  MlxArray repeat(int repeats, {int? axis}) => MlxMisc.repeat(this, repeats, axis: axis);

  MlxArray roll(List<int> shift, {int? axis, List<int>? axes}) =>
      MlxMisc.roll(this, shift, axis: axis, axes: axes);

  MlxArray median({List<int>? axes, bool keepDims = false}) =>
      MlxMisc.median(this, axes: axes, keepDims: keepDims);

  MlxArray nanToNum({double nan = 0, double? posInf, double? negInf}) =>
      MlxMisc.nanToNum(this, nan: nan, posInf: posInf, negInf: negInf);
}
