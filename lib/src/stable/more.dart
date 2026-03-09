part of '../stable_api.dart';

abstract final class MlxMore {
  static MlxArray greater(MlxArray a, MlxArray b) =>
      MlxArray._(_checkHandle('dart_mlx_greater', shim.dart_mlx_greater(a._handle, b._handle)));

  static MlxArray greaterEqual(MlxArray a, MlxArray b) => MlxArray._(
    _checkHandle(
      'dart_mlx_greater_equal',
      shim.dart_mlx_greater_equal(a._handle, b._handle),
    ),
  );

  static MlxArray less(MlxArray a, MlxArray b) =>
      MlxArray._(_checkHandle('dart_mlx_less', shim.dart_mlx_less(a._handle, b._handle)));

  static MlxArray lessEqual(MlxArray a, MlxArray b) => MlxArray._(
    _checkHandle(
      'dart_mlx_less_equal',
      shim.dart_mlx_less_equal(a._handle, b._handle),
    ),
  );

  static MlxArray floorDivide(MlxArray a, MlxArray b) => MlxArray._(
    _checkHandle(
      'dart_mlx_floor_divide',
      shim.dart_mlx_floor_divide(a._handle, b._handle),
    ),
  );

  static MlxArray logaddexp(MlxArray a, MlxArray b) => MlxArray._(
    _checkHandle(
      'dart_mlx_logaddexp',
      shim.dart_mlx_logaddexp(a._handle, b._handle),
    ),
  );

  static MlxArray inner(MlxArray a, MlxArray b) =>
      MlxArray._(_checkHandle('dart_mlx_inner', shim.dart_mlx_inner(a._handle, b._handle)));

  static MlxArray hadamardTransform(MlxArray input, {double? scale}) => MlxArray._(
    _checkHandle(
      'dart_mlx_hadamard_transform',
      shim.dart_mlx_hadamard_transform(input._handle, scale != null, scale ?? 0),
    ),
  );

  static MlxArray floor(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_floor', shim.dart_mlx_floor(input._handle)));

  static MlxArray sqrt(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_sqrt', shim.dart_mlx_sqrt(input._handle)));

  static MlxArray rsqrt(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_rsqrt', shim.dart_mlx_rsqrt(input._handle)));

  static MlxArray square(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_square', shim.dart_mlx_square(input._handle)));

  static MlxArray reciprocal(MlxArray input) => MlxArray._(
    _checkHandle(
      'dart_mlx_reciprocal',
      shim.dart_mlx_reciprocal(input._handle),
    ),
  );

  static MlxArray sigmoid(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_sigmoid', shim.dart_mlx_sigmoid(input._handle)));

  static MlxArray degrees(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_degrees', shim.dart_mlx_degrees(input._handle)));

  static MlxArray radians(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_radians', shim.dart_mlx_radians(input._handle)));

  static MlxArray expm1(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_expm1', shim.dart_mlx_expm1(input._handle)));

  static MlxArray erf(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_erf', shim.dart_mlx_erf(input._handle)));

  static MlxArray erfinv(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_erfinv', shim.dart_mlx_erfinv(input._handle)));

  static MlxArray log1p(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_log1p', shim.dart_mlx_log1p(input._handle)));

  static MlxArray log2(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_log2', shim.dart_mlx_log2(input._handle)));

  static MlxArray log10(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_log10', shim.dart_mlx_log10(input._handle)));

  static MlxArray round(MlxArray input, {int decimals = 0}) => MlxArray._(
    _checkHandle('dart_mlx_round', shim.dart_mlx_round(input._handle, decimals)),
  );

  static MlxArray stopGradient(MlxArray input) => MlxArray._(
    _checkHandle(
      'dart_mlx_stop_gradient',
      shim.dart_mlx_stop_gradient(input._handle),
    ),
  );

  static MlxArray isFinite(MlxArray input) => MlxArray._(
    _checkHandle('dart_mlx_isfinite', shim.dart_mlx_isfinite(input._handle)),
  );

  static MlxArray isInf(MlxArray input) => MlxArray._(
    _checkHandle('dart_mlx_isinf', shim.dart_mlx_isinf(input._handle)),
  );

  static MlxArray isNaN(MlxArray input) => MlxArray._(
    _checkHandle('dart_mlx_isnan', shim.dart_mlx_isnan(input._handle)),
  );

  static MlxArray isNegInf(MlxArray input) => MlxArray._(
    _checkHandle('dart_mlx_isneginf', shim.dart_mlx_isneginf(input._handle)),
  );

  static MlxArray isPosInf(MlxArray input) => MlxArray._(
    _checkHandle('dart_mlx_isposinf', shim.dart_mlx_isposinf(input._handle)),
  );

  static MlxArray zerosLike(MlxArray input) => MlxArray._(
    _checkHandle('dart_mlx_zeros_like', shim.dart_mlx_zeros_like(input._handle)),
  );

  static MlxArray onesLike(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_ones_like', shim.dart_mlx_ones_like(input._handle)));

  static MlxArray fullLike(
    MlxArray input,
    double value, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxArray._(
    _checkHandle(
      'dart_mlx_full_like',
      shim.dart_mlx_full_like(input._handle, value, dtype.value),
    ),
  );

  static MlxArray toFp8(MlxArray input) =>
      MlxArray._(_checkHandle('dart_mlx_to_fp8', shim.dart_mlx_to_fp8(input._handle)));

  static MlxArray fromFp8(
    MlxArray input, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxArray._(
    _checkHandle(
      'dart_mlx_from_fp8',
      shim.dart_mlx_from_fp8(input._handle, dtype.value),
    ),
  );

  static MlxArray putAlongAxis(
    MlxArray input,
    MlxArray indices,
    MlxArray values, {
    required int axis,
  }) => MlxArray._(
    _checkHandle(
      'dart_mlx_put_along_axis',
      shim.dart_mlx_put_along_axis(
        input._handle,
        indices._handle,
        values._handle,
        axis,
      ),
    ),
  );

  static MlxArray scatterAddAxis(
    MlxArray input,
    MlxArray indices,
    MlxArray values, {
    required int axis,
  }) => MlxArray._(
    _checkHandle(
      'dart_mlx_scatter_add_axis',
      shim.dart_mlx_scatter_add_axis(
        input._handle,
        indices._handle,
        values._handle,
        axis,
      ),
    ),
  );
}

extension MlxModuleMoreExt on MlxModule {
  MlxArray greater(MlxArray a, MlxArray b) => MlxMore.greater(a, b);
  MlxArray greaterEqual(MlxArray a, MlxArray b) => MlxMore.greaterEqual(a, b);
  MlxArray less(MlxArray a, MlxArray b) => MlxMore.less(a, b);
  MlxArray lessEqual(MlxArray a, MlxArray b) => MlxMore.lessEqual(a, b);
  MlxArray floorDivide(MlxArray a, MlxArray b) => MlxMore.floorDivide(a, b);
  MlxArray logaddexp(MlxArray a, MlxArray b) => MlxMore.logaddexp(a, b);
  MlxArray inner(MlxArray a, MlxArray b) => MlxMore.inner(a, b);
  MlxArray hadamardTransform(MlxArray input, {double? scale}) =>
      MlxMore.hadamardTransform(input, scale: scale);
  MlxArray floor(MlxArray input) => MlxMore.floor(input);
  MlxArray sqrt(MlxArray input) => MlxMore.sqrt(input);
  MlxArray rsqrt(MlxArray input) => MlxMore.rsqrt(input);
  MlxArray square(MlxArray input) => MlxMore.square(input);
  MlxArray reciprocal(MlxArray input) => MlxMore.reciprocal(input);
  MlxArray sigmoid(MlxArray input) => MlxMore.sigmoid(input);
  MlxArray degrees(MlxArray input) => MlxMore.degrees(input);
  MlxArray radians(MlxArray input) => MlxMore.radians(input);
  MlxArray expm1(MlxArray input) => MlxMore.expm1(input);
  MlxArray erf(MlxArray input) => MlxMore.erf(input);
  MlxArray erfinv(MlxArray input) => MlxMore.erfinv(input);
  MlxArray log1p(MlxArray input) => MlxMore.log1p(input);
  MlxArray log2(MlxArray input) => MlxMore.log2(input);
  MlxArray log10(MlxArray input) => MlxMore.log10(input);
  MlxArray round(MlxArray input, {int decimals = 0}) =>
      MlxMore.round(input, decimals: decimals);
  MlxArray stopGradient(MlxArray input) => MlxMore.stopGradient(input);
  MlxArray isFinite(MlxArray input) => MlxMore.isFinite(input);
  MlxArray isInf(MlxArray input) => MlxMore.isInf(input);
  MlxArray isNaN(MlxArray input) => MlxMore.isNaN(input);
  MlxArray isNegInf(MlxArray input) => MlxMore.isNegInf(input);
  MlxArray isPosInf(MlxArray input) => MlxMore.isPosInf(input);
  MlxArray zerosLike(MlxArray input) => MlxMore.zerosLike(input);
  MlxArray onesLike(MlxArray input) => MlxMore.onesLike(input);
  MlxArray fullLike(
    MlxArray input,
    double value, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxMore.fullLike(input, value, dtype: dtype);
  MlxArray toFp8(MlxArray input) => MlxMore.toFp8(input);
  MlxArray fromFp8(MlxArray input, {MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32}) =>
      MlxMore.fromFp8(input, dtype: dtype);
  MlxArray putAlongAxis(
    MlxArray input,
    MlxArray indices,
    MlxArray values, {
    required int axis,
  }) => MlxMore.putAlongAxis(input, indices, values, axis: axis);

  MlxArray scatterAddAxis(
    MlxArray input,
    MlxArray indices,
    MlxArray values, {
    required int axis,
  }) => MlxMore.scatterAddAxis(input, indices, values, axis: axis);
}

extension MlxArrayMoreExt on MlxArray {
  MlxArray floor() => MlxMore.floor(this);
  MlxArray sqrt() => MlxMore.sqrt(this);
  MlxArray rsqrt() => MlxMore.rsqrt(this);
  MlxArray square() => MlxMore.square(this);
  MlxArray reciprocal() => MlxMore.reciprocal(this);
  MlxArray sigmoid() => MlxMore.sigmoid(this);
  MlxArray degrees() => MlxMore.degrees(this);
  MlxArray radians() => MlxMore.radians(this);
  MlxArray expm1() => MlxMore.expm1(this);
  MlxArray erf() => MlxMore.erf(this);
  MlxArray erfinv() => MlxMore.erfinv(this);
  MlxArray log1p() => MlxMore.log1p(this);
  MlxArray log2() => MlxMore.log2(this);
  MlxArray log10() => MlxMore.log10(this);
  MlxArray round({int decimals = 0}) => MlxMore.round(this, decimals: decimals);
  MlxArray stopGradient() => MlxMore.stopGradient(this);
  MlxArray isFinite() => MlxMore.isFinite(this);
  MlxArray isInf() => MlxMore.isInf(this);
  MlxArray isNaN() => MlxMore.isNaN(this);
  MlxArray isNegInf() => MlxMore.isNegInf(this);
  MlxArray isPosInf() => MlxMore.isPosInf(this);
  MlxArray zerosLike() => MlxMore.zerosLike(this);
  MlxArray onesLike() => MlxMore.onesLike(this);
  MlxArray fullLike(double value, {MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32}) =>
      MlxMore.fullLike(this, value, dtype: dtype);
  MlxArray toFp8() => MlxMore.toFp8(this);
  MlxArray fromFp8({MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32}) =>
      MlxMore.fromFp8(this, dtype: dtype);
  MlxArray hadamardTransform({double? scale}) =>
      MlxMore.hadamardTransform(this, scale: scale);
}
