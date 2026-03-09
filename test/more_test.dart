// ignore_for_file: unused_import

@TestOn('mac-os')

library;

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/raw.dart' as raw;
import 'package:dart_mlx_ffi/src/internal_hooks.dart' as hooks;

void main() {
  test('supports comparison, transforms, and like/fp8 helpers', () {
    final a = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final b = MlxArray.fromFloat32List([2, 2], shape: [2]);
    final greater = mx.greater(a, b);
    final greaterEqual = mx.greaterEqual(a, b);
    final less = mx.less(a, b);
    final lessEqual = mx.lessEqual(a, b);
    final floorDivide = mx.floorDivide(
      MlxArray.fromFloat32List([5, 7], shape: [2]),
      MlxArray.fromFloat32List([2, 2], shape: [2]),
    );
    final logaddexp = mx.logaddexp(a, a);
    final inner = mx.inner(a, b);
    final floor = MlxArray.fromFloat32List([1.2, 2.9], shape: [2]).floor();
    final sqrt = MlxArray.fromFloat32List([1, 4], shape: [2]).sqrt();
    final rsqrt = MlxArray.fromFloat32List([1, 4], shape: [2]).rsqrt();
    final square = a.square();
    final reciprocal = MlxArray.fromFloat32List([2, 4], shape: [2]).reciprocal();
    final sigmoid = MlxArray.fromFloat32List([0], shape: [1]).sigmoid();
    final degrees = MlxArray.fromFloat32List([3.14159265], shape: [1]).degrees();
    final radians = MlxArray.fromFloat32List([180], shape: [1]).radians();
    final expm1 = MlxArray.fromFloat32List([0], shape: [1]).expm1();
    final erf = MlxArray.fromFloat32List([0], shape: [1]).erf();
    final erfinv = MlxArray.fromFloat32List([0], shape: [1]).erfinv();
    final log1p = MlxArray.fromFloat32List([1], shape: [1]).log1p();
    final log2 = MlxArray.fromFloat32List([8], shape: [1]).log2();
    final log10 = MlxArray.fromFloat32List([100], shape: [1]).log10();
    final stopped = a.stopGradient();
    final zerosLike = a.zerosLike();
    final onesLike = a.onesLike();
    final fullLike = a.fullLike(7);
    final fp8 = a.toFp8();
    final fromFp8 = fp8.fromFp8();
    final special = MlxArray.fromFloat32List(
      [double.nan, double.infinity, double.negativeInfinity, 1],
      shape: [4],
    );
    final isFinite = special.isFinite();
    final isInf = special.isInf();
    final isNaN = special.isNaN();
    final isNegInf = special.isNegInf();
    final isPosInf = special.isPosInf();
    final hadamard = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [4]).hadamardTransform();
    final putAlong = mx.putAlongAxis(
      MlxArray.zeros([3]),
      MlxArray.fromInt32List([1, 2], shape: [2]),
      MlxArray.fromFloat32List([5, 7], shape: [2]),
      axis: 0,
    );
    final rounded = MlxArray.fromFloat32List([1.234], shape: [1]).round(decimals: 2);
    final scatterAddAxis = mx.scatterAddAxis(
      MlxArray.zeros([3]),
      MlxArray.fromInt32List([1, 1], shape: [2]),
      MlxArray.fromFloat32List([2, 3], shape: [2]),
      axis: 0,
    );

    try {
      expect(greater.toList(), <Object>[false, false]);
      expect(greaterEqual.toList(), <Object>[false, true]);
      expect(less.toList(), <Object>[true, false]);
      expect(lessEqual.toList(), <Object>[true, true]);
      expect(floorDivide.toList(), <Object>[2.0, 3.0]);
      expect(
        (logaddexp.toList()[0] as double),
        closeTo(1.693147, 1e-5),
      );
      expect(inner.toList(), <Object>[6.0]);
      expect(floor.toList(), <Object>[1.0, 2.0]);
      expect(sqrt.toList(), <Object>[1.0, 2.0]);
      expect(rsqrt.toList(), <Object>[1.0, 0.5]);
      expect(square.toList(), <Object>[1.0, 4.0]);
      expect(reciprocal.toList(), <Object>[0.5, 0.25]);
      expect((sigmoid.toList()[0] as double), closeTo(0.5, 1e-5));
      expect((degrees.toList()[0] as double), closeTo(180.0, 1e-3));
      expect((radians.toList()[0] as double), closeTo(3.14159265, 1e-3));
      expect((expm1.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((erf.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((erfinv.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((log1p.toList()[0] as double), closeTo(0.693147, 1e-5));
      expect((log2.toList()[0] as double), closeTo(3.0, 1e-5));
      expect((log10.toList()[0] as double), closeTo(2.0, 1e-5));
      expect(stopped.toList(), <Object>[1.0, 2.0]);
      expect(zerosLike.toList(), <Object>[0.0, 0.0]);
      expect(onesLike.toList(), <Object>[1.0, 1.0]);
      expect(fullLike.toList(), <Object>[7.0, 7.0]);
      expect(fp8.dtype, MlxDType.MLX_UINT8);
      expect(fromFp8.shape, a.shape);
      expect(isFinite.toList(), <Object>[false, false, false, true]);
      expect(isInf.toList(), <Object>[false, true, true, false]);
      expect(isNaN.toList(), <Object>[true, false, false, false]);
      expect(isNegInf.toList(), <Object>[false, false, true, false]);
      expect(isPosInf.toList(), <Object>[false, true, false, false]);
      expect(hadamard.shape, <int>[4]);
      expect(putAlong.toList(), <Object>[0.0, 5.0, 7.0]);
      expect((rounded.toList()[0] as double), closeTo(1.23, 1e-6));
      expect(scatterAddAxis.toList(), <Object>[0.0, 5.0, 0.0]);
    } finally {
      scatterAddAxis.close();
      rounded.close();
      putAlong.close();
      hadamard.close();
      isPosInf.close();
      isNegInf.close();
      isNaN.close();
      isInf.close();
      isFinite.close();
      special.close();
      fromFp8.close();
      fp8.close();
      fullLike.close();
      onesLike.close();
      zerosLike.close();
      stopped.close();
      log10.close();
      log2.close();
      log1p.close();
      erfinv.close();
      erf.close();
      expm1.close();
      radians.close();
      degrees.close();
      sigmoid.close();
      reciprocal.close();
      square.close();
      rsqrt.close();
      sqrt.close();
      floor.close();
      inner.close();
      logaddexp.close();
      floorDivide.close();
      lessEqual.close();
      less.close();
      greaterEqual.close();
      greater.close();
      b.close();
      a.close();
    }
  });
}
