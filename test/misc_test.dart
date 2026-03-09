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
  test('supports miscellaneous numeric helpers', () {
    final linspace = mx.linspace(0, 1, 5);
    final outer = mx.outer(
      MlxArray.fromFloat32List([1, 2], shape: [2]),
      MlxArray.fromFloat32List([3, 4], shape: [2]),
    );
    final isClose = mx.isClose(
      MlxArray.fromFloat32List([1, 2], shape: [2]),
      MlxArray.fromFloat32List([1.0, 2.000001], shape: [2]),
    );
    final a = MlxArray.fromBoolList([true, false], shape: [2]);
    final b = MlxArray.fromBoolList([true, true], shape: [2]);
    final anded = mx.logicalAnd(a, b);
    final ored = mx.logicalOr(a, b);
    final noted = a.logicalNot();
    final repeated = MlxArray.fromFloat32List([1, 2], shape: [2]).repeat(2);
    final rolled = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [4]).roll([1]);
    final median = MlxArray.fromFloat32List([3, 1, 2], shape: [3]).median();
    final masked = mx.maskedScatter(
      MlxArray.zeros([3]),
      MlxArray.fromBoolList([true, false, true], shape: [3]),
      MlxArray.fromFloat32List([5, 7], shape: [2]),
    );
    final ntn = MlxArray.fromFloat32List(
      [double.nan, double.infinity, double.negativeInfinity, 1],
      shape: [4],
    ).nanToNum(posInf: 9, negInf: -9);
    final divmod = mx.divmod(
      MlxArray.fromInt32List([7, 8], shape: [2]),
      MlxArray.fromInt32List([3, 3], shape: [2]),
    );

    try {
      expect(linspace.toList(), <Object>[0.0, 0.25, 0.5, 0.75, 1.0]);
      expect(outer.toList(), <Object>[3.0, 4.0, 6.0, 8.0]);
      expect(isClose.toList(), <Object>[true, true]);
      expect(anded.toList(), <Object>[true, false]);
      expect(ored.toList(), <Object>[true, true]);
      expect(noted.toList(), <Object>[false, true]);
      expect(repeated.toList(), <Object>[1.0, 1.0, 2.0, 2.0]);
      expect(rolled.toList(), <Object>[4.0, 1.0, 2.0, 3.0]);
      expect(median.toList(), <Object>[2.0]);
      expect(masked.toList(), <Object>[5.0, 0.0, 7.0]);
      expect(ntn.toList(), <Object>[0.0, 9.0, -9.0, 1.0]);
      expect(divmod.quotient.toList(), <Object>[2, 2]);
      expect(divmod.remainder.toList(), <Object>[1, 2]);
    } finally {
      divmod.remainder.close();
      divmod.quotient.close();
      ntn.close();
      masked.close();
      median.close();
      rolled.close();
      repeated.close();
      noted.close();
      ored.close();
      anded.close();
      b.close();
      a.close();
      isClose.close();
      outer.close();
      linspace.close();
    }
  });
}
