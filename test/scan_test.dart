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
  test('supports scan and matrix construction helpers', () {
    final input = MlxArray.fromFloat32List([1, 2, 3], shape: [3]);
    final cumsum = input.cumsum();
    final cumprod = input.cumprod();
    final cummax = input.cummax();
    final cummin = input.cummin();
    final logcumsumexp = MlxArray.fromFloat32List([0, 0], shape: [2]).logcumsumexp();

    final eye = mx.eye(3);
    final identity = mx.identity(3);
    final hamming = mx.hamming(4);
    final hanning = mx.hanning(4);
    final tri = mx.tri(3);
    final matrix = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final tril = matrix.tril();
    final triu = matrix.triu();
    final trace = matrix.trace();

    try {
      expect(cumsum.toList(), <Object>[1.0, 3.0, 6.0]);
      expect(cumprod.toList(), <Object>[1.0, 2.0, 6.0]);
      expect(cummax.toList(), <Object>[1.0, 2.0, 3.0]);
      expect(cummin.toList(), <Object>[1.0, 1.0, 1.0]);
      expect(
        (logcumsumexp.toList()[1] as double),
        closeTo(0.693147, 1e-5),
      );

      expect(eye.toList(), <Object>[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
      expect(identity.toList(), <Object>[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
      expect(hamming.shape, <int>[4]);
      expect(hanning.shape, <int>[4]);
      expect(tri.toList(), <Object>[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
      expect(tril.toList(), <Object>[1.0, 0.0, 3.0, 4.0]);
      expect(triu.toList(), <Object>[1.0, 2.0, 0.0, 4.0]);
      expect(trace.toList(), <Object>[5.0]);
    } finally {
      trace.close();
      triu.close();
      tril.close();
      matrix.close();
      tri.close();
      hanning.close();
      hamming.close();
      identity.close();
      eye.close();
      logcumsumexp.close();
      cummin.close();
      cummax.close();
      cumprod.close();
      cumsum.close();
      input.close();
    }
  });
}
