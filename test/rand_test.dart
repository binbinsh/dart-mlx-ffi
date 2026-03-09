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
  test('supports additional random distributions', () {
    final gumbel = mx.random.gumbel([2, 2]);
    final laplace = mx.random.laplace([2, 2], loc: 1, scale: 0.5);
    final randint = mx.random.randint(0, 10, [8]);
    final mean = MlxArray.fromFloat32List([0, 0], shape: [2]);
    final cov = MlxArray.fromFloat32List([1, 0, 0, 1], shape: [2, 2]);
    final mvn = mx.random.multivariateNormal(mean, cov, shape: [3]);

    try {
      expect(gumbel.shape, <int>[2, 2]);
      expect(gumbel.dtype, MlxDType.MLX_FLOAT32);
      expect(laplace.shape, <int>[2, 2]);
      expect(laplace.dtype, MlxDType.MLX_FLOAT32);
      expect(randint.shape, <int>[8]);
      expect(randint.dtype, MlxDType.MLX_INT32);
      for (final value in randint.toList().cast<int>()) {
        expect(value, inInclusiveRange(0, 9));
      }
      expect(mvn.shape, <int>[3, 2]);
      expect(mvn.dtype, MlxDType.MLX_FLOAT32);
    } finally {
      mvn.close();
      cov.close();
      mean.close();
      randint.close();
      laplace.close();
      gumbel.close();
    }
  });
}
