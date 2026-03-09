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
  test('supports advanced FFT helpers', () {
    final signal = MlxArray.fromFloat32List([0, 1, 0, -1], shape: [2, 2]);
    final fft2 = mx.fft.fft2(signal);
    final fftn = mx.fft.fftn(signal);
    final rfft2 = mx.fft.rfft2(signal);
    final irfft2 = mx.fft.irfft2(rfft2, n: [2, 2], axes: [0, 1]);
    final rfftn = mx.fft.rfftn(signal);
    final irfftn = mx.fft.irfftn(rfftn, n: [2, 2], axes: [0, 1]);
    final shifted = mx.fft.fftshift(
      MlxArray.fromFloat32List([1, 2, 3, 4], shape: [4]),
      axes: [0],
    );
    final unshifted = mx.fft.ifftshift(shifted, axes: [0]);

    try {
      expect(fft2.shape, <int>[2, 2]);
      expect(fftn.shape, <int>[2, 2]);

      expect(rfft2.shape, <int>[2, 2]);
      expect(irfft2.shape, <int>[2, 2]);
      expect(irfftn.shape, <int>[2, 2]);
      expect(shifted.toList(), <Object>[3.0, 4.0, 1.0, 2.0]);
      expect(unshifted.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
    } finally {
      unshifted.close();
      shifted.close();
      irfftn.close();
      rfftn.close();
      irfft2.close();
      rfft2.close();
      fftn.close();
      fft2.close();
      signal.close();
    }
  });

  test('supports advanced linalg helpers', () {
    final spd = MlxArray.fromFloat32List([4, 1, 1, 3], shape: [2, 2]);
    final diag = MlxArray.fromFloat32List([3, 0, 0, 2], shape: [2, 2]);
    final fullRank = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);

    final chol = mx.linalg.cholesky(spd);
    final cross = mx.linalg.cross(
      MlxArray.fromFloat32List([1, 0, 0], shape: [3]),
      MlxArray.fromFloat32List([0, 1, 0], shape: [3]),
    );
    final eigh = mx.linalg.eigh(spd);
    final eigvals = mx.linalg.eigvals(spd);
    final eigvalsh = mx.linalg.eigvalsh(spd);
    final lu = mx.linalg.lu(fullRank);
    final luFactor = mx.linalg.luFactor(fullRank);
    final pinv = mx.linalg.pinv(fullRank);
    final norm = mx.linalg.norm(MlxArray.fromFloat32List([3, 4], shape: [2]));
    final matrixNorm = mx.linalg.matrixNorm(fullRank);
    final l2Norm = mx.linalg.l2Norm(MlxArray.fromFloat32List([3, 4], shape: [2]));
    final solveTriangular = mx.linalg.solveTriangular(
      MlxArray.fromFloat32List([2, 0, 1, 3], shape: [2, 2]),
      MlxArray.fromFloat32List([2, 7], shape: [2]),
    );
    final svd = mx.linalg.svd(diag);
    final svdValuesOnly = mx.linalg.svd(diag, computeUv: false);

    try {
      final cholValues = List<double>.from(chol.toList().cast<double>());
      expect(chol.shape, <int>[2, 2]);
      expect(cholValues[0], closeTo(2.0, 1e-4));
      expect(cholValues[1], closeTo(0.0, 1e-4));
      expect(cholValues[2], closeTo(0.5, 1e-4));
      expect(cholValues[3], closeTo(1.658312, 1e-4));

      expect(cross.toList(), <Object>[0.0, 0.0, 1.0]);
      final eighValues = List<double>.from(eigh.values.toList().cast<double>());
      expect(eigh.values.shape, <int>[2]);
      expect(eigh.vectors.shape, <int>[2, 2]);
      expect(eighValues[0], closeTo(2.381966, 1e-4));
      expect(eighValues[1], closeTo(4.618034, 1e-4));
      expect(eigvals.shape, <int>[2]);
      expect(eigvalsh.toList(), eigh.values.toList());
      expect(lu.rowPivots.shape, <int>[2]);
      expect(lu.l.shape, <int>[2, 2]);
      expect(lu.u.shape, <int>[2, 2]);
      expect(luFactor.lu.shape, <int>[2, 2]);
      expect(luFactor.pivots.shape, <int>[2]);

      final pinvValues = List<double>.from(pinv.toList().cast<double>());
      expect(pinvValues[0], closeTo(-2.0, 1e-4));
      expect(pinvValues[1], closeTo(1.0, 1e-4));
      expect(pinvValues[2], closeTo(1.5, 1e-4));
      expect(pinvValues[3], closeTo(-0.5, 1e-4));

      expect(norm.toList(), <Object>[5.0]);
      expect((matrixNorm.toList()[0] as double), closeTo(5.477225, 1e-4));
      expect(l2Norm.toList(), <Object>[5.0]);
      final solveTriangularValues = List<double>.from(
        solveTriangular.toList().cast<double>(),
      );
      expect(solveTriangularValues[0], closeTo(1.0, 1e-6));
      expect(solveTriangularValues[1], closeTo(2.0, 1e-6));

      expect(svd.u, isNotNull);
      expect(svd.vt, isNotNull);
      expect(svd.s.toList(), <Object>[3.0, 2.0]);
      expect(svdValuesOnly.u, isNull);
      expect(svdValuesOnly.vt, isNull);
      expect(svdValuesOnly.s.toList(), <Object>[3.0, 2.0]);
    } finally {
      svdValuesOnly.s.close();
      svd.vt?.close();
      svd.s.close();
      svd.u?.close();
      solveTriangular.close();
      l2Norm.close();
      matrixNorm.close();
      norm.close();
      pinv.close();
      luFactor.pivots.close();
      luFactor.lu.close();
      lu.u.close();
      lu.l.close();
      lu.rowPivots.close();
      eigvalsh.close();
      eigvals.close();
      eigh.vectors.close();
      eigh.values.close();
      cross.close();
      chol.close();
      fullRank.close();
      diag.close();
      spd.close();
    }
  });
}
