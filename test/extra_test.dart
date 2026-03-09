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
  test('supports diag, diagonal, kron, meshgrid, partition, and scatter', () {
    final vector = MlxArray.fromFloat32List([1, 2, 3], shape: [3]);
    final matrix = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final a = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final b = MlxArray.fromFloat32List([10, 20, 30], shape: [3]);
    final partitionInput = MlxArray.fromFloat32List([3, 1, 2], shape: [3]);

    final diag = mx.diag(vector);
    final diagonal = mx.diagonal(matrix);
    final kron = mx.kron(a, b);
    final mesh = mx.meshgrid([a, b], indexing: 'ij');
    final partition = mx.partition(partitionInput, 1);

    final scatterBase = MlxArray.zeros([3]);
    final scatterIdx = MlxArray.fromInt32List([1, 2], shape: [2]);
    final scatterUpdates = MlxArray.fromFloat32List([5, 7], shape: [2, 1]);
    final scatterSingleUpdates = MlxArray.fromFloat32List([5, 7], shape: [2, 1]);
    final scatter = mx.scatter(scatterBase, [scatterIdx], scatterUpdates, axes: [0]);
    final scatterSingle =
        mx.scatterSingle(scatterBase, scatterIdx, scatterSingleUpdates, axis: 0);

    final addIdx = MlxArray.fromInt32List([1, 1], shape: [2]);
    final addUpdates = MlxArray.fromFloat32List([2, 3], shape: [2, 1]);
    final scatterAdd = mx.scatterAddSingle(scatterBase, addIdx, addUpdates, axis: 0);

    final maxUpdates = MlxArray.fromFloat32List([2, 3], shape: [2, 1]);
    final scatterMax = mx.scatterMaxSingle(scatterBase, addIdx, maxUpdates, axis: 0);

    final minBase = MlxArray.full([3], 10);
    final minUpdates = MlxArray.fromFloat32List([2, 3], shape: [2, 1]);
    final scatterMin = mx.scatterMinSingle(minBase, addIdx, minUpdates, axis: 0);

    final prodBase = MlxArray.ones([3]);
    final prodUpdates = MlxArray.fromFloat32List([2, 3], shape: [2, 1]);
    final scatterProd = mx.scatterProdSingle(prodBase, addIdx, prodUpdates, axis: 0);

    try {
      expect(diag.shape, <int>[3, 3]);
      expect(diag.toList(), <Object>[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
      expect(diagonal.toList(), <Object>[1.0, 4.0]);
      expect(kron.shape, <int>[6]);
      expect(kron.toList(), <Object>[10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);

      expect(mesh, hasLength(2));
      expect(mesh[0].shape, <int>[2, 3]);
      expect(mesh[1].shape, <int>[2, 3]);
      expect(mesh[0].toList(), <Object>[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
      expect(mesh[1].toList(), <Object>[10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);

      final partitionValues = List<double>.from(partition.toList().cast<double>());
      expect(partitionValues[1], closeTo(2.0, 1e-6));
      expect(partitionValues.toSet(), {1.0, 2.0, 3.0});

      expect(scatter.toList(), <Object>[0.0, 5.0, 7.0]);
      expect(scatterSingle.toList(), <Object>[0.0, 5.0, 7.0]);
      expect(scatterAdd.toList(), <Object>[0.0, 5.0, 0.0]);
      expect(scatterMax.toList(), <Object>[0.0, 3.0, 0.0]);
      expect(scatterMin.toList(), <Object>[10.0, 2.0, 10.0]);
      expect(scatterProd.toList(), <Object>[1.0, 6.0, 1.0]);
    } finally {
      scatterProd.close();
      prodUpdates.close();
      prodBase.close();
      scatterMin.close();
      minUpdates.close();
      minBase.close();
      scatterMax.close();
      maxUpdates.close();
      scatterAdd.close();
      addUpdates.close();
      addIdx.close();
      scatterSingle.close();
      scatter.close();
      scatterSingleUpdates.close();
      scatterUpdates.close();
      scatterIdx.close();
      scatterBase.close();
      partition.close();
      for (final value in mesh) {
        value.close();
      }
      kron.close();
      diagonal.close();
      diag.close();
      partitionInput.close();
      b.close();
      a.close();
      matrix.close();
      vector.close();
    }
  });
}
