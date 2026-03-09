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
  test('supports dynamic slice helpers and gatherMm', () {
    final vector = MlxArray.fromFloat32List([10, 20, 30, 40], shape: [4]);
    final start = MlxArray.fromInt32List([1], shape: [1]);
    final update = MlxArray.fromFloat32List([99, 88], shape: [2]);
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
    final base = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final padValue = MlxArray.full([], 0);
    final reshaped = base.reshape([1, 2, 2]);
    final small = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final paddedBase = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final paddedSymBase = MlxArray.fromFloat32List([1, 2], shape: [2]);

    final sliced = vector.sliceDynamic(start: start, axes: [0], sliceSize: [2]);
    final updated = vector.sliceUpdate(update, start: [1], stop: [3]);
    final updatedDynamic = vector.sliceUpdateDynamic(update, start: start, axes: [0]);
    final gatheredMm = mx.gatherMm(a, b);
    final ref = a.matmul(b);
    final flattened = base.flatten();
    final tiled = small.tile([2]);
    final moved = reshaped.moveaxis(0, 2);
    final swapped = base.swapaxes(0, 1);
    final transposed = base.transposeAxes([1, 0]);
    final padded = paddedBase.pad(
      lowPads: [1],
      highPads: [2],
      padValue: padValue,
    );
    final paddedSym = paddedSymBase.padSymmetric(
      1,
      padValue: padValue,
    );
    final unflattened = flattened.unflatten(axis: 0, shape: [2, 2]);

    try {
      expect(sliced.toList(), <Object>[20.0, 30.0]);
      expect(updated.toList(), <Object>[10.0, 99.0, 88.0, 40.0]);
      expect(updatedDynamic.toList(), <Object>[10.0, 99.0, 88.0, 40.0]);
      expect(gatheredMm.toList(), ref.toList());
      expect(flattened.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(tiled.toList(), <Object>[1.0, 2.0, 1.0, 2.0]);
      expect(moved.shape, <int>[2, 2, 1]);
      expect(swapped.toList(), <Object>[1.0, 3.0, 2.0, 4.0]);
      expect(transposed.toList(), <Object>[1.0, 3.0, 2.0, 4.0]);
      expect(padded.toList(), <Object>[0.0, 1.0, 2.0, 0.0, 0.0]);
      expect(paddedSym.toList(), <Object>[0.0, 1.0, 2.0, 0.0]);
      expect(unflattened.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
    } finally {
      unflattened.close();
      paddedSym.close();
      padded.close();
      transposed.close();
      swapped.close();
      moved.close();
      tiled.close();
      flattened.close();
      paddedSymBase.close();
      paddedBase.close();
      small.close();
      reshaped.close();
      ref.close();
      gatheredMm.close();
      updatedDynamic.close();
      updated.close();
      sliced.close();
      padValue.close();
      base.close();
      b.close();
      a.close();
      update.close();
      start.close();
      vector.close();
    }
  });
}
