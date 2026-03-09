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
  test('supports broadcastArrays and splitSections', () {
    final a = MlxArray.fromFloat32List([1, 2], shape: [1, 2]);
    final b = MlxArray.fromFloat32List([10, 20], shape: [2, 1]);
    final splitsInput = MlxArray.fromFloat32List([0, 1, 2, 3, 4], shape: [5]);

    final broadcasted = mx.broadcastArrays([a, b]);
    final splits = mx.splitSections(splitsInput, [2, 4]);

    try {
      expect(broadcasted, hasLength(2));
      expect(broadcasted[0].shape, <int>[2, 2]);
      expect(broadcasted[1].shape, <int>[2, 2]);
      expect(broadcasted[0].toList(), <Object>[1.0, 2.0, 1.0, 2.0]);
      expect(broadcasted[1].toList(), <Object>[10.0, 10.0, 20.0, 20.0]);

      expect(splits, hasLength(3));
      expect(splits[0].toList(), <Object>[0.0, 1.0]);
      expect(splits[1].toList(), <Object>[2.0, 3.0]);
      expect(splits[2].toList(), <Object>[4.0]);
    } finally {
      for (final value in splits) {
        value.close();
      }
      for (final value in broadcasted) {
        value.close();
      }
      splitsInput.close();
      b.close();
      a.close();
    }
  });

  test('supports segmentedMm and blockMaskedMm', () {
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
    final segments = MlxArray.fromInt32List([0, 2], shape: [1, 2]);

    final segmented = mx.segmentedMm(a, b, segments);
    final masked = mx.blockMaskedMm(a, b, blockSize: 32);
    final ref = a.matmul(b);

    try {
      expect(segmented.shape, <int>[1, 2, 2]);
      expect(segmented.toList(), <Object>[19.0, 22.0, 43.0, 50.0]);
      expect(masked.toList(), ref.toList());
    } finally {
      ref.close();
      masked.close();
      segmented.close();
      segments.close();
      b.close();
      a.close();
    }
  });
}
