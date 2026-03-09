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
  test('supports conv1d, conv2d, conv3d, and convGeneral', () {
    final input1d = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 4, 1]);
    final weight1d = MlxArray.fromFloat32List([1, 1], shape: [1, 2, 1]);

    final input2d = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 2, 2, 1]);
    final weight2d = MlxArray.fromFloat32List([1, 1, 1, 1], shape: [1, 2, 2, 1]);

    final input3d = MlxArray.fromFloat32List(
      [1, 2, 3, 4, 5, 6, 7, 8],
      shape: [1, 2, 2, 2, 1],
    );
    final weight3d = MlxArray.fromFloat32List(
      List<double>.filled(8, 1),
      shape: [1, 2, 2, 2, 1],
    );

    final out1d = mx.conv1d(input1d, weight1d);
    final out2d = mx.conv2d(input2d, weight2d);
    final out3d = mx.conv3d(input3d, weight3d);
    final outGeneral = mx.convGeneral(
      input2d,
      weight2d,
      stride: [1, 1],
      padding: [0, 0],
      kernelDilation: [1, 1],
      inputDilation: [1, 1],
    );
    final tInput1d = MlxArray.fromFloat32List([1, 2, 3], shape: [1, 3, 1]);
    final tWeight1d = MlxArray.fromFloat32List([1, 1], shape: [1, 2, 1]);
    final tInput2d = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 2, 2, 1]);
    final tWeight2d = MlxArray.fromFloat32List([1, 1, 1, 1], shape: [1, 2, 2, 1]);
    final tInput3d = MlxArray.fromFloat32List(
      [1, 2, 3, 4, 5, 6, 7, 8],
      shape: [1, 2, 2, 2, 1],
    );
    final tWeight3d = MlxArray.fromFloat32List(
      List<double>.filled(8, 1),
      shape: [1, 2, 2, 2, 1],
    );
    final tOut1d = mx.convTranspose1d(tInput1d, tWeight1d);
    final tOut2d = mx.convTranspose2d(tInput2d, tWeight2d);
    final tOut3d = mx.convTranspose3d(tInput3d, tWeight3d);
    final tOut3dSum = tOut3d.sum();

    try {
      expect(out1d.shape, <int>[1, 3, 1]);
      expect(out1d.toList(), <Object>[3.0, 5.0, 7.0]);
      expect(out2d.shape, <int>[1, 1, 1, 1]);
      expect(out2d.toList(), <Object>[10.0]);
      expect(out3d.shape, <int>[1, 1, 1, 1, 1]);
      expect(out3d.toList(), <Object>[36.0]);
      expect(outGeneral.toList(), out2d.toList());
      expect(tOut1d.shape, <int>[1, 4, 1]);
      expect(tOut1d.toList(), <Object>[1.0, 3.0, 5.0, 3.0]);
      expect(tOut2d.shape, <int>[1, 3, 3, 1]);
      expect(tOut2d.toList(), <Object>[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
      expect(tOut3d.shape, <int>[1, 3, 3, 3, 1]);
      expect(tOut3dSum.toList(), <Object>[288.0]);
    } finally {
      tOut3dSum.close();
      tOut3d.close();
      tOut2d.close();
      tOut1d.close();
      tWeight3d.close();
      tInput3d.close();
      tWeight2d.close();
      tInput2d.close();
      tWeight1d.close();
      tInput1d.close();
      outGeneral.close();
      out3d.close();
      out2d.close();
      out1d.close();
      weight3d.close();
      input3d.close();
      weight2d.close();
      input2d.close();
      weight1d.close();
      input1d.close();
    }
  });
}
