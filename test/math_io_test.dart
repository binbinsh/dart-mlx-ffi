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
  test('supports linalg operations', () {
    final a = MlxArray.fromFloat32List([4, 7, 2, 6], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([1, 0], shape: [2, 1]);
    final inv = mx.linalg.inv(a);
    final solved = mx.linalg.solve(a, b);
    final qr = mx.linalg.qr(a);
    final eig = mx.linalg.eig(a);

    try {
      final invValues = inv.toList();
      expect(inv.shape, <int>[2, 2]);
      expect(invValues[0] as double, closeTo(0.6, 1e-4));
      expect(invValues[1] as double, closeTo(-0.7, 1e-4));
      expect(invValues[2] as double, closeTo(-0.2, 1e-4));
      expect(invValues[3] as double, closeTo(0.4, 1e-4));

      final solvedValues = solved.toList();
      expect(solved.shape, <int>[2, 1]);
      expect(solvedValues[0] as double, closeTo(0.6, 1e-4));
      expect(solvedValues[1] as double, closeTo(-0.2, 1e-4));

      expect(qr.q.shape, <int>[2, 2]);
      expect(qr.r.shape, <int>[2, 2]);
      expect(eig.values.shape, <int>[2]);
      expect(eig.vectors.shape, <int>[2, 2]);
    } finally {
      eig.vectors.close();
      eig.values.close();
      qr.r.close();
      qr.q.close();
      solved.close();
      inv.close();
      b.close();
      a.close();
    }
  });

  test('supports fft operations', () {
    final signal = MlxArray.fromFloat32List([0, 1, 0, -1], shape: [4]);
    final rfft = mx.fft.rfft(signal);
    final irfft = mx.fft.irfft(rfft, n: 4);
    final fft = mx.fft.fft(signal);
    final ifft = mx.fft.ifft(fft, n: 4);

    try {
      expect(rfft.shape, <int>[3]);
      expect(irfft.shape, <int>[4]);
      final irfftValues = irfft.toList();
      expect(irfftValues[0] as double, closeTo(0.0, 1e-4));
      expect(irfftValues[1] as double, closeTo(1.0, 1e-4));
      expect(irfftValues[2] as double, closeTo(0.0, 1e-4));
      expect(irfftValues[3] as double, closeTo(-1.0, 1e-4));

      expect(fft.shape, <int>[4]);
      expect(ifft.shape, <int>[4]);
      expect(ifft.dtype, MlxDType.MLX_COMPLEX64);
      expect(() => fft.toList(), throwsUnsupportedError);
    } finally {
      ifft.close();
      fft.close();
      irfft.close();
      rfft.close();
      signal.close();
    }
  });

  test('supports io save/load roundtrip', () {
    final array = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_io_');
    final file = '${dir.path}/array.npy';

    try {
      mx.io.save(file, array);
      final loaded = mx.io.load(file);
      try {
        expect(loaded.shape, <int>[2, 2]);
        expect(loaded.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      } finally {
        loaded.close();
      }
    } finally {
      array.close();
      if (Directory(dir.path).existsSync()) {
        dir.deleteSync(recursive: true);
      }
    }
  });

  test('supports fast helpers', () {
    final input = MlxArray.fromFloat32List([1, 1, 1, 1], shape: [1, 4]);
    final weight = MlxArray.ones([4]);
    final bias = MlxArray.zeros([4]);
    final queries = MlxArray.fromFloat32List(
      List<double>.generate(256, (index) => ((index * 3) % 17 - 8) / 8),
      shape: [1, 1, 4, 64],
    );
    final keys = MlxArray.fromFloat32List(
      List<double>.generate(256, (index) => ((index * 5) % 19 - 9) / 8),
      shape: [1, 1, 4, 64],
    );
    final values = MlxArray.fromFloat32List(
      List<double>.generate(256, (index) => ((index * 7) % 23 - 11) / 8),
      shape: [1, 1, 4, 64],
    );

    final ln = mx.fast.layerNorm(input, weight: weight, bias: bias);
    final rms = mx.fast.rmsNorm(input, weight: weight);
    final ropeInput = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 1, 1, 4]);
    final ropeOffset = MlxArray.full([], 0);
    MlxArray? sdpa;
    MlxArray? rope;
    MlxArray? ropeDynamic;

    try {
      try {
        rope = mx.fast.rope(ropeInput, dims: 2);
        expect(rope.shape, <int>[1, 1, 1, 4]);
      } on MlxException {
        rope = null;
      }
      try {
        ropeDynamic = mx.fast.ropeDynamic(
          ropeInput,
          dims: 2,
          offset: ropeOffset,
        );
        expect(ropeDynamic.shape, <int>[1, 1, 1, 4]);
      } on MlxException {
        ropeDynamic = null;
      }
      if (MlxMetal.isAvailable()) {
        sdpa = mx.fast.scaledDotProductAttention(queries, keys, values);
        expect(sdpa.shape, <int>[1, 1, 4, 64]);
      } else {
        try {
          sdpa = mx.fast.scaledDotProductAttention(queries, keys, values);
          expect(sdpa.shape, <int>[1, 1, 4, 64]);
        } on MlxException {
          sdpa = null;
        }
      }
      expect(ln.shape, <int>[1, 4]);
      expect(rms.shape, <int>[1, 4]);
      expect(ln.toList(), <Object>[0.0, 0.0, 0.0, 0.0]);
    } finally {
      sdpa?.close();
      ropeDynamic?.close();
      rope?.close();
      ropeOffset.close();
      ropeInput.close();
      rms.close();
      ln.close();
      values.close();
      keys.close();
      queries.close();
      bias.close();
      weight.close();
      input.close();
    }
  });

  test('supports fast kernel config wrappers', () {
    final metalConfig = mx.fast.metalConfig();
    final cudaConfig = mx.fast.cudaConfig();
    final input = MlxArray.fromFloat32List([1, 2], shape: [2]);

    try {
      metalConfig.addOutputArg([2], MlxDType.MLX_FLOAT32);
      metalConfig.setGrid(2, 1, 1);
      metalConfig.setThreadGroup(2, 1, 1);
      metalConfig.setInitValue(0);
      metalConfig.setVerbose(false);
      metalConfig.addTemplateDtype('T', MlxDType.MLX_FLOAT32);
      metalConfig.addTemplateInt('N', 2);
      metalConfig.addTemplateBool('Flag', true);

      cudaConfig.addOutputArg([2], MlxDType.MLX_FLOAT32);
      cudaConfig.setGrid(2, 1, 1);
      cudaConfig.setThreadGroup(2, 1, 1);
      cudaConfig.setInitValue(0);
      cudaConfig.setVerbose(false);
      cudaConfig.addTemplateDtype('T', MlxDType.MLX_FLOAT32);
      cudaConfig.addTemplateInt('N', 2);
      cudaConfig.addTemplateBool('Flag', true);

      MlxMetalKernel? metalKernel;
      MlxCudaKernel? cudaKernel;
      try {
        try {
          metalKernel = mx.fast.metalKernel(
            'copy_kernel',
            ['inp'],
            ['out'],
            'uint elem = thread_position_in_grid.x; out[elem] = inp[elem];',
          );
          final outputs = metalKernel.apply([input], metalConfig);
          for (final output in outputs) {
            output.close();
          }
        } on MlxException {
          metalKernel = null;
        }
        try {
          cudaKernel = mx.fast.cudaKernel(
            'copy_kernel',
            ['inp'],
            ['out'],
            'extern "C" __global__ void copy_kernel() {}',
          );
          final outputs = cudaKernel.apply([input], cudaConfig);
          for (final output in outputs) {
            output.close();
          }
        } on MlxException {
          cudaKernel = null;
        }
      } finally {
        cudaKernel?.close();
        metalKernel?.close();
      }
    } finally {
      input.close();
      cudaConfig.close();
      metalConfig.close();
    }
  });

}
