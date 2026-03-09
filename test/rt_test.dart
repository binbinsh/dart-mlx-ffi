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
  test('supports runtime helpers and deterministic seeding', () {
    MlxRuntime.seed(123);
    final first = MlxRandom.uniform([2, 2]);
    MlxRuntime.seed(123);
    final second = MlxRandom.uniform([2, 2]);
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
    final sum = a + b;
    final product = a.matmul(b);

    try {
      MlxRuntime.evalAll([sum, product]);
      expect(first.toList(), second.toList());
      expect(sum.toList(), <Object>[6.0, 8.0, 10.0, 12.0]);
      expect(product.toList(), <Object>[19.0, 22.0, 43.0, 50.0]);

      expect(MlxMemory.activeBytes(), greaterThanOrEqualTo(0));
      expect(MlxMemory.cacheBytes(), greaterThanOrEqualTo(0));
      expect(MlxMemory.peakBytes(), greaterThanOrEqualTo(0));
      expect(MlxMemory.memoryLimitBytes(), greaterThanOrEqualTo(0));
      expect(MlxMetal.isAvailable(), anyOf(isTrue, isFalse));
      expect(MlxDevice.count(raw.mlx_device_type_.MLX_CPU), greaterThanOrEqualTo(1));
    } finally {
      product.close();
      sum.close();
      b.close();
      a.close();
      second.close();
      first.close();
    }
  });

  test('exposes memory, metal, and device runtime helpers', () {
    final device = MlxDevice.defaultDevice();
    final currentLimit = MlxMemory.memoryLimitBytes();

    try {
      expect(device.isAvailable, anyOf(isTrue, isFalse));
      expect(MlxDevice.count(raw.mlx_device_type_.MLX_CPU), greaterThanOrEqualTo(1));
      expect(mx.memory.activeBytes(), greaterThanOrEqualTo(0));
      expect(mx.memory.cacheBytes(), greaterThanOrEqualTo(0));
      expect(mx.memory.peakBytes(), greaterThanOrEqualTo(0));
      expect(mx.memory.memoryLimitBytes(), greaterThanOrEqualTo(0));
      expect(mx.memory.setMemoryLimitBytes(currentLimit), greaterThanOrEqualTo(0));
      expect(mx.memory.setCacheLimitBytes(mx.memory.cacheBytes()), greaterThanOrEqualTo(0));
      try {
        expect(mx.memory.setWiredLimitBytes(currentLimit), greaterThanOrEqualTo(0));
      } on MlxException {
        // Some Metal-enabled environments reject wired-limit changes.
      }
      mx.memory.clearCache();
      mx.memory.resetPeak();
      expect(mx.metal.isAvailable(), anyOf(isTrue, isFalse));
      final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_capture_');
      final capturePath = '${dir.path}/trace.gputrace';
      try {
        try {
          mx.metal.startCapture(capturePath);
        } on MlxException {
          // CPU-only or Metal-disabled environments are allowed here.
        }
        try {
          mx.metal.stopCapture();
        } on MlxException {
          // CPU-only or Metal-disabled environments are allowed here.
        }
      } finally {
        dir.deleteSync(recursive: true);
      }

      final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
      final b = a + a;
      try {
        mx.evalAll([b]);
        mx.asyncEvalAll([b]);
        expect(b.toList(), <Object>[2.0, 4.0, 6.0, 8.0]);
      } finally {
        b.close();
        a.close();
      }
    } finally {
      device.close();
    }
  });

  test('surfaces runtime helper failures as MlxException', () {
    final device = MlxDevice.defaultDevice();
    hooks.debugDeviceIsAvailableOverride = (_) => -1;
    hooks.debugDeviceCountOverride = (_) => -1;
    try {
      expect(() => device.isAvailable, throwsA(isA<MlxException>()));
      expect(
        () => MlxDevice.count(raw.mlx_device_type_.MLX_CPU),
        throwsA(isA<MlxException>()),
      );
    } finally {
      hooks.resetDebugHooks();
      device.close();
    }
  });

  test('supports python-like mx namespace entrypoint', () {
    final a = mx.arange(0, 4, 1).reshape([2, 2]);
    final b = mx.ones([2, 2]);
    final zeros = mx.zeros([2, 2]);
    final full = mx.full([2, 2], 3);
    final added = mx.add(a, b);
    final subtracted = mx.subtract(full, b);
    final multiplied = mx.multiply(a, b);
    final divided = mx.divide(full, b);
    final matmul = mx.matmul(a, b);
    final abs = mx.abs(MlxArray.fromFloat32List([-1, 2], shape: [2]));
    final exp = mx.exp(MlxArray.fromFloat32List([0], shape: [1]));
    final log = mx.log(MlxArray.fromFloat32List([1], shape: [1]));
    final sin = mx.sin(MlxArray.fromFloat32List([0], shape: [1]));
    final cos = mx.cos(MlxArray.fromFloat32List([0], shape: [1]));
    final reduced = mx.sum(added);
    final mean = mx.mean(full);
    final logSumExp = mx.logSumExp(MlxArray.fromFloat32List([0, 0], shape: [2]));
    final softmax = mx.softmax(MlxArray.fromFloat32List([0, 0], shape: [2]));
    final topK = mx.topK(MlxArray.fromFloat32List([3, 1, 2], shape: [3]), 2);
    final equal = mx.equal(a, a);
    final where = mx.where(
      MlxArray.fromBoolList([true, false, true, false], shape: [2, 2]),
      a,
      full,
    );
    final concatenated = mx.concatenate([a, b], axis: 0);
    final stacked = mx.stack([a, b], axis: 0);
    final broadcasted = mx.broadcastTo(MlxArray.fromFloat32List([1, 2], shape: [2]), [2, 2]);
    final expanded = mx.expandDims(MlxArray.fromFloat32List([1, 2], shape: [2]), 0);
    final squeezed = mx.squeeze(expanded);
    final clipped = mx.clip(MlxArray.fromFloat32List([-1, 2], shape: [2]), min: 0, max: 1);
    final minned = mx.minimum(a, full);
    final maxed = mx.maximum(a, full);
    final argmax = mx.argmax(full);
    final argmin = mx.argmin(full, axis: 1, keepDims: true);
    final sorted = mx.sort(MlxArray.fromFloat32List([3, 1, 2], shape: [3]));
    final argsorted = mx.argsort(MlxArray.fromFloat32List([3, 1, 2], shape: [3]));
    final random = mx.random.uniform([2, 2]);
    final randomNormal = mx.random.normal([2, 2]);
    final bernoulli = mx.random.bernoulli(MlxArray.zeros([3]));
    final categorical = mx.random.categorical(
      MlxArray.fromFloat32List([0, 50], shape: [2]),
    );
    final permutation = mx.random.permutation(
      MlxArray.fromInt32List([1, 2, 3], shape: [3]),
    );
    final permutationArange = mx.random.permutationArange(3);
    final key = mx.random.key(7);
    final split = mx.random.split(key);

    try {
      expect(zeros.toList(), <Object>[0.0, 0.0, 0.0, 0.0]);
      expect(full.toList(), <Object>[3.0, 3.0, 3.0, 3.0]);
      expect(added.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(subtracted.toList(), <Object>[2.0, 2.0, 2.0, 2.0]);
      expect(multiplied.toList(), <Object>[0.0, 1.0, 2.0, 3.0]);
      expect(divided.toList(), <Object>[3.0, 3.0, 3.0, 3.0]);
      expect(matmul.toList(), <Object>[1.0, 1.0, 5.0, 5.0]);
      expect(abs.toList(), <Object>[1.0, 2.0]);
      expect((exp.toList()[0] as double), closeTo(1.0, 1e-5));
      expect((log.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((sin.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((cos.toList()[0] as double), closeTo(1.0, 1e-5));
      expect(reduced.toList(), <Object>[10.0]);
      expect(mean.toList(), <Object>[3.0]);
      expect((logSumExp.toList()[0] as double), closeTo(0.693147, 1e-4));
      expect(softmax.toList(), <Object>[0.5, 0.5]);
      expect(topK.toList(), <Object>[2.0, 3.0]);
      expect(equal.toList(), <Object>[true, true, true, true]);
      expect(where.toList(), <Object>[0.0, 3.0, 2.0, 3.0]);
      expect(concatenated.shape, <int>[4, 2]);
      expect(stacked.shape, <int>[2, 2, 2]);
      expect(broadcasted.toList(), <Object>[1.0, 2.0, 1.0, 2.0]);
      expect(expanded.shape, <int>[1, 2]);
      expect(squeezed.shape, <int>[2]);
      expect(clipped.toList(), <Object>[0.0, 1.0]);
      expect(minned.toList(), <Object>[0.0, 1.0, 2.0, 3.0]);
      expect(maxed.toList(), <Object>[3.0, 3.0, 3.0, 3.0]);
      expect(argmax.toList(), <Object>[0]);
      expect(argmin.toList(), <Object>[0, 0]);
      expect(sorted.toList(), <Object>[1.0, 2.0, 3.0]);
      expect(argsorted.toList(), <Object>[1, 2, 0]);
      expect(random.shape, <int>[2, 2]);
      expect(randomNormal.shape, <int>[2, 2]);
      expect(bernoulli.toList(), <Object>[false, false, false]);
      expect(categorical.toList(), everyElement(1));
      expect(
        (List<int>.from(permutation.toList())..sort()),
        <int>[1, 2, 3],
      );
      expect(
        (List<int>.from(permutationArange.toList())..sort()),
        <int>[0, 1, 2],
      );
      expect(split.first.shape, key.shape);
      expect(split.second.shape, key.shape);
    } finally {
      split.second.close();
      split.first.close();
      key.close();
      permutationArange.close();
      permutation.close();
      categorical.close();
      bernoulli.close();
      randomNormal.close();
      random.close();
      argsorted.close();
      sorted.close();
      argmin.close();
      argmax.close();
      maxed.close();
      minned.close();
      clipped.close();
      squeezed.close();
      expanded.close();
      broadcasted.close();
      stacked.close();
      concatenated.close();
      where.close();
      equal.close();
      topK.close();
      softmax.close();
      logSumExp.close();
      mean.close();
      reduced.close();
      cos.close();
      sin.close();
      log.close();
      exp.close();
      abs.close();
      matmul.close();
      divided.close();
      multiplied.close();
      subtracted.close();
      added.close();
      full.close();
      zeros.close();
      b.close();
      a.close();
    }
  });

}
