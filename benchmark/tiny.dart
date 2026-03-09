import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main(List<String> args) {
  final warmup = _readIntArg(args, '--warmup', fallback: 10);
  final iters = _readIntArg(args, '--iters', fallback: 50);

  final device = MlxDevice.defaultDevice();
  final stream = MlxStream.defaultFor(device);

  try {
    final result = _runTinyBench(warmup: warmup, iters: iters);
    stdout.writeln('dart_mlx_ffi tiny benchmark');
    stdout.writeln('version: ${MlxVersion.current()}');
    stdout.writeln('device: ${device.toString()}');
    stdout.writeln('metal: ${MlxMetal.isAvailable()}');
    stdout.writeln('warmup: $warmup');
    stdout.writeln('iters: $iters');
    stdout.writeln('total_ms: ${result.totalMs.toStringAsFixed(2)}');
    stdout.writeln('per_iter_ms: ${result.perIterMs.toStringAsFixed(4)}');
    stdout.writeln('peak_bytes_delta: ${result.peakBytesDelta}');
    stdout.writeln('last_topk: ${result.topkText}');
    stream.synchronize();
  } finally {
    stream.close();
    device.close();
  }
}

({double totalMs, double perIterMs, int peakBytesDelta, String topkText})
_runTinyBench({required int warmup, required int iters}) {
  final features = MlxArray.fromFloat32List(
    [0.2, -0.1, 0.4, 0.8, 0.3, -0.5, 0.7, 0.1, -0.2, 0.9, -0.4, 0.6],
    shape: [4, 3],
  );
  final weights = MlxArray.fromFloat32List(
    [0.5, -0.2, 0.3, -0.4, 0.8, 0.1, 0.2, 0.2, 0.6],
    shape: [3, 3],
  );
  final bias = MlxArray.fromFloat32List([0.1, -0.1, 0.05], shape: [1, 3]);

  try {
    for (var i = 0; i < warmup; i++) {
      final topk = _runTinyStep(features, weights, bias);
      topk.close();
    }

    MlxMemory.resetPeak();
    final beforePeak = MlxMemory.peakBytes();
    final stopwatch = Stopwatch()..start();
    MlxArray? lastTopk;
    try {
      for (var i = 0; i < iters; i++) {
        lastTopk?.close();
        lastTopk = _runTinyStep(features, weights, bias);
      }
      stopwatch.stop();
      final afterPeak = MlxMemory.peakBytes();
      final totalMs = stopwatch.elapsedMicroseconds / 1000.0;
      return (
        totalMs: totalMs,
        perIterMs: totalMs / iters,
        peakBytesDelta: afterPeak - beforePeak,
        topkText: lastTopk.toString(),
      );
    } finally {
      lastTopk?.close();
    }
  } finally {
    bias.close();
    weights.close();
    features.close();
  }
}

MlxArray _runTinyStep(MlxArray features, MlxArray weights, MlxArray bias) {
  final logits = mx.add(mx.matmul(features, weights), bias);
  final probs = mx.softmax(logits, axis: 1);
  final topk = mx.topK(probs, 2, axis: 1);
  MlxRuntime.evalAll([topk]);
  probs.close();
  logits.close();
  return topk;
}

int _readIntArg(List<String> args, String name, {required int fallback}) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return int.tryParse(arg.substring(prefix.length)) ?? fallback;
    }
  }
  return fallback;
}
