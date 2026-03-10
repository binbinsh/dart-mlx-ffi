import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main(List<String> args) {
  final warmup = _readIntArg(args, '--warmup', fallback: 20);
  final iters = _readIntArg(args, '--iters', fallback: 100);
  final asJson = args.contains('--json');

  final device = MlxDevice.defaultDevice();
  final stream = MlxStream.defaultFor(device);

  try {
    final report = <String, Object?>{
      'runtime': 'dart_mlx_ffi',
      'mlx_version': MlxVersion.current(),
      'device': device.toString(),
      'metal': MlxMetal.isAvailable(),
      'warmup': warmup,
      'iters': iters,
      'models': <Map<String, Object?>>[
        _benchMlp(warmup: warmup, iters: iters, stream: stream),
        _benchConv(warmup: warmup, iters: iters, stream: stream),
        _benchAttention(warmup: warmup, iters: iters, stream: stream),
      ],
    };

    if (asJson) {
      stdout.writeln(jsonEncode(report));
      return;
    }

    _printHuman(report);
  } finally {
    stream.close();
    device.close();
  }
}

Map<String, Object?> _benchMlp({
  required int warmup,
  required int iters,
  required MlxStream stream,
}) {
  const inputShape = [32, 64];
  const hiddenShape = [64, 96];
  const outputShape = [96, 6];

  final inputValues = _values(_numel(inputShape), seed: 3, divisor: 64);
  final w1Values = _values(_numel(hiddenShape), seed: 5, divisor: 128);
  final b1Values = _values(96, seed: 7, divisor: 256);
  final w2Values = _values(_numel(outputShape), seed: 11, divisor: 128);
  final b2Values = _values(6, seed: 13, divisor: 256);

  final input = MlxArray.fromFloat32List(inputValues, shape: inputShape);
  final w1 = MlxArray.fromFloat32List(w1Values, shape: hiddenShape);
  final b1 = MlxArray.fromFloat32List(b1Values, shape: [1, 96]);
  final w2 = MlxArray.fromFloat32List(w2Values, shape: outputShape);
  final b2 = MlxArray.fromFloat32List(b2Values, shape: [1, 6]);

  try {
    MlxArray forward() {
      final hiddenLinear = mx.add(mx.matmul(input, w1), b1);
      final hidden = hiddenLinear.sigmoid();
      final logits = mx.add(mx.matmul(hidden, w2), b2);
      final output = mx.softmax(logits, axis: 1);
      MlxRuntime.evalAll([output]);
      stream.synchronize();
      logits.close();
      hidden.close();
      hiddenLinear.close();
      return output;
    }

    return _measure(
      name: 'tiny_mlp',
      description: 'matmul + sigmoid + softmax',
      warmup: warmup,
      iters: iters,
      inputShapes: const {
        'input': inputShape,
        'w1': hiddenShape,
        'b1': [1, 96],
        'w2': outputShape,
        'b2': [1, 6],
      },
      inputPreview: inputValues,
      forward: forward,
    );
  } finally {
    b2.close();
    w2.close();
    b1.close();
    w1.close();
    input.close();
  }
}

Map<String, Object?> _benchConv({
  required int warmup,
  required int iters,
  required MlxStream stream,
}) {
  const inputShape = [8, 32, 32, 3];
  const kernelShape = [8, 3, 3, 3];
  const biasShape = [1, 1, 1, 8];
  const headShape = [8, 5];

  final inputValues = _values(_numel(inputShape), seed: 17, divisor: 128);
  final kernelValues = _values(_numel(kernelShape), seed: 19, divisor: 256);
  final biasValues = _values(_numel(biasShape), seed: 23, divisor: 512);
  final headValues = _values(_numel(headShape), seed: 29, divisor: 256);
  final headBiasValues = _values(5, seed: 31, divisor: 512);

  final input = MlxArray.fromFloat32List(inputValues, shape: inputShape);
  final kernel = MlxArray.fromFloat32List(kernelValues, shape: kernelShape);
  final bias = MlxArray.fromFloat32List(biasValues, shape: biasShape);
  final head = MlxArray.fromFloat32List(headValues, shape: headShape);
  final headBias = MlxArray.fromFloat32List(headBiasValues, shape: [1, 5]);

  try {
    MlxArray forward() {
      final conv = mx.conv2d(input, kernel, padding: const [1, 1]);
      final biased = mx.add(conv, bias);
      final activated = biased.sigmoid();
      final pooledH = mx.mean(activated, axis: 1);
      final pooledW = mx.mean(pooledH, axis: 1);
      final logits = mx.add(mx.matmul(pooledW, head), headBias);
      final output = mx.softmax(logits, axis: 1);
      MlxRuntime.evalAll([output]);
      stream.synchronize();
      logits.close();
      pooledW.close();
      pooledH.close();
      activated.close();
      biased.close();
      conv.close();
      return output;
    }

    return _measure(
      name: 'tiny_conv',
      description: 'conv2d + sigmoid + global average pooling + softmax',
      warmup: warmup,
      iters: iters,
      inputShapes: const {
        'input': inputShape,
        'kernel': kernelShape,
        'bias': biasShape,
        'head': headShape,
        'head_bias': [1, 5],
      },
      inputPreview: inputValues,
      forward: forward,
    );
  } finally {
    headBias.close();
    head.close();
    bias.close();
    kernel.close();
    input.close();
  }
}

Map<String, Object?> _benchAttention({
  required int warmup,
  required int iters,
  required MlxStream stream,
}) {
  const qShape = [2, 4, 16, 64];
  const normShape = [64];
  const headShape = [64, 4];

  final qValues = _values(_numel(qShape), seed: 37, divisor: 128);
  final kValues = _values(_numel(qShape), seed: 41, divisor: 128);
  final vValues = _values(_numel(qShape), seed: 43, divisor: 128);
  final normWeightValues = _values(_numel(normShape), seed: 47, divisor: 256);
  final normBiasValues = _values(_numel(normShape), seed: 53, divisor: 512);
  final headValues = _values(_numel(headShape), seed: 59, divisor: 256);
  final headBiasValues = _values(4, seed: 61, divisor: 512);

  final q = MlxArray.fromFloat32List(qValues, shape: qShape);
  final k = MlxArray.fromFloat32List(kValues, shape: qShape);
  final v = MlxArray.fromFloat32List(vValues, shape: qShape);
  final normWeight = MlxArray.fromFloat32List(normWeightValues, shape: normShape);
  final normBias = MlxArray.fromFloat32List(normBiasValues, shape: normShape);
  final head = MlxArray.fromFloat32List(headValues, shape: headShape);
  final headBias = MlxArray.fromFloat32List(headBiasValues, shape: [1, 4]);

  try {
    MlxArray forward() {
      final attention = mx.fast.scaledDotProductAttention(
        q,
        k,
        v,
        scale: 0.25,
      );
      final normalized = mx.fast.layerNorm(
        attention,
        weight: normWeight,
        bias: normBias,
        eps: 1e-5,
      );
      final pooledSeq = mx.mean(normalized, axis: 2);
      final pooledHeads = mx.mean(pooledSeq, axis: 1);
      final logits = mx.add(mx.matmul(pooledHeads, head), headBias);
      final output = mx.softmax(logits, axis: 1);
      MlxRuntime.evalAll([output]);
      stream.synchronize();
      logits.close();
      pooledHeads.close();
      pooledSeq.close();
      normalized.close();
      attention.close();
      return output;
    }

    return _measure(
      name: 'tiny_attention',
      description: 'scaled dot product attention + layer norm + softmax',
      warmup: warmup,
      iters: iters,
      inputShapes: const {
        'q': qShape,
        'k': qShape,
        'v': qShape,
        'norm_weight': normShape,
        'norm_bias': normShape,
        'head': headShape,
        'head_bias': [1, 4],
      },
      inputPreview: qValues,
      forward: forward,
    );
  } finally {
    headBias.close();
    head.close();
    normBias.close();
    normWeight.close();
    v.close();
    k.close();
    q.close();
  }
}

Map<String, Object?> _measure({
  required String name,
  required String description,
  required int warmup,
  required int iters,
  required Map<String, List<int>> inputShapes,
  required List<double> inputPreview,
  required MlxArray Function() forward,
}) {
  for (var i = 0; i < warmup; i++) {
    final output = forward();
    output.close();
  }

  MlxMemory.resetPeak();
  final beforePeak = MlxMemory.peakBytes();
  final stopwatch = Stopwatch()..start();
  MlxArray? last;

  try {
    for (var i = 0; i < iters; i++) {
      last?.close();
      last = forward();
    }
    stopwatch.stop();

    final outputFlat = List<double>.from(last!.toList().cast<double>());
    final afterPeak = MlxMemory.peakBytes();
    final totalMs = stopwatch.elapsedMicroseconds / 1000.0;

    return <String, Object?>{
      'name': name,
      'description': description,
      'input_shapes': inputShapes,
      'input_preview': _preview(inputPreview),
      'output_shape': last.shape,
      'output_preview': _preview(outputFlat),
      'output_flat': outputFlat,
      'output_sum': outputFlat.fold<double>(0, (sum, value) => sum + value),
      'total_ms': totalMs,
      'per_iter_ms': totalMs / iters,
      'peak_bytes_delta': afterPeak - beforePeak,
    };
  } finally {
    last?.close();
  }
}

List<double> _values(int count, {required int seed, required int divisor}) {
  return List<double>.generate(count, (index) {
    final numerator = ((index * (seed * 2 + 1) + seed * 7 + 13) % 257) - 128;
    return numerator / divisor;
  });
}

List<double> _preview(List<double> values, {int limit = 8}) {
  final end = math.min(limit, values.length);
  return values.sublist(0, end);
}

int _numel(List<int> shape) => shape.fold<int>(1, (value, dim) => value * dim);

int _readIntArg(List<String> args, String name, {required int fallback}) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return int.tryParse(arg.substring(prefix.length)) ?? fallback;
    }
  }
  return fallback;
}

void _printHuman(Map<String, Object?> report) {
  stdout.writeln('dart_mlx_ffi model benchmark');
  stdout.writeln('mlx_version: ${report['mlx_version']}');
  stdout.writeln('device: ${report['device']}');
  stdout.writeln('metal: ${report['metal']}');
  stdout.writeln('warmup: ${report['warmup']}');
  stdout.writeln('iters: ${report['iters']}');
  for (final model in (report['models'] as List<Object?>).cast<Map<String, Object?>>()) {
    stdout.writeln('');
    stdout.writeln('${model['name']}: ${model['description']}');
    stdout.writeln('  output_shape: ${model['output_shape']}');
    stdout.writeln('  output_preview: ${model['output_preview']}');
    stdout.writeln(
      '  per_iter_ms: ${(model['per_iter_ms'] as double).toStringAsFixed(4)}',
    );
    stdout.writeln('  peak_bytes_delta: ${model['peak_bytes_delta']}');
  }
}
