import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/raw.dart' as raw;

import '../private_ane/models/qwen3_5_0_8b/dart/qwen3_5.dart';

void main(List<String> args) {
  _trace('start');
  final options = _parseArgs(args);
  final snapshotDir = options['snapshot-dir'];
  final artifactsDir = options['artifacts-dir'];
  final tokenIdsFile = options['token-ids-file'];
  if (snapshotDir == null || artifactsDir == null || tokenIdsFile == null) {
    stderr.writeln(
      'Usage: dart run tool/qwen35_private_ane_bench.dart '
      '--snapshot-dir <dir> --artifacts-dir <dir> --token-ids-file <json> '
      '[--baseline-device cpu|gpu] [--ane-device cpu|gpu] '
      '[--warmup N] [--iters N] [--json]',
    );
    exitCode = 64;
    return;
  }

  final warmup = int.tryParse(options['warmup'] ?? '') ?? 1;
  final iters = int.tryParse(options['iters'] ?? '') ?? 3;
  final emitJson = options.containsKey('json');

  final tokenPayload = Map<String, Object?>.from(
    jsonDecode(File(tokenIdsFile).readAsStringSync()) as Map,
  );
  final tokenIds = List<int>.from(
    (tokenPayload['token_ids'] as List).cast<num>(),
  );
  final prompt = tokenPayload['prompt'] as String?;
  final originalDevice = MlxDevice.defaultDevice();

  final baselineLoad = Stopwatch()..start();
  final baselineDevice = _parseDevice(options['baseline-device']);
  if (baselineDevice != null) {
    MlxDevice.setDefault(baselineDevice);
  }
  _trace('baseline load start');
  final baselineRunner = Qwen3_5Runner.load(snapshotDir);
  baselineLoad.stop();
  _trace('baseline load done');
  final baselineValues = _forward(baselineRunner, tokenIds);
  _trace('baseline forward done');
  final baselinePerIterMs = _benchmark(
    baselineRunner,
    tokenIds,
    warmup: warmup,
    iters: iters,
  );
  baselineRunner.close();
  _trace('baseline close done');

  final aneLoad = Stopwatch()..start();
  final aneDevice = _parseDevice(options['ane-device']);
  if (aneDevice != null) {
    MlxDevice.setDefault(aneDevice);
  }
  _trace('ane load start');
  final aneRunner = Qwen3_5Runner.load(
    snapshotDir,
    privateAneArtifactsDir: artifactsDir,
  );
  aneLoad.stop();
  _trace('ane load done');
  final aneValues = _forward(aneRunner, tokenIds);
  _trace('ane forward done');
  final anePerIterMs = _benchmark(
    aneRunner,
    tokenIds,
    warmup: warmup,
    iters: iters,
  );
  aneRunner.close();
  _trace('ane close done');

  final diffs = <double>[];
  for (var index = 0; index < baselineValues.length; index++) {
    diffs.add((aneValues[index] - baselineValues[index]).abs());
  }

  final report = <String, Object?>{
    'runtime': 'qwen35_private_ane_hybrid',
    'snapshot_dir': snapshotDir,
    'artifacts_dir': artifactsDir,
    'baseline_device': options['baseline-device'],
    'ane_device': options['ane-device'],
    'prompt': prompt,
    'token_ids': tokenIds,
    'token_count': tokenIds.length,
    'warmup': warmup,
    'iters': iters,
    'baseline_load_ms': baselineLoad.elapsedMicroseconds / 1000.0,
    'ane_load_ms': aneLoad.elapsedMicroseconds / 1000.0,
    'baseline_per_iter_ms': baselinePerIterMs,
    'ane_per_iter_ms': anePerIterMs,
    'ane_speedup_vs_baseline': anePerIterMs == 0
        ? null
        : baselinePerIterMs / anePerIterMs,
    'baseline_values': baselineValues,
    'ane_values': aneValues,
    'max_abs_diff': diffs.reduce(_max),
    'mean_abs_diff': diffs.reduce((a, b) => a + b) / diffs.length,
    'argmax_match': _argmax(baselineValues) == _argmax(aneValues),
    'baseline_argmax': _argmax(baselineValues),
    'ane_argmax': _argmax(aneValues),
  };

  if (emitJson) {
    _trace('emit json');
    stdout.writeln(jsonEncode(report));
    if (baselineDevice != null || aneDevice != null) {
      MlxDevice.setDefault(originalDevice);
    }
    originalDevice.close();
    baselineDevice?.close();
    aneDevice?.close();
    exit(0);
  }

  _trace('emit pretty json');
  stdout.writeln(const JsonEncoder.withIndent('  ').convert(report));
  if (baselineDevice != null || aneDevice != null) {
    MlxDevice.setDefault(originalDevice);
  }
  originalDevice.close();
  baselineDevice?.close();
  aneDevice?.close();
  exit(0);
}

Map<String, String?> _parseArgs(List<String> args) {
  final out = <String, String?>{};
  for (var index = 0; index < args.length; index++) {
    final arg = args[index];
    if (!arg.startsWith('--')) {
      continue;
    }
    final key = arg.substring(2);
    if (index + 1 < args.length && !args[index + 1].startsWith('--')) {
      out[key] = args[index + 1];
      index++;
    } else {
      out[key] = null;
    }
  }
  return out;
}

MlxDevice? _parseDevice(String? value) {
  switch (value?.toLowerCase()) {
    case null:
    case '':
      return null;
    case 'cpu':
      return MlxDevice.type(raw.mlx_device_type_.MLX_CPU);
    case 'gpu':
      return MlxDevice.type(raw.mlx_device_type_.MLX_GPU);
    default:
      throw ArgumentError.value(value, 'device', 'Expected cpu or gpu.');
  }
}

List<double> _forward(Qwen3_5Runner runner, List<int> tokenIds) {
  final out = runner.run(tokenIds).astype(MlxDType.MLX_FLOAT32);
  try {
    final raw = out.toList();
    return List<double>.generate(
      raw.length,
      (index) => (raw[index] as num).toDouble(),
    );
  } finally {
    out.close();
  }
}

double _benchmark(
  Qwen3_5Runner runner,
  List<int> tokenIds, {
  required int warmup,
  required int iters,
}) {
  for (var index = 0; index < warmup; index++) {
    _forward(runner, tokenIds);
  }
  final watch = Stopwatch()..start();
  for (var index = 0; index < iters; index++) {
    _forward(runner, tokenIds);
  }
  watch.stop();
  return watch.elapsedMicroseconds / 1000.0 / iters;
}

int _argmax(List<double> values) {
  var bestIndex = 0;
  var bestValue = values.first;
  for (var index = 1; index < values.length; index++) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }
  return bestIndex;
}

double _max(double a, double b) => a > b ? a : b;

void _trace(String message) {
  if (Platform.environment['QWEN35_PRIVATE_BENCH_TRACE'] != '1') {
    return;
  }
  stderr.writeln('qwen35_private_ane_bench: $message');
}
