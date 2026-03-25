import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import '../private_ane/models/qwen3_5_0_8b/dart/qwen3_5.dart';

void main(List<String> args) {
  final options = _parseArgs(args);
  final snapshotDir = options['snapshot-dir'];
  final artifactsDir = options['artifacts-dir'];
  final tokenIdsFile = options['token-ids-file'];
  if (snapshotDir == null || artifactsDir == null || tokenIdsFile == null) {
    stderr.writeln(
      'Usage: dart run tool/qwen35_private_ane_layer_bench.dart '
      '--snapshot-dir <dir> --artifacts-dir <dir> --token-ids-file <json> '
      '[--warmup N] [--iters N] [--json]',
    );
    exitCode = 64;
    return;
  }

  final warmup = int.tryParse(options['warmup'] ?? '') ?? 1;
  final iters = int.tryParse(options['iters'] ?? '') ?? 5;
  final emitJson = options.containsKey('json');

  final tokenPayload = Map<String, Object?>.from(
    jsonDecode(File(tokenIdsFile).readAsStringSync()) as Map,
  );
  final tokenIds = List<int>.from(
    (tokenPayload['token_ids'] as List).cast<num>(),
  );
  final prompt = tokenPayload['prompt'] as String?;

  final metadata = Map<String, Object?>.from(
    jsonDecode(File('$artifactsDir/metadata.json').readAsStringSync()) as Map,
  );
  final lane = (metadata['lane'] as num).toInt();
  final dim = (metadata['dim'] as num).toInt();
  final layerSpecs = <int, Map<String, Object?>>{
    for (final spec in List<Map<String, Object?>>.from(
      (metadata['layers'] as List).cast<Map>(),
    ))
      (spec['layer'] as num).toInt(): spec,
  };

  final runner = Qwen3_5Runner.load(snapshotDir);
  final captures = runner.captureDenseMlp(tokenIds);
  final baselineForwardMs = _benchmarkForward(
    runner,
    tokenIds,
    warmup: warmup,
    iters: iters,
  );

  final rows = <Map<String, Object?>>[];
  final selectedDense = <int>[];
  final selectedArrayInput = <int>[];
  var baselineSelectedDenseMs = 0.0;
  var aneSelectedDenseMs = 0.0;
  var baselineSelectedArrayInputMs = 0.0;
  var aneSelectedArrayInputMs = 0.0;

  for (final capture in captures) {
    final spec = layerSpecs[capture.layerIndex];
    if (spec == null) {
      continue;
    }
    final baselineMs = _benchmarkBaselineLayer(
      runner,
      capture,
      warmup: warmup,
      iters: iters,
    );
    final aneMs = _benchmarkAneLayer(
      spec,
      capture,
      dim: dim,
      lane: lane,
      warmup: warmup,
      iters: iters,
    );
    final aneArrayInputMs = _benchmarkAneArrayInputLayer(
      spec,
      capture,
      dim: dim,
      lane: lane,
      warmup: warmup,
      iters: iters,
    );
    final speedup = aneMs == 0 ? null : baselineMs / aneMs;
    final arrayInputSpeedup = aneArrayInputMs == 0
        ? null
        : baselineMs / aneArrayInputMs;
    final chooseDense = aneMs < baselineMs;
    final chooseArrayInput = aneArrayInputMs < baselineMs;
    if (chooseDense) {
      selectedDense.add(capture.layerIndex);
      baselineSelectedDenseMs += baselineMs;
      aneSelectedDenseMs += aneMs;
    }
    if (chooseArrayInput) {
      selectedArrayInput.add(capture.layerIndex);
      baselineSelectedArrayInputMs += baselineMs;
      aneSelectedArrayInputMs += aneArrayInputMs;
    }
    rows.add({
      'layer': capture.layerIndex,
      'seq_len': capture.seqLen,
      'baseline_dense_ms': baselineMs,
      'ane_dense_ms': aneMs,
      'ane_array_input_ms': aneArrayInputMs,
      'ane_speedup_vs_baseline_dense': speedup,
      'ane_speedup_vs_baseline_array_input': arrayInputSpeedup,
      'select_for_ane_dense': chooseDense,
      'select_for_ane_array_input': chooseArrayInput,
    });
  }

  runner.close();

  final report = <String, Object?>{
    'runtime': 'qwen35_private_ane_layer_bench',
    'snapshot_dir': snapshotDir,
    'artifacts_dir': artifactsDir,
    'prompt': prompt,
    'token_ids': tokenIds,
    'token_count': tokenIds.length,
    'warmup': warmup,
    'iters': iters,
    'baseline_forward_per_iter_ms': baselineForwardMs,
    'selected_dense_layers': selectedDense,
    'selected_dense_layers_csv': selectedDense.join(','),
    'baseline_selected_dense_ms': baselineSelectedDenseMs,
    'ane_selected_dense_ms': aneSelectedDenseMs,
    'estimated_dense_hybrid_forward_per_iter_ms':
        baselineForwardMs - baselineSelectedDenseMs + aneSelectedDenseMs,
    'estimated_dense_hybrid_speedup_vs_baseline':
        (baselineForwardMs - baselineSelectedDenseMs + aneSelectedDenseMs) == 0
        ? null
        : baselineForwardMs /
              (baselineForwardMs -
                  baselineSelectedDenseMs +
                  aneSelectedDenseMs),
    'selected_array_input_layers': selectedArrayInput,
    'selected_array_input_layers_csv': selectedArrayInput.join(','),
    'baseline_selected_array_input_ms': baselineSelectedArrayInputMs,
    'ane_selected_array_input_ms': aneSelectedArrayInputMs,
    'estimated_array_input_hybrid_forward_per_iter_ms':
        baselineForwardMs -
        baselineSelectedArrayInputMs +
        aneSelectedArrayInputMs,
    'estimated_array_input_hybrid_speedup_vs_baseline':
        (baselineForwardMs -
                baselineSelectedArrayInputMs +
                aneSelectedArrayInputMs) ==
            0
        ? null
        : baselineForwardMs /
              (baselineForwardMs -
                  baselineSelectedArrayInputMs +
                  aneSelectedArrayInputMs),
    'layers': rows,
  };

  if (emitJson) {
    stdout.writeln(jsonEncode(report));
    exit(0);
  }
  stdout.writeln(const JsonEncoder.withIndent('  ').convert(report));
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

double _benchmarkForward(
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

double _benchmarkBaselineLayer(
  Qwen3_5Runner runner,
  Qwen3_5DenseMlpCapture capture, {
  required int warmup,
  required int iters,
}) {
  for (var index = 0; index < warmup; index++) {
    runner.runDenseMlpForLayer(
      capture.layerIndex,
      capture.input,
      seqLen: capture.seqLen,
    );
  }
  final watch = Stopwatch()..start();
  for (var index = 0; index < iters; index++) {
    runner.runDenseMlpForLayer(
      capture.layerIndex,
      capture.input,
      seqLen: capture.seqLen,
    );
  }
  watch.stop();
  return watch.elapsedMicroseconds / 1000.0 / iters;
}

double _benchmarkAneLayer(
  Map<String, Object?> spec,
  Qwen3_5DenseMlpCapture capture, {
  required int dim,
  required int lane,
  required int warmup,
  required int iters,
}) {
  final model = _loadModel(spec);
  try {
    model.compile();
    model.load();
    final session = model.createSession(
      inputByteSizes: [(spec['input_bytes'] as num).toInt()],
      outputByteSizes: [(spec['output_bytes'] as num).toInt()],
    );
    try {
      final packed = _pack(
        capture.input,
        dim: dim,
        lane: lane,
        seqLen: capture.seqLen,
      );
      for (var index = 0; index < warmup; index++) {
        session.writeInputRawFloat32(0, packed);
        session.evaluate();
        session.readOutputRawFloat32View(0);
      }
      final watch = Stopwatch()..start();
      for (var index = 0; index < iters; index++) {
        session.writeInputRawFloat32(0, packed);
        session.evaluate();
        session.readOutputRawFloat32View(0);
      }
      watch.stop();
      return watch.elapsedMicroseconds / 1000.0 / iters;
    } finally {
      session.close();
    }
  } finally {
    model.close();
  }
}

double _benchmarkAneArrayInputLayer(
  Map<String, Object?> spec,
  Qwen3_5DenseMlpCapture capture, {
  required int dim,
  required int lane,
  required int warmup,
  required int iters,
}) {
  final model = _loadModel(spec);
  try {
    model.compile();
    model.load();
    final session = model.createSession(
      inputByteSizes: [(spec['input_bytes'] as num).toInt()],
      outputByteSizes: [(spec['output_bytes'] as num).toInt()],
    );
    final inputArray = MlxArray.fromFloat32List(
      capture.input,
      shape: [1, capture.seqLen, dim],
    );
    try {
      for (var index = 0; index < warmup; index++) {
        session.writeInputPackedArrayFloat32(
          0,
          inputArray,
          seqLen: capture.seqLen,
          dim: dim,
          lane: lane,
        );
        session.evaluate();
        session.readOutputRawFloat32View(0);
      }
      final watch = Stopwatch()..start();
      for (var index = 0; index < iters; index++) {
        session.writeInputPackedArrayFloat32(
          0,
          inputArray,
          seqLen: capture.seqLen,
          dim: dim,
          lane: lane,
        );
        session.evaluate();
        session.readOutputRawFloat32View(0);
      }
      watch.stop();
      return watch.elapsedMicroseconds / 1000.0 / iters;
    } finally {
      inputArray.close();
      session.close();
    }
  } finally {
    model.close();
  }
}

MlxAnePrivateModel _loadModel(Map<String, Object?> spec) {
  final milText = File(spec['model_mil']! as String).readAsStringSync();
  final weights = <MlxAneWeightWithOffset>[
    for (final weight in List<Map<String, Object?>>.from(
      (spec['weights'] as List).cast<Map>(),
    ))
      (
        path: weight['path']! as String,
        data: Uint8List.fromList(
          File(weight['file']! as String).readAsBytesSync(),
        ),
        offset: (weight['offset']! as num).toInt(),
      ),
  ];
  return mx.anePrivate.modelFromMilWithOffsets(milText, weights: weights);
}

Float32List _pack(
  Float32List values, {
  required int dim,
  required int lane,
  required int seqLen,
}) {
  final packed = Float32List(dim * lane);
  for (var token = 0; token < seqLen; token++) {
    final srcBase = token * dim;
    for (var channel = 0; channel < dim; channel++) {
      packed[channel * lane + token] = values[srcBase + channel];
    }
  }
  return packed;
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
