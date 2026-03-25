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
      'Usage: dart run tool/qwen35_private_ane_pack_bench.dart '
      '--snapshot-dir <dir> --artifacts-dir <dir> --token-ids-file <json> '
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

  final metadata = Map<String, Object?>.from(
    jsonDecode(File('$artifactsDir/metadata.json').readAsStringSync()) as Map,
  );
  final allFfn = Map<String, Object?>.from(metadata['all_ffn']! as Map);
  final lane = (allFfn['lane'] as num).toInt();
  final dim = (allFfn['dim'] as num).toInt();
  final layerFilter = List<int>.from((allFfn['layers'] as List).cast<num>());

  final runnerLoad = Stopwatch()..start();
  final runner = Qwen3_5Runner.load(snapshotDir);
  runnerLoad.stop();

  final captures = runner
      .captureDenseMlp(tokenIds)
      .where((capture) => layerFilter.contains(capture.layerIndex))
      .toList();
  final baselineForwardMs = _benchmarkForward(
    runner,
    tokenIds,
    warmup: warmup,
    iters: iters,
  );
  final baselineDenseMs = _benchmarkDenseReplay(
    runner,
    captures,
    warmup: warmup,
    iters: iters,
  );

  final packedInputs = <Float32List>[
    for (final capture in captures)
      _pack(capture.input, dim: dim, lane: lane, seqLen: capture.seqLen),
  ];

  final aneLoad = Stopwatch()..start();
  final model = _loadAllFfnModel(allFfn);
  model.compile();
  model.load();
  final session = model.createSession(
    inputByteSizes: List<int>.from(
      (allFfn['input_byte_sizes'] as List).cast<num>(),
    ),
    outputByteSizes: List<int>.from(
      (allFfn['output_byte_sizes'] as List).cast<num>(),
    ),
  );
  aneLoad.stop();

  final aneOutputs = _runAllFfnOnce(session, packedInputs);
  final unpacked = <Float32List>[
    for (var index = 0; index < captures.length; index++)
      _unpack(
        aneOutputs[index],
        dim: dim,
        lane: lane,
        seqLen: captures[index].seqLen,
      ),
  ];
  final aneDenseMs = _benchmarkAllFfn(
    session,
    packedInputs,
    warmup: warmup,
    iters: iters,
  );

  final perLayer = <Map<String, Object?>>[];
  var maxAbsDiff = 0.0;
  var sumAbsDiff = 0.0;
  var diffCount = 0;
  for (var index = 0; index < captures.length; index++) {
    final expected = captures[index].output;
    final actual = unpacked[index];
    var layerMax = 0.0;
    var layerSum = 0.0;
    for (var valueIndex = 0; valueIndex < expected.length; valueIndex++) {
      final diff = (actual[valueIndex] - expected[valueIndex]).abs();
      if (diff > layerMax) {
        layerMax = diff;
      }
      if (diff > maxAbsDiff) {
        maxAbsDiff = diff;
      }
      layerSum += diff;
      sumAbsDiff += diff;
      diffCount++;
    }
    perLayer.add({
      'layer': captures[index].layerIndex,
      'seq_len': captures[index].seqLen,
      'max_abs_diff': layerMax,
      'mean_abs_diff': layerSum / expected.length,
    });
  }

  session.close();
  model.close();
  runner.close();

  final report = <String, Object?>{
    'runtime': 'qwen35_private_ane_all_ffn',
    'snapshot_dir': snapshotDir,
    'artifacts_dir': artifactsDir,
    'prompt': prompt,
    'token_ids': tokenIds,
    'token_count': tokenIds.length,
    'runner_load_ms': runnerLoad.elapsedMicroseconds / 1000.0,
    'ane_load_ms': aneLoad.elapsedMicroseconds / 1000.0,
    'warmup': warmup,
    'iters': iters,
    'dense_layer_count': captures.length,
    'baseline_forward_per_iter_ms': baselineForwardMs,
    'baseline_dense_ffn_per_iter_ms': baselineDenseMs,
    'ane_all_ffn_per_iter_ms': aneDenseMs,
    'ane_speedup_vs_dense_ffn': aneDenseMs == 0
        ? null
        : baselineDenseMs / aneDenseMs,
    'max_abs_diff': maxAbsDiff,
    'mean_abs_diff': diffCount == 0 ? 0.0 : sumAbsDiff / diffCount,
    'layers': perLayer,
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

double _benchmarkDenseReplay(
  Qwen3_5Runner runner,
  List<Qwen3_5DenseMlpCapture> captures, {
  required int warmup,
  required int iters,
}) {
  for (var index = 0; index < warmup; index++) {
    for (final capture in captures) {
      runner.runDenseMlpForLayer(
        capture.layerIndex,
        capture.input,
        seqLen: capture.seqLen,
      );
    }
  }
  final watch = Stopwatch()..start();
  for (var index = 0; index < iters; index++) {
    for (final capture in captures) {
      runner.runDenseMlpForLayer(
        capture.layerIndex,
        capture.input,
        seqLen: capture.seqLen,
      );
    }
  }
  watch.stop();
  return watch.elapsedMicroseconds / 1000.0 / iters;
}

double _benchmarkAllFfn(
  MlxAnePrivateSession session,
  List<Float32List> packedInputs, {
  required int warmup,
  required int iters,
}) {
  for (var index = 0; index < warmup; index++) {
    session.runRawFloat32(packedInputs);
  }
  final watch = Stopwatch()..start();
  for (var index = 0; index < iters; index++) {
    session.runRawFloat32(packedInputs);
  }
  watch.stop();
  return watch.elapsedMicroseconds / 1000.0 / iters;
}

List<Float32List> _runAllFfnOnce(
  MlxAnePrivateSession session,
  List<Float32List> packedInputs,
) => session.runRawFloat32(packedInputs);

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

MlxAnePrivateModel _loadAllFfnModel(Map<String, Object?> allFfn) {
  final milText = File(allFfn['model_mil']! as String).readAsStringSync();
  final weights = <MlxAneWeightWithOffset>[
    for (final weight in List<Map<String, Object?>>.from(
      (allFfn['weights'] as List).cast<Map>(),
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

Float32List _unpack(
  Float32List packed, {
  required int dim,
  required int lane,
  required int seqLen,
}) {
  final out = Float32List(seqLen * dim);
  for (var token = 0; token < seqLen; token++) {
    final dstBase = token * dim;
    for (var channel = 0; channel < dim; channel++) {
      out[dstBase + channel] = packed[channel * lane + token];
    }
  }
  return out;
}
