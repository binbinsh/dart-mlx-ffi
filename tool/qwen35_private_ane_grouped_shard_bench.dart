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
      'Usage: dart run tool/qwen35_private_ane_grouped_shard_bench.dart '
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
  final lane = (metadata['lane'] as num).toInt();
  final dim = (metadata['dim'] as num).toInt();
  final shardSpecs = List<Map<String, Object?>>.from(
    (metadata['grouped_shards'] as List).cast<Map>(),
  );

  final runnerLoad = Stopwatch()..start();
  final runner = Qwen3_5Runner.load(snapshotDir);
  runnerLoad.stop();

  final captures = runner.captureDenseMlp(tokenIds);
  final captureByLayer = <int, Qwen3_5DenseMlpCapture>{
    for (final capture in captures) capture.layerIndex: capture,
  };

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

  final shardLoad = Stopwatch()..start();
  final shards = <_LoadedGroupedShard>[];
  try {
    for (final spec in shardSpecs) {
      final layers = List<int>.from((spec['layers'] as List).cast<num>());
      final model = _loadPackedModel(spec);
      model.compile();
      model.load();
      final session = model.createSession(
        inputByteSizes: List<int>.from(
          (spec['input_byte_sizes'] as List).cast<num>(),
        ),
        outputByteSizes: List<int>.from(
          (spec['output_byte_sizes'] as List).cast<num>(),
        ),
      );
      final packedInput = _packGrouped(
        [for (final layer in layers) captureByLayer[layer]!.input],
        dim: dim,
        lane: lane,
        seqLen: captureByLayer[layers.first]!.seqLen,
      );
      shards.add(
        _LoadedGroupedShard(
          layers: layers,
          session: session,
          model: model,
          packedInput: packedInput,
        ),
      );
    }
    shardLoad.stop();

    final shardOutputs = <int, Float32List>{};
    for (final shard in shards) {
      final output = shard.session.runRawFloat32([shard.packedInput]).single;
      final unpacked = _unpackGrouped(
        output,
        dim: dim,
        lane: lane,
        seqLen: captureByLayer[shard.layers.first]!.seqLen,
        groups: shard.layers.length,
      );
      for (var index = 0; index < shard.layers.length; index++) {
        shardOutputs[shard.layers[index]] = unpacked[index];
      }
    }

    final shardDenseMs = _benchmarkGroupedShards(
      shards,
      warmup: warmup,
      iters: iters,
    );

    final perLayer = <Map<String, Object?>>[];
    var maxAbsDiff = 0.0;
    var sumAbsDiff = 0.0;
    var diffCount = 0;
    for (final capture in captures) {
      final actual = shardOutputs[capture.layerIndex];
      if (actual == null) {
        continue;
      }
      var layerMax = 0.0;
      var layerSum = 0.0;
      for (var index = 0; index < capture.output.length; index++) {
        final diff = (actual[index] - capture.output[index]).abs();
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
        'layer': capture.layerIndex,
        'seq_len': capture.seqLen,
        'max_abs_diff': layerMax,
        'mean_abs_diff': layerSum / capture.output.length,
      });
    }

    final report = <String, Object?>{
      'runtime': 'qwen35_private_ane_grouped_shards',
      'snapshot_dir': snapshotDir,
      'artifacts_dir': artifactsDir,
      'prompt': prompt,
      'token_ids': tokenIds,
      'token_count': tokenIds.length,
      'runner_load_ms': runnerLoad.elapsedMicroseconds / 1000.0,
      'grouped_shard_load_ms': shardLoad.elapsedMicroseconds / 1000.0,
      'warmup': warmup,
      'iters': iters,
      'dense_layer_count': captures.length,
      'grouped_shard_count': shards.length,
      'grouped_shard_layers': [for (final shard in shards) shard.layers],
      'baseline_forward_per_iter_ms': baselineForwardMs,
      'baseline_dense_ffn_per_iter_ms': baselineDenseMs,
      'ane_grouped_shards_per_iter_ms': shardDenseMs,
      'ane_speedup_vs_dense_ffn': shardDenseMs == 0
          ? null
          : baselineDenseMs / shardDenseMs,
      'estimated_hybrid_forward_per_iter_ms':
          baselineForwardMs - baselineDenseMs + shardDenseMs,
      'estimated_hybrid_speedup_vs_baseline_forward':
          (baselineForwardMs - baselineDenseMs + shardDenseMs) == 0
          ? null
          : baselineForwardMs /
                (baselineForwardMs - baselineDenseMs + shardDenseMs),
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
  } finally {
    for (final shard in shards) {
      shard.close();
    }
    runner.close();
  }
}

final class _LoadedGroupedShard {
  _LoadedGroupedShard({
    required this.layers,
    required this.session,
    required this.model,
    required this.packedInput,
  });

  final List<int> layers;
  final MlxAnePrivateSession session;
  final MlxAnePrivateModel model;
  final Float32List packedInput;

  void close() {
    session.close();
    model.close();
  }
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

double _benchmarkGroupedShards(
  List<_LoadedGroupedShard> shards, {
  required int warmup,
  required int iters,
}) {
  for (var index = 0; index < warmup; index++) {
    for (final shard in shards) {
      shard.session.runRawFloat32([shard.packedInput]);
    }
  }
  final watch = Stopwatch()..start();
  for (var index = 0; index < iters; index++) {
    for (final shard in shards) {
      shard.session.runRawFloat32([shard.packedInput]);
    }
  }
  watch.stop();
  return watch.elapsedMicroseconds / 1000.0 / iters;
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

MlxAnePrivateModel _loadPackedModel(Map<String, Object?> spec) {
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

Float32List _packGrouped(
  List<Float32List> inputs, {
  required int dim,
  required int lane,
  required int seqLen,
}) {
  final packed = Float32List(inputs.length * dim * lane);
  for (var group = 0; group < inputs.length; group++) {
    final input = inputs[group];
    for (var token = 0; token < seqLen; token++) {
      final srcBase = token * dim;
      final dstGroupBase = group * dim * lane;
      for (var channel = 0; channel < dim; channel++) {
        packed[dstGroupBase + channel * lane + token] =
            input[srcBase + channel];
      }
    }
  }
  return packed;
}

List<Float32List> _unpackGrouped(
  Float32List packed, {
  required int dim,
  required int lane,
  required int seqLen,
  required int groups,
}) {
  final outputs = <Float32List>[
    for (var index = 0; index < groups; index++) Float32List(seqLen * dim),
  ];
  for (var group = 0; group < groups; group++) {
    final srcGroupBase = group * dim * lane;
    final output = outputs[group];
    for (var token = 0; token < seqLen; token++) {
      final dstBase = token * dim;
      for (var channel = 0; channel < dim; channel++) {
        output[dstBase + channel] =
            packed[srcGroupBase + channel * lane + token];
      }
    }
  }
  return outputs;
}
