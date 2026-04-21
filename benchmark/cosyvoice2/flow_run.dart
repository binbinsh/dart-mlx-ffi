import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  if (args.length != 1) {
    stderr.writeln(
      'usage: dart run benchmark/cosyvoice2/flow_run.dart <bundle_dir>',
    );
    exitCode = 64;
    return;
  }

  final bundleDir = args[0];
  final warmup = int.tryParse(Platform.environment['COSY_WARMUP'] ?? '') ?? 0;
  final iters = int.tryParse(Platform.environment['COSY_ITERS'] ?? '') ?? 1;
  final seed = int.tryParse(Platform.environment['COSY_SEED'] ?? '') ?? 0;
  final valuesPath = Platform.environment['COSY_VALUES_PATH'];
  final fullValues = Platform.environment['COSY_FULL_VALUES'] == '1';
  final meta =
      jsonDecode(File('$bundleDir/meta.json').readAsStringSync())
          as Map<String, Object?>;
  final buckets =
      (meta['flow_buckets'] as List<Object?>?)
          ?.map((value) => (value as num).toInt())
          .toList(growable: false) ??
      const <int>[1024];
  final inputs = File('$bundleDir/sample_inputs.safetensors');
  final sample = mx.io.loadSafetensors(inputs.path).tensors;
  final tokensArray = sample['tokens'];
  if (tokensArray == null) {
    stderr.writeln('Missing tokens tensor in ${inputs.path}');
    exitCode = 66;
    return;
  }
  final tokens = tokensArray.reshape([tokensArray.size]).toList().cast<int>();
  final bucket = buckets.firstWhere(
    (value) => tokens.length <= value,
    orElse: () => buckets.last,
  );
  final flow = CosyVoice2FlowBundle.load(bundleDir, bucketTokens: bucket);

  try {
    for (var index = 0; index < warmup; index++) {
      flow.synthesiseTokens(tokens, seed: seed);
    }
    final stopwatch = Stopwatch()..start();
    Float32List mel = Float32List(0);
    for (var index = 0; index < iters; index++) {
      mel = flow.synthesiseTokens(tokens, seed: seed);
    }
    stopwatch.stop();
    final preview = mel.take(16).toList(growable: false);
    if (valuesPath != null && valuesPath.isNotEmpty) {
      final melArray = MlxArray.fromFloat32List(
        mel,
        shape: [1, 80, tokens.length * 2],
      );
      try {
        mx.io.saveSafetensors(valuesPath, <String, MlxArray>{'mel': melArray});
      } finally {
        melArray.close();
      }
    }
    stdout.writeln(
      jsonEncode(<String, Object?>{
        'shape': <int>[1, 80, tokens.length * 2],
        'per_iter_ms': stopwatch.elapsedMicroseconds / 1000.0 / iters,
        'output_preview': preview,
        if (fullValues) 'values': mel,
      }),
    );
  } finally {
    flow.close();
    tokensArray.close();
  }
}
