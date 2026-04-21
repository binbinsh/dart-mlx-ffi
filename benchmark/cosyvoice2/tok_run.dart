import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  if (args.length != 1) {
    stderr.writeln(
      'usage: dart run benchmark/cosyvoice2/tok_run.dart <bundle_dir>',
    );
    exitCode = 64;
    return;
  }

  final bundleDir = args[0];
  final inputPath = File('$bundleDir/sample_inputs.safetensors');
  if (!inputPath.existsSync()) {
    stderr.writeln('Missing sample_inputs.safetensors in $bundleDir');
    exitCode = 66;
    return;
  }

  final warmup = int.tryParse(Platform.environment['COSY_WARMUP'] ?? '') ?? 0;
  final iters = int.tryParse(Platform.environment['COSY_ITERS'] ?? '') ?? 1;
  final seed = int.tryParse(Platform.environment['COSY_SEED'] ?? '') ?? 0;
  final valuesPath = Platform.environment['COSY_VALUES_PATH'];
  final fullValues = Platform.environment['COSY_FULL_VALUES'] == '1';

  final bundle = CosyVoice2LowerBundle.load(bundleDir);
  final inputs = mx.io.loadSafetensors(inputPath.path).tensors;
  final tokens = inputs['tokens'];
  if (tokens == null) {
    stderr.writeln('Missing tokens tensor in ${inputPath.path}');
    exitCode = 66;
    return;
  }
  final tokenIds = tokens.reshape([tokens.size]).toList().cast<int>();

  try {
    for (var index = 0; index < warmup; index++) {
      final result = bundle.synthesise(tokenIds, seed: seed);
      result.close();
    }

    final stopwatch = Stopwatch()..start();
    CosyVoice2LowerResult? last;
    for (var index = 0; index < iters; index++) {
      last?.close();
      last = bundle.synthesise(tokenIds, seed: seed);
    }
    stopwatch.stop();
    if (last == null) {
      throw StateError('No CosyVoice2 lower iterations executed.');
    }

    final flat = last.audio.reshape([last.audio.size]);
    final preview = flat
        .slice(start: [0], stop: [flat.shape[0] < 16 ? flat.shape[0] : 16])
        .astype(MlxDType.MLX_FLOAT32);
    final payload = <String, Object?>{
      'shape': last.audio.shape,
      'sample_rate': last.sampleRate,
      'per_iter_ms': stopwatch.elapsedMicroseconds / 1000.0 / iters,
      'output_preview': List<double>.from(preview.toList().cast<double>()),
    };
    if (valuesPath != null && valuesPath.isNotEmpty) {
      mx.io.saveSafetensors(valuesPath, <String, MlxArray>{
        'audio': last.audio,
      });
      payload['values_path'] = valuesPath;
    }
    if (fullValues) {
      payload['values'] = List<double>.from(flat.toList().cast<double>());
    }
    preview.close();
    flat.close();
    stdout.writeln(jsonEncode(payload));
    last.close();
    exit(0);
  } finally {
    bundle.close();
    tokens.close();
  }
}
