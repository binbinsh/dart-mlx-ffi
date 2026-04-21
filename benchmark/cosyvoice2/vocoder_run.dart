import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  if (args.length != 1) {
    stderr.writeln(
      'usage: dart run benchmark/cosyvoice2/vocoder_run.dart <bundle_dir>',
    );
    exitCode = 64;
    return;
  }

  final bundleDir = args[0];
  final warmup = int.tryParse(Platform.environment['COSY_WARMUP'] ?? '') ?? 0;
  final iters = int.tryParse(Platform.environment['COSY_ITERS'] ?? '') ?? 1;
  final valuesPath = Platform.environment['COSY_VALUES_PATH'];
  final fullValues = Platform.environment['COSY_FULL_VALUES'] == '1';
  final meta =
      jsonDecode(File('$bundleDir/meta.json').readAsStringSync())
          as Map<String, Object?>;
  final buckets =
      (meta['vocoder_buckets'] as List<Object?>?)
          ?.map((value) => (value as num).toInt())
          .toList(growable: false) ??
      const <int>[4096];

  final prompt = CosyVoice2PromptBundle.load(bundleDir);

  final mel = prompt.promptMel
      .transposeAxes([0, 2, 1])
      .astype(MlxDType.MLX_FLOAT32);
  final melShape = mel.shape;
  final bucket = buckets.firstWhere(
    (value) => melShape[2] <= value,
    orElse: () => buckets.last,
  );
  final bundle = CosyVoice2VocoderBundle.load(bundleDir, bucketFrames: bucket);
  final melFlat = mel.reshape([mel.size]);
  final melValues = melFlat.toFloat32List();

  try {
    for (var index = 0; index < warmup; index++) {
      final result = bundle.synthesiseMel(melValues, shape: melShape);
      result.close();
    }

    final stopwatch = Stopwatch()..start();
    CosyVoice2LowerResult? last;
    for (var index = 0; index < iters; index++) {
      last?.close();
      last = bundle.synthesiseMel(melValues, shape: melShape);
    }
    stopwatch.stop();
    if (last == null) {
      throw StateError('No CosyVoice2 vocoder iterations executed.');
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
  } finally {
    melFlat.close();
    mel.close();
    bundle.close();
    prompt.close();
  }
}
