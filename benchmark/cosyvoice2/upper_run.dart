import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  if (args.length != 1) {
    stderr.writeln(
      'usage: dart run benchmark/cosyvoice2/upper_run.dart <bundle_dir>',
    );
    exitCode = 64;
    return;
  }

  final bundleDir = args[0];
  final metaFile = File('$bundleDir/meta.json');
  if (!metaFile.existsSync()) {
    stderr.writeln('Missing meta.json in $bundleDir');
    exitCode = 66;
    return;
  }

  final meta = jsonDecode(metaFile.readAsStringSync()) as Map<String, Object?>;
  final snapshotPath = meta['snapshot_path']?.toString() ?? '';
  final sampleText = meta['sample_text']?.toString() ?? '';
  final refText = meta['ref_text']?.toString() ?? '';
  if (snapshotPath.isEmpty || sampleText.isEmpty || refText.isEmpty) {
    stderr.writeln('Incomplete CosyVoice2 bundle metadata in $bundleDir');
    exitCode = 66;
    return;
  }

  final warmup = int.tryParse(Platform.environment['COSY_WARMUP'] ?? '') ?? 0;
  final iters = int.tryParse(Platform.environment['COSY_ITERS'] ?? '') ?? 1;
  final seed = int.tryParse(Platform.environment['COSY_SEED'] ?? '') ?? 0;
  final minRatio =
      double.tryParse(Platform.environment['COSY_MIN_RATIO'] ?? '') ?? 1.0;
  final maxRatio =
      double.tryParse(Platform.environment['COSY_MAX_RATIO'] ?? '') ?? 4.0;
  final greedy = Platform.environment['COSY_GREEDY'] == '1';
  final runner = CosyVoice2UpperRunner.load(
    snapshotPath,
    tokenizerPath: bundleDir,
  );
  final prompt = CosyVoice2PromptBundle.load(bundleDir);
  final promptTokens = prompt.promptSpeechToken
      .reshape([prompt.promptSpeechToken.size])
      .toList()
      .cast<int>();

  try {
    for (var index = 0; index < warmup; index++) {
      runner.generateSpeechTokens(
        text: sampleText,
        refText: refText,
        promptSpeechTokens: promptTokens,
        seed: seed,
        minTokenTextRatio: minRatio,
        maxTokenTextRatio: maxRatio,
        greedy: greedy,
      );
    }

    final stopwatch = Stopwatch()..start();
    List<int> tokens = const <int>[];
    for (var index = 0; index < iters; index++) {
      tokens = runner.generateSpeechTokens(
        text: sampleText,
        refText: refText,
        promptSpeechTokens: promptTokens,
        seed: seed,
        minTokenTextRatio: minRatio,
        maxTokenTextRatio: maxRatio,
        greedy: greedy,
      );
    }
    stopwatch.stop();

    stdout.writeln(
      jsonEncode(<String, Object?>{
        'count': tokens.length,
        'per_iter_ms': stopwatch.elapsedMicroseconds / 1000.0 / iters,
        'tokens': tokens,
        'preview': tokens.take(32).toList(growable: false),
      }),
    );
  } finally {
    prompt.close();
    runner.close();
  }
}
