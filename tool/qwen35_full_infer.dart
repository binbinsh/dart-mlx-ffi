import 'dart:convert';
import 'dart:io';

import '../private_ane/models/qwen3_5_0_8b/dart/qwen3_5.dart';

void main(List<String> args) {
  final options = _parseArgs(args);
  final snapshotDir = options['snapshot-dir'];
  final tokenIdsFile = options['token-ids-file'];
  if (snapshotDir == null || tokenIdsFile == null) {
    stderr.writeln(
      'Usage: dart run tool/qwen35_full_infer.dart '
      '--snapshot-dir <dir> --token-ids-file <json> '
      '[--private-ane-artifacts-dir <dir>] [--max-new-tokens N] [--json]',
    );
    exitCode = 64;
    return;
  }

  final payload = Map<String, Object?>.from(
    jsonDecode(File(tokenIdsFile).readAsStringSync()) as Map,
  );
  final promptTokenIds = List<int>.from(
    (payload['token_ids'] as List).cast<num>(),
  );
  final eosTokenId = (payload['eos_token_id'] as num?)?.toInt();
  final maxNewTokens = int.tryParse(options['max-new-tokens'] ?? '') ?? 8;
  final emitJson = options.containsKey('json');

  final loadWatch = Stopwatch()..start();
  final runner = Qwen3_5Runner.load(
    snapshotDir,
    privateAneArtifactsDir: options['private-ane-artifacts-dir'],
  );
  loadWatch.stop();
  try {
    final runWatch = Stopwatch()..start();
    final generated = runner.generateGreedy(
      promptTokenIds,
      maxNewTokens,
      eosTokenId: eosTokenId,
    );
    runWatch.stop();

    final promptLength = promptTokenIds.length;
    final newTokenIds = generated.sublist(promptLength);
    final report = <String, Object?>{
      'snapshot_dir': snapshotDir,
      'private_ane_artifacts_dir': options['private-ane-artifacts-dir'],
      'prompt_token_ids': promptTokenIds,
      'generated_token_ids': generated,
      'new_token_ids': newTokenIds,
      'prompt_length': promptLength,
      'generated_length': generated.length,
      'max_new_tokens': maxNewTokens,
      'load_ms': loadWatch.elapsedMicroseconds / 1000.0,
      'generate_ms': runWatch.elapsedMicroseconds / 1000.0,
      'per_new_token_ms': newTokenIds.isEmpty
          ? 0.0
          : runWatch.elapsedMicroseconds / 1000.0 / newTokenIds.length,
      'eos_token_id': eosTokenId,
    };

    if (emitJson) {
      stdout.writeln(jsonEncode(report));
      exit(0);
    }
    stdout.writeln(const JsonEncoder.withIndent('  ').convert(report));
    exit(0);
  } finally {
    runner.close();
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
