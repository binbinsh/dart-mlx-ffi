import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'mlx_vlm/models/qwen3_5/qwen3_5.dart';

void main(List<String> args) {
  if (args.length != 2) {
    stderr.writeln(
      'usage: dart run benchmark/qwen3_5/qwen35_run.dart <snapshot_path> <token_ids_json>',
    );
    exitCode = 64;
    return;
  }

  final snapshotPath = args[0];
  final debug = Platform.environment['QWEN35_DEBUG'] == '1';
  final compile = Platform.environment['QWEN35_COMPILE'] == '1';
  final tokenIds = (jsonDecode(args[1]) as List<Object?>)
      .cast<num>()
      .map((value) => value.toInt())
      .toList();
  if (debug) {
    stderr.writeln('qwen35_run: loading $snapshotPath');
  }
  final runner = Qwen3_5Runner.load(snapshotPath);
  try {
    if (debug) {
      stderr.writeln('qwen35_run: running ${tokenIds.length} tokens');
    }
    if (compile) {
      MlxCompile.enable();
      MlxCompile.setMode(MlxCompileMode.MLX_COMPILE_MODE_ENABLED);
    }
    final logits = runner.run(tokenIds).astype(MlxDType.MLX_FLOAT32);
    try {
      if (debug) {
        stderr.writeln('qwen35_run: completed');
      }
      final payload = <String, Object?>{
        'shape': logits.shape,
        'values': List<double>.from(logits.toList().cast<double>()),
      };
      stdout.writeln(jsonEncode(payload));
      exit(0);
    } finally {
      logits.close();
      if (compile) {
        MlxCompile.disable();
      }
    }
  } finally {
    runner.close();
  }
}
