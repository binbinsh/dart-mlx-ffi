import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/src/models/paddle_ocr_vl/paddle_ocr_vl.dart';

String _arg(List<String> args, String name) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return arg.substring(prefix.length);
    }
  }
  throw ArgumentError('Missing $name');
}

int _intArg(List<String> args, String name, int fallback) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return int.tryParse(arg.substring(prefix.length)) ?? fallback;
    }
  }
  return fallback;
}

void main(List<String> args) {
  final snapshotPath = _arg(args, '--snapshot');
  final inputIdsPath = _arg(args, '--input-ids');
  final imagePath = _arg(args, '--image');
  final warmup = _intArg(args, '--warmup', 0);
  final iters = _intArg(args, '--iters', 1);

  final runner = PaddleOcrVlRunner.load(snapshotPath);
  final inputIds = mx.io.load(inputIdsPath);
  final image = mx.io.load(imagePath);
  String? output;
  try {
    MlxRuntime.evalAll([inputIds, image]);
    final promptIds = inputIds
        .toList()
        .cast<num>()
        .map((n) => n.toInt())
        .toList(growable: false);

    MlxArray runOnce() {
      final logits = runner.debugPrefillLogitsFromImage(promptIds, image);
      MlxRuntime.evalAll([logits]);
      return logits;
    }

    for (var i = 0; i < warmup; i++) {
      final logits = runOnce();
      logits.close();
    }

    final watch = Stopwatch()..start();
    MlxArray? last;
    for (var i = 0; i < iters; i++) {
      last?.close();
      last = runOnce();
    }
    watch.stop();

    if (last == null) {
      throw StateError('No iterations executed.');
    }

    final preview = last
        .reshape([last.size])
        .slice(start: [0], stop: [16])
        .astype(MlxDType.MLX_FLOAT32);
    try {
      MlxRuntime.evalAll([preview]);
      output = jsonEncode(<String, Object?>{
          'per_iter_ms': watch.elapsedMicroseconds / 1000.0 / iters,
          'values': List<double>.from(preview.toList().cast<double>()),
        });
    } finally {
      preview.close();
      last.close();
    }
  } finally {
    image.close();
    inputIds.close();
    runner.close();
  }
  stdout.writeln(output);
  exit(0);
}
