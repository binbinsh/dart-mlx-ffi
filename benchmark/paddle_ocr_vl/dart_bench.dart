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
  final pixelValuesPath = args.any((arg) => arg.startsWith('--pixel-values='))
      ? _arg(args, '--pixel-values')
      : '';
  final imageGridThwPath = args.any((arg) => arg.startsWith('--image-grid-thw='))
      ? _arg(args, '--image-grid-thw')
      : '';
  final warmup = _intArg(args, '--warmup', 0);
  final iters = _intArg(args, '--iters', 1);

  final runner = PaddleOcrVlRunner.load(snapshotPath);
  final inputIds = mx.io.load(inputIdsPath);
  final image = mx.io.load(imagePath);
  final pixelValues = pixelValuesPath.isEmpty ? null : mx.io.load(pixelValuesPath);
  final imageGridThw = imageGridThwPath.isEmpty ? null : mx.io.load(imageGridThwPath);
  String? output;
  try {
    final preloaded = <MlxArray>[inputIds, image];
    if (pixelValues != null) preloaded.add(pixelValues);
    if (imageGridThw != null) preloaded.add(imageGridThw);
    MlxRuntime.evalAll(preloaded);
    final promptIds = inputIds
        .toList()
        .cast<num>()
        .map((n) => n.toInt())
        .toList(growable: false);

    MlxArray runOnce() {
      final logits = (pixelValues != null && imageGridThw != null)
          ? (() {
              final full = runner.debugPrefillLogitsFromPixelValues(
                promptIds,
                pixelValues,
                imageGridThw,
              );
              try {
                return full
                    .slice(start: [0, 0], stop: [1, 16])
                    .reshape([1, 16]);
              } finally {
                full.close();
              }
            })()
          : runner.debugPrefillLogitsFromImage(promptIds, image);
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
      output = jsonEncode(<String, Object?>{
          'per_iter_ms': watch.elapsedMicroseconds / 1000.0 / iters,
          'values': List<double>.from(preview.toList().cast<double>()),
        });
    } finally {
      preview.close();
      last.close();
    }
  } finally {
    imageGridThw?.close();
    pixelValues?.close();
    image.close();
    inputIds.close();
    runner.close();
  }
  stdout.writeln(output);
  exit(0);
}
