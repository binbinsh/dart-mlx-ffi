// PaddleOCR-VL benchmark harness.
//
// Two modes are supported (selected via --mode):
//
//   * mlx-only (default): the legacy path used by every release before
//     commit #9 of issue #1. Loads the full MLX runner (vision + decoder)
//     and runs `debugPrefillLogitsFromPixelValues` for K iterations,
//     reporting per-iter ms + the first 16 logits of the final-position
//     distribution.
//
//   * hybrid: the new CoreML-vision + MLX-decoder path introduced in
//     commit #8. Loads `PaddleOcrVlHybridRunner.load(...)` and exercises
//     the matching debug helper `debugFirstTokenLogitsPrefix(...)`. The
//     report's `max_abs_diff` is the diff between this path's logits[:16]
//     and the python_ref logits[:16] — drift between runs of this number
//     means either the CoreML vision_embed or the MLX decoder regressed.
//
// JSON contract emitted on stdout (last line, single line):
//   { "mode": "mlx-only"|"hybrid",
//     "per_iter_ms": <double>,
//     "values": [<float>, ... 16 entries],
//     "iters": <int>,
//     "warmup": <int> }

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/mlx.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/hybrid_runner.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/paddle_ocr_vl.dart';

String _arg(List<String> args, String name) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return arg.substring(prefix.length);
    }
  }
  throw ArgumentError('Missing $name');
}

String? _optArg(List<String> args, String name) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return arg.substring(prefix.length);
    }
  }
  return null;
}

int _intArg(List<String> args, String name, int fallback) {
  final raw = _optArg(args, name);
  if (raw == null) return fallback;
  return int.tryParse(raw) ?? fallback;
}

void _runMlxOnly(List<String> args) {
  final snapshotPath = _arg(args, '--snapshot');
  final inputIdsPath = _arg(args, '--input-ids');
  final imagePath = _arg(args, '--image');
  final pixelValuesPath = _optArg(args, '--pixel-values') ?? '';
  final imageGridThwPath = _optArg(args, '--image-grid-thw') ?? '';
  final warmup = _intArg(args, '--warmup', 0);
  final iters = _intArg(args, '--iters', 1);

  final runner = PaddleOcrVlRunner.load(snapshotPath);
  final inputIds = mx.io.load(inputIdsPath);
  final image = mx.io.load(imagePath);
  final pixelValues = pixelValuesPath.isEmpty ? null : mx.io.load(pixelValuesPath);
  final imageGridThw =
      imageGridThwPath.isEmpty ? null : mx.io.load(imageGridThwPath);
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
      runOnce().close();
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
        'mode': 'mlx-only',
        'per_iter_ms': watch.elapsedMicroseconds / 1000.0 / iters,
        'values': List<double>.from(preview.toList().cast<double>()),
        'iters': iters,
        'warmup': warmup,
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
}

Uint8List _loadRgbHwcU8(String path) {
  // python_ref dumps the original PIL image as a uint8 .npy of shape
  // [H, W, 3]. mx.io.load can read uint8 .npy but our high-level toList()
  // doesn't expose uint8 — cast to int32 and downcast to bytes.
  final arr = mx.io.load(path);
  try {
    final asI32 = arr.astype(MlxDType.MLX_INT32);
    try {
      MlxRuntime.evalAll([asI32]);
      final list = asI32.toList().cast<num>();
      final bytes = Uint8List(list.length);
      for (var i = 0; i < list.length; i++) {
        bytes[i] = list[i].toInt() & 0xff;
      }
      return bytes;
    } finally {
      asI32.close();
    }
  } finally {
    arr.close();
  }
}

Future<void> _runHybrid(List<String> args) async {
  final snapshotPath = _arg(args, '--snapshot');
  final coremlBundle = _arg(args, '--coreml-bundle');
  final inputIdsPath = _arg(args, '--input-ids');
  final rgbPath = _arg(args, '--image-rgb-hwc-u8');
  final imageHeight = _intArg(args, '--image-height', -1);
  final imageWidth = _intArg(args, '--image-width', -1);
  if (imageHeight <= 0 || imageWidth <= 0) {
    throw ArgumentError(
      'Hybrid mode requires --image-height and --image-width '
      '(original image dims; the runner re-runs smart_resize).',
    );
  }
  final warmup = _intArg(args, '--warmup', 0);
  final iters = _intArg(args, '--iters', 1);

  final runner = await PaddleOcrVlHybridRunner.load(
    coremlBundlePath: coremlBundle,
    mlxSnapshotPath: snapshotPath,
  );

  final inputIds = mx.io.load(inputIdsPath);
  String? output;
  try {
    MlxRuntime.evalAll([inputIds]);
    final promptIds = inputIds
        .toList()
        .cast<num>()
        .map((n) => n.toInt())
        .toList(growable: false);
    final imageBytes = _loadRgbHwcU8(rgbPath);

    MlxArray runOnce() {
      final logits = runner.debugFirstTokenLogitsPrefix(
        imageBytes: imageBytes,
        imageHeight: imageHeight,
        imageWidth: imageWidth,
        promptIds: promptIds,
        width: 16,
      );
      MlxRuntime.evalAll([logits]);
      return logits;
    }

    for (var i = 0; i < warmup; i++) {
      runOnce().close();
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
        'mode': 'hybrid',
        'per_iter_ms': watch.elapsedMicroseconds / 1000.0 / iters,
        'values': List<double>.from(preview.toList().cast<double>()),
        'iters': iters,
        'warmup': warmup,
      });
    } finally {
      preview.close();
      last.close();
    }
  } finally {
    inputIds.close();
    runner.close();
  }
  stdout.writeln(output);
}

void _printHelp() {
  stdout.writeln(
    'Usage: dart run benchmark/paddle_ocr_vl/dart_bench.dart \\\n'
    '         --mode=<mlx-only|hybrid> \\\n'
    '         --snapshot=<mlx snapshot dir> \\\n'
    '         --input-ids=<input_ids.npy> \\\n'
    '         --warmup=<int> --iters=<int> \\\n'
    '         (mlx-only) --image=<image_nhwc.npy> \\\n'
    '                    [--pixel-values=<.npy> --image-grid-thw=<.npy>] \\\n'
    '         (hybrid)   --coreml-bundle=<dir with pipeline.json> \\\n'
    '                    --image-rgb-hwc-u8=<.npy> \\\n'
    '                    --image-height=<int> --image-width=<int>',
  );
}

Future<void> main(List<String> args) async {
  if (args.contains('--help') || args.contains('-h')) {
    _printHelp();
    exit(0);
  }
  final mode = _optArg(args, '--mode') ?? 'mlx-only';
  switch (mode) {
    case 'mlx-only':
      _runMlxOnly(args);
    case 'hybrid':
      await _runHybrid(args);
    default:
      stderr.writeln('Unknown --mode=$mode (expected mlx-only|hybrid)');
      exit(2);
  }
  exit(0);
}
