// Dart-side per-step token dumper for the MLX-decoder parity audit
// (issue #1, commit #10 of the hybrid OCR refactor).
//
// Loads an MLX PaddleOCR-VL decoder snapshot with `keepVisionWeights:false`
// — i.e. the same load mode `PaddleOcrVlHybridRunner` uses — then runs
// K greedy decode steps from a HF-projected `image_embeds` tensor and a
// HF-tokenised `prompt_ids` array. The result is a JSON file containing
// the per-step argmax tokens, ready to be diffed against the HF reference
// by `mlx_decode_parity.py`.
//
// This script is the Dart-side of `mlx_decode_parity.py`. It is NOT
// intended for benchmarking — it does no warmup, no timing, and it
// writes its tokens to a file for reproducible cross-process comparison.
//
// CLI:
//
//   dart run benchmark/paddle_ocr_vl/dart_decode_dump.dart \
//     --snapshot=/path/to/snapshot \
//     --image-embeds=/tmp/image_embeds.npy \
//     --prompt-ids=/tmp/prompt_ids.npy \
//     --grid-thw=1,H,W \
//     --steps=8 \
//     --out=/tmp/dart_tokens.json
//
// Output JSON:
//
//   {
//     "decode_tokens": [t0, t1, ..., tK],
//     "prompt_length": <int>,
//     "expanded_prompt_length": <int>,
//     "grid_height": H,
//     "grid_width":  W
//   }
//
// `decode_tokens[0]` is the seed (= the token sampled from prefill logits),
// so the array has length `steps + 1` to align with the HF reference's
// `[seed_token, *K_continuations]` convention used by parity.py:Stage C.

import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/mlx.dart';
import 'package:dart_inference/models.dart';

String _arg(List<String> args, String name) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) return arg.substring(prefix.length);
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

({int t, int h, int w}) _parseGrid(String spec) {
  final parts = spec.split(',');
  if (parts.length != 3) {
    throw ArgumentError('--grid-thw must be T,H,W (got "$spec")');
  }
  return (
    t: int.parse(parts[0].trim()),
    h: int.parse(parts[1].trim()),
    w: int.parse(parts[2].trim()),
  );
}

void main(List<String> args) {
  final snapshotPath = _arg(args, '--snapshot');
  final imageEmbedsPath = _arg(args, '--image-embeds');
  final promptIdsPath = _arg(args, '--prompt-ids');
  final gridSpec = _arg(args, '--grid-thw');
  final steps = _intArg(args, '--steps', 8);
  final outPath = _arg(args, '--out');

  final grid = _parseGrid(gridSpec);

  // Match the hybrid runner's load mode: drop visual.* tensors at parse
  // time so we exercise the same MLX surface the production hybrid path
  // exercises.
  final runner = PaddleOcrVlRunner.load(
    snapshotPath,
    keepVisionWeights: false,
  );
  final imageEmbeds = MlxIo.load(imageEmbedsPath);
  final promptIdsArr = MlxIo.load(promptIdsPath);

  try {
    MlxRuntime.evalAll([imageEmbeds, promptIdsArr]);

    if (imageEmbeds.shape.length != 2) {
      throw StateError(
        'image_embeds must be rank-2 [num_image_tokens, hidden]; '
        'got shape=${imageEmbeds.shape}',
      );
    }
    final numImageTokens = imageEmbeds.shape[0];
    final hidden = imageEmbeds.shape[1];
    final mergedExpected = grid.t * (grid.h ~/ 2) * (grid.w ~/ 2);
    if (numImageTokens != mergedExpected) {
      throw StateError(
        'image_embeds row count $numImageTokens does not match '
        'merged-token count $mergedExpected for grid '
        '(${grid.t},${grid.h},${grid.w}). '
        'spatial_merge_size assumed 2; if the model uses a different '
        'merge size, the caller must regenerate inputs.',
      );
    }
    if (hidden != runner.config.hiddenSize) {
      throw StateError(
        'image_embeds hidden=$hidden differs from decoder '
        'hiddenSize=${runner.config.hiddenSize}; HF projector and MLX '
        'decoder snapshots are out of sync.',
      );
    }

    final promptIds = promptIdsArr
        .toList()
        .cast<num>()
        .map((n) => n.toInt())
        .toList(growable: false);

    // generate one seed token (from prefill) + `steps` continuations.
    final result = runner.generateFromVisionFeaturesDetailed(
      promptIds,
      imageEmbeds,
      gridHeight: grid.h,
      gridWidth: grid.w,
      maxNewTokens: steps + 1,
    );

    final fullTokens = result.fullTokenIds;
    final expanded = result.expandedPromptLength;
    final decodeTokens = fullTokens.sublist(expanded);

    final payload = jsonEncode(<String, Object?>{
      'decode_tokens': decodeTokens,
      'prompt_length': promptIds.length,
      'expanded_prompt_length': expanded,
      'grid_height': grid.h,
      'grid_width': grid.w,
      'requested_steps': steps,
    });

    final outFile = File(outPath);
    outFile.parent.createSync(recursive: true);
    outFile.writeAsStringSync('$payload\n');

    // Echo to stdout for debuggability when invoked directly.
    stdout.writeln(payload);
  } finally {
    promptIdsArr.close();
    imageEmbeds.close();
    runner.close();
  }

  exit(0);
}
