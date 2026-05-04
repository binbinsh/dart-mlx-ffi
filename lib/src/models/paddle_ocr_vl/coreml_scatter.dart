/// Float32-only sibling of [paddleOcrVlScatterImageEmbeddings] for the
/// hybrid CoreML path (issue #1, commit #6).
///
/// The MLX-array variant in `embed.dart` is part of the MLX runner library
/// and pulls in `package:dart_inference/mlx.dart`. The CoreML runner
/// deliberately stays MLX-free at the boundary — its stages produce raw
/// `Float32List` buffers — so we mirror the scatter algorithm on
/// `Float32List` here.
///
/// Both helpers share the same row-by-row contract: copy each row of
/// `imageHidden` into the corresponding `imagePositions[i]` row of
/// `textEmbed`. The Float32 path is a straight loop; we don't need the
/// MLX-side contiguous/sparse split because there's no graph cost to
/// optimise — direct typed-list copies are already cheap.
library;

import 'dart:typed_data';

/// Scatter `imageHidden` rows into `textEmbed` at `imagePositions`,
/// returning a freshly allocated `Float32List` of length
/// `promptLen * hiddenSize` shaped logically as `[1, promptLen, hiddenSize]`.
///
/// - [textEmbed] holds `promptLen * hiddenSize` floats; rows are laid out
///   row-major. The buffer is read but not mutated.
/// - [imageHidden] holds at least `imagePositions.length * hiddenSize`
///   floats. Only the first `imagePositions.length` rows are consumed; any
///   trailing rows are ignored, matching the MLX helper's slicing.
/// - [imagePositions] are indices in `[0, promptLen)` where image-token
///   placeholders occur, in the same order their features appear in
///   `imageHidden`. Must be non-empty.
/// - [promptLen] is the prompt length (rows of `textEmbed`).
/// - [hiddenSize] is the model hidden dim (e.g. 1024 for PaddleOCR-VL-1.5).
///
/// Throws [ArgumentError] for empty positions, out-of-range positions, or
/// undersized buffers.
Float32List paddleOcrVlScatterImageEmbeddingsFloat32({
  required Float32List textEmbed,
  required Float32List imageHidden,
  required List<int> imagePositions,
  required int promptLen,
  required int hiddenSize,
}) {
  if (imagePositions.isEmpty) {
    throw ArgumentError(
      'imagePositions must be non-empty; the caller should short-circuit '
      'the zero-image case.',
    );
  }
  if (textEmbed.length < promptLen * hiddenSize) {
    throw ArgumentError(
      'textEmbed has ${textEmbed.length} floats but promptLen=$promptLen * '
      'hiddenSize=$hiddenSize requires ${promptLen * hiddenSize}.',
    );
  }
  final numImage = imagePositions.length;
  if (imageHidden.length < numImage * hiddenSize) {
    throw ArgumentError(
      'imageHidden has ${imageHidden.length} floats but '
      'imagePositions.length=$numImage * hiddenSize=$hiddenSize requires '
      '${numImage * hiddenSize}.',
    );
  }
  final out = Float32List(promptLen * hiddenSize);
  out.setRange(0, promptLen * hiddenSize, textEmbed);
  for (var i = 0; i < numImage; i++) {
    final pos = imagePositions[i];
    if (pos < 0 || pos >= promptLen) {
      throw ArgumentError(
        'imagePositions[$i]=$pos is outside [0, promptLen=$promptLen).',
      );
    }
    final dstStart = pos * hiddenSize;
    final srcStart = i * hiddenSize;
    out.setRange(
      dstStart,
      dstStart + hiddenSize,
      imageHidden,
      srcStart,
    );
  }
  return out;
}
