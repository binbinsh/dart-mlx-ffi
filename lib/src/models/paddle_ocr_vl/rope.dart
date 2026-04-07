part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// M-RoPE (Multimodal Rotary Position Embedding) for ERNIE-4.5
//
// - NeoX-style rotate_half: [-x2, x1] interleaved halves
// - mrope_section = [16, 24, 24] → 64 rotary dims = headDim/2
// - 3 position streams (temporal, height, width); for text-only all equal
// - rope_theta = 500000
//
// ---------------------------------------------------------------------------

extension PaddleOcrVlRope on PaddleOcrVlRunner {
  /// Compute RoPE inverse frequencies for the full head dimension.
  ///
  /// Returns a 1-D array of shape [headDim/2].
  MlxArray _getInvFreq() {
    final existing = _ropeInvFreq;
    if (existing != null) return existing;

    final halfDim = config.headDim ~/ 2; // 64
    final created = MlxArray.fromFloat32List(
      [
        for (var i = 0; i < halfDim; i++)
          math.exp(-(i / halfDim) * math.log(config.ropeTheta)),
      ],
      shape: [halfDim],
    );
    _ropeInvFreq = created;
    return created;
  }

  // -----------------------------------------------------------------------
  // Cos/sin table for a given sequence
  // -----------------------------------------------------------------------

  /// Build cos/sin tables for M-RoPE given per-token 3D position IDs.
  ///
  /// [positionIds] has shape `[3, 1, seqLen]` (temporal, height, width).
  /// Returns cos/sin each of shape `[1, 1, seqLen, headDim]`.
  ({MlxArray cos, MlxArray sin}) _buildMropeCosSin(
    MlxArray positionIds,
    MlxDType dtype,
  ) {
    final invFreq = _getInvFreq(); // [halfDim]
    final seqLen = positionIds.shape[2];
    final invExpanded = invFreq
        .reshape([1, 1, invFreq.shape[0], 1])
        .broadcastTo([3, positionIds.shape[1], invFreq.shape[0], 1]);
    final positionExpanded = positionIds.astype(MlxDType.MLX_FLOAT32).reshape([
      3,
      positionIds.shape[1],
      1,
      seqLen,
    ]);

    try {
      final freqs = mx.matmul(invExpanded, positionExpanded).transposeAxes([
        0,
        1,
        3,
        2,
      ]);
      try {
        final emb = mx.concatenate([freqs, freqs], axis: 3);
        try {
          final splitSections = <int>[
            ...config.mropeSection,
            ...config.mropeSection,
          ];
          final parts = <MlxArray>[];
          var start = 0;
          for (var i = 0; i < splitSections.length; i++) {
            final end = start + splitSections[i];
            final part = emb
                .slice(
                  start: [i % 3, 0, 0, start],
                  stop: [i % 3 + 1, 1, seqLen, end],
                )
                .reshape([1, seqLen, end - start]);
            parts.add(part);
            start = end;
          }
          final interleaved = mx.concatenate(parts, axis: 2);
          for (final part in parts) {
            part.close();
          }
          try {
            final cosArr = interleaved.cos().astype(dtype).reshape([
              1,
              1,
              seqLen,
              config.headDim,
            ]);
            final sinArr = interleaved.sin().astype(dtype).reshape([
              1,
              1,
              seqLen,
              config.headDim,
            ]);
            return (cos: cosArr, sin: sinArr);
          } finally {
            interleaved.close();
          }
        } finally {
          emb.close();
        }
      } finally {
        freqs.close();
      }
    } finally {
      positionExpanded.close();
      invExpanded.close();
    }
  }

  // -----------------------------------------------------------------------
  // Apply RoPE to Q and K
  // -----------------------------------------------------------------------

  /// Apply M-RoPE to query and key tensors.
  ///
  /// [q] shape: `[1, numHeads, seqLen, headDim]`
  /// [k] shape: `[1, numKvHeads, seqLen, headDim]`
  /// [positionIds] shape: `[3, 1, seqLen]`
  ({MlxArray q, MlxArray k}) _applyMrope(
    MlxArray q,
    MlxArray k,
    MlxArray positionIds,
  ) {
    final pair = _buildMropeCosSin(positionIds, q.dtype);
    // cos/sin shape: [1, 1, seqLen, headDim] — broadcasts over heads
    final qRot = _rotaryEmbed(q, pair.cos, pair.sin);
    final kRot = _rotaryEmbed(k, pair.cos, pair.sin);
    pair.cos.close();
    pair.sin.close();
    return (q: qRot, k: kRot);
  }

  /// NeoX-style rotary embedding: x * cos + rotate_half(x) * sin
  MlxArray _rotaryEmbed(MlxArray x, MlxArray cos, MlxArray sin) {
    final rotated = _rotateHalf(x);
    try {
      final a = x * cos;
      final b = rotated * sin;
      final result = mx.add(a, b);
      a.close();
      b.close();
      return result;
    } finally {
      rotated.close();
    }
  }

  /// NeoX rotate_half: split last dim in two halves, negate first, swap.
  /// [-x2, x1]
  MlxArray _rotateHalf(MlxArray x) {
    final half = x.shape[3] ~/ 2;
    final x1 = x.slice(
      start: [0, 0, 0, 0],
      stop: [x.shape[0], x.shape[1], x.shape[2], half],
    );
    final x2 = x.slice(
      start: [0, 0, 0, half],
      stop: [x.shape[0], x.shape[1], x.shape[2], x.shape[3]],
    );
    try {
      final negX2 = x2.negative();
      try {
        return mx.concatenate([negX2, x1], axis: 3);
      } finally {
        negX2.close();
      }
    } finally {
      x2.close();
      x1.close();
    }
  }

  // -----------------------------------------------------------------------
  // Position ID construction helpers
  // -----------------------------------------------------------------------

  /// Build sequential text-only position IDs: all 3 streams get [offset..offset+len).
  ///
  /// Returns shape `[3, 1, seqLen]`.
  MlxArray _textPositionIds(int seqLen, {int offset = 0}) {
    final ids = <int>[for (var i = 0; i < seqLen; i++) offset + i];
    // Repeat for 3 streams
    final flat = <int>[...ids, ...ids, ...ids];
    return MlxArray.fromInt32List(flat, shape: [3, 1, seqLen]);
  }

  /// Build multimodal position IDs for a single-image prompt.
  ///
  /// This follows the official MLX / Hugging Face `get_rope_index()` logic:
  /// text preceding the image uses sequential 1D positions, the image token run
  /// uses compact 3D positions derived from the merged vision grid, and the
  /// trailing text resumes from `max(vision_positions) + 1`.
  ///
  /// Returns the position tensor plus the first position ID to use for the
  /// next generated token.
  ({MlxArray ids, int nextTextPosition}) _multimodalPositionIds(
    List<int> tokenIds,
    int gridH,
    int gridW,
  ) {
    final mergeSize = config._vision.spatialMergeSize;
    final mergedH = gridH ~/ mergeSize;
    final mergedW = gridW ~/ mergeSize;
    final totalLen = tokenIds.length;

    final imageStart = tokenIds.indexOf(config.imageTokenId);
    if (imageStart < 0) {
      return (ids: _textPositionIds(totalLen), nextTextPosition: totalLen);
    }

    var imageEnd = imageStart;
    while (imageEnd < totalLen && tokenIds[imageEnd] == config.imageTokenId) {
      imageEnd++;
    }

    final imageTokenCount = imageEnd - imageStart;
    final expectedImageTokenCount = mergedH * mergedW;
    if (imageTokenCount != expectedImageTokenCount) {
      throw StateError(
        'Expected $expectedImageTokenCount image tokens for grid '
        '$mergedH x $mergedW, but prompt contains $imageTokenCount.',
      );
    }

    final temporal = List<int>.filled(totalLen, 0);
    final height = List<int>.filled(totalLen, 0);
    final width = List<int>.filled(totalLen, 0);

    for (var i = 0; i < imageStart; i++) {
      temporal[i] = i;
      height[i] = i;
      width[i] = i;
    }

    final imageBase = imageStart;
    for (var i = 0; i < imageTokenCount; i++) {
      final tokenIndex = imageStart + i;
      temporal[tokenIndex] = imageBase;
      height[tokenIndex] = imageBase + (i ~/ mergedW);
      width[tokenIndex] = imageBase + (i % mergedW);
    }

    final imageMaxPosition = imageBase + math.max(mergedH, mergedW).toInt() - 1;
    final trailingTextBase = imageMaxPosition + 1;
    for (var i = imageEnd; i < totalLen; i++) {
      final textPosition = trailingTextBase + (i - imageEnd);
      temporal[i] = textPosition;
      height[i] = textPosition;
      width[i] = textPosition;
    }

    final flat = <int>[...temporal, ...height, ...width];
    return (
      ids: MlxArray.fromInt32List(flat, shape: [3, 1, totalLen]),
      nextTextPosition: trailingTextBase + (totalLen - imageEnd),
    );
  }
}
