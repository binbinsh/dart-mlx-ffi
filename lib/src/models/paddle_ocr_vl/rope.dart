part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// M-RoPE (Multimodal Rotary Position Embedding) for ERNIE-4.5
//
// - NeoX-style rotate_half: [-x2, x1] interleaved halves
// - mrope_section = [16, 24, 24] → 64 rotary dims = headDim/2
// - 3 position streams (temporal, height, width); for text-only all equal
// - rope_theta = 500000
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
    final halfDim = invFreq.shape[0]; // 64
    final seqLen = positionIds.shape[2];

    // Each mrope section applies a different position-ID stream.
    // sections = [16, 24, 24], sum = 64 = halfDim
    final sections = config.mropeSection;

    // Build per-section frequencies and interleave.
    final freqParts = <MlxArray>[];
    var dimOffset = 0;
    for (var s = 0; s < sections.length; s++) {
      final secLen = sections[s]; // number of freq dims in this section
      if (secLen == 0) {
        dimOffset += secLen;
        continue;
      }

      // Position IDs for this stream: [1, seqLen]
      final posStream = positionIds
          .slice(start: [s, 0, 0], stop: [s + 1, 1, seqLen])
          .reshape([1, seqLen])
          .astype(MlxDType.MLX_FLOAT32);

      // Inverse frequencies for this section: [secLen]
      final secFreq = invFreq
          .slice(start: [dimOffset], stop: [dimOffset + secLen])
          .reshape([1, secLen]);

      // Outer product: [1, seqLen] × [1, secLen] → [seqLen, secLen]
      final posExpanded = posStream.reshape([seqLen, 1]);
      posStream.close();
      final angles = mx.matmul(posExpanded, secFreq);
      posExpanded.close();
      secFreq.close();
      freqParts.add(angles); // [seqLen, secLen]
      dimOffset += secLen;
    }

    // Concatenate all section frequencies → [seqLen, halfDim]
    final freqs = mx.concatenate(freqParts, axis: 1);
    for (final part in freqParts) {
      part.close();
    }

    // Double to full headDim: [seqLen, headDim]
    final fullFreqs = mx.concatenate([freqs, freqs], axis: 1);
    freqs.close();

    // cos/sin: [1, 1, seqLen, headDim]
    final cosArr = fullFreqs.cos().astype(dtype).reshape([
      1,
      1,
      seqLen,
      halfDim * 2,
    ]);
    final sinArr = fullFreqs.sin().astype(dtype).reshape([
      1,
      1,
      seqLen,
      halfDim * 2,
    ]);
    fullFreqs.close();
    return (cos: cosArr, sin: sinArr);
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

  /// Build multimodal position IDs for a sequence containing vision tokens.
  ///
  /// Text tokens get sequential IDs on all 3 streams.
  /// Vision tokens get (temporal=t, height=row, width=col) positions.
  ///
  /// [tokenIds] is the full token sequence.
  /// [gridH] / [gridW] are the vision grid dimensions after patch embedding.
  /// Returns shape `[3, 1, totalSeqLen]`.
  MlxArray _multimodalPositionIds(List<int> tokenIds, int gridH, int gridW) {
    final mergeSize = config._vision.spatialMergeSize;
    final mergedH = gridH ~/ mergeSize;
    final mergedW = gridW ~/ mergeSize;
    final numVisionTokens = mergedH * mergedW;
    final totalLen = tokenIds.length;

    final temporal = List<int>.filled(totalLen, 0);
    final height = List<int>.filled(totalLen, 0);
    final width = List<int>.filled(totalLen, 0);

    var textPos = 0;
    var i = 0;
    while (i < totalLen) {
      if (tokenIds[i] == config.visionStartTokenId) {
        // vision_start token
        temporal[i] = textPos;
        height[i] = textPos;
        width[i] = textPos;
        i++;

        // Vision content tokens
        var vIdx = 0;
        while (vIdx < numVisionTokens && i < totalLen) {
          final row = vIdx ~/ mergedW;
          final col = vIdx % mergedW;
          temporal[i] = textPos;
          height[i] = textPos + row;
          width[i] = textPos + col;
          i++;
          vIdx++;
        }

        // vision_end token
        if (i < totalLen && tokenIds[i] == config.visionEndTokenId) {
          temporal[i] = textPos;
          height[i] = textPos + mergedH - 1;
          width[i] = textPos + mergedW - 1;
          i++;
        }

        // Advance text position past the vision span
        textPos += math.max(mergedH, mergedW);
      } else {
        temporal[i] = textPos;
        height[i] = textPos;
        width[i] = textPos;
        textPos++;
        i++;
      }
    }

    final flat = <int>[...temporal, ...height, ...width];
    return MlxArray.fromInt32List(flat, shape: [3, 1, totalLen]);
  }
}
