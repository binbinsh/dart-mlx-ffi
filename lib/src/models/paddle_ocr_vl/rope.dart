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
  static bool _loggedDisabledSingleTokenMrope = false;

  bool get _debugMropeTraceEnabled {
    final raw = Platform.environment['DART_MLX_PADDLE_DEBUG_MROPE'];
    return raw == '1' || raw == 'true';
  }

  bool get _disableFastSingleTokenMropeForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.disableFastSingleTokenMrope;
    if (debug != null) return debug;
    final override = Platform.environment[
      'DART_MLX_PADDLE_DISABLE_FAST_SINGLE_TOKEN_MROPE'
    ];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    // The Metal fast path can still native-crash on iPhone during decode.
    return Platform.isIOS;
  }

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
          1.0 /
              math.pow(
                config.ropeTheta,
                (2 * i) / config.headDim,
              ).toDouble(),
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
  /// Returns cos/sin each of shape `[3, 1, seqLen, headDim]`.
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
          final cosArr = emb.cos().astype(dtype);
          final sinArr = emb.sin().astype(dtype);
          return (cos: cosArr, sin: sinArr);
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
    final disableFastPath = _disableFastSingleTokenMropeForCurrentPlatform;
    if (disableFastPath &&
        Platform.isIOS &&
        !_loggedDisabledSingleTokenMrope &&
        q.shape[2] == 1 &&
        k.shape[2] == 1 &&
        (config.enableDecoderTailTraceForCurrentPlatform ||
            _debugMropeTraceEnabled)) {
      _loggedDisabledSingleTokenMrope = true;
      PaddleOcrVlDebugOverrides.traceSink?.call(
        'mrope fast path forced off on ios q=${q.shape} k=${k.shape}',
      );
    }
    if (!disableFastPath &&
        q.shape[2] == 1 &&
        k.shape[2] == 1) {
      final fast = _applyMropeSingleTokenFast(q, k, positionIds);
      if (fast != null) {
        if (config.enableDecoderTailTraceForCurrentPlatform ||
            _debugMropeTraceEnabled) {
          PaddleOcrVlDebugOverrides.traceSink?.call(
            'mrope fast path q=${q.shape} k=${k.shape}',
          );
        }
        return fast;
      }
      if (config.enableDecoderTailTraceForCurrentPlatform ||
          _debugMropeTraceEnabled) {
        PaddleOcrVlDebugOverrides.traceSink?.call(
          'mrope fast path skipped q=${q.shape} k=${k.shape}',
        );
      }
    }
    final pair = _buildMropeCosSin(positionIds, q.dtype);
    final cos = _applyMultimodalRotarySections(pair.cos);
    final sin = _applyMultimodalRotarySections(pair.sin);
    pair.cos.close();
    pair.sin.close();
    final rotaryDim = cos.shape[3];
    final qRotPart = q.slice(
      start: [0, 0, 0, 0],
      stop: [q.shape[0], q.shape[1], q.shape[2], rotaryDim],
    );
    final qPass = q.slice(
      start: [0, 0, 0, rotaryDim],
      stop: [q.shape[0], q.shape[1], q.shape[2], q.shape[3]],
    );
    final kRotPart = k.slice(
      start: [0, 0, 0, 0],
      stop: [k.shape[0], k.shape[1], k.shape[2], rotaryDim],
    );
    final kPass = k.slice(
      start: [0, 0, 0, rotaryDim],
      stop: [k.shape[0], k.shape[1], k.shape[2], k.shape[3]],
    );
    final qEmbed = _rotaryEmbed(qRotPart, cos, sin);
    final kEmbed = _rotaryEmbed(kRotPart, cos, sin);
    qRotPart.close();
    kRotPart.close();
    cos.close();
    sin.close();
    final qRot = mx.concatenate([qEmbed, qPass], axis: 3);
    final kRot = mx.concatenate([kEmbed, kPass], axis: 3);
    qEmbed.close();
    kEmbed.close();
    qPass.close();
    kPass.close();
    return (q: qRot, k: kRot);
  }

  ({MlxArray q, MlxArray k})? _applyMropeSingleTokenFast(
    MlxArray q,
    MlxArray k,
    MlxArray positionIds,
  ) {
    if (q.shape[3] != config.headDim || k.shape[3] != config.headDim) {
      return null;
    }
    final sections = config.mropeSection;
    final halfDim = config.headDim ~/ 2;
    final total = sections.fold<int>(0, (sum, dim) => sum + dim);
    if (total != halfDim) {
      return null;
    }
    final invFreq = _getInvFreq();
    final qRot = _applyMropeFastToTensor(
      q,
      positionIds,
      invFreq,
      sections,
      halfDim,
    );
    final kRot = _applyMropeFastToTensor(
      k,
      positionIds,
      invFreq,
      sections,
      halfDim,
    );
    return (q: qRot, k: kRot);
  }

  MlxArray _applyMropeFastToTensor(
    MlxArray tensor,
    MlxArray positionIds,
    MlxArray invFreq,
    List<int> sections,
    int halfDim,
  ) {
    final firstParts = <MlxArray>[];
    final secondParts = <MlxArray>[];
    var cursor = 0;
    for (var streamIdx = 0; streamIdx < sections.length; streamIdx++) {
      final width = sections[streamIdx];
      final first = tensor.slice(
        start: [0, 0, 0, cursor],
        stop: [
          tensor.shape[0],
          tensor.shape[1],
          tensor.shape[2],
          cursor + width,
        ],
      );
      final second = tensor.slice(
        start: [0, 0, 0, halfDim + cursor],
        stop: [
          tensor.shape[0],
          tensor.shape[1],
          tensor.shape[2],
          halfDim + cursor + width,
        ],
      );
      final chunk = mx.concatenate([first, second], axis: 3);
      first.close();
      second.close();
      final offsetView = positionIds
          .slice(
            start: [streamIdx, 0, 0],
            stop: [streamIdx + 1, 1, 1],
          )
          .reshape([1]);
      final offset = offsetView.dtype == MlxDType.MLX_INT32
          ? offsetView
          : offsetView.astype(MlxDType.MLX_INT32);
      final invSlice = invFreq.slice(
        start: [cursor],
        stop: [cursor + width],
      );
      final freqs = MlxMore.reciprocal(invSlice);
      invSlice.close();
      final rotated = mx.fast.ropeDynamic(
        chunk,
        dims: width * 2,
        offset: offset,
        freqs: freqs,
      );
      chunk.close();
      if (!identical(offset, offsetView)) {
        offset.close();
      }
      offsetView.close();
      freqs.close();
      final rotatedFirst = rotated.slice(
        start: [0, 0, 0, 0],
        stop: [rotated.shape[0], rotated.shape[1], rotated.shape[2], width],
      );
      final rotatedSecond = rotated.slice(
        start: [0, 0, 0, width],
        stop: [
          rotated.shape[0],
          rotated.shape[1],
          rotated.shape[2],
          width * 2,
        ],
      );
      rotated.close();
      firstParts.add(rotatedFirst);
      secondParts.add(rotatedSecond);
      cursor += width;
    }
    final out = mx.concatenate([...firstParts, ...secondParts], axis: 3);
    for (final part in firstParts) {
      part.close();
    }
    for (final part in secondParts) {
      part.close();
    }
    return out;
  }

  MlxArray _applyMultimodalRotarySections(MlxArray x) {
    final sections = <int>[
      ...config.mropeSection,
      ...config.mropeSection,
    ];
    final parts = <MlxArray>[];
    var start = 0;
    for (var i = 0; i < sections.length; i++) {
      final end = start + sections[i];
      final part = x
          .slice(
            start: [i % 3, 0, 0, start],
            stop: [i % 3 + 1, 1, x.shape[2], end],
          )
          .reshape([1, 1, x.shape[2], end - start]);
      parts.add(part);
      start = end;
    }
    final out = mx.concatenate(parts, axis: 3);
    for (final part in parts) {
      part.close();
    }
    return out;
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
