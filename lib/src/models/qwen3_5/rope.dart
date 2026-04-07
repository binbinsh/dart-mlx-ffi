part of 'qwen3_5.dart';

extension on Qwen3_5Runner {
  MlxArray _getRopeInvFreq() {
    final existing = _ropeInvFreq;
    if (existing != null) {
      return existing;
    }
    final created = MlxArray.fromFloat32List(
      [
        for (var i = 0; i < config.rotaryDims; i += 2)
          math.exp(-(i / config.rotaryDims) * math.log(config.ropeTheta)),
      ],
      shape: [config.rotaryDims ~/ 2],
    );
    _ropeInvFreq = created;
    return created;
  }

  ({MlxArray cos, MlxArray sin}) _mropeCosSin(
    int seqLen, {
    required int offset,
    required MlxDType dtype,
  }) {
    final cacheKey = '$seqLen:$offset:${dtype.value}';
    final cached = _ropeCache[cacheKey];
    if (cached != null) {
      return cached;
    }
    final invFreq = _getRopeInvFreq();
    final basePositionIds = MlxArray.fromInt32List(
      [for (var i = 0; i < seqLen; i++) offset + i],
      shape: [1, seqLen],
    );
    try {
      final positionIds = basePositionIds.broadcastTo([3, 1, seqLen]);
      try {
        final freqs = _interleavedMropeFreqs(invFreq, positionIds);
        try {
          final emb = mx.concatenate([freqs, freqs], axis: -1);
          try {
            final pair = (
              cos: emb.cos().astype(dtype),
              sin: emb.sin().astype(dtype),
            );
            if (seqLen > 1 || offset == 0 || (seqLen == 1 && offset < 1024)) {
              _ropeCache[cacheKey] = pair;
            }
            return pair;
          } finally {
            emb.close();
          }
        } finally {
          freqs.close();
        }
      } finally {
        positionIds.close();
      }
    } finally {
      basePositionIds.close();
    }
  }

  ({MlxArray q, MlxArray k}) _applyMrope(
    MlxArray q,
    MlxArray k,
    int seqLen, {
    int offset = 0,
  }) {
    if (Platform.environment['QWEN35_SIMPLE_ROPE'] == '1') {
      return (
        q: mx.fast.rope(
          q,
          dims: config.rotaryDims,
          traditional: false,
          base: config.ropeTheta,
        ),
        k: mx.fast.rope(
          k,
          dims: config.rotaryDims,
          traditional: false,
          base: config.ropeTheta,
        ),
      );
    }
    final pair = _mropeCosSin(seqLen, offset: offset, dtype: q.dtype);
    final qRot = _applyMultimodalRotary(q, pair.cos, pair.sin);
    final kRot = _applyMultimodalRotary(k, pair.cos, pair.sin);
    final cacheKey = '$seqLen:$offset:${q.dtype.value}';
    if (!_ropeCache.containsKey(cacheKey)) {
      pair.sin.close();
      pair.cos.close();
    }
    return (q: qRot, k: kRot);
  }

  /// Apply M-RoPE using explicit multimodal position IDs.
  ///
  /// [positionIds] has shape `[3, 1, seqLen]` — one row each for temporal,
  /// height, and width dimensions.  Used for vision-language prompts where
  /// image tokens have 2D spatial positions.
  ({MlxArray q, MlxArray k}) applyMropeWithPositionIds(
    MlxArray q,
    MlxArray k,
    MlxArray positionIds,
    int seqLen,
  ) {
    final invFreq = _getRopeInvFreq();
    final freqs = _interleavedMropeFreqs(invFreq, positionIds);
    final emb = mx.concatenate([freqs, freqs], axis: -1);
    freqs.close();
    final cos = emb.cos().astype(q.dtype);
    final sin = emb.sin().astype(q.dtype);
    emb.close();
    final qRot = _applyMultimodalRotary(q, cos, sin);
    final kRot = _applyMultimodalRotary(k, cos, sin);
    cos.close();
    sin.close();
    return (q: qRot, k: kRot);
  }

  MlxArray _interleavedMropeFreqs(MlxArray invFreq, MlxArray positionIds) {
    final invExpanded = invFreq
        .reshape([1, 1, invFreq.shape[0], 1])
        .broadcastTo([3, positionIds.shape[1], invFreq.shape[0], 1]);
    final positionExpanded = positionIds.astype(MlxDType.MLX_FLOAT32).reshape([
      3,
      positionIds.shape[1],
      1,
      positionIds.shape[2],
    ]);
    try {
      final freqs = mx.matmul(invExpanded, positionExpanded).transposeAxes([
        0,
        1,
        3,
        2,
      ]);
      return _applyInterleavedMrope(freqs);
    } finally {
      positionExpanded.close();
      invExpanded.close();
    }
  }

  MlxArray _applyInterleavedMrope(MlxArray freqs) {
    final parts = <MlxArray>[];
    try {
      parts.add(
        freqs
            .slice(
              start: [0, 0, 0, 0],
              stop: [1, freqs.shape[1], freqs.shape[2], freqs.shape[3]],
            )
            .reshape([freqs.shape[1], freqs.shape[2], freqs.shape[3]]),
      );
      for (var dim = 1; dim <= 2; dim++) {
        final length = config.mropeSection[dim] * 3;
        if (length <= dim) {
          continue;
        }
        final replacement = freqs
            .slice(
              start: [dim, 0, 0, 0],
              stop: [dim + 1, freqs.shape[1], freqs.shape[2], freqs.shape[3]],
            )
            .reshape([freqs.shape[1], freqs.shape[2], freqs.shape[3]]);
        final current = parts[0];
        parts[0] = _replaceStrideSlice(current, replacement, dim, length, 3);
        current.close();
        replacement.close();
      }
      return parts[0];
    } finally {
      for (var i = 1; i < parts.length; i++) {
        parts[i].close();
      }
    }
  }

  MlxArray _replaceStrideSlice(
    MlxArray base,
    MlxArray replacement,
    int start,
    int stop,
    int step,
  ) {
    final columns = <MlxArray>[];
    try {
      for (var index = 0; index < base.shape[2]; index++) {
        final source =
            (index >= start && index < stop && (index - start) % step == 0)
            ? replacement
            : base;
        final column = source.slice(
          start: [0, 0, index],
          stop: [source.shape[0], source.shape[1], index + 1],
        );
        columns.add(column);
      }
      return mx.concatenate(columns, axis: 2);
    } finally {
      for (final column in columns) {
        column.close();
      }
    }
  }

  MlxArray _applyMultimodalRotary(MlxArray input, MlxArray cos, MlxArray sin) {
    final rotaryDim = cos.shape[2];
    final rot = input.slice(
      start: [0, 0, 0, 0],
      stop: [input.shape[0], input.shape[1], input.shape[2], rotaryDim],
    );
    final pass = input.slice(
      start: [0, 0, 0, rotaryDim],
      stop: [input.shape[0], input.shape[1], input.shape[2], input.shape[3]],
    );
    final cos4 = cos.reshape([cos.shape[0], 1, cos.shape[1], cos.shape[2]]);
    final sin4 = sin.reshape([sin.shape[0], 1, sin.shape[1], sin.shape[2]]);
    try {
      final rotated = _rotateHalf(rot);
      try {
        final embed = (rot * cos4) + (rotated * sin4);
        try {
          return mx.concatenate([embed, pass], axis: 3);
        } finally {
          embed.close();
        }
      } finally {
        rotated.close();
      }
    } finally {
      sin4.close();
      cos4.close();
      pass.close();
      rot.close();
    }
  }

  MlxArray _rotateHalf(MlxArray input) {
    final half = input.shape[3] ~/ 2;
    final x1 = input.slice(
      start: [0, 0, 0, 0],
      stop: [input.shape[0], input.shape[1], input.shape[2], half],
    );
    final x2 = input.slice(
      start: [0, 0, 0, half],
      stop: [input.shape[0], input.shape[1], input.shape[2], input.shape[3]],
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
}
