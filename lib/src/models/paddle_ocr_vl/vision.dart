part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// Vision encoder (ViT) + spatial-merge projector forward pass
// ---------------------------------------------------------------------------

extension PaddleOcrVlVision on PaddleOcrVlRunner {
  /// Encode a pre-processed image tensor into LM-space hidden states.
  ///
  /// [pixels] has shape `[1, H, W, C]` (NHWC, float16/float32), already
  /// normalised to the model's expected range.
  ///
  /// Returns merged hidden states plus the patch grid dimensions before
  /// spatial merging.
  ({MlxArray hidden, int gridHeight, int gridWidth}) _encodeImage(
    MlxArray pixels, {
    void Function(String message)? onStage,
  }) {
    final vCfg = config._vision;

    // 1. Patch embedding (Conv2d, stride = patchSize)
    final patchOut = mx.conv2d(
      pixels,
      _visionWeights.patchEmbedWeight,
      stride: [vCfg.patchSize, vCfg.patchSize],
    );
    final gridH = patchOut.shape[1];
    final gridW = patchOut.shape[2];
    var hidden = patchOut.reshape([gridH * gridW, vCfg.hiddenSize]);
    patchOut.close();

    // Add patch embedding bias if present
    if (_visionWeights.patchEmbedBias != null) {
      final biased = mx.add(hidden, _visionWeights.patchEmbedBias!);
      hidden.close();
      hidden = biased;
    }

    // 2. Interpolated position embedding
    final posEmbed = _interpolateVisionPositionEmbedding(
      gridH,
      gridW,
      hidden.dtype,
    );
    final withPos = mx.add(hidden, posEmbed);
    hidden.close();
    posEmbed.close();
    hidden = withPos;
    if (config.enableVisionLayerwiseEvalForCurrentPlatform) {
      MlxRuntime.evalAll([hidden]);
    }
    onStage?.call('encodeImage: embeddings ready shape=${hidden.shape}');

    // 3. Vision rotary positions depend on the current image grid.
    final baseRotaryPosEmb = _buildVisionRotaryPosEmbedding(
      gridH,
      gridW,
      MlxDType.MLX_FLOAT32,
    );
    MlxArray? windowRotaryPosEmb;

    final windowSize = config.recommendedVisionWindowSizeForCurrentPlatform;
    final windowedLayerCount = math.min(
      config.recommendedVisionWindowedLayerCountForCurrentPlatform,
      _visionWeights.blocks.length,
    );
    _VisionWindowLayout? windowLayout;
    var isWindowOrdered = false;
    if (windowSize > 0 &&
        windowedLayerCount > 0 &&
        (gridH > windowSize || gridW > windowSize)) {
      windowLayout = _buildVisionWindowLayout(gridH, gridW, windowSize);
      final reorder = MlxArray.fromInt32List(
        windowLayout.windowIndices,
        shape: [windowLayout.windowIndices.length],
      );
      final reorderedHidden = hidden.take(reorder, axis: 0);
      final reorderedRotary = baseRotaryPosEmb.take(reorder, axis: 0);
      hidden.close();
      reorder.close();
      hidden = reorderedHidden;
      windowRotaryPosEmb = reorderedRotary;
      isWindowOrdered = true;
      if (config.enableVisionLayerwiseEvalForCurrentPlatform) {
        MlxRuntime.evalAll([hidden, windowRotaryPosEmb]);
      }
      onStage?.call(
        'encodeImage: window attention enabled windowSize=$windowSize '
        'windows=${windowLayout.windowLengths.length} '
        'layers=$windowedLayerCount',
      );
    }

    // 4. ViT transformer blocks
    final evalBatch = config.visionEvalBatchSizeForCurrentPlatform;
    for (var i = 0; i < _visionWeights.blocks.length; i++) {
      final useWindowForLayer =
          windowLayout != null &&
          i < windowedLayerCount &&
          windowRotaryPosEmb != null;
      hidden = _visionBlock(
        _visionWeights.blocks[i],
        hidden,
        vCfg,
        useWindowForLayer ? windowRotaryPosEmb : baseRotaryPosEmb,
        windowLayout: useWindowForLayer ? windowLayout : null,
      );
      // Batch-eval: materialize every N layers (or at the last layer, or
      // right before the windowed→global transition).  This reduces GPU
      // dispatch overhead compared to evaluating after every single layer.
      final isLastLayer = i + 1 == _visionWeights.blocks.length;
      final isTransitionLayer = isWindowOrdered && i + 1 == windowedLayerCount;
      final isGlobalAfterWindow =
          windowLayout != null && !useWindowForLayer && i + 1 > windowedLayerCount;
      final effectiveEvalBatch = isGlobalAfterWindow ? 1 : evalBatch;
      if (config.enableVisionLayerwiseEvalForCurrentPlatform &&
          (isLastLayer ||
              isTransitionLayer ||
              (i + 1) % effectiveEvalBatch == 0)) {
        MlxRuntime.evalAll([hidden]);
      }
      if (isWindowOrdered && i + 1 == windowedLayerCount) {
        final restore = MlxArray.fromInt32List(
          windowLayout!.restoreIndices,
          shape: [windowLayout.restoreIndices.length],
        );
        final restored = hidden.take(restore, axis: 0);
        hidden.close();
        restore.close();
        hidden = restored;
        if (config.enableVisionLayerwiseEvalForCurrentPlatform) {
          MlxRuntime.evalAll([hidden]);
        }
        final activeWindowRotary = windowRotaryPosEmb;
        if (activeWindowRotary != null) {
          activeWindowRotary.close();
        }
        windowRotaryPosEmb = null;
        isWindowOrdered = false;
        onStage?.call(
          'encodeImage: restored patch order after '
          '$windowedLayerCount windowed layers',
        );
      }
      if ((i + 1) % 3 == 0 || i + 1 == _visionWeights.blocks.length) {
        onStage?.call(
          'encodeImage: vision layer ${i + 1}/${_visionWeights.blocks.length}',
        );
      }
    }
    windowRotaryPosEmb?.close();
    baseRotaryPosEmb.close();

    if (isWindowOrdered && windowLayout != null) {
      final restore = MlxArray.fromInt32List(
        windowLayout.restoreIndices,
        shape: [windowLayout.restoreIndices.length],
      );
      final restored = hidden.take(restore, axis: 0);
      hidden.close();
      restore.close();
      hidden = restored;
      MlxRuntime.evalAll([hidden]);
      onStage?.call('encodeImage: restored patch order after window attention');
    }

    final postNorm = _visionLayerNorm(
      hidden,
      weight: _visionWeights.postLayerNormWeight,
      bias: _visionWeights.postLayerNormBias,
      eps: vCfg.layerNormEps,
    );
    hidden.close();
    if (config.enableVisionLayerwiseEvalForCurrentPlatform) {
      MlxRuntime.evalAll([postNorm]);
    }

    // 5. Spatial-merge projector
    final merged = _spatialMergeProject(postNorm, gridH, gridW, vCfg);
    postNorm.close();
    return (hidden: merged, gridHeight: gridH, gridWidth: gridW);
  }

  // -----------------------------------------------------------------------
  // Position embedding + vision rotary helpers
  // -----------------------------------------------------------------------

  Float32List _fullVisionPositionEmbedding() {
    final cached = _visionPositionEmbeddingCache;
    if (cached != null) return cached;

    final posEmbed = _visionWeights.positionEmbedding;
    late final Float32List dense;
    if (posEmbed case final _QuantLinear q) {
      final dequantized = mx.quant.dequantize(
        q.matrix,
        groupSize: q.quantSpec.groupSize,
        bits: q.quantSpec.bits,
        mode: q.quantSpec.mode,
        dtype: MlxDType.MLX_FLOAT32,
      );
      try {
        dense = dequantized.toFloat32List();
      } finally {
        dequantized.close();
      }
    } else if (posEmbed case final _DenseLinear d) {
      dense = d.weight.toFloat32List();
    } else {
      throw StateError('Unsupported position embedding type.');
    }
    _visionPositionEmbeddingCache = dense;
    return dense;
  }

  MlxArray _interpolateVisionPositionEmbedding(
    int gridH,
    int gridW,
    MlxDType dtype,
  ) {
    final hiddenSize = config._vision.hiddenSize;
    final base = _fullVisionPositionEmbedding();
    final baseGrid = math.sqrt(base.length / hiddenSize).round();
    final result = Float32List(gridH * gridW * hiddenSize);

    final rowFloor = List<int>.filled(gridH, 0);
    final rowCeil = List<int>.filled(gridH, 0);
    final rowWeight = List<double>.filled(gridH, 0);
    final colFloor = List<int>.filled(gridW, 0);
    final colCeil = List<int>.filled(gridW, 0);
    final colWeight = List<double>.filled(gridW, 0);

    for (var y = 0; y < gridH; y++) {
      final rowPosition = gridH == 1
          ? 0.0
          : ((y + 0.5) * baseGrid / gridH) - 0.5;
      final unclampedFloor = rowPosition.floor();
      final unclampedCeil = unclampedFloor + 1;
      rowFloor[y] = unclampedFloor.clamp(0, baseGrid - 1);
      rowCeil[y] = unclampedCeil.clamp(0, baseGrid - 1);
      rowWeight[y] = rowPosition - rowFloor[y];
    }

    for (var x = 0; x < gridW; x++) {
      final colPosition = gridW == 1
          ? 0.0
          : ((x + 0.5) * baseGrid / gridW) - 0.5;
      final unclampedFloor = colPosition.floor();
      final unclampedCeil = unclampedFloor + 1;
      colFloor[x] = unclampedFloor.clamp(0, baseGrid - 1);
      colCeil[x] = unclampedCeil.clamp(0, baseGrid - 1);
      colWeight[x] = colPosition - colFloor[x];
    }

    for (var y = 0; y < gridH; y++) {
      final rw = rowWeight[y];
      final oneMinusRw = 1.0 - rw;
      for (var x = 0; x < gridW; x++) {
        final cw = colWeight[x];
        final oneMinusCw = 1.0 - cw;
        final topLeft = ((rowFloor[y] * baseGrid) + colFloor[x]) * hiddenSize;
        final topRight = ((rowFloor[y] * baseGrid) + colCeil[x]) * hiddenSize;
        final bottomLeft = ((rowCeil[y] * baseGrid) + colFloor[x]) * hiddenSize;
        final bottomRight = ((rowCeil[y] * baseGrid) + colCeil[x]) * hiddenSize;
        final target = ((y * gridW) + x) * hiddenSize;
        for (var c = 0; c < hiddenSize; c++) {
          result[target + c] =
              (oneMinusRw * oneMinusCw * base[topLeft + c]) +
              (oneMinusRw * cw * base[topRight + c]) +
              (rw * oneMinusCw * base[bottomLeft + c]) +
              (rw * cw * base[bottomRight + c]);
        }
      }
    }

    final out = MlxArray.fromFloat32List(
      result,
      shape: [gridH * gridW, hiddenSize],
    );
    if (dtype == MlxDType.MLX_FLOAT32) return out;
    final cast = out.astype(dtype);
    out.close();
    return cast;
  }

  MlxArray _buildVisionRotaryPosEmbedding(
    int gridH,
    int gridW,
    MlxDType dtype,
  ) {
    final seqLen = gridH * gridW;
    final rotaryDim = config._vision.headDim ~/ 2;
    final invFreqCount = rotaryDim ~/ 2;
    final maxGrid = math.max(gridH, gridW);

    final base = Float32List(maxGrid * invFreqCount);
    for (var pos = 0; pos < maxGrid; pos++) {
      for (var i = 0; i < invFreqCount; i++) {
        final exponent = (2 * i) / rotaryDim;
        base[(pos * invFreqCount) + i] =
            pos / math.pow(10000.0, exponent).toDouble();
      }
    }

    final freqs = Float32List(seqLen * rotaryDim);
    for (var idx = 0; idx < seqLen; idx++) {
      final row = idx ~/ gridW;
      final col = idx % gridW;
      final target = idx * rotaryDim;
      final rowSource = row * invFreqCount;
      final colSource = col * invFreqCount;
      for (var i = 0; i < invFreqCount; i++) {
        freqs[target + i] = base[rowSource + i];
        freqs[target + invFreqCount + i] = base[colSource + i];
      }
    }

    final out = MlxArray.fromFloat32List(freqs, shape: [seqLen, rotaryDim]);
    if (dtype == MlxDType.MLX_FLOAT32) return out;
    final cast = out.astype(dtype);
    out.close();
    return cast;
  }

  MlxArray _applyVisionRotary(MlxArray tensor, MlxArray rotaryPosEmb) {
    final cosBase = rotaryPosEmb.cos();
    final sinBase = rotaryPosEmb.sin();
    final cos = cosBase.expandDims(1).tile([1, 1, 2]).expandDims(0);
    final sin = sinBase.expandDims(1).tile([1, 1, 2]).expandDims(0);
    cosBase.close();
    sinBase.close();
    final rotated = _rotateHalfVision(tensor);
    try {
      final left = tensor * cos;
      final right = rotated * sin;
      final out = mx.add(left, right);
      left.close();
      right.close();
      cos.close();
      sin.close();
      if (out.dtype == tensor.dtype) return out;
      final cast = out.astype(tensor.dtype);
      out.close();
      return cast;
    } finally {
      rotated.close();
    }
  }

  MlxArray _rotateHalfVision(MlxArray tensor) {
    final half = tensor.shape[3] ~/ 2;
    final x1 = tensor.slice(
      start: [0, 0, 0, 0],
      stop: [tensor.shape[0], tensor.shape[1], tensor.shape[2], half],
    );
    final x2 = tensor.slice(
      start: [0, 0, 0, half],
      stop: [
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
      ],
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
  // Single ViT block
  // -----------------------------------------------------------------------

  MlxArray _visionBlock(
    _VisionBlockWeights block,
    MlxArray input,
    _VisionConfig vCfg,
    MlxArray rotaryPosEmb, {
    _VisionWindowLayout? windowLayout,
  }) {
    // ── Pre-norm 1 ──
    final norm1 = _visionLayerNorm(
      input,
      weight: block.layerNorm1Weight,
      bias: block.layerNorm1Bias,
      eps: vCfg.layerNormEps,
    );

    // ── Self-attention (fused QKV) ──
    final attnOut = _visionAttention(
      block,
      norm1,
      vCfg,
      rotaryPosEmb,
      windowLayout: windowLayout,
    );
    norm1.close();

    // ── Residual 1 ──
    final h = mx.add(input, attnOut);
    attnOut.close();
    input.close();

    // ── Pre-norm 2 ──
    final norm2 = _visionLayerNorm(
      h,
      weight: block.layerNorm2Weight,
      bias: block.layerNorm2Bias,
      eps: vCfg.layerNormEps,
    );

    // ── MLP: fc1 → GELU → fc2 ──
    final mlpOut = _visionMlp(block, norm2);
    norm2.close();

    // ── Residual 2 ──
    final out = mx.add(h, mlpOut);
    mlpOut.close();
    h.close();
    return out;
  }

  // -----------------------------------------------------------------------
  // Vision self-attention (fused QKV)
  // -----------------------------------------------------------------------

  MlxArray _visionAttention(
    _VisionBlockWeights block,
    MlxArray input,
    _VisionConfig vCfg,
    MlxArray rotaryPosEmb, {
    _VisionWindowLayout? windowLayout,
  }) {
    final seqLen = input.shape[0];
    final numHeads = vCfg.numAttentionHeads;
    final headDim = vCfg.headDim;

    // Fused QKV projection: [seq, 3*hidden]
    final qkv = block.qkv.apply(input);
    final qkv4d = qkv.reshape([seqLen, 3, numHeads, headDim]).transposeAxes([
      1,
      0,
      2,
      3,
    ]);
    qkv.close();

    // Split into Q, K, V
    final q = qkv4d.slice(start: [0, 0, 0, 0], stop: [1, seqLen, numHeads, headDim]);
    final k = qkv4d.slice(start: [1, 0, 0, 0], stop: [2, seqLen, numHeads, headDim]);
    final v = qkv4d.slice(start: [2, 0, 0, 0], stop: [3, seqLen, numHeads, headDim]);
    qkv4d.close();

    final qRot = _applyVisionRotary(q, rotaryPosEmb);
    final kRot = _applyVisionRotary(k, rotaryPosEmb);
    q.close();
    k.close();

    final qForAttn = qRot.transposeAxes([0, 2, 1, 3]);
    final kForAttn = kRot.transposeAxes([0, 2, 1, 3]);
    final vForAttn = v.transposeAxes([0, 2, 1, 3]);
    qRot.close();
    kRot.close();

    // Scaled dot-product attention (no causal mask for vision).
    // Large OCR images can produce 10k+ patch tokens; chunk queries to keep
    // peak memory bounded on iPhone while preserving exact full-attention
    // semantics.
    final chunkSize =
        config.recommendedVisionAttentionChunkSizeForCurrentPlatform;
    final attn = windowLayout != null
        ? _windowedVisionAttention(
            qForAttn,
            kForAttn,
            vForAttn,
            headDim: headDim,
            dtype: input.dtype,
            windowLayout: windowLayout,
          )
        : chunkSize > 0 && seqLen > chunkSize
        ? _chunkedVisionAttention(
            qForAttn,
            kForAttn,
            vForAttn,
            headDim: headDim,
            chunkSize: chunkSize,
            dtype: input.dtype,
          )
        : (() {
            final mask = MlxArray.zeros([1, seqLen, seqLen], dtype: input.dtype);
            try {
              return mx.fast.scaledDotProductAttention(
                qForAttn,
                kForAttn,
                vForAttn,
                scale: 1.0 / math.sqrt(headDim.toDouble()),
                mask: mask,
              );
            } finally {
              mask.close();
            }
          })();
    qForAttn.close();
    kForAttn.close();
    vForAttn.close();
    v.close();

    // Merge heads and project output
    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      numHeads * headDim,
    ]);
    attn.close();

    final out = block.outProj.apply(merged);
    merged.close();
    return out;
  }

  MlxArray _chunkedVisionAttention(
    MlxArray q,
    MlxArray k,
    MlxArray v, {
    required int headDim,
    required int chunkSize,
    required MlxDType dtype,
  }) {
    final seqLen = q.shape[2];
    final numHeads = q.shape[1];
    var combined = MlxArray.zeros(
      [1, numHeads, seqLen, headDim],
      dtype: q.dtype,
    );
    for (var start = 0; start < seqLen; start += chunkSize) {
      final end = math.min(start + chunkSize, seqLen);
      final qChunk = q.slice(
        start: [0, 0, start, 0],
        stop: [1, numHeads, end, headDim],
      );
      final mask = MlxArray.zeros([1, end - start, seqLen], dtype: dtype);
      final chunkOut = mx.fast.scaledDotProductAttention(
        qChunk,
        k,
        v,
        scale: 1.0 / math.sqrt(headDim.toDouble()),
        mask: mask,
      );
      mask.close();
      qChunk.close();
      final updated = combined.sliceUpdate(
        chunkOut,
        start: [0, 0, start, 0],
        stop: [1, numHeads, end, headDim],
      );
      chunkOut.close();
      combined.close();
      combined = updated;
    }
    return combined;
  }

  MlxArray _windowedVisionAttention(
    MlxArray q,
    MlxArray k,
    MlxArray v, {
    required int headDim,
    required MlxDType dtype,
    required _VisionWindowLayout windowLayout,
  }) {
    final seqLen = q.shape[2];
    var combined = MlxArray.zeros(
      [1, q.shape[1], seqLen, headDim],
      dtype: q.dtype,
    );
    var start = 0;
    for (final length in windowLayout.windowLengths) {
      final end = start + length;
      final qChunk = q.slice(
        start: [0, 0, start, 0],
        stop: [1, q.shape[1], end, headDim],
      );
      final kChunk = k.slice(
        start: [0, 0, start, 0],
        stop: [1, k.shape[1], end, headDim],
      );
      final vChunk = v.slice(
        start: [0, 0, start, 0],
        stop: [1, v.shape[1], end, headDim],
      );
      final mask = MlxArray.zeros([1, length, length], dtype: dtype);
      final out = mx.fast.scaledDotProductAttention(
        qChunk,
        kChunk,
        vChunk,
        scale: 1.0 / math.sqrt(headDim.toDouble()),
        mask: mask,
      );
      mask.close();
      qChunk.close();
      kChunk.close();
      vChunk.close();
      final updated = combined.sliceUpdate(
        out,
        start: [0, 0, start, 0],
        stop: [1, q.shape[1], end, headDim],
      );
      out.close();
      combined.close();
      combined = updated;
      start = end;
    }
    return combined;
  }

  // -----------------------------------------------------------------------
  // Vision MLP (fc1 → GELU → fc2)
  // -----------------------------------------------------------------------

  MlxArray _visionMlp(_VisionBlockWeights block, MlxArray input) {
    final h = block.fc1.apply(input);
    final activated = _gelu(h);
    h.close();
    final out = block.fc2.apply(activated);
    activated.close();
    return out;
  }

  // -----------------------------------------------------------------------
  // Spatial-merge projector
  // -----------------------------------------------------------------------

  /// Performs 2×2 spatial merging then projects to LM hidden size.
  ///
  /// Input `hidden`: `[gridH*gridW, visionHidden]`
  /// Output:         `[mergedTokens, lmHidden]`
  MlxArray _spatialMergeProject(
    MlxArray hidden,
    int gridH,
    int gridW,
    _VisionConfig vCfg,
  ) {
    final m = vCfg.spatialMergeSize; // 2
    final mergedH = gridH ~/ m;
    final mergedW = gridW ~/ m;
    final proj = _visionWeights.projector;

    // The projector's pre-norm runs on the per-patch hidden dimension before
    // 2x2 spatial merging. The merged tensor is then fed into linear_1.
    final normed = _visionLayerNorm(
      hidden,
      weight: proj.preNormWeight,
      bias: proj.preNormBias,
      eps: vCfg.layerNormEps,
    );

    // Reshape to [1, gridH, gridW, visionHidden] to mirror the Python MLX
    // projector path (single-image batch with t=1).
    final grid = normed.reshape([1, gridH, gridW, vCfg.hiddenSize]);
    normed.close();

    // Gather 2×2 patches: reshape to
    //   [1, mergedH, m, mergedW, m, visionHidden]
    // then transpose to [1, mergedH, mergedW, m, m, visionHidden]
    // then flatten last 3 dims: [1, mergedH, mergedW, m*m*visionHidden]
    final reshaped = grid.reshape([1, mergedH, m, mergedW, m, vCfg.hiddenSize]);
    grid.close();
    final transposed = reshaped.transposeAxes([0, 1, 3, 2, 4, 5]);
    reshaped.close();
    final flat = transposed.reshape([
      mergedH * mergedW,
      m * m * vCfg.hiddenSize,
    ]);
    transposed.close();

    // linear1 → GELU → linear2
    final h = proj.linear1.apply(flat);
    flat.close();
    final activated = _gelu(h);
    h.close();
    final out = proj.linear2.apply(activated);
    activated.close();
    MlxRuntime.evalAll([out]);
    return out; // [mergedTokens, lmHidden]
  }

  // -----------------------------------------------------------------------
  // GELU activation — MLX "precise" approximation
  //
  // Matches `nn.GELU(approx="precise")` in the Python MLX runtime:
  //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  // -----------------------------------------------------------------------

  MlxArray _gelu(MlxArray x) {
    final cubicCoeff = MlxArray.fromFloat32List(
      [0.044715],
      shape: [1],
    ).astype(x.dtype);
    final scale = MlxArray.fromFloat32List(
      [math.sqrt(2 / math.pi)],
      shape: [1],
    ).astype(x.dtype);
    final half = MlxArray.fromFloat32List([0.5], shape: [1]).astype(x.dtype);
    final one = MlxArray.fromFloat32List([1.0], shape: [1]).astype(x.dtype);
    try {
      final xSquared = x * x;
      final xCubed = xSquared * x;
      xSquared.close();
      final cubicTerm = xCubed * cubicCoeff;
      xCubed.close();
      final inner = mx.add(x, cubicTerm);
      cubicTerm.close();
      final scaled = inner * scale;
      inner.close();
      final tanhVal = scaled.tanh();
      scaled.close();
      final sum = mx.add(one, tanhVal);
      tanhVal.close();
      final left = x * half;
      final result = left * sum;
      left.close();
      sum.close();
      return result;
    } finally {
      cubicCoeff.close();
      scale.close();
      half.close();
      one.close();
    }
  }

  MlxArray _visionLayerNorm(
    MlxArray input, {
    required MlxArray weight,
    required MlxArray bias,
    required double eps,
  }) {
    return mx.fast.layerNorm(
      input,
      weight: weight,
      bias: bias,
      eps: eps,
    );
  }
}

final class _VisionWindowLayout {
  const _VisionWindowLayout({
    required this.windowIndices,
    required this.restoreIndices,
    required this.windowLengths,
  });

  final List<int> windowIndices;
  final List<int> restoreIndices;
  final List<int> windowLengths;
}

_VisionWindowLayout _buildVisionWindowLayout(
  int gridH,
  int gridW,
  int windowSize,
) {
  final padH = ((-gridH) % windowSize + windowSize) % windowSize;
  final padW = ((-gridW) % windowSize + windowSize) % windowSize;
  final paddedH = gridH + padH;
  final paddedW = gridW + padW;

  final padded = List<int>.filled(paddedH * paddedW, -1);
  for (var row = 0; row < gridH; row++) {
    for (var col = 0; col < gridW; col++) {
      padded[(row * paddedW) + col] = (row * gridW) + col;
    }
  }

  final windowIndices = <int>[];
  final windowLengths = <int>[];
  for (var row = 0; row < paddedH; row += windowSize) {
    for (var col = 0; col < paddedW; col += windowSize) {
      var length = 0;
      for (var dy = 0; dy < windowSize; dy++) {
        for (var dx = 0; dx < windowSize; dx++) {
          final index = padded[((row + dy) * paddedW) + col + dx];
          if (index >= 0) {
            windowIndices.add(index);
            length++;
          }
        }
      }
      if (length > 0) {
        windowLengths.add(length);
      }
    }
  }

  final restoreIndices = List<int>.filled(windowIndices.length, 0);
  for (var reordered = 0; reordered < windowIndices.length; reordered++) {
    restoreIndices[windowIndices[reordered]] = reordered;
  }

  return _VisionWindowLayout(
    windowIndices: windowIndices,
    restoreIndices: restoreIndices,
    windowLengths: windowLengths,
  );
}
