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
  /// Returns a 2-D tensor `[numMergedTokens, lmHiddenSize]`.
  MlxArray _encodeImage(MlxArray pixels) {
    final vCfg = config._vision;

    // 1. Patch embedding (Conv2d, stride = patchSize)
    final patchOut = mx.conv2d(
      pixels,
      _visionWeights.patchEmbedWeight,
      stride: [vCfg.patchSize, vCfg.patchSize],
    );
    // patchOut: [1, gridH, gridW, hiddenSize]
    final gridH = patchOut.shape[1];
    final gridW = patchOut.shape[2];
    final numPatches = gridH * gridW;
    var hidden = patchOut.reshape([1, numPatches, vCfg.hiddenSize]);
    patchOut.close();

    // Add patch embedding bias if present
    if (_visionWeights.patchEmbedBias != null) {
      final biased = mx.add(hidden, _visionWeights.patchEmbedBias!);
      hidden.close();
      hidden = biased;
    }

    // 2. Position embedding
    hidden = _addPositionEmbedding(hidden, numPatches);

    // 3. ViT transformer blocks
    for (var i = 0; i < _visionWeights.blocks.length; i++) {
      hidden = _visionBlock(_visionWeights.blocks[i], hidden, vCfg);
    }

    // 4. Spatial-merge projector
    final merged = _spatialMergeProject(hidden, gridH, gridW, vCfg);
    hidden.close();
    return merged;
  }

  // -----------------------------------------------------------------------
  // Position embedding
  // -----------------------------------------------------------------------

  MlxArray _addPositionEmbedding(MlxArray hidden, int numPatches) {
    final posEmbed = _visionWeights.positionEmbedding;
    final indices = MlxArray.fromInt32List(
      List<int>.generate(numPatches, (i) => i),
      shape: [numPatches],
    );
    try {
      MlxArray posVec;
      if (posEmbed case final _QuantLinear q) {
        final rowsW = q.weight.take(indices, axis: 0);
        final rowsS = q.scales.take(indices, axis: 0);
        final rowsB = q.biases?.take(indices, axis: 0);
        final gathered = MlxQuantizedMatrix(rowsW, rowsS, rowsB);
        try {
          posVec = mx.quant.dequantize(
            gathered,
            groupSize: q.quantSpec.groupSize,
            bits: q.quantSpec.bits,
            mode: q.quantSpec.mode,
            dtype: hidden.dtype,
          );
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      } else if (posEmbed case final _DenseLinear d) {
        posVec = d.weight.take(indices, axis: 0);
      } else {
        throw StateError('Unsupported position embedding type.');
      }
      final pos3d = posVec.reshape([1, numPatches, config._vision.hiddenSize]);
      posVec.close();
      final result = mx.add(hidden, pos3d);
      pos3d.close();
      hidden.close();
      return result;
    } finally {
      indices.close();
    }
  }

  // -----------------------------------------------------------------------
  // Single ViT block
  // -----------------------------------------------------------------------

  MlxArray _visionBlock(
    _VisionBlockWeights block,
    MlxArray input,
    _VisionConfig vCfg,
  ) {
    // ── Pre-norm 1 ──
    final norm1 = mx.fast.layerNorm(
      input,
      weight: block.layerNorm1Weight,
      bias: block.layerNorm1Bias,
      eps: vCfg.layerNormEps,
    );

    // ── Self-attention (fused QKV) ──
    final attnOut = _visionAttention(block, norm1, vCfg);
    norm1.close();

    // ── Residual 1 ──
    final h = mx.add(input, attnOut);
    attnOut.close();
    input.close();

    // ── Pre-norm 2 ──
    final norm2 = mx.fast.layerNorm(
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
  ) {
    final seqLen = input.shape[1];
    final numHeads = vCfg.numAttentionHeads;
    final headDim = vCfg.headDim;

    // Fused QKV projection: [1, seq, 3*hidden]
    final qkv = block.qkv.apply(input.reshape([seqLen, vCfg.hiddenSize]));
    final qkv3d = qkv.reshape([1, seqLen, 3 * vCfg.hiddenSize]);
    qkv.close();

    // Split into Q, K, V
    final q = qkv3d
        .slice(start: [0, 0, 0], stop: [1, seqLen, vCfg.hiddenSize])
        .reshape([1, seqLen, numHeads, headDim])
        .transposeAxes([0, 2, 1, 3]);
    final k = qkv3d
        .slice(
          start: [0, 0, vCfg.hiddenSize],
          stop: [1, seqLen, 2 * vCfg.hiddenSize],
        )
        .reshape([1, seqLen, numHeads, headDim])
        .transposeAxes([0, 2, 1, 3]);
    final v = qkv3d
        .slice(
          start: [0, 0, 2 * vCfg.hiddenSize],
          stop: [1, seqLen, 3 * vCfg.hiddenSize],
        )
        .reshape([1, seqLen, numHeads, headDim])
        .transposeAxes([0, 2, 1, 3]);
    qkv3d.close();

    // Scaled dot-product attention (no causal mask for vision)
    final attn = mx.fast.scaledDotProductAttention(
      q,
      k,
      v,
      scale: 1.0 / math.sqrt(headDim.toDouble()),
    );
    q.close();
    k.close();
    v.close();

    // Merge heads and project output
    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      numHeads * headDim,
    ]);
    attn.close();

    final out = block.outProj.apply(merged);
    merged.close();
    return out.reshape([1, seqLen, vCfg.hiddenSize]);
  }

  // -----------------------------------------------------------------------
  // Vision MLP (fc1 → GELU → fc2)
  // -----------------------------------------------------------------------

  MlxArray _visionMlp(_VisionBlockWeights block, MlxArray input) {
    final seqLen = input.shape[1];
    final flat = input.reshape([seqLen, config._vision.hiddenSize]);
    final h = block.fc1.apply(flat);
    flat.close();
    final activated = _gelu(h);
    h.close();
    final out = block.fc2.apply(activated);
    activated.close();
    return out.reshape([1, seqLen, config._vision.hiddenSize]);
  }

  // -----------------------------------------------------------------------
  // Spatial-merge projector
  // -----------------------------------------------------------------------

  /// Performs 2×2 spatial merging then projects to LM hidden size.
  ///
  /// Input `hidden`: `[1, gridH*gridW, visionHidden]`
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

    // Reshape to [1, gridH, gridW, visionHidden]
    final grid = hidden.reshape([1, gridH, gridW, vCfg.hiddenSize]);

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

    // Pre-norm → linear1 → GELU → linear2
    final proj = _visionWeights.projector;
    final normed = mx.fast.layerNorm(
      flat,
      weight: proj.preNormWeight,
      bias: proj.preNormBias,
      eps: vCfg.layerNormEps,
    );
    flat.close();
    final h = proj.linear1.apply(normed);
    normed.close();
    final activated = _gelu(h);
    h.close();
    final out = proj.linear2.apply(activated);
    activated.close();
    MlxRuntime.evalAll([out]);
    return out; // [mergedTokens, lmHidden]
  }

  // -----------------------------------------------------------------------
  // GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
  // -----------------------------------------------------------------------

  MlxArray _gelu(MlxArray x) {
    final invSqrt2 = MlxArray.fromFloat32List(
      [1.0 / math.sqrt(2.0)],
      shape: [1],
    ).astype(x.dtype);
    final half = MlxArray.fromFloat32List([0.5], shape: [1]).astype(x.dtype);
    final one = MlxArray.fromFloat32List([1.0], shape: [1]).astype(x.dtype);
    try {
      final scaled = x * invSqrt2;
      final erfVal = scaled.erf();
      scaled.close();
      final sum = mx.add(one, erfVal);
      erfVal.close();
      final left = x * half;
      final result = left * sum;
      left.close();
      sum.close();
      return result;
    } finally {
      invSqrt2.close();
      half.close();
      one.close();
    }
  }
}
