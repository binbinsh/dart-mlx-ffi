part of 'qwen3_5.dart';

// ---------------------------------------------------------------------------
// Vision encoder (ViT) + spatial-merge projector for Qwen3.5 VLM
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Vision weight data structures
// ---------------------------------------------------------------------------

/// Weights for a single ViT transformer block.
final class _Qwen35VisionBlockWeights {
  const _Qwen35VisionBlockWeights({
    required this.norm1Weight,
    required this.norm1Bias,
    required this.norm2Weight,
    required this.norm2Bias,
    required this.attnQkvWeight,
    required this.attnQkvBias,
    required this.attnProjWeight,
    required this.attnProjBias,
    required this.mlpFc1Weight,
    required this.mlpFc1Bias,
    required this.mlpFc2Weight,
    required this.mlpFc2Bias,
  });

  // LayerNorm (standard with bias)
  final MlxArray norm1Weight;
  final MlxArray norm1Bias;
  final MlxArray norm2Weight;
  final MlxArray norm2Bias;

  // Fused QKV attention
  final MlxArray attnQkvWeight;
  final MlxArray attnQkvBias;
  final MlxArray attnProjWeight;
  final MlxArray attnProjBias;

  // MLP (fc1 → GELU → fc2)
  final MlxArray mlpFc1Weight;
  final MlxArray mlpFc1Bias;
  final MlxArray mlpFc2Weight;
  final MlxArray mlpFc2Bias;
}

/// Weights for the spatial-merge projector (merger).
final class _Qwen35MergerWeights {
  const _Qwen35MergerWeights({
    required this.normWeight,
    required this.normBias,
    required this.fc1Weight,
    required this.fc1Bias,
    required this.fc2Weight,
    required this.fc2Bias,
  });

  final MlxArray normWeight;
  final MlxArray normBias;
  final MlxArray fc1Weight;
  final MlxArray fc1Bias;
  final MlxArray fc2Weight;
  final MlxArray fc2Bias;
}

/// Complete vision encoder weights for Qwen3.5 VLM.
final class _Qwen35VisionWeights {
  const _Qwen35VisionWeights({
    required this.patchEmbedWeight,
    required this.patchEmbedBias,
    required this.posEmbedWeight,
    required this.blocks,
    required this.merger,
  });

  /// Conv3D patch embedding: [outCh, temporal_ps, ps, ps, inChannels].
  final MlxArray patchEmbedWeight;
  final MlxArray patchEmbedBias;

  /// Position embedding: [numPositions, hiddenSize].
  final MlxArray posEmbedWeight;

  /// ViT transformer blocks.
  final List<_Qwen35VisionBlockWeights> blocks;

  /// Spatial-merge projector.
  final _Qwen35MergerWeights merger;
}

// ---------------------------------------------------------------------------
// Vision forward pass (extension on Qwen3_5Runner)
// ---------------------------------------------------------------------------

extension Qwen35Vision on Qwen3_5Runner {
  /// Encode a pre-processed image tensor into LM-space hidden states.
  ///
  /// [patchedPixels] has shape `[N_patches, C * temporal_ps * ps * ps]`,
  /// already normalised and in **merge-grouped** patch order (matching
  /// the Qwen2VLImageProcessor output format).
  ///
  /// [gridH] and [gridW] are the spatial patch grid dimensions
  /// (e.g. 36 and 64 for a 576×1024 image with patch_size=16).
  ///
  /// Returns merged hidden states [mergedTokens, lmHidden] plus the grid dims.
  ({MlxArray hidden, int gridHeight, int gridWidth}) encodeImage(
    MlxArray patchedPixels,
    int gridH,
    int gridW,
    _Qwen35VisionWeights visionWeights, {
    void Function(String message)? onStage,
    void Function(String stage, MlxArray value)? onDumpIntermediate,
  }) {
    final vCfg = config.visionConfig!;

    // 1. Patch embedding (Conv3D-as-matmul)
    //
    // patchedPixels: [N_patches, C * temporal_ps * ps * ps]
    //   → reshape to: [N_patches, C, temporal_ps, ps, ps]
    //   → moveaxis(1,4): [N_patches, temporal_ps, ps, ps, C]
    //
    // patchEmbedWeight: [outCh, temporal_ps, ps, ps, C]
    //   Conv3D with kernel=stride = matmul on flattened vectors.
    //
    // Since kernel_size == stride, each patch is independent:
    //   output = patchedPixels @ weight_reshaped.T + bias
    //
    // Weight shape from safetensors: [outCh, temporal_ps, ps, ps, C]
    // We need: [outCh, temporal_ps * ps * ps * C] for matmul.
    final patchVecSize = patchedPixels.shape[1];
    final nPatches = patchedPixels.shape[0];

    // Reshape weight from [outCh, temp, ps, ps, C] to [outCh, temp*ps*ps*C]
    final weightFlat = visionWeights.patchEmbedWeight.reshape([
      vCfg.hiddenSize,
      patchVecSize,
    ]);

    // However, the Python PatchEmbed does:
    //   input.reshape(-1, C, temp_ps, ps, ps).moveaxis(1, 4)
    //   → input becomes [N, temp, ps, ps, C]
    //   → Conv3D(in_channels=C, kernel=(temp,ps,ps), stride=same)
    //
    // Our patchedPixels is already ordered as [N, C*temp*ps*ps].
    // The reshape(-1, C, temp, ps, ps).moveaxis(1,4) reorders to
    //   [N, temp, ps, ps, C] which flattened is:
    //   [N, temp*ps*ps*C] — but in a DIFFERENT order than C*temp*ps*ps.
    //
    // We need to apply the same reordering.
    // Original: [C, temp, ps, ps] → moveaxis(0,3) → [temp, ps, ps, C]
    // So we reshape patchedPixels to [N, C, temp*ps*ps]
    //   → reshape to [N, C, temp, ps, ps]
    //   → transpose to [N, temp, ps, ps, C]
    //   → reshape back to [N, temp*ps*ps*C]
    final C = vCfg.inChannels; // 3
    final temp = vCfg.temporalPatchSize; // 2
    final ps = vCfg.patchSize; // 16
    final reordered5d = patchedPixels.reshape([nPatches, C, temp, ps, ps]);
    final transposed5d = reordered5d.transposeAxes([0, 2, 3, 4, 1]);
    reordered5d.close();
    final reorderedFlat = transposed5d.reshape([nPatches, temp * ps * ps * C]);
    transposed5d.close();

    // Now matmul: [N, patchVecSize] @ [hiddenSize, patchVecSize].T
    //           = [N, hiddenSize]
    final patchMat = mx.matmul(reorderedFlat, weightFlat.transpose());
    reorderedFlat.close();
    weightFlat.close();
    var hidden = mx.add(patchMat, visionWeights.patchEmbedBias);
    patchMat.close();
    if (onDumpIntermediate != null) {
      MlxRuntime.evalAll([hidden]);
      onDumpIntermediate('patch_embed', hidden);
    }

    // 2. Position embedding (interpolated, in merge-grouped order)
    final posEmbed = _interpolateVisionPosEmbed(
      visionWeights.posEmbedWeight,
      gridH,
      gridW,
      vCfg,
      hidden.dtype,
    );
    final withPos = mx.add(hidden, posEmbed);
    hidden.close();
    posEmbed.close();
    hidden = withPos;
    MlxRuntime.evalAll([hidden]);
    if (onDumpIntermediate != null) {
      onDumpIntermediate('patch_plus_pos', hidden);
    }
    onStage?.call('encodeImage: embeddings ready shape=${hidden.shape}');

    // 3. Build vision rotary position embeddings (merge-grouped order)
    final rotaryPosEmb = _buildVisionRotaryPosEmb(
      gridH,
      gridW,
      vCfg,
      MlxDType.MLX_FLOAT32,
    );

    // 4. ViT transformer blocks
    for (var i = 0; i < visionWeights.blocks.length; i++) {
      hidden = _visionBlock(
        visionWeights.blocks[i],
        hidden,
        vCfg,
        rotaryPosEmb,
      );
      MlxRuntime.evalAll([hidden]);
      if ((i + 1) % 3 == 0 || i + 1 == visionWeights.blocks.length) {
        onStage?.call(
          'encodeImage: vision layer ${i + 1}/${visionWeights.blocks.length}',
        );
      }
    }
    rotaryPosEmb.close();
    if (onDumpIntermediate != null) {
      MlxRuntime.evalAll([hidden]);
      onDumpIntermediate('after_blocks', hidden);
    }

    // 5. Spatial-merge projector
    final merged = _spatialMergeProject(
      hidden,
      gridH,
      gridW,
      vCfg,
      visionWeights.merger,
    );
    hidden.close();
    if (onDumpIntermediate != null) {
      MlxRuntime.evalAll([merged]);
      onDumpIntermediate('vision_output', merged);
    }
    onStage?.call('encodeImage: projector done shape=${merged.shape}');
    return (hidden: merged, gridHeight: gridH, gridWidth: gridW);
  }

  // -----------------------------------------------------------------------
  // Position embedding interpolation
  // -----------------------------------------------------------------------

  /// Interpolate position embeddings then reorder to merge-grouped order.
  ///
  /// Python uses `np.linspace(0, baseGrid-1, gridDim)` for coordinate
  /// computation, followed by bilinear interpolation, then a merge-size
  /// reordering: `reshape(mergedH, merge, mergedW, merge, hidden)` →
  /// `transpose(0, 2, 1, 3, 4)` → `reshape(N, hidden)`.
  MlxArray _interpolateVisionPosEmbed(
    MlxArray posEmbedWeight,
    int gridH,
    int gridW,
    Qwen3_5VisionConfig vCfg,
    MlxDType dtype,
  ) {
    final hiddenSize = vCfg.hiddenSize;
    final m = vCfg.spatialMergeSize;

    // posEmbedWeight shape: [numPositions, hiddenSize]
    final base = posEmbedWeight.toFloat32List();
    final baseGrid = math.sqrt(base.length / hiddenSize).round();

    // Step 1: Bilinear interpolation using linspace coordinates.
    // Python: np.linspace(0, baseGrid-1, gridDim)
    final rowCoords = List<double>.generate(
      gridH,
      (y) => gridH == 1 ? 0.0 : y * (baseGrid - 1) / (gridH - 1),
    );
    final colCoords = List<double>.generate(
      gridW,
      (x) => gridW == 1 ? 0.0 : x * (baseGrid - 1) / (gridW - 1),
    );

    // Precompute floor/ceil/weight for rows and cols.
    final rowFloor = List<int>.filled(gridH, 0);
    final rowCeil = List<int>.filled(gridH, 0);
    final rowFrac = List<double>.filled(gridH, 0);
    for (var y = 0; y < gridH; y++) {
      final ry = rowCoords[y];
      rowFloor[y] = ry.floor().clamp(0, baseGrid - 1);
      rowCeil[y] = (rowFloor[y] + 1).clamp(0, baseGrid - 1);
      rowFrac[y] = ry - rowFloor[y];
    }

    final colFloor = List<int>.filled(gridW, 0);
    final colCeil = List<int>.filled(gridW, 0);
    final colFrac = List<double>.filled(gridW, 0);
    for (var x = 0; x < gridW; x++) {
      final rx = colCoords[x];
      colFloor[x] = rx.floor().clamp(0, baseGrid - 1);
      colCeil[x] = (colFloor[x] + 1).clamp(0, baseGrid - 1);
      colFrac[x] = rx - colFloor[x];
    }

    // Interpolate in row-major order: [gridH * gridW, hiddenSize].
    final rowMajor = Float32List(gridH * gridW * hiddenSize);
    for (var y = 0; y < gridH; y++) {
      final rf = rowFrac[y];
      final oneMinusRf = 1.0 - rf;
      for (var x = 0; x < gridW; x++) {
        final cf = colFrac[x];
        final oneMinusCf = 1.0 - cf;
        final tl = (rowFloor[y] * baseGrid + colFloor[x]) * hiddenSize;
        final tr = (rowFloor[y] * baseGrid + colCeil[x]) * hiddenSize;
        final bl = (rowCeil[y] * baseGrid + colFloor[x]) * hiddenSize;
        final br = (rowCeil[y] * baseGrid + colCeil[x]) * hiddenSize;
        final target = (y * gridW + x) * hiddenSize;
        for (var c = 0; c < hiddenSize; c++) {
          rowMajor[target + c] =
              (oneMinusRf * oneMinusCf * base[tl + c]) +
              (oneMinusRf * cf * base[tr + c]) +
              (rf * oneMinusCf * base[bl + c]) +
              (rf * cf * base[br + c]);
        }
      }
    }

    // Step 2: Reorder from row-major to merge-grouped order.
    // Python: reshape(mergedH, merge, mergedW, merge, hidden)
    //       → transpose(0, 2, 1, 3, 4)
    //       → reshape(N, hidden)
    final mergedH = gridH ~/ m;
    final mergedW = gridW ~/ m;
    final result = Float32List(gridH * gridW * hiddenSize);
    var dstIdx = 0;
    for (var mh = 0; mh < mergedH; mh++) {
      for (var mw = 0; mw < mergedW; mw++) {
        for (var sh = 0; sh < m; sh++) {
          for (var sw = 0; sw < m; sw++) {
            final row = mh * m + sh;
            final col = mw * m + sw;
            final srcOff = (row * gridW + col) * hiddenSize;
            final dstOff = dstIdx * hiddenSize;
            for (var c = 0; c < hiddenSize; c++) {
              result[dstOff + c] = rowMajor[srcOff + c];
            }
            dstIdx++;
          }
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

  // -----------------------------------------------------------------------
  // Vision rotary position embedding
  // -----------------------------------------------------------------------

  /// Build vision rotary position embeddings in **merge-grouped** order.
  ///
  /// Patches are iterated as:
  ///   for each merged block (mh, mw):
  ///     for each sub-patch (sh, sw) within the merge window:
  ///       row = mh * merge + sh
  ///       col = mw * merge + sw
  ///
  /// Returns `[seqLen, rotaryDim]` where `rotaryDim = headDim / 2`.
  /// Each row contains `[row_freqs..., col_freqs...]`.
  MlxArray _buildVisionRotaryPosEmb(
    int gridH,
    int gridW,
    Qwen3_5VisionConfig vCfg,
    MlxDType dtype,
  ) {
    final seqLen = gridH * gridW;
    final m = vCfg.spatialMergeSize;
    final rotaryDim = vCfg.headDim ~/ 2;
    final invFreqCount = rotaryDim ~/ 2;
    final mergedH = gridH ~/ m;
    final mergedW = gridW ~/ m;

    // Precompute inverse frequencies.
    final invFreq = Float32List(invFreqCount);
    for (var i = 0; i < invFreqCount; i++) {
      final exponent = (2 * i) / rotaryDim;
      invFreq[i] = 1.0 / math.pow(10000.0, exponent).toDouble();
    }

    // Fill frequency table in merge-grouped order.
    final freqs = Float32List(seqLen * rotaryDim);
    var idx = 0;
    for (var mh = 0; mh < mergedH; mh++) {
      for (var mw = 0; mw < mergedW; mw++) {
        for (var sh = 0; sh < m; sh++) {
          for (var sw = 0; sw < m; sw++) {
            final row = mh * m + sh;
            final col = mw * m + sw;
            final target = idx * rotaryDim;
            for (var i = 0; i < invFreqCount; i++) {
              freqs[target + i] = row * invFreq[i];
              freqs[target + invFreqCount + i] = col * invFreq[i];
            }
            idx++;
          }
        }
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
    _Qwen35VisionBlockWeights block,
    MlxArray input,
    Qwen3_5VisionConfig vCfg,
    MlxArray rotaryPosEmb,
  ) {
    // ── Pre-norm 1 ──
    final norm1 = _visionLayerNorm(
      input,
      weight: block.norm1Weight,
      bias: block.norm1Bias,
    );

    // ── Self-attention (fused QKV) ──
    final attnOut = _visionAttention(block, norm1, vCfg, rotaryPosEmb);
    norm1.close();

    // ── Residual 1 ──
    final h = mx.add(input, attnOut);
    attnOut.close();
    input.close();

    // ── Pre-norm 2 ──
    final norm2 = _visionLayerNorm(
      h,
      weight: block.norm2Weight,
      bias: block.norm2Bias,
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
  // Vision self-attention (fused QKV, dense, no quantization)
  // -----------------------------------------------------------------------

  MlxArray _visionAttention(
    _Qwen35VisionBlockWeights block,
    MlxArray input,
    Qwen3_5VisionConfig vCfg,
    MlxArray rotaryPosEmb,
  ) {
    final seqLen = input.shape[0];
    final numHeads = vCfg.numHeads;
    final headDim = vCfg.headDim;

    // Fused QKV projection: input [seq, hidden] → [seq, 3*hidden]
    final qkvMat = mx.matmul(input, block.attnQkvWeight.transpose());
    final qkv = mx.add(qkvMat, block.attnQkvBias);
    qkvMat.close();

    final qkv4d = qkv.reshape([seqLen, 3, numHeads, headDim]).transposeAxes([
      1,
      0,
      2,
      3,
    ]);
    qkv.close();

    // Split into Q, K, V: each [1, seq, heads, headDim]
    final q = qkv4d.slice(
      start: [0, 0, 0, 0],
      stop: [1, seqLen, numHeads, headDim],
    );
    final k = qkv4d.slice(
      start: [1, 0, 0, 0],
      stop: [2, seqLen, numHeads, headDim],
    );
    final v = qkv4d.slice(
      start: [2, 0, 0, 0],
      stop: [3, seqLen, numHeads, headDim],
    );
    qkv4d.close();

    // Apply vision rotary embedding
    final qRot = _applyVisionRotary(q, rotaryPosEmb);
    final kRot = _applyVisionRotary(k, rotaryPosEmb);
    q.close();
    k.close();

    // Transpose to [1, heads, seq, headDim] for attention
    final qForAttn = qRot.transposeAxes([0, 2, 1, 3]);
    final kForAttn = kRot.transposeAxes([0, 2, 1, 3]);
    final vForAttn = v.transposeAxes([0, 2, 1, 3]);
    qRot.close();
    kRot.close();

    // Scaled dot-product attention (no causal mask for vision)
    final mask = MlxArray.zeros([1, seqLen, seqLen], dtype: input.dtype);
    final attn = mx.fast.scaledDotProductAttention(
      qForAttn,
      kForAttn,
      vForAttn,
      scale: 1.0 / math.sqrt(headDim.toDouble()),
      mask: mask,
    );
    mask.close();
    qForAttn.close();
    kForAttn.close();
    vForAttn.close();
    v.close();

    // Merge heads: [1, heads, seq, headDim] → [seq, hidden]
    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      numHeads * headDim,
    ]);
    attn.close();

    // Output projection: [seq, hidden] → [seq, hidden]
    final outMat = mx.matmul(merged, block.attnProjWeight.transpose());
    final out = mx.add(outMat, block.attnProjBias);
    outMat.close();
    merged.close();
    return out;
  }

  // -----------------------------------------------------------------------
  // Vision MLP (fc1 → GELU → fc2)
  // -----------------------------------------------------------------------

  MlxArray _visionMlp(_Qwen35VisionBlockWeights block, MlxArray input) {
    final fc1Mat = mx.matmul(input, block.mlpFc1Weight.transpose());
    final h = mx.add(fc1Mat, block.mlpFc1Bias);
    fc1Mat.close();
    final activated = _geluVision(h);
    h.close();
    final fc2Mat = mx.matmul(activated, block.mlpFc2Weight.transpose());
    final out = mx.add(fc2Mat, block.mlpFc2Bias);
    fc2Mat.close();
    activated.close();
    return out;
  }

  // -----------------------------------------------------------------------
  // Spatial-merge projector
  // -----------------------------------------------------------------------

  /// Performs 2×2 spatial merging then projects to LM hidden size.
  ///
  /// Input `hidden`: `[gridH*gridW, visionHidden]` in **merge-grouped** order
  /// (i.e. every 4 consecutive rows already belong to the same 2×2 merge block).
  /// Output:         `[mergedTokens, lmHidden]`
  ///
  /// Python reference (use_postshuffle_norm=False):
  ///   x = self.norm(x).reshape(-1, hidden_size * merge^2)
  ///   x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
  MlxArray _spatialMergeProject(
    MlxArray hidden,
    int gridH,
    int gridW,
    Qwen3_5VisionConfig vCfg,
    _Qwen35MergerWeights merger,
  ) {
    final m = vCfg.spatialMergeSize; // 2
    final mergedTokens = (gridH ~/ m) * (gridW ~/ m);

    // Pre-norm (standard LayerNorm on [N, visionHidden])
    final normed = _visionLayerNorm(
      hidden,
      weight: merger.normWeight,
      bias: merger.normBias,
    );

    // Since data is already in merge-grouped order, a simple reshape
    // groups 4 consecutive patches (one 2×2 block) into one row:
    //   [N_patches, visionHidden] → [mergedTokens, merge^2 * visionHidden]
    final flat = normed.reshape([mergedTokens, m * m * vCfg.hiddenSize]);
    normed.close();

    // linear_fc1 → GELU → linear_fc2
    final fc1Mat = mx.matmul(flat, merger.fc1Weight.transpose());
    final h = mx.add(fc1Mat, merger.fc1Bias);
    fc1Mat.close();
    flat.close();
    final activated = _geluVision(h);
    h.close();
    final fc2Mat = mx.matmul(activated, merger.fc2Weight.transpose());
    final out = mx.add(fc2Mat, merger.fc2Bias);
    fc2Mat.close();
    activated.close();
    MlxRuntime.evalAll([out]);
    return out; // [mergedTokens, outHiddenSize]
  }

  // -----------------------------------------------------------------------
  // GELU activation (precise, matching PyTorch gelu_pytorch_tanh)
  //
  //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  // -----------------------------------------------------------------------

  MlxArray _geluVision(MlxArray x) {
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
  }) {
    return mx.fast.layerNorm(input, weight: weight, bias: bias, eps: 1e-6);
  }

  // -----------------------------------------------------------------------
  // Multimodal embedding construction
  // -----------------------------------------------------------------------

  /// Build a multimodal embedding tensor by replacing image token placeholders
  /// with encoded vision hidden states.
  ///
  /// [tokenIds] are token IDs with `imageTokenId` placeholders.
  /// [imageHidden] are encoded vision features `[numImageTokens, hiddenSize]`.
  MlxArray buildMultimodalEmbedding(List<int> tokenIds, MlxArray imageHidden) {
    final totalLen = tokenIds.length;
    final imageTokenId = config.imageTokenId!;

    // Get text embeddings for all tokens
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, totalLen]);
    final textEmbed = _embed(ids);
    ids.close();

    // Find image token ranges and replace
    var imgIdx = 0;
    final segments = <MlxArray>[];
    var segStart = 0;

    for (var i = 0; i < totalLen; i++) {
      if (tokenIds[i] == imageTokenId) {
        // Emit preceding text segment
        if (i > segStart) {
          segments.add(
            textEmbed.slice(
              start: [0, segStart, 0],
              stop: [1, i, config.hiddenSize],
            ),
          );
        }
        // Find contiguous run of image tokens
        var runEnd = i;
        while (runEnd < totalLen && tokenIds[runEnd] == imageTokenId) {
          runEnd++;
        }
        final runLen = runEnd - i;
        // Insert image hidden states
        segments.add(
          imageHidden
              .slice(
                start: [imgIdx, 0],
                stop: [imgIdx + runLen, config.hiddenSize],
              )
              .reshape([1, runLen, config.hiddenSize]),
        );
        imgIdx += runLen;
        segStart = runEnd;
        i = runEnd - 1; // loop will increment
      }
    }

    // Trailing text segment
    if (segStart < totalLen) {
      segments.add(
        textEmbed.slice(
          start: [0, segStart, 0],
          stop: [1, totalLen, config.hiddenSize],
        ),
      );
    }

    final result = segments.length == 1
        ? segments.first
        : mx.concatenate(segments, axis: 1);
    for (final seg in segments) {
      if (seg != result) seg.close();
    }
    textEmbed.close();
    return result;
  }

  // -----------------------------------------------------------------------
  // Multimodal M-RoPE position IDs
  // -----------------------------------------------------------------------

  /// Build M-RoPE position IDs for a multimodal prompt.
  ///
  /// Text tokens use sequential positions. Image tokens use 2D spatial
  /// positions (row/col within the merged grid).
  ({MlxArray ids, int nextTextPosition}) multimodalPositionIds(
    List<int> tokenIds,
    int gridH,
    int gridW,
  ) {
    final imageTokenId = config.imageTokenId!;
    final mergeSize = config.visionConfig!.spatialMergeSize;
    final mergedH = gridH ~/ mergeSize;
    final mergedW = gridW ~/ mergeSize;
    final totalLen = tokenIds.length;

    final imageStart = tokenIds.indexOf(imageTokenId);
    if (imageStart < 0) {
      // No image tokens — simple text-only positions
      final flat = <int>[
        for (var i = 0; i < totalLen; i++) i,
        for (var i = 0; i < totalLen; i++) i,
        for (var i = 0; i < totalLen; i++) i,
      ];
      return (
        ids: MlxArray.fromInt32List(flat, shape: [3, 1, totalLen]),
        nextTextPosition: totalLen,
      );
    }

    var imageEnd = imageStart;
    while (imageEnd < totalLen && tokenIds[imageEnd] == imageTokenId) {
      imageEnd++;
    }

    final temporal = List<int>.filled(totalLen, 0);
    final height = List<int>.filled(totalLen, 0);
    final width = List<int>.filled(totalLen, 0);

    // Leading text tokens
    for (var i = 0; i < imageStart; i++) {
      temporal[i] = i;
      height[i] = i;
      width[i] = i;
    }

    // Image tokens: 2D spatial positions
    final imageBase = imageStart;
    final imageTokenCount = imageEnd - imageStart;
    for (var i = 0; i < imageTokenCount; i++) {
      final tokenIndex = imageStart + i;
      temporal[tokenIndex] = imageBase;
      height[tokenIndex] = imageBase + (i ~/ mergedW);
      width[tokenIndex] = imageBase + (i % mergedW);
    }

    // Trailing text tokens
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

  // -----------------------------------------------------------------------
  // Vision weight loading
  // -----------------------------------------------------------------------

  /// Load vision encoder weights from the tensor map.
  ///
  /// Returns null if no vision weights are found (text-only model).
  static _Qwen35VisionWeights? loadVisionWeights(
    Map<String, MlxArray> tensors,
  ) {
    // Check if vision weights exist
    if (!tensors.containsKey('vision_tower.patch_embed.proj.weight')) {
      return null;
    }

    const vp = 'vision_tower.';

    // Patch embedding (Conv2D)
    final patchWeight = tensors['${vp}patch_embed.proj.weight']!;
    final patchBias = tensors['${vp}patch_embed.proj.bias']!;

    // Position embedding
    final posEmbed = tensors['${vp}pos_embed.weight']!;

    // ViT blocks — detect depth by scanning keys
    var depth = 0;
    while (tensors.containsKey('${vp}blocks.$depth.norm1.weight')) {
      depth++;
    }

    final blocks = List<_Qwen35VisionBlockWeights>.generate(depth, (i) {
      final bp = '${vp}blocks.$i.';
      return _Qwen35VisionBlockWeights(
        norm1Weight: tensors['${bp}norm1.weight']!,
        norm1Bias: tensors['${bp}norm1.bias']!,
        norm2Weight: tensors['${bp}norm2.weight']!,
        norm2Bias: tensors['${bp}norm2.bias']!,
        attnQkvWeight: tensors['${bp}attn.qkv.weight']!,
        attnQkvBias: tensors['${bp}attn.qkv.bias']!,
        attnProjWeight: tensors['${bp}attn.proj.weight']!,
        attnProjBias: tensors['${bp}attn.proj.bias']!,
        mlpFc1Weight: tensors['${bp}mlp.linear_fc1.weight']!,
        mlpFc1Bias: tensors['${bp}mlp.linear_fc1.bias']!,
        mlpFc2Weight: tensors['${bp}mlp.linear_fc2.weight']!,
        mlpFc2Bias: tensors['${bp}mlp.linear_fc2.bias']!,
      );
    });

    // Spatial-merge projector (merger)
    final merger = _Qwen35MergerWeights(
      normWeight: tensors['${vp}merger.norm.weight']!,
      normBias: tensors['${vp}merger.norm.bias']!,
      fc1Weight: tensors['${vp}merger.linear_fc1.weight']!,
      fc1Bias: tensors['${vp}merger.linear_fc1.bias']!,
      fc2Weight: tensors['${vp}merger.linear_fc2.weight']!,
      fc2Bias: tensors['${vp}merger.linear_fc2.bias']!,
    );

    return _Qwen35VisionWeights(
      patchEmbedWeight: patchWeight,
      patchEmbedBias: patchBias,
      posEmbedWeight: posEmbed,
      blocks: blocks,
      merger: merger,
    );
  }
}
