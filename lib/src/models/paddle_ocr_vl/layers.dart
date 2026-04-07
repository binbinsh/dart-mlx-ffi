part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// ERNIE-4.5 decoder layer forward pass
// ---------------------------------------------------------------------------

extension PaddleOcrVlLayers on PaddleOcrVlRunner {
  /// Run one decoder layer with optional KV cache.
  ///
  /// [input] shape: `[1, seqLen, hiddenSize]`
  /// [positionIds] shape: `[3, 1, seqLen]`
  MlxArray _decoderLayer(
    _LmLayerWeights layer,
    MlxArray input,
    int seqLen,
    MlxArray positionIds, {
    _KvCache? cache,
  }) {
    // ── Pre-attention norm ──
    final norm1 = mx.fast.rmsNorm(
      input,
      weight: layer.inputNorm,
      eps: config.rmsNormEps,
    );

    // ── Self-attention ──
    final attnOut = _lmAttention(
      layer.attention,
      norm1,
      seqLen,
      positionIds,
      cache: cache,
    );
    norm1.close();

    // ── Residual 1 ──
    final h = mx.add(input, attnOut);
    attnOut.close();

    // ── Post-attention norm ──
    final norm2 = mx.fast.rmsNorm(
      h,
      weight: layer.postNorm,
      eps: config.rmsNormEps,
    );

    // ── MLP ──
    final mlpOut = _lmMlp(layer.mlp, norm2, seqLen);
    norm2.close();

    // ── Residual 2 ──
    final out = mx.add(h, mlpOut);
    mlpOut.close();
    h.close();
    return out;
  }

  // -----------------------------------------------------------------------
  // Language model attention (GQA with M-RoPE)
  // -----------------------------------------------------------------------

  MlxArray _lmAttention(
    _LmAttentionWeights attn,
    MlxArray input,
    int seqLen,
    MlxArray positionIds, {
    _KvCache? cache,
  }) {
    final numHeads = config.numAttentionHeads;
    final numKvHeads = config.numKeyValueHeads;
    final headDim = config.headDim;

    // Project Q, K, V (each is separate linear)
    final flat = input.reshape([seqLen, config.hiddenSize]);
    final q = attn.qProj
        .apply(flat)
        .reshape([1, seqLen, numHeads, headDim])
        .transposeAxes([0, 2, 1, 3]);
    final k = attn.kProj
        .apply(flat)
        .reshape([1, seqLen, numKvHeads, headDim])
        .transposeAxes([0, 2, 1, 3]);
    final v = attn.vProj
        .apply(flat)
        .reshape([1, seqLen, numKvHeads, headDim])
        .transposeAxes([0, 2, 1, 3]);
    flat.close();

    // Apply M-RoPE
    final rope = _applyMrope(q, k, positionIds);
    q.close();
    k.close();
    final qRope = rope.q;
    final kRope = rope.k;

    // Update KV cache
    // When using the pre-allocated cache, updateAndFetch returns *new slice
    // views* into the buffer that the caller must close after use.
    var kAttn = kRope;
    var vAttn = v;
    final ownsKV = cache != null; // true → we must close kAttn/vAttn
    if (cache != null) {
      final fetched = cache.updateAndFetch(kRope, vAttn);
      kAttn = fetched.$1;
      vAttn = fetched.$2;
      // kRope and v are now closed by updateAndFetch — don't double-close.
    }

    // Scaled dot-product attention. MLX supports grouped-query attention
    // natively, so keep the 2 KV heads instead of manually repeating them to
    // 16 heads.
    final attnOut = mx.fast.scaledDotProductAttention(
      qRope,
      kAttn,
      vAttn,
      scale: 1.0 / math.sqrt(headDim.toDouble()),
      maskMode: cache != null && seqLen == 1 ? '' : 'causal',
    );
    qRope.close();
    if (ownsKV) {
      kAttn.close();
      vAttn.close();
    } else {
      kRope.close();
      v.close();
    }

    // Merge heads and project output
    final merged = attnOut.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      numHeads * headDim,
    ]);
    attnOut.close();

    final out = attn.oProj.apply(merged);
    merged.close();
    return out.reshape([1, seqLen, config.hiddenSize]);
  }

  // -----------------------------------------------------------------------
  // SiLU-gated MLP
  // -----------------------------------------------------------------------

  MlxArray _lmMlp(_LmMlpWeights mlp, MlxArray input, int seqLen) {
    final flat = input.reshape([seqLen, config.hiddenSize]);
    final gate = mlp.gateProj.apply(flat);
    final up = mlp.upProj.apply(flat);
    flat.close();

    // SiLU(gate) * up
    final sig = gate.sigmoid();
    final silu = gate * sig;
    sig.close();
    gate.close();
    final fused = silu * up;
    silu.close();
    up.close();

    final down = mlp.downProj.apply(fused);
    fused.close();
    return down.reshape([1, seqLen, config.hiddenSize]);
  }
}
