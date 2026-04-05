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
    var qRope = rope.q;
    var kRope = rope.k;
    var vAttn = v;

    // Update KV cache
    if (cache != null) {
      final fetched = cache.updateAndFetch(kRope, vAttn);
      kRope = fetched.$1;
      vAttn = fetched.$2;
    }

    // GQA: repeat KV heads to match Q heads
    final repeatKv = numHeads ~/ numKvHeads;
    var kForAttn = kRope;
    var vForAttn = vAttn;
    if (repeatKv > 1) {
      kForAttn = _repeatKvHeads(
        kRope,
        numHeads: numHeads,
        numKvHeads: numKvHeads,
        seqLen: kRope.shape[2],
        headDim: headDim,
      );
      vForAttn = _repeatKvHeads(
        vAttn,
        numHeads: numHeads,
        numKvHeads: numKvHeads,
        seqLen: vAttn.shape[2],
        headDim: headDim,
      );
      if (cache == null) {
        kRope.close();
        vAttn.close();
      }
    }

    // Scaled dot-product attention
    final attnOut = mx.fast.scaledDotProductAttention(
      qRope,
      kForAttn,
      vForAttn,
      scale: 1.0 / math.sqrt(headDim.toDouble()),
      maskMode: cache != null && seqLen == 1 ? '' : 'causal',
    );
    qRope.close();
    if (repeatKv > 1 || cache == null) {
      kForAttn.close();
      vForAttn.close();
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
  // GQA head repetition
  // -----------------------------------------------------------------------

  MlxArray _repeatKvHeads(
    MlxArray tensor, {
    required int numHeads,
    required int numKvHeads,
    required int seqLen,
    required int headDim,
  }) {
    final repeat = numHeads ~/ numKvHeads;
    final expanded = tensor.expandDims(2);
    try {
      return expanded
          .broadcastTo([1, numKvHeads, repeat, seqLen, headDim])
          .reshape([1, numHeads, seqLen, headDim]);
    } finally {
      expanded.close();
    }
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
