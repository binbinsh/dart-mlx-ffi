part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// ERNIE-4.5 decoder layer forward pass
// ---------------------------------------------------------------------------

bool get _debugTurboAttentionTraceEnabled {
  final raw = Platform.environment['DART_INFERENCE_PADDLE_DEBUG_TURBO_ATTN'];
  return raw == '1' || raw == 'true';
}

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
    ({MlxArray cos, MlxArray sin})? positionEmbeddings,
    required int layerIndex,
    _LayerCache? cache,
  }) {
    // ── Pre-attention norm ──
    final norm1Trace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'norm1',
    );
    var norm1 = _lmRmsNormCompat(
      input,
      weight: layer.inputNorm,
      eps: config.rmsNormEps,
    );
    if (seqLen > 1 && norm1.dtype != input.dtype) {
      final cast = norm1.astype(input.dtype);
      norm1.close();
      norm1 = cast;
    }
    _endDecoderSubstepTrace(this, norm1Trace);

    // ── Self-attention ──
    final attnTrace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'attn',
    );
    final attnOut = _lmAttention(
      layer.attention,
      norm1,
      seqLen,
      positionIds,
      positionEmbeddings: positionEmbeddings,
      layerIndex: layerIndex,
      cache: cache,
    );
    _endDecoderSubstepTrace(this, attnTrace);
    norm1.close();

    // ── Residual 1 ──
    final res1Trace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'residual1',
    );
    final h = mx.add(input, attnOut);
    _endDecoderSubstepTrace(this, res1Trace);
    attnOut.close();

    // ── Post-attention norm ──
    final norm2Trace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'norm2',
    );
    var norm2 = _lmRmsNormCompat(
      h,
      weight: layer.postNorm,
      eps: config.rmsNormEps,
    );
    if (seqLen > 1 && norm2.dtype != h.dtype) {
      final cast = norm2.astype(h.dtype);
      norm2.close();
      norm2 = cast;
    }
    _endDecoderSubstepTrace(this, norm2Trace);

    // ── MLP ──
    final mlpTrace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'mlp',
    );
    final mlpOut = _lmMlp(layer.mlp, norm2, seqLen);
    _endDecoderSubstepTrace(this, mlpTrace);
    norm2.close();

    // ── Residual 2 ──
    final res2Trace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'residual2',
    );
    final out = mx.add(h, mlpOut);
    _endDecoderSubstepTrace(this, res2Trace);
    if (seqLen == 1 && config.enableDecoderHiddenDetachForCurrentPlatform) {
      final detachEvalTrace = _beginDecoderSubstepTrace(
        this,
        layerIndex,
        seqLen,
        cache,
        'detach_eval_out',
      );
      MlxRuntime.evalAll([out]);
      _endDecoderSubstepTrace(this, detachEvalTrace);
      final closeMlpTrace = _beginDecoderSubstepTrace(
        this,
        layerIndex,
        seqLen,
        cache,
        'detach_close_mlpOut',
      );
      mlpOut.close();
      _endDecoderSubstepTrace(this, closeMlpTrace);
      final closeHTrace = _beginDecoderSubstepTrace(
        this,
        layerIndex,
        seqLen,
        cache,
        'detach_close_h',
      );
      h.close();
      _endDecoderSubstepTrace(this, closeHTrace);
      return out;
    }
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
    ({MlxArray cos, MlxArray sin})? positionEmbeddings,
    required int layerIndex,
    _LayerCache? cache,
  }) {
    final numHeads = config.numAttentionHeads;
    final numKvHeads = config.numKeyValueHeads;
    final headDim = config.headDim;

    // Project Q, K, V (each is separate linear)
    final qkvTrace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'attn_qkv',
    );
    final flat = input.reshape([seqLen, config.hiddenSize]);
    final qWidth = numHeads * headDim;
    final kvWidth = numKvHeads * headDim;
    late final MlxArray q;
    late final MlxArray k;
    late final MlxArray v;
    if (attn.qkvProj case final fused?) {
      final qkv = fused.apply(flat);
      final q2d = qkv.slice(start: [0, 0], stop: [seqLen, qWidth]);
      final k2d = qkv.slice(
        start: [0, qWidth],
        stop: [seqLen, qWidth + kvWidth],
      );
      final v2d = qkv.slice(
        start: [0, qWidth + kvWidth],
        stop: [seqLen, qWidth + kvWidth * 2],
      );
      qkv.close();
      q = q2d.reshape([1, seqLen, numHeads, headDim]).transposeAxes([
        0,
        2,
        1,
        3,
      ]);
      k = k2d.reshape([1, seqLen, numKvHeads, headDim]).transposeAxes([
        0,
        2,
        1,
        3,
      ]);
      v = v2d.reshape([1, seqLen, numKvHeads, headDim]).transposeAxes([
        0,
        2,
        1,
        3,
      ]);
      q2d.close();
      k2d.close();
      v2d.close();
    } else {
      q = attn.qProj!
          .apply(flat)
          .reshape([1, seqLen, numHeads, headDim])
          .transposeAxes([0, 2, 1, 3]);
      k = attn.kProj!
          .apply(flat)
          .reshape([1, seqLen, numKvHeads, headDim])
          .transposeAxes([0, 2, 1, 3]);
      v = attn.vProj!
          .apply(flat)
          .reshape([1, seqLen, numKvHeads, headDim])
          .transposeAxes([0, 2, 1, 3]);
    }
    flat.close();
    _endDecoderSubstepTrace(this, qkvTrace);

    // Apply M-RoPE
    final ropeTrace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'attn_rope',
    );
    final rope = positionEmbeddings == null
        ? _applyMrope(q, k, positionIds)
        : _applyMropeWithCosSin(
            q,
            k,
            positionEmbeddings.cos,
            positionEmbeddings.sin,
          );
    q.close();
    k.close();
    final qRope = rope.q;
    final kRope = rope.k;
    _endDecoderSubstepTrace(this, ropeTrace);

    // Update KV cache
    // When using the pre-allocated cache, updateAndFetch returns *new slice
    // views* into the buffer that the caller must close after use.
    final mask = cache != null && seqLen == 1 ? null : 'causal';
    late MlxArray attnOut;
    if (cache case final _TurboQuantKvCache turbo) {
      final cacheTrace = _beginDecoderSubstepTrace(
        this,
        layerIndex,
        seqLen,
        cache,
        'attn_cache',
      );
      turbo.update(kRope, v);
      if (seqLen == 1 && _shouldEvalDecodeCacheState(turbo)) {
        turbo.evalState();
      }
      _endDecoderSubstepTrace(this, cacheTrace);
      try {
        final sdpaTrace = _beginDecoderSubstepTrace(
          this,
          layerIndex,
          seqLen,
          cache,
          'attn_sdpa',
        );
        attnOut = turbo.quantizedAttention(
          qRope,
          scale: 1.0 / math.sqrt(headDim.toDouble()),
          mask: mask,
        );
        _endDecoderSubstepTrace(this, sdpaTrace);
      } finally {
        qRope.close();
      }
    } else if (cache case final _QuantizedKvCache quantized) {
      if (seqLen == 1) {
        final cacheTrace = _beginDecoderSubstepTrace(
          this,
          layerIndex,
          seqLen,
          cache,
          'attn_cache',
        );
        quantized.update(kRope, v);
        if (_shouldEvalDecodeCacheState(quantized)) {
          quantized.evalState();
        }
        _endDecoderSubstepTrace(this, cacheTrace);
        try {
          final sdpaTrace = _beginDecoderSubstepTrace(
            this,
            layerIndex,
            seqLen,
            cache,
            'attn_sdpa',
          );
          attnOut = _quantizedScaledDotProductAttention(
            qRope,
            quantized.borrowedKeys,
            quantized.borrowedValues,
            scale: 1.0 / math.sqrt(headDim.toDouble()),
            mask: null,
            validKeyLen: quantized.offset,
            groupSize: quantized.groupSize,
            bits: quantized.bits,
          );
          _endDecoderSubstepTrace(this, sdpaTrace);
        } finally {
          qRope.close();
        }
      } else {
        final cacheTrace = _beginDecoderSubstepTrace(
          this,
          layerIndex,
          seqLen,
          cache,
          'attn_cache',
        );
        final fetched = quantized.updateAndFetch(kRope, v);
        _endDecoderSubstepTrace(this, cacheTrace);
        try {
          final sdpaTrace = _beginDecoderSubstepTrace(
            this,
            layerIndex,
            seqLen,
            cache,
            'attn_sdpa',
          );
          attnOut = _quantizedScaledDotProductAttention(
            qRope,
            fetched.keys,
            fetched.values,
            scale: 1.0 / math.sqrt(headDim.toDouble()),
            mask: mask,
            groupSize: quantized.groupSize,
            bits: quantized.bits,
          );
          _endDecoderSubstepTrace(this, sdpaTrace);
        } finally {
          fetched.close();
          qRope.close();
        }
      }
    } else {
      var kAttn = kRope;
      var vAttn = v;
      final ownsKV = cache != null;
      if (cache case final _KvCache dense) {
        final cacheTrace = _beginDecoderSubstepTrace(
          this,
          layerIndex,
          seqLen,
          cache,
          'attn_cache',
        );
        final fetched = dense.updateAndFetch(kRope, vAttn);
        if (seqLen == 1 && _shouldEvalDecodeCacheState(dense)) {
          dense.evalState();
        }
        kAttn = fetched.$1;
        vAttn = fetched.$2;
        _endDecoderSubstepTrace(this, cacheTrace);
      }
      final sdpaTrace = _beginDecoderSubstepTrace(
        this,
        layerIndex,
        seqLen,
        cache,
        'attn_sdpa',
      );
      attnOut = mx.fast.scaledDotProductAttention(
        qRope,
        kAttn,
        vAttn,
        scale: 1.0 / math.sqrt(headDim.toDouble()),
        maskMode: mask == null ? '' : 'causal',
      );
      _endDecoderSubstepTrace(this, sdpaTrace);
      qRope.close();
      if (ownsKV) {
        kAttn.close();
        vAttn.close();
      } else {
        kRope.close();
        v.close();
      }
    }

    if (seqLen == 1 &&
        cache is _TurboQuantKvCache &&
        layerIndex == 0 &&
        (config.enableDecoderTailTraceForCurrentPlatform ||
            _debugTurboAttentionTraceEnabled)) {
      final line =
          '[pocr][turbo-attn] '
          'layer=${layerIndex + 1} '
          'path=${_turboLastAttentionPath ?? "-"} '
          'out=${attnOut.dtype} '
          'input=${input.dtype}';
      final sink = PaddleOcrVlDebugOverrides.traceSink;
      if (sink != null) {
        sink(line);
      } else {
        stderr.writeln(line);
      }
    }

    if (seqLen == 1 && attnOut.dtype != input.dtype) {
      final cast = attnOut.astype(input.dtype);
      attnOut.close();
      attnOut = cast;
    }

    // Merge heads and project output
    final merged = attnOut.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      numHeads * headDim,
    ]);
    attnOut.close();

    final projTrace = _beginDecoderSubstepTrace(
      this,
      layerIndex,
      seqLen,
      cache,
      'attn_proj',
    );
    final out = attn.oProj.apply(merged);
    _endDecoderSubstepTrace(this, projTrace);
    merged.close();
    return out.reshape([1, seqLen, config.hiddenSize]);
  }

  // -----------------------------------------------------------------------
  // SiLU-gated MLP
  // -----------------------------------------------------------------------

  MlxArray _lmMlp(_LmMlpWeights mlp, MlxArray input, int seqLen) {
    final flat = input.reshape([seqLen, config.hiddenSize]);
    late final MlxArray gate;
    late final MlxArray up;
    if (mlp.gateUpProj case final fused?) {
      final gateUp = fused.apply(flat);
      final width = gateUp.shape[1] ~/ 2;
      gate = gateUp.slice(start: [0, 0], stop: [seqLen, width]);
      up = gateUp.slice(start: [0, width], stop: [seqLen, width * 2]);
      gateUp.close();
    } else {
      gate = mlp.gateProj!.apply(flat);
      up = mlp.upProj!.apply(flat);
    }
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
