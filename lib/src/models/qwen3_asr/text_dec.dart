import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'config.dart';
import 'mrope.dart';

/// Qwen3 text decoder for Qwen3-ASR.
///
/// Standard Qwen3 transformer: RMSNorm → GQA self-attention (with q/k norm
/// and MRoPE) → RMSNorm → SwiGLU MLP. 28 layers, 16 heads, 8 KV heads,
/// head_dim=128.
///
/// Uses InterleavedMRoPE with sections [24, 20, 20] instead of standard RoPE.
/// Uses explicit additive causal masks instead of mx.fast.sdpa maskMode.
///
/// Text decoder weights are quantized 8-bit affine (group_size=64).
/// Audio encoder weights are dense (BF16) and handled separately.
final class Qwen3AsrTextDecoder {
  Qwen3AsrTextDecoder._(
    this.config,
    this._embedWeight,
    this._embedScales,
    this._embedBiases,
    this._lmHeadWeight,
    this._layers,
    this._finalNormW,
    this._mrope,
  );

  factory Qwen3AsrTextDecoder.load(
    Map<String, MlxArray> tensors,
    Qwen3AsrConfig config,
  ) {
    const p = 'model.';
    final layers = List<_DecoderLayer>.generate(
      config.textNumLayers,
      (i) => _DecoderLayer.load(tensors, '${p}layers.$i.', config),
    );

    final mrope = AsrMRoPE(
      headDim: config.textHeadDim,
      base: config.textRopeTheta,
      sections: config.textMropeSections,
    );

    final embedW = tensors['${p}embed_tokens.weight']!;
    final embedS = tensors['${p}embed_tokens.scales'];
    final embedB = tensors['${p}embed_tokens.biases'];

    // In the reference implementation, lm_head.weight is tied to
    // embed_tokens.weight *before* quantization. After quantization,
    // embed_tokens becomes QuantizedEmbedding (uint32) but lm_head retains
    // the original float weight. Since safetensors only stores the quantized
    // embed_tokens, we reconstruct the dense lm_head weight by dequantizing.
    // This matches: lm_head does dense matmul, not quantized matmul.
    MlxArray? lmHeadWeight;
    if (embedS != null) {
      lmHeadWeight = mx.quant.dequantize(
        MlxQuantizedMatrix(embedW, embedS, embedB),
        groupSize: config.quantGroupSize,
        bits: config.quantBits,
        mode: config.quantMode,
      );
      lmHeadWeight.eval();
    }

    return Qwen3AsrTextDecoder._(
      config,
      embedW,
      embedS,
      embedB,
      lmHeadWeight,
      layers,
      tensors['${p}norm.weight']!,
      mrope,
    );
  }

  final Qwen3AsrConfig config;
  final MlxArray _embedWeight;
  final MlxArray? _embedScales;
  final MlxArray? _embedBiases;

  /// Dense float32 lm_head weight, dequantized from embed_tokens.
  /// Null only when embed_tokens is not quantized (dense model).
  final MlxArray? _lmHeadWeight;
  final List<_DecoderLayer> _layers;
  final MlxArray _finalNormW;
  final AsrMRoPE _mrope;

  /// Embed token IDs → [1, seqLen, hiddenSize].
  MlxArray embed(List<int> tokenIds) {
    final ids = MlxArray.fromInt32List(tokenIds, shape: [tokenIds.length]);
    try {
      if (_embedScales != null) {
        final rowsW = _embedWeight.take(ids, axis: 0);
        final rowsS = _embedScales.take(ids, axis: 0);
        final rowsB = _embedBiases?.take(ids, axis: 0);
        try {
          final out = mx.quant.dequantize(
            MlxQuantizedMatrix(rowsW, rowsS, rowsB),
            groupSize: config.quantGroupSize,
            bits: config.quantBits,
            mode: config.quantMode,
          );
          return out.reshape([1, tokenIds.length, config.textHiddenSize]);
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      final out = _embedWeight.take(ids, axis: 0);
      return out.reshape([1, tokenIds.length, config.textHiddenSize]);
    } finally {
      ids.close();
    }
  }

  /// Run the decoder stack on hidden states with MRoPE position IDs.
  ///
  /// [hidden]: [1, seqLen, hiddenSize].
  /// [positionIds]: [1, 3, seqLen] MRoPE position IDs.
  /// [cache]: optional KV cache for autoregressive generation.
  ///
  /// Returns [1, seqLen, hiddenSize] after final RMSNorm.
  MlxArray forward(
    MlxArray hidden, {
    required MlxArray positionIds,
    AsrKvCache? cache,
  }) {
    final seqLen = hidden.shape[1];

    // Compute MRoPE cos/sin from position IDs.
    final rope = _mrope.compute(positionIds, hidden.dtype);
    try {
      // Build causal mask.
      MlxArray? mask;
      if (cache != null && cache.offset > 0) {
        // Incremental decode with existing cache.
        if (seqLen == 1) {
          // Single-token decode: no mask needed (attends to all cached KV).
          mask = null;
        } else {
          // Multi-token decode after prefix: causal among new tokens,
          // full visibility to cached prefix.
          mask = _createCausalMaskWithPrefix(
            seqLen: seqLen,
            prefixLen: cache.offset,
            dtype: hidden.dtype,
          );
        }
      } else {
        // First prefill: standard causal mask.
        mask = seqLen > 1 ? _createCausalMask(seqLen, hidden.dtype) : null;
      }

      var h = hidden;
      for (var i = 0; i < _layers.length; i++) {
        final next = _decoderLayer(
          _layers[i],
          h,
          seqLen,
          cos: rope.cos,
          sin: rope.sin,
          mask: mask,
          cache: cache,
          layerIdx: i,
        );
        if (h != hidden) h.close();
        h = next;
      }

      // Update KV cache offset after all layers have processed.
      if (cache != null) {
        cache._offset += seqLen;
      }

      mask?.close();
      final normed = mx.fast.rmsNorm(
        h,
        weight: _finalNormW,
        eps: config.textRmsNormEps,
      );
      if (h != hidden) h.close();
      return normed;
    } finally {
      rope.cos.close();
      rope.sin.close();
    }
  }

  /// Compute logits from decoder output.
  /// Input: [1, seqLen, hiddenSize] (typically just last token).
  /// Output: [1, vocabSize] logits.
  ///
  /// Uses the dequantized dense lm_head weight (matching the reference
  /// implementation where lm_head is a plain nn.Linear with float weight
  /// tied to embed_tokens *before* quantization).
  MlxArray lmHead(MlxArray hidden) {
    final lastH = hidden.shape[1] > 1
        ? hidden.slice(
            start: [0, hidden.shape[1] - 1, 0],
            stop: [1, hidden.shape[1], config.textHiddenSize],
          )
        : hidden;
    final h2d = lastH.reshape([1, config.textHiddenSize]);
    if (lastH != hidden) lastH.close();
    try {
      final w = _lmHeadWeight ?? _embedWeight;
      final wT = w.transpose();
      try {
        return mx.matmul(h2d, wT);
      } finally {
        wT.close();
      }
    } finally {
      h2d.close();
    }
  }

  /// Create a new KV cache for autoregressive decoding.
  AsrKvCache createCache() => AsrKvCache(config.textNumLayers);

  MlxArray _decoderLayer(
    _DecoderLayer layer,
    MlxArray input,
    int seqLen, {
    required MlxArray cos,
    required MlxArray sin,
    MlxArray? mask,
    AsrKvCache? cache,
    required int layerIdx,
  }) {
    // Pre-norm self-attention.
    final norm1 = mx.fast.rmsNorm(
      input,
      weight: layer.inputNormW,
      eps: config.textRmsNormEps,
    );
    final attn = _selfAttention(
      layer,
      norm1,
      seqLen,
      cos: cos,
      sin: sin,
      mask: mask,
      cache: cache,
      layerIdx: layerIdx,
    );
    norm1.close();
    final residual1 = mx.add(input, attn);
    attn.close();

    // Pre-norm MLP.
    final norm2 = mx.fast.rmsNorm(
      residual1,
      weight: layer.postNormW,
      eps: config.textRmsNormEps,
    );
    final mlp = _swiGluMlp(layer, norm2);
    norm2.close();
    final residual2 = mx.add(residual1, mlp);
    residual1.close();
    mlp.close();
    return residual2;
  }

  MlxArray _selfAttention(
    _DecoderLayer layer,
    MlxArray input,
    int seqLen, {
    required MlxArray cos,
    required MlxArray sin,
    MlxArray? mask,
    AsrKvCache? cache,
    required int layerIdx,
  }) {
    final nH = config.textNumHeads;
    final nKv = config.textNumKvHeads;
    final hd = config.textHeadDim;

    // Q, K, V projections.
    final q = _quantLinear(input, layer.qW, layer.qS, layer.qB);
    final k = _quantLinear(input, layer.kW, layer.kS, layer.kB);
    final v = _quantLinear(input, layer.vW, layer.vS, layer.vB);

    // Reshape to [1, seqLen, nHeads, headDim].
    final q4r = q.reshape([1, seqLen, nH, hd]);
    final k4r = k.reshape([1, seqLen, nKv, hd]);
    final v4r = v.reshape([1, seqLen, nKv, hd]);
    q.close();
    k.close();
    v.close();

    // Q/K RMSNorm (Qwen3 has q_norm and k_norm per layer).
    final qNorm = mx.fast.rmsNorm(
      q4r,
      weight: layer.qNormW,
      eps: config.textRmsNormEps,
    );
    final kNorm = mx.fast.rmsNorm(
      k4r,
      weight: layer.kNormW,
      eps: config.textRmsNormEps,
    );
    q4r.close();
    k4r.close();

    // Transpose to [1, nHeads, seqLen, headDim] for attention.
    final qT = qNorm.transposeAxes([0, 2, 1, 3]);
    final kT = kNorm.transposeAxes([0, 2, 1, 3]);
    final vT = v4r.transposeAxes([0, 2, 1, 3]);
    qNorm.close();
    kNorm.close();
    v4r.close();

    // Apply MRoPE.
    final rotated = applyRotaryPosEmb(qT, kT, cos, sin);
    final qRope = rotated.q;
    var kRope = rotated.k;
    qT.close();
    kT.close();

    // KV cache update.
    var vAttn = vT;
    if (cache != null) {
      final updated = cache._updateLayer(kRope, vAttn, layerIdx);
      kRope = updated.$1;
      vAttn = updated.$2;
    }

    // GQA: repeat KV heads if needed.
    final repeat = nH ~/ nKv;
    var kSdpa = kRope;
    var vSdpa = vAttn;
    if (repeat > 1) {
      kSdpa = _repeatKvHeads(kRope, nH, nKv, kRope.shape[2], hd);
      vSdpa = _repeatKvHeads(vAttn, nH, nKv, vAttn.shape[2], hd);
      if (cache == null) {
        kRope.close();
        vT.close();
      }
    }

    // Scaled dot-product attention with explicit additive mask.
    final scale = 1.0 / math.sqrt(hd);
    MlxArray attn;
    if (mask != null) {
      attn = mx.fast.scaledDotProductAttention(
        qRope,
        kSdpa,
        vSdpa,
        scale: scale,
        mask: mask,
      );
    } else {
      attn = mx.fast.scaledDotProductAttention(
        qRope,
        kSdpa,
        vSdpa,
        scale: scale,
      );
    }
    qRope.close();
    if (repeat > 1 || cache == null) {
      kSdpa.close();
      vSdpa.close();
    }

    // Merge heads → output projection.
    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      1,
      seqLen,
      nH * hd,
    ]);
    attn.close();
    final out = _quantLinear(merged, layer.oW, layer.oS, layer.oB);
    merged.close();
    return out.reshape([1, seqLen, config.textHiddenSize]);
  }

  MlxArray _swiGluMlp(_DecoderLayer layer, MlxArray input) {
    final gate = _quantLinear(input, layer.gateW, layer.gateS, layer.gateB);
    final up = _quantLinear(input, layer.upW, layer.upS, layer.upB);
    final sig = gate.sigmoid();
    final activated = gate * sig;
    sig.close();
    gate.close();
    final fused = activated * up;
    activated.close();
    up.close();
    final down = _quantLinear(fused, layer.downW, layer.downS, layer.downB);
    fused.close();
    return down.reshape([1, input.shape[1], config.textHiddenSize]);
  }

  MlxArray _quantLinear(
    MlxArray input,
    MlxArray weight,
    MlxArray? scales,
    MlxArray? biases,
  ) {
    if (scales != null) {
      return mx.quant.matmul(
        input,
        MlxQuantizedMatrix(weight, scales, biases),
        transpose: true,
        groupSize: config.quantGroupSize,
        bits: config.quantBits,
        mode: config.quantMode,
      );
    }
    return mx.matmul(input, weight.transpose());
  }

  MlxArray _repeatKvHeads(
    MlxArray tensor,
    int numHeads,
    int numKvHeads,
    int seqLen,
    int headDim,
  ) {
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

  void close() {
    _lmHeadWeight?.close();
    _mrope.close();
  }
}

/// KV cache for autoregressive decoding.
///
/// Offset is updated globally after all layers process a step (matching
/// the reference implementation where offset updates on the last layer).
final class AsrKvCache {
  AsrKvCache(int numLayers)
    : _keys = List<MlxArray?>.filled(numLayers, null),
      _values = List<MlxArray?>.filled(numLayers, null),
      _offset = 0;

  final List<MlxArray?> _keys;
  final List<MlxArray?> _values;
  int _offset;

  /// Current cached sequence length.
  int get offset => _offset;

  /// Number of decoder layers.
  int get numLayers => _keys.length;

  /// Update a single layer's KV cache. Does NOT update offset.
  /// Offset is updated by [Qwen3AsrTextDecoder.forward] after all layers.
  (MlxArray, MlxArray) _updateLayer(
    MlxArray nextK,
    MlxArray nextV,
    int layerIdx,
  ) {
    final curK = _keys[layerIdx];
    final curV = _values[layerIdx];
    if (curK == null || curV == null) {
      _keys[layerIdx] = nextK;
      _values[layerIdx] = nextV;
      return (nextK, nextV);
    }
    final mergedK = mx.concatenate([curK, nextK], axis: 2);
    final mergedV = mx.concatenate([curV, nextV], axis: 2);
    curK.close();
    curV.close();
    nextK.close();
    nextV.close();
    _keys[layerIdx] = mergedK;
    _values[layerIdx] = mergedV;
    return (mergedK, mergedV);
  }

  /// Trim recently appended tokens from all layer caches.
  void trim(int numTokens) {
    if (numTokens <= 0 || numTokens > _offset) return;
    final newLen = _offset - numTokens;
    for (var i = 0; i < _keys.length; i++) {
      final k = _keys[i];
      final v = _values[i];
      if (k != null) {
        _keys[i] = k.slice(
          start: [0, 0, 0, 0],
          stop: [k.shape[0], k.shape[1], newLen, k.shape[3]],
        );
        k.close();
      }
      if (v != null) {
        _values[i] = v.slice(
          start: [0, 0, 0, 0],
          stop: [v.shape[0], v.shape[1], newLen, v.shape[3]],
        );
        v.close();
      }
    }
    _offset = newLen;
  }

  void close() {
    for (var i = 0; i < _keys.length; i++) {
      _keys[i]?.close();
      _values[i]?.close();
      _keys[i] = null;
      _values[i] = null;
    }
    _offset = 0;
  }
}

// ── Causal mask utilities ──

/// Standard causal mask: (1, 1, L, L) with -1e9 above diagonal, 0 elsewhere.
MlxArray _createCausalMask(int seqLen, MlxDType dtype) {
  final mask = MlxArray.full([seqLen, seqLen], -1e9, dtype: dtype);
  final causal = mask.triu(k: 1); // zero on/below diagonal, -1e9 above
  mask.close();
  final expanded = causal.reshape([1, 1, seqLen, seqLen]);
  causal.close();
  return expanded;
}

/// Causal mask for appending tokens after a cached prefix.
///
/// The prefix (already in cache) is fully visible to all new queries.
/// Causality only among the newly appended tokens.
/// Returns (1, 1, seqLen, prefixLen + seqLen).
MlxArray _createCausalMaskWithPrefix({
  required int seqLen,
  required int prefixLen,
  required MlxDType dtype,
}) {
  // Left block: prefix keys always visible.
  final left = MlxArray.zeros([seqLen, prefixLen], dtype: dtype);
  // Right block: causal among new tokens.
  final right = MlxArray.full([seqLen, seqLen], -1e9, dtype: dtype);
  final rightCausal = right.triu(k: 1);
  right.close();
  final mask = mx.concatenate([left, rightCausal], axis: 1);
  left.close();
  rightCausal.close();
  final expanded = mask.reshape([1, 1, seqLen, prefixLen + seqLen]);
  mask.close();
  return expanded;
}

// ── Decoder layer weight holder ──

final class _DecoderLayer {
  const _DecoderLayer({
    required this.inputNormW,
    required this.postNormW,
    required this.qW,
    required this.qS,
    required this.qB,
    required this.kW,
    required this.kS,
    required this.kB,
    required this.vW,
    required this.vS,
    required this.vB,
    required this.oW,
    required this.oS,
    required this.oB,
    required this.qNormW,
    required this.kNormW,
    required this.gateW,
    required this.gateS,
    required this.gateB,
    required this.upW,
    required this.upS,
    required this.upB,
    required this.downW,
    required this.downS,
    required this.downB,
  });

  factory _DecoderLayer.load(
    Map<String, MlxArray> t,
    String p,
    Qwen3AsrConfig config,
  ) {
    return _DecoderLayer(
      inputNormW: t['${p}input_layernorm.weight']!,
      postNormW: t['${p}post_attention_layernorm.weight']!,
      qW: t['${p}self_attn.q_proj.weight']!,
      qS: t['${p}self_attn.q_proj.scales'],
      qB: t['${p}self_attn.q_proj.biases'],
      kW: t['${p}self_attn.k_proj.weight']!,
      kS: t['${p}self_attn.k_proj.scales'],
      kB: t['${p}self_attn.k_proj.biases'],
      vW: t['${p}self_attn.v_proj.weight']!,
      vS: t['${p}self_attn.v_proj.scales'],
      vB: t['${p}self_attn.v_proj.biases'],
      oW: t['${p}self_attn.o_proj.weight']!,
      oS: t['${p}self_attn.o_proj.scales'],
      oB: t['${p}self_attn.o_proj.biases'],
      qNormW: t['${p}self_attn.q_norm.weight']!,
      kNormW: t['${p}self_attn.k_norm.weight']!,
      gateW: t['${p}mlp.gate_proj.weight']!,
      gateS: t['${p}mlp.gate_proj.scales'],
      gateB: t['${p}mlp.gate_proj.biases'],
      upW: t['${p}mlp.up_proj.weight']!,
      upS: t['${p}mlp.up_proj.scales'],
      upB: t['${p}mlp.up_proj.biases'],
      downW: t['${p}mlp.down_proj.weight']!,
      downS: t['${p}mlp.down_proj.scales'],
      downB: t['${p}mlp.down_proj.biases'],
    );
  }

  // Norms.
  final MlxArray inputNormW;
  final MlxArray postNormW;

  // Self-attention projections (quantized).
  final MlxArray qW;
  final MlxArray? qS, qB;
  final MlxArray kW;
  final MlxArray? kS, kB;
  final MlxArray vW;
  final MlxArray? vS, vB;
  final MlxArray oW;
  final MlxArray? oS, oB;

  // Q/K norm weights.
  final MlxArray qNormW;
  final MlxArray kNormW;

  // MLP projections (quantized).
  final MlxArray gateW;
  final MlxArray? gateS, gateB;
  final MlxArray upW;
  final MlxArray? upS, upB;
  final MlxArray downW;
  final MlxArray? downS, downB;
}
