import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'config.dart';

/// Qwen3 text decoder for Qwen3-ASR.
///
/// Standard Qwen3 transformer: RMSNorm → GQA self-attention (with q/k norm)
/// → RMSNorm → SwiGLU MLP. 28 layers, 16 heads, 8 KV heads, head_dim=128.
///
/// Text decoder weights are quantized 8-bit affine (group_size=64).
/// Audio encoder weights are dense (BF16) and handled separately.
///
/// Tensor prefix: "model.layers.{i}." for decoder layers,
///               "model.embed_tokens." for embedding,
///               "model.norm." for final norm.
final class Qwen3AsrTextDecoder {
  Qwen3AsrTextDecoder._(
    this.config,
    this._embedWeight,
    this._embedScales,
    this._embedBiases,
    this._layers,
    this._finalNormW,
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

    return Qwen3AsrTextDecoder._(
      config,
      tensors['${p}embed_tokens.weight']!,
      tensors['${p}embed_tokens.scales'],
      tensors['${p}embed_tokens.biases'],
      layers,
      tensors['${p}norm.weight']!,
    );
  }

  final Qwen3AsrConfig config;
  final MlxArray _embedWeight;
  final MlxArray? _embedScales;
  final MlxArray? _embedBiases;
  final List<_DecoderLayer> _layers;
  final MlxArray _finalNormW;
  MlxArray? _ropeInvFreq;

  /// Embed token IDs → [1, seqLen, hiddenSize].
  MlxArray embed(List<int> tokenIds) {
    final ids = MlxArray.fromInt32List(tokenIds, shape: [tokenIds.length]);
    try {
      if (_embedScales != null) {
        // Quantized embedding: gather rows → dequantize.
        final rowsW = _embedWeight.take(ids, axis: 0);
        final rowsS = _embedScales!.take(ids, axis: 0);
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
      // Dense embedding.
      final out = _embedWeight.take(ids, axis: 0);
      return out.reshape([1, tokenIds.length, config.textHiddenSize]);
    } finally {
      ids.close();
    }
  }

  /// Run the decoder stack on hidden states.
  /// Input: [1, seqLen, hiddenSize] (may be mixed audio+text embeddings).
  /// Output: [1, seqLen, hiddenSize] after final RMSNorm.
  MlxArray forward(MlxArray hidden, {AsrKvCache? cache}) {
    final seqLen = hidden.shape[1];
    var h = hidden;
    for (var i = 0; i < _layers.length; i++) {
      final next = _decoderLayer(
        _layers[i],
        h,
        seqLen,
        cache: cache?.layers[i],
      );
      if (h != hidden) h.close();
      h = next;
    }
    final normed = mx.fast.rmsNorm(
      h,
      weight: _finalNormW,
      eps: config.textRmsNormEps,
    );
    if (h != hidden) h.close();
    return normed;
  }

  /// Compute logits from decoder output.
  /// Input: [1, seqLen, hiddenSize] (typically just last token).
  /// Output: [1, vocabSize] logits.
  MlxArray lmHead(MlxArray hidden) {
    // tie_word_embeddings: reuse embed weight as LM head.
    final lastH = hidden.shape[1] > 1
        ? hidden.slice(
            start: [0, hidden.shape[1] - 1, 0],
            stop: [1, hidden.shape[1], config.textHiddenSize],
          )
        : hidden;
    final h2d = lastH.reshape([1, config.textHiddenSize]);
    if (lastH != hidden) lastH.close();
    try {
      if (_embedScales != null) {
        final result = mx.quant.matmul(
          h2d,
          MlxQuantizedMatrix(_embedWeight, _embedScales!, _embedBiases),
          transpose: true,
          groupSize: config.quantGroupSize,
          bits: config.quantBits,
          mode: config.quantMode,
        );
        return result;
      }
      return mx.matmul(h2d, _embedWeight.transpose());
    } finally {
      h2d.close();
    }
  }

  /// Create a new KV cache for autoregressive decoding.
  AsrKvCache createCache() =>
      AsrKvCache(List.generate(config.textNumLayers, (_) => AsrKvLayerCache()));

  MlxArray _decoderLayer(
    _DecoderLayer layer,
    MlxArray input,
    int seqLen, {
    AsrKvLayerCache? cache,
  }) {
    // Pre-norm self-attention.
    final norm1 = mx.fast.rmsNorm(
      input,
      weight: layer.inputNormW,
      eps: config.textRmsNormEps,
    );
    final attn = _selfAttention(layer, norm1, seqLen, cache: cache);
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
    AsrKvLayerCache? cache,
  }) {
    final nH = config.textNumHeads;
    final nKv = config.textNumKvHeads;
    final hd = config.textHeadDim;

    // Q, K, V projections.
    final q = _quantLinear(input, layer.qW, layer.qS, layer.qB);
    final k = _quantLinear(input, layer.kW, layer.kS, layer.kB);
    final v = _quantLinear(input, layer.vW, layer.vS, layer.vB);

    // Reshape to [1, seqLen, nHeads, headDim] → [1, nHeads, seqLen, headDim].
    final q4 = q.reshape([1, seqLen, nH, hd]).transposeAxes([0, 2, 1, 3]);
    final k4 = k.reshape([1, seqLen, nKv, hd]).transposeAxes([0, 2, 1, 3]);
    final v4 = v.reshape([1, seqLen, nKv, hd]).transposeAxes([0, 2, 1, 3]);
    q.close();
    k.close();
    v.close();

    // Q/K RMSNorm (Qwen3 has q_norm and k_norm per layer).
    final qNorm = mx.fast.rmsNorm(
      q4,
      weight: layer.qNormW,
      eps: config.textRmsNormEps,
    );
    final kNorm = mx.fast.rmsNorm(
      k4,
      weight: layer.kNormW,
      eps: config.textRmsNormEps,
    );
    q4.close();
    k4.close();

    // Apply RoPE.
    final offset = cache?.offset ?? 0;
    final qRope = mx.fast.rope(
      qNorm,
      dims: hd,
      traditional: false,
      base: config.textRopeTheta,
      offset: offset,
    );
    var kRope = mx.fast.rope(
      kNorm,
      dims: hd,
      traditional: false,
      base: config.textRopeTheta,
      offset: offset,
    );
    qNorm.close();
    kNorm.close();

    // KV cache update.
    var vAttn = v4;
    if (cache != null) {
      final updated = cache.updateAndFetch(kRope, vAttn);
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
        v4.close();
      }
    }

    // Scaled dot-product attention.
    // Use causal mask for prefill (seqLen > 1), no mask for decode step.
    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kSdpa,
      vSdpa,
      scale: 1.0 / math.sqrt(hd),
      maskMode: cache != null && seqLen == 1 ? '' : 'causal',
    );
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
    // SwiGLU: silu(gate) * up.
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

  /// Quantized linear: input @ weight^T using quantized matmul.
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
    _ropeInvFreq?.close();
    _ropeInvFreq = null;
  }
}

/// KV cache for autoregressive decoding.
final class AsrKvCache {
  AsrKvCache(this.layers);

  final List<AsrKvLayerCache> layers;

  void close() {
    for (final layer in layers) {
      layer.close();
    }
  }
}

/// Per-layer KV cache.
final class AsrKvLayerCache {
  MlxArray? keys;
  MlxArray? values;
  int offset = 0;

  (MlxArray, MlxArray) updateAndFetch(MlxArray nextK, MlxArray nextV) {
    final curK = keys;
    final curV = values;
    if (curK == null || curV == null) {
      keys = nextK;
      values = nextV;
      offset = nextK.shape[2];
      return (nextK, nextV);
    }
    final mergedK = mx.concatenate([curK, nextK], axis: 2);
    final mergedV = mx.concatenate([curV, nextV], axis: 2);
    curK.close();
    curV.close();
    nextK.close();
    nextV.close();
    keys = mergedK;
    values = mergedV;
    offset = mergedK.shape[2];
    return (mergedK, mergedV);
  }

  void close() {
    keys?.close();
    values?.close();
    keys = null;
    values = null;
    offset = 0;
  }
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
