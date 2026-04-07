part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// Generic linear building blocks (shared by both vision and language paths)
// ---------------------------------------------------------------------------

sealed class _LinearBase {
  const _LinearBase();

  MlxArray apply(MlxArray input);

  /// Close all MlxArray handles held by this linear layer and remove them
  /// from [tensors] so they are not double-closed by the runner.
  void release(Map<String, MlxArray> tensors);

  static _LinearBase load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required _QuantSpec defaultQuant,
  }) {
    final weight = tensors['$prefix.weight'];
    final scales = tensors['$prefix.scales'];
    final biases = tensors['$prefix.biases'];
    final bias = tensors['$prefix.bias'];
    if (weight == null) {
      throw StateError('Missing weight tensor for $prefix.');
    }
    if (scales != null) {
      return _QuantLinear(weight, scales, biases, bias, defaultQuant, prefix);
    }
    return _DenseLinear(weight, bias, prefix);
  }

  static _LinearBase? maybeLoad(
    Map<String, MlxArray> tensors,
    String prefix, {
    required _QuantSpec defaultQuant,
  }) {
    if (!tensors.containsKey('$prefix.weight')) return null;
    return load(tensors, prefix, defaultQuant: defaultQuant);
  }
}

final class _DenseLinear extends _LinearBase {
  const _DenseLinear(this.weight, this.bias, this.prefix);

  final MlxArray weight;
  final MlxArray? bias;
  final String prefix;

  @override
  MlxArray apply(MlxArray input) {
    final y = mx.matmul(input, weight.transpose());
    if (bias == null) return y;
    final b = bias!.reshape([1, bias!.shape[0]]);
    try {
      return mx.add(y, b);
    } finally {
      b.close();
      y.close();
    }
  }

  @override
  void release(Map<String, MlxArray> tensors) {
    weight.close();
    tensors.remove('$prefix.weight');
    final b = bias;
    if (b != null) {
      b.close();
      tensors.remove('$prefix.bias');
    }
  }
}

final class _QuantLinear extends _LinearBase {
  const _QuantLinear(
    this.weight,
    this.scales,
    this.biases,
    this.bias,
    this.quantSpec,
    this.prefix,
  );

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray? biases;
  final MlxArray? bias;
  final _QuantSpec quantSpec;
  final String prefix;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);

  @override
  MlxArray apply(MlxArray input) {
    final y = mx.quant.matmul(
      input,
      matrix,
      transpose: true,
      groupSize: quantSpec.groupSize,
      bits: quantSpec.bits,
      mode: quantSpec.mode,
    );
    if (bias == null) return y;
    final b = bias!.reshape([1, bias!.shape[0]]);
    try {
      return mx.add(y, b);
    } finally {
      b.close();
      y.close();
    }
  }

  @override
  void release(Map<String, MlxArray> tensors) {
    weight.close();
    tensors.remove('$prefix.weight');
    scales.close();
    tensors.remove('$prefix.scales');
    final qBiases = biases;
    if (qBiases != null) {
      qBiases.close();
      tensors.remove('$prefix.biases');
    }
    final b = bias;
    if (b != null) {
      b.close();
      tensors.remove('$prefix.bias');
    }
  }
}

// ---------------------------------------------------------------------------
// Vision encoder weight containers
// ---------------------------------------------------------------------------

/// Weights for a single ViT transformer block.
final class _VisionBlockWeights {
  const _VisionBlockWeights({
    required this.layerNorm1Weight,
    required this.layerNorm1Bias,
    required this.layerNorm2Weight,
    required this.layerNorm2Bias,
    required this.layerNorm1Key,
    required this.layerNorm1BiasKey,
    required this.layerNorm2Key,
    required this.layerNorm2BiasKey,
    required this.qkv,
    required this.outProj,
    required this.fc1,
    required this.fc2,
  });

  // Layer norms (standard, not RMS)
  final MlxArray layerNorm1Weight;
  final MlxArray layerNorm1Bias;
  final MlxArray layerNorm2Weight;
  final MlxArray layerNorm2Bias;
  // Keys for removing from tensor map on release
  final String layerNorm1Key;
  final String layerNorm1BiasKey;
  final String layerNorm2Key;
  final String layerNorm2BiasKey;

  // Fused QKV attention projection + output projection
  final _LinearBase qkv;
  final _LinearBase outProj;

  // MLP (fc1 → GELU → fc2)
  final _LinearBase fc1;
  final _LinearBase fc2;

  /// Close all weight tensors and remove them from [tensors].
  void release(Map<String, MlxArray> tensors) {
    layerNorm1Weight.close();
    tensors.remove(layerNorm1Key);
    layerNorm1Bias.close();
    tensors.remove(layerNorm1BiasKey);
    layerNorm2Weight.close();
    tensors.remove(layerNorm2Key);
    layerNorm2Bias.close();
    tensors.remove(layerNorm2BiasKey);
    qkv.release(tensors);
    outProj.release(tensors);
    fc1.release(tensors);
    fc2.release(tensors);
  }
}

/// Weights for the spatial-merge projector.
final class _ProjectorWeights {
  const _ProjectorWeights({
    required this.preNormWeight,
    required this.preNormBias,
    required this.preNormWeightKey,
    required this.preNormBiasKey,
    required this.linear1,
    required this.linear2,
  });

  final MlxArray preNormWeight;
  final MlxArray preNormBias;
  final String preNormWeightKey;
  final String preNormBiasKey;
  final _LinearBase linear1;
  final _LinearBase linear2;

  /// Close all weight tensors and remove them from [tensors].
  void release(Map<String, MlxArray> tensors) {
    preNormWeight.close();
    tensors.remove(preNormWeightKey);
    preNormBias.close();
    tensors.remove(preNormBiasKey);
    linear1.release(tensors);
    linear2.release(tensors);
  }
}

/// Complete vision encoder weights (ViT + projector).
final class _VisionWeights {
  _VisionWeights({
    required this.patchEmbedWeight,
    required this.patchEmbedBias,
    required this.patchEmbedWeightKey,
    required this.patchEmbedBiasKey,
    required this.positionEmbedding,
    required this.blocks,
    required this.postLayerNormWeight,
    required this.postLayerNormBias,
    required this.postLayerNormWeightKey,
    required this.postLayerNormBiasKey,
    required this.projector,
  });

  /// Conv2d patch embedding kernel: [outChannels, inChannels, pH, pW].
  final MlxArray patchEmbedWeight;
  final MlxArray? patchEmbedBias;
  final String patchEmbedWeightKey;
  final String? patchEmbedBiasKey;

  /// Position embedding — may be quantized.
  final _LinearBase positionEmbedding;

  final List<_VisionBlockWeights> blocks;
  final MlxArray postLayerNormWeight;
  final MlxArray postLayerNormBias;
  final String postLayerNormWeightKey;
  final String postLayerNormBiasKey;
  final _ProjectorWeights projector;

  bool _released = false;

  /// Whether the vision weights have already been released.
  bool get isReleased => _released;

  /// Close all vision weight tensors and remove them from [tensors].
  ///
  /// After this call, the vision encoder cannot be used again.
  void release(Map<String, MlxArray> tensors) {
    if (_released) return;
    _released = true;

    patchEmbedWeight.close();
    tensors.remove(patchEmbedWeightKey);
    final pBias = patchEmbedBias;
    if (pBias != null && patchEmbedBiasKey != null) {
      pBias.close();
      tensors.remove(patchEmbedBiasKey);
    }

    positionEmbedding.release(tensors);

    for (final block in blocks) {
      block.release(tensors);
    }

    postLayerNormWeight.close();
    tensors.remove(postLayerNormWeightKey);
    postLayerNormBias.close();
    tensors.remove(postLayerNormBiasKey);

    projector.release(tensors);
  }
}

// ---------------------------------------------------------------------------
// Language model weight containers
// ---------------------------------------------------------------------------

/// Attention weights for an ERNIE-4.5 decoder layer.
final class _LmAttentionWeights {
  const _LmAttentionWeights({
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
  });

  final _LinearBase qProj;
  final _LinearBase kProj;
  final _LinearBase vProj;
  final _LinearBase oProj;
}

/// SiLU-gated MLP weights for an ERNIE-4.5 decoder layer.
final class _LmMlpWeights {
  const _LmMlpWeights({
    required this.gateProj,
    required this.upProj,
    required this.downProj,
  });

  final _LinearBase gateProj;
  final _LinearBase upProj;
  final _LinearBase downProj;
}

/// Weights for a single ERNIE-4.5 decoder layer.
final class _LmLayerWeights {
  const _LmLayerWeights({
    required this.inputNorm,
    required this.postNorm,
    required this.attention,
    required this.mlp,
  });

  /// RMS layer norm weight.
  final MlxArray inputNorm;
  final MlxArray postNorm;
  final _LmAttentionWeights attention;
  final _LmMlpWeights mlp;
}
