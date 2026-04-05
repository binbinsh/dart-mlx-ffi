part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// Generic linear building blocks (shared by both vision and language paths)
// ---------------------------------------------------------------------------

sealed class _LinearBase {
  const _LinearBase();

  MlxArray apply(MlxArray input);

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
      return _QuantLinear(weight, scales, biases, bias, defaultQuant);
    }
    return _DenseLinear(weight, bias);
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
  const _DenseLinear(this.weight, this.bias);

  final MlxArray weight;
  final MlxArray? bias;

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
}

final class _QuantLinear extends _LinearBase {
  const _QuantLinear(
    this.weight,
    this.scales,
    this.biases,
    this.bias,
    this.quantSpec,
  );

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray? biases;
  final MlxArray? bias;
  final _QuantSpec quantSpec;

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

  // Fused QKV attention projection + output projection
  final _LinearBase qkv;
  final _LinearBase outProj;

  // MLP (fc1 → GELU → fc2)
  final _LinearBase fc1;
  final _LinearBase fc2;
}

/// Weights for the spatial-merge projector.
final class _ProjectorWeights {
  const _ProjectorWeights({
    required this.preNormWeight,
    required this.preNormBias,
    required this.linear1,
    required this.linear2,
  });

  final MlxArray preNormWeight;
  final MlxArray preNormBias;
  final _LinearBase linear1;
  final _LinearBase linear2;
}

/// Complete vision encoder weights (ViT + projector).
final class _VisionWeights {
  const _VisionWeights({
    required this.patchEmbedWeight,
    required this.patchEmbedBias,
    required this.positionEmbedding,
    required this.blocks,
    required this.projector,
  });

  /// Conv2d patch embedding kernel: [outChannels, inChannels, pH, pW].
  final MlxArray patchEmbedWeight;
  final MlxArray? patchEmbedBias;

  /// Position embedding — may be quantized.
  final _LinearBase positionEmbedding;

  final List<_VisionBlockWeights> blocks;
  final _ProjectorWeights projector;
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
