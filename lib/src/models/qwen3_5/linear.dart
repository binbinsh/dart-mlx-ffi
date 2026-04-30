part of 'qwen3_5.dart';

sealed class _LinearBase {
  const _LinearBase();

  MlxArray apply(MlxArray input, {required Qwen3_5Config config});

  static _LinearBase load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required Qwen3_5QuantSpec defaultQuantSpec,
    Qwen3_5QuantSpec? quantSpec,
  }) {
    final weight = tensors['$prefix.weight'];
    final scales = tensors['$prefix.scales'];
    final biases = tensors['$prefix.biases'];
    final bias = tensors['$prefix.bias'];
    if (weight == null) {
      throw StateError('Missing weight tensor for $prefix.');
    }
    if (scales != null) {
      return _QuantLinear(
        weight,
        scales,
        biases,
        bias,
        quantSpec ?? defaultQuantSpec,
      );
    }
    return _DenseLinear(weight, bias);
  }
}

final class _DenseLinear extends _LinearBase {
  const _DenseLinear(this.weight, this.bias);

  final MlxArray weight;
  final MlxArray? bias;

  @override
  MlxArray apply(MlxArray input, {required Qwen3_5Config config}) {
    final y = mx.matmul(input, weight.transpose());
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([1, bias!.shape[0]]);
    try {
      return mx.add(y, reshapedBias);
    } finally {
      reshapedBias.close();
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
  final Qwen3_5QuantSpec quantSpec;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);

  @override
  MlxArray apply(MlxArray input, {required Qwen3_5Config config}) {
    final y = mx.quant.matmul(
      input,
      matrix,
      transpose: true,
      groupSize: quantSpec.groupSize,
      bits: quantSpec.bits,
      mode: quantSpec.mode,
    );
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([1, bias!.shape[0]]);
    try {
      return mx.add(y, reshapedBias);
    } finally {
      reshapedBias.close();
      y.close();
    }
  }
}

final class _FullAttentionWeights {
  const _FullAttentionWeights({
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.qNormWeight,
    required this.kNormWeight,
  });

  final _LinearBase qProj;
  final _LinearBase kProj;
  final _LinearBase vProj;
  final _LinearBase oProj;
  final MlxArray? qNormWeight;
  final MlxArray? kNormWeight;
}

final class _LinearAttentionWeights {
  const _LinearAttentionWeights({
    required this.convWeight,
    required this.inProjQkv,
    required this.inProjZ,
    required this.inProjB,
    required this.inProjA,
    required this.dtBias,
    required this.aLog,
    required this.normWeight,
    required this.outProj,
  });

  final MlxArray convWeight;
  final _LinearBase inProjQkv;
  final _LinearBase inProjZ;
  final _LinearBase inProjB;
  final _LinearBase inProjA;
  final MlxArray dtBias;
  final MlxArray aLog;
  final MlxArray normWeight;
  final _LinearBase outProj;
}

final class _DenseMlpWeights {
  const _DenseMlpWeights({
    required this.gateProj,
    required this.upProj,
    required this.downProj,
  });

  final _LinearBase gateProj;
  final _LinearBase upProj;
  final _LinearBase downProj;
}

final class _LayerWeights {
  const _LayerWeights({
    required this.inputNorm,
    required this.postNorm,
    required this.fullAttention,
    required this.linearAttention,
    required this.denseMlp,
  });

  final MlxArray inputNorm;
  final MlxArray postNorm;
  final _FullAttentionWeights? fullAttention;
  final _LinearAttentionWeights? linearAttention;
  final _DenseMlpWeights denseMlp;
}
