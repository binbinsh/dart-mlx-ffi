part of 'qwen3_5.dart';

sealed class _LinearBase {
  const _LinearBase();

  MlxArray apply(MlxArray input, {required Qwen3_5Config config});

  void close();

  static _LinearBase load(Map<String, MlxArray> tensors, String prefix) {
    final weight = tensors['$prefix.weight'];
    final scales = tensors['$prefix.scales'];
    final biases = tensors['$prefix.biases'];
    final bias = tensors['$prefix.bias'];
    if (weight == null) {
      throw StateError('Missing weight tensor for $prefix.');
    }
    if (scales != null) {
      return _QuantLinear(weight, scales, biases, bias);
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

  @override
  void close() {}
}

final class _QuantLinear extends _LinearBase {
  const _QuantLinear(this.weight, this.scales, this.biases, this.bias);

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray? biases;
  final MlxArray? bias;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);

  @override
  MlxArray apply(MlxArray input, {required Qwen3_5Config config}) {
    final y = mx.quant.matmul(
      input,
      matrix,
      transpose: true,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: config.mode,
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

  @override
  void close() {}
}

sealed class _SwitchLinearBase {
  const _SwitchLinearBase();

  MlxArray applyExpert(
    MlxArray input,
    int expert, {
    required Qwen3_5Config config,
  });

  MlxArray? applyExperts(
    MlxArray input,
    MlxArray experts, {
    required Qwen3_5Config config,
    bool sortedIndices = false,
  }) => null;

  static _SwitchLinearBase load(Map<String, MlxArray> tensors, String prefix) {
    final weight = tensors['$prefix.weight'];
    final scales = tensors['$prefix.scales'];
    final biases = tensors['$prefix.biases'];
    final bias = tensors['$prefix.bias'];
    if (weight == null) {
      throw StateError('Missing switch weight tensor for $prefix.');
    }
    if (scales != null) {
      return _QuantSwitchLinear(weight, scales, biases, bias);
    }
    return _DenseSwitchLinear(weight, bias);
  }
}

final class _DenseSwitchLinear extends _SwitchLinearBase {
  const _DenseSwitchLinear(this.weight, this.bias);

  final MlxArray weight;
  final MlxArray? bias;

  @override
  MlxArray applyExpert(
    MlxArray input,
    int expert, {
    required Qwen3_5Config config,
  }) {
    final index = MlxArray.fromInt32List([expert], shape: [1]);
    try {
      final selected = weight.take(index, axis: 0).reshape([
        weight.shape[1],
        weight.shape[2],
      ]);
      try {
        final y = mx.matmul(input, selected.transpose());
        if (bias == null) {
          return y;
        }
        final selectedBias = bias!.take(index, axis: 0).reshape([
          1,
          bias!.shape[1],
        ]);
        try {
          return mx.add(y, selectedBias);
        } finally {
          selectedBias.close();
          y.close();
        }
      } finally {
        selected.close();
      }
    } finally {
      index.close();
    }
  }
}

final class _QuantSwitchLinear extends _SwitchLinearBase {
  const _QuantSwitchLinear(this.weight, this.scales, this.biases, this.bias);

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray? biases;
  final MlxArray? bias;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);

  @override
  MlxArray applyExpert(
    MlxArray input,
    int expert, {
    required Qwen3_5Config config,
  }) {
    final index = MlxArray.fromInt32List([expert], shape: [1]);
    try {
      final selectedWeight = weight.take(index, axis: 0).reshape([
        weight.shape[1],
        weight.shape[2],
      ]);
      final selectedScales = scales.take(index, axis: 0).reshape([
        scales.shape[1],
        scales.shape[2],
      ]);
      final selectedBiases = biases?.take(index, axis: 0).reshape([
        if (biases != null) biases!.shape[1],
        if (biases != null) biases!.shape[2],
      ]);
      final matrix = MlxQuantizedMatrix(
        selectedWeight,
        selectedScales,
        selectedBiases,
      );
      try {
        final y = mx.quant.matmul(
          input,
          matrix,
          transpose: true,
          groupSize: config.groupSize,
          bits: config.bits,
          mode: config.mode,
        );
        if (bias == null) {
          return y;
        }
        final selectedBias = bias!.take(index, axis: 0).reshape([
          1,
          bias!.shape[1],
        ]);
        try {
          return mx.add(y, selectedBias);
        } finally {
          selectedBias.close();
          y.close();
        }
      } finally {
        selectedBiases?.close();
        selectedScales.close();
        selectedWeight.close();
      }
    } finally {
      index.close();
    }
  }

  @override
  MlxArray applyExperts(
    MlxArray input,
    MlxArray experts, {
    required Qwen3_5Config config,
    bool sortedIndices = false,
  }) {
    final y = mx.quant.gatherQmm(
      input,
      matrix,
      rhsIndices: experts,
      transpose: true,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: config.mode,
      sortedIndices: sortedIndices,
    );
    if (bias == null) {
      return y;
    }
    final flatExperts = experts.reshape([experts.size]);
    try {
      final gatheredBias = bias!.take(flatExperts, axis: 0).reshape([
        ...experts.shape,
        1,
        bias!.shape[1],
      ]);
      try {
        return mx.add(y, gatheredBias);
      } finally {
        gatheredBias.close();
        y.close();
      }
    } finally {
      flatExperts.close();
    }
  }
}

final class _FullAttentionWeights {
  _FullAttentionWeights({
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
  _LinearAttentionWeights({
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
  _DenseMlpWeights({
    required this.gateProj,
    required this.upProj,
    required this.downProj,
  });

  final _LinearBase gateProj;
  final _LinearBase upProj;
  final _LinearBase downProj;
}

final class _MoeWeights {
  _MoeWeights({
    required this.gate,
    required this.switchGateProj,
    required this.switchUpProj,
    required this.switchDownProj,
    required this.sharedGateProj,
    required this.sharedUpProj,
    required this.sharedDownProj,
    required this.sharedExpertGate,
  });

  final _LinearBase gate;
  final _SwitchLinearBase switchGateProj;
  final _SwitchLinearBase switchUpProj;
  final _SwitchLinearBase switchDownProj;
  final _LinearBase sharedGateProj;
  final _LinearBase sharedUpProj;
  final _LinearBase sharedDownProj;
  final _LinearBase sharedExpertGate;
}

final class _LayerWeights {
  _LayerWeights({
    required this.inputNorm,
    required this.postNorm,
    this.fullAttention,
    this.linearAttention,
    this.denseMlp,
    this.moe,
  });

  final MlxArray inputNorm;
  final MlxArray postNorm;
  final _FullAttentionWeights? fullAttention;
  final _LinearAttentionWeights? linearAttention;
  final _DenseMlpWeights? denseMlp;
  final _MoeWeights? moe;
}
