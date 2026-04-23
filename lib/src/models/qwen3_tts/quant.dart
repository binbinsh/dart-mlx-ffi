part of 'qwen3_tts.dart';

final class _Qwen3TtsQuantLinear {
  const _Qwen3TtsQuantLinear({
    required this.weight,
    required this.scales,
    required this.biases,
    this.bias,
  });

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray biases;
  final MlxArray? bias;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);
}

final class _Qwen3TtsKvCache {
  MlxArray? keys;
  MlxArray? values;
  int offset = 0;

  void reset() {
    keys?.close();
    values?.close();
    keys = null;
    values = null;
    offset = 0;
  }

  (MlxArray, MlxArray) updateAndFetch(MlxArray nextKeys, MlxArray nextValues) {
    final currentKeys = keys;
    final currentValues = values;
    if (currentKeys == null || currentValues == null) {
      keys = nextKeys;
      values = nextValues;
      offset = nextKeys.shape[2];
      return (nextKeys, nextValues);
    }
    final mergedKeys = mx.concatenate([currentKeys, nextKeys], axis: 2);
    final mergedValues = mx.concatenate([currentValues, nextValues], axis: 2);
    currentKeys.close();
    currentValues.close();
    nextKeys.close();
    nextValues.close();
    keys = mergedKeys;
    values = mergedValues;
    offset = mergedKeys.shape[2];
    return (mergedKeys, mergedValues);
  }

  void close() => reset();
}

_Qwen3TtsQuantLinear _qLinear(Map<String, MlxArray> tensors, String prefix) {
  return _Qwen3TtsQuantLinear(
    weight: tensors['$prefix.weight']!,
    scales: tensors['$prefix.scales']!,
    biases: tensors['$prefix.biases']!,
    bias: tensors['$prefix.bias'],
  );
}

MlxArray _quantLinear(
  MlxArray input,
  _Qwen3TtsQuantLinear linear,
  Qwen3TtsQuantConfig quant, {
  required int outDim,
  required bool addBias,
}) {
  final out = mx.quant.matmul(
    input,
    linear.matrix,
    transpose: true,
    groupSize: quant.groupSize,
    bits: quant.bits,
    mode: quant.mode,
  );
  if (!addBias || linear.bias == null) {
    return out;
  }
  final bias = linear.bias!.reshape([1, outDim]);
  try {
    final added = mx.add(out, bias);
    out.close();
    return added;
  } finally {
    bias.close();
  }
}

MlxArray _silu(MlxArray input) {
  final sig = input.sigmoid();
  try {
    return input * sig;
  } finally {
    sig.close();
  }
}
