part of 'qwen3_tts.dart';

({MlxArray cos, MlxArray sin}) _standardRopeCosSin(
  int seqLen, {
  required int offset,
  required int headDim,
  required double base,
  required MlxDType dtype,
}) {
  final halfDim = headDim ~/ 2;
  final invFreq = <double>[
    for (var i = 0; i < headDim; i += 2) math.exp(-(i / headDim) * math.log(base)),
  ];
  final freqs = <double>[
    for (var pos = 0; pos < seqLen; pos++)
      for (var i = 0; i < halfDim; i++) (offset + pos) * invFreq[i],
  ];
  final freqArr = MlxArray.fromFloat32List(freqs, shape: [1, seqLen, halfDim]);
  try {
    final emb = mx.concatenate([freqArr, freqArr], axis: 2);
    try {
      return (cos: emb.cos().astype(dtype), sin: emb.sin().astype(dtype));
    } finally {
      emb.close();
    }
  } finally {
    freqArr.close();
  }
}

({MlxArray q, MlxArray k}) _applyStandardRope(
  MlxArray q,
  MlxArray k,
  MlxArray cos,
  MlxArray sin,
) {
  final cos4 = cos.expandDims(1);
  final sin4 = sin.expandDims(1);
  try {
    final qRot = _rotateHalf(q);
    final kRot = _rotateHalf(k);
    try {
      return (q: (q * cos4) + (qRot * sin4), k: (k * cos4) + (kRot * sin4));
    } finally {
      qRot.close();
      kRot.close();
    }
  } finally {
    cos4.close();
    sin4.close();
  }
}

MlxArray _rotateHalf(MlxArray input) {
  final half = input.shape[3] ~/ 2;
  final x1 = input.slice(
    start: [0, 0, 0, 0],
    stop: [input.shape[0], input.shape[1], input.shape[2], half],
  );
  final x2 = input.slice(
    start: [0, 0, 0, half],
    stop: [input.shape[0], input.shape[1], input.shape[2], input.shape[3]],
  );
  try {
    final negX2 = x2.negative();
    try {
      return mx.concatenate([negX2, x1], axis: 3);
    } finally {
      negX2.close();
    }
  } finally {
    x1.close();
    x2.close();
  }
}

MlxArray _createCausalMask(int seqLen, MlxDType dtype) {
  final mask = MlxArray.full([seqLen, seqLen], -1e9, dtype: dtype);
  final causal = mask.triu(k: 1);
  mask.close();
  final expanded = causal.reshape([1, 1, seqLen, seqLen]);
  causal.close();
  return expanded;
}

MlxArray _createCausalMaskWithPrefix({
  required int seqLen,
  required int prefixLen,
  required MlxDType dtype,
}) {
  final left = MlxArray.zeros([seqLen, prefixLen], dtype: dtype);
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
