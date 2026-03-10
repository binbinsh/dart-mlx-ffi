library;

import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

MlxArray vlmSilu(MlxArray input) {
  final sig = input.sigmoid();
  try {
    return input * sig;
  } finally {
    sig.close();
  }
}

MlxArray vlmSliceStep(MlxArray input, int index, List<int> shape) => input
    .slice(
      start: [0, index, 0, if (shape.length == 4) 0],
      stop: [1, index + 1, shape[1], if (shape.length == 4) shape[2]],
    )
    .reshape(shape);

({MlxArray output, MlxArray state}) vlmGatedDelta({
  required MlxArray q,
  required MlxArray k,
  required MlxArray v,
  required MlxArray a,
  required MlxArray b,
  required MlxArray aLog,
  required MlxArray dtBias,
  required int linearNumValueHeads,
  required int linearNumKeyHeads,
  required int linearValueHeadDim,
  required int linearKeyHeadDim,
}) {
  final repeatFactor = linearNumValueHeads ~/ linearNumKeyHeads;
  MlxArray qHeads = q;
  MlxArray kHeads = k;
  if (repeatFactor > 1) {
    qHeads = q.repeat(repeatFactor, axis: 2);
    kHeads = k.repeat(repeatFactor, axis: 2);
    q.close();
    k.close();
  }

  final beta = b.sigmoid();
  final dt = a + dtBias.reshape([1, 1, linearNumValueHeads]).broadcastTo(a.shape);
  final zeros = MlxArray.zeros(dt.shape, dtype: dt.dtype);
  final softplus = mx.logaddexp(dt, zeros);
  zeros.close();
  dt.close();
  final expA = mx.exp(aLog.astype(MlxDType.MLX_FLOAT32)).astype(a.dtype);
  final scaled =
      expA.reshape([1, 1, linearNumValueHeads]).broadcastTo(a.shape) * softplus;
  final negative = scaled.negative();
  scaled.close();
  final g = negative.exp();
  negative.close();
  expA.close();
  softplus.close();
  a.close();

  var state = MlxArray.zeros([
    1,
    linearNumValueHeads,
    linearValueHeadDim,
    linearKeyHeadDim,
  ], dtype: qHeads.dtype);
  final outputs = <MlxArray>[];
  for (var index = 0; index < qHeads.shape[1]; index++) {
    final qStep = vlmSliceStep(qHeads, index, [
      1,
      linearNumValueHeads,
      linearKeyHeadDim,
    ]);
    final kStep = vlmSliceStep(kHeads, index, [
      1,
      linearNumValueHeads,
      linearKeyHeadDim,
    ]);
    final vStep = vlmSliceStep(v, index, [
      1,
      linearNumValueHeads,
      linearValueHeadDim,
    ]);
    final gStep = vlmSliceStep(g, index, [1, linearNumValueHeads]);
    final betaStep = vlmSliceStep(beta, index, [1, linearNumValueHeads]);

    final decay = gStep.reshape([1, linearNumValueHeads, 1, 1]);
    final decayed = state * decay;
    state.close();
    decay.close();
    final kExpanded = kStep.reshape([
      1,
      linearNumValueHeads,
      1,
      linearKeyHeadDim,
    ]);
    final kvMem = (decayed * kExpanded).sum(axis: 3);
    final delta =
        (vStep - kvMem) * betaStep.reshape([1, linearNumValueHeads, 1]);
    kvMem.close();
    betaStep.close();
    final newState =
        decayed +
        kExpanded *
            delta.reshape([
              1,
              linearNumValueHeads,
              linearValueHeadDim,
              1,
            ]);
    decayed.close();
    delta.close();
    final y =
        (newState *
                qStep.reshape([
                  1,
                  linearNumValueHeads,
                  1,
                  linearKeyHeadDim,
                ]))
            .sum(axis: 3)
            .reshape([
              1,
              1,
              linearNumValueHeads,
              linearValueHeadDim,
            ]);
    outputs.add(y);
    qStep.close();
    kStep.close();
    vStep.close();
    gStep.close();
    kExpanded.close();
    state = newState;
  }
  qHeads.close();
  kHeads.close();
  v.close();
  g.close();
  beta.close();
  final output = mx.concatenate(outputs, axis: 1);
  for (final value in outputs) {
    value.close();
  }
  return (output: output, state: state);
}

double vlmInverseSqrt(int value) => 1 / math.sqrt(value.toDouble());
