/// MLX implementation of the ECAPA-TDNN speaker embedding backbone.
///
/// All modules here consume and return MLX arrays in **NTC** layout
/// `(batch, time, channels)` to align with MLX's native `mx.conv1d` signature
/// (`input (N,L,C_in)`, `weight (C_out,kW,C_in)`). This differs from the
/// SpeechBrain reference implementation which works in channel-first `NCL`
/// throughout; parity is preserved by matching:
///
/// * Conv weights are pre-transposed at export from `(C_out,C_in,kW)` to
///   `(C_out,kW,C_in)`, so MLX conv is a single call with `padding=0`.
/// * SpeechBrain `padding='same'` with `padding_mode='reflect'` is replicated
///   with an explicit [mx.pad] using `mode: 'reflect'` around time axis 1.
///   The pad amount per side is `floor(dilation * (kernel_size - 1) / 2)`;
///   only the four dilated convs (block 0, and res2net inside blocks 1..3)
///   need any padding at all.
/// * BatchNorm1d is pre-fused into `(scale, bias)` pairs so each norm is a
///   single `x * scale + bias` broadcast over the channel axis.
/// * AttentiveStatisticsPooling follows the `lengths=None` fast path
///   (uniform mask), so the attn mask is all-ones and the mask-fill step
///   becomes a no-op. This matches `encode_batch` on a single clip.
library;

import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

/// Fused (scale, bias) pair replacing a BatchNorm1d in eval mode.
class FusedBn {
  const FusedBn({required this.scale, required this.bias});

  final MlxArray scale;
  final MlxArray bias;

  /// `y = x * scale + bias` broadcast across all but the channel axis.
  /// [input] is `(B, T, C)` and the scale/bias tensors are `(C,)` so MLX
  /// broadcasting handles the rest.
  MlxArray apply(MlxArray input) {
    final scaled = mx.multiply(input, scale);
    final biased = mx.add(scaled, bias);
    scaled.close();
    return biased;
  }
}

/// Reflect-pad `input` along the time axis (axis 1, NTC layout) by [pad]
/// positions on each side. Matches PyTorch `F.pad(..., mode='reflect')`,
/// which reflects *without* repeating the border. MLX's own `mx.pad` only
/// supports `constant` and `edge`, so we build the mirrored region with a
/// gather (`take`) over the time axis.
MlxArray _reflectPadTime(MlxArray input, int pad) {
  if (pad <= 0) return input;
  final t = input.shape[1];
  if (pad >= t) {
    throw ArgumentError(
      'reflect pad ($pad) must be smaller than time length ($t).',
    );
  }
  // Build the gather index: [pad, pad-1, ..., 1, 0, 1, ..., T-1, T-2, ..., T-pad-1].
  final total = t + 2 * pad;
  final idx = List<int>.filled(total, 0);
  for (var i = 0; i < pad; i++) {
    idx[i] = pad - i; // left mirror
  }
  for (var i = 0; i < t; i++) {
    idx[pad + i] = i; // original
  }
  for (var i = 0; i < pad; i++) {
    idx[pad + t + i] = t - 2 - i; // right mirror
  }
  final idxArr = MlxArray.fromInt32List(idx, shape: [total]);
  try {
    return input.take(idxArr, axis: 1);
  } finally {
    idxArr.close();
  }
}

/// Conv1d wrapper that mirrors SpeechBrain `Conv1d(padding='same',
/// padding_mode='reflect')` when `padSame=true`. [weight] must be in MLX
/// layout `(C_out, kW, C_in)` and [bias] must be `(C_out,)`.
MlxArray speechBrainConv1d(
  MlxArray input, {
  required MlxArray weight,
  required MlxArray bias,
  required int kernelSize,
  required int dilation,
  required bool padSame,
}) {
  MlxArray? padded;
  MlxArray? conv;
  MlxArray? added;
  try {
    var x = input;
    if (padSame && kernelSize > 1) {
      // SpeechBrain get_padding_elem reduces to floor(d*(k-1)/2) each side
      // for stride=1 and the kernels/dilations used in ECAPA-TDNN.
      final pad = (dilation * (kernelSize - 1)) ~/ 2;
      if (pad > 0) {
        padded = _reflectPadTime(input, pad);
        x = padded;
      }
    }
    conv = mx.conv1d(x, weight, stride: 1, padding: 0, dilation: dilation);
    added = mx.add(conv, bias);
    return added;
  } catch (_) {
    added?.close();
    rethrow;
  } finally {
    padded?.close();
    conv?.close();
    // `added` is returned so don't close it here.
  }
}

/// A `TDNNBlock` = Conv1d -> ReLU -> BatchNorm1d. Returns `(B, T, C_out)`.
MlxArray tdnnBlock(
  MlxArray input, {
  required MlxArray weight,
  required MlxArray bias,
  required FusedBn bn,
  required int kernelSize,
  required int dilation,
}) {
  MlxArray? conv;
  MlxArray? relu;
  try {
    conv = speechBrainConv1d(
      input,
      weight: weight,
      bias: bias,
      kernelSize: kernelSize,
      dilation: dilation,
      padSame: true,
    );
    relu = mx.maximum(conv, MlxArray.full([], 0.0));
    final out = bn.apply(relu);
    return out;
  } finally {
    conv?.close();
    // If relu is created, close it after bn.apply consumed it.
    if (relu != null) relu.close();
  }
}

/// Res2NetBlock: chunk channel axis into `scale` groups. First chunk is
/// passed through untouched; remaining chunks are fed through a TDNNBlock,
/// with a residual accumulator `y_{i} = block_{i}(x_{i} + y_{i-1})` starting
/// at i=2.
MlxArray res2netBlock(
  MlxArray input, {
  required List<MlxArray> weights,
  required List<MlxArray> biases,
  required List<FusedBn> bns,
  required int scale,
  required int kernelSize,
  required int dilation,
}) {
  // Input is (B, T, C). Channel axis = 2.
  final c = input.shape[2];
  if (c % scale != 0) {
    throw StateError('Res2Net expected channels divisible by scale=$scale.');
  }
  final chunk = c ~/ scale;
  final parts = <MlxArray>[];
  MlxArray? running;
  try {
    for (var i = 0; i < scale; i++) {
      final xi = input.slice(
        start: [0, 0, i * chunk],
        stop: [input.shape[0], input.shape[1], (i + 1) * chunk],
      );
      MlxArray yi;
      if (i == 0) {
        yi = xi;
      } else if (i == 1) {
        yi = tdnnBlock(
          xi,
          weight: weights[i - 1],
          bias: biases[i - 1],
          bn: bns[i - 1],
          kernelSize: kernelSize,
          dilation: dilation,
        );
        xi.close();
      } else {
        final summed = mx.add(xi, running!);
        xi.close();
        yi = tdnnBlock(
          summed,
          weight: weights[i - 1],
          bias: biases[i - 1],
          bn: bns[i - 1],
          kernelSize: kernelSize,
          dilation: dilation,
        );
        summed.close();
      }
      if (i >= 1) {
        // `running` aliases parts[i-1] already (it is owned by parts),
        // so we do NOT close it here — reassignment is enough.
        running = yi;
      }
      parts.add(yi);
    }
    return mx.concatenate(parts, axis: 2);
  } finally {
    for (final p in parts) {
      p.close();
    }
    // `running` aliases the last `parts` entry which has already been closed.
  }
}

/// SEBlock: squeeze-and-excite with sigmoid scaling.
/// `s = mean(x over time)` -> conv1 -> ReLU -> conv2 -> sigmoid -> x * s.
MlxArray seBlock(
  MlxArray input, {
  required MlxArray conv1W,
  required MlxArray conv1B,
  required MlxArray conv2W,
  required MlxArray conv2B,
}) {
  // input: (B, T, C)
  MlxArray? pooled;
  MlxArray? c1;
  MlxArray? relu;
  MlxArray? c2;
  MlxArray? sig;
  try {
    pooled = input.mean(axis: 1, keepDims: true); // (B, 1, C)
    c1 = speechBrainConv1d(
      pooled,
      weight: conv1W,
      bias: conv1B,
      kernelSize: 1,
      dilation: 1,
      padSame: false,
    );
    relu = mx.maximum(c1, MlxArray.full([], 0.0));
    c2 = speechBrainConv1d(
      relu,
      weight: conv2W,
      bias: conv2B,
      kernelSize: 1,
      dilation: 1,
      padSame: false,
    );
    sig = c2.sigmoid();
    // Broadcast (B,1,C) * (B,T,C) = (B,T,C).
    return mx.multiply(input, sig);
  } finally {
    pooled?.close();
    c1?.close();
    relu?.close();
    c2?.close();
    sig?.close();
  }
}

/// SERes2NetBlock forward. Since every retained block has `in==out`, the
/// shortcut is identity.
MlxArray seRes2NetBlock(
  MlxArray input, {
  required MlxArray tdnn1W,
  required MlxArray tdnn1B,
  required FusedBn tdnn1Bn,
  required List<MlxArray> resWeights,
  required List<MlxArray> resBiases,
  required List<FusedBn> resBns,
  required MlxArray tdnn2W,
  required MlxArray tdnn2B,
  required FusedBn tdnn2Bn,
  required MlxArray seConv1W,
  required MlxArray seConv1B,
  required MlxArray seConv2W,
  required MlxArray seConv2B,
  required int res2netScale,
  required int kernelSize,
  required int dilation,
}) {
  MlxArray? a;
  MlxArray? b;
  MlxArray? c;
  MlxArray? d;
  try {
    a = tdnnBlock(
      input,
      weight: tdnn1W,
      bias: tdnn1B,
      bn: tdnn1Bn,
      kernelSize: 1,
      dilation: 1,
    );
    b = res2netBlock(
      a,
      weights: resWeights,
      biases: resBiases,
      bns: resBns,
      scale: res2netScale,
      kernelSize: kernelSize,
      dilation: dilation,
    );
    c = tdnnBlock(
      b,
      weight: tdnn2W,
      bias: tdnn2B,
      bn: tdnn2Bn,
      kernelSize: 1,
      dilation: 1,
    );
    d = seBlock(
      c,
      conv1W: seConv1W,
      conv1B: seConv1B,
      conv2W: seConv2W,
      conv2B: seConv2B,
    );
    return mx.add(d, input);
  } finally {
    a?.close();
    b?.close();
    c?.close();
    d?.close();
  }
}

/// Attentive statistics pooling (global_context=True, lengths=None).
///
/// 1. Global mean/std of `x` over time (mask all ones, eps-clamped).
/// 2. Broadcast mean/std to `(B, T, C)` and concat with x along C -> `(B,T,3C)`.
/// 3. `attn = conv(tanh(tdnn(attn)))` producing `(B, T, C)`.
/// 4. softmax over time.
/// 5. Weighted mean + std over time -> `(B, 1, 2C)`.
MlxArray attentiveStatisticsPool(
  MlxArray input, {
  required MlxArray tdnnW,
  required MlxArray tdnnB,
  required FusedBn tdnnBn,
  required MlxArray convW,
  required MlxArray convB,
  required double eps,
}) {
  // input: (B, T, C). C is the speaker channel dimension.
  final b = input.shape[0];
  final t = input.shape[1];
  final c = input.shape[2];
  MlxArray? mean1;
  MlxArray? centered;
  MlxArray? sq;
  MlxArray? varRaw;
  MlxArray? varClamped;
  MlxArray? std1;
  MlxArray? meanBcast;
  MlxArray? stdBcast;
  MlxArray? concat;
  MlxArray? tdnnOut;
  MlxArray? tanh;
  MlxArray? conv;
  MlxArray? attn;
  MlxArray? weighted;
  MlxArray? weightedMean;
  MlxArray? diff;
  MlxArray? diffSq;
  MlxArray? weightedDiffSq;
  MlxArray? varAttn;
  MlxArray? varAttnClamped;
  MlxArray? stdAttn;
  MlxArray? pooled;
  try {
    // 1. Global stats (mask uniform 1/T).
    mean1 = input.mean(axis: 1, keepDims: true); // (B,1,C)
    final centeredTmp = mx.subtract(input, mean1);
    centered = centeredTmp;
    sq = mx.multiply(centered, centered);
    varRaw = sq.mean(axis: 1, keepDims: true); // (B,1,C)
    varClamped = mx.maximum(varRaw, MlxArray.full([], eps));
    std1 = mx.sqrt(varClamped);

    // 2. Broadcast mean/std to (B,T,C) then concat along C.
    meanBcast = mx.broadcastTo(mean1, [b, t, c]);
    stdBcast = mx.broadcastTo(std1, [b, t, c]);
    concat = mx.concatenate([input, meanBcast, stdBcast], axis: 2);

    // 3. attn = conv(tanh(tdnn(attn)))
    tdnnOut = tdnnBlock(
      concat,
      weight: tdnnW,
      bias: tdnnB,
      bn: tdnnBn,
      kernelSize: 1,
      dilation: 1,
    );
    tanh = tdnnOut.tanh();
    conv = speechBrainConv1d(
      tanh,
      weight: convW,
      bias: convB,
      kernelSize: 1,
      dilation: 1,
      padSame: false,
    );

    // 4. Softmax over time. In NTC, time axis = 1.
    attn = mx.softmax(conv, axis: 1);

    // 5. Weighted mean + std.
    weighted = mx.multiply(input, attn);
    weightedMean = weighted.sum(axis: 1, keepDims: true); // (B,1,C)
    diff = mx.subtract(input, weightedMean);
    diffSq = mx.multiply(diff, diff);
    weightedDiffSq = mx.multiply(diffSq, attn);
    varAttn = weightedDiffSq.sum(axis: 1, keepDims: true);
    varAttnClamped = mx.maximum(varAttn, MlxArray.full([], eps));
    stdAttn = mx.sqrt(varAttnClamped);

    pooled = mx.concatenate([weightedMean, stdAttn], axis: 2);
    return pooled;
  } catch (_) {
    pooled?.close();
    rethrow;
  } finally {
    mean1?.close();
    centered?.close();
    sq?.close();
    varRaw?.close();
    varClamped?.close();
    std1?.close();
    meanBcast?.close();
    stdBcast?.close();
    concat?.close();
    tdnnOut?.close();
    tanh?.close();
    conv?.close();
    attn?.close();
    weighted?.close();
    weightedMean?.close();
    diff?.close();
    diffSq?.close();
    weightedDiffSq?.close();
    varAttn?.close();
    varAttnClamped?.close();
    stdAttn?.close();
    // pooled is the returned array, not closed.
  }
}

/// Minimal helper: natural log of 10 for log10 via ln.
const double ln10 = math.ln10;
