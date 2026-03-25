library;

import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'args.dart';
import 'quant.dart';

MlxArray requireTensor(Map<String, MlxArray> tensors, String key) {
  final tensor = tensors[key];
  if (tensor == null) {
    throw StateError('Missing tensor: $key');
  }
  return tensor;
}

MlxArray? lookupTensor(Map<String, MlxArray> tensors, String key) =>
    tensors[key];

MlxArray scalar(double value, {MlxDType dtype = MlxDType.MLX_FLOAT32}) =>
    MlxArray.full([], value, dtype: dtype);

MlxArray maskFillZero(MlxArray x, MlxArray mask) {
  final boolMask = mask.astype(MlxDType.MLX_BOOL);
  final zeros = x.zerosLike();
  try {
    return mx.where(boolMask, zeros, x);
  } finally {
    zeros.close();
    boolMask.close();
  }
}

MlxArray leakyRelu(MlxArray x, {double negativeSlope = 0.2}) {
  final zero = scalar(0.0);
  final slope = scalar(negativeSlope);
  final mask = MlxMore.greater(x, zero);
  final scaled = x * slope;
  try {
    return mx.where(mask, x, scaled);
  } finally {
    scaled.close();
    mask.close();
    slope.close();
    zero.close();
  }
}

MlxArray tanhGelu(MlxArray x) {
  final half = scalar(0.5);
  final one = scalar(1.0);
  final c0 = scalar(0.7978846);
  final c1 = scalar(0.044715);
  final x3 = (x * x) * x;
  final inner = x + (c1 * x3);
  final tanhArg = c0 * inner;
  final tanhTerm = tanhArg.tanh();
  final sum = one + tanhTerm;
  final left = half * x;
  try {
    return left * sum;
  } finally {
    left.close();
    sum.close();
    tanhTerm.close();
    tanhArg.close();
    inner.close();
    x3.close();
    c1.close();
    c0.close();
    one.close();
    half.close();
  }
}

final class LayerNorm {
  const LayerNorm(this.weight, this.bias, {this.eps = 1e-5});

  factory LayerNorm.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    double? eps,
  }) {
    return LayerNorm(
      requireTensor(tensors, '$prefix.weight'),
      requireTensor(tensors, '$prefix.bias'),
      eps: eps ?? 1e-5,
    );
  }

  final MlxArray weight;
  final MlxArray bias;
  final double eps;

  MlxArray call(MlxArray input) =>
      mx.fast.layerNorm(input, weight: weight, bias: bias, eps: eps);
}

sealed class Linear {
  const Linear({required this.activationQuant});

  factory Linear.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    final weight = requireTensor(tensors, '$prefix.weight');
    final scales = lookupTensor(tensors, '$prefix.scales');
    final biases = lookupTensor(tensors, '$prefix.biases');
    final bias = lookupTensor(tensors, '$prefix.bias');
    if (scales == null) {
      return DenseLinear(weight, bias, activationQuant: activationQuant);
    }
    return QuantLinear(
      weight,
      scales,
      biases,
      bias,
      quant: quant,
      activationQuant: activationQuant,
    );
  }

  final bool activationQuant;

  MlxArray call(MlxArray input);
}

final class DenseLinear extends Linear {
  const DenseLinear(this.weight, this.bias, {required super.activationQuant});

  final MlxArray weight;
  final MlxArray? bias;

  @override
  MlxArray call(MlxArray input) {
    final x = maybeFakeQuant(input, activationQuant);
    final y = mx.matmul(x, weight.transpose());
    if (identical(x, input) && bias == null) {
      return y;
    }
    if (!identical(x, input)) {
      x.close();
    }
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([
      ...List<int>.filled(y.ndim - 1, 1),
      bias!.shape[0],
    ]);
    try {
      return y + reshapedBias;
    } finally {
      reshapedBias.close();
      y.close();
    }
  }
}

final class QuantLinear extends Linear {
  QuantLinear(
    this.weight,
    this.scales,
    this.biases,
    this.bias, {
    required this.quant,
    required super.activationQuant,
  });

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray? biases;
  final MlxArray? bias;
  final KittenQuantConfig quant;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);

  @override
  MlxArray call(MlxArray input) {
    final x = maybeFakeQuant(input, activationQuant);
    final y = mx.quant.matmul(
      x,
      matrix,
      transpose: true,
      groupSize: quant.groupSize,
      bits: quant.bits,
      mode: quant.mode,
    );
    if (!identical(x, input)) {
      x.close();
    }
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([
      ...List<int>.filled(y.ndim - 1, 1),
      bias!.shape[0],
    ]);
    try {
      return y + reshapedBias;
    } finally {
      reshapedBias.close();
      y.close();
    }
  }
}

final class LinearNorm {
  LinearNorm(this.linearLayer);

  factory LinearNorm.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    return LinearNorm(
      Linear.load(
        tensors,
        '$prefix.linear_layer',
        quant: quant,
        activationQuant: activationQuant,
      ),
    );
  }

  final Linear linearLayer;

  MlxArray call(MlxArray input) => linearLayer(input);
}

final class Embedding {
  Embedding._(this.weight);

  factory Embedding.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required KittenQuantConfig quant,
    MlxDType dtype = MlxDType.MLX_FLOAT32,
  }) {
    final weight = requireTensor(tensors, '$prefix.weight');
    final scales = lookupTensor(tensors, '$prefix.scales');
    final biases = lookupTensor(tensors, '$prefix.biases');
    if (scales == null) {
      return Embedding._(weight);
    }
    final dequantized = mx.quant.dequantize(
      MlxQuantizedMatrix(weight, scales, biases),
      groupSize: quant.groupSize,
      bits: quant.bits,
      mode: quant.mode,
      dtype: dtype,
    );
    return Embedding._(dequantized);
  }

  final MlxArray weight;

  MlxArray call(MlxArray inputIds) {
    final flat = inputIds.flatten();
    final gathered = weight.take(flat, axis: 0);
    final shape = <int>[...inputIds.shape, weight.shape[1]];
    try {
      return gathered.reshape(shape);
    } finally {
      flat.close();
      gathered.close();
    }
  }
}

final class Conv1d {
  const Conv1d(this.weight, this.bias, {this.activationQuant = false});

  factory Conv1d.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    bool activationQuant = false,
  }) {
    return Conv1d(
      requireTensor(tensors, '$prefix.weight'),
      lookupTensor(tensors, '$prefix.bias'),
      activationQuant: activationQuant,
    );
  }

  final MlxArray weight;
  final MlxArray? bias;
  final bool activationQuant;

  MlxArray call(
    MlxArray input, {
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
  }) {
    final x = maybeFakeQuant(input, activationQuant);
    final y = mx.conv1d(
      x,
      weight,
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups,
    );
    if (!identical(x, input)) {
      x.close();
    }
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([1, 1, bias!.shape[0]]);
    try {
      return y + reshapedBias;
    } finally {
      reshapedBias.close();
      y.close();
    }
  }
}

MlxArray _weightNorm(MlxArray weightV, MlxArray weightG) {
  final square = weightV.square();
  final sumK = square.sum(axis: 1, keepDims: true);
  final norm = sumK.sum(axis: 2, keepDims: true).sqrt();
  final eps = scalar(1e-7);
  final denom = norm + eps;
  final normalized = weightV / denom;
  try {
    return normalized * weightG;
  } finally {
    normalized.close();
    denom.close();
    eps.close();
    norm.close();
    sumK.close();
    square.close();
  }
}

final class ConvWeighted {
  const ConvWeighted(
    this.weightG,
    this.weightV, {
    required this.stride,
    required this.padding,
    required this.dilation,
    required this.groups,
    this.bias,
    this.activationQuant = false,
  });

  factory ConvWeighted.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    int stride = 1,
    int padding = 1,
    int dilation = 1,
    int groups = 1,
    bool activationQuant = false,
  }) {
    return ConvWeighted(
      requireTensor(tensors, '$prefix.weight_g'),
      requireTensor(tensors, '$prefix.weight_v'),
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups,
      bias: lookupTensor(tensors, '$prefix.bias'),
      activationQuant: activationQuant,
    );
  }

  final MlxArray weightG;
  final MlxArray weightV;
  final int stride;
  final int padding;
  final int dilation;
  final int groups;
  final MlxArray? bias;
  final bool activationQuant;

  MlxArray call(MlxArray input, {bool transpose = false}) {
    final x = maybeFakeQuant(input, activationQuant);
    final weight = _weightNorm(weightV, weightG);
    final useAsIs = x.shape.last == weight.shape.last || groups > 1;
    final effectiveWeight = useAsIs
        ? weight
        : mx.transposeAxes(weight, [2, 1, 0]);
    final y = transpose
        ? mx.convTranspose1d(
            x,
            effectiveWeight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
          )
        : mx.conv1d(
            x,
            effectiveWeight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
          );
    if (!identical(effectiveWeight, weight)) {
      effectiveWeight.close();
    }
    weight.close();
    if (!identical(x, input)) {
      x.close();
    }
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([1, 1, bias!.shape[0]]);
    try {
      return y + reshapedBias;
    } finally {
      reshapedBias.close();
      y.close();
    }
  }
}

final class InstanceNorm1d {
  const InstanceNorm1d(this.numFeatures, {this.eps = 1e-5});

  final int numFeatures;
  final double eps;

  MlxArray call(MlxArray input) {
    if (input.ndim == 2) {
      final expanded = input.expandDims(0);
      final normalized = call(expanded);
      try {
        return normalized.squeeze();
      } finally {
        expanded.close();
        normalized.close();
      }
    }
    if (input.ndim != 3) {
      throw ArgumentError('InstanceNorm1d expects 2D or 3D input.');
    }
    final mean = input.mean(axis: 2, keepDims: true);
    final variance = mx.variance(input, axis: 2, keepDims: true);
    final epsArray = scalar(eps);
    final centered = input - mean;
    final denom = (variance + epsArray).sqrt();
    try {
      return centered / denom;
    } finally {
      denom.close();
      centered.close();
      epsArray.close();
      variance.close();
      mean.close();
    }
  }
}

final class AdaIN1d {
  AdaIN1d(this.norm, this.fc, {this.activationQuant = false});

  factory AdaIN1d.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int numFeatures,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    return AdaIN1d(
      InstanceNorm1d(numFeatures),
      Linear.load(
        tensors,
        '$prefix.fc',
        quant: quant,
        activationQuant: activationQuant,
      ),
      activationQuant: activationQuant,
    );
  }

  final InstanceNorm1d norm;
  final Linear fc;
  final bool activationQuant;

  MlxArray call(MlxArray input, MlxArray style) {
    final styleIn = maybeFakeQuant(style, activationQuant);
    final affine = fc(styleIn);
    if (!identical(styleIn, style)) {
      styleIn.close();
    }
    final expanded = affine.expandDims(2);
    final split = mx.splitSections(expanded, [expanded.shape[1] ~/ 2], axis: 1);
    final normalized = norm(input);
    final one = scalar(1.0);
    final scaled = (one + split[0]) * normalized;
    try {
      return scaled + split[1];
    } finally {
      scaled.close();
      one.close();
      normalized.close();
      for (final item in split) {
        item.close();
      }
      expanded.close();
      affine.close();
    }
  }
}

final class AdainResBlk1d {
  AdainResBlk1d({
    required this.dimIn,
    required this.dimOut,
    required this.conv1,
    required this.conv2,
    required this.norm1,
    required this.norm2,
    required this.upsample,
    this.conv1x1,
    this.pool,
  }) : learnedShortcut = dimIn != dimOut;

  factory AdainResBlk1d.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int dimIn,
    required int dimOut,
    required int styleDim,
    bool upsample = false,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    return AdainResBlk1d(
      dimIn: dimIn,
      dimOut: dimOut,
      conv1: ConvWeighted.load(
        tensors,
        '$prefix.conv1',
        padding: 1,
        activationQuant: activationQuant,
      ),
      conv2: ConvWeighted.load(
        tensors,
        '$prefix.conv2',
        padding: 1,
        activationQuant: activationQuant,
      ),
      norm1: AdaIN1d.load(
        tensors,
        '$prefix.norm1',
        numFeatures: dimIn,
        quant: quant,
        activationQuant: activationQuant,
      ),
      norm2: AdaIN1d.load(
        tensors,
        '$prefix.norm2',
        numFeatures: dimOut,
        quant: quant,
        activationQuant: activationQuant,
      ),
      upsample: upsample,
      conv1x1: dimIn == dimOut
          ? null
          : ConvWeighted.load(
              tensors,
              '$prefix.conv1x1',
              padding: 0,
              activationQuant: activationQuant,
            ),
      pool: upsample
          ? ConvWeighted.load(
              tensors,
              '$prefix.pool',
              stride: 2,
              padding: 1,
              groups: dimIn,
              activationQuant: activationQuant,
            )
          : null,
    );
  }

  final int dimIn;
  final int dimOut;
  final bool learnedShortcut;
  final ConvWeighted conv1;
  final ConvWeighted conv2;
  final AdaIN1d norm1;
  final AdaIN1d norm2;
  final ConvWeighted? conv1x1;
  final ConvWeighted? pool;
  final bool upsample;

  MlxArray call(MlxArray input, MlxArray style) {
    final residual = _residual(input, style);
    final shortcut = _shortcut(input);
    final sum = residual + shortcut;
    final denom = scalar(math.sqrt(2.0));
    try {
      return sum / denom;
    } finally {
      denom.close();
      sum.close();
      shortcut.close();
      residual.close();
    }
  }

  MlxArray _shortcut(MlxArray input) {
    var x = mx.transposeAxes(input, [0, 2, 1]);
    if (upsample) {
      final repeated = x.repeat(2, axis: 1);
      x.close();
      x = repeated;
    }
    final back = mx.transposeAxes(x, [0, 2, 1]);
    x.close();
    if (!learnedShortcut) {
      return back;
    }
    final convIn = mx.transposeAxes(back, [0, 2, 1]);
    back.close();
    final convOut = conv1x1!.call(convIn);
    convIn.close();
    final restored = mx.transposeAxes(convOut, [0, 2, 1]);
    convOut.close();
    return restored;
  }

  MlxArray _residual(MlxArray input, MlxArray style) {
    var x = norm1(input, style);
    final act1 = leakyRelu(x);
    x.close();
    x = act1;

    var work = mx.transposeAxes(x, [0, 2, 1]);
    x.close();
    if (upsample) {
      final pooled = pool!.call(work, transpose: true);
      work.close();
      final zero = scalar(0.0, dtype: pooled.dtype);
      final padded = mx.pad(
        pooled,
        axes: [1],
        lowPads: [0],
        highPads: [1],
        padValue: zero,
      );
      zero.close();
      pooled.close();
      work = padded;
    }
    x = mx.transposeAxes(work, [0, 2, 1]);
    work.close();

    work = mx.transposeAxes(x, [0, 2, 1]);
    x.close();
    final conv1Out = conv1.call(work);
    work.close();
    x = mx.transposeAxes(conv1Out, [0, 2, 1]);
    conv1Out.close();

    final normed = norm2(x, style);
    x.close();
    x = leakyRelu(normed);
    normed.close();

    work = mx.transposeAxes(x, [0, 2, 1]);
    x.close();
    final conv2Out = conv2.call(work);
    work.close();
    final out = mx.transposeAxes(conv2Out, [0, 2, 1]);
    conv2Out.close();
    return out;
  }
}
