library;

import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'config.dart';
import 'core.dart';
import 'quant.dart';

final class KittenSourceOutput {
  const KittenSourceOutput({
    required this.sineMerge,
    required this.noise,
    required this.uv,
  });

  final MlxArray sineMerge;
  final MlxArray noise;
  final MlxArray uv;

  void close() {
    uv.close();
    noise.close();
    sineMerge.close();
  }
}

final class KittenGeneratorProjection {
  const KittenGeneratorProjection({
    required this.spec,
    required this.phase,
  });

  final MlxArray spec;
  final MlxArray phase;

  void close() {
    phase.close();
    spec.close();
  }
}

final class KittenConv1dLayer {
  const KittenConv1dLayer({
    required this.weight,
    this.bias,
    this.stride = 1,
    this.padding = 0,
    this.dilation = 1,
    this.groups = 1,
    this.activationQuant = false,
  });

  factory KittenConv1dLayer.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
    bool activationQuant = false,
  }) {
    return KittenConv1dLayer(
      weight: requireTensor(tensors, '$prefix.weight'),
      bias: lookupTensor(tensors, '$prefix.bias'),
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups,
      activationQuant: activationQuant,
    );
  }

  final MlxArray weight;
  final MlxArray? bias;
  final int stride;
  final int padding;
  final int dilation;
  final int groups;
  final bool activationQuant;

  MlxArray call(MlxArray input) {
    final x = maybeFakeQuant(input, activationQuant);
    final useAsIs = x.shape.last == weight.shape.last || groups > 1;
    final effectiveWeight =
        useAsIs ? weight : mx.transposeAxes(weight, [2, 1, 0]);
    final y = mx.conv1d(
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

final class AdaINResBlock1 {
  const AdaINResBlock1({
    required this.convs1,
    required this.convs2,
    required this.adain1,
    required this.adain2,
    required this.alpha1,
    required this.alpha2,
  });

  factory AdaINResBlock1.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int channels,
    required KittenQuantConfig quant,
    required int kernelSize,
    required List<int> dilation,
    bool activationQuant = false,
  }) {
    return AdaINResBlock1(
      convs1: List<ConvWeighted>.generate(
        3,
        (index) => ConvWeighted.load(
          tensors,
          '$prefix.convs1.$index',
          padding: getPadding1d(kernelSize, dilation[index]),
          dilation: dilation[index],
          activationQuant: activationQuant,
        ),
        growable: false,
      ),
      convs2: List<ConvWeighted>.generate(
        3,
        (index) => ConvWeighted.load(
          tensors,
          '$prefix.convs2.$index',
          padding: getPadding1d(kernelSize, 1),
          activationQuant: activationQuant,
        ),
        growable: false,
      ),
      adain1: List<AdaIN1d>.generate(
        3,
        (index) => AdaIN1d.load(
          tensors,
          '$prefix.adain1.$index',
          numFeatures: channels,
          quant: quant,
          activationQuant: activationQuant,
        ),
        growable: false,
      ),
      adain2: List<AdaIN1d>.generate(
        3,
        (index) => AdaIN1d.load(
          tensors,
          '$prefix.adain2.$index',
          numFeatures: channels,
          quant: quant,
          activationQuant: activationQuant,
        ),
        growable: false,
      ),
      alpha1: <MlxArray>[
        requireTensor(tensors, '$prefix.alpha1_0'),
        requireTensor(tensors, '$prefix.alpha1_1'),
        requireTensor(tensors, '$prefix.alpha1_2'),
      ],
      alpha2: <MlxArray>[
        requireTensor(tensors, '$prefix.alpha2_0'),
        requireTensor(tensors, '$prefix.alpha2_1'),
        requireTensor(tensors, '$prefix.alpha2_2'),
      ],
    );
  }

  final List<ConvWeighted> convs1;
  final List<ConvWeighted> convs2;
  final List<AdaIN1d> adain1;
  final List<AdaIN1d> adain2;
  final List<MlxArray> alpha1;
  final List<MlxArray> alpha2;

  MlxArray call(MlxArray input, MlxArray style) {
    var current = input;
    for (var index = 0; index < convs1.length; index++) {
      final norm1 = adain1[index](current, style);
      final snake1 = snake1d(norm1, alpha1[index]);
      final conv1In = mx.transposeAxes(snake1, [0, 2, 1]);
      final conv1Out = convs1[index](conv1In);
      final afterConv1 = mx.transposeAxes(conv1Out, [0, 2, 1]);
      norm1.close();
      snake1.close();
      conv1In.close();
      conv1Out.close();

      final norm2 = adain2[index](afterConv1, style);
      final snake2 = snake1d(norm2, alpha2[index]);
      final conv2In = mx.transposeAxes(snake2, [0, 2, 1]);
      final conv2Out = convs2[index](conv2In);
      final afterConv2 = mx.transposeAxes(conv2Out, [0, 2, 1]);
      afterConv1.close();
      norm2.close();
      snake2.close();
      conv2In.close();
      conv2Out.close();

      final residual = afterConv2 + current;
      afterConv2.close();
      if (!identical(current, input)) {
        current.close();
      }
      current = residual;
    }
    return current;
  }
}

final class SineGen {
  const SineGen({
    required this.samplingRate,
    required this.upsampleScale,
    this.harmonicNum = 0,
    this.sineAmp = 0.1,
    this.noiseStd = 0.003,
    this.voicedThreshold = 0,
  });

  final int samplingRate;
  final int upsampleScale;
  final int harmonicNum;
  final double sineAmp;
  final double noiseStd;
  final double voicedThreshold;

  int get dim => harmonicNum + 1;

  MlxArray _f02uv(MlxArray f0) {
    final threshold = scalar(voicedThreshold, dtype: f0.dtype);
    try {
      return mx.greater(f0, threshold).astype(MlxDType.MLX_FLOAT32);
    } finally {
      threshold.close();
    }
  }

  MlxArray _f02sine(MlxArray f0Values) {
    final rate = scalar(samplingRate.toDouble(), dtype: f0Values.dtype);
    final normalized = f0Values / rate;
    final wrapped = normalized - normalized.floor();
    rate.close();
    normalized.close();

    final rand = mx.random.normal([f0Values.shape[0], f0Values.shape[2]]);
    final zeros = MlxArray.zeros([f0Values.shape[0], 1], dtype: rand.dtype);
    final tail = rand.slice(start: [0, 1], stop: [rand.shape[0], rand.shape[1]]);
    final randInit = mx.concatenate([zeros, tail], axis: 1).reshape([
      f0Values.shape[0],
      1,
      f0Values.shape[2],
    ]);
    final first = wrapped.slice(
      start: [0, 0, 0],
      stop: [f0Values.shape[0], 1, f0Values.shape[2]],
    );
    final rest = wrapped.slice(
      start: [0, 1, 0],
      stop: [f0Values.shape[0], f0Values.shape[1], f0Values.shape[2]],
    );
    final seeded = first + randInit;
    wrapped.close();
    rand.close();
    zeros.close();
    tail.close();
    randInit.close();
    first.close();
    final radValues = mx.concatenate([seeded, rest], axis: 1);
    seeded.close();
    rest.close();

    final downSize = math.max(1, f0Values.shape[1] ~/ upsampleScale);
    final down = interpolate1dChannelsFirst(
      mx.transposeAxes(radValues, [0, 2, 1]),
      size: downSize,
      mode: 'linear',
    );
    final downTime = mx.transposeAxes(down, [0, 2, 1]);
    final phaseScale = scalar(2 * math.pi, dtype: downTime.dtype);
    final phase = downTime.cumsum(axis: 1) * phaseScale;
    final up = interpolate1dChannelsFirst(
      mx.transposeAxes(phase, [0, 2, 1]),
      size: f0Values.shape[1],
      mode: 'linear',
    );
    final upTime = mx.transposeAxes(up, [0, 2, 1]);
    final scale = scalar(upsampleScale.toDouble(), dtype: upTime.dtype);
    final sine = (upTime * scale).sin();
    radValues.close();
    down.close();
    downTime.close();
    phaseScale.close();
    phase.close();
    up.close();
    upTime.close();
    scale.close();
    return sine;
  }

  ({MlxArray sineWaves, MlxArray uv, MlxArray noise}) call(MlxArray f0) {
    final harmonics = mx.arange(
      1,
      (harmonicNum + 2).toDouble(),
      1,
      dtype: f0.dtype,
    ).reshape([1, 1, dim]);
    final harmonicF0 = f0 * harmonics;
    final sine = _f02sine(harmonicF0);
    final amp = scalar(sineAmp, dtype: sine.dtype);
    final sineWaves = sine * amp;
    final uv = _f02uv(f0);
    final noiseAmpVoiced = scalar(noiseStd, dtype: uv.dtype);
    final noiseAmpUnvoiced = scalar(sineAmp / 3.0, dtype: uv.dtype);
    final one = scalar(1.0, dtype: uv.dtype);
    final noiseAmp = (uv * noiseAmpVoiced) + ((one - uv) * noiseAmpUnvoiced);
    final noise = noiseAmp * mx.random.normal(sineWaves.shape);
    final merged =
        (sineWaves * uv.astype(sineWaves.dtype)) + noise.astype(sineWaves.dtype);
    harmonics.close();
    harmonicF0.close();
    sine.close();
    amp.close();
    noiseAmpVoiced.close();
    noiseAmpUnvoiced.close();
    one.close();
    noiseAmp.close();
    noise.close();
    return (
      sineWaves: merged,
      uv: uv,
      noise: mx.random.normal(uv.shape),
    );
  }
}

final class SourceModuleHnNSF {
  SourceModuleHnNSF({
    required this.sineGen,
    required this.linear,
    this.sineAmp = 0.1,
  });

  factory SourceModuleHnNSF.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required ModelConfig config,
    bool activationQuant = false,
  }) {
    return SourceModuleHnNSF(
      sineGen: SineGen(
        samplingRate: config.sampleRate,
        upsampleScale:
            config.istftnetConfig.upsampleRates.fold(1, (a, b) => a * b) *
            config.istftnetConfig.genIstftHopSize,
        harmonicNum: 8,
        voicedThreshold: 10,
      ),
      linear: Linear.load(
        tensors,
        '$prefix.l_linear',
        quant: config.quantization,
        activationQuant: activationQuant,
      ),
    );
  }

  final SineGen sineGen;
  final Linear linear;
  final double sineAmp;

  KittenSourceOutput call(MlxArray f0) {
    final source = sineGen(f0);
    final merged = mx.tanh(linear(source.sineWaves));
    source.sineWaves.close();
    source.noise.close();
    final scale = scalar(sineAmp / 3.0, dtype: source.uv.dtype);
    final noise = mx.random.normal(source.uv.shape) * scale;
    scale.close();
    return KittenSourceOutput(
      sineMerge: merged,
      noise: noise,
      uv: source.uv,
    );
  }
}

final class ReflectionPad1d {
  const ReflectionPad1d({required this.left, required this.right});

  final int left;
  final int right;

  MlxArray call(MlxArray input) {
    final parts = <MlxArray>[];
    if (left > 0) {
      final leftIndices = List<int>.generate(left, (i) => left - i);
      final index = MlxArray.fromInt32List(leftIndices, shape: [left]);
      parts.add(input.take(index, axis: 2));
      index.close();
    }
    parts.add(input);
    if (right > 0) {
      final width = input.shape[2];
      final rightIndices = List<int>.generate(right, (i) => width - 2 - i);
      final index = MlxArray.fromInt32List(rightIndices, shape: [right]);
      parts.add(input.take(index, axis: 2));
      index.close();
    }
    if (parts.length == 1) {
      return input;
    }
    final output = mx.concatenate(parts, axis: 2);
    for (final part in parts) {
      if (!identical(part, input)) {
        part.close();
      }
    }
    return output;
  }
}

int getPadding1d(int kernelSize, int dilation) =>
    ((kernelSize * dilation) - dilation) ~/ 2;

MlxArray snake1d(MlxArray input, MlxArray alpha) {
  final one = scalar(1.0, dtype: input.dtype);
  final scaled = alpha * input;
  final sine = scaled.sin();
  final squared = sine * sine;
  final update = (one / alpha) * squared;
  try {
    return input + update;
  } finally {
    update.close();
    squared.close();
    sine.close();
    scaled.close();
    one.close();
  }
}

MlxArray interpolate1dChannelsFirst(
  MlxArray input, {
  required int size,
  String mode = 'linear',
  bool alignCorners = false,
}) {
  if (input.ndim != 3) {
    throw ArgumentError('Expected [batch, channels, width] input.');
  }
  final inWidth = input.shape[2];
  if (size == inWidth) {
    return input;
  }
  if (mode == 'nearest') {
    final indices = _nearestIndices(inWidth, size);
    try {
      return input.take(indices, axis: 2);
    } finally {
      indices.close();
    }
  }
  if (inWidth == 1) {
    return input.broadcastTo([input.shape[0], input.shape[1], size]);
  }
  final positions = _linearPositions(
    inWidth,
    size,
    alignCorners: alignCorners,
  );
  final xLow = positions.floor().astype(MlxDType.MLX_INT32);
  final oneInt = MlxArray.full([], 1, dtype: MlxDType.MLX_INT32);
  final maxIndex = MlxArray.full([], inWidth - 1, dtype: MlxDType.MLX_INT32);
  final xHigh = mx.minimum(xLow + oneInt, maxIndex);
  final frac = (positions - xLow.astype(MlxDType.MLX_FLOAT32)).reshape([
    1,
    1,
    size,
  ]);
  final one = scalar(1.0);
  try {
    final yLow = input.take(xLow, axis: 2);
    final yHigh = input.take(xHigh, axis: 2);
    final left = yLow * (one - frac);
    final right = yHigh * frac;
    try {
      return left + right;
    } finally {
      right.close();
      left.close();
      yHigh.close();
      yLow.close();
    }
  } finally {
    one.close();
    frac.close();
    xHigh.close();
    maxIndex.close();
    oneInt.close();
    xLow.close();
    positions.close();
  }
}

MlxArray _nearestIndices(int inWidth, int size) {
  if (size == 1) {
    return MlxArray.fromInt32List([0], shape: [1]);
  }
  final scale = scalar(inWidth / size);
  final base = mx.arange(0, size.toDouble(), 1);
  try {
    return mx.clip(
      (base * scale).floor(),
      min: 0.0,
      max: (inWidth - 1).toDouble(),
    ).astype(MlxDType.MLX_INT32);
  } finally {
    base.close();
    scale.close();
  }
}

MlxArray _linearPositions(
  int inWidth,
  int size, {
  required bool alignCorners,
}) {
  if (size == 1) {
    return MlxArray.full([1], 0.0);
  }
  final base = mx.arange(0, size.toDouble(), 1);
  if (alignCorners) {
    final scale = scalar((inWidth - 1) / (size - 1));
    try {
      return base * scale;
    } finally {
      scale.close();
      base.close();
    }
  }
  final step = scalar(inWidth / size);
  final offset = scalar(0.5 * (inWidth / size) - 0.5);
  try {
    return (base * step) + offset;
  } finally {
    offset.close();
    step.close();
    base.close();
  }
}
