library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'args.dart';
import 'core.dart';
import 'lstm.dart';
import 'quant.dart';

final class AdaLayerNorm {
  AdaLayerNorm(
    this.channels,
    this.fc, {
    this.eps = 1e-5,
    this.activationQuant = false,
  });

  factory AdaLayerNorm.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int channels,
    required KittenQuantConfig quant,
    double eps = 1e-5,
    bool activationQuant = false,
  }) {
    return AdaLayerNorm(
      channels,
      Linear.load(
        tensors,
        '$prefix.fc',
        quant: quant,
        activationQuant: activationQuant,
      ),
      eps: eps,
      activationQuant: activationQuant,
    );
  }

  final int channels;
  final Linear fc;
  final double eps;
  final bool activationQuant;

  MlxArray call(MlxArray input, MlxArray style) {
    final styleIn = maybeFakeQuant(style, activationQuant);
    final affine = fc(styleIn);
    if (!identical(styleIn, style)) {
      styleIn.close();
    }
    final split = mx.splitSections(affine, [channels], axis: 1);
    final gamma = split[0].expandDims(1);
    final beta = split[1].expandDims(1);
    final mean = input.mean(axis: -1, keepDims: true);
    final variance = mx.variance(input, axis: -1, keepDims: true);
    final epsArray = scalar(eps);
    final centered = input - mean;
    final normalized = centered / (variance + epsArray).sqrt();
    final one = scalar(1.0);
    final scaled = (one + gamma) * normalized;
    try {
      return scaled + beta;
    } finally {
      scaled.close();
      one.close();
      normalized.close();
      centered.close();
      epsArray.close();
      variance.close();
      mean.close();
      beta.close();
      gamma.close();
      for (final item in split) {
        item.close();
      }
      affine.close();
    }
  }
}

final class DurationEncoder {
  DurationEncoder._({
    required this.styleDim,
    required this.dModel,
    required List<_DurationBlock> blocks,
  }) : _blocks = blocks;

  factory DurationEncoder.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required int styleDim,
    required int dModel,
    required int nLayers,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    final blocks = List<_DurationBlock>.generate(
      nLayers,
      (index) => _DurationBlock(
        lstm: LSTM.load(
          tensors,
          '$prefix.lstms.${index * 2}',
          inputSize: dModel + styleDim,
          hiddenSize: dModel ~/ 2,
          activationQuant: activationQuant,
        ),
        norm: AdaLayerNorm.load(
          tensors,
          '$prefix.lstms.${index * 2 + 1}',
          channels: dModel,
          quant: quant,
          activationQuant: activationQuant,
        ),
      ),
      growable: false,
    );
    return DurationEncoder._(
      styleDim: styleDim,
      dModel: dModel,
      blocks: blocks,
    );
  }

  final int styleDim;
  final int dModel;
  final List<_DurationBlock> _blocks;

  MlxArray call(MlxArray input, MlxArray style, MlxArray textMask) {
    final batch = input.shape[0];
    final steps = input.shape[2];
    final styleExpanded = style.expandDims(1).broadcastTo([
      batch,
      steps,
      styleDim,
    ]);
    final maskBTC = textMask.expandDims(2);
    var hidden = mx.transposeAxes(input, [0, 2, 1]);
    var concat = mx.concatenate([hidden, styleExpanded], axis: 2);
    hidden.close();
    hidden = maskFillZero(concat, maskBTC);
    concat.close();
    for (final block in _blocks) {
      final lstmOut = block.lstm(hidden).output;
      hidden.close();
      hidden = maskFillZero(lstmOut, maskBTC);
      lstmOut.close();
      final normed = block.norm(hidden, style);
      hidden.close();
      final masked = maskFillZero(normed, maskBTC);
      normed.close();
      final next = mx.concatenate([masked, styleExpanded], axis: 2);
      masked.close();
      hidden = maskFillZero(next, maskBTC);
      next.close();
    }
    styleExpanded.close();
    maskBTC.close();
    return hidden;
  }
}

final class ProsodyPredictorResult {
  const ProsodyPredictorResult(this.duration, this.encoding);

  final MlxArray duration;
  final MlxArray encoding;
}

final class ProsodyFeatures {
  const ProsodyFeatures(this.f0, this.noise);

  final MlxArray f0;
  final MlxArray noise;
}

final class ProsodyPredictor {
  ProsodyPredictor._({
    required this.textEncoder,
    required this.lstm,
    required this.durationProj,
    required this.shared,
    required this.f0Blocks,
    required this.nBlocks,
    required this.f0Proj,
    required this.nProj,
  });

  factory ProsodyPredictor.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required ModelConfig config,
    bool activationQuant = false,
  }) {
    final quant = config.quantization;
    return ProsodyPredictor._(
      textEncoder: DurationEncoder.load(
        tensors,
        prefix: '$prefix.text_encoder',
        styleDim: config.styleDim,
        dModel: config.hiddenDim,
        nLayers: config.nLayer,
        quant: quant,
        activationQuant: activationQuant,
      ),
      lstm: LSTM.load(
        tensors,
        '$prefix.lstm',
        inputSize: config.hiddenDim + config.styleDim,
        hiddenSize: config.hiddenDim ~/ 2,
        activationQuant: activationQuant,
      ),
      durationProj: LinearNorm.load(
        tensors,
        '$prefix.duration_proj',
        quant: quant,
        activationQuant: activationQuant,
      ),
      shared: LSTM.load(
        tensors,
        '$prefix.shared',
        inputSize: config.hiddenDim + config.styleDim,
        hiddenSize: config.hiddenDim ~/ 2,
        activationQuant: activationQuant,
      ),
      f0Blocks: <AdainResBlk1d>[
        AdainResBlk1d.load(
          tensors,
          '$prefix.F0.0',
          dimIn: config.hiddenDim,
          dimOut: config.hiddenDim,
          styleDim: config.styleDim,
          quant: quant,
          activationQuant: activationQuant,
        ),
        AdainResBlk1d.load(
          tensors,
          '$prefix.F0.1',
          dimIn: config.hiddenDim,
          dimOut: config.hiddenDim ~/ 2,
          styleDim: config.styleDim,
          upsample: true,
          quant: quant,
          activationQuant: activationQuant,
        ),
        AdainResBlk1d.load(
          tensors,
          '$prefix.F0.2',
          dimIn: config.hiddenDim ~/ 2,
          dimOut: config.hiddenDim ~/ 2,
          styleDim: config.styleDim,
          quant: quant,
          activationQuant: activationQuant,
        ),
      ],
      nBlocks: <AdainResBlk1d>[
        AdainResBlk1d.load(
          tensors,
          '$prefix.N.0',
          dimIn: config.hiddenDim,
          dimOut: config.hiddenDim,
          styleDim: config.styleDim,
          quant: quant,
          activationQuant: activationQuant,
        ),
        AdainResBlk1d.load(
          tensors,
          '$prefix.N.1',
          dimIn: config.hiddenDim,
          dimOut: config.hiddenDim ~/ 2,
          styleDim: config.styleDim,
          upsample: true,
          quant: quant,
          activationQuant: activationQuant,
        ),
        AdainResBlk1d.load(
          tensors,
          '$prefix.N.2',
          dimIn: config.hiddenDim ~/ 2,
          dimOut: config.hiddenDim ~/ 2,
          styleDim: config.styleDim,
          quant: quant,
          activationQuant: activationQuant,
        ),
      ],
      f0Proj: Conv1d.load(
        tensors,
        '$prefix.F0_proj',
        activationQuant: activationQuant,
      ),
      nProj: Conv1d.load(
        tensors,
        '$prefix.N_proj',
        activationQuant: activationQuant,
      ),
    );
  }

  final DurationEncoder textEncoder;
  final LSTM lstm;
  final LinearNorm durationProj;
  final LSTM shared;
  final List<AdainResBlk1d> f0Blocks;
  final List<AdainResBlk1d> nBlocks;
  final Conv1d f0Proj;
  final Conv1d nProj;

  ProsodyPredictorResult call(
    MlxArray texts,
    MlxArray style,
    MlxArray alignment,
    MlxArray textMask,
  ) {
    final encoded = textEncoder(texts, style, textMask);
    final lstmOut = lstm(encoded).output;
    final duration = durationProj(lstmOut);
    lstmOut.close();
    final transposed = mx.transposeAxes(encoded, [0, 2, 1]);
    final aligned = mx.matmul(transposed, alignment);
    transposed.close();
    return ProsodyPredictorResult(duration.squeeze(), aligned);
  }

  ProsodyFeatures f0Ntrain(MlxArray input, MlxArray style) {
    final sharedIn = mx.transposeAxes(input, [0, 2, 1]);
    final sharedOut = shared(sharedIn).output;
    sharedIn.close();
    final f0 = _runFeatureBranch(sharedOut, style, f0Blocks, f0Proj);
    final noise = _runFeatureBranch(sharedOut, style, nBlocks, nProj);
    sharedOut.close();
    return ProsodyFeatures(f0, noise);
  }

  MlxArray _runFeatureBranch(
    MlxArray sharedOut,
    MlxArray style,
    List<AdainResBlk1d> blocks,
    Conv1d proj,
  ) {
    var current = mx.transposeAxes(sharedOut, [0, 2, 1]);
    for (final block in blocks) {
      final next = block(current, style);
      current.close();
      current = next;
    }
    final projIn = mx.transposeAxes(current, [0, 2, 1]);
    current.close();
    final projected = proj(projIn);
    projIn.close();
    final back = mx.transposeAxes(projected, [0, 2, 1]);
    projected.close();
    final squeezed = back.reshape([back.shape[0], back.shape[2]]);
    back.close();
    return squeezed;
  }
}

final class _DurationBlock {
  const _DurationBlock({required this.lstm, required this.norm});

  final LSTM lstm;
  final AdaLayerNorm norm;
}
