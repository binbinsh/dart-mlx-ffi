import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'nn.dart';

final class ParakeetTdtPositionalEncoding {
  ParakeetTdtPositionalEncoding({
    required this.dModel,
    this.scaleInput = true,
  });

  final int dModel;
  final bool scaleInput;
  MlxArray? _pe;
  int _maxLen = 0;
  MlxArray? _scaleTensor;

  ({MlxArray scaledInput, MlxArray posEmb}) call(
    MlxArray input, {
    int offset = 0,
  }) {
    final batch = input.shape[0];
    final inputLen = input.shape[1] + offset;
    _ensureCache(inputLen);
    final pos = _pe!;
    final bufferLen = pos.shape[1];
    final startIdx = bufferLen ~/ 2 - (inputLen - 1);
    final endIdx = bufferLen ~/ 2 + (inputLen - 1) + 1;
    final scaled = (() {
      if (!scaleInput) {
        return input;
      }
      _scaleTensor ??= MlxArray.full([], math.sqrt(dModel.toDouble()));
      try {
        return mx.multiply(input, _scaleTensor!);
      } finally {
      }
    })();
    final sliced = pos.slice(
      start: <int>[0, startIdx, 0],
      stop: <int>[1, endIdx, dModel],
    );
    final typed = sliced.astype(input.dtype);
    sliced.close();
    final posEmb = batch == 1
        ? typed
        : (() {
            final out = mx.broadcastTo(
              typed,
              <int>[batch, endIdx - startIdx, dModel],
            );
            typed.close();
            return out;
          })();
    return (scaledInput: scaled, posEmb: posEmb);
  }

  void _ensureCache(int inputLen) {
    if (_pe != null && inputLen <= _maxLen) {
      return;
    }
    _pe?.close();
    _maxLen = inputLen + 1;
    final totalPositions = 2 * _maxLen - 1;
    final positions = MlxArray.arange(
      (_maxLen - 1).toDouble(),
      (-_maxLen).toDouble(),
      -1,
      dtype: MlxDType.MLX_FLOAT32,
    ).reshape(<int>[totalPositions, 1]);
    final divIndices = MlxArray.arange(
      0,
      dModel.toDouble(),
      2,
      dtype: MlxDType.MLX_FLOAT32,
    );
    final scale = MlxArray.full(
      <int>[],
      -(math.log(10000.0) / dModel),
      dtype: MlxDType.MLX_FLOAT32,
    );
    final scaledDiv = mx.multiply(divIndices, scale);
    final divTerm = mx.exp(scaledDiv).reshape(<int>[1, dModel ~/ 2]);
    final angles = mx.multiply(positions, divTerm);
    final sin = mx.sin(angles);
    final cos = mx.cos(angles);
    final stacked = mx.stack(<MlxArray>[sin, cos], axis: 2);
    final pe = stacked.reshape(<int>[totalPositions, dModel]).expandDims(0);
    _pe = pe;
    positions.close();
    divIndices.close();
    scale.close();
    scaledDiv.close();
    divTerm.close();
    angles.close();
    sin.close();
    cos.close();
    stacked.close();
  }
}

final class ParakeetTdtFeedForward {
  const ParakeetTdtFeedForward(this.linear1, this.linear2);

  final ParakeetDenseLinear linear1;
  final ParakeetDenseLinear linear2;

  factory ParakeetTdtFeedForward.load(
    Map<String, MlxArray> tensors,
    String prefix,
  ) {
    return ParakeetTdtFeedForward(
      ParakeetDenseLinear.load(tensors, '$prefix.linear1'),
      ParakeetDenseLinear.load(tensors, '$prefix.linear2'),
    );
  }

  MlxArray call(MlxArray input) {
    final hidden = linear1(input);
    final activated = parakeetSilu(hidden);
    hidden.close();
    final output = linear2(activated);
    activated.close();
    return output;
  }
}

final class ParakeetTdtRelPosAttention {
  ParakeetTdtRelPosAttention({
    required this.nHeads,
    required this.dModel,
    required this.linearQ,
    required this.linearK,
    required this.linearV,
    required this.linearOut,
    required this.linearPos,
    required this.posBiasU,
    required this.posBiasV,
  }) : headDim = dModel ~/ nHeads,
       scale = 1.0 / math.sqrt((dModel ~/ nHeads).toDouble());

  final int nHeads;
  final int dModel;
  final int headDim;
  final double scale;
  final ParakeetDenseLinear linearQ;
  final ParakeetDenseLinear linearK;
  final ParakeetDenseLinear linearV;
  final ParakeetDenseLinear linearOut;
  final ParakeetDenseLinear linearPos;
  final MlxArray posBiasU;
  final MlxArray posBiasV;
  late final MlxArray _biasU = posBiasU.reshape(<int>[1, 1, nHeads, headDim]);
  late final MlxArray _biasV = posBiasV.reshape(<int>[1, 1, nHeads, headDim]);

  factory ParakeetTdtRelPosAttention.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int nHeads,
    required int dModel,
  }) {
    return ParakeetTdtRelPosAttention(
      nHeads: nHeads,
      dModel: dModel,
      linearQ: ParakeetDenseLinear.load(tensors, '$prefix.linear_q'),
      linearK: ParakeetDenseLinear.load(tensors, '$prefix.linear_k'),
      linearV: ParakeetDenseLinear.load(tensors, '$prefix.linear_v'),
      linearOut: ParakeetDenseLinear.load(tensors, '$prefix.linear_out'),
      linearPos: ParakeetDenseLinear.load(tensors, '$prefix.linear_pos'),
      posBiasU: requireParakeetTensor(tensors, '$prefix.pos_bias_u'),
      posBiasV: requireParakeetTensor(tensors, '$prefix.pos_bias_v'),
    );
  }

  MlxArray call(MlxArray input, {required MlxArray posEmb}) {
    final batch = input.shape[0];
    final qSeq = input.shape[1];
    final qFlat = input.reshape(<int>[batch * qSeq, dModel]);
    final pFlat = posEmb.reshape(<int>[
      posEmb.shape[0] * posEmb.shape[1],
      dModel,
    ]);
    final qProj = linearQ(qFlat).reshape(<int>[batch, qSeq, nHeads, headDim]);
    final kProj = linearK(qFlat).reshape(<int>[batch, qSeq, nHeads, headDim]);
    final vProj = linearV(qFlat).reshape(<int>[batch, qSeq, nHeads, headDim]);
    final pProj = linearPos(
      pFlat,
    ).reshape(<int>[posEmb.shape[0], posEmb.shape[1], nHeads, headDim]);
    qFlat.close();
    pFlat.close();

    final qU = mx.add(qProj, _biasU).transposeAxes(<int>[0, 2, 1, 3]);
    final qV = mx.add(qProj, _biasV).transposeAxes(<int>[0, 2, 1, 3]);
    final k = kProj.transposeAxes(<int>[0, 2, 1, 3]);
    final v = vProj.transposeAxes(<int>[0, 2, 1, 3]);
    final p = pProj.transposeAxes(<int>[0, 2, 1, 3]);
    qProj.close();
    kProj.close();
    vProj.close();
    pProj.close();

    final pT = p.transposeAxes(<int>[0, 1, 3, 2]);
    final matrixBd = _relShift(mx.matmul(qV, pT), qSeq);
    pT.close();
    qV.close();
    p.close();
    final scaleTensor = MlxArray.full([], scale, dtype: matrixBd.dtype);
    final scaledBias = mx.multiply(matrixBd, scaleTensor);
    scaleTensor.close();
    matrixBd.close();

    final attn = mx.fast.scaledDotProductAttention(
      qU,
      k,
      v,
      scale: scale,
      mask: scaledBias,
    );
    qU.close();
    k.close();
    v.close();
    scaledBias.close();

    final merged = attn.transposeAxes(<int>[0, 2, 1, 3]).reshape(<int>[
      batch * qSeq,
      dModel,
    ]);
    attn.close();
    final out = linearOut(merged).reshape(<int>[batch, qSeq, dModel]);
    merged.close();
    return out;
  }

  MlxArray _relShift(MlxArray input, int queryLen) {
    final b = input.shape[0];
    final h = input.shape[1];
    final posLen = input.shape[3];
    final zeroPad = mx.zeros(<int>[b, h, queryLen, 1], dtype: input.dtype);
    final padded = mx.concatenate(<MlxArray>[zeroPad, input], axis: 3);
    zeroPad.close();
    final reshaped = padded.reshape(<int>[b, h, posLen + 1, queryLen]);
    padded.close();
    final sliced = reshaped.slice(
      start: <int>[0, 0, 1, 0],
      stop: <int>[b, h, posLen + 1, queryLen],
    );
    reshaped.close();
    final shifted = sliced.reshape(<int>[b, h, queryLen, posLen]);
    sliced.close();
    final out = shifted.slice(
      start: <int>[0, 0, 0, 0],
      stop: <int>[b, h, queryLen, queryLen],
    );
    shifted.close();
    return out;
  }
}

final class ParakeetTdtConvModule {
  const ParakeetTdtConvModule._({
    required _Conv1dBias pointwiseConv1,
    required _DepthwiseConv1dBias depthwiseConv,
    required this.batchNorm,
    required _Conv1dBias pointwiseConv2,
    required this.padding,
  }) : _pointwiseConv1 = pointwiseConv1,
       _depthwiseConv = depthwiseConv,
       _pointwiseConv2 = pointwiseConv2;

  final _Conv1dBias _pointwiseConv1;
  final _DepthwiseConv1dBias _depthwiseConv;
  final ParakeetBatchNorm batchNorm;
  final _Conv1dBias _pointwiseConv2;
  final int padding;

  factory ParakeetTdtConvModule.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int padding,
  }) {
    return ParakeetTdtConvModule._(
      pointwiseConv1: _Conv1dBias.load(tensors, '$prefix.pointwise_conv1'),
      depthwiseConv: _DepthwiseConv1dBias.load(
        tensors,
        '$prefix.depthwise_conv',
      ),
      batchNorm: ParakeetBatchNorm.load(tensors, '$prefix.batch_norm'),
      pointwiseConv2: _Conv1dBias.load(tensors, '$prefix.pointwise_conv2'),
      padding: padding,
    );
  }

  MlxArray call(MlxArray input) {
    final pw1 = _pointwiseConv1(input);
    final glu = parakeetGlu(pw1);
    pw1.close();
    final leftPad = mx.zeros(<int>[
      glu.shape[0],
      padding,
      glu.shape[2],
    ], dtype: glu.dtype);
    final rightPad = mx.zeros(<int>[
      glu.shape[0],
      padding,
      glu.shape[2],
    ], dtype: glu.dtype);
    final padded = mx.concatenate(<MlxArray>[leftPad, glu, rightPad], axis: 1);
    leftPad.close();
    rightPad.close();
    glu.close();
    final depth = _depthwiseConv(padded);
    padded.close();
    final normed = batchNorm(depth);
    depth.close();
    final activated = parakeetSilu(normed);
    normed.close();
    final out = _pointwiseConv2(activated);
    activated.close();
    return out;
  }
}

final class ParakeetTdtConformerBlock {
  ParakeetTdtConformerBlock({
    required this.normFeedForward1,
    required this.feedForward1,
    required this.normSelfAtt,
    required this.selfAttention,
    required this.normConv,
    required this.conv,
    required this.normFeedForward2,
    required this.feedForward2,
    required this.normOut,
  });

  final ParakeetLayerNorm normFeedForward1;
  final ParakeetTdtFeedForward feedForward1;
  final ParakeetLayerNorm normSelfAtt;
  final ParakeetTdtRelPosAttention selfAttention;
  final ParakeetLayerNorm normConv;
  final ParakeetTdtConvModule conv;
  final ParakeetLayerNorm normFeedForward2;
  final ParakeetTdtFeedForward feedForward2;
  final ParakeetLayerNorm normOut;
  final MlxArray _half = MlxArray.full([], 0.5);

  factory ParakeetTdtConformerBlock.load(
    Map<String, MlxArray> tensors,
    String prefix,
  ) {
    final posBias = requireParakeetTensor(
      tensors,
      '$prefix.self_attn.pos_bias_u',
    );
    final depthwise = requireParakeetTensor(
      tensors,
      '$prefix.conv.depthwise_conv.weight',
    );
    final linearQ = requireParakeetTensor(
      tensors,
      '$prefix.self_attn.linear_q.weight',
    );
    final dModel = linearQ.shape[0];
    final nHeads = posBias.shape[0];
    final convKernelSize = depthwise.shape[1];
    return ParakeetTdtConformerBlock(
      normFeedForward1: ParakeetLayerNorm.load(
        tensors,
        '$prefix.norm_feed_forward1',
      ),
      feedForward1: ParakeetTdtFeedForward.load(
        tensors,
        '$prefix.feed_forward1',
      ),
      normSelfAtt: ParakeetLayerNorm.load(tensors, '$prefix.norm_self_att'),
      selfAttention: ParakeetTdtRelPosAttention.load(
        tensors,
        '$prefix.self_attn',
        nHeads: nHeads,
        dModel: dModel,
      ),
      normConv: ParakeetLayerNorm.load(tensors, '$prefix.norm_conv'),
      conv: ParakeetTdtConvModule.load(
        tensors,
        '$prefix.conv',
        padding: (convKernelSize - 1) ~/ 2,
      ),
      normFeedForward2: ParakeetLayerNorm.load(
        tensors,
        '$prefix.norm_feed_forward2',
      ),
      feedForward2: ParakeetTdtFeedForward.load(
        tensors,
        '$prefix.feed_forward2',
      ),
      normOut: ParakeetLayerNorm.load(tensors, '$prefix.norm_out'),
    );
  }

  MlxArray call(MlxArray input, {required MlxArray posEmb}) {
    final ff1Norm = normFeedForward1(input);
    final ff1 = feedForward1(ff1Norm);
    ff1Norm.close();
    final ff1Scaled = mx.multiply(ff1, _half);
    ff1.close();
    final x1 = mx.add(input, ff1Scaled);
    ff1Scaled.close();

    final attNorm = normSelfAtt(x1);
    final att = selfAttention(attNorm, posEmb: posEmb);
    attNorm.close();
    final x2 = mx.add(x1, att);
    x1.close();
    att.close();

    final convNorm = normConv(x2);
    final convOut = conv(convNorm);
    convNorm.close();
    final x3 = mx.add(x2, convOut);
    x2.close();
    convOut.close();

    final ff2Norm = normFeedForward2(x3);
    final ff2 = feedForward2(ff2Norm);
    ff2Norm.close();
    final ff2Scaled = mx.multiply(ff2, _half);
    ff2.close();
    final x4 = mx.add(x3, ff2Scaled);
    x3.close();
    ff2Scaled.close();
    final out = normOut(x4);
    x4.close();
    return out;
  }
}

final class _Conv1dBias {
  const _Conv1dBias(this.weight, this.bias);

  final MlxArray weight;
  final MlxArray? bias;

  factory _Conv1dBias.load(Map<String, MlxArray> tensors, String prefix) {
    return _Conv1dBias(
      requireParakeetTensor(tensors, '$prefix.weight'),
      tensors['$prefix.bias'],
    );
  }

  MlxArray call(
    MlxArray input, {
    int stride = 1,
    int padding = 0,
    int groups = 1,
  }) {
    final out = mx.conv1d(
      input,
      weight,
      stride: stride,
      padding: padding,
      groups: groups,
    );
    if (bias == null) {
      return out;
    }
    final reshapedBias = bias!.reshape(<int>[1, 1, bias!.shape[0]]);
    try {
      return mx.add(out, reshapedBias);
    } finally {
      reshapedBias.close();
      out.close();
    }
  }
}

final class _DepthwiseConv1dBias extends _Conv1dBias {
  const _DepthwiseConv1dBias(super.weight, super.bias);

  factory _DepthwiseConv1dBias.load(
    Map<String, MlxArray> tensors,
    String prefix,
  ) {
    return _DepthwiseConv1dBias(
      requireParakeetTensor(tensors, '$prefix.weight'),
      tensors['$prefix.bias'],
    );
  }

  @override
  MlxArray call(
    MlxArray input, {
    int stride = 1,
    int padding = 0,
    int groups = 1,
  }) {
    return super.call(
      input,
      stride: stride,
      padding: padding,
      groups: input.shape[2],
    );
  }
}
