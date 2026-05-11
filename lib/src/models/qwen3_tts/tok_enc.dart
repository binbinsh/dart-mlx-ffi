part of 'qwen3_tts.dart';

final class Qwen3TtsTokenizerEncoder {
  Qwen3TtsTokenizerEncoder(this.bundle) : _cfg = bundle.manifest.tokenizerEncoder;

  final Qwen3TtsBundle bundle;
  final Qwen3TtsTokenizerEncConfig _cfg;

  List<Int32List> encode(Float32List samples) {
    final audio = MlxArray.fromFloat32List(samples, shape: [1, samples.length, 1]);
    MlxArray? encoded;
    try {
      encoded = _encodeArray(audio);
      MlxRuntime.evalAll([encoded]);
      final flat = encoded.toList();
      final q = encoded.shape[1];
      final t = encoded.shape[2];
      final out = <Int32List>[];
      for (var qi = 0; qi < q; qi++) {
        final row = Int32List(t);
        for (var ti = 0; ti < t; ti++) {
          final index = (qi * t) + ti;
          row[ti] = (flat[index] as num).toInt();
        }
        out.add(row);
      }
      return out;
    } finally {
      audio.close();
      encoded?.close();
    }
  }

  MlxArray _encodeArray(MlxArray audio) {
    MlxArray? x;
    MlxArray? transformed;
    MlxArray? downsampled;
    MlxArray? codes;
    try {
      x = _seanetEncode(audio);
      transformed = _transformEncode(x);
      x.close();
      x = null;
      downsampled = _convDownsample(transformed);
      transformed.close();
      transformed = null;
      codes = _splitRvqEncode(downsampled);
      downsampled.close();
      downsampled = null;
      return codes.slice(
        start: [0, 0, 0],
        stop: [codes.shape[0], 16, codes.shape[2]],
      );
    } finally {
      x?.close();
      transformed?.close();
      downsampled?.close();
      codes?.close();
    }
  }

  MlxArray _seanetEncode(MlxArray audio) {
    final ratios = _cfg.upsamplingRatios.reversed.toList(growable: false);
    MlxArray? x;
    MlxArray? out;
    try {
      x = _streamableConvCall(
        audio,
        weight: bundle.requireEncoder('encoder.encoder.layers.0.conv.weight'),
        bias: bundle.requireEncoder('encoder.encoder.layers.0.conv.bias'),
        stride: 1,
        dilation: 1,
        kernelSize: _cfg.kernelSize,
        causal: _cfg.useCausalConv,
        padMode: 'constant',
      );
      for (var i = 0; i < ratios.length; i++) {
        final residual = _seanetLayer(x!, layerIndex: i, ratio: ratios[i], mult: 1 << i);
        x.close();
        x = residual;
      }
      final activated = _elu(x!);
      x.close();
      x = _streamableConvCall(
        activated,
        weight: bundle.requireEncoder('encoder.encoder.layers.14.conv.weight'),
        bias: bundle.requireEncoder('encoder.encoder.layers.14.conv.bias'),
        stride: 1,
        dilation: 1,
        kernelSize: _cfg.lastKernelSize,
        causal: _cfg.useCausalConv,
        padMode: 'constant',
      );
      activated.close();
      out = x;
      x = null;
      return out;
    } finally {
      x?.close();
    }
  }

  MlxArray _seanetLayer(MlxArray input, {required int layerIndex, required int ratio, required int mult}) {
    MlxArray? x;
    MlxArray? out;
    try {
      x = input;
      for (var residualIndex = 0; residualIndex < _cfg.numResidualLayers; residualIndex++) {
        final next = _seanetResnet(x!, layerIndex: layerIndex, residualIndex: residualIndex, dim: mult * _cfg.numFilters);
        if (!identical(x, input)) {
          x.close();
        }
        x = next;
      }
      final activated = _elu(x!);
      if (!identical(x, input)) {
        x.close();
      }
      x = _streamableConvCall(
        activated,
        weight: bundle.requireEncoder('encoder.encoder.layers.${3 + layerIndex * 3}.conv.weight'),
        bias: bundle.requireEncoder('encoder.encoder.layers.${3 + layerIndex * 3}.conv.bias'),
        stride: ratio,
        dilation: 1,
        kernelSize: ratio * 2,
        causal: true,
        padMode: 'constant',
      );
      activated.close();
      out = x;
      x = null;
      return out;
    } finally {
      if (!identical(x, input)) {
        x?.close();
      }
    }
  }

  MlxArray _seanetResnet(MlxArray input, {required int layerIndex, required int residualIndex, required int dim}) {
    MlxArray? x;
    try {
      x = input;
      for (var blockIndex = 0; blockIndex < 2; blockIndex++) {
        final convNode = blockIndex == 0 ? 1 : 3;
        final kernel = blockIndex == 0 ? _cfg.residualKernelSize : 1;
        final dilation = blockIndex == 0 ? math.pow(_cfg.dilationGrowthRate, residualIndex).toInt() : 1;
        final activated = _elu(x!);
        if (!identical(x, input)) {
          x.close();
        }
        x = _streamableConvCall(
          activated,
          weight: bundle.requireEncoder(
            'encoder.encoder.layers.${1 + layerIndex * 3}.block.$convNode.conv.weight',
          ),
          bias: bundle.requireEncoder(
            'encoder.encoder.layers.${1 + layerIndex * 3}.block.$convNode.conv.bias',
          ),
          stride: 1,
          dilation: dilation,
          kernelSize: kernel,
          causal: _cfg.useCausalConv,
          padMode: 'constant',
        );
        activated.close();
      }
      final added = mx.add(x!, input);
      x.close();
      return added;
    } finally {
      if (!identical(x, input)) {
        x?.close();
      }
    }
  }

  MlxArray _transformEncode(MlxArray input) {
    final seqLen = input.shape[1];
    var hidden = input;
    try {
      final mask = _createCausalMask(seqLen, hidden.dtype);
      try {
        for (var index = 0; index < _cfg.numHiddenLayers; index++) {
          final layer = _EncoderLayer(bundle, index, _cfg);
          final next = layer.forward(hidden, mask);
          hidden.close();
          hidden = next;
        }
      } finally {
        mask.close();
      }
      return hidden;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray _convDownsample(MlxArray input) {
    final encoderFrameRate =
        _cfg.sampleRate /
        _cfg.upsamplingRatios.fold<int>(1, (acc, value) => acc * value);
    final stride = (encoderFrameRate / _cfg.frameRate).round();
    return _streamableConvCall(
      input,
      weight: bundle.requireEncoder('encoder.downsample.conv.weight'),
      bias: null,
      stride: stride,
      dilation: 1,
      kernelSize: 2 * stride,
      causal: _cfg.useCausalConv,
      padMode: 'edge',
    );
  }

  MlxArray _splitRvqEncode(MlxArray input) {
    final semantic = _rvqEncode(
      input,
      inputProjWeight: bundle.requireEncoder('encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight'),
      layerPrefixes: const ['encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook'],
    );
    MlxArray? acoustic;
    try {
      acoustic = _rvqEncode(
        input,
        inputProjWeight: bundle.requireEncoder('encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight'),
        layerPrefixes: [
          for (var i = 0; i < _cfg.numQuantizers - 1; i++)
            'encoder.quantizer.acoustic_residual_vector_quantizer.layers.$i.codebook',
        ],
      );
      final all = mx.concatenate([semantic, acoustic], axis: 1);
      return all;
    } finally {
      semantic.close();
      acoustic?.close();
    }
  }

  MlxArray _rvqEncode(
    MlxArray input, {
    required MlxArray inputProjWeight,
    required List<String> layerPrefixes,
  }) {
    MlxArray? residual = _conv1x1(input, inputProjWeight);
    final codes = <MlxArray>[];
    MlxArray? stacked;
    try {
      for (final prefix in layerPrefixes) {
          final embedding = _codebookEmbedding(prefix);
          try {
            final indices = _codebookEncode(residual!, embedding);
            codes.add(indices);
            final quantized = _codebookDecode(indices, embedding);
            final nextResidual = mx.subtract(
              residual,
              quantized.astype(residual.dtype),
            );
            residual.close();
            quantized.close();
            residual = nextResidual;
          } finally {
            embedding.close();
        }
      }
      stacked = mx.stack(codes, axis: 1);
      return stacked;
    } finally {
      residual?.close();
      for (final code in codes) {
        code.close();
      }
      stacked = null;
    }
  }

  MlxArray _codebookEmbedding(String prefix) {
    final usage = bundle.requireEncoder('$prefix.cluster_usage');
    final sum = bundle.requireEncoder('$prefix.embed_sum');
    final denom = mx.maximum(usage.reshape([usage.shape[0], 1]), MlxArray.full([], 1e-5).astype(usage.dtype));
    try {
      return mx.divide(sum, denom);
    } finally {
      denom.close();
    }
  }

  MlxArray _codebookEncode(MlxArray input, MlxArray embedding) {
    final b = input.shape[0];
    final t = input.shape[1];
    final d = input.shape[2];
    final flat = input.reshape([b * t, d]);
    final embedT = embedding.transpose();
    final dot = mx.matmul(flat.astype(MlxDType.MLX_FLOAT32), embedT.astype(MlxDType.MLX_FLOAT32));
    embedT.close();
    final half = MlxArray.full([], 2.0).astype(MlxDType.MLX_FLOAT32);
    final c2 = mx.divide(embedding.astype(MlxDType.MLX_FLOAT32).square().sum(axis: 1), half);
    half.close();
    final scores = mx.subtract(c2.reshape([1, c2.shape[0]]), dot);
    flat.close();
    dot.close();
    c2.close();
    final indices = scores.argmin(axis: 1).reshape([b, t]);
    scores.close();
    return indices;
  }

  MlxArray _codebookDecode(MlxArray indices, MlxArray embedding) {
    final flat = indices.reshape([indices.shape[0] * indices.shape[1]]);
    final decoded = embedding.take(flat, axis: 0).reshape([indices.shape[0], indices.shape[1], embedding.shape[1]]);
    flat.close();
    return decoded;
  }
}

final class _EncoderLayer {
  const _EncoderLayer(this.bundle, this.index, this.cfg);

  final Qwen3TtsBundle bundle;
  final int index;
  final Qwen3TtsTokenizerEncConfig cfg;

  MlxArray forward(MlxArray input, MlxArray mask) {
    final prefix = 'encoder.encoder_transformer.layers.$index';
    final norm1 = mx.fast.layerNorm(
      input,
      weight: bundle.requireEncoder('$prefix.input_layernorm.weight'),
      bias: bundle.requireEncoder('$prefix.input_layernorm.bias'),
      eps: cfg.normEps,
    );
    final attn = _selfAttention(norm1, mask, prefix);
    norm1.close();
    final scaledAttn = attn * bundle.requireEncoder('$prefix.self_attn_layer_scale.scale').reshape([1, 1, cfg.hiddenSize]);
    attn.close();
    final residual1 = mx.add(input, scaledAttn);
    scaledAttn.close();
    final norm2 = mx.fast.layerNorm(
      residual1,
      weight: bundle.requireEncoder('$prefix.post_attention_layernorm.weight'),
      bias: bundle.requireEncoder('$prefix.post_attention_layernorm.bias'),
      eps: cfg.normEps,
    );
    final mlp = _mlp(norm2, prefix);
    norm2.close();
    final scaledMlp = mlp * bundle.requireEncoder('$prefix.mlp_layer_scale.scale').reshape([1, 1, cfg.hiddenSize]);
    mlp.close();
    final out = mx.add(residual1, scaledMlp);
    residual1.close();
    scaledMlp.close();
    return out;
  }

  MlxArray _selfAttention(MlxArray input, MlxArray mask, String prefix) {
    final seqLen = input.shape[1];
    final flat = input.reshape([seqLen, cfg.hiddenSize]);
    final q = _linearNoBias(flat, bundle.requireEncoder('$prefix.self_attn.q_proj.weight'));
    final k = _linearNoBias(flat, bundle.requireEncoder('$prefix.self_attn.k_proj.weight'));
    final v = _linearNoBias(flat, bundle.requireEncoder('$prefix.self_attn.v_proj.weight'));
    flat.close();
    final q4 = q.reshape([1, seqLen, cfg.numAttentionHeads, cfg.headDim]).transposeAxes([0, 2, 1, 3]);
    final k4 = k.reshape([1, seqLen, cfg.numKeyValueHeads, cfg.headDim]).transposeAxes([0, 2, 1, 3]);
    final v4 = v.reshape([1, seqLen, cfg.numKeyValueHeads, cfg.headDim]).transposeAxes([0, 2, 1, 3]);
    q.close();
    k.close();
    v.close();
    final pair = _standardRopeCosSin(
      seqLen,
      offset: 0,
      headDim: cfg.headDim,
      base: cfg.ropeTheta,
      dtype: q4.dtype,
    );
    final rope = _applyStandardRope(q4, k4, pair.cos, pair.sin);
    pair.cos.close();
    pair.sin.close();
    q4.close();
    k4.close();
    final attn = mx.fast.scaledDotProductAttention(
      rope.q,
      rope.k,
      v4,
      scale: 1 / math.sqrt(cfg.headDim),
      mask: mask,
    );
    rope.q.close();
    rope.k.close();
    v4.close();
    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([seqLen, cfg.hiddenSize]);
    attn.close();
    final out = _linearNoBias(merged, bundle.requireEncoder('$prefix.self_attn.o_proj.weight'))
        .reshape([1, seqLen, cfg.hiddenSize]);
    merged.close();
    return out;
  }

  MlxArray _mlp(MlxArray input, String prefix) {
    final seqLen = input.shape[1];
    final flat = input.reshape([seqLen, cfg.hiddenSize]);
    final fc1 = _linearNoBias(flat, bundle.requireEncoder('$prefix.mlp.fc1.weight'));
    final gelu = _geluApprox(fc1);
    fc1.close();
    final fc2 = _linearNoBias(gelu, bundle.requireEncoder('$prefix.mlp.fc2.weight'))
        .reshape([1, seqLen, cfg.hiddenSize]);
    gelu.close();
    flat.close();
    return fc2;
  }
}

MlxArray _streamableConvCall(
  MlxArray input, {
  required MlxArray weight,
  required MlxArray? bias,
  required int stride,
  required int dilation,
  required int kernelSize,
  required bool causal,
  required String padMode,
  int groups = 1,
}) {
  final effKernel = ((kernelSize - 1) * dilation) + 1;
  final paddingTotal = effKernel - stride;
  final len = input.shape[1];
  final nframes = math.max(len + paddingTotal - effKernel, 0) / stride + 1.0;
  final idealLen = ((nframes.ceil() - 1) * stride) + effKernel - paddingTotal;
  final extraPadding = math.max(0, idealLen - len).toInt();
  final left = causal ? paddingTotal : paddingTotal - (paddingTotal ~/ 2);
  final right = causal ? extraPadding : (paddingTotal ~/ 2) + extraPadding;
  MlxArray? padded;
  MlxArray? zero;
  MlxArray? conv;
  try {
    zero = MlxArray.full([], 0.0).astype(input.dtype);
    padded = switch (padMode) {
      'constant' => input.pad(
          axes: [1],
          lowPads: [left],
          highPads: [right],
          padValue: zero,
          mode: 'constant',
        ),
      'edge' => _edgePadTime(input, left: left, right: right),
      _ => throw UnsupportedError('Unsupported tokenizer encoder pad mode: $padMode'),
    };
    final w = weight.shape.length == 3 ? weight.transposeAxes([0, 2, 1]) : weight;
    try {
      conv = mx.conv1d(padded, w, stride: stride, padding: 0, dilation: dilation, groups: groups);
    } finally {
      if (!identical(w, weight)) {
        w.close();
      }
    }
    if (bias == null) return conv;
    final added = mx.add(conv, bias.reshape([1, 1, bias.shape[0]]));
    conv.close();
    return added;
  } finally {
    padded?.close();
    zero?.close();
  }
}

MlxArray _edgePadTime(MlxArray input, {required int left, required int right}) {
  final t = input.shape[1];
  final total = left + t + right;
  final indices = <int>[
    for (var i = 0; i < left; i++) 0,
    for (var i = 0; i < t; i++) i,
    for (var i = 0; i < right; i++) t - 1,
  ];
  final idxArr = MlxArray.fromInt32List(indices, shape: [total]);
  try {
    return input.take(idxArr, axis: 1);
  } finally {
    idxArr.close();
  }
}

MlxArray _conv1x1(MlxArray input, MlxArray weight) {
  return _streamableConvCall(
    input,
    weight: weight,
    bias: null,
    stride: 1,
    dilation: 1,
    kernelSize: 1,
    causal: false,
    padMode: 'constant',
  );
}

MlxArray _linearNoBias(MlxArray input, MlxArray weight) {
  return mx.matmul(input, weight.transpose());
}

MlxArray _geluApprox(MlxArray x) {
  final cubicCoeff = MlxArray.fromFloat32List([0.044715], shape: [1]).astype(x.dtype);
  final scale = MlxArray.fromFloat32List([math.sqrt(2 / math.pi)], shape: [1]).astype(x.dtype);
  final half = MlxArray.fromFloat32List([0.5], shape: [1]).astype(x.dtype);
  final one = MlxArray.fromFloat32List([1.0], shape: [1]).astype(x.dtype);
  try {
    final xSquared = x * x;
    final xCubed = xSquared * x;
    xSquared.close();
    final inner = mx.add(x, xCubed * cubicCoeff);
    xCubed.close();
    final scaled = inner * scale;
    inner.close();
    final tanhVal = scaled.tanh();
    scaled.close();
    final sum = mx.add(one, tanhVal);
    tanhVal.close();
    final left = x * half;
    final result = left * sum;
    left.close();
    sum.close();
    return result;
  } finally {
    cubicCoeff.close();
    scale.close();
    half.close();
    one.close();
  }
}

MlxArray _elu(MlxArray input) {
  final zero = MlxArray.full([], 0.0).astype(input.dtype);
  final one = MlxArray.full([], 1.0).astype(input.dtype);
  final positive = mx.maximum(input, zero);
  final negativeMask = mx.less(input, zero);
  final exp = input.exp();
  final shifted = mx.subtract(exp, one);
  exp.close();
  final negative = mx.where(negativeMask, shifted, MlxArray.zeros(input.shape, dtype: input.dtype));
  shifted.close();
  negativeMask.close();
  final out = mx.add(positive, negative);
  positive.close();
  negative.close();
  zero.close();
  one.close();
  return out;
}
