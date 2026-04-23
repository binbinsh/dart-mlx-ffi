part of 'qwen3_tts.dart';

MlxArray _geluQ3(MlxArray x) {
  final invSqrt2 = MlxArray.fromFloat32List([1.0 / math.sqrt(2.0)], shape: [1]).astype(x.dtype);
  final half = MlxArray.fromFloat32List([0.5], shape: [1]).astype(x.dtype);
  final one = MlxArray.fromFloat32List([1.0], shape: [1]).astype(x.dtype);
  try {
    final scaled = x * invSqrt2;
    final erfVal = scaled.erf();
    scaled.close();
    final sum = mx.add(one, erfVal);
    erfVal.close();
    final left = x * half;
    final result = left * sum;
    left.close();
    sum.close();
    return result;
  } finally {
    invSqrt2.close();
    half.close();
    one.close();
  }
}

MlxArray _linear2d(MlxArray input, MlxArray weight, {MlxArray? bias}) {
  final out = mx.matmul(input, weight.transpose());
  if (bias == null) return out;
  final bias2d = bias.reshape([1, bias.shape[0]]);
  try {
    final added = mx.add(out, bias2d);
    out.close();
    return added;
  } finally {
    bias2d.close();
  }
}

final class _Qwen3TtsCodebook {
  const _Qwen3TtsCodebook(this.embedWeight);

  final MlxArray embedWeight;

  MlxArray decode(MlxArray codes) => embedWeight.take(codes, axis: 0);
}

final class _Qwen3TtsVectorQuantization {
  const _Qwen3TtsVectorQuantization({required this.codebook});

  final _Qwen3TtsCodebook codebook;

  MlxArray decode(MlxArray codes) {
    var quantized = codebook.decode(codes); // [1, time, dim]
    return quantized.transposeAxes([0, 2, 1]);
  }
}

final class _Qwen3TtsResidualVectorQuantization {
  const _Qwen3TtsResidualVectorQuantization(this.layers);

  final List<_Qwen3TtsVectorQuantization> layers;

  MlxArray decode(MlxArray codes) {
    MlxArray? total;
    try {
      for (var index = 0; index < layers.length; index++) {
        final slice = codes.slice(
          start: [index, 0, 0],
          stop: [index + 1, codes.shape[1], codes.shape[2]],
        ).reshape([codes.shape[1], codes.shape[2]]);
        final decoded = layers[index].decode(slice);
        slice.close();
        if (total == null) {
          total = decoded;
        } else {
          final next = mx.add(total, decoded);
          total.close();
          decoded.close();
          total = next;
        }
      }
      if (total == null) {
        throw StateError('Qwen3-TTS decoder RVQ received no codebooks.');
      }
      return total;
    } catch (_) {
      total?.close();
      rethrow;
    }
  }
}

final class _Qwen3TtsResidualVectorQuantizer {
  const _Qwen3TtsResidualVectorQuantizer({required this.vq, this.outputProj});

  final _Qwen3TtsResidualVectorQuantization vq;
  final MlxArray? outputProj;

  MlxArray decode(MlxArray codes) {
    final transposed = codes.transposeAxes([1, 0, 2]);
    final quantized = vq.decode(transposed);
    transposed.close();
    final proj = outputProj;
    if (proj == null) {
      return quantized;
    }
    final ntc = quantized.transposeAxes([0, 2, 1]);
    quantized.close();
    final projected = mx.conv1d(ntc, proj, stride: 1, padding: 0, dilation: 1);
    ntc.close();
    return projected.transposeAxes([0, 2, 1]);
  }
}

final class _Qwen3TtsSplitResidualVectorQuantizer {
  const _Qwen3TtsSplitResidualVectorQuantizer({required this.first, required this.rest});

  final _Qwen3TtsResidualVectorQuantizer first;
  final _Qwen3TtsResidualVectorQuantizer rest;

  MlxArray decode(MlxArray codes) {
    final firstCodes = codes.slice(
      start: [0, 0, 0],
      stop: [codes.shape[0], 1, codes.shape[2]],
    );
    final firstOut = first.decode(firstCodes);
    firstCodes.close();
    if (codes.shape[1] <= 1) {
      return firstOut;
    }
    final restCodes = codes.slice(
      start: [0, 1, 0],
      stop: [codes.shape[0], codes.shape[1], codes.shape[2]],
    );
    final restOut = rest.decode(restCodes);
    restCodes.close();
    final out = mx.add(firstOut, restOut);
    firstOut.close();
    restOut.close();
    return out;
  }
}

final class _Qwen3TtsCausalConv1d {
  _Qwen3TtsCausalConv1d({
    required this.weight,
    required this.bias,
    required this.kernelSize,
    this.dilation = 1,
    this.groups = 1,
  }) : _padding = (kernelSize - 1) * dilation;

  final MlxArray weight;
  final MlxArray bias;
  final int kernelSize;
  final int dilation;
  final int groups;
  final int _padding;
  MlxArray? _buffer;

  void resetState() {
    _buffer?.close();
    _buffer = null;
  }

  MlxArray step(MlxArray input) {
    var x = input;
    if (_padding > 0) {
      final existing = _buffer;
      if (existing != null) {
        x = mx.concatenate([existing, input], axis: 1);
      } else {
        final zero = MlxArray.full([], 0.0);
        x = input.pad(
          axes: [1],
          lowPads: [_padding],
          highPads: [0],
          padValue: zero,
          mode: 'constant',
        );
        zero.close();
      }
      final next = x.slice(
        start: [0, x.shape[1] - _padding, 0],
        stop: [x.shape[0], x.shape[1], x.shape[2]],
      );
      _buffer?.close();
      _buffer = next;
    }
    final conv = mx.conv1d(x, weight, stride: 1, padding: 0, dilation: dilation, groups: groups);
    if (!identical(x, input)) {
      x.close();
    }
    final bias3d = bias.reshape([1, 1, bias.shape[0]]);
    try {
      final out = mx.add(conv, bias3d);
      conv.close();
      return out;
    } finally {
      bias3d.close();
    }
  }
}

final class _Qwen3TtsDecoderBlockUpsample {
  _Qwen3TtsDecoderBlockUpsample({
    required this.weight,
    required this.bias,
    required this.stride,
  }) : _trimRight = weight.shape[1] - stride;

  final MlxArray weight;
  final MlxArray bias;
  final int stride;
  final int _trimRight;
  MlxArray? _overflow;

  void resetState() {
    _overflow?.close();
    _overflow = null;
  }

  MlxArray step(MlxArray input) {
    var y = mx.convTranspose1d(input, weight, stride: stride, padding: 0, dilation: 1);
    final bias3d = bias.reshape([1, 1, bias.shape[0]]);
    try {
      final added = mx.add(y, bias3d);
      y.close();
      y = added;
    } finally {
      bias3d.close();
    }
    final existing = _overflow;
    if (existing != null) {
      final ovLen = existing.shape[1];
      final prefix = y.slice(start: [0, 0, 0], stop: [y.shape[0], ovLen, y.shape[2]]);
      final mergedPrefix = mx.add(prefix, existing);
      prefix.close();
      existing.close();
      final suffix = y.slice(
        start: [0, ovLen, 0],
        stop: [y.shape[0], y.shape[1], y.shape[2]],
      );
      final merged = mx.concatenate([mergedPrefix, suffix], axis: 1);
      mergedPrefix.close();
      suffix.close();
      y.close();
      y = merged;
    }
    if (_trimRight <= 0) {
      _overflow = null;
      return y;
    }
    final overflow = y.slice(
      start: [0, y.shape[1] - _trimRight, 0],
      stop: [y.shape[0], y.shape[1], y.shape[2]],
    );
    final out = y.slice(
      start: [0, 0, 0],
      stop: [y.shape[0], y.shape[1] - _trimRight, y.shape[2]],
    );
    y.close();
    _overflow = overflow;
    return out;
  }
}

final class _Qwen3TtsSnakeBeta {
  const _Qwen3TtsSnakeBeta(this.alpha, this.beta);

  final MlxArray alpha;
  final MlxArray beta;

  MlxArray call(MlxArray input) {
    final a = alpha.exp();
    final b = beta.exp();
    final eps = MlxArray.full([], 1e-9).astype(input.dtype);
    try {
      final inv = mx.divide(MlxArray.full([], 1.0).astype(input.dtype), mx.add(b, eps));
      final sinSq = (input * a).sin();
      final sinPow = sinSq * sinSq;
      sinSq.close();
      final scaled = sinPow * inv;
      sinPow.close();
      inv.close();
      final out = mx.add(input, scaled);
      scaled.close();
      return out;
    } finally {
      a.close();
      b.close();
      eps.close();
    }
  }
}

final class _Qwen3TtsConvNeXtBlock {
  _Qwen3TtsConvNeXtBlock({
    required this.dwconv,
    required this.normWeight,
    required this.normBias,
    required this.pw1Weight,
    required this.pw1Bias,
    required this.pw2Weight,
    required this.pw2Bias,
    required this.gamma,
  });

  final _Qwen3TtsCausalConv1d dwconv;
  final MlxArray normWeight;
  final MlxArray normBias;
  final MlxArray pw1Weight;
  final MlxArray pw1Bias;
  final MlxArray pw2Weight;
  final MlxArray pw2Bias;
  final MlxArray gamma;

  void resetState() => dwconv.resetState();

  MlxArray step(MlxArray input) {
    final residual = input;
    var x = dwconv.step(input);
    final norm = mx.fast.layerNorm(x, weight: normWeight, bias: normBias, eps: 1e-6);
    x.close();
    x = norm;
    final pw1Bias3d = pw1Bias.reshape([1, 1, pw1Bias.shape[0]]);
    final pw1 = mx.add(mx.matmul(x, pw1Weight.transpose()), pw1Bias3d);
    pw1Bias3d.close();
    final gelu = _geluQ3(pw1);
    pw1.close();
    final pw2Bias3d = pw2Bias.reshape([1, 1, pw2Bias.shape[0]]);
    final pw2 = mx.add(mx.matmul(gelu, pw2Weight.transpose()), pw2Bias3d);
    pw2Bias3d.close();
    gelu.close();
    final scaled = pw2 * gamma.reshape([1, 1, gamma.shape[0]]);
    pw2.close();
    final out = mx.add(residual, scaled);
    scaled.close();
    return out;
  }
}

final class _Qwen3TtsDecoderResidualUnit {
  _Qwen3TtsDecoderResidualUnit({
    required this.act1,
    required this.conv1,
    required this.act2,
    required this.conv2,
  });

  final _Qwen3TtsSnakeBeta act1;
  final _Qwen3TtsCausalConv1d conv1;
  final _Qwen3TtsSnakeBeta act2;
  final _Qwen3TtsCausalConv1d conv2;

  void resetState() {
    conv1.resetState();
    conv2.resetState();
  }

  MlxArray step(MlxArray input) {
    final residual = input;
    final a1 = act1.call(input);
    final c1 = conv1.step(a1);
    a1.close();
    final a2 = act2.call(c1);
    c1.close();
    final c2 = conv2.step(a2);
    a2.close();
    final out = mx.add(residual, c2);
    c2.close();
    return out;
  }
}

final class _Qwen3TtsDecoderBlock {
  _Qwen3TtsDecoderBlock({required this.snake, required this.upsample, required this.units});

  final _Qwen3TtsSnakeBeta snake;
  final _Qwen3TtsDecoderBlockUpsample upsample;
  final List<_Qwen3TtsDecoderResidualUnit> units;

  void resetState() {
    upsample.resetState();
    for (final unit in units) {
      unit.resetState();
    }
  }

  MlxArray step(MlxArray input) {
    var x = snake.call(input);
    final up = upsample.step(x);
    x.close();
    x = up;
    for (final unit in units) {
      final next = unit.step(x);
      x.close();
      x = next;
    }
    return x;
  }
}

final class _Qwen3TtsPreTransformerLayer {
  const _Qwen3TtsPreTransformerLayer({
    required this.qWeight,
    required this.kWeight,
    required this.vWeight,
    required this.oWeight,
    required this.inputNorm,
    required this.postNorm,
    required this.gateWeight,
    required this.upWeight,
    required this.downWeight,
    required this.selfAttnScale,
    required this.mlpScale,
  });

  final MlxArray qWeight;
  final MlxArray kWeight;
  final MlxArray vWeight;
  final MlxArray oWeight;
  final MlxArray inputNorm;
  final MlxArray postNorm;
  final MlxArray gateWeight;
  final MlxArray upWeight;
  final MlxArray downWeight;
  final MlxArray selfAttnScale;
  final MlxArray mlpScale;
}

final class _Qwen3TtsPreTransformer {
  _Qwen3TtsPreTransformer(this.bundle)
    : _cfg = bundle.manifest.decoder,
      _inputProjWeight = bundle.requireDecoder('decoder.pre_transformer.input_proj.weight'),
      _inputProjBias = bundle.requireDecoder('decoder.pre_transformer.input_proj.bias'),
      _outputProjWeight = bundle.requireDecoder('decoder.pre_transformer.output_proj.weight'),
      _outputProjBias = bundle.requireDecoder('decoder.pre_transformer.output_proj.bias'),
      _norm = bundle.requireDecoder('decoder.pre_transformer.norm.weight'),
      _layers = List<_Qwen3TtsPreTransformerLayer>.generate(
        bundle.manifest.decoder.numHiddenLayers,
        (index) {
          final prefix = 'decoder.pre_transformer.layers.$index';
          return _Qwen3TtsPreTransformerLayer(
            qWeight: bundle.requireDecoder('$prefix.self_attn.q_proj.weight'),
            kWeight: bundle.requireDecoder('$prefix.self_attn.k_proj.weight'),
            vWeight: bundle.requireDecoder('$prefix.self_attn.v_proj.weight'),
            oWeight: bundle.requireDecoder('$prefix.self_attn.o_proj.weight'),
            inputNorm: bundle.requireDecoder('$prefix.input_layernorm.weight'),
            postNorm: bundle.requireDecoder('$prefix.post_attention_layernorm.weight'),
            gateWeight: bundle.requireDecoder('$prefix.mlp.gate_proj.weight'),
            upWeight: bundle.requireDecoder('$prefix.mlp.up_proj.weight'),
            downWeight: bundle.requireDecoder('$prefix.mlp.down_proj.weight'),
            selfAttnScale: bundle.requireDecoder('$prefix.self_attn_layer_scale.scale'),
            mlpScale: bundle.requireDecoder('$prefix.mlp_layer_scale.scale'),
          );
        },
      );

  final Qwen3TtsBundle bundle;
  final Qwen3TtsDecoderConfig _cfg;
  final MlxArray _inputProjWeight;
  final MlxArray _inputProjBias;
  final MlxArray _outputProjWeight;
  final MlxArray _outputProjBias;
  final MlxArray _norm;
  final List<_Qwen3TtsPreTransformerLayer> _layers;

  List<_Qwen3TtsKvCache> createCache() =>
      List<_Qwen3TtsKvCache>.generate(_layers.length, (_) => _Qwen3TtsKvCache());

  MlxArray forward(MlxArray input, List<_Qwen3TtsKvCache> cache) {
    final seqLen = input.shape[1];
    final flat = input.reshape([seqLen, input.shape[2]]);
    var hidden = _linear2d(flat, _inputProjWeight, bias: _inputProjBias)
        .reshape([1, seqLen, _cfg.hiddenSize]);
    flat.close();
    for (var index = 0; index < _layers.length; index++) {
      final layer = _layers[index];
      final norm1 = mx.fast.rmsNorm(hidden, weight: layer.inputNorm, eps: _cfg.rmsNormEps);
      final attn = _attend(layer, norm1, cache[index]);
      norm1.close();
      final scaledAttn = attn * layer.selfAttnScale.reshape([1, 1, _cfg.hiddenSize]);
      attn.close();
      final residual1 = mx.add(hidden, scaledAttn);
      hidden.close();
      scaledAttn.close();
      final norm2 = mx.fast.rmsNorm(
        residual1,
        weight: layer.postNorm,
        eps: _cfg.rmsNormEps,
      );
      final mlp = _mlp(layer, norm2);
      norm2.close();
      final scaledMlp = mlp * layer.mlpScale.reshape([1, 1, _cfg.hiddenSize]);
      mlp.close();
      final next = mx.add(residual1, scaledMlp);
      residual1.close();
      scaledMlp.close();
      hidden = next;
    }
    final norm = mx.fast.rmsNorm(hidden, weight: _norm, eps: _cfg.rmsNormEps);
    hidden.close();
    final outFlat = norm.reshape([seqLen, _cfg.hiddenSize]);
    norm.close();
    final projected = _linear2d(outFlat, _outputProjWeight, bias: _outputProjBias);
    outFlat.close();
    return projected.reshape([1, seqLen, _cfg.latentDim]);
  }

  MlxArray _attend(
    _Qwen3TtsPreTransformerLayer layer,
    MlxArray input,
    _Qwen3TtsKvCache cache,
  ) {
    final seqLen = input.shape[1];
    final offset = cache.offset;
    final flat = input.reshape([seqLen, _cfg.hiddenSize]);
    final q = _linear2d(flat, layer.qWeight).reshape([1, seqLen, _cfg.numAttentionHeads, _cfg.headDim]);
    final k = _linear2d(flat, layer.kWeight).reshape([1, seqLen, _cfg.numKeyValueHeads, _cfg.headDim]);
    final v = _linear2d(flat, layer.vWeight).reshape([1, seqLen, _cfg.numKeyValueHeads, _cfg.headDim]);
    flat.close();
    final qT = q.transposeAxes([0, 2, 1, 3]);
    final kT = k.transposeAxes([0, 2, 1, 3]);
    final vT = v.transposeAxes([0, 2, 1, 3]);
    q.close();
    k.close();
    v.close();
    final useExplicitRope = Platform.environment['QWEN3_TTS_DECODER_EXPLICIT_ROPE'] == '1';
    late final MlxArray qRope;
    late final MlxArray kRope;
    if (useExplicitRope) {
      final pair = _standardRopeCosSin(
        seqLen,
        offset: offset,
        headDim: _cfg.headDim,
        base: _cfg.ropeTheta,
        dtype: qT.dtype,
      );
      final rope = _applyStandardRope(qT, kT, pair.cos, pair.sin);
      pair.cos.close();
      pair.sin.close();
      qRope = rope.q;
      kRope = rope.k;
    } else {
      qRope = mx.fast.rope(qT, dims: _cfg.headDim, traditional: false, base: _cfg.ropeTheta, offset: offset);
      kRope = mx.fast.rope(kT, dims: _cfg.headDim, traditional: false, base: _cfg.ropeTheta, offset: offset);
    }
    qT.close();
    kT.close();
    final fetched = cache.updateAndFetch(kRope, vT);
    final kBase = fetched.$1;
    final vBase = fetched.$2;
    final mask = useExplicitRope
        ? seqLen <= 1
            ? null
            : offset > 0
            ? _createCausalMaskWithPrefix(seqLen: seqLen, prefixLen: offset, dtype: input.dtype)
            : _createCausalMask(seqLen, input.dtype)
        : null;
    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kBase,
      vBase,
      scale: 1 / math.sqrt(_cfg.headDim),
      mask: mask,
      maskMode: useExplicitRope ? '' : offset > 0 && seqLen == 1 ? '' : 'causal',
    );
    mask?.close();
    qRope.close();
    final merged = attn
        .transposeAxes([0, 2, 1, 3])
        .reshape([seqLen, _cfg.numAttentionHeads * _cfg.headDim]);
    attn.close();
    final out = _linear2d(merged, layer.oWeight).reshape([1, seqLen, _cfg.hiddenSize]);
    merged.close();
    return out;
  }

  MlxArray _mlp(_Qwen3TtsPreTransformerLayer layer, MlxArray input) {
    final seqLen = input.shape[1];
    final flat = input.reshape([seqLen, _cfg.hiddenSize]);
    final gate = _linear2d(flat, layer.gateWeight);
    final up = _linear2d(flat, layer.upWeight);
    flat.close();
    final activated = _silu(gate);
    gate.close();
    final fused = activated * up;
    activated.close();
    up.close();
    final down = _linear2d(fused, layer.downWeight).reshape([1, seqLen, _cfg.hiddenSize]);
    fused.close();
    return down;
  }
}

final class _Qwen3TtsSpeechDecoder {
  _Qwen3TtsSpeechDecoder(this.bundle)
    : _cfg = bundle.manifest.decoder,
      _quantizer = _buildQuantizer(bundle),
      _preConv = _Qwen3TtsCausalConv1d(
        weight: bundle.requireDecoder('decoder.pre_conv.conv.weight'),
        bias: bundle.requireDecoder('decoder.pre_conv.conv.bias'),
        kernelSize: 3,
      ),
      _preTransformer = _Qwen3TtsPreTransformer(bundle),
      _upsample = List.generate(
        bundle.manifest.decoder.upsamplingRatios.length,
        (index) => (
          conv: _Qwen3TtsDecoderBlockUpsample(
            weight: bundle.requireDecoder('decoder.upsample.$index.0.conv.weight'),
            bias: bundle.requireDecoder('decoder.upsample.$index.0.conv.bias'),
            stride: bundle.manifest.decoder.upsamplingRatios[index],
          ),
          block: _Qwen3TtsConvNeXtBlock(
            dwconv: _Qwen3TtsCausalConv1d(
              weight: bundle.requireDecoder('decoder.upsample.$index.1.dwconv.conv.weight'),
              bias: bundle.requireDecoder('decoder.upsample.$index.1.dwconv.conv.bias'),
              kernelSize: 7,
              groups: bundle.manifest.decoder.latentDim,
            ),
            normWeight: bundle.requireDecoder('decoder.upsample.$index.1.norm.weight'),
            normBias: bundle.requireDecoder('decoder.upsample.$index.1.norm.bias'),
            pw1Weight: bundle.requireDecoder('decoder.upsample.$index.1.pwconv1.weight'),
            pw1Bias: bundle.requireDecoder('decoder.upsample.$index.1.pwconv1.bias'),
            pw2Weight: bundle.requireDecoder('decoder.upsample.$index.1.pwconv2.weight'),
            pw2Bias: bundle.requireDecoder('decoder.upsample.$index.1.pwconv2.bias'),
            gamma: bundle.requireDecoder('decoder.upsample.$index.1.gamma'),
          ),
        ),
      ),
      _decoderInit = _Qwen3TtsCausalConv1d(
        weight: bundle.requireDecoder('decoder.decoder.0.conv.weight'),
        bias: bundle.requireDecoder('decoder.decoder.0.conv.bias'),
        kernelSize: 7,
      ),
      _decoderBlocks = List.generate(
        bundle.manifest.decoder.upsampleRates.length,
        (index) => _Qwen3TtsDecoderBlock(
          snake: _Qwen3TtsSnakeBeta(
            bundle.requireDecoder('decoder.decoder.${index + 1}.block.0.alpha'),
            bundle.requireDecoder('decoder.decoder.${index + 1}.block.0.beta'),
          ),
          upsample: _Qwen3TtsDecoderBlockUpsample(
            weight: bundle.requireDecoder('decoder.decoder.${index + 1}.block.1.conv.weight'),
            bias: bundle.requireDecoder('decoder.decoder.${index + 1}.block.1.conv.bias'),
            stride: bundle.manifest.decoder.upsampleRates[index],
          ),
          units: List.generate(
            3,
            (unitIndex) {
              final prefix = 'decoder.decoder.${index + 1}.block.${unitIndex + 2}';
              return _Qwen3TtsDecoderResidualUnit(
                act1: _Qwen3TtsSnakeBeta(
                  bundle.requireDecoder('$prefix.act1.alpha'),
                  bundle.requireDecoder('$prefix.act1.beta'),
                ),
                conv1: _Qwen3TtsCausalConv1d(
                  weight: bundle.requireDecoder('$prefix.conv1.conv.weight'),
                  bias: bundle.requireDecoder('$prefix.conv1.conv.bias'),
                  kernelSize: 7,
                  dilation: [1, 3, 9][unitIndex],
                ),
                act2: _Qwen3TtsSnakeBeta(
                  bundle.requireDecoder('$prefix.act2.alpha'),
                  bundle.requireDecoder('$prefix.act2.beta'),
                ),
                conv2: _Qwen3TtsCausalConv1d(
                  weight: bundle.requireDecoder('$prefix.conv2.conv.weight'),
                  bias: bundle.requireDecoder('$prefix.conv2.conv.bias'),
                  kernelSize: 1,
                ),
              );
            },
          ),
        ),
      ),
      _outputSnake = _Qwen3TtsSnakeBeta(
        bundle.requireDecoder('decoder.decoder.5.alpha'),
        bundle.requireDecoder('decoder.decoder.5.beta'),
      ),
      _outputConv = _Qwen3TtsCausalConv1d(
        weight: bundle.requireDecoder('decoder.decoder.6.conv.weight'),
        bias: bundle.requireDecoder('decoder.decoder.6.conv.bias'),
        kernelSize: 7,
      );

  final Qwen3TtsBundle bundle;
  final Qwen3TtsDecoderConfig _cfg;
  final _Qwen3TtsSplitResidualVectorQuantizer _quantizer;
  final _Qwen3TtsCausalConv1d _preConv;
  final _Qwen3TtsPreTransformer _preTransformer;
  final List<({
    _Qwen3TtsDecoderBlockUpsample conv,
    _Qwen3TtsConvNeXtBlock block,
  })> _upsample;
  final _Qwen3TtsCausalConv1d _decoderInit;
  final List<_Qwen3TtsDecoderBlock> _decoderBlocks;
  final _Qwen3TtsSnakeBeta _outputSnake;
  final _Qwen3TtsCausalConv1d _outputConv;
  List<_Qwen3TtsKvCache>? _transformerCache;

  int get totalUpsample => _cfg.totalUpsample;

  void resetStreamingState() {
    if (_transformerCache case final cache?) {
      for (final item in cache) {
        item.close();
      }
    }
    _transformerCache = null;
    _preConv.resetState();
    for (final layer in _upsample) {
      layer.conv.resetState();
      layer.block.resetState();
    }
    _decoderInit.resetState();
    for (final block in _decoderBlocks) {
      block.resetState();
    }
    _outputConv.resetState();
  }

  MlxArray streamingStep(MlxArray codes) {
    _transformerCache ??= _preTransformer.createCache();
    final quantized = _quantizer.decode(codes);
    var hidden = quantized.transposeAxes([0, 2, 1]);
    quantized.close();
    final pre = _preConv.step(hidden);
    hidden.close();
    hidden = pre;
    final transformed = _preTransformer.forward(hidden, _transformerCache!);
    hidden.close();
    hidden = transformed;
    for (final layer in _upsample) {
      final up = layer.conv.step(hidden);
      hidden.close();
      final next = layer.block.step(up);
      up.close();
      hidden = next;
    }
    var wav = _decoderInit.step(hidden);
    hidden.close();
    for (final block in _decoderBlocks) {
      final next = block.step(wav);
      wav.close();
      wav = next;
    }
    final shaped = _outputSnake.call(wav);
    wav.close();
    final out = _outputConv.step(shaped);
    shaped.close();
    final ncl = out.transposeAxes([0, 2, 1]);
    out.close();
    final clipped = ncl.clip(min: -1.0, max: 1.0);
    ncl.close();
    return clipped;
  }

  static _Qwen3TtsSplitResidualVectorQuantizer _buildQuantizer(Qwen3TtsBundle bundle) {
    final firstLayers = <_Qwen3TtsVectorQuantization>[
      _Qwen3TtsVectorQuantization(
        codebook: _Qwen3TtsCodebook(
          bundle.requireDecoder('decoder.quantizer.rvq_first.vq.layers.0.codebook.embed.weight'),
        ),
      ),
    ];
    final restLayers = List<_Qwen3TtsVectorQuantization>.generate(
      bundle.manifest.decoder.numQuantizers - 1,
      (index) => _Qwen3TtsVectorQuantization(
        codebook: _Qwen3TtsCodebook(
          bundle.requireDecoder('decoder.quantizer.rvq_rest.vq.layers.$index.codebook.embed.weight'),
        ),
      ),
    );
    return _Qwen3TtsSplitResidualVectorQuantizer(
      first: _Qwen3TtsResidualVectorQuantizer(
        vq: _Qwen3TtsResidualVectorQuantization(firstLayers),
        outputProj: bundle.requireDecoder('decoder.quantizer.rvq_first.output_proj.weight'),
      ),
      rest: _Qwen3TtsResidualVectorQuantizer(
        vq: _Qwen3TtsResidualVectorQuantization(restLayers),
        outputProj: bundle.requireDecoder('decoder.quantizer.rvq_rest.output_proj.weight'),
      ),
    );
  }
}
