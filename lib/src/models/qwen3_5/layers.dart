part of 'qwen3_5.dart';

extension on Qwen3_5Runner {
  MlxArray _fullAttention(
    _FullAttentionWeights layer,
    MlxArray input,
    int seqLen, {
    _KvDecodeCache? cache,
  }) {
    final useHighRank = _useHighRankLinear(seqLen);
    final linearInput = useHighRank
        ? input
        : input.reshape([seqLen, config.hiddenSize]);
    final qGate = layer.qProj.apply(linearInput, config: config).reshape([
      1,
      seqLen,
      config.numAttentionHeads,
      config.headDim * 2,
    ]);
    final split = mx.splitSections(qGate, [config.headDim], axis: 3);
    qGate.close();
    final q = split[0].reshape([
      seqLen,
      config.numAttentionHeads * config.headDim,
    ]);
    final gate = split[1].reshape([
      seqLen,
      config.numAttentionHeads * config.headDim,
    ]);
    final k = layer.kProj.apply(linearInput, config: config);
    final v = layer.vProj.apply(linearInput, config: config);
    if (linearInput != input) {
      linearInput.close();
    }

    final q4 = q
        .reshape([1, seqLen, config.numAttentionHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    final k4 = k
        .reshape([1, seqLen, config.numKeyValueHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    final v4 = v
        .reshape([1, seqLen, config.numKeyValueHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    q.close();
    k.close();
    v.close();

    final qNorm = mx.fast.rmsNorm(
      q4,
      weight: layer.qNormWeight,
      eps: config.rmsNormEps,
    );
    final kNorm = mx.fast.rmsNorm(
      k4,
      weight: layer.kNormWeight,
      eps: config.rmsNormEps,
    );
    q4.close();
    k4.close();

    final rope = _applyMrope(qNorm, kNorm, seqLen, offset: cache?.offset ?? 0);
    final qRope = rope.q;
    var kRope = rope.k;
    var vAttn = v4;
    if (cache != null) {
      final fetched = cache.updateAndFetch(kRope, vAttn);
      kRope = fetched.$1;
      vAttn = fetched.$2;
    }
    qNorm.close();
    kNorm.close();

    final directGqaMode = Platform.environment['QWEN35_USE_DIRECT_GQA'];
    final useDirectGqa =
        directGqaMode == '1' ||
        (directGqaMode == 'decode' && cache != null && seqLen == 1);
    final repeatKv = config.numAttentionHeads ~/ config.numKeyValueHeads;
    if (repeatKv > 1 && !useDirectGqa) {
      kRope = _repeatKvHeads(
        kRope,
        numHeads: config.numAttentionHeads,
        numKvHeads: config.numKeyValueHeads,
        seqLen: kRope.shape[2],
        headDim: config.headDim,
      );
      vAttn = _repeatKvHeads(
        vAttn,
        numHeads: config.numAttentionHeads,
        numKvHeads: config.numKeyValueHeads,
        seqLen: vAttn.shape[2],
        headDim: config.headDim,
      );
      if (cache == null) {
        rope.k.close();
        v4.close();
      }
    }

    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kRope,
      vAttn,
      scale: 1 / math.sqrt(config.headDim),
      maskMode: cache != null && seqLen == 1 ? '' : 'causal',
    );
    qRope.close();
    if ((repeatKv > 1 && !useDirectGqa) || cache == null) {
      kRope.close();
      vAttn.close();
    }

    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      config.numAttentionHeads * config.headDim,
    ]);
    attn.close();
    final gated =
        merged *
        gate.reshape([
          seqLen,
          config.numAttentionHeads * config.headDim,
        ]).sigmoid();
    gate.close();
    merged.close();
    final outInput = useHighRank
        ? gated.reshape([1, seqLen, config.numAttentionHeads * config.headDim])
        : gated;
    final out = layer.oProj.apply(outInput, config: config);
    if (outInput != gated) {
      outInput.close();
    }
    gated.close();
    return out.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray _linearAttention(
    _LinearAttentionWeights layer,
    MlxArray input,
    int seqLen, {
    _LinearDecodeCache? cache,
  }) {
    final useHighRank = _useHighRankLinear(seqLen);
    final linearInput = useHighRank
        ? input
        : input.reshape([seqLen, config.hiddenSize]);
    final mixedQkv = layer.inProjQkv.apply(linearInput, config: config).reshape(
      [1, seqLen, layer.convWeight.shape[0]],
    );
    final z = layer.inProjZ.apply(linearInput, config: config).reshape([
      1,
      seqLen,
      config.linearNumValueHeads,
      config.linearValueHeadDim,
    ]);
    final b = layer.inProjB.apply(linearInput, config: config).reshape([
      1,
      seqLen,
      config.linearNumValueHeads,
    ]);
    final a = layer.inProjA.apply(linearInput, config: config).reshape([
      1,
      seqLen,
      config.linearNumValueHeads,
    ]);
    if (linearInput != input) {
      linearInput.close();
    }

    final cachedConvState = cache?.takeConvState();
    final convState =
        cachedConvState ??
        _linearConvPrefix(layer.convWeight.shape[0], mixedQkv.dtype);
    final convInput = mx.concatenate([convState, mixedQkv], axis: 1);
    if (cachedConvState != null) {
      cachedConvState.close();
    }
    mixedQkv.close();
    if (cache != null) {
      final nextConvState = convInput.slice(
        start: [0, convInput.shape[1] - (config.linearConvKernelDim - 1), 0],
        stop: [1, convInput.shape[1], convInput.shape[2]],
      );
      cache.replaceConvState(nextConvState);
    }
    final convOut = mx.conv1d(
      convInput,
      layer.convWeight,
      groups: layer.convWeight.shape[0],
    );
    convInput.close();
    final siluConv = _silu(convOut);
    convOut.close();

    final keyDim = config.linearNumKeyHeads * config.linearKeyHeadDim;
    final valueDim = config.linearNumValueHeads * config.linearValueHeadDim;
    final q = siluConv
        .slice(start: [0, 0, 0], stop: [1, seqLen, keyDim])
        .reshape([
          1,
          seqLen,
          config.linearNumKeyHeads,
          config.linearKeyHeadDim,
        ]);
    final k = siluConv
        .slice(start: [0, 0, keyDim], stop: [1, seqLen, 2 * keyDim])
        .reshape([
          1,
          seqLen,
          config.linearNumKeyHeads,
          config.linearKeyHeadDim,
        ]);
    final v = siluConv
        .slice(
          start: [0, 0, 2 * keyDim],
          stop: [1, seqLen, 2 * keyDim + valueDim],
        )
        .reshape([
          1,
          seqLen,
          config.linearNumValueHeads,
          config.linearValueHeadDim,
        ]);
    siluConv.close();

    final invScale = 1 / math.sqrt(config.linearKeyHeadDim.toDouble());
    final qNorm = mx.fast.rmsNorm(q, eps: 1e-6);
    final qScale = _linearScale(
      seqLen,
      qNorm.dtype,
      invScale * invScale,
      kind: 'q',
      shape: qNorm.shape,
    );
    final qScaled = qNorm * qScale;
    qNorm.close();

    final kNorm = mx.fast.rmsNorm(k, eps: 1e-6);
    final kScale = _linearScale(
      seqLen,
      kNorm.dtype,
      invScale,
      kind: 'k',
      shape: kNorm.shape,
    );
    final kScaled = kNorm * kScale;
    kNorm.close();
    q.close();
    k.close();

    final outState = _gatedDelta(
      qScaled,
      kScaled,
      v,
      a,
      b,
      layer.aLog,
      layer.dtBias,
      initialState: cache?.takeState(),
    );
    qScaled.close();
    kScaled.close();
    v.close();
    a.close();
    b.close();

    final out = outState.output;
    final state = outState.state;
    if (cache != null) {
      cache.replaceState(state);
    } else {
      state.close();
    }

    final norm = mx.fast.rmsNorm(
      out,
      weight: layer.normWeight,
      eps: config.rmsNormEps,
    );
    out.close();
    final gated = _swiglu(z, norm);
    norm.close();
    z.close();
    final outInput = useHighRank
        ? gated.reshape([
            1,
            seqLen,
            config.linearNumValueHeads * config.linearValueHeadDim,
          ])
        : gated.reshape([
            seqLen,
            config.linearNumValueHeads * config.linearValueHeadDim,
          ]);
    final projected = layer.outProj.apply(outInput, config: config);
    outInput.close();
    gated.close();
    return projected.reshape([1, seqLen, config.hiddenSize]);
  }

  ({MlxArray output, MlxArray state}) _gatedDelta(
    MlxArray q,
    MlxArray k,
    MlxArray v,
    MlxArray a,
    MlxArray b,
    MlxArray aLog,
    MlxArray dtBias, {
    MlxArray? initialState,
  }) {
    var state =
        initialState ??
        MlxArray.zeros([
          1,
          config.linearNumValueHeads,
          config.linearValueHeadDim,
          config.linearKeyHeadDim,
        ], dtype: q.dtype);
    final beta = b.sigmoid();
    final aLogExpanded = aLog
        .reshape([1, 1, config.linearNumValueHeads])
        .broadcastTo(a.shape);
    final dtBiasExpanded = dtBias
        .reshape([1, 1, config.linearNumValueHeads])
        .broadcastTo(a.shape);
    final g = Platform.environment['QWEN35_DISABLE_COMPILED_HELPERS'] == '1'
        ? _computeGEager(aLogExpanded, a, dtBiasExpanded)
        : _getComputeGCompiled()([aLogExpanded, a, dtBiasExpanded]).first;
    aLogExpanded.close();
    dtBiasExpanded.close();
    a.close();

    final kernelResult = _tryGatedDeltaKernel(q, k, v, g, beta, state);
    if (kernelResult != null) {
      state.close();
      g.close();
      beta.close();
      return kernelResult;
    }
    return _runGatedDeltaFallback(q, k, v, g, beta, state);
  }

  MlxArray _denseMlp(_DenseMlpWeights layer, MlxArray input, int seqLen) {
    final useHighRank = _useHighRankLinear(seqLen);
    final linearInput = useHighRank
        ? input
        : input.reshape([seqLen, config.hiddenSize]);
    final gate = layer.gateProj.apply(linearInput, config: config);
    final up = layer.upProj.apply(linearInput, config: config);
    if (linearInput != input) {
      linearInput.close();
    }
    final fused = _swiglu(gate, up);
    gate.close();
    up.close();
    final down = layer.downProj.apply(fused, config: config);
    fused.close();
    return down.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray _sliceStep(MlxArray input, int index, List<int> shape) {
    if (input.shape.length == 4) {
      return input
          .slice(
            start: [0, index, 0, 0],
            stop: [1, index + 1, input.shape[2], input.shape[3]],
          )
          .reshape(shape);
    }
    if (input.shape.length == 3) {
      return input
          .slice(start: [0, index, 0], stop: [1, index + 1, input.shape[2]])
          .reshape(shape);
    }
    throw StateError('Unsupported slice rank: ${input.shape.length}');
  }

  MlxArray _silu(MlxArray input) {
    final sig = input.sigmoid();
    try {
      return input * sig;
    } finally {
      sig.close();
    }
  }

  MlxArray _swiglu(MlxArray gate, MlxArray x) {
    if (Platform.environment['QWEN35_DISABLE_COMPILED_HELPERS'] == '1') {
      final sig = gate.sigmoid();
      try {
        return (gate * sig) * x;
      } finally {
        sig.close();
      }
    }
    return _getSwiGluCompiled()([gate, x]).first;
  }

  MlxArray _repeatKvHeads(
    MlxArray tensor, {
    required int numHeads,
    required int numKvHeads,
    required int seqLen,
    required int headDim,
  }) {
    final repeat = numHeads ~/ numKvHeads;
    final expanded = tensor.expandDims(2);
    try {
      return expanded
          .broadcastTo([1, numKvHeads, repeat, seqLen, headDim])
          .reshape([1, numHeads, seqLen, headDim]);
    } finally {
      expanded.close();
    }
  }

  MlxArray _linearScale(
    int seqLen,
    MlxDType dtype,
    double value, {
    required String kind,
    required List<int> shape,
  }) {
    final key = '$kind:$seqLen:${dtype.value}';
    final cached = _linearScaleCache[key];
    if (cached != null) {
      return cached;
    }
    final scalar = MlxArray.fromFloat32List([value], shape: [1]).astype(dtype);
    try {
      final out = scalar.broadcastTo(shape);
      _linearScaleCache[key] = out;
      return out;
    } finally {
      scalar.close();
    }
  }

  MlxArray _linearConvPrefix(int channels, MlxDType dtype) {
    final key = '$channels:${dtype.value}';
    final cached = _linearConvPrefixCache[key];
    if (cached != null) {
      return cached;
    }
    final out = MlxArray.zeros([
      1,
      config.linearConvKernelDim - 1,
      channels,
    ], dtype: dtype);
    _linearConvPrefixCache[key] = out;
    return out;
  }
}
