part of 'qwen3_5.dart';

extension on Qwen3_5Runner {
  static final Map<int, MlxArray> _causalMasks = <int, MlxArray>{};
  bool get _useHighRankLinear =>
      Platform.environment['QWEN35_USE_HIGHRANK_LINEAR'] == '1';

  MlxArray _fullAttention(
    _FullAttentionWeights layer,
    MlxArray input,
    int seqLen, {
    int? layerIndex,
  }) {
    final wantedLayer = int.tryParse(
      Platform.environment['QWEN35_TRACE_LAYER'] ?? '',
    );
    final traceFull =
        Platform.environment['QWEN35_TRACE_FULL'] == '1' &&
        (wantedLayer == null || wantedLayer == layerIndex);
    final linearInput = _useHighRankLinear
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
    if (!_useHighRankLinear) {
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
    if (traceFull) {
      stderr.writeln('qwen35_run: full q_raw ${_previewAttn(q4)}');
      stderr.writeln('qwen35_run: full k_raw ${_previewAttn(k4)}');
    }

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
    if (traceFull) {
      stderr.writeln('qwen35_run: full qnorm ${_previewAttn(qNorm)}');
      stderr.writeln('qwen35_run: full knorm ${_previewAttn(kNorm)}');
    }

    final rope = _applyMrope(qNorm, kNorm, seqLen);
    final qRope = rope.q;
    final kRope = rope.k;
    if (traceFull) {
      stderr.writeln('qwen35_run: full qrope ${_previewAttn(qRope)}');
      stderr.writeln('qwen35_run: full krope ${_previewAttn(kRope)}');
    }
    qNorm.close();
    kNorm.close();

    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kRope,
      v4,
      scale: 1 / math.sqrt(config.headDim),
      maskMode: Platform.environment['QWEN35_FORCE_MASK'] == '1'
          ? ''
          : 'causal',
      mask: Platform.environment['QWEN35_FORCE_MASK'] == '1'
          ? _causalMask(seqLen)
          : null,
    );
    if (traceFull) {
      stderr.writeln('qwen35_run: full attn ${_previewAttn(attn)}');
    }
    qRope.close();
    kRope.close();
    v4.close();

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
    final outInput = _useHighRankLinear
        ? gated.reshape([1, seqLen, config.numAttentionHeads * config.headDim])
        : gated;
    final out = layer.oProj.apply(outInput, config: config);
    if (_useHighRankLinear) {
      outInput.close();
    }
    if (traceFull) {
      stderr.writeln(
        'qwen35_run: full projected ${_previewLastSeq(out.reshape([1, seqLen, config.hiddenSize]))}',
      );
    }
    gated.close();
    return out.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray _linearAttention(
    _LinearAttentionWeights layer,
    MlxArray input,
    int seqLen, {
    int? layerIndex,
  }) {
    final wantedLayer = int.tryParse(
      Platform.environment['QWEN35_TRACE_LAYER'] ?? '',
    );
    final traceLinear =
        Platform.environment['QWEN35_TRACE_LINEAR'] == '1' &&
        (wantedLayer == null || wantedLayer == layerIndex);
    final dumpLayer = int.tryParse(
      Platform.environment['QWEN35_DUMP_LAYER'] ?? '',
    );
    final dumpStage = Platform.environment['QWEN35_DUMP_STAGE'];
    final dumpPath = Platform.environment['QWEN35_DUMP_PATH'];
    final linearInput = _useHighRankLinear
        ? input
        : input.reshape([seqLen, config.hiddenSize]);
    final mixedQkv = layer.inProjQkv.apply(linearInput, config: config).reshape(
      [1, seqLen, layer.convWeight.shape[0]],
    );
    if (traceLinear) {
      stderr.writeln(
        'qwen35_run: linear mixed_qkv ${_previewLastSeq(mixedQkv)}',
      );
    }
    final z = layer.inProjZ.apply(linearInput, config: config).reshape([
      1,
      seqLen,
      config.linearNumValueHeads,
      config.linearValueHeadDim,
    ]);
    if (traceLinear) {
      stderr.writeln('qwen35_run: linear z ${_previewLastHead(z)}');
    }
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
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'a' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(a, dumpPath);
    }
    if (!_useHighRankLinear) {
      linearInput.close();
    }

    final convState = MlxArray.zeros([
      1,
      config.linearConvKernelDim - 1,
      layer.convWeight.shape[0],
    ], dtype: mixedQkv.dtype);
    final convInput = mx.concatenate([convState, mixedQkv], axis: 1);
    convState.close();
    mixedQkv.close();
    final convOut = mx.conv1d(
      convInput,
      layer.convWeight,
      groups: layer.convWeight.shape[0],
    );
    convInput.close();
    if (traceLinear) {
      stderr.writeln('qwen35_run: linear conv_raw ${_previewLastSeq(convOut)}');
    }
    final siluConv = _silu(convOut);
    if (traceLinear) {
      stderr.writeln(
        'qwen35_run: linear conv_out ${_previewLastSeq(siluConv)}',
      );
    }
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
    if (traceLinear) {
      stderr.writeln('qwen35_run: linear q_raw ${_previewLastHead(q)}');
      stderr.writeln('qwen35_run: linear k_raw ${_previewLastHead(k)}');
    }

    final invScale = 1 / math.sqrt(config.linearKeyHeadDim.toDouble());
    final qNorm = mx.fast.rmsNorm(q, eps: 1e-6);
    final qScale = _broadcastScalar(
      invScale * invScale,
      qNorm.shape,
      qNorm.dtype,
    );
    final qScaled = qNorm * qScale;
    qScale.close();
    qNorm.close();
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'linear_q' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(qScaled, dumpPath);
    }
    if (traceLinear) {
      stderr.writeln(
        'qwen35_run: linear q_scaled ${_previewLastHead(qScaled)}',
      );
    }
    final kNorm = mx.fast.rmsNorm(k, eps: 1e-6);
    final kScale = _broadcastScalar(invScale, kNorm.shape, kNorm.dtype);
    final kScaled = kNorm * kScale;
    kScale.close();
    kNorm.close();
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'linear_k' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(kScaled, dumpPath);
    }
    if (traceLinear) {
      stderr.writeln(
        'qwen35_run: linear k_scaled ${_previewLastHead(kScaled)}',
      );
    }
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
      layerIndex: layerIndex,
    );
    qScaled.close();
    kScaled.close();
    v.close();
    a.close();
    b.close();
    final out = outState.output;
    final state = outState.state;
    state.close();
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'gdelta' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(out, dumpPath);
    }
    if (traceLinear) {
      stderr.writeln(
        'qwen35_run: linear gdelta_out ${_previewLastLinear(out)}',
      );
    }

    final norm = mx.fast.rmsNorm(
      out,
      weight: layer.normWeight,
      eps: config.rmsNormEps,
    );
    out.close();
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'pre' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(norm, dumpPath);
    }
    if (traceLinear) {
      stderr.writeln('qwen35_run: linear norm ${_previewLastLinear(norm)}');
    }
    final gated = _swiglu(z, norm);
    norm.close();
    z.close();
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'gated' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(gated, dumpPath);
    }
    if (traceLinear) {
      stderr.writeln('qwen35_run: linear gated ${_previewLastLinear(gated)}');
    }
    final outInput = _useHighRankLinear
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
    if (traceLinear) {
      final proj3d = projected.reshape([1, seqLen, config.hiddenSize]);
      stderr.writeln('qwen35_run: linear projected ${_previewLastSeq(proj3d)}');
      proj3d.close();
    }
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
    int? layerIndex,
  }) {
    final dumpLayer = int.tryParse(
      Platform.environment['QWEN35_DUMP_LAYER'] ?? '',
    );
    final dumpStage = Platform.environment['QWEN35_DUMP_STAGE'];
    final dumpPath = Platform.environment['QWEN35_DUMP_PATH'];
    final beta = b.sigmoid();
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'beta' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(beta, dumpPath);
    }
    final aLogExpanded = aLog
        .reshape([1, 1, config.linearNumValueHeads])
        .broadcastTo(a.shape);
    final dtBiasExpanded = dtBias
        .reshape([1, 1, config.linearNumValueHeads])
        .broadcastTo(a.shape);
    final g = _getComputeGCompiled()([aLogExpanded, a, dtBiasExpanded]).first;
    if (dumpLayer != null &&
        dumpPath != null &&
        dumpStage == 'g' &&
        dumpLayer == layerIndex) {
      Qwen3_5Runner._dumpAny(g, dumpPath);
    }
    aLogExpanded.close();
    dtBiasExpanded.close();
    a.close();

    var state = MlxArray.zeros([
      1,
      config.linearNumValueHeads,
      config.linearValueHeadDim,
      config.linearKeyHeadDim,
    ], dtype: q.dtype);

    final kernelResult = _tryGatedDeltaKernel(q, k, v, g, beta, state);
    if (kernelResult != null) {
      state.close();
      g.close();
      beta.close();
      return kernelResult;
    }

    final repeatFactor = config.linearNumValueHeads ~/ config.linearNumKeyHeads;
    MlxArray qHeads = q;
    MlxArray kHeads = k;
    if (repeatFactor > 1) {
      qHeads = q.repeat(repeatFactor, axis: 2);
      kHeads = k.repeat(repeatFactor, axis: 2);
    }

    final outputs = <MlxArray>[];
    for (var index = 0; index < qHeads.shape[1]; index++) {
      final qStep = _sliceStep(qHeads, index, [
        1,
        config.linearNumValueHeads,
        config.linearKeyHeadDim,
      ]);
      final kStep = _sliceStep(kHeads, index, [
        1,
        config.linearNumValueHeads,
        config.linearKeyHeadDim,
      ]);
      final vStep = _sliceStep(v, index, [
        1,
        config.linearNumValueHeads,
        config.linearValueHeadDim,
      ]);
      final gStep = _sliceStep(g, index, [1, config.linearNumValueHeads]);
      final betaStep = _sliceStep(beta, index, [1, config.linearNumValueHeads]);
      final decay = gStep.reshape([1, config.linearNumValueHeads, 1, 1]);
      final decayed = state * decay;
      state.close();
      decay.close();
      final kvMem =
          (decayed *
                  kStep.reshape([
                    1,
                    config.linearNumValueHeads,
                    1,
                    config.linearKeyHeadDim,
                  ]))
              .sum(axis: 3);
      final delta =
          (vStep - kvMem) *
          betaStep.reshape([1, config.linearNumValueHeads, 1]);
      kvMem.close();
      betaStep.close();
      final newState =
          decayed +
          kStep.reshape([
                1,
                config.linearNumValueHeads,
                1,
                config.linearKeyHeadDim,
              ]) *
              delta.reshape([
                1,
                config.linearNumValueHeads,
                config.linearValueHeadDim,
                1,
              ]);
      decayed.close();
      delta.close();
      final y =
          (newState *
                  qStep.reshape([
                    1,
                    config.linearNumValueHeads,
                    1,
                    config.linearKeyHeadDim,
                  ]))
              .sum(axis: 3)
              .reshape([
                1,
                1,
                config.linearNumValueHeads,
                config.linearValueHeadDim,
              ]);
      outputs.add(y);
      qStep.close();
      kStep.close();
      vStep.close();
      gStep.close();
      state = newState;
    }
    if (!identical(qHeads, q)) {
      qHeads.close();
    }
    if (!identical(kHeads, k)) {
      kHeads.close();
    }
    g.close();
    beta.close();
    final output = mx.concatenate(outputs, axis: 1);
    for (final value in outputs) {
      value.close();
    }
    return (output: output, state: state);
  }

  MlxArray _denseMlp(
    _DenseMlpWeights layer,
    MlxArray input,
    int seqLen, {
    int? layerIndex,
  }) {
    final wantedLayer = int.tryParse(
      Platform.environment['QWEN35_TRACE_LAYER'] ?? '',
    );
    final traceMlp =
        Platform.environment['QWEN35_TRACE_MLP'] == '1' &&
        (wantedLayer == null || wantedLayer == layerIndex);
    final linearInput = _useHighRankLinear
        ? input
        : input.reshape([seqLen, config.hiddenSize]);
    final gate = layer.gateProj.apply(linearInput, config: config);
    final up = layer.upProj.apply(linearInput, config: config);
    if (!_useHighRankLinear) {
      linearInput.close();
    }
    if (traceMlp) {
      stderr.writeln('qwen35_run: mlp gate ${_previewLastSeq(gate)}');
      stderr.writeln('qwen35_run: mlp up ${_previewLastSeq(up)}');
    }
    final fused = _swiglu(gate, up);
    if (traceMlp) {
      stderr.writeln('qwen35_run: mlp fused ${_previewLastSeq(fused)}');
    }
    gate.close();
    up.close();
    final down = layer.downProj.apply(fused, config: config);
    fused.close();
    if (traceMlp) {
      stderr.writeln('qwen35_run: mlp down ${_previewLastSeq(down)}');
    }
    return down.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray _moeMlp(_MoeWeights layer, MlxArray input, int seqLen) {
    final x2d = input.reshape([seqLen, config.hiddenSize]);
    final gates = layer.gate.apply(x2d, config: config);
    final probs = mx.softmax(gates, axis: 1, precise: true);
    gates.close();
    final sorted = probs.argsort(axis: 1);
    final expertCount = sorted.shape[1];
    final topK = config.numExpertsPerTok ?? 0;
    final inds = sorted.slice(
      start: [0, expertCount - topK],
      stop: [seqLen, expertCount],
    );
    sorted.close();
    final scores = probs.takeAlongAxis(inds, axis: 1);
    probs.close();
    final scoreSum = scores.sum(axis: 1, keepDims: true);
    final normScores = scores / scoreSum;
    scores.close();
    scoreSum.close();
    if (layer.switchGateProj is _QuantSwitchLinear &&
        layer.switchUpProj is _QuantSwitchLinear &&
        layer.switchDownProj is _QuantSwitchLinear) {
      final gateProjQ = layer.switchGateProj as _QuantSwitchLinear;
      final upProjQ = layer.switchUpProj as _QuantSwitchLinear;
      final downProjQ = layer.switchDownProj as _QuantSwitchLinear;
      final moeSize = config.moeIntermediateSize ?? 0;
      final inds3d = inds.reshape([1, seqLen, topK]);
      final inputForExperts = x2d.reshape([1, seqLen, 1, 1, config.hiddenSize]);
      final expertsForApply = inds3d;

      final gateRaw = gateProjQ.applyExperts(
        inputForExperts,
        expertsForApply,
        config: config,
        sortedIndices: false,
      );
      final gateOut = gateRaw.reshape([1, seqLen, topK, moeSize]);
      gateRaw.close();
      final upRaw = upProjQ.applyExperts(
        inputForExperts,
        expertsForApply,
        config: config,
        sortedIndices: false,
      );
      final upOut = upRaw.reshape([1, seqLen, topK, moeSize]);
      upRaw.close();
      inputForExperts.close();
      final fused = _silu(gateOut) * upOut;
      gateOut.close();
      upOut.close();
      final downOut = downProjQ
          .applyExperts(
            fused.reshape([1, seqLen, topK, 1, moeSize]),
            inds3d,
            config: config,
            sortedIndices: false,
          )
          .reshape([1, seqLen, topK, config.hiddenSize]);
      fused.close();
      final weighted = downOut * normScores.reshape([1, seqLen, topK, 1]);
      downOut.close();
      final moeY = weighted.sum(axis: 2).reshape([
        1,
        seqLen,
        config.hiddenSize,
      ]);
      weighted.close();

      final sharedGate = layer.sharedGateProj.apply(x2d, config: config);
      final sharedUp = layer.sharedUpProj.apply(x2d, config: config);
      final sharedMul = _silu(sharedGate) * sharedUp;
      sharedGate.close();
      sharedUp.close();
      final sharedDown = layer.sharedDownProj
          .apply(sharedMul, config: config)
          .reshape([1, seqLen, config.hiddenSize]);
      sharedMul.close();
      final mixGate = layer.sharedExpertGate
          .apply(x2d, config: config)
          .sigmoid()
          .reshape([1, seqLen, 1])
          .broadcastTo(sharedDown.shape);
      final sharedMix = sharedDown * mixGate;
      mixGate.close();
      sharedDown.close();
      final out = moeY + sharedMix;
      moeY.close();
      sharedMix.close();
      inds.close();
      normScores.close();
      return out;
    }

    final topIndices = List<int>.from(inds.toList().cast<int>());
    final topScoresArray = normScores.astype(MlxDType.MLX_FLOAT32);
    final topScores = List<double>.from(topScoresArray.toList().cast<double>());
    inds.close();
    normScores.close();
    topScoresArray.close();

    final outputs = <MlxArray>[];
    for (var token = 0; token < seqLen; token++) {
      final tokenInput = x2d.slice(
        start: [token, 0],
        stop: [token + 1, config.hiddenSize],
      );
      var acc = MlxArray.zeros([1, config.hiddenSize], dtype: tokenInput.dtype);
      for (var rank = 0; rank < topK; rank++) {
        final expert = topIndices[token * topK + rank];
        final score = topScores[token * topK + rank];
        final gate = layer.switchGateProj.applyExpert(
          tokenInput,
          expert,
          config: config,
        );
        final up = layer.switchUpProj.applyExpert(
          tokenInput,
          expert,
          config: config,
        );
        final fused = _silu(gate) * up;
        gate.close();
        up.close();
        final down = layer.switchDownProj.applyExpert(
          fused,
          expert,
          config: config,
        );
        fused.close();
        final scale = MlxArray.full(down.shape, score, dtype: down.dtype);
        final scaled = down * scale;
        scale.close();
        down.close();
        final next = acc + scaled;
        acc.close();
        scaled.close();
        acc = next;
      }
      final sharedGate = layer.sharedGateProj.apply(tokenInput, config: config);
      final sharedUp = layer.sharedUpProj.apply(tokenInput, config: config);
      final sharedGateAct = _silu(sharedGate);
      sharedGate.close();
      final sharedMul = sharedGateAct * sharedUp;
      sharedGateAct.close();
      sharedUp.close();
      final sharedDown = layer.sharedDownProj.apply(sharedMul, config: config);
      sharedMul.close();
      final mixGate = layer.sharedExpertGate
          .apply(tokenInput, config: config)
          .sigmoid()
          .broadcastTo(sharedDown.shape);
      final sharedMix = sharedDown * mixGate;
      mixGate.close();
      sharedDown.close();
      tokenInput.close();
      final out = acc + sharedMix;
      acc.close();
      sharedMix.close();
      outputs.add(out);
    }
    x2d.close();
    final stacked = mx.concatenate(outputs, axis: 0).reshape([
      1,
      seqLen,
      config.hiddenSize,
    ]);
    for (final value in outputs) {
      value.close();
    }
    return stacked;
  }

  MlxArray _sliceStep(MlxArray input, int index, List<int> shape) {
    final rank = input.shape.length;
    if (rank == 4) {
      return input
          .slice(
            start: [0, index, 0, 0],
            stop: [1, index + 1, input.shape[2], input.shape[3]],
          )
          .reshape(shape);
    }
    if (rank == 3) {
      return input
          .slice(start: [0, index, 0], stop: [1, index + 1, input.shape[2]])
          .reshape(shape);
    }
    throw StateError('Unsupported slice rank: $rank');
  }

  MlxArray _silu(MlxArray input) {
    final sig = input.sigmoid();
    try {
      return (input * sig).astype(input.dtype);
    } finally {
      sig.close();
    }
  }

  MlxArray _swiglu(MlxArray gate, MlxArray x) {
    final out = _getSwiGluCompiled()([gate, x]).first;
    try {
      return out.astype(gate.dtype);
    } finally {
      out.close();
    }
  }

  MlxArray _causalMask(int seqLen) {
    return _causalMasks.putIfAbsent(seqLen, () {
      final values = <bool>[
        for (var row = 0; row < seqLen; row++)
          for (var col = 0; col < seqLen; col++) row >= col,
      ];
      return MlxArray.fromBoolList(values, shape: [1, 1, seqLen, seqLen]);
    });
  }

  List<double> _previewAttn(MlxArray array, {int limit = 8}) {
    final resolvedLimit =
        int.tryParse(Platform.environment['QWEN35_TRACE_LIMIT'] ?? '') ?? limit;
    final last = array
        .slice(
          start: [0, 0, array.shape[2] - 1, 0],
          stop: [1, 1, array.shape[2], resolvedLimit],
        )
        .reshape([resolvedLimit])
        .astype(MlxDType.MLX_FLOAT32);
    try {
      return List<double>.from(last.toList().cast<double>());
    } finally {
      last.close();
    }
  }

  MlxArray _broadcastScalar(double value, List<int> shape, MlxDType dtype) {
    final scalar = MlxArray.fromFloat32List([value], shape: [1]).astype(dtype);
    try {
      return scalar.broadcastTo(shape);
    } finally {
      scalar.close();
    }
  }

  List<double> _previewLastSeq(MlxArray array, {int limit = 8}) {
    final resolvedLimit =
        int.tryParse(Platform.environment['QWEN35_TRACE_LIMIT'] ?? '') ?? limit;
    final last = array.shape.length == 3
        ? array
              .slice(
                start: [0, array.shape[1] - 1, 0],
                stop: [1, array.shape[1], resolvedLimit],
              )
              .reshape([resolvedLimit])
              .astype(MlxDType.MLX_FLOAT32)
        : array
              .slice(
                start: [array.shape[0] - 1, 0],
                stop: [array.shape[0], resolvedLimit],
              )
              .reshape([resolvedLimit])
              .astype(MlxDType.MLX_FLOAT32);
    try {
      return List<double>.from(last.toList().cast<double>());
    } finally {
      last.close();
    }
  }

  List<double> _previewLastHead(MlxArray array, {int limit = 8}) {
    final resolvedLimit =
        int.tryParse(Platform.environment['QWEN35_TRACE_LIMIT'] ?? '') ?? limit;
    final last = array
        .slice(
          start: [0, array.shape[1] - 1, 0, 0],
          stop: [1, array.shape[1], 1, resolvedLimit],
        )
        .reshape([resolvedLimit])
        .astype(MlxDType.MLX_FLOAT32);
    try {
      return List<double>.from(last.toList().cast<double>());
    } finally {
      last.close();
    }
  }

  List<double> _previewLastLinear(MlxArray array, {int limit = 8}) {
    final resolvedLimit =
        int.tryParse(Platform.environment['QWEN35_TRACE_LIMIT'] ?? '') ?? limit;
    final flat = array
        .slice(
          start: [0, array.shape[1] - 1, 0, 0],
          stop: [1, array.shape[1], array.shape[2], array.shape[3]],
        )
        .reshape([array.shape[2] * array.shape[3]]);
    try {
      final last = flat
          .slice(start: [0], stop: [resolvedLimit])
          .astype(MlxDType.MLX_FLOAT32);
      try {
        return List<double>.from(last.toList().cast<double>());
      } finally {
        last.close();
      }
    } finally {
      flat.close();
    }
  }
}
