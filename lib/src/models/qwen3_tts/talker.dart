part of 'qwen3_tts.dart';

final class _Qwen3TtsBlockConfig {
  const _Qwen3TtsBlockConfig({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.vocabSize,
    required this.rmsNormEps,
    required this.ropeTheta,
  });

  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int vocabSize;
  final double rmsNormEps;
  final double ropeTheta;
}

final class _Qwen3TtsBlockLayer {
  const _Qwen3TtsBlockLayer({
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.gateProj,
    required this.upProj,
    required this.downProj,
    required this.inputNorm,
    required this.postNorm,
    required this.qNorm,
    required this.kNorm,
  });

  final _Qwen3TtsQuantLinear qProj;
  final _Qwen3TtsQuantLinear kProj;
  final _Qwen3TtsQuantLinear vProj;
  final _Qwen3TtsQuantLinear oProj;
  final _Qwen3TtsQuantLinear gateProj;
  final _Qwen3TtsQuantLinear upProj;
  final _Qwen3TtsQuantLinear downProj;
  final MlxArray inputNorm;
  final MlxArray postNorm;
  final MlxArray qNorm;
  final MlxArray kNorm;
}

final class _Qwen3TtsTalker {
  _Qwen3TtsTalker(this.bundle)
    : _quant = bundle.manifest.quantization,
      _talkerCfg = bundle.manifest.talker,
      _predictorCfg = bundle.manifest.talker.codePredictor,
      _codecEmbedding = bundle.require('talker.model.codec_embedding.weight'),
      _textEmbedding = bundle.require('talker.model.text_embedding.weight'),
      _finalNorm = bundle.require('talker.model.norm.weight'),
      _codecHead = _qLinear(bundle.tensors, 'talker.codec_head'),
      _textFc1 = _qLinear(bundle.tensors, 'talker.text_projection.linear_fc1'),
      _textFc2 = _qLinear(bundle.tensors, 'talker.text_projection.linear_fc2'),
      _predictorNorm = bundle.require('talker.code_predictor.model.norm.weight'),
      _talkerLayers = List<_Qwen3TtsBlockLayer>.generate(
        bundle.manifest.talker.numHiddenLayers,
        (index) {
          final prefix = 'talker.model.layers.$index';
          return _Qwen3TtsBlockLayer(
            qProj: _qLinear(bundle.tensors, '$prefix.self_attn.q_proj'),
            kProj: _qLinear(bundle.tensors, '$prefix.self_attn.k_proj'),
            vProj: _qLinear(bundle.tensors, '$prefix.self_attn.v_proj'),
            oProj: _qLinear(bundle.tensors, '$prefix.self_attn.o_proj'),
            gateProj: _qLinear(bundle.tensors, '$prefix.mlp.gate_proj'),
            upProj: _qLinear(bundle.tensors, '$prefix.mlp.up_proj'),
            downProj: _qLinear(bundle.tensors, '$prefix.mlp.down_proj'),
            inputNorm: bundle.require('$prefix.input_layernorm.weight'),
            postNorm: bundle.require('$prefix.post_attention_layernorm.weight'),
            qNorm: bundle.require('$prefix.self_attn.q_norm.weight'),
            kNorm: bundle.require('$prefix.self_attn.k_norm.weight'),
          );
        },
      ),
      _predictorLayers = List<_Qwen3TtsBlockLayer>.generate(
        bundle.manifest.talker.codePredictor.numHiddenLayers,
        (index) {
          final prefix = 'talker.code_predictor.model.layers.$index';
          return _Qwen3TtsBlockLayer(
            qProj: _qLinear(bundle.tensors, '$prefix.self_attn.q_proj'),
            kProj: _qLinear(bundle.tensors, '$prefix.self_attn.k_proj'),
            vProj: _qLinear(bundle.tensors, '$prefix.self_attn.v_proj'),
            oProj: _qLinear(bundle.tensors, '$prefix.self_attn.o_proj'),
            gateProj: _qLinear(bundle.tensors, '$prefix.mlp.gate_proj'),
            upProj: _qLinear(bundle.tensors, '$prefix.mlp.up_proj'),
            downProj: _qLinear(bundle.tensors, '$prefix.mlp.down_proj'),
            inputNorm: bundle.require('$prefix.input_layernorm.weight'),
            postNorm: bundle.require('$prefix.post_attention_layernorm.weight'),
            qNorm: bundle.require('$prefix.self_attn.q_norm.weight'),
            kNorm: bundle.require('$prefix.self_attn.k_norm.weight'),
          );
        },
      ),
      _codecPredictorEmbeds = List<MlxArray>.generate(
        bundle.manifest.talker.codePredictor.numCodeGroups - 1,
        (index) => bundle.require('talker.code_predictor.model.codec_embedding.$index.weight'),
      ),
      _lmHeads = List<_Qwen3TtsQuantLinear>.generate(
        bundle.manifest.talker.codePredictor.numCodeGroups - 1,
        (index) => _qLinear(bundle.tensors, 'talker.code_predictor.lm_head.$index'),
      );

  final Qwen3TtsBundle bundle;
  final Qwen3TtsQuantConfig _quant;
  final Qwen3TtsTalkerConfig _talkerCfg;
  final Qwen3TtsCodePredictorConfig _predictorCfg;
  final List<_Qwen3TtsBlockLayer> _talkerLayers;
  final List<_Qwen3TtsBlockLayer> _predictorLayers;
  final List<MlxArray> _codecPredictorEmbeds;
  final List<_Qwen3TtsQuantLinear> _lmHeads;
  final MlxArray _codecEmbedding;
  final MlxArray _textEmbedding;
  final MlxArray _finalNorm;
  final _Qwen3TtsQuantLinear _codecHead;
  final _Qwen3TtsQuantLinear _textFc1;
  final _Qwen3TtsQuantLinear _textFc2;
  final MlxArray _predictorNorm;

  List<_Qwen3TtsKvCache> createTalkerCache() =>
      List<_Qwen3TtsKvCache>.generate(_talkerLayers.length, (_) => _Qwen3TtsKvCache());

  List<_Qwen3TtsKvCache> createCodePredictorCache() =>
      List<_Qwen3TtsKvCache>.generate(_predictorLayers.length, (_) => _Qwen3TtsKvCache());

  void resetCache(List<_Qwen3TtsKvCache> cache) {
    for (final layer in cache) {
      layer.reset();
    }
  }

  void closeCache(List<_Qwen3TtsKvCache> cache) {
    for (final layer in cache) {
      layer.close();
    }
  }

  MlxArray embedTextIds(List<int> ids) {
    if (ids.isEmpty) {
      return MlxArray.fromFloat32List(const <double>[], shape: [1, 0, _talkerCfg.textHiddenSize]);
    }
    final arr = MlxArray.fromInt32List(ids, shape: [1, ids.length]);
    try {
      return _textEmbedding.take(arr, axis: 0);
    } finally {
      arr.close();
    }
  }

  MlxArray embedCodecIds(List<int> ids) {
    final arr = MlxArray.fromInt32List(ids, shape: [1, ids.length]);
    try {
      return _codecEmbedding.take(arr, axis: 0);
    } finally {
      arr.close();
    }
  }

  MlxArray embedCodecToken(MlxArray ids) => _codecEmbedding.take(ids, axis: 0);

  MlxArray embedPredictorToken(int index, MlxArray ids) =>
      _codecPredictorEmbeds[index].take(ids, axis: 0);

  MlxArray projectText(MlxArray input) {
    final seqLen = input.shape[1];
    final flat = input.reshape([seqLen, _talkerCfg.textHiddenSize]);
    final fc1 = _quantLinear(
      flat,
      _textFc1,
      _quant,
      outDim: _talkerCfg.textHiddenSize,
      addBias: true,
    );
    flat.close();
    final activated = _silu(fc1);
    fc1.close();
    final fc2 = _quantLinear(
      activated,
      _textFc2,
      _quant,
      outDim: _talkerCfg.hiddenSize,
      addBias: true,
    );
    activated.close();
    return fc2.reshape([1, seqLen, _talkerCfg.hiddenSize]);
  }

  ({MlxArray logits, MlxArray hidden}) forwardTalker(
    MlxArray hidden,
    List<_Qwen3TtsKvCache> cache,
  ) {
    final out = _runBlocks(
      hidden,
      _talkerLayers,
      _Qwen3TtsBlockConfig(
        hiddenSize: _talkerCfg.hiddenSize,
        intermediateSize: _talkerCfg.intermediateSize,
        numHiddenLayers: _talkerCfg.numHiddenLayers,
        numAttentionHeads: _talkerCfg.numAttentionHeads,
        numKeyValueHeads: _talkerCfg.numKeyValueHeads,
        headDim: _talkerCfg.headDim,
        vocabSize: _talkerCfg.vocabSize,
        rmsNormEps: _talkerCfg.rmsNormEps,
        ropeTheta: _talkerCfg.ropeTheta,
      ),
      cache,
    );
    final seqLen = out.shape[1];
    // Match Python: `model.__call__` returns post-norm hidden states. The
    // returned hidden is re-used downstream (code_predictor's code_idx=0
    // concatenates `hidden[:, -1:, :]` with the codec embedding), so we must
    // propagate the normalized tensor — not the raw pre-norm residual stream.
    final norm = mx.fast.rmsNorm(out, weight: _finalNorm, eps: _talkerCfg.rmsNormEps);
    out.close();
    final last = norm
        .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, _talkerCfg.hiddenSize])
        .reshape([1, _talkerCfg.hiddenSize]);
    final logits = _quantLinear(
      last,
      _codecHead,
      _quant,
      outDim: _talkerCfg.vocabSize,
      addBias: false,
    );
    last.close();
    return (logits: logits.reshape([1, _talkerCfg.vocabSize]), hidden: norm);
  }

  MlxArray forwardCodePredictor(
    MlxArray hidden,
    int generationStep,
    List<_Qwen3TtsKvCache> cache,
  ) {
    if (_predictorCfg.hiddenSize != _talkerCfg.hiddenSize) {
      throw UnsupportedError('Qwen3-TTS code predictor projection mismatch is not implemented yet.');
    }
    final out = _runBlocks(
      hidden,
      _predictorLayers,
      _Qwen3TtsBlockConfig(
        hiddenSize: _predictorCfg.hiddenSize,
        intermediateSize: _predictorCfg.intermediateSize,
        numHiddenLayers: _predictorCfg.numHiddenLayers,
        numAttentionHeads: _predictorCfg.numAttentionHeads,
        numKeyValueHeads: _predictorCfg.numKeyValueHeads,
        headDim: _predictorCfg.headDim,
        vocabSize: _predictorCfg.vocabSize,
        rmsNormEps: _predictorCfg.rmsNormEps,
        ropeTheta: _predictorCfg.ropeTheta,
      ),
      cache,
    );
    final seqLen = out.shape[1];
    final norm = mx.fast.rmsNorm(out, weight: _predictorNorm, eps: _predictorCfg.rmsNormEps);
    out.close();
    final last = norm
        .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, _predictorCfg.hiddenSize])
        .reshape([1, _predictorCfg.hiddenSize]);
    norm.close();
    final logits = _quantLinear(
      last,
      _lmHeads[generationStep],
      _quant,
      outDim: _predictorCfg.vocabSize,
      addBias: false,
    );
    last.close();
    return logits.reshape([1, _predictorCfg.vocabSize]);
  }

  MlxArray _runBlocks(
    MlxArray hidden,
    List<_Qwen3TtsBlockLayer> layers,
    _Qwen3TtsBlockConfig cfg,
    List<_Qwen3TtsKvCache> cache,
  ) {
    try {
      for (var index = 0; index < layers.length; index++) {
        final layer = layers[index];
        final seqLen = hidden.shape[1];
        final norm1 = mx.fast.rmsNorm(hidden, weight: layer.inputNorm, eps: cfg.rmsNormEps);
        final attn = _selfAttention(layer, norm1, seqLen, cfg, cache[index]);
        final residual1 = mx.add(hidden, attn);
        norm1.close();
        attn.close();
        hidden.close();

        final norm2 = mx.fast.rmsNorm(
          residual1,
          weight: layer.postNorm,
          eps: cfg.rmsNormEps,
        );
        final mlp = _mlp(layer, norm2, seqLen, cfg);
        norm2.close();
        final next = mx.add(residual1, mlp);
        residual1.close();
        mlp.close();
        hidden = next;
      }
      return hidden;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray _selfAttention(
    _Qwen3TtsBlockLayer layer,
    MlxArray input,
    int seqLen,
    _Qwen3TtsBlockConfig cfg,
    _Qwen3TtsKvCache cache,
  ) {
    final offset = cache.offset;
    final x2d = input.reshape([seqLen, cfg.hiddenSize]);
    final q = _quantLinear(
      x2d,
      layer.qProj,
      _quant,
      outDim: cfg.numAttentionHeads * cfg.headDim,
      addBias: true,
    );
    final k = _quantLinear(
      x2d,
      layer.kProj,
      _quant,
      outDim: cfg.numKeyValueHeads * cfg.headDim,
      addBias: true,
    );
    final v = _quantLinear(
      x2d,
      layer.vProj,
      _quant,
      outDim: cfg.numKeyValueHeads * cfg.headDim,
      addBias: true,
    );
    x2d.close();

    final q4r = q.reshape([1, seqLen, cfg.numAttentionHeads, cfg.headDim]);
    final k4r = k.reshape([1, seqLen, cfg.numKeyValueHeads, cfg.headDim]);
    final v4r = v.reshape([1, seqLen, cfg.numKeyValueHeads, cfg.headDim]);
    q.close();
    k.close();
    v.close();

    final qNorm = mx.fast.rmsNorm(q4r, weight: layer.qNorm, eps: cfg.rmsNormEps);
    final kNorm = mx.fast.rmsNorm(k4r, weight: layer.kNorm, eps: cfg.rmsNormEps);
    q4r.close();
    k4r.close();

    final qT = qNorm.transposeAxes([0, 2, 1, 3]);
    final kT = kNorm.transposeAxes([0, 2, 1, 3]);
    final vT = v4r.transposeAxes([0, 2, 1, 3]);
    qNorm.close();
    kNorm.close();
    v4r.close();

    final useExplicitRope = Platform.environment['QWEN3_TTS_TALKER_FAST_ROPE'] != '1';
    late final MlxArray qRope;
    late final MlxArray kRopeNext;
    if (useExplicitRope) {
      final pair = _standardRopeCosSin(
        seqLen,
        offset: offset,
        headDim: cfg.headDim,
        base: cfg.ropeTheta,
        dtype: qT.dtype,
      );
      final rope = _applyStandardRope(qT, kT, pair.cos, pair.sin);
      pair.cos.close();
      pair.sin.close();
      qRope = rope.q;
      kRopeNext = rope.k;
    } else {
      qRope = mx.fast.rope(
        qT,
        dims: cfg.headDim,
        traditional: false,
        base: cfg.ropeTheta,
        offset: offset,
      );
      kRopeNext = mx.fast.rope(
        kT,
        dims: cfg.headDim,
        traditional: false,
        base: cfg.ropeTheta,
        offset: offset,
      );
    }
    qT.close();
    kT.close();

    final fetched = cache.updateAndFetch(kRopeNext, vT);
    final kBase = fetched.$1;
    final vBase = fetched.$2;
    final directGqaMode = Platform.environment['QWEN3_TTS_USE_DIRECT_GQA'];
    final useDirectGqa =
        directGqaMode == '1' ||
        (directGqaMode == 'decode' && offset > 0 && seqLen == 1);
    final repeatKv = cfg.numAttentionHeads ~/ cfg.numKeyValueHeads;
    final kAttn = repeatKv > 1 && !useDirectGqa ? kBase.repeat(repeatKv, axis: 1) : kBase;
    final vAttn = repeatKv > 1 && !useDirectGqa ? vBase.repeat(repeatKv, axis: 1) : vBase;
    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kAttn,
      vAttn,
      scale: 1 / math.sqrt(cfg.headDim),
      maskMode: offset > 0 && seqLen == 1 ? '' : 'causal',
    );
    qRope.close();
    if (!identical(kAttn, kBase)) {
      kAttn.close();
    }
    if (!identical(vAttn, vBase)) {
      vAttn.close();
    }

    final merged = attn
        .transposeAxes([0, 2, 1, 3])
        .reshape([seqLen, cfg.numAttentionHeads * cfg.headDim]);
    attn.close();
    final projected = _quantLinear(
      merged,
      layer.oProj,
      _quant,
      outDim: cfg.hiddenSize,
      addBias: true,
    );
    merged.close();
    return projected.reshape([1, seqLen, cfg.hiddenSize]);
  }

  MlxArray _mlp(
    _Qwen3TtsBlockLayer layer,
    MlxArray input,
    int seqLen,
    _Qwen3TtsBlockConfig cfg,
  ) {
    final x2d = input.reshape([seqLen, cfg.hiddenSize]);
    final gate = _quantLinear(
      x2d,
      layer.gateProj,
      _quant,
      outDim: cfg.intermediateSize,
      addBias: false,
    );
    final up = _quantLinear(
      x2d,
      layer.upProj,
      _quant,
      outDim: cfg.intermediateSize,
      addBias: false,
    );
    x2d.close();
    final activated = _silu(gate);
    gate.close();
    final fused = activated * up;
    activated.close();
    up.close();
    final down = _quantLinear(
      fused,
      layer.downProj,
      _quant,
      outDim: cfg.hiddenSize,
      addBias: false,
    );
    fused.close();
    return down.reshape([1, seqLen, cfg.hiddenSize]);
  }
}
