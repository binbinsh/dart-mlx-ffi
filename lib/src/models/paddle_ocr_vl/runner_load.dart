part of 'paddle_ocr_vl.dart';

PaddleOcrVlRunner _loadPaddleOcrVlRunner(
  String snapshotPath, {
  bool keepVisionWeights = true,
}) {
  final config = PaddleOcrVlConfig.fromSnapshot(snapshotPath);
  final tensors = loadTensorMap(snapshotPath);
  final defaultQuant = config._defaultQuantSpec();

  // When the caller opts out of the vision encoder, drop every `visual.*`
  // tensor before any wrapper structure is built. `loadTensorMap` resolves
  // the safetensors map eagerly (see `lib/src/models/shared/tensor_map.dart`)
  // so we cannot avoid the initial materialization at the byte level without
  // changing that shared helper. Closing the MlxArray handles here releases
  // the underlying buffers immediately, which still removes the largest
  // chunk of GPU memory (~385 MB on the real PaddleOCR-VL-1.5 weights) and
  // keeps every `_LinearBase.load` call below from touching `visual.*`.
  if (!keepVisionWeights) {
    final visualKeys =
        tensors.keys.where((k) => k.startsWith('visual.')).toList();
    for (final k in visualKeys) {
      tensors.remove(k)?.close();
    }
  }

  final _VisionWeights? visionWeights = keepVisionWeights
      ? _loadRunnerVisionWeights(tensors, config, defaultQuant)
      : null;

  const lmPrefix = 'language_model.model.';
  final embedWeights = _LinearBase.load(
    tensors,
    '${lmPrefix}embed_tokens',
    defaultQuant: defaultQuant,
  );
  final finalNorm = _prepareRunnerLmNormWeight(tensors, '${lmPrefix}norm.weight');
  final lmHead = config.tieWordEmbeddings
      ? null
      : _LinearBase.maybeLoad(
          tensors,
          'language_model.lm_head',
          defaultQuant: defaultQuant,
        );

  final layers = List<_LmLayerWeights>.generate(config.numHiddenLayers, (i) {
    final p = '${lmPrefix}layers.$i.';
    final qProj = _LinearBase.load(
      tensors,
      '${p}self_attn.q_proj',
      defaultQuant: defaultQuant,
    );
    final kProj = _LinearBase.load(
      tensors,
      '${p}self_attn.k_proj',
      defaultQuant: defaultQuant,
    );
    final vProj = _LinearBase.load(
      tensors,
      '${p}self_attn.v_proj',
      defaultQuant: defaultQuant,
    );
    final fusedQkv = _LinearBase.maybeFuse(
      [qProj, kProj, vProj],
      prefix: '${p}self_attn.qkv_fused',
    );
    if (fusedQkv != null) {
      qProj.release(tensors);
      kProj.release(tensors);
      vProj.release(tensors);
    }
    final gateProj = _LinearBase.load(
      tensors,
      '${p}mlp.gate_proj',
      defaultQuant: defaultQuant,
    );
    final upProj = _LinearBase.load(
      tensors,
      '${p}mlp.up_proj',
      defaultQuant: defaultQuant,
    );
    final fusedGateUp = _LinearBase.maybeFuse(
      [gateProj, upProj],
      prefix: '${p}mlp.gate_up_fused',
    );
    if (fusedGateUp != null) {
      gateProj.release(tensors);
      upProj.release(tensors);
    }
    return _LmLayerWeights(
      inputNorm: _prepareRunnerLmNormWeight(
        tensors,
        '${p}input_layernorm.weight',
      ),
      postNorm: _prepareRunnerLmNormWeight(
        tensors,
        '${p}post_attention_layernorm.weight',
      ),
      attention: _LmAttentionWeights(
        qProj: fusedQkv == null ? qProj : null,
        kProj: fusedQkv == null ? kProj : null,
        vProj: fusedQkv == null ? vProj : null,
        qkvProj: fusedQkv,
        oProj: _LinearBase.load(
          tensors,
          '${p}self_attn.o_proj',
          defaultQuant: defaultQuant,
        ),
      ),
      mlp: _LmMlpWeights(
        gateProj: fusedGateUp == null ? gateProj : null,
        upProj: fusedGateUp == null ? upProj : null,
        gateUpProj: fusedGateUp,
        downProj: _LinearBase.load(
          tensors,
          '${p}mlp.down_proj',
          defaultQuant: defaultQuant,
        ),
      ),
    );
  });

  final runner = PaddleOcrVlRunner._(
    config,
    tensors,
    visionWeights,
    layers,
    embedWeights,
    finalNorm,
    lmHead,
  );
  _applyRunnerMemoryLimits(runner);

  RuntimeTuning.instance.register('paddle_ocr_vl', paddleOcrVlTuning);
  return runner;
}

void _applyRunnerMemoryLimits(PaddleOcrVlRunner runner) {
  final memLimit =
      PaddleOcrVlDebugOverrides.memoryLimitBytes ??
      runner.config.recommendedMemoryLimitBytesForCurrentPlatform;
  if (memLimit >= 0) {
    try {
      MlxMemory.setMemoryLimitBytes(memLimit);
    } catch (_) {}
  }
  final cacheLimit =
      PaddleOcrVlDebugOverrides.cacheLimitBytes ??
      runner.config.recommendedCacheLimitBytesForCurrentPlatform;
  if (cacheLimit >= 0) {
    try {
      MlxMemory.setCacheLimitBytes(cacheLimit);
    } catch (_) {}
  }
  final wiredLimit =
      PaddleOcrVlDebugOverrides.wiredLimitBytes ??
      runner.config.recommendedWiredLimitBytesForCurrentPlatform;
  if (wiredLimit >= 0) {
    try {
      MlxMemory.setWiredLimitBytes(wiredLimit);
    } catch (_) {}
  }
}

void _maybeSynchronizeRunnerGpuPerToken(PaddleOcrVlRunner runner) {
  if (!runner.config.enableForceGpuSynchronizePerTokenForCurrentPlatform) {
    return;
  }
  MlxStream? stream;
  try {
    stream = MlxStream.defaultGpu();
    stream.synchronize();
  } catch (_) {
    try {
      MlxRuntime.evalAll(const []);
    } catch (_) {}
  } finally {
    stream?.close();
  }
}

bool _runnerShouldEvalDecodeCacheState(
  PaddleOcrVlRunner runner,
  _LayerCache cache,
) {
  if (!runner.config.enableDecodeCacheStateEvalForCurrentPlatform) {
    return false;
  }
  final interval = runner.config.decodeCacheStateEvalIntervalForCurrentPlatform;
  return interval <= 1 || (cache.offset % interval) == 0;
}

void _releaseRunnerVisionWeights(PaddleOcrVlRunner runner) {
  final vw = runner._visionWeightsOrNull;
  if (vw == null) return;
  if (vw.isReleased) return;
  vw.release(runner._tensors);
  runner._visionPositionEmbeddingCache = null;
  try {
    MlxMemory.clearCache();
  } catch (_) {}
}

_VisionWeights _loadRunnerVisionWeights(
  Map<String, MlxArray> tensors,
  PaddleOcrVlConfig config,
  _QuantSpec defaultQuant,
) {
  const vp = 'visual.';
  final embeddingPrefix = _firstExistingTensorPrefix(tensors, const [
    '${vp}patch_embedding',
    '${vp}embeddings.patch_embedding',
  ]);
  final positionPrefix = _firstExistingTensorPrefix(tensors, const [
    '${vp}position_embedding',
    '${vp}embeddings.position_embedding',
  ]);
  final blockPrefixBase = _firstExistingTensorPrefix(tensors, const [
    '${vp}blocks.0',
    '${vp}layers.0',
  ]);
  final blockCollectionPrefix = blockPrefixBase.substring(
    0,
    blockPrefixBase.length - 1,
  );
  final projectorPrefix = _firstExistingTensorPrefix(tensors, const [
    '${vp}merger',
    '${vp}projector',
  ]);

  final patchWeight = tensors['$embeddingPrefix.weight']!;
  final patchBias = tensors['$embeddingPrefix.bias'];
  final posEmbed = _LinearBase.load(
    tensors,
    positionPrefix,
    defaultQuant: defaultQuant,
  );

  final blocks = List<_VisionBlockWeights>.generate(
    config._vision.numHiddenLayers,
    (i) {
      final bp = '$blockCollectionPrefix$i.';
      final attentionPrefix = _firstExistingTensorPrefix(tensors, [
        '${bp}attn',
        '${bp}self_attn',
      ]);
      return _VisionBlockWeights(
        layerNorm1Weight: tensors['${bp}layer_norm1.weight']!,
        layerNorm1Bias: tensors['${bp}layer_norm1.bias']!,
        layerNorm2Weight: tensors['${bp}layer_norm2.weight']!,
        layerNorm2Bias: tensors['${bp}layer_norm2.bias']!,
        layerNorm1Key: '${bp}layer_norm1.weight',
        layerNorm1BiasKey: '${bp}layer_norm1.bias',
        layerNorm2Key: '${bp}layer_norm2.weight',
        layerNorm2BiasKey: '${bp}layer_norm2.bias',
        qkv: _LinearBase.load(
          tensors,
          '$attentionPrefix.qkv',
          defaultQuant: defaultQuant,
        ),
        outProj: _LinearBase.load(
          tensors,
          '$attentionPrefix.out_proj',
          defaultQuant: defaultQuant,
        ),
        fc1: _LinearBase.load(
          tensors,
          '${bp}mlp.fc1',
          defaultQuant: defaultQuant,
        ),
        fc2: _LinearBase.load(
          tensors,
          '${bp}mlp.fc2',
          defaultQuant: defaultQuant,
        ),
      );
    },
  );

  final projector = _ProjectorWeights(
    preNormWeight: tensors['$projectorPrefix.pre_norm.weight']!,
    preNormBias: tensors['$projectorPrefix.pre_norm.bias']!,
    preNormWeightKey: '$projectorPrefix.pre_norm.weight',
    preNormBiasKey: '$projectorPrefix.pre_norm.bias',
    linear1: _LinearBase.load(
      tensors,
      '$projectorPrefix.linear_1',
      defaultQuant: defaultQuant,
    ),
    linear2: _LinearBase.load(
      tensors,
      '$projectorPrefix.linear_2',
      defaultQuant: defaultQuant,
    ),
  );

  return _VisionWeights(
    patchEmbedWeight: patchWeight,
    patchEmbedBias: patchBias,
    patchEmbedWeightKey: '$embeddingPrefix.weight',
    patchEmbedBiasKey: patchBias != null ? '$embeddingPrefix.bias' : null,
    positionEmbedding: posEmbed,
    blocks: blocks,
    postLayerNormWeight: tensors['${vp}post_layernorm.weight']!,
    postLayerNormBias: tensors['${vp}post_layernorm.bias']!,
    postLayerNormWeightKey: '${vp}post_layernorm.weight',
    postLayerNormBiasKey: '${vp}post_layernorm.bias',
    projector: projector,
  );
}

String _firstExistingTensorPrefix(
  Map<String, MlxArray> tensors,
  List<String> candidates,
) {
  for (final candidate in candidates) {
    if (tensors.keys.any((key) => key.startsWith(candidate))) {
      return candidate;
    }
  }
  throw StateError('Missing tensors for prefixes: ${candidates.join(', ')}');
}

MlxArray _prepareRunnerLmNormWeight(
  Map<String, MlxArray> tensors,
  String key,
) {
  return tensors[key]!;
}
