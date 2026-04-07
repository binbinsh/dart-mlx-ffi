part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// PaddleOcrVlRunner — load, forward, generate
// ---------------------------------------------------------------------------

/// Runner for the PaddleOCR-VL-1.5 vision-language model.
///
/// Load from a local snapshot directory (containing config.json and
/// *.safetensors files):
///
/// ```dart
/// final runner = PaddleOcrVlRunner.load('/path/to/snapshot');
/// final ids = runner.generate(tokenIds, positionIds, maxNewTokens: 256);
/// runner.close();
/// ```
final class PaddleOcrVlRunner {
  PaddleOcrVlRunner._(
    this.config,
    this._tensors,
    this._visionWeights,
    this._lmLayers,
    this._embedWeights,
    this._finalNorm,
    this._lmHead,
  );

  /// Load model weights from a snapshot directory.
  factory PaddleOcrVlRunner.load(String snapshotPath) {
    final config = PaddleOcrVlConfig.fromSnapshot(snapshotPath);
    final tensors = loadTensorMap(snapshotPath);
    final defaultQuant = config._defaultQuantSpec();

    // ── Vision encoder ──
    final visionWeights = _loadVisionWeights(tensors, config, defaultQuant);

    // ── Language model ──
    const lmPrefix = 'language_model.model.';
    final embedWeights = _LinearBase.load(
      tensors,
      '${lmPrefix}embed_tokens',
      defaultQuant: defaultQuant,
    );
    final finalNorm = tensors['${lmPrefix}norm.weight']!;
    final lmHead = config.tieWordEmbeddings
        ? null
        : _LinearBase.maybeLoad(
            tensors,
            'language_model.lm_head',
            defaultQuant: defaultQuant,
          );

    final layers = List<_LmLayerWeights>.generate(config.numHiddenLayers, (i) {
      final p = '${lmPrefix}layers.$i.';
      return _LmLayerWeights(
        inputNorm: tensors['${p}input_layernorm.weight']!,
        postNorm: tensors['${p}post_attention_layernorm.weight']!,
        attention: _LmAttentionWeights(
          qProj: _LinearBase.load(
            tensors,
            '${p}self_attn.q_proj',
            defaultQuant: defaultQuant,
          ),
          kProj: _LinearBase.load(
            tensors,
            '${p}self_attn.k_proj',
            defaultQuant: defaultQuant,
          ),
          vProj: _LinearBase.load(
            tensors,
            '${p}self_attn.v_proj',
            defaultQuant: defaultQuant,
          ),
          oProj: _LinearBase.load(
            tensors,
            '${p}self_attn.o_proj',
            defaultQuant: defaultQuant,
          ),
        ),
        mlp: _LmMlpWeights(
          gateProj: _LinearBase.load(
            tensors,
            '${p}mlp.gate_proj',
            defaultQuant: defaultQuant,
          ),
          upProj: _LinearBase.load(
            tensors,
            '${p}mlp.up_proj',
            defaultQuant: defaultQuant,
          ),
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

    // Apply platform-specific memory limits so the OS doesn't kill us.
    runner._applyMemoryLimits();

    return runner;
  }

  /// Applies recommended memory/cache limits for the current platform.
  void _applyMemoryLimits() {
    final memLimit = config.recommendedMemoryLimitBytesForCurrentPlatform;
    if (memLimit > 0) {
      try {
        MlxMemory.setMemoryLimitBytes(memLimit);
      } catch (_) {}
    }
    final cacheLimit = config.recommendedCacheLimitBytesForCurrentPlatform;
    if (cacheLimit > 0) {
      try {
        MlxMemory.setCacheLimitBytes(cacheLimit);
      } catch (_) {}
    }
  }

  final PaddleOcrVlConfig config;
  final Map<String, MlxArray> _tensors;
  final _VisionWeights _visionWeights;
  final List<_LmLayerWeights> _lmLayers;
  final _LinearBase _embedWeights;
  final MlxArray _finalNorm;
  final _LinearBase? _lmHead;
  MlxArray? _ropeInvFreq;
  Float32List? _visionPositionEmbeddingCache;

  MlxDType get visionInputDType => config.forceFloat32VisionForCurrentPlatform
      ? MlxDType.MLX_FLOAT32
      : _visionWeights.patchEmbedWeight.dtype;

  /// Release all vision encoder weights (ViT blocks + projector + patch
  /// embedding + position embedding + post-layernorm) to free ~385 MB of
  /// GPU memory.  After this call, the vision encoder cannot be used again.
  void _releaseVisionWeights() {
    if (_visionWeights.isReleased) return;
    _visionWeights.release(_tensors);
    _visionPositionEmbeddingCache = null;
    try {
      MlxMemory.clearCache();
    } catch (_) {}
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /// Generate token IDs from a text-only prompt.
  ///
  /// [promptIds] are token IDs for the text prompt.
  /// Returns the full sequence (prompt + generated tokens).
  List<int> generate(List<int> promptIds, {int maxNewTokens = 512}) {
    final posIds = _textPositionIds(promptIds.length);
    try {
      return _generateGreedy(
        this,
        promptIds,
        posIds,
        maxNewTokens,
        eosTokenId: config.eosTokenId,
      );
    } finally {
      posIds.close();
    }
  }

  /// Generate token IDs from a vision-language prompt.
  ///
  /// [promptIds] are token IDs with image_token_id placeholders.
  /// [imagePixels] is a pre-processed image tensor `[1, H, W, C]` (NHWC).
  /// Returns the full sequence (prompt + generated tokens).
  ({List<int> fullTokenIds, int expandedPromptLength})
  generateFromImageDetailed(
    List<int> promptIds,
    MlxArray imagePixels, {
    int maxNewTokens = 512,
    void Function(String message)? onStage,
  }) {
    // 1. Encode the image through ViT + projector
    onStage?.call('generateFromImage: encodeImage start');
    final imageEncoding = _encodeImage(imagePixels, onStage: onStage);
    final imageHidden = imageEncoding.hidden;
    onStage?.call(
      'generateFromImage: encodeImage done shape=${imageHidden.shape} '
      'grid=${imageEncoding.gridHeight}x${imageEncoding.gridWidth}',
    );
    // imageHidden: [mergedTokens, lmHidden]
    final numImageTokens = imageHidden.shape[0];

    // 2. Expand legacy single-placeholder prompts if needed.
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    onStage?.call(
      'generateFromImage: promptImageTokens=$imageTokenCountInPrompt '
      'expandedIds=${expandedIds.length}',
    );

    // 3. Build multimodal M-RoPE positions from the actual image grid.
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    onStage?.call('generateFromImage: positionIds ready');
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    onStage?.call(
      'generateFromImage: multimodal embeddings shape=${embeddings.shape}',
    );
    imageHidden.close();
    if (config.enableAggressiveCacheClearingForCurrentPlatform) {
      try {
        MlxMemory.clearCache();
      } catch (_) {}
    }

    // 3b. Release vision encoder weights — they are no longer needed and
    //     freeing them reclaims ~385 MB of GPU memory for the decode phase.
    if (config.enableVisionWeightReleaseForCurrentPlatform) {
      _releaseVisionWeights();
      onStage?.call('generateFromImage: vision weights released');
    }

    // 4. Run LM forward pass on embeddings (not token IDs)
    final cache = _ModelCache.create(
      numLayers: config.numHiddenLayers,
      numKvHeads: config.numKeyValueHeads,
      headDim: config.headDim,
      maxSeqLen: config.maxKvCacheSeqLenForCurrentPlatform,
    );
    try {
      onStage?.call('generateFromImage: forwardFromEmbedding start');
      var logits = _forwardFromEmbedding(embeddings, posIds, cache);
      onStage?.call('generateFromImage: forwardFromEmbedding done');
      embeddings.close();
      posIds.close();

      final tokens = List<int>.from(expandedIds);
      try {
        var nextTextPosition = positionInfo.nextTextPosition;
        for (var step = 0; step < maxNewTokens; step++) {
          final next = _nextTokenFromLogits(logits);
          tokens.add(next);
          if (step == 0) {
            onStage?.call('generateFromImage: first token=$next');
          }
          if (next == config.eosTokenId) break;
          if (step + 1 >= maxNewTokens) break;

          logits.close();
          final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
          final stepPos = _textPositionIds(1, offset: nextTextPosition);
          logits = _forwardWithCache(stepArr, stepPos, cache);
          if (step == 0) {
            onStage?.call('generateFromImage: first cached forward done');
          }
          stepArr.close();
          stepPos.close();
          if (config.enableAggressiveCacheClearingForCurrentPlatform) {
            try {
              MlxMemory.clearCache();
            } catch (_) {}
          }
          if ((step + 1) % 8 == 0) {
            onStage?.call('generateFromImage: generated ${step + 1} tokens');
          }
          nextTextPosition++;
        }
      } finally {
        logits.close();
      }
      return (fullTokenIds: tokens, expandedPromptLength: expandedIds.length);
    } finally {
      cache.close();
    }
  }

  List<int> generateFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    int maxNewTokens = 512,
    void Function(String message)? onStage,
  }) {
    return generateFromImageDetailed(
      promptIds,
      imagePixels,
      maxNewTokens: maxNewTokens,
      onStage: onStage,
    ).fullTokenIds;
  }

  ({List<int> fullTokenIds, int expandedPromptLength})
  generateFromVisionFeaturesDetailed(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    int maxNewTokens = 512,
    void Function(String message)? onStage,
  }) {
    final numImageTokens = imageHidden.shape[0];

    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    onStage?.call(
      'generateFromVisionFeatures: promptImageTokens=$imageTokenCountInPrompt '
      'expandedIds=${expandedIds.length}',
    );

    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    onStage?.call('generateFromVisionFeatures: positionIds ready');
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    onStage?.call(
      'generateFromVisionFeatures: multimodal embeddings '
      'shape=${embeddings.shape}',
    );

    final cache = _ModelCache.create(
      numLayers: config.numHiddenLayers,
      numKvHeads: config.numKeyValueHeads,
      headDim: config.headDim,
      maxSeqLen: config.maxKvCacheSeqLenForCurrentPlatform,
    );
    try {
      var logits = _forwardFromEmbedding(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();

      final tokens = List<int>.from(expandedIds);
      try {
        var nextTextPosition = positionInfo.nextTextPosition;
        for (var step = 0; step < maxNewTokens; step++) {
          final next = _nextTokenFromLogits(logits);
          tokens.add(next);
          if (next == config.eosTokenId) break;
          if (step + 1 >= maxNewTokens) break;

          logits.close();
          final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
          final stepPos = _textPositionIds(1, offset: nextTextPosition);
          logits = _forwardWithCache(stepArr, stepPos, cache);
          stepArr.close();
          stepPos.close();
          nextTextPosition++;
        }
      } finally {
        logits.close();
      }
      return (fullTokenIds: tokens, expandedPromptLength: expandedIds.length);
    } finally {
      cache.close();
    }
  }

  /// Encode an image and return the projected vision features.
  ///
  /// Intended for debugging/parity checks against reference implementations.
  MlxArray encodeImageFeatures(
    MlxArray imagePixels, {
    void Function(String message)? onStage,
  }) {
    final encoded = _encodeImage(imagePixels, onStage: onStage);
    return encoded.hidden;
  }

  /// Return vision embeddings after patch embedding + position interpolation,
  /// before any transformer layers. Intended for parity debugging only.
  MlxArray encodeImageEmbeddingsOnly(MlxArray imagePixels) {
    final vCfg = config._vision;
    final patchOut = mx.conv2d(
      imagePixels,
      _visionWeights.patchEmbedWeight,
      stride: [vCfg.patchSize, vCfg.patchSize],
    );
    final gridH = patchOut.shape[1];
    final gridW = patchOut.shape[2];
    var hidden = patchOut.reshape([gridH * gridW, vCfg.hiddenSize]);
    patchOut.close();
    if (_visionWeights.patchEmbedBias != null) {
      final biased = mx.add(hidden, _visionWeights.patchEmbedBias!);
      hidden.close();
      hidden = biased;
    }
    final posEmbed = _interpolateVisionPositionEmbedding(
      gridH,
      gridW,
      hidden.dtype,
    );
    final withPos = mx.add(hidden, posEmbed);
    hidden.close();
    posEmbed.close();
    return withPos;
  }

  /// Return vision hidden states after a given number of transformer layers,
  /// before post-layernorm/projector. Intended for parity debugging only.
  MlxArray encodeImageAfterLayerCount(
    MlxArray imagePixels,
    int layerCount,
  ) {
    final vCfg = config._vision;
    var hidden = encodeImageEmbeddingsOnly(imagePixels);
    final gridH = imagePixels.shape[1] ~/ vCfg.patchSize;
    final gridW = imagePixels.shape[2] ~/ vCfg.patchSize;
    final rotaryPosEmb = _buildVisionRotaryPosEmbedding(
      gridH,
      gridW,
      MlxDType.MLX_FLOAT32,
    );
    try {
      final total = math.min(layerCount, _visionWeights.blocks.length);
      for (var i = 0; i < total; i++) {
        hidden = _visionBlock(_visionWeights.blocks[i], hidden, vCfg, rotaryPosEmb);
      }
      return hidden;
    } finally {
      rotaryPosEmb.close();
    }
  }

  /// Return the vision hidden states after all transformer layers and
  /// post-layernorm, before projector.
  MlxArray encodeImagePostNormHidden(MlxArray imagePixels) {
    final hidden = encodeImageAfterLayerCount(
      imagePixels,
      _visionWeights.blocks.length,
    );
    final out = _visionLayerNorm(
      hidden,
      weight: _visionWeights.postLayerNormWeight,
      bias: _visionWeights.postLayerNormBias,
      eps: config._vision.layerNormEps,
    );
    hidden.close();
    return out;
  }

  /// Return the vision rotary embedding for the current image grid.
  MlxArray encodeVisionRotaryEmbedding(MlxArray imagePixels) {
    final vCfg = config._vision;
    final gridH = imagePixels.shape[1] ~/ vCfg.patchSize;
    final gridW = imagePixels.shape[2] ~/ vCfg.patchSize;
    return _buildVisionRotaryPosEmbedding(gridH, gridW, MlxDType.MLX_FLOAT32);
  }

  ({MlxArray ids, int nextTextPosition}) debugMultimodalPositionIds(
    List<int> tokenIds,
    int gridH,
    int gridW,
  ) => _multimodalPositionIds(tokenIds, gridH, gridW);

  ({MlxArray cos, MlxArray sin}) debugMropeCosSin(
    List<int> tokenIds,
    int gridH,
    int gridW,
  ) {
    final info = _multimodalPositionIds(tokenIds, gridH, gridW);
    try {
      return _buildMropeCosSin(info.ids, MlxDType.MLX_FLOAT32);
    } finally {
      info.ids.close();
    }
  }

  /// Return the first vision-layer self-attention output before residual add.
  /// Intended for parity debugging only.
  MlxArray debugFirstVisionAttentionOutput(MlxArray imagePixels) {
    return debugVisionAttentionOutput(imagePixels, 0);
  }

  /// Return a vision-layer self-attention output before residual add.
  /// [layerIndex] is zero-based.
  MlxArray debugVisionAttentionOutput(MlxArray imagePixels, int layerIndex) {
    final input = encodeImageEmbeddingsOnly(imagePixels);
    final vCfg = config._vision;
    final gridH = imagePixels.shape[1] ~/ vCfg.patchSize;
    final gridW = imagePixels.shape[2] ~/ vCfg.patchSize;
    final rotary32 = _buildVisionRotaryPosEmbedding(
      gridH,
      gridW,
      MlxDType.MLX_FLOAT32,
    );
    try {
      final clampedIndex = layerIndex.clamp(0, _visionWeights.blocks.length - 1);
      var hidden = input;
      for (var i = 0; i < clampedIndex; i++) {
        hidden = _visionBlock(_visionWeights.blocks[i], hidden, vCfg, rotary32);
      }
      final block = _visionWeights.blocks[clampedIndex];
      final norm1 = _visionLayerNorm(
        hidden,
        weight: block.layerNorm1Weight,
        bias: block.layerNorm1Bias,
        eps: vCfg.layerNormEps,
      );
      try {
        return _visionAttention(block, norm1, vCfg, rotary32);
      } finally {
        norm1.close();
        if (hidden != input) hidden.close();
      }
    } finally {
      input.close();
      rotary32.close();
    }
  }

  /// Return the first vision-layer MLP output before residual add.
  /// Intended for parity debugging only.
  MlxArray debugFirstVisionMlpOutput(MlxArray imagePixels) {
    return debugVisionMlpOutput(imagePixels, 0);
  }

  /// Return a vision-layer MLP output before residual add.
  /// [layerIndex] is zero-based.
  MlxArray debugVisionMlpOutput(MlxArray imagePixels, int layerIndex) {
    final vCfg = config._vision;
    final input = encodeImageEmbeddingsOnly(imagePixels);
    final rotary = _buildVisionRotaryPosEmbedding(
      imagePixels.shape[1] ~/ vCfg.patchSize,
      imagePixels.shape[2] ~/ vCfg.patchSize,
      MlxDType.MLX_FLOAT32,
    );
    try {
      final clampedIndex = layerIndex.clamp(0, _visionWeights.blocks.length - 1);
      var hidden = input;
      for (var i = 0; i < clampedIndex; i++) {
        hidden = _visionBlock(_visionWeights.blocks[i], hidden, vCfg, rotary);
      }
      final block = _visionWeights.blocks[clampedIndex];
      final norm1 = _visionLayerNorm(
        hidden,
        weight: block.layerNorm1Weight,
        bias: block.layerNorm1Bias,
        eps: vCfg.layerNormEps,
      );
      try {
        final attnOut = _visionAttention(block, norm1, vCfg, rotary);
        try {
          final h = mx.add(hidden, attnOut);
          try {
            final norm2 = _visionLayerNorm(
              h,
              weight: block.layerNorm2Weight,
              bias: block.layerNorm2Bias,
              eps: vCfg.layerNormEps,
            );
            try {
              return _visionMlp(block, norm2);
            } finally {
              norm2.close();
            }
          } finally {
            h.close();
          }
        } finally {
          attnOut.close();
        }
      } finally {
        norm1.close();
        if (hidden != input) hidden.close();
      }
    } finally {
      input.close();
      rotary.close();
    }
  }

  /// Close all resources. The runner must not be used after this call.
  void close() {
    _ropeInvFreq?.close();
    _ropeInvFreq = null;
    for (final tensor in _tensors.values) {
      tensor.close();
    }
  }

  // -----------------------------------------------------------------------
  // Forward passes
  // -----------------------------------------------------------------------

  /// Forward pass from token IDs with KV cache. Returns logits.
  MlxArray _forwardWithCache(
    MlxArray ids,
    MlxArray positionIds,
    _ModelCache cache,
  ) {
    final hidden = _embed(ids);
    return _forwardFromEmbedding(hidden, positionIds, cache);
  }

  /// Forward pass from pre-computed embeddings with KV cache.
  MlxArray _forwardFromEmbedding(
    MlxArray hidden,
    MlxArray positionIds,
    _ModelCache cache,
  ) {
    final seqLen = hidden.shape[1];
    var h = hidden;
    try {
      for (var i = 0; i < _lmLayers.length; i++) {
        final next = _decoderLayer(
          _lmLayers[i],
          h,
          seqLen,
          positionIds,
          cache: cache.layers[i],
        );
        if (config.enableDecoderLayerwiseEvalForCurrentPlatform) {
          MlxRuntime.evalAll([next]);
        }
        if (config.enableAggressiveCacheClearingForCurrentPlatform) {
          try {
            MlxMemory.clearCache();
          } catch (_) {}
        }
        if (h != hidden) h.close();
        h = next;
      }

      // Final norm
      final norm = mx.fast.rmsNorm(
        h,
        weight: _finalNorm,
        eps: config.rmsNormEps,
      );
      h.close();

      // LM head — only last token
      final last = norm.slice(
        start: [0, seqLen - 1, 0],
        stop: [1, seqLen, config.hiddenSize],
      );
      norm.close();

      final last2d = last.reshape([1, config.hiddenSize]);
      last.close();

      final linear = config.tieWordEmbeddings ? _embedWeights : _lmHead!;
      final logits = linear.apply(last2d);
      last2d.close();

      final shaped = logits.reshape([1, config.vocabSize]);
      if (shaped != logits) logits.close();
      MlxRuntime.evalAll([shaped]);
      if (config.enableAggressiveCacheClearingForCurrentPlatform) {
        try {
          MlxMemory.clearCache();
        } catch (_) {}
      }
      return shaped;
    } catch (_) {
      if (h != hidden) h.close();
      rethrow;
    }
  }

  // -----------------------------------------------------------------------
  // Embedding
  // -----------------------------------------------------------------------

  MlxArray _embed(MlxArray ids) {
    final seqLen = ids.shape[1];
    final rows = ids.reshape([seqLen]);
    try {
      if (_embedWeights case final _QuantLinear q) {
        final rowsW = q.weight.take(rows, axis: 0);
        final rowsS = q.scales.take(rows, axis: 0);
        final rowsB = q.biases?.take(rows, axis: 0);
        final gathered = MlxQuantizedMatrix(rowsW, rowsS, rowsB);
        try {
          final out = mx.quant.dequantize(
            gathered,
            groupSize: q.quantSpec.groupSize,
            bits: q.quantSpec.bits,
            mode: q.quantSpec.mode,
          );
          MlxRuntime.evalAll([out]);
          return out.reshape([1, seqLen, config.hiddenSize]);
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      if (_embedWeights case final _DenseLinear d) {
        final out = d.weight.take(rows, axis: 0);
        return out.reshape([1, seqLen, config.hiddenSize]);
      }
      throw StateError('Unsupported embedding type.');
    } finally {
      rows.close();
    }
  }

  // -----------------------------------------------------------------------
  // Multimodal embedding construction
  // -----------------------------------------------------------------------

  /// Expand a single `image_token_id` into [numImageTokens] copies.
  List<int> _expandImageTokens(List<int> tokenIds, int numImageTokens) {
    final result = <int>[];
    for (final id in tokenIds) {
      if (id == config.imageTokenId) {
        for (var j = 0; j < numImageTokens; j++) {
          result.add(config.imageTokenId);
        }
      } else {
        result.add(id);
      }
    }
    return result;
  }

  /// Build embeddings where image_token_id positions are replaced with
  /// vision encoder hidden states.
  ///
  /// [tokenIds] is the expanded token list.
  /// [imageHidden] shape: `[numImageTokens, lmHidden]`.
  /// Returns shape: `[1, totalLen, lmHidden]`.
  MlxArray _buildMultimodalEmbedding(List<int> tokenIds, MlxArray imageHidden) {
    final totalLen = tokenIds.length;

    // Get text embeddings for all tokens
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, totalLen]);
    final textEmbed = _embed(ids);
    ids.close();

    // Find image token ranges and replace
    var imgIdx = 0;
    final segments = <MlxArray>[];
    var segStart = 0;

    for (var i = 0; i < totalLen; i++) {
      if (tokenIds[i] == config.imageTokenId) {
        // Emit preceding text segment
        if (i > segStart) {
          segments.add(
            textEmbed.slice(
              start: [0, segStart, 0],
              stop: [1, i, config.hiddenSize],
            ),
          );
        }
        // Find contiguous run of image tokens
        var runEnd = i;
        while (runEnd < totalLen && tokenIds[runEnd] == config.imageTokenId) {
          runEnd++;
        }
        final runLen = runEnd - i;
        // Insert image hidden states
        segments.add(
          imageHidden
              .slice(
                start: [imgIdx, 0],
                stop: [imgIdx + runLen, config.hiddenSize],
              )
              .reshape([1, runLen, config.hiddenSize]),
        );
        imgIdx += runLen;
        segStart = runEnd;
        i = runEnd - 1; // loop will increment
      }
    }

    // Trailing text segment
    if (segStart < totalLen) {
      segments.add(
        textEmbed.slice(
          start: [0, segStart, 0],
          stop: [1, totalLen, config.hiddenSize],
        ),
      );
    }

    final result = segments.length == 1
        ? segments.first
        : mx.concatenate(segments, axis: 1);
    for (final seg in segments) {
      if (seg != result) seg.close();
    }
    textEmbed.close();
    return result;
  }

  // -----------------------------------------------------------------------
  // Vision weight loading
  // -----------------------------------------------------------------------

  static _VisionWeights _loadVisionWeights(
    Map<String, MlxArray> tensors,
    PaddleOcrVlConfig config,
    _QuantSpec defaultQuant,
  ) {
    const vp = 'visual.';
    final embeddingPrefix = _firstExistingPrefix(tensors, const [
      '${vp}patch_embedding',
      '${vp}embeddings.patch_embedding',
    ]);
    final positionPrefix = _firstExistingPrefix(tensors, const [
      '${vp}position_embedding',
      '${vp}embeddings.position_embedding',
    ]);
    final blockPrefixBase = _firstExistingPrefix(tensors, const [
      '${vp}blocks.0',
      '${vp}layers.0',
    ]);
    final blockCollectionPrefix = blockPrefixBase.substring(
      0,
      blockPrefixBase.length - 1,
    );
    final projectorPrefix = _firstExistingPrefix(tensors, const [
      '${vp}merger',
      '${vp}projector',
    ]);

    // Patch embedding (Conv2d)
    final patchWeight = tensors['$embeddingPrefix.weight']!;
    final patchBias = tensors['$embeddingPrefix.bias'];

    // Position embedding
    final posEmbed = _LinearBase.load(
      tensors,
      positionPrefix,
      defaultQuant: defaultQuant,
    );

    // ViT blocks
    final blocks = List<_VisionBlockWeights>.generate(
      config._vision.numHiddenLayers,
      (i) {
        final bp = '$blockCollectionPrefix$i.';
        final attentionPrefix = _firstExistingPrefix(tensors, [
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

    // Spatial-merge projector
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

  static String _firstExistingPrefix(
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
}
