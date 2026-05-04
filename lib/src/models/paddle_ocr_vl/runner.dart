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
  factory PaddleOcrVlRunner.load(String snapshotPath) =>
      _loadPaddleOcrVlRunner(snapshotPath);

  void _maybeSynchronizeGpuPerToken() => _maybeSynchronizeRunnerGpuPerToken(this);

  bool _shouldEvalDecodeCacheState(_LayerCache cache) =>
      _runnerShouldEvalDecodeCacheState(this, cache);

  final PaddleOcrVlConfig config;
  final Map<String, MlxArray> _tensors;
  final _VisionWeights _visionWeights;
  final List<_LmLayerWeights> _lmLayers;
  final _LinearBase _embedWeights;
  final MlxArray _finalNorm;
  final _LinearBase? _lmHead;
  MlxArray? _ropeInvFreq;
  Float32List? _visionPositionEmbeddingCache;
  final Map<String, MlxArray> _visionInterpolatedPositionEmbeddingArrayCache = {};
  final Map<String, MlxArray> _visionRotaryEmbeddingArrayCache = {};
  final Map<int, ({MlxArray weight, MlxArray scales, MlxArray? biases})>
      _debugLmHeadPrefixQuantCache = {};

  MlxDType get visionInputDType => config.forceFloat32VisionForCurrentPlatform
      ? MlxDType.MLX_FLOAT32
      : _visionWeights.patchEmbedWeight.dtype;

  /// Release all vision encoder weights (ViT blocks + projector + patch
  /// embedding + position embedding + post-layernorm) to free ~385 MB of
  /// GPU memory.  After this call, the vision encoder cannot be used again.
  ///
  /// See issue #1 (hybrid CoreML-NaViT + MLX-decoder refactor) — the new
  /// `keepVisionWeights: false` load path on `runner_load.dart` makes this
  /// release structurally unnecessary in the hybrid runner; this method
  /// remains for the legacy full-model load path.
  void _releaseVisionWeights() => _releaseRunnerVisionWeights(this);

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
    // Release vision weights before building multimodal embeddings. At this
    // point we only need the projected imageHidden features, not the encoder.
    if (config.enableVisionWeightReleaseForCurrentPlatform) {
      _releaseVisionWeights();
      onStage?.call('generateFromImage: vision weights released');
    }
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

    // 4. Run LM forward pass on embeddings (not token IDs)
    final cache = _ModelCache.create(config: config);
    try {
      onStage?.call(
        'generateFromImage: decode config '
        'layerEval=${config.enableDecoderLayerwiseEvalForCurrentPlatform} '
        'clearCache=${config.enableAggressiveCacheClearingForCurrentPlatform} '
        'cacheEvalInterval=${config.decodeCacheStateEvalIntervalForCurrentPlatform} '
        'layerTrace=${config.enableDecoderPerLayerTraceForCurrentPlatform} '
        'substepTrace=${config.enableDecoderSubstepTraceForCurrentPlatform} '
        'tailTrace=${config.enableDecoderTailTraceForCurrentPlatform} '
        'syncEachToken=${config.enableForceGpuSynchronizePerTokenForCurrentPlatform} '
        'logitsDetach=${config.enableDecodeLogitsDetachForCurrentPlatform}',
      );
      onStage?.call('generateFromImage: forwardFromEmbedding start');
      MlxArray? logits = _prefillFromEmbeddingWithCache(
        embeddings,
        posIds,
        cache,
        onStage: onStage,
      );
      onStage?.call('generateFromImage: forwardFromEmbedding done');
      MlxMemory.resetGpuPrimitiveTraceBudgets();
      onStage?.call(
        'generateFromImage: cache kinds after prefill ${cache.describeLayerKinds()}',
      );
      embeddings.close();
      posIds.close();

      final tokens = List<int>.from(expandedIds);
      try {
        var nextTextPosition = positionInfo.nextTextPosition;
        var prefetchedDecodeToken = null as int?;
        for (var step = 0; step < maxNewTokens; step++) {
          if (config.enableDecoderTailTraceForCurrentPlatform && step >= 24) {
            final top2 = logits!.topK(2, axis: -1);
            try {
              MlxRuntime.evalAll([top2]);
              final values = top2.toFloat32List();
              final top2Value = values.length >= 2 ? values.first : double.nan;
              final top1Value = values.isNotEmpty ? values.last : double.nan;
              onStage?.call(
                'generateFromImage: token ${step + 1} gap='
                '${top1Value - top2Value} top1=$top1Value top2=$top2Value',
              );
            } finally {
              top2.close();
            }
          }
          final int next =
              prefetchedDecodeToken ??
              _traceDecodeLoopValue(this, cache, 'sample_token', () {
                return _nextTokenFromLogits(logits!);
              });
          prefetchedDecodeToken = null;
          tokens.add(next);
          if (step == 0) {
            onStage?.call('generateFromImage: first token=$next');
          }
          if (next == config.eosTokenId || step + 1 >= maxNewTokens) break;

          if (config.enableDecodeLogitsDetachForCurrentPlatform) {
            if (logits != null) {
              _traceDecoderTailVoid(this, 1, cache, 'close_logits', () {
                logits!.close();
                logits = null;
              });
              if (config.enableClearAfterCloseLogitsForCurrentPlatform) {
                _traceDecoderTailVoid(
                  this,
                  1,
                  cache,
                  'clear_after_close_logits',
                  () {
                    try {
                      MlxMemory.clearCache();
                    } catch (_) {}
                  },
                );
              }
            }
            prefetchedDecodeToken = _traceDecodeLoopValue<int>(
              this,
              cache,
              'forward_chunked_sample',
              () => _sampleNextTokenWithCache(
                this,
                next,
                nextTextPosition,
                cache,
              ),
            );
            if (step == 0) {
              onStage?.call('generateFromImage: first cached forward done');
            }
            if (config.enableAggressiveCacheClearingForCurrentPlatform) {
              try {
                MlxMemory.clearCache();
              } catch (_) {}
            }
            _traceDecoderTailVoid(this, 1, cache, 'sync_per_token', () {
              _maybeSynchronizeGpuPerToken();
            });
            if ((step + 1) % 8 == 0) {
              onStage?.call(
                'generateFromImage: generated ${step + 1} tokens '
                '${cache.describeLayerKinds()}',
              );
            } else if (step + 1 >= 24) {
              onStage?.call(
                'generateFromImage: generated ${step + 1} tokens '
                '${cache.describeLayerKinds()}',
              );
            }
            nextTextPosition++;
            continue;
          }

          _traceDecoderTailVoid(this, 1, cache, 'close_logits', () {
            logits!.close();
          });
          if (config.enableClearAfterCloseLogitsForCurrentPlatform) {
            _traceDecoderTailVoid(
              this,
              1,
              cache,
              'clear_after_close_logits',
              () {
                try {
                  MlxMemory.clearCache();
                } catch (_) {}
              },
            );
          }
          final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
          final stepPos = _textPositionIds(1, offset: nextTextPosition);
          logits = _traceDecoderTailArray(
            this,
            1,
            cache,
            'forward_total',
            () => _forwardWithCache(stepArr, stepPos, cache),
          );
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
          _traceDecoderTailVoid(this, 1, cache, 'sync_per_token', () {
            _maybeSynchronizeGpuPerToken();
          });
          if ((step + 1) % 8 == 0) {
            onStage?.call(
              'generateFromImage: generated ${step + 1} tokens '
              '${cache.describeLayerKinds()}',
            );
          } else if (step + 1 >= 24) {
            onStage?.call(
              'generateFromImage: generated ${step + 1} tokens '
              '${cache.describeLayerKinds()}',
            );
          }
          nextTextPosition++;
        }
      } finally {
        logits?.close();
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

    final cache = _ModelCache.create(config: config);
    try {
      var logits = _prefillFromEmbeddingWithCache(
        embeddings,
        posIds,
        cache,
        onStage: onStage,
      );
      MlxMemory.resetGpuPrimitiveTraceBudgets();
      onStage?.call(
        'generateFromVisionFeatures: cache kinds after prefill ${cache.describeLayerKinds()}',
      );
      embeddings.close();
      posIds.close();

      final tokens = List<int>.from(expandedIds);
      try {
        var nextTextPosition = positionInfo.nextTextPosition;
        for (var step = 0; step < maxNewTokens; step++) {
          final next = _traceDecodeLoopValue(this, cache, 'sample_token', () {
            return _nextTokenFromLogits(logits);
          });
          tokens.add(next);
          if (next == config.eosTokenId || step + 1 >= maxNewTokens) break;

          _traceDecoderTailVoid(this, 1, cache, 'close_logits', () {
            logits.close();
          });
          if (config.enableClearAfterCloseLogitsForCurrentPlatform) {
            _traceDecoderTailVoid(
              this,
              1,
              cache,
              'clear_after_close_logits',
              () {
                try {
                  MlxMemory.clearCache();
                } catch (_) {}
              },
            );
          }
          final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
          final stepPos = _textPositionIds(1, offset: nextTextPosition);
          logits = _traceDecoderTailArray(
            this,
            1,
            cache,
            'forward_total',
            () => _forwardWithCache(stepArr, stepPos, cache),
          );
          stepArr.close();
          stepPos.close();
          _traceDecoderTailVoid(this, 1, cache, 'sync_per_token', () {
            _maybeSynchronizeGpuPerToken();
          });
          if ((step + 1) % 8 == 0) {
            onStage?.call(
              'generateFromVisionFeatures: generated ${step + 1} tokens '
              '${cache.describeLayerKinds()}',
            );
          } else if (step + 1 >= 24) {
            onStage?.call(
              'generateFromVisionFeatures: generated ${step + 1} tokens '
              '${cache.describeLayerKinds()}',
            );
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

  /// Close all resources. The runner must not be used after this call.
  void close() {
    _ropeInvFreq?.close();
    _ropeInvFreq = null;
    for (final tensor in _visionInterpolatedPositionEmbeddingArrayCache.values) {
      tensor.close();
    }
    _visionInterpolatedPositionEmbeddingArrayCache.clear();
    for (final tensor in _visionRotaryEmbeddingArrayCache.values) {
      tensor.close();
    }
    _visionRotaryEmbeddingArrayCache.clear();
    for (final quant in _debugLmHeadPrefixQuantCache.values) {
      quant.weight.close();
      quant.scales.close();
      quant.biases?.close();
    }
    _debugLmHeadPrefixQuantCache.clear();
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
    _ModelCache cache, {
    bool maybeQuantizeAfter = true,
  }) {
    if (ids.shape[1] == 1) {
      _maybeStartDecoderMetalCapture(cache);
    }
    final embedded = _traceDecoderTailArray(
      this,
      ids.shape[1],
      cache,
      'embed_step',
      () => _embed(ids),
    );
    try {
      return _traceDecoderTailArray(
        this,
        ids.shape[1],
        cache,
        'forward_from_embedding',
        () => _forwardFromEmbedding(
          embedded,
          positionIds,
          cache,
          maybeQuantizeAfter: maybeQuantizeAfter,
        ),
      );
    } finally {
      embedded.close();
    }
  }

  /// Forward pass from pre-computed embeddings with KV cache.
  MlxArray _forwardFromEmbedding(
    MlxArray hidden,
    MlxArray positionIds,
    _ModelCache cache, {
    bool maybeQuantizeAfter = true,
  }) {
    final seqLen = hidden.shape[1];
    final positionEmbeddings = seqLen > 1
        ? _buildAppliedMropeCosSin(positionIds, hidden.dtype)
        : null;
    var h = hidden;
    try {
      for (var i = 0; i < _lmLayers.length; i++) {
        final trace = _beginDecoderLayerTrace(i, seqLen, cache);
        var next = _decoderLayer(
          _lmLayers[i],
          h,
          seqLen,
          positionIds,
          positionEmbeddings: positionEmbeddings,
          layerIndex: i,
          cache: cache.layers[i],
        );
        if (config.enableDecoderLayerwiseEvalForCurrentPlatform) {
          final trace = _beginDecoderLayerEventTrace(
            this,
            i,
            seqLen,
            cache,
            'eval_next',
          );
          MlxRuntime.evalAll([next]);
          _endDecoderLayerEventTrace(this, trace);
          if (seqLen == 1 &&
              config
                  .enableForceGpuSynchronizeAfterLayerEvalForCurrentPlatform) {
            final syncTrace = _beginDecoderLayerEventTrace(
              this,
              i,
              seqLen,
              cache,
              'sync_after_eval',
            );
            _maybeSynchronizeGpuPerToken();
            _endDecoderLayerEventTrace(this, syncTrace);
          }
        }
        if (h != hidden) {
          final trace = _beginDecoderLayerEventTrace(
            this,
            i,
            seqLen,
            cache,
            'close_prev_h',
          );
          h.close();
          _endDecoderLayerEventTrace(this, trace);
        }
        h = next;
        if (config.enableAggressiveCacheClearingForCurrentPlatform) {
          final trace = _beginDecoderLayerEventTrace(
            this,
            i,
            seqLen,
            cache,
            'clear_cache',
          );
          try {
            MlxMemory.clearCache();
          } catch (_) {}
          _endDecoderLayerEventTrace(this, trace);
        }
        _endDecoderLayerTrace(trace, cache, cache.layers[i]);
      }

      final last = _traceDecoderTailArray(
        this,
        seqLen,
        cache,
        'last_slice',
        () => h.slice(
          start: [0, seqLen - 1, 0],
          stop: [1, seqLen, config.hiddenSize],
        ),
      );
      h.close();

      // Final norm on the last token only.
      final norm = _traceDecoderTailArray(
        this,
        seqLen,
        cache,
        'final_norm',
        () => _lmRmsNormCompat(last, weight: _finalNorm, eps: config.rmsNormEps),
      );
      last.close();

      final last2d = _traceDecoderTailArray(
        this,
        seqLen,
        cache,
        'last_reshape',
        () => norm.reshape([1, config.hiddenSize]),
      );
      norm.close();

      final linear = config.tieWordEmbeddings ? _embedWeights : _lmHead!;
      final shouldCastLmHeadInput =
          (seqLen == 1 && config.forceDecodeLmHeadFloat32ForCurrentPlatform) ||
          (!Platform.isIOS && linear is _QuantLinear && last2d.dtype != MlxDType.MLX_FLOAT32);
      final lmHeadInput = shouldCastLmHeadInput
          ? _traceDecoderTailArray(
              this,
              seqLen,
              cache,
              'lm_head_input_f32',
              () => last2d.astype(MlxDType.MLX_FLOAT32),
            )
          : last2d;
      final logits = _traceDecoderTailArray(
        this,
        seqLen,
        cache,
        'lm_head',
        () => linear.apply(lmHeadInput),
      );
      if (!identical(lmHeadInput, last2d)) {
        lmHeadInput.close();
      }
      last2d.close();

      final shaped = _traceDecoderTailArray(
        this,
        seqLen,
        cache,
        'logits_reshape',
        () => logits.reshape([1, config.vocabSize]),
      );
      if (shaped != logits) logits.close();
      if (maybeQuantizeAfter) {
        _traceDecoderTailVoid(this, seqLen, cache, 'maybe_quantize', () {
          cache.maybeQuantize(config: config);
        });
      }
      if (config.enableDecodeCacheStateEvalForCurrentPlatform) {
        final interval = config.decodeCacheStateEvalIntervalForCurrentPlatform;
        if (seqLen > 1 || interval <= 1 || (cache.offset % interval) == 0) {
          _traceDecoderTailVoid(this, seqLen, cache, 'eval_states', () {
            cache.evalStates();
          });
        }
      }
      if (seqLen == 1) {
        _traceDecoderTailVoid(this, seqLen, cache, 'maybe_compact', () {
          cache.maybeCompact(config: config);
        });
      }
      final shouldEvalLogits = seqLen == 1
          ? config.enableDecodeStepExplicitLogitsEvalForCurrentPlatform
          : config.enableExplicitLogitsEvalForCurrentPlatform;
      if (shouldEvalLogits) {
        _traceDecoderTailVoid(this, seqLen, cache, 'eval_logits', () {
          MlxRuntime.evalAll([shaped]);
        });
      }
      if (config.enableAggressiveCacheClearingForCurrentPlatform) {
        _traceDecoderTailVoid(this, seqLen, cache, 'clear_cache', () {
          try {
            MlxMemory.clearCache();
          } catch (_) {}
        });
      }
      if (seqLen == 1) {
        _maybeStopDecoderMetalCapture(cache);
      }
      if (seqLen == 1 && config.enableDecodeLogitsDetachForCurrentPlatform) {
        final detached = MlxArray.fromFloat32List(
          shaped.toFloat32List(),
          shape: shaped.shape,
        );
        // Keep detached decode logits in float32 so later sampling never has
        // to re-materialize a BF16 device array just to read a scalar index.
        shaped.close();
        return detached;
      }
      return shaped;
    } catch (_) {
      if (h != hidden) h.close();
      rethrow;
    } finally {
      positionEmbeddings?.cos.close();
      positionEmbeddings?.sin.close();
    }
  }
}
