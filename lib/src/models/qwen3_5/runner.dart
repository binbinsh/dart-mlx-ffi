part of 'qwen3_5.dart';

final class Qwen3_5Runner {
  Qwen3_5Runner._(
    this.config,
    this.tensors,
    this.textPrefix,
    this._layers,
    this._embedWeights,
    this.finalNorm,
    this._lmHeadWeights,
    this._logitPrefixMatrix,
    this._visionWeights,
  );

  factory Qwen3_5Runner.load(String snapshotPath) {
    final config = Qwen3_5Config.fromSnapshot(snapshotPath);
    final tensors = loadTensorMap(snapshotPath);
    final textPrefix = _detectTextPrefix(tensors.keys);
    final embed = _loadLinear(
      tensors,
      '${textPrefix}embed_tokens',
      config: config,
    );
    final finalNorm = tensors['${textPrefix}norm.weight']!;
    final lmHead = config.tieWordEmbeddings
        ? null
        : _firstLoadLinear(
            tensors,
            _lmHeadPrefixes(textPrefix),
            config: config,
          );
    final layers = List<_LayerWeights>.generate(config.numHiddenLayers, (
      index,
    ) {
      final prefix = '${textPrefix}layers.$index.';
      return _LayerWeights(
        inputNorm: tensors['${prefix}input_layernorm.weight']!,
        postNorm: tensors['${prefix}post_attention_layernorm.weight']!,
        fullAttention: config.isLinearLayer(index)
            ? null
            : _FullAttentionWeights(
                qProj: _loadLinear(
                  tensors,
                  '${prefix}self_attn.q_proj',
                  config: config,
                ),
                kProj: _loadLinear(
                  tensors,
                  '${prefix}self_attn.k_proj',
                  config: config,
                ),
                vProj: _loadLinear(
                  tensors,
                  '${prefix}self_attn.v_proj',
                  config: config,
                ),
                oProj: _loadLinear(
                  tensors,
                  '${prefix}self_attn.o_proj',
                  config: config,
                ),
                qNormWeight: tensors['${prefix}self_attn.q_norm.weight'],
                kNormWeight: tensors['${prefix}self_attn.k_norm.weight'],
              ),
        linearAttention: !config.isLinearLayer(index)
            ? null
            : _LinearAttentionWeights(
                convWeight: _loadLinearConvWeight(
                  tensors['${prefix}linear_attn.conv1d.weight']!,
                  config: config,
                ),
                inProjQkv: _loadLinear(
                  tensors,
                  '${prefix}linear_attn.in_proj_qkv',
                  config: config,
                ),
                inProjZ: _loadLinear(
                  tensors,
                  '${prefix}linear_attn.in_proj_z',
                  config: config,
                ),
                inProjB: _loadLinear(
                  tensors,
                  '${prefix}linear_attn.in_proj_b',
                  config: config,
                ),
                inProjA: _loadLinear(
                  tensors,
                  '${prefix}linear_attn.in_proj_a',
                  config: config,
                ),
                dtBias: tensors['${prefix}linear_attn.dt_bias']!,
                aLog: tensors['${prefix}linear_attn.A_log']!,
                normWeight: tensors['${prefix}linear_attn.norm.weight']!,
                outProj: _loadLinear(
                  tensors,
                  '${prefix}linear_attn.out_proj',
                  config: config,
                ),
              ),
        denseMlp: _DenseMlpWeights(
          gateProj: _loadLinear(
            tensors,
            '${prefix}mlp.gate_proj',
            config: config,
          ),
          upProj: _loadLinear(tensors, '${prefix}mlp.up_proj', config: config),
          downProj: _loadLinear(
            tensors,
            '${prefix}mlp.down_proj',
            config: config,
          ),
        ),
      );
    });
    final logitPrefixMatrix = _buildLogitPrefixMatrix(
      config.tieWordEmbeddings ? embed : lmHead!,
      config: config,
    );

    // Load vision weights if present (VLM mode).
    final visionWeights = config.hasVision
        ? Qwen35Vision.loadVisionWeights(tensors)
        : null;

    final runner = Qwen3_5Runner._(
      config,
      tensors,
      textPrefix,
      layers,
      embed,
      finalNorm,
      lmHead,
      logitPrefixMatrix,
      visionWeights,
    );
    _warmQwen35CompiledHelpers(config);
    _warmQwen35GDeltaStep(
      numValueHeads: config.linearNumValueHeads,
      keyHeadDim: config.linearKeyHeadDim,
      valueHeadDim: config.linearValueHeadDim,
      dtype: config.computeDType,
    );
    _warmQwen35GDeltaKernel(
      numValueHeads: config.linearNumValueHeads,
      numKeyHeads: config.linearNumKeyHeads,
      keyHeadDim: config.linearKeyHeadDim,
      valueHeadDim: config.linearValueHeadDim,
      dtype: config.computeDType,
    );
    runner._warmForwardPath();
    return runner;
  }

  final Qwen3_5Config config;
  final Map<String, MlxArray> tensors;
  final String textPrefix;
  final List<_LayerWeights> _layers;
  final _LinearBase _embedWeights;
  final MlxArray finalNorm;
  final _LinearBase? _lmHeadWeights;
  final MlxArray _logitPrefixMatrix;
  final _Qwen35VisionWeights? _visionWeights;
  MlxArray? _ropeInvFreq;
  final Map<String, ({MlxArray cos, MlxArray sin})> _ropeCache = {};
  final Map<String, MlxArray> _linearScaleCache = {};
  final Map<String, MlxArray> _linearConvPrefixCache = {};

  MlxArray run(List<int> tokenIds) {
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, tokenIds.length]);
    try {
      final hidden = _embed(ids);
      return _runHidden(hidden, tokenIds.length);
    } finally {
      ids.close();
    }
  }

  MlxArray runFullLogits(List<int> tokenIds) {
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, tokenIds.length]);
    try {
      final hidden = _embed(ids);
      return _runHidden(hidden, tokenIds.length, fullLogits: true);
    } finally {
      ids.close();
    }
  }

  Map<String, Object?> debugTraceLastToken(List<int> tokenIds) {
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, tokenIds.length]);
    try {
      var hidden = _embed(ids);
      final layers = <Map<String, Object?>>[];
      try {
        for (var index = 0; index < _layers.length; index++) {
          final layer = _layers[index];
          final norm1 = mx.fast.rmsNorm(
            hidden,
            weight: layer.inputNorm,
            eps: config.rmsNormEps,
          );
          final attn = layer.fullAttention != null
              ? _fullAttention(layer.fullAttention!, norm1, tokenIds.length)
              : _linearAttention(
                  layer.linearAttention!,
                  norm1,
                  tokenIds.length,
                );
          final h = mx.add(hidden, attn);
          attn.close();
          norm1.close();
          hidden.close();

          final norm2 = mx.fast.rmsNorm(
            h,
            weight: layer.postNorm,
            eps: config.rmsNormEps,
          );
          final mlp = _denseMlp(layer.denseMlp, norm2, tokenIds.length);
          norm2.close();
          final next = mx.add(h, mlp);
          mlp.close();
          h.close();
          hidden = next;

          layers.add({
            'index': index,
            'kind': layer.fullAttention != null ? 'full' : 'linear',
            'last_token': _lastTokenFloat32(hidden, tokenIds.length),
          });
        }
        final norm = mx.fast.rmsNorm(
          hidden,
          weight: finalNorm,
          eps: config.rmsNormEps,
        );
        try {
          return {
            'layers': layers,
            'final_norm_last_token': _lastTokenFloat32(norm, tokenIds.length),
          };
        } finally {
          norm.close();
        }
      } finally {
        hidden.close();
      }
    } finally {
      ids.close();
    }
  }

  Map<String, Object?> debugLinearCacheSummary(
    List<int> promptIds,
    int maxNewTokens, {
    int? eosTokenId,
    int? layerIndex,
  }) {
    final cache = _makeDecodeCache(this);
    try {
      final tokens = List<int>.from(promptIds);
      final prompt = MlxArray.fromInt32List(
        promptIds,
        shape: [1, promptIds.length],
      );
      var logits = _runWithCache(prompt, cache, fullLogits: true);
      prompt.close();
      try {
        for (var index = 0; index < maxNewTokens; index++) {
          final next = _nextTokenFromLogits(logits);
          tokens.add(next);
          if (eosTokenId != null && next == eosTokenId) {
            break;
          }
          if (index + 1 >= maxNewTokens) {
            break;
          }
          logits.close();
          final step = MlxArray.fromInt32List([next], shape: [1, 1]);
          logits = _runWithCache(step, cache, fullLogits: true);
          step.close();
        }
      } finally {
        logits.close();
      }

      final layers = <Map<String, Object?>>[];
      for (var index = 0; index < cache.layers.length; index++) {
        if (layerIndex != null && layerIndex != index) {
          continue;
        }
        final layerCache = cache.layers[index];
        if (layerCache is! _LinearDecodeCache) {
          continue;
        }
        final conv = layerCache._convState;
        final state = layerCache._state;
        layers.add({
          'index': index,
          'conv_state': conv == null ? null : _summarizeArray(conv),
          'state': state == null ? null : _summarizeArray(state),
        });
      }
      return {'generated_token_ids': tokens, 'linear_layers': layers};
    } finally {
      cache.close();
    }
  }

  Map<String, Object?> debugDecodeCacheSummary(
    List<int> promptIds,
    int maxNewTokens, {
    int? eosTokenId,
    int? layerIndex,
  }) {
    final cache = _makeDecodeCache(this);
    try {
      final tokens = List<int>.from(promptIds);
      final prompt = MlxArray.fromInt32List(
        promptIds,
        shape: [1, promptIds.length],
      );
      var logits = _runWithCache(prompt, cache, fullLogits: true);
      prompt.close();
      try {
        for (var index = 0; index < maxNewTokens; index++) {
          final next = _nextTokenFromLogits(logits);
          tokens.add(next);
          if (eosTokenId != null && next == eosTokenId) {
            break;
          }
          if (index + 1 >= maxNewTokens) {
            break;
          }
          logits.close();
          final step = MlxArray.fromInt32List([next], shape: [1, 1]);
          logits = _runWithCache(step, cache, fullLogits: true);
          step.close();
        }
      } finally {
        logits.close();
      }

      final layers = <Map<String, Object?>>[];
      for (var index = 0; index < cache.layers.length; index++) {
        if (layerIndex != null && layerIndex != index) {
          continue;
        }
        final layerCache = cache.layers[index];
        if (layerCache case final _LinearDecodeCache linear) {
          layers.add({
            'index': index,
            'kind': 'linear',
            'conv_state': linear._convState == null
                ? null
                : _summarizeArray(linear._convState!),
            'state': linear._state == null
                ? null
                : _summarizeArray(linear._state!),
          });
        } else if (layerCache case final _KvDecodeCache kv) {
          layers.add({
            'index': index,
            'kind': 'full',
            'offset': kv.offset,
            'keys': kv.keys == null ? null : _summarizeArray(kv.keys!),
            'values': kv.values == null ? null : _summarizeArray(kv.values!),
          });
        }
      }
      return {'generated_token_ids': tokens, 'layers': layers};
    } finally {
      cache.close();
    }
  }

  Map<String, Object?> debugCachedTopLogits(
    List<int> promptIds,
    int steps, {
    int topK = 8,
    int? eosTokenId,
  }) {
    final cache = _makeDecodeCache(this);
    try {
      final prompt = MlxArray.fromInt32List(
        promptIds,
        shape: [1, promptIds.length],
      );
      var logits = _runWithCache(
        prompt,
        cache,
        fullLogits: true,
      ).astype(MlxDType.MLX_FLOAT32);
      prompt.close();
      try {
        var next = _nextTokenFromLogits(logits);
        for (var index = 0; index < steps; index++) {
          if (eosTokenId != null && next == eosTokenId) {
            break;
          }
          logits.close();
          final step = MlxArray.fromInt32List([next], shape: [1, 1]);
          logits = _runWithCache(
            step,
            cache,
            fullLogits: true,
          ).astype(MlxDType.MLX_FLOAT32);
          step.close();
          next = _nextTokenFromLogits(logits);
        }
        final values = logits.toList().cast<double>();
        final pairs = <Map<String, Object?>>[
          for (var i = 0; i < values.length; i++) {'idx': i, 'v': values[i]},
        ]..sort((a, b) => (b['v'] as double).compareTo(a['v'] as double));
        return {
          'next_token_id': _nextTokenFromLogits(logits),
          'top': pairs.take(topK).toList(growable: false),
        };
      } finally {
        logits.close();
      }
    } finally {
      cache.close();
    }
  }

  Map<String, Object?> debugSingleTokenLinearProbe(
    int tokenId,
    int layerIndex,
  ) {
    final ids = MlxArray.fromInt32List([tokenId], shape: [1, 1]);
    try {
      final embed = _embed(ids);
      try {
        final layer = _layers[layerIndex];
        if (layer.linearAttention == null) {
          throw StateError(
            'Layer $layerIndex is not a linear-attention layer.',
          );
        }
        final norm1 = mx.fast.rmsNorm(
          embed,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );
        try {
          final linearInput =
              Platform.environment['QWEN35_USE_HIGHRANK_LINEAR'] == '1'
              ? norm1
              : norm1.reshape([1, config.hiddenSize]);
          final mixedQkv = layer.linearAttention!.inProjQkv
              .apply(linearInput, config: config)
              .reshape([1, layer.linearAttention!.convWeight.shape[0]]);
          try {
            return {
              'token_id': tokenId,
              'layer_index': layerIndex,
              'embed': _summarizeArray(embed),
              'norm1': _summarizeArray(norm1),
              'mixed_qkv': _summarizeArray(mixedQkv),
            };
          } finally {
            mixedQkv.close();
            if (linearInput != norm1) {
              linearInput.close();
            }
          }
        } finally {
          norm1.close();
        }
      } finally {
        embed.close();
      }
    } finally {
      ids.close();
    }
  }

  int nextTokenId(List<int> tokenIds) {
    final logits = runFullLogits(tokenIds);
    try {
      return _nextTokenFromLogits(logits);
    } finally {
      logits.close();
    }
  }

  List<int> generateGreedy(
    List<int> promptIds,
    int maxNewTokens, {
    int? eosTokenId,
  }) {
    if (maxNewTokens < 0) {
      throw ArgumentError.value(
        maxNewTokens,
        'maxNewTokens',
        'Must be non-negative.',
      );
    }
    if (maxNewTokens == 0) {
      return List<int>.from(promptIds);
    }
    return _generateGreedyCached(
      this,
      promptIds,
      maxNewTokens,
      eosTokenId: eosTokenId,
    );
  }

  void _warmForwardPath() {
    if (Platform.environment['QWEN35_WARM_GENERATE'] == '0') {
      return;
    }
    final warmDecodeTokens =
        int.tryParse(Platform.environment['QWEN35_WARM_DECODE_TOKENS'] ?? '') ??
        4;
    final promptIds = List<int>.generate(24, (index) => index + 1);
    final full = runFullLogits(promptIds);
    final step = runFullLogits([promptIds.first]);
    full.close();
    step.close();
    final cache = _makeDecodeCache(this);
    try {
      final prompt = MlxArray.fromInt32List(
        promptIds,
        shape: [1, promptIds.length],
      );
      final promptOut = _runWithCache(prompt, cache, fullLogits: true);
      prompt.close();
      promptOut.close();
      final stepIds = MlxArray.fromInt32List([promptIds.first], shape: [1, 1]);
      final stepOut = _runWithCache(stepIds, cache, fullLogits: true);
      stepIds.close();
      stepOut.close();
    } finally {
      cache.close();
    }
    if (warmDecodeTokens > 0) {
      _warmCachedDecodeSteps(this, promptIds, warmDecodeTokens);
    }
  }

  MlxArray buildGraph(MlxArray ids) {
    final hidden = _embed(ids);
    return _runHidden(hidden, ids.shape[1]);
  }

  MlxArray _embed(MlxArray ids) {
    final rows = ids.reshape([ids.shape[1]]);
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
            dtype: config.computeDType,
          );
          MlxRuntime.evalAll([out]);
          return out.reshape([1, ids.shape[1], config.hiddenSize]);
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      if (_embedWeights case final _DenseLinear d) {
        final out = d.weight.take(rows, axis: 0);
        return out.reshape([1, ids.shape[1], config.hiddenSize]);
      }
      throw StateError('Unsupported embedding type.');
    } finally {
      rows.close();
    }
  }

  MlxArray _runWithCache(
    MlxArray ids,
    _ModelDecodeCache cache, {
    required bool fullLogits,
  }) {
    final hidden = _embed(ids);
    return _runHidden(
      hidden,
      ids.shape[1],
      fullLogits: fullLogits,
      cache: cache,
    );
  }

  MlxArray _runHidden(
    MlxArray hidden,
    int seqLen, {
    bool fullLogits = false,
    _ModelDecodeCache? cache,
  }) {
    try {
      final norm = _runHiddenNorm(hidden, seqLen, cache: cache);
      final output = fullLogits
          ? _lmHeadFull(norm, seqLen)
          : _lmHead(norm, seqLen);
      norm.close();
      return output;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray _runHiddenNorm(
    MlxArray hidden,
    int seqLen, {
    _ModelDecodeCache? cache,
  }) {
    try {
      for (var index = 0; index < _layers.length; index++) {
        final layer = _layers[index];
        final norm1 = mx.fast.rmsNorm(
          hidden,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );
        final attn = layer.fullAttention != null
            ? _fullAttention(
                layer.fullAttention!,
                norm1,
                seqLen,
                cache: cache?.layers[index] as _KvDecodeCache?,
              )
            : _linearAttention(
                layer.linearAttention!,
                norm1,
                seqLen,
                cache: cache?.layers[index] as _LinearDecodeCache?,
              );
        final h = mx.add(hidden, attn);
        attn.close();
        norm1.close();
        hidden.close();

        final norm2 = mx.fast.rmsNorm(
          h,
          weight: layer.postNorm,
          eps: config.rmsNormEps,
        );
        final mlp = _denseMlp(layer.denseMlp, norm2, seqLen);
        norm2.close();
        final next = mx.add(h, mlp);
        mlp.close();
        h.close();
        hidden = next;
      }

      final norm = mx.fast.rmsNorm(
        hidden,
        weight: finalNorm,
        eps: config.rmsNormEps,
      );
      hidden.close();
      return norm;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray _lmHead(MlxArray hidden, int seqLen) {
    final last = hidden.slice(
      start: [0, seqLen - 1, 0],
      stop: [1, seqLen, config.hiddenSize],
    );
    try {
      final out = _logitPrefix(last.reshape([1, config.hiddenSize]));
      MlxRuntime.evalAll([out]);
      return out;
    } finally {
      last.close();
    }
  }

  MlxArray _lmHeadFull(MlxArray hidden, int seqLen) {
    final last = hidden.slice(
      start: [0, seqLen - 1, 0],
      stop: [1, seqLen, config.hiddenSize],
    );
    try {
      final linear = config.tieWordEmbeddings ? _embedWeights : _lmHeadWeights!;
      final useHighRank = _useHighRankLinear(seqLen);
      final useF32 = Platform.environment['QWEN35_LMHEAD_F32'] == '1';
      final linearInput = useHighRank
          ? last
          : last.reshape([1, config.hiddenSize]);
      final matmulInput = useF32
          ? linearInput.astype(MlxDType.MLX_FLOAT32)
          : linearInput;
      final out = linear.apply(matmulInput, config: config);
      if (matmulInput != linearInput) {
        matmulInput.close();
      }
      if (!useHighRank) {
        linearInput.close();
      }
      MlxRuntime.evalAll([out]);
      return out.reshape([1, config.vocabSize]);
    } finally {
      last.close();
    }
  }

  MlxArray _logitPrefix(MlxArray last2d) =>
      mx.matmul(last2d, _logitPrefixMatrix.transpose()).reshape([1, 16]);

  bool _useHighRankLinear(int seqLen) {
    final flag = Platform.environment['QWEN35_USE_HIGHRANK_LINEAR'];
    if (flag == '1') {
      return true;
    }
    if (flag == '0') {
      return false;
    }
    return seqLen > 1;
  }

  List<double> _lastTokenFloat32(MlxArray hidden, int seqLen) {
    final last = hidden
        .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, config.hiddenSize])
        .reshape([config.hiddenSize])
        .astype(MlxDType.MLX_FLOAT32);
    try {
      return last.toList().cast<double>();
    } finally {
      last.close();
    }
  }

  Map<String, Object?> _summarizeArray(MlxArray value) {
    final f32 = value.astype(MlxDType.MLX_FLOAT32);
    try {
      final flat = f32.reshape([f32.size]);
      try {
        final values = flat.toList().cast<double>();
        var sum = 0.0;
        var sumSquares = 0.0;
        var maxAbs = 0.0;
        for (final v in values) {
          final absV = v.abs();
          sum += v;
          sumSquares += v * v;
          if (absV > maxAbs) {
            maxAbs = absV;
          }
        }
        return {
          'shape': value.shape,
          'head16': values.take(16).toList(),
          'mean': values.isEmpty ? 0.0 : sum / values.length,
          'l2': sumSquares == 0.0 ? 0.0 : math.sqrt(sumSquares),
          'max_abs': maxAbs,
        };
      } finally {
        flat.close();
      }
    } finally {
      f32.close();
    }
  }

  Map<String, Object?> debugCachedExactRerank(
    List<int> promptIds,
    int steps,
    List<int> candidateIds, {
    int? eosTokenId,
  }) {
    final cache = _makeDecodeCache(this);
    try {
      final prompt = MlxArray.fromInt32List(
        promptIds,
        shape: [1, promptIds.length],
      );
      var norm = _runWithCacheNorm(prompt, cache);
      prompt.close();
      try {
        var logits = _lmHeadFull(
          norm,
          norm.shape[1],
        ).astype(MlxDType.MLX_FLOAT32);
        try {
          var next = _nextTokenFromLogits(logits);
          for (var index = 0; index < steps; index++) {
            if (eosTokenId != null && next == eosTokenId) {
              break;
            }
            logits.close();
            norm.close();
            final step = MlxArray.fromInt32List([next], shape: [1, 1]);
            norm = _runWithCacheNorm(step, cache);
            step.close();
            logits = _lmHeadFull(
              norm,
              norm.shape[1],
            ).astype(MlxDType.MLX_FLOAT32);
            next = _nextTokenFromLogits(logits);
          }
          final exact = _exactTokenScores(norm, candidateIds);
          try {
            final exactValues = exact.toList().cast<double>();
            return {
              'approx': [
                for (final id in candidateIds)
                  {'idx': id, 'v': _scalarAt(logits, id)},
              ],
              'exact': [
                for (var i = 0; i < candidateIds.length; i++)
                  {'idx': candidateIds[i], 'v': exactValues[i]},
              ],
            };
          } finally {
            exact.close();
          }
        } finally {
          logits.close();
        }
      } finally {
        norm.close();
      }
    } finally {
      cache.close();
    }
  }

  MlxArray _runWithCacheNorm(MlxArray ids, _ModelDecodeCache cache) {
    final hidden = _embed(ids);
    return _runHiddenNorm(hidden, ids.shape[1], cache: cache);
  }

  MlxArray _exactTokenScores(MlxArray norm, List<int> tokenIds) {
    final last2d = norm
        .slice(start: [0, 0, 0], stop: [1, 1, config.hiddenSize])
        .reshape([1, config.hiddenSize])
        .astype(MlxDType.MLX_FLOAT32);
    final indices = MlxArray.fromInt32List(tokenIds, shape: [tokenIds.length]);
    try {
      final linear = config.tieWordEmbeddings ? _embedWeights : _lmHeadWeights!;
      if (linear case final _QuantLinear q) {
        final rowsW = q.weight.take(indices, axis: 0);
        final rowsS = q.scales.take(indices, axis: 0);
        final rowsB = q.biases?.take(indices, axis: 0);
        try {
          final rows = mx.quant.dequantize(
            MlxQuantizedMatrix(rowsW, rowsS, rowsB),
            groupSize: q.quantSpec.groupSize,
            bits: q.quantSpec.bits,
            mode: q.quantSpec.mode,
            dtype: MlxDType.MLX_FLOAT32,
          );
          try {
            return mx.matmul(last2d, rows.transpose()).reshape([
              tokenIds.length,
            ]);
          } finally {
            rows.close();
          }
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      if (linear case final _DenseLinear d) {
        final rows = d.weight
            .take(indices, axis: 0)
            .astype(MlxDType.MLX_FLOAT32);
        try {
          return mx.matmul(last2d, rows.transpose()).reshape([tokenIds.length]);
        } finally {
          rows.close();
        }
      }
      throw StateError('Unsupported lm_head type.');
    } finally {
      indices.close();
      last2d.close();
    }
  }

  double _scalarAt(MlxArray logits, int index) {
    final values = logits.toList().cast<double>();
    return values[index];
  }

  /// Whether this runner has vision encoder weights loaded.
  bool get hasVision => _visionWeights != null;

  /// Generate token IDs from a vision-language prompt.
  ///
  /// [promptIds] are token IDs with `imageTokenId` placeholders for where
  /// the image tokens should be inserted.
  /// [patchedPixels] is a pre-processed patch tensor
  ///   `[N_patches, C * temporal_ps * ps * ps]` in merge-grouped order.
  /// [gridH] and [gridW] are the spatial patch grid dimensions.
  ///
  /// Returns the generated token IDs (prompt + generated).
  List<int> generateFromImage(
    List<int> promptIds,
    MlxArray patchedPixels,
    int gridH,
    int gridW, {
    int maxNewTokens = 512,
    int? eosTokenId,
    void Function(String message)? onStage,
    void Function(int tokenId)? onToken,
    void Function(String stage, MlxArray value)? onDumpIntermediate,
  }) {
    final visionWeights = _visionWeights;
    if (visionWeights == null) {
      throw StateError('No vision weights loaded. Cannot generate from image.');
    }

    // 1. Encode image through ViT + projector
    onStage?.call('generateFromImage: encoding image...');
    final imageEncoding = encodeImage(
      patchedPixels,
      gridH,
      gridW,
      visionWeights,
      onStage: onStage,
      onDumpIntermediate: onDumpIntermediate,
    );
    final imageHidden = imageEncoding.hidden;
    onStage?.call(
      'generateFromImage: image encoded, '
      'grid=${gridH}x${gridW}, tokens=${imageHidden.shape[0]}',
    );

    // 2. Expand image token placeholders to match actual token count
    final numImageTokens = imageHidden.shape[0];
    final imageTokenId = config.imageTokenId!;
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    onStage?.call(
      'generateFromImage: expanded prompt ${promptIds.length} → '
      '${expandedIds.length} tokens',
    );

    // 3. Build multimodal embedding (replace image placeholders with vision)
    final embeddings = buildMultimodalEmbedding(expandedIds, imageHidden);
    onStage?.call(
      'generateFromImage: multimodal embeddings shape=${embeddings.shape}',
    );
    imageHidden.close();

    // 4. Build multimodal M-RoPE position IDs
    final positionInfo = multimodalPositionIds(expandedIds, gridH, gridW);
    final posIds = positionInfo.ids;
    onStage?.call('generateFromImage: position IDs ready');

    // 5. Run LM forward pass from embeddings with multimodal positions
    final cache = _makeDecodeCache(this);
    try {
      var logits = _runFromEmbeddingWithPositionIds(
        embeddings,
        posIds,
        expandedIds.length,
        cache,
      );
      embeddings.close();
      posIds.close();
      onStage?.call('generateFromImage: prompt forward done');

      // 6. Greedy decode
      final tokens = List<int>.from(expandedIds);
      try {
        var nextTextPosition = positionInfo.nextTextPosition;
        for (var step = 0; step < maxNewTokens; step++) {
          final next = _nextTokenFromLogits(logits);
          tokens.add(next);
          onToken?.call(next);
          if (step == 0) {
            onStage?.call('generateFromImage: first token=$next');
          }
          if (eosTokenId != null && next == eosTokenId) break;
          if (step + 1 >= maxNewTokens) break;

          logits.close();
          final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
          logits = _runWithCacheAtOffset(
            stepArr,
            cache,
            offset: nextTextPosition,
          );
          stepArr.close();
          nextTextPosition++;
          if ((step + 1) % 16 == 0) {
            onStage?.call('generateFromImage: generated ${step + 1} tokens');
          }
        }
      } finally {
        logits.close();
      }
      return tokens;
    } finally {
      cache.close();
    }
  }

  /// Forward pass from pre-built embeddings with explicit M-RoPE position IDs.
  MlxArray _runFromEmbeddingWithPositionIds(
    MlxArray embeddings,
    MlxArray positionIds,
    int seqLen,
    _ModelDecodeCache cache,
  ) {
    var hidden = embeddings;
    try {
      for (var index = 0; index < _layers.length; index++) {
        final layer = _layers[index];
        final norm1 = mx.fast.rmsNorm(
          hidden,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );

        final MlxArray attn;
        if (layer.fullAttention != null) {
          attn = _fullAttentionWithPositionIds(
            layer.fullAttention!,
            norm1,
            seqLen,
            positionIds,
            cache: cache.layers[index] as _KvDecodeCache,
          );
        } else {
          attn = _linearAttention(
            layer.linearAttention!,
            norm1,
            seqLen,
            cache: cache.layers[index] as _LinearDecodeCache,
          );
        }

        final h = mx.add(hidden, attn);
        attn.close();
        norm1.close();
        hidden.close();

        final norm2 = mx.fast.rmsNorm(
          h,
          weight: layer.postNorm,
          eps: config.rmsNormEps,
        );
        final mlp = _denseMlp(layer.denseMlp, norm2, seqLen);
        norm2.close();
        final next = mx.add(h, mlp);
        mlp.close();
        h.close();
        hidden = next;
      }

      final norm = mx.fast.rmsNorm(
        hidden,
        weight: finalNorm,
        eps: config.rmsNormEps,
      );
      hidden.close();
      final output = _lmHeadFull(norm, seqLen);
      norm.close();
      return output;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  /// Full attention with explicit M-RoPE position IDs (for multimodal prompt).
  MlxArray _fullAttentionWithPositionIds(
    _FullAttentionWeights layer,
    MlxArray input,
    int seqLen,
    MlxArray positionIds, {
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

    // Use explicit multimodal position IDs for M-RoPE
    final rope = applyMropeWithPositionIds(qNorm, kNorm, positionIds, seqLen);
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

    final repeatKv = config.numAttentionHeads ~/ config.numKeyValueHeads;
    if (repeatKv > 1) {
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
    if ((repeatKv > 1) || cache == null) {
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

  /// Run a single decode step using the cache with a specific M-RoPE offset.
  ///
  /// For VLM decode: the M-RoPE position must be [offset] (which may differ
  /// from the KV-cache length due to spatial image positions).
  MlxArray _runWithCacheAtOffset(
    MlxArray ids,
    _ModelDecodeCache cache, {
    required int offset,
  }) {
    final hidden = _embed(ids);
    final seqLen = ids.shape[1];
    // Build position IDs for this decode step: all 3 dims use [offset].
    final flat = <int>[offset, offset, offset];
    final posIds = MlxArray.fromInt32List(flat, shape: [3, 1, seqLen]);
    try {
      var h = hidden;
      for (var index = 0; index < _layers.length; index++) {
        final layer = _layers[index];
        final norm1 = mx.fast.rmsNorm(
          h,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );

        final MlxArray attn;
        if (layer.fullAttention != null) {
          attn = _fullAttentionWithPositionIds(
            layer.fullAttention!,
            norm1,
            seqLen,
            posIds,
            cache: cache.layers[index] as _KvDecodeCache,
          );
        } else {
          attn = _linearAttention(
            layer.linearAttention!,
            norm1,
            seqLen,
            cache: cache.layers[index] as _LinearDecodeCache,
          );
        }

        final nextH = mx.add(h, attn);
        attn.close();
        norm1.close();
        h.close();

        final norm2 = mx.fast.rmsNorm(
          nextH,
          weight: layer.postNorm,
          eps: config.rmsNormEps,
        );
        final mlp = _denseMlp(layer.denseMlp, norm2, seqLen);
        norm2.close();
        final next = mx.add(nextH, mlp);
        mlp.close();
        nextH.close();
        h = next;
      }

      final norm = mx.fast.rmsNorm(
        h,
        weight: finalNorm,
        eps: config.rmsNormEps,
      );
      h.close();
      final output = _lmHeadFull(norm, seqLen);
      norm.close();
      return output;
    } catch (_) {
      hidden.close();
      rethrow;
    } finally {
      posIds.close();
    }
  }

  /// Expand a single image_token_id placeholder to the actual number of
  /// image tokens needed.
  List<int> _expandImageTokens(List<int> promptIds, int numImageTokens) {
    final imageTokenId = config.imageTokenId!;
    final result = <int>[];
    var expanded = false;
    for (final id in promptIds) {
      if (id == imageTokenId && !expanded) {
        for (var i = 0; i < numImageTokens; i++) {
          result.add(imageTokenId);
        }
        expanded = true;
      } else if (id == imageTokenId && expanded) {
        // Skip additional image token placeholders
        continue;
      } else {
        result.add(id);
      }
    }
    return result;
  }

  void close() {
    for (final entry in _ropeCache.values) {
      entry.cos.close();
      entry.sin.close();
    }
    _ropeCache.clear();
    for (final value in _linearScaleCache.values) {
      value.close();
    }
    _linearScaleCache.clear();
    for (final value in _linearConvPrefixCache.values) {
      value.close();
    }
    _linearConvPrefixCache.clear();
    _ropeInvFreq?.close();
    _ropeInvFreq = null;
    _logitPrefixMatrix.close();
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }

  static _LinearBase _loadLinear(
    Map<String, MlxArray> tensors,
    String prefix, {
    required Qwen3_5Config config,
  }) => _LinearBase.load(
    tensors,
    prefix,
    defaultQuantSpec: config.defaultQuantSpec(),
    quantSpec: config.quantizationByPath[prefix],
  );

  static _LinearBase? _maybeLoadLinear(
    Map<String, MlxArray> tensors,
    String prefix, {
    required Qwen3_5Config config,
  }) {
    if (!tensors.containsKey('$prefix.weight')) {
      return null;
    }
    return _loadLinear(tensors, prefix, config: config);
  }

  static _LinearBase? _firstLoadLinear(
    Map<String, MlxArray> tensors,
    Iterable<String> prefixes, {
    required Qwen3_5Config config,
  }) {
    for (final prefix in prefixes) {
      final linear = _maybeLoadLinear(tensors, prefix, config: config);
      if (linear != null) {
        return linear;
      }
    }
    return null;
  }

  static Iterable<String> _lmHeadPrefixes(String textPrefix) sync* {
    yield 'lm_head';
    yield '${textPrefix}lm_head';
    if (textPrefix.endsWith('model.')) {
      yield '${textPrefix.substring(0, textPrefix.length - 'model.'.length)}lm_head';
    }
  }

  static MlxArray _buildLogitPrefixMatrix(
    _LinearBase linear, {
    required Qwen3_5Config config,
  }) {
    final indices = MlxArray.fromInt32List(
      List<int>.generate(16, (index) => index),
      shape: [16],
    );
    try {
      if (linear is _QuantLinear) {
        final rowsW = linear.weight.take(indices, axis: 0);
        final rowsS = linear.scales.take(indices, axis: 0);
        final rowsB = linear.biases?.take(indices, axis: 0);
        try {
          return mx.quant.dequantize(
            MlxQuantizedMatrix(rowsW, rowsS, rowsB),
            groupSize: linear.quantSpec.groupSize,
            bits: linear.quantSpec.bits,
            mode: linear.quantSpec.mode,
            dtype: config.computeDType,
          );
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      if (linear is _DenseLinear) {
        return linear.weight.take(indices, axis: 0);
      }
      throw StateError('Unsupported lm_head type.');
    } finally {
      indices.close();
    }
  }

  static String _detectTextPrefix(Iterable<String> keys) {
    final candidates = [
      for (final key in keys)
        if (key.endsWith('layers.0.input_layernorm.weight'))
          key.substring(
            0,
            key.length - 'layers.0.input_layernorm.weight'.length,
          ),
    ]..sort((a, b) => a.length.compareTo(b.length));
    if (candidates.isEmpty) {
      throw StateError('Unable to detect Qwen3.5 text tensor prefix.');
    }
    return candidates.first;
  }

  static MlxArray _loadLinearConvWeight(
    MlxArray weight, {
    required Qwen3_5Config config,
  }) {
    if (weight.shape.length == 3 &&
        weight.shape[1] == 1 &&
        weight.shape[2] == config.linearConvKernelDim) {
      return weight.transposeAxes([0, 2, 1]);
    }
    return weight;
  }
}
