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

    return PaddleOcrVlRunner._(
      config,
      tensors,
      visionWeights,
      layers,
      embedWeights,
      finalNorm,
      lmHead,
    );
  }

  final PaddleOcrVlConfig config;
  final Map<String, MlxArray> _tensors;
  final _VisionWeights _visionWeights;
  final List<_LmLayerWeights> _lmLayers;
  final _LinearBase _embedWeights;
  final MlxArray _finalNorm;
  final _LinearBase? _lmHead;
  MlxArray? _ropeInvFreq;

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
  List<int> generateFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    int maxNewTokens = 512,
  }) {
    // 1. Encode the image through ViT + projector
    final imageHidden = _encodeImage(imagePixels);
    // imageHidden: [mergedTokens, lmHidden]
    final numImageTokens = imageHidden.shape[0];

    // 2. Compute grid dimensions for position IDs
    final gridSide = config._vision.imageSize ~/ config._vision.patchSize;

    // 3. Replace image_token_id placeholders with actual image hidden states
    //    in the embedding
    final expandedIds = _expandImageTokens(promptIds, numImageTokens);
    final posIds = _multimodalPositionIds(expandedIds, gridSide, gridSide);
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();

    // 4. Run LM forward pass on embeddings (not token IDs)
    final cache = _ModelCache.create(config.numHiddenLayers);
    try {
      var logits = _forwardFromEmbedding(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();

      final tokens = List<int>.from(expandedIds);
      try {
        for (var step = 0; step < maxNewTokens; step++) {
          final next = _nextTokenFromLogits(logits);
          tokens.add(next);
          if (next == config.eosTokenId) break;
          if (step + 1 >= maxNewTokens) break;

          logits.close();
          final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
          final stepPos = _textPositionIds(1, offset: cache.offset);
          logits = _forwardWithCache(stepArr, stepPos, cache);
          stepArr.close();
          stepPos.close();
        }
      } finally {
        logits.close();
      }
      return tokens;
    } finally {
      cache.close();
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

    // Patch embedding (Conv2d)
    final patchWeight = tensors['${vp}patch_embedding.weight']!;
    final patchBias = tensors['${vp}patch_embedding.bias'];

    // Position embedding
    final posEmbed = _LinearBase.load(
      tensors,
      '${vp}position_embedding',
      defaultQuant: defaultQuant,
    );

    // ViT blocks
    final blocks = List<_VisionBlockWeights>.generate(
      config._vision.numHiddenLayers,
      (i) {
        final bp = '${vp}blocks.$i.';
        return _VisionBlockWeights(
          layerNorm1Weight: tensors['${bp}layer_norm1.weight']!,
          layerNorm1Bias: tensors['${bp}layer_norm1.bias']!,
          layerNorm2Weight: tensors['${bp}layer_norm2.weight']!,
          layerNorm2Bias: tensors['${bp}layer_norm2.bias']!,
          qkv: _LinearBase.load(
            tensors,
            '${bp}attn.qkv',
            defaultQuant: defaultQuant,
          ),
          outProj: _LinearBase.load(
            tensors,
            '${bp}attn.out_proj',
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
      preNormWeight: tensors['${vp}merger.pre_norm.weight']!,
      preNormBias: tensors['${vp}merger.pre_norm.bias']!,
      linear1: _LinearBase.load(
        tensors,
        '${vp}merger.linear_1',
        defaultQuant: defaultQuant,
      ),
      linear2: _LinearBase.load(
        tensors,
        '${vp}merger.linear_2',
        defaultQuant: defaultQuant,
      ),
    );

    return _VisionWeights(
      patchEmbedWeight: patchWeight,
      patchEmbedBias: patchBias,
      positionEmbedding: posEmbed,
      blocks: blocks,
      projector: projector,
    );
  }
}
