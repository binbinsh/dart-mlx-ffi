part of 'paddle_ocr_vl.dart';

extension PaddleOcrVlRunnerDebug on PaddleOcrVlRunner {
  ({MlxArray weight, MlxArray scales, MlxArray? biases}) _debugLmHeadPrefixQuant(
    _QuantLinear quant,
    int width,
  ) {
    final cached = _debugLmHeadPrefixQuantCache[width];
    if (cached != null) return cached;
    final slicedWeight = quant.weight.slice(
      start: [0, 0],
      stop: [width, quant.weight.shape[1]],
    );
    final slicedScales = quant.scales.slice(
      start: [0, 0],
      stop: [width, quant.scales.shape[1]],
    );
    final slicedBiases = quant.biases?.slice(
      start: [0, 0],
      stop: [width, quant.biases!.shape[1]],
    );
    final packed = (
      weight: slicedWeight,
      scales: slicedScales,
      biases: slicedBiases,
    );
    _debugLmHeadPrefixQuantCache[width] = packed;
    return packed;
  }

  MlxArray _debugPrefillLastHiddenFromEmbeddings(
    MlxArray embeddings,
    MlxArray posIds,
  ) {
    final seqLen = embeddings.shape[1];
    final positionEmbeddings = seqLen > 1
        ? _buildAppliedMropeCosSin(posIds, embeddings.dtype)
        : null;
    var h = embeddings;
    try {
      for (var i = 0; i < _lmLayers.length; i++) {
        final next = _decoderLayer(
          _lmLayers[i],
          h,
          seqLen,
          posIds,
          positionEmbeddings: positionEmbeddings,
          layerIndex: i,
          cache: null,
        );
        if (h != embeddings) h.close();
        h = next;
      }
      final last = h
          .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, config.hiddenSize])
          .reshape([1, config.hiddenSize]);
      h.close();
      final norm = _lmRmsNormCompat(last, weight: _finalNorm, eps: config.rmsNormEps);
      last.close();
      try {
        return norm.reshape([1, config.hiddenSize]);
      } finally {
        norm.close();
      }
    } catch (_) {
      if (h != embeddings) h.close();
      rethrow;
    } finally {
      positionEmbeddings?.cos.close();
      positionEmbeddings?.sin.close();
    }
  }

  MlxArray _debugPrefillLogitsPrefixFromEmbeddings(
    MlxArray embeddings,
    MlxArray posIds, {
    int width = 16,
  }) {
    final last = _debugPrefillLastHiddenFromEmbeddings(embeddings, posIds);
    try {
      final linear = config.tieWordEmbeddings ? _embedWeights : _lmHead!;
      if (linear case final _QuantLinear quant) {
        final prefix = _debugLmHeadPrefixQuant(quant, width);
        final logits = mx.quant.matmul(
          last,
          MlxQuantizedMatrix(prefix.weight, prefix.scales, prefix.biases),
          transpose: true,
          groupSize: quant.quantSpec.groupSize,
          bits: quant.quantSpec.bits,
          mode: quant.quantSpec.mode,
        );
        try {
          return logits
              .reshape([1, width])
              .astype(MlxDType.MLX_FLOAT32);
        } finally {
          logits.close();
        }
      }
      final logits = linear.apply(last);
      try {
        return logits
            .slice(start: [0, 0], stop: [1, width])
            .reshape([1, width])
            .astype(MlxDType.MLX_FLOAT32);
      } finally {
        logits.close();
      }
    } finally {
      last.close();
    }
  }

  MlxArray debugPrefillLogitsPrefixFromPixelValues(
    List<int> promptIds,
    MlxArray pixelValues,
    MlxArray imageGridThw, {
    int width = 16,
  }) {
    final imageEncoding = _encodeImageFromPixelValues(pixelValues, imageGridThw);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    try {
      return _debugPrefillLogitsPrefixFromEmbeddings(
        embeddings,
        posIds,
        width: width,
      );
    } finally {
      embeddings.close();
      posIds.close();
    }
  }

  MlxArray debugEncodeImageFeaturesFromPixelValues(
    MlxArray pixelValues,
    MlxArray imageGridThw,
  ) {
    final imageEncoding = _encodeImageFromPixelValues(pixelValues, imageGridThw);
    return imageEncoding.hidden;
  }

  MlxArray debugPrefillFinalHiddenFromPixelValues(
    List<int> promptIds,
    MlxArray pixelValues,
    MlxArray imageGridThw,
  ) {
    final imageEncoding = _encodeImageFromPixelValues(pixelValues, imageGridThw);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    try {
      final hidden = _debugPrefillLastHiddenFromEmbeddings(embeddings, posIds);
      try {
        return hidden.astype(MlxDType.MLX_FLOAT32);
      } finally {
        hidden.close();
      }
    } finally {
      embeddings.close();
      posIds.close();
    }
  }

  MlxArray debugPrefillLogitsFromPixelValues(
    List<int> promptIds,
    MlxArray pixelValues,
    MlxArray imageGridThw,
  ) {
    final imageEncoding = _encodeImageFromPixelValues(pixelValues, imageGridThw);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();

    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      try {
        return logits.reshape([1, config.vocabSize]);
      } finally {
        logits.close();
      }
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  MlxArray debugPrefillLogitsFromImage(
    List<int> promptIds,
    MlxArray imagePixels,
  ) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();

    final cache = _ModelCache.create(config: config);
    try {
      return _prefillFromEmbeddingWithCache(embeddings, posIds, cache)
          .reshape([1, config.vocabSize]);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  List<Map<String, Object?>> debugTokenMarginsFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    int maxNewTokens = 32,
  }) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();

    final cache = _ModelCache.create(config: config);
    try {
      var logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();

      final rows = <Map<String, Object?>>[];
      var nextTextPosition = positionInfo.nextTextPosition;
      for (var step = 0; step < maxNewTokens; step++) {
        final top2 = logits.topK(2, axis: -1);
        try {
          MlxRuntime.evalAll([top2]);
          final values = top2.toFloat32List();
          final top2Value = values.length >= 2 ? values.first : double.nan;
          final top1Value = values.isNotEmpty ? values.last : double.nan;
          final next = _nextTokenFromLogits(logits);
          rows.add({
            'step': step + 1,
            'offset': cache.offset,
            'token': next,
            'top1': top1Value,
            'top2': top2Value,
            'gap': top1Value - top2Value,
          });
          if (next == config.eosTokenId || step + 1 >= maxNewTokens) {
            break;
          }
          logits.close();
          final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
          final stepPos = _textPositionIds(1, offset: nextTextPosition);
          try {
            logits = _forwardWithCache(stepArr, stepPos, cache);
          } finally {
            stepArr.close();
            stepPos.close();
          }
          nextTextPosition++;
        } finally {
          top2.close();
        }
      }
      logits.close();
      return rows;
    } finally {
      cache.close();
    }
  }

  String? debugTurboLastAttentionPath() => _turboLastAttentionPath;

  ({MlxArray q, MlxArray k, MlxArray v}) _debugApplyQkvFlat(
    _LmAttentionWeights attn,
    MlxArray flat,
    int seqLen,
  ) {
    final qWidth = config.numAttentionHeads * config.headDim;
    final kvWidth = config.numKeyValueHeads * config.headDim;
    if (attn.qkvProj case final fused?) {
      final qkv = fused.apply(flat);
      final q = qkv.slice(start: [0, 0], stop: [seqLen, qWidth]);
      final k = qkv.slice(
        start: [0, qWidth],
        stop: [seqLen, qWidth + kvWidth],
      );
      final v = qkv.slice(
        start: [0, qWidth + kvWidth],
        stop: [seqLen, qWidth + kvWidth * 2],
      );
      qkv.close();
      return (q: q, k: k, v: v);
    }
    return (
      q: attn.qProj!.apply(flat),
      k: attn.kProj!.apply(flat),
      v: attn.vProj!.apply(flat),
    );
  }

  ({MlxArray q, MlxArray k, MlxArray v}) _debugApplyQkvHeads(
    _LmAttentionWeights attn,
    MlxArray flat,
    int seqLen,
  ) {
    final qkv = _debugApplyQkvFlat(attn, flat, seqLen);
    final q = qkv.q.reshape([
      1,
      seqLen,
      config.numAttentionHeads,
      config.headDim,
    ]).transposeAxes([0, 2, 1, 3]);
    final k = qkv.k.reshape([
      1,
      seqLen,
      config.numKeyValueHeads,
      config.headDim,
    ]).transposeAxes([0, 2, 1, 3]);
    final v = qkv.v.reshape([
      1,
      seqLen,
      config.numKeyValueHeads,
      config.headDim,
    ]).transposeAxes([0, 2, 1, 3]);
    qkv.q.close();
    qkv.k.close();
    qkv.v.close();
    return (q: q, k: k, v: v);
  }

  MlxArray debugEmbedIds(List<int> tokenIds) {
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, tokenIds.length]);
    try {
      return _embed(ids);
    } finally {
      ids.close();
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
    final targetPixelDType = _visionWeights.patchEmbedWeight.dtype;
    final pixels = imagePixels.dtype == targetPixelDType
        ? imagePixels
        : imagePixels.astype(targetPixelDType);
    final patchInfo = _patchifyVisionImage(pixels, vCfg.patchSize);
    final gridH = patchInfo.gridHeight;
    final gridW = patchInfo.gridWidth;
    final patches = patchInfo.patches;
    final flatPatches = patches
        .reshape([gridH * gridW, pixels.shape[3], vCfg.patchSize, vCfg.patchSize])
        .transposeAxes([0, 2, 3, 1]);
    patches.close();
    if (!identical(pixels, imagePixels)) {
      pixels.close();
    }
    final patchOut = mx.conv2d(flatPatches, _visionWeights.patchEmbedWeight);
    flatPatches.close();
    var hidden = patchOut.reshape([gridH * gridW, vCfg.hiddenSize]);
    patchOut.close();
    if (_visionWeights.patchEmbedBias != null) {
      final biased = mx.add(hidden, _visionWeights.patchEmbedBias!);
      hidden.close();
      hidden = biased;
    }
    final targetPatchDType = _visionWeights.patchEmbedWeight.dtype;
    if (hidden.dtype != targetPatchDType) {
      final cast = hidden.astype(targetPatchDType);
      hidden.close();
      hidden = cast;
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

  /// Return raw patch embeddings before position interpolation.
  MlxArray encodeImagePatchOnly(MlxArray imagePixels) {
    final vCfg = config._vision;
    final targetPixelDType = _visionWeights.patchEmbedWeight.dtype;
    final pixels = imagePixels.dtype == targetPixelDType
        ? imagePixels
        : imagePixels.astype(targetPixelDType);
    final patchInfo = _patchifyVisionImage(pixels, vCfg.patchSize);
    final gridH = patchInfo.gridHeight;
    final gridW = patchInfo.gridWidth;
    final patches = patchInfo.patches;
    final flatPatches = patches
        .reshape([gridH * gridW, pixels.shape[3], vCfg.patchSize, vCfg.patchSize])
        .transposeAxes([0, 2, 3, 1]);
    patches.close();
    if (!identical(pixels, imagePixels)) {
      pixels.close();
    }
    final patchOut = mx.conv2d(flatPatches, _visionWeights.patchEmbedWeight);
    flatPatches.close();
    var hidden = patchOut.reshape([gridH * gridW, vCfg.hiddenSize]);
    patchOut.close();
    if (_visionWeights.patchEmbedBias != null) {
      final biased = mx.add(hidden, _visionWeights.patchEmbedBias!);
      hidden.close();
      hidden = biased;
    }
    final targetPatchDType = _visionWeights.patchEmbedWeight.dtype;
    if (hidden.dtype != targetPatchDType) {
      final cast = hidden.astype(targetPatchDType);
      hidden.close();
      hidden = cast;
    }
    return hidden;
  }


  /// Return patchified image tensor `[1, seq, C, patch, patch]`.
  MlxArray encodeImagePatchesOnly(MlxArray imagePixels) {
    final vCfg = config._vision;
    final patchInfo = _patchifyVisionImage(imagePixels, vCfg.patchSize);
    return patchInfo.patches;
  }

  MlxArray debugVisionPatchWeight() => _visionWeights.patchEmbedWeight;

  MlxArray? debugVisionPatchBias() => _visionWeights.patchEmbedBias;

  /// Return vision hidden states after a given number of transformer layers,
  /// before post-layernorm/projector. Intended for parity debugging only.
  MlxArray encodeImageAfterLayerCount(MlxArray imagePixels, int layerCount) {
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
        hidden = _visionBlock(
          _visionWeights.blocks[i],
          hidden,
          vCfg,
          rotaryPosEmb,
        );
      }
      return hidden;
    } finally {
      rotaryPosEmb.close();
    }
  }

  MlxArray encodeVisionFromEmbeddings(
    MlxArray hidden,
    int gridH,
    int gridW, {
    int? layerCount,
  }) {
    final vCfg = config._vision;
    final rotaryPosEmb = _buildVisionRotaryPosEmbedding(
      gridH,
      gridW,
      MlxDType.MLX_FLOAT32,
    );
    try {
      final total = layerCount ?? _visionWeights.blocks.length;
      var h = hidden;
      for (var i = 0; i < total; i++) {
        final next = _visionBlock(
          _visionWeights.blocks[i],
          h,
          vCfg,
          rotaryPosEmb,
        );
        if (h != hidden) h.close();
        h = next;
      }
      if (total == _visionWeights.blocks.length) {
        final postNorm = _visionLayerNorm(
          h,
          weight: _visionWeights.postLayerNormWeight,
          bias: _visionWeights.postLayerNormBias,
          eps: vCfg.layerNormEps,
        );
        h.close();
        final merged = _spatialMergeProject(postNorm, gridH, gridW, vCfg);
        postNorm.close();
        return merged;
      }
      return h;
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

  MlxArray debugVisionPostNormFromHidden(MlxArray hidden) {
    return _visionLayerNorm(
      hidden,
      weight: _visionWeights.postLayerNormWeight,
      bias: _visionWeights.postLayerNormBias,
      eps: config._vision.layerNormEps,
    );
  }

  MlxArray debugVisionProjectorFromPostNorm(
    MlxArray hidden,
    int gridH,
    int gridW,
  ) {
    return _spatialMergeProject(hidden, gridH, gridW, config._vision);
  }

  MlxArray debugVisionProjectorPreNorm(MlxArray hidden) {
    final proj = _visionWeights.projector;
    return _visionLayerNorm(
      hidden,
      weight: proj.preNormWeight,
      bias: proj.preNormBias,
      eps: config._vision.layerNormEps,
    );
  }

  MlxArray debugVisionProjectorFlat(
    MlxArray hidden,
    int gridH,
    int gridW,
  ) {
    final proj = _visionWeights.projector;
    final vHidden = config._vision.hiddenSize;
    final m = config._vision.spatialMergeSize;
    final mergedH = gridH ~/ m;
    final mergedW = gridW ~/ m;
    final normed = _visionLayerNorm(
      hidden,
      weight: proj.preNormWeight,
      bias: proj.preNormBias,
      eps: config._vision.layerNormEps,
    );
    final grid = normed.reshape([1, gridH, gridW, vHidden]);
    normed.close();
    final reshaped = grid.reshape([
      1,
      mergedH,
      m,
      mergedW,
      m,
      vHidden,
    ]);
    grid.close();
    final transposed = reshaped.transposeAxes([0, 1, 3, 2, 4, 5]);
    reshaped.close();
    final flat = transposed.reshape([
      mergedH * mergedW,
      m * m * vHidden,
    ]);
    transposed.close();
    return flat;
  }

  MlxArray debugVisionProjectorLinear1(MlxArray hidden) =>
      _visionWeights.projector.linear1.apply(hidden);

  MlxArray debugVisionProjectorGelu(MlxArray hidden) => _geluDefault(hidden);

  MlxArray debugVisionProjectorLinear2(MlxArray hidden) =>
      _visionWeights.projector.linear2.apply(hidden);

  /// Return the vision rotary embedding for the current image grid.
  MlxArray encodeVisionRotaryEmbedding(MlxArray imagePixels) {
    final vCfg = config._vision;
    final gridH = imagePixels.shape[1] ~/ vCfg.patchSize;
    final gridW = imagePixels.shape[2] ~/ vCfg.patchSize;
    return _buildVisionRotaryPosEmbedding(gridH, gridW, MlxDType.MLX_FLOAT32);
  }

  MlxArray encodeVisionPositionEmbedding(int gridH, int gridW) =>
      _interpolateVisionPositionEmbedding(gridH, gridW, MlxDType.MLX_FLOAT32);

  MlxArray encodeVisionPositionEmbeddingUsed(MlxArray imagePixels) {
    final vCfg = config._vision;
    final gridH = imagePixels.shape[1] ~/ vCfg.patchSize;
    final gridW = imagePixels.shape[2] ~/ vCfg.patchSize;
    return _interpolateVisionPositionEmbedding(
      gridH,
      gridW,
      _visionWeights.patchEmbedWeight.dtype,
    );
  }

  MlxArray debugVisionPositionTable() {
    final base = _fullVisionPositionEmbedding();
    return MlxArray.fromFloat32List(
      base,
      shape: [
        base.length ~/ config._vision.hiddenSize,
        config._vision.hiddenSize,
      ],
    );
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

  ({MlxArray q, MlxArray k}) debugApplyMropeAtTextOffset(
    MlxArray q,
    MlxArray k,
    int offset,
  ) {
    final pos = _textPositionIds(1, offset: offset);
    try {
      return _applyMrope(q, k, pos);
    } finally {
      pos.close();
    }
  }

  /// Return the first vision-layer self-attention output before residual add.
  /// Intended for parity debugging only.
  MlxArray debugFirstVisionAttentionOutput(MlxArray imagePixels) {
    return debugVisionAttentionOutput(imagePixels, 0);
  }

  MlxArray debugVisionQkvOutput(MlxArray imagePixels, int layerIndex) {
    final vCfg = config._vision;
    final hidden = encodeImageEmbeddingsOnly(imagePixels);
    try {
      final block = _visionWeights.blocks[layerIndex];
      final norm1 = _visionLayerNorm(
        hidden,
        weight: block.layerNorm1Weight,
        bias: block.layerNorm1Bias,
        eps: vCfg.layerNormEps,
      );
      hidden.close();
      try {
        return block.qkv.apply(norm1);
      } finally {
        norm1.close();
      }
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray debugVisionQkvApply(MlxArray hidden, int layerIndex) {
    final block = _visionWeights.blocks[layerIndex];
    return block.qkv.apply(hidden);
  }

  MlxArray debugVisionBlockFromHidden(
    MlxArray hidden,
    int layerIndex,
    MlxArray rotaryPosEmb,
  ) {
    return _visionBlock(
      _visionWeights.blocks[layerIndex],
      hidden,
      config._vision,
      rotaryPosEmb,
    );
  }

  MlxArray debugVisionAttentionFromQkv(
    MlxArray qkv,
    MlxArray rotaryPosEmb,
    int layerIndex,
  ) {
    final vCfg = config._vision;
    final block = _visionWeights.blocks[layerIndex];
    final seqLen = qkv.shape[0];
    final numHeads = vCfg.numAttentionHeads;
    final headDim = vCfg.headDim;

    final qkv4d = qkv.reshape([seqLen, 3, numHeads, headDim]).transposeAxes([
      1,
      0,
      2,
      3,
    ]);

    final q = qkv4d.slice(
      start: [0, 0, 0, 0],
      stop: [1, seqLen, numHeads, headDim],
    );
    final k = qkv4d.slice(
      start: [1, 0, 0, 0],
      stop: [2, seqLen, numHeads, headDim],
    );
    final v = qkv4d.slice(
      start: [2, 0, 0, 0],
      stop: [3, seqLen, numHeads, headDim],
    );
    qkv4d.close();

    final qRot = _applyVisionRotary(q, rotaryPosEmb);
    final kRot = _applyVisionRotary(k, rotaryPosEmb);
    q.close();
    k.close();

    final qForAttn = qRot.transposeAxes([0, 2, 1, 3]);
    final kForAttn = kRot.transposeAxes([0, 2, 1, 3]);
    final vForAttn = v.transposeAxes([0, 2, 1, 3]);
    qRot.close();
    kRot.close();
    v.close();

    final mask = MlxArray.zeros([1, seqLen, seqLen], dtype: qForAttn.dtype);
    try {
      final attn = mx.fast.scaledDotProductAttention(
        qForAttn,
        kForAttn,
        vForAttn,
        scale: 1.0 / math.sqrt(headDim.toDouble()),
        mask: mask,
      );
      qForAttn.close();
      kForAttn.close();
      vForAttn.close();
      final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
        seqLen,
        numHeads * headDim,
      ]);
      attn.close();
      final out = block.outProj.apply(merged);
      merged.close();
      return out;
    } finally {
      mask.close();
    }
  }

  ({MlxArray q, MlxArray k}) debugVisionRopedQkFromQkv(
    MlxArray qkv,
    MlxArray rotaryPosEmb,
  ) {
    final seqLen = qkv.shape[0];
    final numHeads = config._vision.numAttentionHeads;
    final headDim = config._vision.headDim;

    final qkv4d = qkv.reshape([seqLen, 3, numHeads, headDim]).transposeAxes([
      1,
      0,
      2,
      3,
    ]);

    final q = qkv4d.slice(
      start: [0, 0, 0, 0],
      stop: [1, seqLen, numHeads, headDim],
    );
    final k = qkv4d.slice(
      start: [1, 0, 0, 0],
      stop: [2, seqLen, numHeads, headDim],
    );
    qkv4d.close();

    final qRot = _applyVisionRotary(q, rotaryPosEmb);
    final kRot = _applyVisionRotary(k, rotaryPosEmb);
    q.close();
    k.close();
    return (q: qRot, k: kRot);
  }

  MlxArray debugVisionLayerNorm1Output(MlxArray imagePixels, int layerIndex) {
    final vCfg = config._vision;
    final hidden = encodeImageEmbeddingsOnly(imagePixels);
    try {
      final block = _visionWeights.blocks[layerIndex];
      return _visionLayerNorm(
        hidden,
        weight: block.layerNorm1Weight,
        bias: block.layerNorm1Bias,
        eps: vCfg.layerNormEps,
      );
    } finally {
      hidden.close();
    }
  }

  MlxArray debugApplyVisionLayerNorm1(MlxArray hidden, int layerIndex) {
    final vCfg = config._vision;
    final block = _visionWeights.blocks[layerIndex];
    return _visionLayerNorm(
      hidden,
      weight: block.layerNorm1Weight,
      bias: block.layerNorm1Bias,
      eps: vCfg.layerNormEps,
    );
  }

  MlxArray debugApplyVisionLayerNorm2(MlxArray hidden, int layerIndex) {
    final vCfg = config._vision;
    final block = _visionWeights.blocks[layerIndex];
    return _visionLayerNorm(
      hidden,
      weight: block.layerNorm2Weight,
      bias: block.layerNorm2Bias,
      eps: vCfg.layerNormEps,
    );
  }

  MlxArray debugVisionLayerNorm2Weight(int layerIndex) =>
      _visionWeights.blocks[layerIndex].layerNorm2Weight;

  MlxArray debugVisionLayerNorm2Bias(int layerIndex) =>
      _visionWeights.blocks[layerIndex].layerNorm2Bias;

  MlxArray debugVisionMlpApply(MlxArray hidden, int layerIndex) {
    return _visionMlp(_visionWeights.blocks[layerIndex], hidden);
  }

  MlxArray debugVisionMlpFc1(MlxArray hidden, int layerIndex) {
    return _visionWeights.blocks[layerIndex].fc1.apply(hidden);
  }

  MlxArray debugVisionGelu(MlxArray hidden) => _gelu(hidden);

  MlxArray debugVisionMlpFc2(MlxArray hidden, int layerIndex) {
    return _visionWeights.blocks[layerIndex].fc2.apply(hidden);
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
      final clampedIndex = layerIndex.clamp(
        0,
        _visionWeights.blocks.length - 1,
      );
      var hidden = input;
      for (var i = 0; i < clampedIndex; i++) {
        hidden = _visionBlock(
          _visionWeights.blocks[i],
          hidden,
          vCfg,
          rotary32,
        );
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
      final clampedIndex = layerIndex.clamp(
        0,
        _visionWeights.blocks.length - 1,
      );
      var hidden = input;
      for (var i = 0; i < clampedIndex; i++) {
        hidden = _visionBlock(
          _visionWeights.blocks[i],
          hidden,
          vCfg,
          rotary,
        );
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

  ({MlxArray norms, MlxArray indices}) debugTurboQuantMseQuantize(
    MlxArray vectors, {
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(
      dim: vectors.shape.last,
      bits: bits,
      seed: seed,
    );
    try {
      final state = codec.quantize(vectors);
      return (norms: state.norms, indices: state.indices);
    } finally {
      codec.close();
    }
  }

  ({MlxArray norms, MlxArray unit, MlxArray rotated}) debugTurboQuantMsePrepare(
    MlxArray vectors, {
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(
      dim: vectors.shape.last,
      bits: bits,
      seed: seed,
    );
    try {
      final flat = vectors
          .reshape([vectors.size ~/ vectors.shape.last, vectors.shape.last])
          .astype(MlxDType.MLX_FLOAT32);
      final norms = mx.linalg.norm(flat, axes: [1]);
      final eps = MlxArray.full([], 1e-6, dtype: MlxDType.MLX_FLOAT32);
      final safeNorms = mx.maximum(norms, eps);
      eps.close();
      final unit = flat / safeNorms.expandDims(1);
      flat.close();
      safeNorms.close();
      final rotated = codec.prepareQueries(unit);
      return (norms: norms, unit: unit, rotated: rotated);
    } finally {
      codec.close();
    }
  }

  MlxArray debugTurboQuantUnpackIndices(
    MlxArray packed, {
    required int bits,
    required int length,
  }) => _turboUnpackLowbit(packed, bits, length);

  MlxArray debugTurboQuantRotation(
    int dim, {
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(dim: dim, bits: bits, seed: seed);
    try {
      return codec.rotation.astype(MlxDType.MLX_FLOAT32);
    } finally {
      codec.close();
    }
  }

  ({MlxArray codebook, MlxArray midpoints}) debugTurboQuantCodebook(
    int dim, {
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(dim: dim, bits: bits, seed: seed);
    try {
      return (
        codebook: codec.codebook.astype(MlxDType.MLX_FLOAT32),
        midpoints: codec.midpoints.astype(MlxDType.MLX_FLOAT32),
      );
    } finally {
      codec.close();
    }
  }

  MlxArray debugTurboQuantPrepareQueries(
    MlxArray queries, {
    required int dim,
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(dim: dim, bits: bits, seed: seed);
    try {
      return codec.prepareQueries(queries);
    } finally {
      codec.close();
    }
  }

  MlxArray debugTurboQuantInverseRotate(
    MlxArray rotated, {
    required int dim,
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(dim: dim, bits: bits, seed: seed);
    try {
      return codec._rotateInverse(rotated);
    } finally {
      codec.close();
    }
  }

  MlxArray debugTurboQuantRotate(
    MlxArray vectors, {
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(
      dim: vectors.shape.last,
      bits: bits,
      seed: seed,
    );
    try {
      final vectorsF32 = vectors.astype(MlxDType.MLX_FLOAT32);
      final squares = MlxMore.square(vectorsF32);
      final sqSum = mx.sum(squares, axis: vectors.ndim - 1);
      squares.close();
      final norms = MlxMore.sqrt(sqSum);
      sqSum.close();
      final eps = MlxArray.full([], 1e-6, dtype: MlxDType.MLX_FLOAT32);
      final safeNorms = mx.maximum(norms, eps);
      eps.close();
      final unit = vectorsF32 / safeNorms.expandDims(vectors.ndim - 1);
      vectorsF32.close();
      norms.close();
      safeNorms.close();
      final rotated = codec.prepareQueries(unit);
      unit.close();
      return rotated;
    } finally {
      codec.close();
    }
  }

  MlxArray debugTurboNoRotQuantize(
    MlxArray rotated, {
    required int bits,
    int seed = 0,
  }) {
    final codec = _TurboQuantMseCodec(
      dim: rotated.shape.last,
      bits: bits,
      seed: seed,
    );
    try {
      final flat = rotated.reshape([rotated.size ~/ rotated.shape.last, rotated.shape.last]);
      final packedWidth = _turboPackedWidth(rotated.shape.last, bits);
      final kernel = _getTurboNoRotQuantizeKernel(bits);
      final config = mx.fast.metalConfig();
      config.addOutputArg([flat.shape[0], packedWidth], MlxDType.MLX_UINT32);
      config.setGrid(rotated.shape.last * flat.shape[0], 1, 1);
      config.setThreadGroup(rotated.shape.last, 1, 1);
      config.addTemplateInt('Dim', rotated.shape.last);
      config.addTemplateInt('Bits', bits);
      config.addTemplateInt('PackedWidth', packedWidth);
      final out = kernel.apply([flat, codec.midpoints], config).first;
      flat.close();
      return out.reshape([
        ...rotated.shape.sublist(0, rotated.ndim - 1),
        packedWidth,
      ]);
    } finally {
      codec.close();
    }
  }

  ({
    MlxArray keyNorms,
    MlxArray keyIndices,
    MlxArray valueNorms,
    MlxArray valueIndices,
  }) debugTurboQuantCacheAppend(
    MlxArray keys,
    MlxArray values, {
    required double bits,
  }) {
    final cache = _TurboQuantKvCache(
      numKvHeads: keys.shape[1],
      headDim: keys.shape[3],
      maxSeqLen: keys.shape[2],
      bits: bits,
    );
    try {
      for (var i = 0; i < keys.shape[2]; i++) {
        final keySlice = keys.slice(
          start: [0, 0, i, 0],
          stop: [keys.shape[0], keys.shape[1], i + 1, keys.shape[3]],
        );
        final valueSlice = values.slice(
          start: [0, 0, i, 0],
          stop: [values.shape[0], values.shape[1], i + 1, values.shape[3]],
        );
        cache.update(keySlice, valueSlice);
      }
      final keyState = cache._keys!;
      final valueState = cache._values!;
      return (
        keyNorms: keyState.norms,
        keyIndices: keyState.indices,
        valueNorms: valueState.norms,
        valueIndices: valueState.indices,
      );
    } finally {
      cache._keys = null;
      cache._values = null;
      cache.close();
    }
  }

  ({
    MlxArray keyNorms,
    MlxArray keyIndices,
    MlxArray valueNorms,
    MlxArray valueIndices,
  }) debugTurboQuantFusedKv(
    MlxArray keys,
    MlxArray values,
  ) {
    final keyCodec = _TurboQuantMseCodec(dim: 128, bits: 3, seed: 0);
    final valueCodec = _TurboQuantMseCodec(dim: 128, bits: 4, seed: 1);
    try {
      final out = _turboTryFusedKvQuantize(keys, values, keyCodec, valueCodec);
      if (out == null) {
        throw StateError('TurboQuant fused kv path unavailable');
      }
      return (
        keyNorms: out.$1.norms,
        keyIndices: out.$1.indices,
        valueNorms: out.$2.norms,
        valueIndices: out.$2.indices,
      );
    } finally {
      keyCodec.close();
      valueCodec.close();
    }
  }

  MlxArray debugTurboQuantFusedDecodeRaw(
    MlxArray qRotFlat,
    MlxArray keyNorms,
    MlxArray keyIndices,
    MlxArray valueNorms,
    MlxArray valueIndices, {
    required int repeatCount,
    required int dim,
    required int keyBits,
    required int valueBits,
  }) {
    final keyCodec = _TurboQuantMseCodec(dim: dim, bits: keyBits, seed: 0);
    final valueCodec = _TurboQuantMseCodec(dim: dim, bits: valueBits, seed: 1);
    try {
      final kernel = _getTurboFusedMseDecodeKernel(keyBits, valueBits, dim);
      final config = mx.fast.metalConfig();
      config.addOutputArg([qRotFlat.shape[0], dim], MlxDType.MLX_FLOAT32);
      config.setGrid(qRotFlat.shape[0] * 1024, 1, 1);
      config.setThreadGroup(1024, 1, 1);
      config.addTemplateInt('Dim', dim);
      config.addTemplateInt('RepeatCount', repeatCount);
      config.addTemplateInt('KPackedWidth', keyIndices.shape.last);
      config.addTemplateInt('VPackedWidth', valueIndices.shape.last);
      final outputs = kernel.apply([
        qRotFlat,
        keyNorms,
        keyIndices,
        keyCodec.codebook,
        valueNorms,
        valueIndices,
        valueCodec.codebook,
      ], config);
      return outputs.first;
    } finally {
      keyCodec.close();
      valueCodec.close();
    }
  }

  MlxArray? debugTurboQuantFusedDecodeDirect(
    MlxArray queries,
    MlxArray keyNorms,
    MlxArray keyIndices,
    MlxArray valueNorms,
    MlxArray valueIndices, {
    required int numKvHeads,
    required int headDim,
    required int keyBits,
    required int valueBits,
    required double scale,
  }) {
    final keyCodec = _TurboQuantMseCodec(dim: headDim, bits: keyBits, seed: 0);
    final valueCodec = _TurboQuantMseCodec(dim: headDim, bits: valueBits, seed: 1);
    final keyState = _TurboQuantMseState(keyNorms, keyIndices);
    final valueState = _TurboQuantMseState(valueNorms, valueIndices);
    try {
      return _turboMseFusedDecodeAttention(
        queries,
        keyState,
        valueState,
        keyCodec: keyCodec,
        valueCodec: valueCodec,
        repeats: queries.shape[1] ~/ numKvHeads,
        scale: scale,
      );
    } finally {
      keyCodec.close();
      valueCodec.close();
    }
  }

  MlxArray debugTurboQuantAttention(
    MlxArray keys,
    MlxArray values,
    MlxArray queries, {
    required double bits,
    required double scale,
    Object? mask,
  }) {
    final cache = _TurboQuantKvCache(
      numKvHeads: keys.shape[1],
      headDim: keys.shape[3],
      maxSeqLen: keys.shape[2],
      bits: bits,
    );
    try {
      for (var i = 0; i < keys.shape[2]; i++) {
        final keySlice = keys.slice(
          start: [0, 0, i, 0],
          stop: [keys.shape[0], keys.shape[1], i + 1, keys.shape[3]],
        );
        final valueSlice = values.slice(
          start: [0, 0, i, 0],
          stop: [values.shape[0], values.shape[1], i + 1, values.shape[3]],
        );
        cache.update(keySlice, valueSlice);
      }
      return cache.quantizedAttention(
        queries,
        scale: scale,
        mask: mask,
      );
    } finally {
      cache.close();
    }
  }

  MlxArray debugTurboQuantAttentionMixed(
    MlxArray keys,
    MlxArray values,
    MlxArray queries, {
    required int prefillTokens,
    required double bits,
    required double scale,
    Object? mask,
  }) {
    final cache = _TurboQuantKvCache(
      numKvHeads: keys.shape[1],
      headDim: keys.shape[3],
      maxSeqLen: keys.shape[2],
      bits: bits,
    );
    try {
      if (prefillTokens > 0) {
        final keyPrefill = keys.slice(
          start: [0, 0, 0, 0],
          stop: [keys.shape[0], keys.shape[1], prefillTokens, keys.shape[3]],
        );
        final valuePrefill = values.slice(
          start: [0, 0, 0, 0],
          stop: [values.shape[0], values.shape[1], prefillTokens, values.shape[3]],
        );
        cache.update(keyPrefill, valuePrefill);
      }
      for (var i = prefillTokens; i < keys.shape[2]; i++) {
        final keySlice = keys.slice(
          start: [0, 0, i, 0],
          stop: [keys.shape[0], keys.shape[1], i + 1, keys.shape[3]],
        );
        final valueSlice = values.slice(
          start: [0, 0, i, 0],
          stop: [values.shape[0], values.shape[1], i + 1, values.shape[3]],
        );
        cache.update(keySlice, valueSlice);
      }
      return cache.quantizedAttention(
        queries,
        scale: scale,
        mask: mask,
      );
    } finally {
      cache.close();
    }
  }

  MlxArray debugTurboQuantScore(
    MlxArray keys,
    MlxArray queries, {
    required double bits,
    required double scale,
  }) {
    final keyCodec = _TurboQuantMseCodec(
      dim: keys.shape[3],
      bits: bits.floor(),
      seed: 0,
    );
    try {
      final keyState = keyCodec.quantize(keys);
      final repeats = queries.shape[1] ~/ keys.shape[1];
      final scaleArr = MlxArray.full([], scale, dtype: queries.dtype);
      final scaled = queries * scaleArr;
      scaleArr.close();
      final grouped = scaled.reshape([
        queries.shape[0],
        keys.shape[1],
        repeats,
        queries.shape[2],
        queries.shape[3],
      ]);
      scaled.close();
      final prepared = keyCodec.prepareQueries(grouped);
      grouped.close();
      final scores = keyCodec.scorePrepared(prepared, keyState);
      prepared.close();
      keyState.norms.close();
      keyState.indices.close();
      return scores;
    } finally {
      keyCodec.close();
    }
  }

  ({MlxArray output, MlxArray denom, MlxArray maxScores})
  debugTurboQuantValueStats(
    MlxArray values,
    MlxArray scores, {
    required double bits,
  }) {
    final valueCodec = _TurboQuantMseCodec(
      dim: values.shape[3],
      bits: bits.ceil(),
      seed: 1,
    );
    try {
      final valueState = valueCodec.quantize(values);
      final out = _turboMseWeightedSumStatsFromScores(
        scores,
        valueState,
        valueCodec,
      );
      if (out == null) {
        throw StateError('TurboQuant value stats path unavailable');
      }
      valueState.norms.close();
      valueState.indices.close();
      return out;
    } finally {
      valueCodec.close();
    }
  }

  ({MlxArray keys, MlxArray values}) debugTurboQuantCacheDequantize(
    MlxArray keys,
    MlxArray values, {
    required double bits,
  }) {
    final cache = _TurboQuantKvCache(
      numKvHeads: keys.shape[1],
      headDim: keys.shape[3],
      maxSeqLen: keys.shape[2],
      bits: bits,
    );
    try {
      for (var i = 0; i < keys.shape[2]; i++) {
        final keySlice = keys.slice(
          start: [0, 0, i, 0],
          stop: [keys.shape[0], keys.shape[1], i + 1, keys.shape[3]],
        );
        final valueSlice = values.slice(
          start: [0, 0, i, 0],
          stop: [values.shape[0], values.shape[1], i + 1, values.shape[3]],
        );
        cache.update(keySlice, valueSlice);
      }
      final keyState = _turboSliceStateRange(cache._keys!, 0, cache.offset);
      final valueState = _turboSliceStateRange(cache._values!, 0, cache.offset);
      try {
        return (
          keys: cache._keyCodec.dequantize(keyState),
          values: cache._valueCodec.dequantize(valueState),
        );
      } finally {
        keyState.close();
        valueState.close();
      }
    } finally {
      cache.close();
    }
  }

  ({int firstToken, MlxArray secondLogits}) debugSecondDecodeLogitsFromImage(
    List<int> promptIds,
    MlxArray imagePixels,
  ) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    final cache = _ModelCache.create(config: config);
    try {
      var logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: cache.offset);
      try {
        final secondLogits = _forwardWithCache(stepArr, stepPos, cache);
        logits.close();
        return (firstToken: firstToken, secondLogits: secondLogits);
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      cache.close();
    }
  }

  ({MlxArray keys, MlxArray values}) debugPrefillQuantizedLayerDequantizedStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      logits.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _QuantizedKvCache) {
        throw StateError('Layer $layerIndex is not using quantized KV cache');
      }
      final keys = mx.quant.dequantize(
        layer.borrowedKeys,
        groupSize: layer.groupSize,
        bits: layer.bits,
        mode: 'affine',
      );
      final values = mx.quant.dequantize(
        layer.borrowedValues,
        groupSize: layer.groupSize,
        bits: layer.bits,
        mode: 'affine',
      );
      final validKeys = keys.slice(
        start: [0, 0, 0, 0],
        stop: [keys.shape[0], keys.shape[1], layer.offset, keys.shape[3]],
      );
      final validValues = values.slice(
        start: [0, 0, 0, 0],
        stop: [values.shape[0], values.shape[1], layer.offset, values.shape[3]],
      );
      keys.close();
      values.close();
      return (keys: validKeys, values: validValues);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({MlxArray keyWeights, MlxArray keyScales, MlxArray keyBiases, MlxArray valueWeights, MlxArray valueScales, MlxArray valueBiases})
  debugPrefillQuantizedLayerRawStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      logits.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _QuantizedKvCache) {
        throw StateError('Layer $layerIndex is not using quantized KV cache');
      }
      final keyWeights = layer.borrowedKeys.weights.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedKeys.weights.shape[0],
          layer.borrowedKeys.weights.shape[1],
          layer.offset,
          layer.borrowedKeys.weights.shape[3],
        ],
      );
      final keyScales = layer.borrowedKeys.scales.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedKeys.scales.shape[0],
          layer.borrowedKeys.scales.shape[1],
          layer.offset,
          layer.borrowedKeys.scales.shape[3],
        ],
      );
      final keyBiases = layer.borrowedKeys.biases!.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedKeys.biases!.shape[0],
          layer.borrowedKeys.biases!.shape[1],
          layer.offset,
          layer.borrowedKeys.biases!.shape[3],
        ],
      );
      final valueWeights = layer.borrowedValues.weights.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedValues.weights.shape[0],
          layer.borrowedValues.weights.shape[1],
          layer.offset,
          layer.borrowedValues.weights.shape[3],
        ],
      );
      final valueScales = layer.borrowedValues.scales.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedValues.scales.shape[0],
          layer.borrowedValues.scales.shape[1],
          layer.offset,
          layer.borrowedValues.scales.shape[3],
        ],
      );
      final valueBiases = layer.borrowedValues.biases!.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedValues.biases!.shape[0],
          layer.borrowedValues.biases!.shape[1],
          layer.offset,
          layer.borrowedValues.biases!.shape[3],
        ],
      );
      return (
        keyWeights: keyWeights,
        keyScales: keyScales,
        keyBiases: keyBiases,
        valueWeights: valueWeights,
        valueScales: valueScales,
        valueBiases: valueBiases,
      );
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray keys, MlxArray values})
  debugSecondDecodeQuantizedLayerDequantizedStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      final second = _forwardWithCache(stepArr, stepPos, cache);
      second.close();
      stepArr.close();
      stepPos.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _QuantizedKvCache) {
        throw StateError('Layer $layerIndex is not using quantized KV cache');
      }
      final keys = mx.quant.dequantize(
        layer.borrowedKeys,
        groupSize: layer.groupSize,
        bits: layer.bits,
        mode: 'affine',
      );
      final values = mx.quant.dequantize(
        layer.borrowedValues,
        groupSize: layer.groupSize,
        bits: layer.bits,
        mode: 'affine',
      );
      final validKeys = keys.slice(
        start: [0, 0, 0, 0],
        stop: [keys.shape[0], keys.shape[1], layer.offset, keys.shape[3]],
      );
      final validValues = values.slice(
        start: [0, 0, 0, 0],
        stop: [values.shape[0], values.shape[1], layer.offset, values.shape[3]],
      );
      keys.close();
      values.close();
      return (firstToken: firstToken, keys: validKeys, values: validValues);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray keyWeights, MlxArray keyScales, MlxArray keyBiases, MlxArray valueWeights, MlxArray valueScales, MlxArray valueBiases})
  debugSecondDecodeQuantizedLayerRawStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      final second = _forwardWithCache(stepArr, stepPos, cache);
      second.close();
      stepArr.close();
      stepPos.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _QuantizedKvCache) {
        throw StateError('Layer $layerIndex is not using quantized KV cache');
      }
      final keyWeights = layer.borrowedKeys.weights.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedKeys.weights.shape[0],
          layer.borrowedKeys.weights.shape[1],
          layer.offset,
          layer.borrowedKeys.weights.shape[3],
        ],
      );
      final keyScales = layer.borrowedKeys.scales.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedKeys.scales.shape[0],
          layer.borrowedKeys.scales.shape[1],
          layer.offset,
          layer.borrowedKeys.scales.shape[3],
        ],
      );
      final keyBiases = layer.borrowedKeys.biases!.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedKeys.biases!.shape[0],
          layer.borrowedKeys.biases!.shape[1],
          layer.offset,
          layer.borrowedKeys.biases!.shape[3],
        ],
      );
      final valueWeights = layer.borrowedValues.weights.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedValues.weights.shape[0],
          layer.borrowedValues.weights.shape[1],
          layer.offset,
          layer.borrowedValues.weights.shape[3],
        ],
      );
      final valueScales = layer.borrowedValues.scales.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedValues.scales.shape[0],
          layer.borrowedValues.scales.shape[1],
          layer.offset,
          layer.borrowedValues.scales.shape[3],
        ],
      );
      final valueBiases = layer.borrowedValues.biases!.slice(
        start: [0, 0, 0, 0],
        stop: [
          layer.borrowedValues.biases!.shape[0],
          layer.borrowedValues.biases!.shape[1],
          layer.offset,
          layer.borrowedValues.biases!.shape[3],
        ],
      );
      return (
        firstToken: firstToken,
        keyWeights: keyWeights,
        keyScales: keyScales,
        keyBiases: keyBiases,
        valueWeights: valueWeights,
        valueScales: valueScales,
        valueBiases: valueBiases,
      );
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({
    List<int> expandedIds,
    MlxArray inputsEmbeds,
    MlxArray positionIds,
    int nextTextPosition,
  }) debugInputsEmbedsAndPositionIdsFromImage(
    List<int> promptIds,
    MlxArray imagePixels,
  ) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    return (
      expandedIds: expandedIds,
      inputsEmbeds: embeddings,
      positionIds: positionInfo.ids,
      nextTextPosition: positionInfo.nextTextPosition,
    );
  }

  MlxArray debugLmLayerOutputFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final seqLen = embeddings.shape[1];
      var h = embeddings;
      try {
        for (var i = 0; i <= layerIndex; i++) {
          final next = _decoderLayer(
            _lmLayers[i],
            h,
            seqLen,
            posIds,
            layerIndex: i,
            cache: cache.layers[i],
          );
          if (h != embeddings) h.close();
          h = next;
        }
        return h;
      } catch (_) {
        if (h != embeddings) h.close();
        rethrow;
      }
    } finally {
      posIds.close();
      cache.close();
    }
  }


  MlxArray debugLmAttentionOutputFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final seqLen = embeddings.shape[1];
      var h = embeddings;
      try {
        for (var i = 0; i <= layerIndex; i++) {
          final layer = _lmLayers[i];
          final norm1 = mx.fast.rmsNorm(
            h,
            weight: layer.inputNorm,
            eps: config.rmsNormEps,
          );
          if (i == layerIndex) {
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              seqLen,
              posIds,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            if (h != embeddings) h.close();
            return attnOut;
          }
          final attnOut = _lmAttention(
            layer.attention,
            norm1,
            seqLen,
            posIds,
            layerIndex: i,
            cache: cache.layers[i],
          );
          norm1.close();
          final h1 = mx.add(h, attnOut);
          attnOut.close();
          if (h != embeddings) h.close();
          final norm2 = mx.fast.rmsNorm(
            h1,
            weight: layer.postNorm,
            eps: config.rmsNormEps,
          );
          final mlpOut = _lmMlp(layer.mlp, norm2, seqLen);
          norm2.close();
          final next = mx.add(h1, mlpOut);
          mlpOut.close();
          h1.close();
          h = next;
        }
        throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
      } catch (_) {
        if (h != embeddings) h.close();
        rethrow;
      }
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({MlxArray q, MlxArray k, MlxArray v}) debugLmProjectedQkvFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    try {
      final layer = _lmLayers[layerIndex];
      final norm1 = mx.fast.rmsNorm(
        embeddings,
        weight: layer.inputNorm,
        eps: config.rmsNormEps,
      );
      embeddings.close();
      try {
        final seqLen = norm1.shape[1];
        final flat = norm1.reshape([seqLen, config.hiddenSize]);
        norm1.close();
        final qkv = _debugApplyQkvFlat(layer.attention, flat, seqLen);
        flat.close();
        return qkv;
      } catch (_) {
        norm1.close();
        rethrow;
      }
    } catch (_) {
      embeddings.close();
      rethrow;
    }
  }

  ({MlxArray q, MlxArray k, MlxArray v}) debugLmProjectedQkvApply(
    MlxArray hidden,
    int layerIndex,
  ) {
    final layer = _lmLayers[layerIndex];
    final seqLen = hidden.shape[1];
    final flat = hidden.reshape([seqLen, config.hiddenSize]);
    final qkv = _debugApplyQkvFlat(layer.attention, flat, seqLen);
    flat.close();
    return qkv;
  }

  ({MlxArray q, MlxArray k}) debugLmRopedQkFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    try {
      final layer = _lmLayers[layerIndex];
      final norm1 = mx.fast.rmsNorm(
        embeddings,
        weight: layer.inputNorm,
        eps: config.rmsNormEps,
      );
      embeddings.close();
      try {
        final seqLen = norm1.shape[1];
        final flat = norm1.reshape([seqLen, config.hiddenSize]);
        norm1.close();
        final qkv = _debugApplyQkvHeads(layer.attention, flat, seqLen);
        flat.close();
        final rope = _applyMrope(qkv.q, qkv.k, positionInfo.ids);
        qkv.q.close();
        qkv.k.close();
        qkv.v.close();
        return rope;
      } catch (_) {
        norm1.close();
        rethrow;
      }
    } finally {
      positionInfo.ids.close();
    }
  }

  MlxArray debugQuantizedAttentionFromRaw(
    MlxArray queries,
    MlxArray keyWeights,
    MlxArray keyScales,
    MlxArray keyBiases,
    MlxArray valueWeights,
    MlxArray valueScales,
    MlxArray valueBiases, {
    required double scale,
    required int groupSize,
    required int bits,
    Object? mask,
  }) {
    final keys = MlxQuantizedMatrix(keyWeights, keyScales, keyBiases);
    final values = MlxQuantizedMatrix(valueWeights, valueScales, valueBiases);
    return _quantizedScaledDotProductAttention(
      queries,
      keys,
      values,
      scale: scale,
      mask: mask,
      groupSize: groupSize,
      bits: bits,
    );
  }

  ({MlxArray scores, MlxArray probs}) debugQuantizedScoresFromRaw(
    MlxArray queries,
    MlxArray keyWeights,
    MlxArray keyScales,
    MlxArray keyBiases, {
    required double scale,
    required int groupSize,
    required int bits,
    Object? mask,
  }) {
    final qKeys = MlxQuantizedMatrix(keyWeights, keyScales, keyBiases);
    final batch = queries.shape[0];
    final numQHeads = queries.shape[1];
    final queryLen = queries.shape[2];
    final dim = queries.shape[3];
    final numKvHeads = qKeys.weights.shape[1];
    final repeats = numQHeads ~/ numKvHeads;

    final scaleArray = MlxArray.fromFloat32List([scale], shape: [1]).astype(
      queries.dtype,
    );
    final scaledQueries = queries * scaleArray;
    scaleArray.close();

    MlxArray qMat = scaledQueries;
    MlxQuantizedMatrix keysMat = qKeys;
    var closeKeyTemps = false;
    if (repeats > 1) {
      qMat = scaledQueries.reshape([batch, numKvHeads, repeats, queryLen, dim]);
      keysMat = MlxQuantizedMatrix(
        qKeys.weights.expandDims(-3),
        qKeys.scales.expandDims(-3),
        qKeys.biases?.expandDims(-3),
      );
      closeKeyTemps = true;
    }

    try {
      var scores = mx.quant.matmul(
        qMat,
        keysMat,
        transpose: true,
        groupSize: groupSize,
        bits: bits,
      );
      final masked = _applyAttentionMask(scores, mask);
      if (masked != scores) {
        scores.close();
        scores = masked;
      }
      final probs = mx.softmax(scores, axis: -1, precise: true);
      return (scores: scores, probs: probs);
    } finally {
      if (scaledQueries != queries) {
        scaledQueries.close();
      }
      if (qMat != scaledQueries) {
        qMat.close();
      }
      if (closeKeyTemps) {
        keysMat.close();
      }
    }
  }

  MlxArray debugLmNorm1FromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    try {
      final layer = _lmLayers[layerIndex];
      return mx.fast.rmsNorm(
        embeddings,
        weight: layer.inputNorm,
        eps: config.rmsNormEps,
      );
    } finally {
      embeddings.close();
    }
  }

  MlxArray debugLmInputNormWeight(int layerIndex) => _lmLayers[layerIndex].inputNorm;

  MlxArray debugLmNorm1ManualFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    try {
      final layer = _lmLayers[layerIndex];
      final x = embeddings.astype(MlxDType.MLX_FLOAT32);
      embeddings.close();
      final sq = MlxMore.square(x);
      final meanSq = mx.mean(sq, axis: x.ndim - 1, keepDims: true);
      sq.close();
      final eps = MlxArray.full([], config.rmsNormEps, dtype: MlxDType.MLX_FLOAT32);
      final denom = MlxMore.sqrt(meanSq + eps);
      meanSq.close();
      eps.close();
      final unit = x / denom;
      x.close();
      denom.close();
      final weight = layer.inputNorm.astype(MlxDType.MLX_FLOAT32).reshape([
        1,
        1,
        config.hiddenSize,
      ]);
      final out = unit * weight;
      unit.close();
      weight.close();
      return out;
    } catch (_) {
      embeddings.close();
      rethrow;
    }
  }

  MlxArray debugLmPostNorm(MlxArray hidden, int layerIndex) {
    final layer = _lmLayers[layerIndex];
    return mx.fast.rmsNorm(
      hidden,
      weight: layer.postNorm,
      eps: config.rmsNormEps,
    );
  }

  MlxArray debugLmMlpApply(MlxArray hidden, int layerIndex, int seqLen) {
    return _lmMlp(_lmLayers[layerIndex].mlp, hidden, seqLen);
  }


  MlxArray debugLmPostNormWeight(int layerIndex) => _lmLayers[layerIndex].postNorm;

  ({
    int firstToken,
    MlxArray keyNorms,
    MlxArray keyIndices,
    MlxArray valueNorms,
    MlxArray valueIndices,
  }) debugSecondDecodeLayerCacheStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      final second = _forwardWithCache(stepArr, stepPos, cache);
      second.close();
      stepArr.close();
      stepPos.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _TurboQuantKvCache) {
        throw StateError('Layer $layerIndex is not using TurboQuant cache');
      }
      final keyState = _turboSliceStateRange(layer._keys!, 0, layer.offset);
      final valueState = _turboSliceStateRange(layer._values!, 0, layer.offset);
      return (
        firstToken: firstToken,
        keyNorms: keyState.norms,
        keyIndices: keyState.indices,
        valueNorms: valueState.norms,
        valueIndices: valueState.indices,
      );
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray keys, MlxArray values})
  debugSecondDecodeDenseLayerStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      final second = _forwardWithCache(stepArr, stepPos, cache);
      second.close();
      stepArr.close();
      stepPos.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _KvCache) {
        throw StateError('Layer $layerIndex is not using dense KV cache');
      }
      final keys = layer._keys!.slice(
        start: [0, 0, 0, 0],
        stop: [1, layer.numKvHeads, layer.offset, layer.headDim],
      );
      final values = layer._values!.slice(
        start: [0, 0, 0, 0],
        stop: [1, layer.numKvHeads, layer.offset, layer.headDim],
      );
      return (firstToken: firstToken, keys: keys, values: values);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray layerOutput})
  debugSecondDecodeLayerOutputFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final next = _decoderLayer(
              _lmLayers[i],
              h,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            if (h != hidden) h.close();
            h = next;
            if (i == layerIndex) {
              return (firstToken: firstToken, layerOutput: h);
            }
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray attentionOutput})
  debugSecondDecodeAttentionOutputFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            if (i == layerIndex) {
              final attnOut = _lmAttention(
                layer.attention,
                norm1,
                1,
                stepPos,
                layerIndex: i,
                cache: cache.layers[i],
              );
              norm1.close();
              if (h != hidden) h.close();
              return (firstToken: firstToken, attentionOutput: attnOut);
            }
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            final h1 = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            final norm2 = mx.fast.rmsNorm(
              h1,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h1, mlpOut);
            mlpOut.close();
            h1.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray finalNorm})
  debugSecondDecodeFinalNormFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final next = _decoderLayer(
              _lmLayers[i],
              h,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            if (h != hidden) h.close();
            h = next;
          }
          final norm = mx.fast.rmsNorm(
            h,
            weight: _finalNorm,
            eps: config.rmsNormEps,
          );
          h.close();
          return (firstToken: firstToken, finalNorm: norm);
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  MlxArray debugLmHeadApply(MlxArray hidden) {
    final last = hidden.reshape([1, config.hiddenSize]);
    final linear = config.tieWordEmbeddings ? _embedWeights : _lmHead!;
    final logits = linear.apply(last);
    last.close();
    return logits.reshape([1, config.vocabSize]);
  }

  MlxArray debugFinalNormApply(MlxArray hidden) => mx.fast.rmsNorm(
    hidden,
    weight: _finalNorm,
    eps: config.rmsNormEps,
  );

  MlxArray debugFinalNormWeight() => _finalNorm;

  ({int firstToken, MlxArray norm2})
  debugSecondDecodePostNormFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i <= layerIndex; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            final h1 = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            h = h1;
            final norm2 = mx.fast.rmsNorm(
              h,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            if (i == layerIndex) {
              return (firstToken: firstToken, norm2: norm2);
            }
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h, mlpOut);
            mlpOut.close();
            h.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray norm1}) debugSecondDecodeNorm1FromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final hidden = _embed(stepArr);
      stepArr.close();
      final norm1 = mx.fast.rmsNorm(
        hidden,
        weight: _lmLayers[layerIndex].inputNorm,
        eps: config.rmsNormEps,
      );
      hidden.close();
      return (firstToken: firstToken, norm1: norm1);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray q, MlxArray k, MlxArray v})
  debugSecondDecodeProjectedQkvFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final hidden = _embed(stepArr);
      stepArr.close();
      final layer = _lmLayers[layerIndex];
      final norm1 = mx.fast.rmsNorm(
        hidden,
        weight: layer.inputNorm,
        eps: config.rmsNormEps,
      );
      hidden.close();
      final flat = norm1.reshape([1, config.hiddenSize]);
      norm1.close();
      final qkv = _debugApplyQkvFlat(layer.attention, flat, 1);
      flat.close();
      return (firstToken: firstToken, q: qkv.q, k: qkv.k, v: qkv.v);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray q, MlxArray k})
  debugSecondDecodeRopedQkFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();
      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      final hidden = _embed(stepArr);
      stepArr.close();
      final layer = _lmLayers[layerIndex];
      final norm1 = mx.fast.rmsNorm(
        hidden,
        weight: layer.inputNorm,
        eps: config.rmsNormEps,
      );
      hidden.close();
      final flat = norm1.reshape([1, config.hiddenSize]);
      norm1.close();
      final qkv = _debugApplyQkvHeads(layer.attention, flat, 1);
      flat.close();
      final rope = _applyMrope(qkv.q, qkv.k, stepPos);
      qkv.q.close();
      qkv.k.close();
      qkv.v.close();
      stepPos.close();
      return (firstToken: firstToken, q: rope.q, k: rope.k);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({
    List<int> expandedIds,
    MlxArray inputsEmbeds,
    MlxArray positionIds,
    int nextTextPosition,
  }) debugInputsEmbedsAndPositionIdsFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    return (
      expandedIds: expandedIds,
      inputsEmbeds: embeddings,
      positionIds: positionInfo.ids,
      nextTextPosition: positionInfo.nextTextPosition,
    );
  }

  ({
    MlxArray keyNorms,
    MlxArray keyIndices,
    MlxArray valueNorms,
    MlxArray valueIndices,
  }) debugPrefillLayerCacheStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      logits.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _TurboQuantKvCache) {
        throw StateError('Layer $layerIndex is not using TurboQuant cache');
      }
      final keyState = _turboSliceStateRange(layer._keys!, 0, layer.offset);
      final valueState = _turboSliceStateRange(layer._values!, 0, layer.offset);
      return (
        keyNorms: keyState.norms,
        keyIndices: keyState.indices,
        valueNorms: valueState.norms,
        valueIndices: valueState.indices,
      );
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({MlxArray keys, MlxArray values}) debugPrefillDenseLayerCacheStateFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache._(
      List.generate(
        config.numHiddenLayers,
        (_) => _KvCache(
          numKvHeads: config.numKeyValueHeads,
          headDim: config.headDim,
          maxSeqLen: config.maxKvCacheSeqLenForCurrentPlatform,
        ),
      ),
      kvBits: -1,
      kvGroupSize: config.kvCacheQuantGroupSizeForCurrentPlatform,
      quantizedStart: 1 << 30,
      kvScheme: 'turboquant',
      turboBits: config.turboQuantBitsForCurrentPlatform,
      turboStart: 1 << 30,
    );
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      logits.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _KvCache) {
        throw StateError('Layer $layerIndex is not using dense KV cache');
      }
      final keys = layer._keys!.slice(
        start: [0, 0, 0, 0],
        stop: [1, layer.numKvHeads, layer.offset, layer.headDim],
      );
      final values = layer._values!.slice(
        start: [0, 0, 0, 0],
        stop: [1, layer.numKvHeads, layer.offset, layer.headDim],
      );
      return (keys: keys, values: values);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({MlxArray keys, MlxArray values}) debugPrefillDenseLayerCacheStateFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    required int layerIndex,
  }) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    final cache = _ModelCache._(
      List.generate(
        config.numHiddenLayers,
        (_) => _KvCache(
          numKvHeads: config.numKeyValueHeads,
          headDim: config.headDim,
          maxSeqLen: config.maxKvCacheSeqLenForCurrentPlatform,
        ),
      ),
      kvBits: -1,
      kvGroupSize: config.kvCacheQuantGroupSizeForCurrentPlatform,
      quantizedStart: 1 << 30,
      kvScheme: 'turboquant',
      turboBits: config.turboQuantBitsForCurrentPlatform,
      turboStart: 1 << 30,
    );
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      logits.close();
      final layer = cache.layers[layerIndex];
      if (layer is! _KvCache) {
        throw StateError('Layer $layerIndex is not using dense KV cache');
      }
      final keys = layer._keys!.slice(
        start: [0, 0, 0, 0],
        stop: [1, layer.numKvHeads, layer.offset, layer.headDim],
      );
      final values = layer._values!.slice(
        start: [0, 0, 0, 0],
        stop: [1, layer.numKvHeads, layer.offset, layer.headDim],
      );
      return (keys: keys, values: values);
    } finally {
      embeddings.close();
      posIds.close();
      cache.close();
    }
  }

  ({int firstToken, MlxArray layerOutput}) debugSecondDecodeLayerOutputFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    required int layerIndex,
  }) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();

      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: cache.offset);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final next = _decoderLayer(
              _lmLayers[i],
              h,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            if (h != hidden) h.close();
            h = next;
            if (i == layerIndex) {
              return (firstToken: firstToken, layerOutput: h);
            }
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      cache.close();
    }
  }

  ({int firstToken, MlxArray attentionOutput})
  debugSecondDecodeAttentionOutputFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    required int layerIndex,
  }) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();

      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: cache.offset);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            if (i == layerIndex) {
              if (h != hidden) {
                h.close();
              } else {
                hidden.close();
              }
              return (firstToken: firstToken, attentionOutput: attnOut);
            }
            final residual = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            h = residual;

            final norm2 = mx.fast.rmsNorm(
              h,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h, mlpOut);
            mlpOut.close();
            h.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      cache.close();
    }
  }

  ({int firstToken, MlxArray directOutput, MlxArray splitOutput})
  debugSecondDecodeAttentionOutputsFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    required int layerIndex,
  }) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();

      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: cache.offset);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            if (i == layerIndex) {
              final numKvHeads = config.numKeyValueHeads;
              final headDim = config.headDim;
              final flat = norm1.reshape([1, config.hiddenSize]);
              final qkv = _debugApplyQkvHeads(layer.attention, flat, 1);
              flat.close();
              norm1.close();
              final rope = _applyMrope(qkv.q, qkv.k, stepPos);
              qkv.q.close();
              qkv.k.close();
              final qRope = rope.q;
              final kRope = rope.k;
              final v = qkv.v;
              final layerCache = cache.layers[i];
              if (layerCache is! _TurboQuantKvCache) {
                qRope.close();
                kRope.close();
                v.close();
                throw StateError('Layer $i is not using TurboQuant cache');
              }
              layerCache.update(kRope, v);
              final direct = layerCache.quantizedAttention(
                qRope,
                scale: 1.0 / math.sqrt(headDim.toDouble()),
                mask: null,
              );
              final keyState = _turboSliceStateRange(
                layerCache._keys!,
                0,
                layerCache.offset,
              );
              final valueState = _turboSliceStateRange(
                layerCache._values!,
                0,
                layerCache.offset,
              );
              try {
                final batch = qRope.shape[0];
                final repeats = qRope.shape[1] ~/ numKvHeads;
                final scaleArr = MlxArray.full(
                  [],
                  1.0 / math.sqrt(headDim.toDouble()),
                  dtype: qRope.dtype,
                );
                final scaled = qRope * scaleArr;
                scaleArr.close();
                final grouped = scaled.reshape([batch, numKvHeads, repeats, 1, headDim]);
                scaled.close();
                final prepared = layerCache._keyCodec.prepareQueries(grouped);
                grouped.close();
                final scores = layerCache._keyCodec.scorePrepared(prepared, keyState);
                prepared.close();
                final stats = _turboMseWeightedSumStatsFromScores(
                  scores,
                  valueState,
                  layerCache._valueCodec,
                );
                scores.close();
                if (stats == null) {
                  direct.close();
                  throw StateError('TurboQuant split stats path unavailable');
                }
                final eps = MlxArray.full([], 1e-6, dtype: MlxDType.MLX_FLOAT32);
                final denomExpanded = stats.denom.expandDims(stats.denom.ndim);
                final denomSafe = mx.maximum(denomExpanded, eps);
                denomExpanded.close();
                eps.close();
                final split = stats.output / denomSafe;
                denomSafe.close();
                stats.output.close();
                stats.denom.close();
                stats.maxScores.close();
                qRope.close();
                if (h != hidden) {
                  h.close();
                } else {
                  hidden.close();
                }
                return (
                  firstToken: firstToken,
                  directOutput: direct,
                  splitOutput: split,
                );
              } finally {
                keyState.close();
                valueState.close();
              }
            }
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            final residual = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            h = residual;

            final norm2 = mx.fast.rmsNorm(
              h,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h, mlpOut);
            mlpOut.close();
            h.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      cache.close();
    }
  }

  ({int firstToken, MlxArray directOutput, MlxArray splitOutput})
  debugSecondDecodeAttentionOutputsFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();

      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: cache.offset);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            if (i == layerIndex) {
              final numKvHeads = config.numKeyValueHeads;
              final headDim = config.headDim;
              final flat = norm1.reshape([1, config.hiddenSize]);
              final qkv = _debugApplyQkvHeads(layer.attention, flat, 1);
              flat.close();
              norm1.close();
              final rope = _applyMrope(qkv.q, qkv.k, stepPos);
              qkv.q.close();
              qkv.k.close();
              final qRope = rope.q;
              final kRope = rope.k;
              final v = qkv.v;
              final layerCache = cache.layers[i];
              if (layerCache is! _TurboQuantKvCache) {
                qRope.close();
                kRope.close();
                v.close();
                throw StateError('Layer $i is not using TurboQuant cache');
              }
              layerCache.update(kRope, v);
              final direct = layerCache.quantizedAttention(
                qRope,
                scale: 1.0 / math.sqrt(headDim.toDouble()),
                mask: null,
              );
              final keyState = _turboSliceStateRange(
                layerCache._keys!,
                0,
                layerCache.offset,
              );
              final valueState = _turboSliceStateRange(
                layerCache._values!,
                0,
                layerCache.offset,
              );
              try {
                final batch = qRope.shape[0];
                final repeats = qRope.shape[1] ~/ numKvHeads;
                final scaleArr = MlxArray.full(
                  [],
                  1.0 / math.sqrt(headDim.toDouble()),
                  dtype: qRope.dtype,
                );
                final scaled = qRope * scaleArr;
                scaleArr.close();
                final grouped = scaled.reshape([
                  batch,
                  numKvHeads,
                  repeats,
                  1,
                  headDim,
                ]);
                scaled.close();
                final prepared = layerCache._keyCodec.prepareQueries(grouped);
                grouped.close();
                final scores = layerCache._keyCodec.scorePrepared(prepared, keyState);
                prepared.close();
                final stats = _turboMseWeightedSumStatsFromScores(
                  scores,
                  valueState,
                  layerCache._valueCodec,
                );
                scores.close();
                if (stats == null) {
                  direct.close();
                  throw StateError('TurboQuant split stats path unavailable');
                }
                final eps = MlxArray.full([], 1e-6, dtype: MlxDType.MLX_FLOAT32);
                final denomExpanded = stats.denom.expandDims(stats.denom.ndim);
                final denomSafe = mx.maximum(denomExpanded, eps);
                denomExpanded.close();
                eps.close();
                final split = stats.output / denomSafe;
                denomSafe.close();
                stats.output.close();
                stats.denom.close();
                stats.maxScores.close();
                qRope.close();
                if (h != hidden) {
                  h.close();
                } else {
                  hidden.close();
                }
                return (
                  firstToken: firstToken,
                  directOutput: direct,
                  splitOutput: split,
                );
              } finally {
                keyState.close();
                valueState.close();
              }
            }
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            final residual = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            h = residual;

            final norm2 = mx.fast.rmsNorm(
              h,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h, mlpOut);
            mlpOut.close();
            h.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      cache.close();
    }
  }

  ({int firstToken, MlxArray directOutput, MlxArray fusedOutput})
  debugSecondDecodeAttentionDirectAndFusedFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();

      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: positionInfo.nextTextPosition);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            if (i == layerIndex) {
              final numKvHeads = config.numKeyValueHeads;
              final headDim = config.headDim;
              final flat = norm1.reshape([1, config.hiddenSize]);
              final qkv = _debugApplyQkvHeads(layer.attention, flat, 1);
              flat.close();
              norm1.close();
              final rope = _applyMrope(qkv.q, qkv.k, stepPos);
              qkv.q.close();
              qkv.k.close();
              final qRope = rope.q;
              final kRope = rope.k;
              final v = qkv.v;
              final layerCache = cache.layers[i];
              if (layerCache is! _TurboQuantKvCache) {
                qRope.close();
                kRope.close();
                v.close();
                throw StateError('Layer $i is not using TurboQuant cache');
              }
              layerCache.update(kRope, v);
              final direct = layerCache.quantizedAttention(
                qRope,
                scale: 1.0 / math.sqrt(headDim.toDouble()),
                mask: null,
              );
              final keyState = _turboSliceStateRange(
                layerCache._keys!,
                0,
                layerCache.offset,
              );
              final valueState = _turboSliceStateRange(
                layerCache._values!,
                0,
                layerCache.offset,
              );
              try {
                final fused = _turboMseFusedDecodeAttention(
                  qRope,
                  keyState,
                  valueState,
                  keyCodec: layerCache._keyCodec,
                  valueCodec: layerCache._valueCodec,
                  repeats: qRope.shape[1] ~/ numKvHeads,
                  scale: 1.0 / math.sqrt(headDim.toDouble()),
                );
                qRope.close();
                if (fused == null) {
                  direct.close();
                  throw StateError('TurboQuant fused decode path unavailable');
                }
                if (h != hidden) {
                  h.close();
                } else {
                  hidden.close();
                }
                return (
                  firstToken: firstToken,
                  directOutput: direct,
                  fusedOutput: fused,
                );
              } finally {
                keyState.close();
                valueState.close();
              }
            }
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            final residual = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            h = residual;

            final norm2 = _lmRmsNormCompat(
              h,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h, mlpOut);
            mlpOut.close();
            h.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      cache.close();
    }
  }

  ({int firstToken, MlxArray fastScores, MlxArray slowScores})
  debugSecondDecodeScoresFromVisionFeatures(
    List<int> promptIds,
    MlxArray imageHidden, {
    required int gridHeight,
    required int gridWidth,
    required int layerIndex,
  }) {
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      gridHeight,
      gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    final cache = _ModelCache.create(config: config);
    final oldDisableFastScore = PaddleOcrVlDebugOverrides.turboDisableFastScore;
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();

      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: cache.offset);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            if (i == layerIndex) {
              final numKvHeads = config.numKeyValueHeads;
              final headDim = config.headDim;
              final flat = norm1.reshape([1, config.hiddenSize]);
              final qkv = _debugApplyQkvHeads(layer.attention, flat, 1);
              flat.close();
              norm1.close();
              final rope = _applyMrope(qkv.q, qkv.k, stepPos);
              qkv.q.close();
              qkv.k.close();
              final qRope = rope.q;
              final kRope = rope.k;
              final v = qkv.v;
              final layerCache = cache.layers[i];
              if (layerCache is! _TurboQuantKvCache) {
                qRope.close();
                kRope.close();
                v.close();
                throw StateError('Layer $i is not using TurboQuant cache');
              }
              layerCache.update(kRope, v);
              final keyState = _turboSliceStateRange(
                layerCache._keys!,
                0,
                layerCache.offset,
              );
              try {
                final repeats = qRope.shape[1] ~/ numKvHeads;
                final scaleArr = MlxArray.full(
                  [],
                  1.0 / math.sqrt(headDim.toDouble()),
                  dtype: qRope.dtype,
                );
                final scaled = qRope * scaleArr;
                scaleArr.close();
                final grouped = scaled.reshape([
                  qRope.shape[0],
                  numKvHeads,
                  repeats,
                  1,
                  headDim,
                ]);
                scaled.close();
                final prepared = layerCache._keyCodec.prepareQueries(grouped);
                grouped.close();
                PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
                final fastScores = layerCache._keyCodec.scorePrepared(
                  prepared,
                  keyState,
                );
                PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
                final slowScores = layerCache._keyCodec.scorePrepared(
                  prepared,
                  keyState,
                );
                prepared.close();
                qRope.close();
                if (h != hidden) {
                  h.close();
                } else {
                  hidden.close();
                }
                return (
                  firstToken: firstToken,
                  fastScores: fastScores,
                  slowScores: slowScores,
                );
              } finally {
                keyState.close();
              }
            }
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            final residual = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            h = residual;

            final norm2 = mx.fast.rmsNorm(
              h,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h, mlpOut);
            mlpOut.close();
            h.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      PaddleOcrVlDebugOverrides.turboDisableFastScore = oldDisableFastScore;
      cache.close();
    }
  }

  ({
    int firstToken,
    MlxArray keyNorms,
    MlxArray keyIndices,
    MlxArray valueNorms,
    MlxArray valueIndices,
  }) debugSecondDecodeLayerCacheStateFromImage(
    List<int> promptIds,
    MlxArray imagePixels, {
    required int layerIndex,
  }) {
    final imageEncoding = _encodeImage(imagePixels);
    final imageHidden = imageEncoding.hidden;
    final numImageTokens = imageHidden.shape[0];
    final imageTokenCountInPrompt = promptIds
        .where((id) => id == config.imageTokenId)
        .length;
    final expandedIds = imageTokenCountInPrompt == numImageTokens
        ? List<int>.from(promptIds)
        : _expandImageTokens(promptIds, numImageTokens);
    final positionInfo = _multimodalPositionIds(
      expandedIds,
      imageEncoding.gridHeight,
      imageEncoding.gridWidth,
    );
    final posIds = positionInfo.ids;
    final embeddings = _buildMultimodalEmbedding(expandedIds, imageHidden);
    imageHidden.close();
    final cache = _ModelCache.create(config: config);
    try {
      final logits = _prefillFromEmbeddingWithCache(embeddings, posIds, cache);
      embeddings.close();
      posIds.close();
      final firstToken = _nextTokenFromLogits(logits);
      logits.close();

      final stepArr = MlxArray.fromInt32List([firstToken], shape: [1, 1]);
      final stepPos = _textPositionIds(1, offset: cache.offset);
      try {
        final hidden = _embed(stepArr);
        var h = hidden;
        try {
          for (var i = 0; i < _lmLayers.length; i++) {
            final layer = _lmLayers[i];
            final norm1 = mx.fast.rmsNorm(
              h,
              weight: layer.inputNorm,
              eps: config.rmsNormEps,
            );
            if (i == layerIndex) {
              final flat = norm1.reshape([1, config.hiddenSize]);
              final qkv = _debugApplyQkvHeads(layer.attention, flat, 1);
              flat.close();
              norm1.close();
              final rope = _applyMrope(qkv.q, qkv.k, stepPos);
              qkv.q.close();
              qkv.k.close();
              final kRope = rope.k;
              final v = qkv.v;
              final layerCache = cache.layers[i];
              if (layerCache is! _TurboQuantKvCache) {
                kRope.close();
                v.close();
                throw StateError('Layer $i is not using TurboQuant cache');
              }
              layerCache.update(kRope, v);
              final keyState = _turboSliceStateRange(
                layerCache._keys!,
                0,
                layerCache.offset,
              );
              final valueState = _turboSliceStateRange(
                layerCache._values!,
                0,
                layerCache.offset,
              );
              if (h != hidden) {
                h.close();
              } else {
                hidden.close();
              }
              return (
                firstToken: firstToken,
                keyNorms: keyState.norms,
                keyIndices: keyState.indices,
                valueNorms: valueState.norms,
                valueIndices: valueState.indices,
              );
            }
            final attnOut = _lmAttention(
              layer.attention,
              norm1,
              1,
              stepPos,
              layerIndex: i,
              cache: cache.layers[i],
            );
            norm1.close();
            final residual = mx.add(h, attnOut);
            attnOut.close();
            if (h != hidden) h.close();
            h = residual;

            final norm2 = mx.fast.rmsNorm(
              h,
              weight: layer.postNorm,
              eps: config.rmsNormEps,
            );
            final mlpOut = _lmMlp(layer.mlp, norm2, 1);
            norm2.close();
            final next = mx.add(h, mlpOut);
            mlpOut.close();
            h.close();
            h = next;
          }
          throw RangeError.index(layerIndex, _lmLayers, 'layerIndex');
        } catch (_) {
          if (h != hidden) h.close();
          rethrow;
        }
      } finally {
        stepArr.close();
        stepPos.close();
      }
    } finally {
      cache.close();
    }
  }
}
