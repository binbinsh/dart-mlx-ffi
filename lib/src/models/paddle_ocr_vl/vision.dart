part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// Vision encoder (ViT) + spatial-merge projector forward pass
// ---------------------------------------------------------------------------

extension PaddleOcrVlVision on PaddleOcrVlRunner {
  static const int _visionCacheEntryLimit = 2;

  void _storeBoundedVisionArrayCache(
    Map<String, MlxArray> cache,
    String key,
    MlxArray value,
  ) {
    if (!cache.containsKey(key) && cache.length >= _visionCacheEntryLimit) {
      final oldestKey = cache.keys.first;
      cache.remove(oldestKey)?.close();
    }
    cache[key] = value;
  }

  ({MlxArray value, bool owned}) _borrowVisionRotaryPosEmbedding(
    int gridH,
    int gridW,
    MlxDType dtype,
  ) {
    if (Platform.isIOS) {
      return (
        value: _buildVisionRotaryPosEmbedding(gridH, gridW, dtype),
        owned: true,
      );
    }
    final key = '$gridH:$gridW:${dtype.name}';
    final cached = _visionRotaryEmbeddingArrayCache[key];
    if (cached != null) {
      return (value: cached, owned: false);
    }
    final created = _buildVisionRotaryPosEmbedding(gridH, gridW, dtype);
    _storeBoundedVisionArrayCache(_visionRotaryEmbeddingArrayCache, key, created);
    return (value: created, owned: false);
  }

  ({MlxArray value, bool owned}) _borrowInterpolatedVisionPositionEmbedding(
    int gridH,
    int gridW,
    MlxDType dtype,
  ) {
    if (Platform.isIOS) {
      return (
        value: _interpolateVisionPositionEmbedding(gridH, gridW, dtype),
        owned: true,
      );
    }
    final key = '$gridH:$gridW:${dtype.name}';
    final cached = _visionInterpolatedPositionEmbeddingArrayCache[key];
    if (cached != null) {
      return (value: cached, owned: false);
    }
    final created = _interpolateVisionPositionEmbedding(gridH, gridW, dtype);
    _storeBoundedVisionArrayCache(
      _visionInterpolatedPositionEmbeddingArrayCache,
      key,
      created,
    );
    return (value: created, owned: false);
  }

  ({MlxArray hidden, int gridHeight, int gridWidth}) _encodeImageFromPatchHidden(
    MlxArray hidden,
    int gridH,
    int gridW, {
    void Function(String message)? onStage,
  }) {
    final vCfg = config._vision;

    final posEmbedRef = _borrowInterpolatedVisionPositionEmbedding(
      gridH,
      gridW,
      hidden.dtype,
    );
    final posEmbed = posEmbedRef.value;
    final withPos = mx.add(hidden, posEmbed);
    hidden.close();
    if (posEmbedRef.owned) {
      posEmbed.close();
    }
    hidden = withPos;
    if (config.enableVisionLayerwiseEvalForCurrentPlatform) {
      MlxRuntime.evalAll([hidden]);
    }
    onStage?.call('encodeImage: embeddings ready shape=${hidden.shape}');

    final rotaryRef = _borrowVisionRotaryPosEmbedding(
      gridH,
      gridW,
      MlxDType.MLX_FLOAT32,
    );
    final baseRotaryPosEmb = rotaryRef.value;

    final evalBatch = config.visionEvalBatchSizeForCurrentPlatform;
    for (var i = 0; i < _visionWeights.blocks.length; i++) {
      hidden = _visionBlock(
        _visionWeights.blocks[i],
        hidden,
        vCfg,
        baseRotaryPosEmb,
      );
      final isLastLayer = i + 1 == _visionWeights.blocks.length;
      if (config.enableVisionLayerwiseEvalForCurrentPlatform &&
          (isLastLayer || (i + 1) % evalBatch == 0)) {
        MlxRuntime.evalAll([hidden]);
      }
      if ((i + 1) % 3 == 0 || i + 1 == _visionWeights.blocks.length) {
        onStage?.call(
          'encodeImage: vision layer ${i + 1}/${_visionWeights.blocks.length}',
        );
      }
    }
    if (rotaryRef.owned) {
      baseRotaryPosEmb.close();
    }

    final postNorm = _visionLayerNorm(
      hidden,
      weight: _visionWeights.postLayerNormWeight,
      bias: _visionWeights.postLayerNormBias,
      eps: vCfg.layerNormEps,
    );
    hidden.close();
    if (config.enableVisionLayerwiseEvalForCurrentPlatform) {
      MlxRuntime.evalAll([postNorm]);
    }

    final merged = _spatialMergeProject(postNorm, gridH, gridW, vCfg);
    postNorm.close();
    return (hidden: merged, gridHeight: gridH, gridWidth: gridW);
  }

  /// Encode a pre-processed image tensor into LM-space hidden states.
  ///
  /// [pixels] has shape `[1, H, W, C]` (NHWC, float16/float32), already
  /// normalised to the model's expected range.
  ///
  /// Returns merged hidden states plus the patch grid dimensions before
  /// spatial merging.
  ({MlxArray hidden, int gridHeight, int gridWidth}) _encodeImage(
    MlxArray pixels, {
    void Function(String message)? onStage,
  }) {
    final vCfg = config._vision;
    final targetPixelDType = _visionWeights.patchEmbedWeight.dtype;
    final visionPixels = pixels.dtype == targetPixelDType
        ? pixels
        : pixels.astype(targetPixelDType);

    // 1. Patch embedding on pre-split 14x14 tiles, matching upstream
    final patchInfo = _patchifyVisionImage(visionPixels, vCfg.patchSize);
    final gridH = patchInfo.gridHeight;
    final gridW = patchInfo.gridWidth;
    final patches = patchInfo.patches;
    final flatPatches = patches
        .reshape([gridH * gridW, visionPixels.shape[3], vCfg.patchSize, vCfg.patchSize])
        .transposeAxes([0, 2, 3, 1]);
    patches.close();
    if (!identical(visionPixels, pixels)) {
      visionPixels.close();
    }
    final patchOut = mx.conv2d(flatPatches, _visionWeights.patchEmbedWeight);
    flatPatches.close();
    var hidden = patchOut.reshape([gridH * gridW, vCfg.hiddenSize]);
    patchOut.close();

    // Add patch embedding bias if present
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
    return _encodeImageFromPatchHidden(
      hidden,
      gridH,
      gridW,
      onStage: onStage,
    );
  }

  ({MlxArray hidden, int gridHeight, int gridWidth}) _encodeImageFromPixelValues(
    MlxArray pixelValues,
    MlxArray imageGridThw, {
    void Function(String message)? onStage,
  }) {
    final gridValues = imageGridThw.toList().cast<num>().map((n) => n.toInt()).toList();
    if (gridValues.length != 3) {
      throw ArgumentError('imageGridThw must have exactly 3 values [t, h, w].');
    }
    final temporal = gridValues[0];
    final gridH = gridValues[1];
    final gridW = gridValues[2];
    final seqLen = temporal * gridH * gridW;
    if (seqLen != pixelValues.shape[1]) {
      throw ArgumentError(
        'pixelValues/imageGridThw mismatch: seqLen=$seqLen shape=${pixelValues.shape}',
      );
    }
    final patchSize = config.visionPatchSize;
    final targetPixelDType = _visionWeights.patchEmbedWeight.dtype;
    final visionPixels = pixelValues.dtype == targetPixelDType
        ? pixelValues
        : pixelValues.astype(targetPixelDType);
    final flatPatches = visionPixels
        .reshape([seqLen, visionPixels.shape[2], patchSize, patchSize])
        .transposeAxes([0, 2, 3, 1]);
    if (!identical(visionPixels, pixelValues)) {
      visionPixels.close();
    }
    final patchOut = mx.conv2d(flatPatches, _visionWeights.patchEmbedWeight);
    flatPatches.close();
    var hidden = patchOut.reshape([seqLen, config._vision.hiddenSize]);
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
    return _encodeImageFromPatchHidden(
      hidden,
      gridH,
      gridW,
      onStage: onStage,
    );
  }

  ({MlxArray patches, int gridHeight, int gridWidth}) _patchifyVisionImage(
    MlxArray pixels,
    int patchSize,
  ) {
    final gridH = pixels.shape[1] ~/ patchSize;
    final gridW = pixels.shape[2] ~/ patchSize;
    final channels = pixels.shape[3];
    final reshaped = pixels.reshape([
      pixels.shape[0],
      gridH,
      patchSize,
      gridW,
      patchSize,
      channels,
    ]);
    final transposed = reshaped.transposeAxes([0, 1, 3, 5, 2, 4]);
    reshaped.close();
    final patches = transposed.reshape([
      pixels.shape[0],
      gridH * gridW,
      channels,
      patchSize,
      patchSize,
    ]);
    transposed.close();
    return (patches: patches, gridHeight: gridH, gridWidth: gridW);
  }

  // -----------------------------------------------------------------------
  // Position embedding + vision rotary helpers
  // -----------------------------------------------------------------------

  Float32List _fullVisionPositionEmbedding() {
    final cached = _visionPositionEmbeddingCache;
    if (cached != null) return cached;

    final posEmbed = _visionWeights.positionEmbedding;
    late final Float32List dense;
    if (posEmbed case final _QuantLinear q) {
      final rowCount = q.scales.shape[0];
      final rowIds = Int32List(rowCount);
      for (var i = 0; i < rowCount; i++) {
        rowIds[i] = i;
      }
      final rows = MlxArray.fromInt32List(rowIds, shape: [rowCount]);
      try {
        final rowsW = q.weight.take(rows, axis: 0);
        final rowsS = q.scales.take(rows, axis: 0);
        final rowsB = q.biases?.take(rows, axis: 0);
        final gathered = MlxQuantizedMatrix(rowsW, rowsS, rowsB);
        try {
          final dequantized = mx.quant.dequantize(
            gathered,
            groupSize: q.quantSpec.groupSize,
            bits: q.quantSpec.bits,
            mode: q.quantSpec.mode,
            dtype: MlxDType.MLX_FLOAT32,
          );
          try {
            dense = dequantized.toFloat32List();
          } finally {
            dequantized.close();
          }
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      } finally {
        rows.close();
      }
    } else if (posEmbed case final _DenseLinear d) {
      dense = d.weight.toFloat32List();
    } else {
      throw StateError('Unsupported position embedding type.');
    }
    _visionPositionEmbeddingCache = dense;
    return dense;
  }

  MlxArray _interpolateVisionPositionEmbedding(
    int gridH,
    int gridW,
    MlxDType dtype,
  ) {
    final hiddenSize = config._vision.hiddenSize;
    final base = _fullVisionPositionEmbedding();
    final baseGrid = math.sqrt(base.length / hiddenSize).round();
    final flatTable32 = MlxArray.fromFloat32List(
      base,
      shape: [baseGrid * baseGrid, hiddenSize],
    );
    late final MlxArray flatTable;
    if (dtype == MlxDType.MLX_FLOAT32) {
      flatTable = flatTable32;
    } else {
      flatTable = flatTable32.astype(dtype);
      flatTable32.close();
    }
    try {
      final table = flatTable.reshape([baseGrid, baseGrid, hiddenSize]);
      try {
        final rowPositions = gridH == 1
            ? MlxArray.fromFloat32List([0.0], shape: [1])
            : ((mx.arange(0, gridH.toDouble(), 1, dtype: MlxDType.MLX_FLOAT32) +
                        MlxArray.full([], 0.5, dtype: MlxDType.MLX_FLOAT32)) *
                    MlxArray.full(
                      [],
                      baseGrid / gridH,
                      dtype: MlxDType.MLX_FLOAT32,
                    )) -
                MlxArray.full([], 0.5, dtype: MlxDType.MLX_FLOAT32);
        final colPositions = gridW == 1
            ? MlxArray.fromFloat32List([0.0], shape: [1])
            : ((mx.arange(0, gridW.toDouble(), 1, dtype: MlxDType.MLX_FLOAT32) +
                        MlxArray.full([], 0.5, dtype: MlxDType.MLX_FLOAT32)) *
                    MlxArray.full(
                      [],
                      baseGrid / gridW,
                      dtype: MlxDType.MLX_FLOAT32,
                    )) -
                MlxArray.full([], 0.5, dtype: MlxDType.MLX_FLOAT32);
        try {
          final rowPositionsList = rowPositions.toFloat32List();
          final colPositionsList = colPositions.toFloat32List();
          final rowFloorList = List<int>.filled(gridH, 0);
          final rowCeilList = List<int>.filled(gridH, 0);
          final colFloorList = List<int>.filled(gridW, 0);
          final colCeilList = List<int>.filled(gridW, 0);
          final rowWeightList = Float32List(gridH);
          final colWeightList = Float32List(gridW);
          for (var i = 0; i < gridH; i++) {
            final pos = rowPositionsList[i];
            final floor = pos.floor();
            final ceil = floor + 1;
            rowFloorList[i] = floor.clamp(0, baseGrid - 1);
            rowCeilList[i] = ceil.clamp(0, baseGrid - 1);
            rowWeightList[i] = pos - rowFloorList[i];
          }
          for (var i = 0; i < gridW; i++) {
            final pos = colPositionsList[i];
            final floor = pos.floor();
            final ceil = floor + 1;
            colFloorList[i] = floor.clamp(0, baseGrid - 1);
            colCeilList[i] = ceil.clamp(0, baseGrid - 1);
            colWeightList[i] = pos - colFloorList[i];
          }
          final rowFloor = MlxArray.fromInt32List(rowFloorList, shape: [gridH]);
          final colFloor = MlxArray.fromInt32List(colFloorList, shape: [gridW]);
          final rowCeil = MlxArray.fromInt32List(rowCeilList, shape: [gridH]);
          final colCeil = MlxArray.fromInt32List(colCeilList, shape: [gridW]);
          try {
            final rowWeight = MlxArray.fromFloat32List(
              rowWeightList,
              shape: [gridH],
            );
            final colWeight = MlxArray.fromFloat32List(
              colWeightList,
              shape: [gridW],
            );
            try {
              List<MlxArray> mesh(MlxArray a, MlxArray b) =>
                  mx.meshgrid([a, b], indexing: 'ij');

              MlxArray gatherPixels(MlxArray rowIdx, MlxArray colIdx) {
                final flatIndices =
                    (rowIdx *
                        MlxArray.full(
                          [],
                          baseGrid.toDouble(),
                          dtype: MlxDType.MLX_INT32,
                        )) +
                    colIdx;
                final gathered = flatTable.take(flatIndices.reshape([gridH * gridW]), axis: 0);
                flatIndices.close();
                return gathered.reshape([gridH, gridW, hiddenSize]);
              }

              final tlGrid = mesh(rowFloor, colFloor);
              final blGrid = mesh(rowCeil, colFloor);
              final trGrid = mesh(rowFloor, colCeil);
              final brGrid = mesh(rowCeil, colCeil);
              final topLeft = gatherPixels(tlGrid[0], tlGrid[1]);
              final bottomLeft = gatherPixels(blGrid[0], blGrid[1]);
              final topRight = gatherPixels(trGrid[0], trGrid[1]);
              final bottomRight = gatherPixels(brGrid[0], brGrid[1]);
              for (final arr in [...tlGrid, ...blGrid, ...trGrid, ...brGrid]) {
                arr.close();
              }
              try {
                final extraDims = List<int>.filled(table.ndim - 2, 1);
                final rWeight = rowWeight.reshape([gridH, 1, ...extraDims]);
                final cWeight = colWeight.reshape([1, gridW, ...extraDims]);
                final one = MlxArray.full([], 1.0, dtype: MlxDType.MLX_FLOAT32);
                try {
                  final result = (((one - rWeight) * (one - cWeight)) * topLeft) +
                      (((one - rWeight) * cWeight) * topRight) +
                      ((rWeight * (one - cWeight)) * bottomLeft) +
                      ((rWeight * cWeight) * bottomRight);
                  try {
                    final cast = result.astype(dtype);
                    try {
                      if (Platform.isIOS) {
                        MlxRuntime.evalAll([cast]);
                      }
                      return cast.reshape([gridH * gridW, hiddenSize]);
                    } finally {
                      cast.close();
                    }
                  } finally {
                    result.close();
                  }
                } finally {
                  one.close();
                  rWeight.close();
                  cWeight.close();
                }
              } finally {
                topLeft.close();
                topRight.close();
                bottomLeft.close();
                bottomRight.close();
              }
            } finally {
              rowWeight.close();
              colWeight.close();
            }
          } finally {
            rowFloor.close();
            colFloor.close();
            rowCeil.close();
            colCeil.close();
          }
        } finally {
          rowPositions.close();
          colPositions.close();
        }
      } finally {
        table.close();
      }
    } finally {
      flatTable.close();
    }
  }

  MlxArray _buildVisionRotaryPosEmbedding(
    int gridH,
    int gridW,
    MlxDType dtype,
  ) {
    final seqLen = gridH * gridW;
    final rotaryDim = config._vision.headDim ~/ 2;
    final maxGrid = math.max(gridH, gridW);
    final invFreq = rotaryDim == 36
        ? MlxArray.fromFloat32List(
            const <double>[
              1.0,
              0.5994842052459717,
              0.3593813478946686,
              0.2154434472322464,
              0.1291549652814865,
              0.07742635905742645,
              0.04641588404774666,
              0.02782559022307396,
              0.01668100617825985,
              0.009999999776482582,
              0.005994840525090694,
              0.0035938136279582977,
              0.002154434099793434,
              0.001291549764573574,
              0.0007742635207250714,
              0.00046415894757956266,
              0.00027825593133457005,
              0.00016681008855812252,
            ],
            shape: [18],
          )
        : (() {
            final idx = mx.arange(
              0,
              rotaryDim.toDouble(),
              2,
              dtype: MlxDType.MLX_FLOAT32,
            );
            final div = idx /
                MlxArray.full(
                  [],
                  rotaryDim.toDouble(),
                  dtype: MlxDType.MLX_FLOAT32,
                );
            idx.close();
            final logTheta = MlxArray.full(
              [],
              10000.0,
              dtype: MlxDType.MLX_FLOAT32,
            ).log();
            final neg = (div * logTheta).negative();
            div.close();
            logTheta.close();
            final created = neg.exp();
            neg.close();
            return created;
          })();
    final seq = mx.arange(0, maxGrid.toDouble(), 1, dtype: MlxDType.MLX_FLOAT32);
    final rotaryFull = mx.outer(seq, invFreq);
    seq.close();
    invFreq.close();

    final rowIds = Int32List(seqLen);
    final colIds = Int32List(seqLen);
    for (var idx = 0; idx < seqLen; idx++) {
      rowIds[idx] = idx ~/ gridW;
      colIds[idx] = idx % gridW;
    }
    final rowIdx = MlxArray.fromInt32List(rowIds, shape: [seqLen]);
    final colIdx = MlxArray.fromInt32List(colIds, shape: [seqLen]);
    final rowPart = rotaryFull.take(rowIdx, axis: 0);
    final colPart = rotaryFull.take(colIdx, axis: 0);
    rotaryFull.close();
    rowIdx.close();
    colIdx.close();
    final out = mx.concatenate([rowPart, colPart], axis: 1);
    rowPart.close();
    colPart.close();
    if (dtype == MlxDType.MLX_FLOAT32) return out;
    final cast = out.astype(dtype);
    out.close();
    return cast;
  }

  MlxArray _applyVisionRotary(MlxArray tensor, MlxArray rotaryPosEmb) {
    final cosBase = rotaryPosEmb.cos();
    final sinBase = rotaryPosEmb.sin();
    final cos1 = cosBase.expandDims(1);
    final sin1 = sinBase.expandDims(1);
    final cos2 = mx.concatenate([cos1, cos1], axis: 2).expandDims(0);
    final sin2 = mx.concatenate([sin1, sin1], axis: 2).expandDims(0);
    cosBase.close();
    sinBase.close();
    cos1.close();
    sin1.close();
    final rotated = _rotateHalfVision(tensor);
    try {
      final left = tensor * cos2;
      final right = rotated * sin2;
      final out = mx.add(left, right);
      try {
        if (Platform.isIOS) {
          MlxRuntime.evalAll([out]);
        }
        left.close();
        right.close();
        cos2.close();
        sin2.close();
        if (out.dtype == tensor.dtype) return out;
        final cast = out.astype(tensor.dtype);
        try {
          if (Platform.isIOS) {
            MlxRuntime.evalAll([cast]);
          }
          return cast;
        } finally {
          out.close();
        }
      } catch (_) {
        out.close();
        rethrow;
      }
    } finally {
      rotated.close();
    }
  }

  MlxArray _rotateHalfVision(MlxArray tensor) {
    final half = tensor.shape[3] ~/ 2;
    final x1 = tensor.slice(
      start: [0, 0, 0, 0],
      stop: [tensor.shape[0], tensor.shape[1], tensor.shape[2], half],
    );
    final x2 = tensor.slice(
      start: [0, 0, 0, half],
      stop: [
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
      ],
    );
    try {
      final negX2 = x2.negative();
      try {
        final out = mx.concatenate([negX2, x1], axis: 3);
        if (Platform.isIOS) {
          MlxRuntime.evalAll([out]);
        }
        return out;
      } finally {
        negX2.close();
      }
    } finally {
      x2.close();
      x1.close();
    }
  }

  // -----------------------------------------------------------------------
  // Single ViT block
  // -----------------------------------------------------------------------

  MlxArray _visionBlock(
    _VisionBlockWeights block,
    MlxArray input,
    _VisionConfig vCfg,
    MlxArray rotaryPosEmb,
  ) {
    // ── Pre-norm 1 ──
    final norm1 = _visionLayerNorm(
      input,
      weight: block.layerNorm1Weight,
      bias: block.layerNorm1Bias,
      eps: vCfg.layerNormEps,
    );

    // ── Self-attention (fused QKV) ──
    final attnOut = _visionAttention(block, norm1, vCfg, rotaryPosEmb);
    norm1.close();

    // ── Residual 1 ──
    final h = mx.add(input, attnOut);
    attnOut.close();
    input.close();

    // ── Pre-norm 2 ──
    final norm2 = _visionLayerNorm(
      h,
      weight: block.layerNorm2Weight,
      bias: block.layerNorm2Bias,
      eps: vCfg.layerNormEps,
    );

    // ── MLP: fc1 → GELU → fc2 ──
    final mlpOut = _visionMlp(block, norm2);
    norm2.close();

    // ── Residual 2 ──
    final out = mx.add(h, mlpOut);
    mlpOut.close();
    h.close();
    return out;
  }

  // -----------------------------------------------------------------------
  // Vision self-attention (fused QKV)
  // -----------------------------------------------------------------------

  MlxArray _visionAttention(
    _VisionBlockWeights block,
    MlxArray input,
    _VisionConfig vCfg,
    MlxArray rotaryPosEmb,
  ) {
    final seqLen = input.shape[0];
    final numHeads = vCfg.numAttentionHeads;
    final headDim = vCfg.headDim;

    // Fused QKV projection: [seq, 3*hidden]
    final qkv = block.qkv.apply(input);
    final qkv4d = qkv.reshape([seqLen, 3, numHeads, headDim]).transposeAxes([
      1,
      0,
      2,
      3,
    ]);
    qkv.close();

    // Split into Q, K, V
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

    // Scaled dot-product attention (no causal mask for vision).
    // Large OCR images can produce 10k+ patch tokens; chunk queries to keep
    // peak memory bounded on iPhone while preserving exact full-attention
    // semantics.
    final chunkSize =
        config.recommendedVisionAttentionChunkSizeForCurrentPlatform;
    final attn = chunkSize > 0 && seqLen > chunkSize
        ? _chunkedVisionAttention(
            qForAttn,
            kForAttn,
            vForAttn,
            headDim: headDim,
            chunkSize: chunkSize,
          )
        : mx.fast.scaledDotProductAttention(
            qForAttn,
            kForAttn,
            vForAttn,
            scale: 1.0 / math.sqrt(headDim.toDouble()),
          );
    qForAttn.close();
    kForAttn.close();
    vForAttn.close();
    v.close();

    // Merge heads and project output
    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      numHeads * headDim,
    ]);
    attn.close();

    final out = block.outProj.apply(merged);
    merged.close();
    return out;
  }

  MlxArray _chunkedVisionAttention(
    MlxArray q,
    MlxArray k,
    MlxArray v, {
    required int headDim,
    required int chunkSize,
  }) {
    final seqLen = q.shape[2];
    final numHeads = q.shape[1];
    var combined = MlxArray.zeros([
      1,
      numHeads,
      seqLen,
      headDim,
    ], dtype: q.dtype);
    for (var start = 0; start < seqLen; start += chunkSize) {
      final end = math.min(start + chunkSize, seqLen);
      final qChunk = q.slice(
        start: [0, 0, start, 0],
        stop: [1, numHeads, end, headDim],
      );
      final chunkOut = mx.fast.scaledDotProductAttention(
        qChunk,
        k,
        v,
        scale: 1.0 / math.sqrt(headDim.toDouble()),
      );
      qChunk.close();
      final updated = combined.sliceUpdate(
        chunkOut,
        start: [0, 0, start, 0],
        stop: [1, numHeads, end, headDim],
      );
      chunkOut.close();
      combined.close();
      combined = updated;
    }
    return combined;
  }
}
