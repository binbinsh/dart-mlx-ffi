part of 'paddle_ocr_vl.dart';

/// Configuration for the PaddleOCR-VL vision-language model.
///
/// Parsed from the model's `config.json` which follows the HuggingFace
/// `PaddleOCRVLForConditionalGeneration` layout (ERNIE-4.5 language backbone
/// + ViT vision encoder + spatial-merge projector).
final class PaddleOcrVlConfig {
  PaddleOcrVlConfig._({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.rmsNormEps,
    required this.vocabSize,
    required this.tieWordEmbeddings,
    required this.ropeTheta,
    required this.mropeSection,
    required this.imageTokenId,
    required this.visionStartTokenId,
    required this.visionEndTokenId,
    required this.eosTokenId,
    required _VisionConfig vision,
    required this.groupSize,
    required this.bits,
    required this.mode,
  }) : _vision = vision;

  factory PaddleOcrVlConfig.fromSnapshot(String snapshotPath) {
    final root =
        jsonDecode(File('$snapshotPath/config.json').readAsStringSync())
            as Map<String, Object?>;

    final vision = root['vision_config'] as Map<String, Object?>;
    final rope = root['rope_scaling'] as Map<String, Object?>?;
    final quant = root['quantization'] as Map<String, Object?>?;

    return PaddleOcrVlConfig._(
      hiddenSize: (root['hidden_size'] as num).toInt(),
      intermediateSize: (root['intermediate_size'] as num).toInt(),
      numHiddenLayers: (root['num_hidden_layers'] as num).toInt(),
      numAttentionHeads: (root['num_attention_heads'] as num).toInt(),
      numKeyValueHeads: (root['num_key_value_heads'] as num).toInt(),
      headDim:
          (root['head_dim'] as num?)?.toInt() ??
          ((root['hidden_size'] as num).toInt() ~/
              (root['num_attention_heads'] as num).toInt()),
      rmsNormEps: (root['rms_norm_eps'] as num).toDouble(),
      vocabSize: (root['vocab_size'] as num).toInt(),
      tieWordEmbeddings: root['tie_word_embeddings'] as bool? ?? false,
      ropeTheta: (root['rope_theta'] as num?)?.toDouble() ?? 500000.0,
      mropeSection:
          (rope?['mrope_section'] as List<Object?>? ??
                  const <Object?>[16, 24, 24])
              .cast<num>()
              .map((v) => v.toInt())
              .toList(growable: false),
      imageTokenId: (root['image_token_id'] as num).toInt(),
      visionStartTokenId: (root['vision_start_token_id'] as num).toInt(),
      visionEndTokenId: (root['vision_end_token_id'] as num).toInt(),
      eosTokenId: (root['eos_token_id'] as num).toInt(),
      vision: _VisionConfig.fromJson(vision),
      groupSize: (quant?['group_size'] as num?)?.toInt() ?? 64,
      bits: (quant?['bits'] as num?)?.toInt() ?? 4,
      mode: quant?['mode'] as String? ?? 'affine',
    );
  }

  // ── Language model config ──
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final double rmsNormEps;
  final int vocabSize;
  final bool tieWordEmbeddings;
  final double ropeTheta;
  final List<int> mropeSection;

  // ── Special tokens ──
  final int imageTokenId;
  final int visionStartTokenId;
  final int visionEndTokenId;
  final int eosTokenId;

  // ── Vision encoder config ──
  final _VisionConfig _vision;

  // ── Quantization ──
  final int groupSize;
  final int bits;
  final String mode;

  _QuantSpec _defaultQuantSpec() =>
      _QuantSpec(groupSize: groupSize, bits: bits, mode: mode);

  int get visionInputImageSize => _vision.alignedImageSize;
  int get visionPatchSize => _vision.patchSize;
  int get visionSpatialMergeSize => _vision.spatialMergeSize;
  int get recommendedMinPixels => 28 * 28 * 188;
  int get recommendedMaxPixelsForCurrentPlatform {
    if (Platform.isIOS) {
      // Reduced from 28*28*1280 to 28*28*896 (~30% fewer pixels) to cut
      // vision encoder peak memory on iPhone while retaining high-quality OCR
      // for typical recipe photos (effective resolution up to ~840×840).
      return 28 * 28 * 896;
    }
    return 2822400;
  }

  int get recommendedVisionWindowSizeForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_VISION_WINDOW_SIZE'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    // Quality-first default: keep exact full-attention semantics unless
    // explicitly overridden for experimentation.
    return -1;
  }

  int get recommendedVisionWindowedLayerCountForCurrentPlatform {
    final override =
        Platform.environment['DART_MLX_PADDLE_VISION_WINDOW_LAYERS'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    // Quality-first default: no windowed layers unless explicitly enabled.
    return 0;
  }

  int get recommendedVisionAttentionChunkSizeForCurrentPlatform {
    final override =
        Platform.environment['DART_MLX_PADDLE_VISION_ATTENTION_CHUNK'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 128;
    }
    return -1;
  }

  bool get enableVisionLayerwiseEvalForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_VISION_LAYER_EVAL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }

  bool get enableDecoderLayerwiseEvalForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_DECODER_LAYER_EVAL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }

  bool get enableAggressiveCacheClearingForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_CLEAR_CACHE'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }

  bool get forceFloat32VisionForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_FORCE_VISION_F32'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return false;
  }

  // ── Memory limits ──

  /// Recommended MLX memory limit in bytes for the current platform.
  ///
  /// On iOS, constrains the Metal allocator so the OS doesn't kill the app.
  /// Returns -1 to leave the limit unchanged.
  int get recommendedMemoryLimitBytesForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_MEMORY_LIMIT_MB'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed * 1024 * 1024;
    }
    if (Platform.isIOS) {
      return 2304 * 1024 * 1024; // 2.25 GB
    }
    return -1;
  }

  /// Recommended MLX cache limit in bytes for the current platform.
  ///
  /// On iOS, limits the amount of freed-but-held GPU memory so the OS can
  /// reclaim it under pressure.  Returns -1 to leave the limit unchanged.
  int get recommendedCacheLimitBytesForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_CACHE_LIMIT_MB'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed * 1024 * 1024;
    }
    if (Platform.isIOS) {
      return 256 * 1024 * 1024; // 256 MB
    }
    return -1;
  }

  // ── Vision encoder tuning ──

  /// Number of vision layers to batch before calling evalAll.
  ///
  /// Higher values reduce GPU dispatch overhead but increase peak memory.
  /// Only used when [enableVisionLayerwiseEvalForCurrentPlatform] is true.
  int get visionEvalBatchSizeForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_VISION_EVAL_BATCH'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    if (Platform.isIOS) {
      return 1; // eval every layer to minimize peak graph memory
    }
    return 1;
  }

  // ── KV Cache tuning ──

  /// Maximum sequence length for pre-allocated KV cache buffers.
  ///
  /// The cache pre-allocates `[1, numKvHeads, maxKvCacheSeqLen, headDim]`
  /// per layer.  A shorter limit saves memory but will fail if the total
  /// sequence (prompt + generated tokens) exceeds this value.
  ///
  /// Default: 2048 tokens — plenty for OCR workloads (prompt ~800 tokens
  /// + up to 512 generated).
  int get maxKvCacheSeqLenForCurrentPlatform {
    final override =
        Platform.environment['DART_MLX_PADDLE_MAX_KV_CACHE_SEQ_LEN'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    if (Platform.isIOS) {
      return 2048;
    }
    return 4096;
  }

  // ── Vision weight lifecycle ──

  /// Whether to release all vision encoder weights after encoding is done.
  ///
  /// This frees ~385 MB of GPU memory but means the model can only encode
  /// one image per runner lifetime.  Enabled by default on iOS where memory
  /// is the binding constraint.
  bool get enableVisionWeightReleaseForCurrentPlatform {
    final override =
        Platform.environment['DART_MLX_PADDLE_RELEASE_VISION_WEIGHTS'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }
}

/// Vision encoder (ViT) config.
final class _VisionConfig {
  const _VisionConfig({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.patchSize,
    required this.imageSize,
    required this.numChannels,
    required this.layerNormEps,
    required this.spatialMergeSize,
  });

  factory _VisionConfig.fromJson(Map<String, Object?> json) => _VisionConfig(
    hiddenSize: (json['hidden_size'] as num).toInt(),
    intermediateSize: (json['intermediate_size'] as num).toInt(),
    numHiddenLayers: (json['num_hidden_layers'] as num).toInt(),
    numAttentionHeads: (json['num_attention_heads'] as num).toInt(),
    patchSize: (json['patch_size'] as num).toInt(),
    imageSize: (json['image_size'] as num).toInt(),
    numChannels: (json['num_channels'] as num?)?.toInt() ?? 3,
    layerNormEps: (json['layer_norm_eps'] as num?)?.toDouble() ?? 1e-6,
    spatialMergeSize: (json['spatial_merge_size'] as num?)?.toInt() ?? 2,
  );

  final int hiddenSize; // 1152
  final int intermediateSize; // 4304
  final int numHiddenLayers; // 27
  final int numAttentionHeads; // 16
  final int patchSize; // 14
  final int imageSize; // 384
  final int numChannels; // 3
  final double layerNormEps; // 1e-6
  final int spatialMergeSize; // 2

  int get headDim => hiddenSize ~/ numAttentionHeads;
  int get numPatches => (imageSize ~/ patchSize) * (imageSize ~/ patchSize);
  int get alignedImageSize {
    final block = patchSize * spatialMergeSize;
    return ((imageSize + block - 1) ~/ block) * block;
  }

  /// Hidden dimension of the projector input after spatial merge.
  int get projectorInputDim => hiddenSize * spatialMergeSize * spatialMergeSize;
}

/// Quantization parameters for a weight group.
final class _QuantSpec {
  const _QuantSpec({
    required this.groupSize,
    required this.bits,
    required this.mode,
  });

  final int groupSize;
  final int bits;
  final String mode;
}
