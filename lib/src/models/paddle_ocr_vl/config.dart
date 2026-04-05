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
