// ignore_for_file: camel_case_types

part of 'qwen3_5.dart';

final class Qwen3_5Config {
  Qwen3_5Config({
    required this.modelType,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.rmsNormEps,
    required this.vocabSize,
    required this.tieWordEmbeddings,
    required this.groupSize,
    required this.bits,
    required this.mode,
    required this.quantizationByPath,
    required this.fullAttentionInterval,
    required this.linearNumValueHeads,
    required this.linearNumKeyHeads,
    required this.linearKeyHeadDim,
    required this.linearValueHeadDim,
    required this.linearConvKernelDim,
    required this.ropeTheta,
    required this.partialRotaryFactor,
    required this.mropeSection,
    required this.dtype,
    this.moeIntermediateSize,
    this.sharedExpertIntermediateSize,
    this.numExperts,
    this.numExpertsPerTok,
    this.visionConfig,
    this.imageTokenId,
    this.visionStartTokenId,
    this.visionEndTokenId,
    this.videoTokenId,
  });

  factory Qwen3_5Config.fromSnapshot(String snapshotPath) {
    final root =
        jsonDecode(File('$snapshotPath/config.json').readAsStringSync())
            as Map<String, Object?>;
    final text = root['text_config'] as Map<String, Object?>;
    final quant = root['quantization'] as Map<String, Object?>;
    final rope = text['rope_parameters'] as Map<String, Object?>;
    final quantizationByPath = _parseQuantizationByPath(quant);

    // Parse optional vision config.
    final visionJson = root['vision_config'] as Map<String, Object?>?;
    final visionConfig = visionJson != null
        ? Qwen3_5VisionConfig.fromJson(visionJson)
        : null;

    return Qwen3_5Config(
      modelType: root['model_type'] as String,
      hiddenSize: (text['hidden_size'] as num).toInt(),
      intermediateSize: (text['intermediate_size'] as num? ?? 0).toInt(),
      numHiddenLayers: (text['num_hidden_layers'] as num).toInt(),
      numAttentionHeads: (text['num_attention_heads'] as num).toInt(),
      numKeyValueHeads: (text['num_key_value_heads'] as num).toInt(),
      headDim:
          (text['head_dim'] as num?)?.toInt() ??
          ((text['hidden_size'] as num).toInt() ~/
              (text['num_attention_heads'] as num).toInt()),
      rmsNormEps: (text['rms_norm_eps'] as num).toDouble(),
      vocabSize: (text['vocab_size'] as num).toInt(),
      tieWordEmbeddings: root['tie_word_embeddings'] as bool? ?? false,
      groupSize: (quant['group_size'] as num).toInt(),
      bits: (quant['bits'] as num).toInt(),
      mode: quant['mode'] as String? ?? 'affine',
      quantizationByPath: quantizationByPath,
      fullAttentionInterval:
          (text['full_attention_interval'] as num?)?.toInt() ?? 4,
      linearNumValueHeads:
          (text['linear_num_value_heads'] as num?)?.toInt() ?? 0,
      linearNumKeyHeads: (text['linear_num_key_heads'] as num?)?.toInt() ?? 0,
      linearKeyHeadDim: (text['linear_key_head_dim'] as num?)?.toInt() ?? 0,
      linearValueHeadDim: (text['linear_value_head_dim'] as num?)?.toInt() ?? 0,
      linearConvKernelDim:
          (text['linear_conv_kernel_dim'] as num?)?.toInt() ?? 0,
      ropeTheta: (rope['rope_theta'] as num).toDouble(),
      partialRotaryFactor:
          (rope['partial_rotary_factor'] as num?)?.toDouble() ?? 1.0,
      mropeSection:
          (rope['mrope_section'] as List<Object?>? ??
                  const <Object?>[11, 11, 0])
              .cast<num>()
              .map((value) => value.toInt())
              .toList(growable: false),
      dtype: text['dtype'] as String? ?? 'float16',
      moeIntermediateSize: (text['moe_intermediate_size'] as num?)?.toInt(),
      sharedExpertIntermediateSize:
          (text['shared_expert_intermediate_size'] as num?)?.toInt(),
      numExperts: (text['num_experts'] as num?)?.toInt(),
      numExpertsPerTok: (text['num_experts_per_tok'] as num?)?.toInt(),
      visionConfig: visionConfig,
      imageTokenId: (root['image_token_id'] as num?)?.toInt(),
      visionStartTokenId: (root['vision_start_token_id'] as num?)?.toInt(),
      visionEndTokenId: (root['vision_end_token_id'] as num?)?.toInt(),
      videoTokenId: (root['video_token_id'] as num?)?.toInt(),
    );
  }

  final String modelType;
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final double rmsNormEps;
  final int vocabSize;
  final bool tieWordEmbeddings;
  final int groupSize;
  final int bits;
  final String mode;
  final Map<String, Qwen3_5QuantSpec> quantizationByPath;
  final int fullAttentionInterval;
  final int linearNumValueHeads;
  final int linearNumKeyHeads;
  final int linearKeyHeadDim;
  final int linearValueHeadDim;
  final int linearConvKernelDim;
  final double ropeTheta;
  final double partialRotaryFactor;
  final List<int> mropeSection;
  final String dtype;
  final int? moeIntermediateSize;
  final int? sharedExpertIntermediateSize;
  final int? numExperts;
  final int? numExpertsPerTok;

  /// Vision encoder configuration (null when text-only weights are used).
  final Qwen3_5VisionConfig? visionConfig;
  final int? imageTokenId;
  final int? visionStartTokenId;
  final int? visionEndTokenId;
  final int? videoTokenId;

  bool get hasVision => visionConfig != null;

  bool get isMoe => modelType == 'qwen3_5_moe';

  bool isLinearLayer(int layerIndex) =>
      (layerIndex + 1) % fullAttentionInterval != 0;

  int get rotaryDims => (headDim * partialRotaryFactor).round();

  MlxDType get computeDType => switch (dtype) {
    'bfloat16' => MlxDType.MLX_BFLOAT16,
    'float32' => MlxDType.MLX_FLOAT32,
    _ => MlxDType.MLX_FLOAT16,
  };

  Qwen3_5QuantSpec defaultQuantSpec() =>
      Qwen3_5QuantSpec(groupSize: groupSize, bits: bits, mode: mode);

  Qwen3_5QuantSpec quantSpecFor(String path) =>
      quantizationByPath[path] ?? defaultQuantSpec();

  static Map<String, Qwen3_5QuantSpec> _parseQuantizationByPath(
    Map<String, Object?> quant,
  ) {
    final out = <String, Qwen3_5QuantSpec>{};
    for (final entry in quant.entries) {
      if (entry.value is! Map<String, Object?>) {
        continue;
      }
      final value = entry.value as Map<String, Object?>;
      out[entry.key] = Qwen3_5QuantSpec.fromJson(value);
    }
    return Map.unmodifiable(out);
  }
}

final class Qwen3_5QuantSpec {
  const Qwen3_5QuantSpec({
    required this.groupSize,
    required this.bits,
    required this.mode,
  });

  factory Qwen3_5QuantSpec.fromJson(Map<String, Object?> json) =>
      Qwen3_5QuantSpec(
        groupSize: (json['group_size'] as num).toInt(),
        bits: (json['bits'] as num).toInt(),
        mode: json['mode'] as String? ?? 'affine',
      );

  final int groupSize;
  final int bits;
  final String mode;
}

/// Vision encoder configuration for Qwen3.5 VLM.
///
/// Maps to the `vision_config` block in config.json:
/// ```json
/// {
///   "depth": 12,
///   "hidden_size": 768,
///   "hidden_act": "gelu_pytorch_tanh",
///   "in_channels": 3,
///   "intermediate_size": 3072,
///   "num_heads": 12,
///   "num_position_embeddings": 2304,
///   "out_hidden_size": 1024,
///   "patch_size": 16,
///   "spatial_merge_size": 2,
///   "temporal_patch_size": 2
/// }
/// ```
final class Qwen3_5VisionConfig {
  const Qwen3_5VisionConfig({
    required this.depth,
    required this.hiddenSize,
    required this.hiddenAct,
    required this.inChannels,
    required this.intermediateSize,
    required this.numHeads,
    required this.numPositionEmbeddings,
    required this.outHiddenSize,
    required this.patchSize,
    required this.spatialMergeSize,
    required this.temporalPatchSize,
  });

  factory Qwen3_5VisionConfig.fromJson(Map<String, Object?> json) =>
      Qwen3_5VisionConfig(
        depth: (json['depth'] as num).toInt(),
        hiddenSize: (json['hidden_size'] as num).toInt(),
        hiddenAct: json['hidden_act'] as String? ?? 'gelu_pytorch_tanh',
        inChannels: (json['in_channels'] as num?)?.toInt() ?? 3,
        intermediateSize: (json['intermediate_size'] as num).toInt(),
        numHeads: (json['num_heads'] as num).toInt(),
        numPositionEmbeddings:
            (json['num_position_embeddings'] as num?)?.toInt() ?? 2304,
        outHiddenSize: (json['out_hidden_size'] as num).toInt(),
        patchSize: (json['patch_size'] as num).toInt(),
        spatialMergeSize: (json['spatial_merge_size'] as num?)?.toInt() ?? 2,
        temporalPatchSize: (json['temporal_patch_size'] as num?)?.toInt() ?? 2,
      );

  /// Number of ViT transformer blocks.
  final int depth;

  /// Hidden dimension inside the ViT.
  final int hiddenSize;

  /// Activation function name (e.g. "gelu_pytorch_tanh").
  final String hiddenAct;

  /// Number of input image channels (typically 3 for RGB).
  final int inChannels;

  /// ViT MLP intermediate dimension.
  final int intermediateSize;

  /// Number of attention heads in each ViT block.
  final int numHeads;

  /// Maximum number of rotary position embeddings for the ViT.
  final int numPositionEmbeddings;

  /// Output dimension after the spatial-merge projector (fed to the LM).
  final int outHiddenSize;

  /// Spatial patch size for the Conv2D patch embedding.
  final int patchSize;

  /// Spatial merge factor (e.g. 2 means 2×2 patches are merged).
  final int spatialMergeSize;

  /// Temporal patch size for video frames.
  final int temporalPatchSize;

  /// Head dimension inside the ViT.
  int get headDim => hiddenSize ~/ numHeads;

  /// The factor used in image resize: patch_size * spatial_merge_size.
  int get factor => patchSize * spatialMergeSize;
}
