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
  });

  factory Qwen3_5Config.fromSnapshot(String snapshotPath) {
    final root =
        jsonDecode(File('$snapshotPath/config.json').readAsStringSync())
            as Map<String, Object?>;
    final text = root['text_config'] as Map<String, Object?>;
    final quant = root['quantization'] as Map<String, Object?>;
    final rope = text['rope_parameters'] as Map<String, Object?>;
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
          (rope['mrope_section'] as List<Object?>? ?? const <Object?>[11, 11, 0])
              .cast<num>()
              .map((value) => value.toInt())
              .toList(growable: false),
      dtype: text['dtype'] as String? ?? 'float16',
      moeIntermediateSize: (text['moe_intermediate_size'] as num?)?.toInt(),
      sharedExpertIntermediateSize:
          (text['shared_expert_intermediate_size'] as num?)?.toInt(),
      numExperts: (text['num_experts'] as num?)?.toInt(),
      numExpertsPerTok: (text['num_experts_per_tok'] as num?)?.toInt(),
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

  bool get isMoe => modelType == 'qwen3_5_moe';

  bool isLinearLayer(int layerIndex) =>
      (layerIndex + 1) % fullAttentionInterval != 0;

  int get rotaryDims => (headDim * partialRotaryFactor).round();

  MlxDType get computeDType => switch (dtype) {
    'bfloat16' => MlxDType.MLX_BFLOAT16,
    'float32' => MlxDType.MLX_FLOAT32,
    _ => MlxDType.MLX_FLOAT16,
  };
}
