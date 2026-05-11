part of 'qwen3_tts.dart';

final class Qwen3TtsTokenizerEncConfig {
  const Qwen3TtsTokenizerEncConfig({
    required this.frameRate,
    required this.audioChannels,
    required this.codebookDim,
    required this.codebookSize,
    required this.compress,
    required this.dilationGrowthRate,
    required this.headDim,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.kernelSize,
    required this.lastKernelSize,
    required this.layerScaleInitialScale,
    required this.maxPositionEmbeddings,
    required this.normEps,
    required this.numAttentionHeads,
    required this.numFilters,
    required this.numHiddenLayers,
    required this.numKeyValueHeads,
    required this.numQuantizers,
    required this.numResidualLayers,
    required this.numSemanticQuantizers,
    required this.residualKernelSize,
    required this.ropeTheta,
    required this.sampleRate,
    required this.slidingWindow,
    required this.upsamplingRatios,
    required this.useCausalConv,
    required this.useConvShortcut,
    required this.vectorQuantizationHiddenDimension,
  });

  final double frameRate;
  final int audioChannels;
  final int codebookDim;
  final int codebookSize;
  final int compress;
  final int dilationGrowthRate;
  final int headDim;
  final int hiddenSize;
  final int intermediateSize;
  final int kernelSize;
  final int lastKernelSize;
  final double layerScaleInitialScale;
  final int maxPositionEmbeddings;
  final double normEps;
  final int numAttentionHeads;
  final int numFilters;
  final int numHiddenLayers;
  final int numKeyValueHeads;
  final int numQuantizers;
  final int numResidualLayers;
  final int numSemanticQuantizers;
  final int residualKernelSize;
  final double ropeTheta;
  final int sampleRate;
  final int slidingWindow;
  final List<int> upsamplingRatios;
  final bool useCausalConv;
  final bool useConvShortcut;
  final int vectorQuantizationHiddenDimension;
}

Qwen3TtsTokenizerEncConfig _buildTokenizerEncConfig(
  Map<String, Object?> json, {
  required int Function(Object?, String) requireInt,
  required double Function(Object?, String) requireDouble,
  required List<int> Function(Object?, String) requireIntList,
}) {
  return Qwen3TtsTokenizerEncConfig(
    frameRate: requireDouble(json['_frame_rate'] ?? 12.5, 'speech_tokenizer.encoder_config._frame_rate'),
    audioChannels: requireInt(json['audio_channels'] ?? 1, 'speech_tokenizer.encoder_config.audio_channels'),
    codebookDim: requireInt(json['codebook_dim'] ?? 256, 'speech_tokenizer.encoder_config.codebook_dim'),
    codebookSize: requireInt(json['codebook_size'] ?? 2048, 'speech_tokenizer.encoder_config.codebook_size'),
    compress: requireInt(json['compress'] ?? 2, 'speech_tokenizer.encoder_config.compress'),
    dilationGrowthRate: requireInt(
      json['dilation_growth_rate'] ?? 2,
      'speech_tokenizer.encoder_config.dilation_growth_rate',
    ),
    headDim: requireInt(json['head_dim'] ?? 64, 'speech_tokenizer.encoder_config.head_dim'),
    hiddenSize: requireInt(json['hidden_size'] ?? 512, 'speech_tokenizer.encoder_config.hidden_size'),
    intermediateSize: requireInt(
      json['intermediate_size'] ?? 2048,
      'speech_tokenizer.encoder_config.intermediate_size',
    ),
    kernelSize: requireInt(json['kernel_size'] ?? 7, 'speech_tokenizer.encoder_config.kernel_size'),
    lastKernelSize: requireInt(
      json['last_kernel_size'] ?? 3,
      'speech_tokenizer.encoder_config.last_kernel_size',
    ),
    layerScaleInitialScale: requireDouble(
      json['layer_scale_initial_scale'] ?? 0.01,
      'speech_tokenizer.encoder_config.layer_scale_initial_scale',
    ),
    maxPositionEmbeddings: requireInt(
      json['max_position_embeddings'] ?? 8000,
      'speech_tokenizer.encoder_config.max_position_embeddings',
    ),
    normEps: requireDouble(json['norm_eps'] ?? 1e-5, 'speech_tokenizer.encoder_config.norm_eps'),
    numAttentionHeads: requireInt(
      json['num_attention_heads'] ?? 8,
      'speech_tokenizer.encoder_config.num_attention_heads',
    ),
    numFilters: requireInt(json['num_filters'] ?? 64, 'speech_tokenizer.encoder_config.num_filters'),
    numHiddenLayers: requireInt(
      json['num_hidden_layers'] ?? 8,
      'speech_tokenizer.encoder_config.num_hidden_layers',
    ),
    numKeyValueHeads: requireInt(
      json['num_key_value_heads'] ?? 8,
      'speech_tokenizer.encoder_config.num_key_value_heads',
    ),
    numQuantizers: requireInt(
      json['num_quantizers'] ?? 32,
      'speech_tokenizer.encoder_config.num_quantizers',
    ),
    numResidualLayers: requireInt(
      json['num_residual_layers'] ?? 1,
      'speech_tokenizer.encoder_config.num_residual_layers',
    ),
    numSemanticQuantizers: requireInt(
      json['num_semantic_quantizers'] ?? 1,
      'speech_tokenizer.encoder_config.num_semantic_quantizers',
    ),
    residualKernelSize: requireInt(
      json['residual_kernel_size'] ?? 3,
      'speech_tokenizer.encoder_config.residual_kernel_size',
    ),
    ropeTheta: requireDouble(
      json['rope_theta'] ?? 10000.0,
      'speech_tokenizer.encoder_config.rope_theta',
    ),
    sampleRate: requireInt(json['sampling_rate'] ?? 24000, 'speech_tokenizer.encoder_config.sampling_rate'),
    slidingWindow: requireInt(
      json['sliding_window'] ?? 250,
      'speech_tokenizer.encoder_config.sliding_window',
    ),
    upsamplingRatios: requireIntList(
      json['upsampling_ratios'] ?? const <int>[8, 6, 5, 4],
      'speech_tokenizer.encoder_config.upsampling_ratios',
    ),
    useCausalConv: json['use_causal_conv'] != false,
    useConvShortcut: json['use_conv_shortcut'] == true,
    vectorQuantizationHiddenDimension: requireInt(
      json['vector_quantization_hidden_dimension'] ?? 256,
      'speech_tokenizer.encoder_config.vector_quantization_hidden_dimension',
    ),
  );
}
