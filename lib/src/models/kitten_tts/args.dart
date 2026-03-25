library;

import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

final class KittenQuantConfig {
  const KittenQuantConfig({
    required this.groupSize,
    required this.bits,
    required this.mode,
  });

  factory KittenQuantConfig.fromJson(Map<String, Object?> json) {
    return KittenQuantConfig(
      groupSize: (json['group_size'] as num).toInt(),
      bits: (json['bits'] as num).toInt(),
      mode: json['mode'] as String? ?? 'affine',
    );
  }

  final int groupSize;
  final int bits;
  final String mode;
}

final class AlbertModelArgs {
  const AlbertModelArgs({
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.maxPositionEmbeddings,
    this.modelType = 'albert',
    this.embeddingSize = 128,
    this.innerGroupNum = 1,
    this.numHiddenGroups = 1,
    this.hiddenDropoutProb = 0.1,
    this.attentionProbsDropoutProb = 0.1,
    this.typeVocabSize = 2,
    this.initializerRange = 0.02,
    this.layerNormEps = 1e-12,
    this.vocabSize = 30522,
    this.dropout = 0.0,
  });

  factory AlbertModelArgs.fromJson(
    Map<String, Object?> json, {
    required int vocabSize,
  }) {
    return AlbertModelArgs(
      numHiddenLayers: (json['num_hidden_layers'] as num).toInt(),
      numAttentionHeads: (json['num_attention_heads'] as num).toInt(),
      hiddenSize: (json['hidden_size'] as num).toInt(),
      intermediateSize: (json['intermediate_size'] as num).toInt(),
      maxPositionEmbeddings: (json['max_position_embeddings'] as num).toInt(),
      modelType: json['model_type'] as String? ?? 'albert',
      embeddingSize: (json['embedding_size'] as num? ?? 128).toInt(),
      innerGroupNum: (json['inner_group_num'] as num? ?? 1).toInt(),
      numHiddenGroups: (json['num_hidden_groups'] as num? ?? 1).toInt(),
      hiddenDropoutProb: (json['hidden_dropout_prob'] as num? ?? 0.1)
          .toDouble(),
      attentionProbsDropoutProb:
          (json['attention_probs_dropout_prob'] as num? ?? 0.1).toDouble(),
      typeVocabSize: (json['type_vocab_size'] as num? ?? 2).toInt(),
      initializerRange: (json['initializer_range'] as num? ?? 0.02).toDouble(),
      layerNormEps: (json['layer_norm_eps'] as num? ?? 1e-12).toDouble(),
      vocabSize: vocabSize,
      dropout: (json['dropout'] as num? ?? 0.0).toDouble(),
    );
  }

  final int numHiddenLayers;
  final int numAttentionHeads;
  final int hiddenSize;
  final int intermediateSize;
  final int maxPositionEmbeddings;
  final String modelType;
  final int embeddingSize;
  final int innerGroupNum;
  final int numHiddenGroups;
  final double hiddenDropoutProb;
  final double attentionProbsDropoutProb;
  final int typeVocabSize;
  final double initializerRange;
  final double layerNormEps;
  final int vocabSize;
  final double dropout;

  int get attentionHeadSize => hiddenSize ~/ numAttentionHeads;
  int get allHeadSize => numAttentionHeads * attentionHeadSize;
}

final class ModelConfig {
  const ModelConfig({
    required this.hiddenDim,
    required this.maxConvDim,
    required this.maxDur,
    required this.nLayer,
    required this.nMels,
    required this.nToken,
    required this.styleDim,
    required this.textEncoderKernelSize,
    required this.asrResDim,
    required this.plbert,
    required this.istftnet,
    required this.quantization,
    this.sampleRate = 24000,
    this.decoderOutDim,
    this.voicesPath = 'voices.npz',
    this.speedPriors = const <String, double>{},
    this.voiceAliases = const <String, String>{},
    this.modelPath,
    this.activationQuantModules = const <String>[],
  });

  factory ModelConfig.fromJson(Map<String, Object?> json) {
    final plbert = json['plbert'] as Map<String, Object?>;
    final quantRoot =
        (json['quantization'] ?? json['quantization_config'])
            as Map<String, Object?>;
    return ModelConfig(
      hiddenDim: (json['hidden_dim'] as num).toInt(),
      maxConvDim: (json['max_conv_dim'] as num).toInt(),
      maxDur: (json['max_dur'] as num).toInt(),
      nLayer: (json['n_layer'] as num).toInt(),
      nMels: (json['n_mels'] as num).toInt(),
      nToken: (json['n_token'] as num).toInt(),
      styleDim: (json['style_dim'] as num).toInt(),
      textEncoderKernelSize: (json['text_encoder_kernel_size'] as num).toInt(),
      asrResDim: (json['asr_res_dim'] as num).toInt(),
      plbert: AlbertModelArgs.fromJson(
        plbert,
        vocabSize: (json['n_token'] as num).toInt(),
      ),
      istftnet: Map<String, Object?>.from(
        json['istftnet'] as Map<String, Object?>,
      ),
      quantization: KittenQuantConfig.fromJson(quantRoot),
      sampleRate: (json['sample_rate'] as num? ?? 24000).toInt(),
      decoderOutDim: (json['decoder_out_dim'] as num?)?.toInt(),
      voicesPath: json['voices_path'] as String? ?? 'voices.npz',
      speedPriors: _readDoubleMap(json['speed_priors']),
      voiceAliases: _readStringMap(json['voice_aliases']),
      modelPath: json['model_path'] as String?,
      activationQuantModules:
          (json['activation_quant_modules'] as List<Object?>? ??
                  const <Object?>[])
              .whereType<String>()
              .toList(growable: false),
    );
  }

  factory ModelConfig.fromSnapshot(String snapshotPath) {
    final json =
        jsonDecode(File('$snapshotPath/config.json').readAsStringSync())
            as Map<String, Object?>;
    return ModelConfig.fromJson(json);
  }

  final int hiddenDim;
  final int maxConvDim;
  final int maxDur;
  final int nLayer;
  final int nMels;
  final int nToken;
  final int styleDim;
  final int textEncoderKernelSize;
  final int asrResDim;
  final AlbertModelArgs plbert;
  final Map<String, Object?> istftnet;
  final KittenQuantConfig quantization;
  final int sampleRate;
  final int? decoderOutDim;
  final String voicesPath;
  final Map<String, double> speedPriors;
  final Map<String, String> voiceAliases;
  final String? modelPath;
  final List<String> activationQuantModules;

  MlxDType get computeDType => MlxDType.MLX_FLOAT32;

  static Map<String, double> _readDoubleMap(Object? value) {
    final map = value as Map<String, Object?>? ?? const <String, Object?>{};
    return <String, double>{
      for (final entry in map.entries)
        entry.key: (entry.value as num).toDouble(),
    };
  }

  static Map<String, String> _readStringMap(Object? value) {
    final map = value as Map<String, Object?>? ?? const <String, Object?>{};
    return <String, String>{
      for (final entry in map.entries) entry.key: entry.value as String,
    };
  }
}
