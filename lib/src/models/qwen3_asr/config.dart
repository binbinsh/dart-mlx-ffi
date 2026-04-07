import 'dart:convert';
import 'dart:io';

/// Configuration for the Qwen3-ASR model, parsed from config.json.
final class Qwen3AsrConfig {
  Qwen3AsrConfig({
    required this.audioEncoderDModel,
    required this.audioEncoderLayers,
    required this.audioEncoderHeads,
    required this.audioEncoderFfnDim,
    required this.audioEncoderMelBins,
    required this.audioDownsampleHidden,
    required this.audioOutputDim,
    required this.textHiddenSize,
    required this.textIntermediateSize,
    required this.textNumLayers,
    required this.textNumHeads,
    required this.textNumKvHeads,
    required this.textHeadDim,
    required this.textVocabSize,
    required this.textRmsNormEps,
    required this.textRopeTheta,
    required this.textMropeSections,
    required this.tieWordEmbeddings,
    required this.quantGroupSize,
    required this.quantBits,
    required this.quantMode,
    required this.audioPadTokenId,
    required this.audioStartTokenId,
    required this.audioEndTokenId,
    required this.asrTextTokenId,
    required this.imStartTokenId,
    required this.imEndTokenId,
    required this.newlineTokenId,
    required this.eosTokenIds,
  });

  factory Qwen3AsrConfig.fromSnapshot(String path) {
    final root =
        jsonDecode(File('$path/config.json').readAsStringSync())
            as Map<String, Object?>;
    final audio = root['audio_config'] as Map<String, Object?>? ?? {};
    final text = root['text_config'] as Map<String, Object?>? ?? {};
    final quant = root['quantization'] as Map<String, Object?>? ?? {};

    // Audio encoder params.
    final dModel = (audio['d_model'] as num?)?.toInt() ?? 896;
    final numLayers = (audio['encoder_layers'] as num?)?.toInt() ?? 18;
    final numHeads = (audio['encoder_attention_heads'] as num?)?.toInt() ?? 14;
    final ffnDim = (audio['encoder_ffn_dim'] as num?)?.toInt() ?? 3584;
    final melBins = (audio['num_mel_bins'] as num?)?.toInt() ?? 128;
    final downsampleHidden =
        (audio['downsample_hidden'] as num?)?.toInt() ?? 480;
    final outputDim = (audio['output_dim'] as num?)?.toInt() ?? 1024;

    // Text decoder params.
    final hiddenSize = (text['hidden_size'] as num?)?.toInt() ?? 1024;
    final intermediateSize =
        (text['intermediate_size'] as num?)?.toInt() ?? 3072;
    final textLayers = (text['num_hidden_layers'] as num?)?.toInt() ?? 28;
    final textHeads = (text['num_attention_heads'] as num?)?.toInt() ?? 16;
    final textKvHeads = (text['num_key_value_heads'] as num?)?.toInt() ?? 8;
    final headDim =
        (text['head_dim'] as num?)?.toInt() ?? (hiddenSize ~/ textHeads);
    final vocabSize = (text['vocab_size'] as num?)?.toInt() ?? 151936;
    final rmsEps = (text['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6;
    final ropeTheta = (text['rope_theta'] as num?)?.toDouble() ?? 1e6;
    final mropeSections =
        ((text['rope_scaling'] as Map<String, Object?>?)?['mrope_section']
                    as List<Object?>? ??
                const [24, 20, 20])
            .cast<num>()
            .map((v) => v.toInt())
            .toList(growable: false);

    // Quantization params.
    final groupSize = (quant['group_size'] as num?)?.toInt() ?? 64;
    final bits = (quant['bits'] as num?)?.toInt() ?? 8;
    final mode = quant['mode'] as String? ?? 'affine';

    return Qwen3AsrConfig(
      audioEncoderDModel: dModel,
      audioEncoderLayers: numLayers,
      audioEncoderHeads: numHeads,
      audioEncoderFfnDim: ffnDim,
      audioEncoderMelBins: melBins,
      audioDownsampleHidden: downsampleHidden,
      audioOutputDim: outputDim,
      textHiddenSize: hiddenSize,
      textIntermediateSize: intermediateSize,
      textNumLayers: textLayers,
      textNumHeads: textHeads,
      textNumKvHeads: textKvHeads,
      textHeadDim: headDim,
      textVocabSize: vocabSize,
      textRmsNormEps: rmsEps,
      textRopeTheta: ropeTheta,
      textMropeSections: mropeSections,
      tieWordEmbeddings: root['tie_word_embeddings'] as bool? ?? true,
      quantGroupSize: groupSize,
      quantBits: bits,
      quantMode: mode,
      audioPadTokenId: 151676,
      audioStartTokenId: 151669,
      audioEndTokenId: 151670,
      asrTextTokenId: 151704,
      imStartTokenId: 151644,
      imEndTokenId: 151645,
      newlineTokenId: 198,
      eosTokenIds: const [151645, 151643],
    );
  }

  // Audio encoder.
  final int audioEncoderDModel; // 896
  final int audioEncoderLayers; // 18
  final int audioEncoderHeads; // 14
  final int audioEncoderFfnDim; // 3584
  final int audioEncoderMelBins; // 128
  final int audioDownsampleHidden; // 480
  final int audioOutputDim; // 1024

  // Text decoder.
  final int textHiddenSize; // 1024
  final int textIntermediateSize; // 3072
  final int textNumLayers; // 28
  final int textNumHeads; // 16
  final int textNumKvHeads; // 8
  final int textHeadDim; // 128
  final int textVocabSize; // 151936
  final double textRmsNormEps; // 1e-6
  final double textRopeTheta; // 1e6
  final List<int> textMropeSections; // [24, 20, 20]
  final bool tieWordEmbeddings;

  // Quantization.
  final int quantGroupSize;
  final int quantBits;
  final String quantMode;

  // Special tokens.
  final int audioPadTokenId;
  final int audioStartTokenId;
  final int audioEndTokenId;
  final int asrTextTokenId;
  final int imStartTokenId;
  final int imEndTokenId;
  final int newlineTokenId;
  final List<int> eosTokenIds;

  /// Audio encoder head dimension.
  int get audioHeadDim => audioEncoderDModel ~/ audioEncoderHeads;

  /// GQA repeat factor for the text decoder.
  int get textGqaRepeat => textNumHeads ~/ textNumKvHeads;
}
