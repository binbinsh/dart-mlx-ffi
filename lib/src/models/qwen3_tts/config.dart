part of 'qwen3_tts.dart';

const String qwen3TtsManifestName = 'cmdspace_mlx_qwen3_tts.json';
const String _qwen3TtsFormat = 'cmdspace-mlx-qwen3tts/v1';

final class Qwen3TtsQuantConfig {
  const Qwen3TtsQuantConfig({
    required this.bits,
    required this.groupSize,
    required this.mode,
  });

  final int bits;
  final int groupSize;
  final String mode;
}

final class Qwen3TtsCodePredictorConfig {
  const Qwen3TtsCodePredictorConfig({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.vocabSize,
    required this.numCodeGroups,
    required this.rmsNormEps,
    required this.ropeTheta,
  });

  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int vocabSize;
  final int numCodeGroups;
  final double rmsNormEps;
  final double ropeTheta;
}

final class Qwen3TtsTalkerConfig {
  const Qwen3TtsTalkerConfig({
    required this.hiddenSize,
    required this.textHiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.vocabSize,
    required this.textVocabSize,
    required this.numCodeGroups,
    required this.rmsNormEps,
    required this.ropeTheta,
    required this.codecBosId,
    required this.codecEosTokenId,
    required this.codecPadId,
    required this.codecThinkId,
    required this.codecNoThinkId,
    required this.codecThinkBosId,
    required this.codecThinkEosId,
    required this.codecLanguageId,
    required this.codePredictor,
  });

  final int hiddenSize;
  final int textHiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int vocabSize;
  final int textVocabSize;
  final int numCodeGroups;
  final double rmsNormEps;
  final double ropeTheta;
  final int codecBosId;
  final int codecEosTokenId;
  final int codecPadId;
  final int codecThinkId;
  final int codecNoThinkId;
  final int codecThinkBosId;
  final int codecThinkEosId;
  final Map<String, int> codecLanguageId;
  final Qwen3TtsCodePredictorConfig codePredictor;
}

final class Qwen3TtsDecoderConfig {
  const Qwen3TtsDecoderConfig({
    required this.latentDim,
    required this.codebookDim,
    required this.codebookSize,
    required this.decoderDim,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.layerScaleInitialScale,
    required this.maxPositionEmbeddings,
    required this.headDim,
    required this.numAttentionHeads,
    required this.numHiddenLayers,
    required this.numKeyValueHeads,
    required this.numQuantizers,
    required this.numSemanticQuantizers,
    required this.rmsNormEps,
    required this.ropeTheta,
    required this.upsampleRates,
    required this.upsamplingRatios,
  });

  final int latentDim;
  final int codebookDim;
  final int codebookSize;
  final int decoderDim;
  final int hiddenSize;
  final int intermediateSize;
  final double layerScaleInitialScale;
  final int maxPositionEmbeddings;
  final int headDim;
  final int numAttentionHeads;
  final int numHiddenLayers;
  final int numKeyValueHeads;
  final int numQuantizers;
  final int numSemanticQuantizers;
  final double rmsNormEps;
  final double ropeTheta;
  final List<int> upsampleRates;
  final List<int> upsamplingRatios;

  int get totalUpsample =>
      upsampleRates.fold<int>(1, (a, b) => a * b) *
      upsamplingRatios.fold<int>(1, (a, b) => a * b);
}

final class Qwen3TtsManifest {
  const Qwen3TtsManifest({
    required this.rootPath,
    required this.modelId,
    required this.sampleRate,
    required this.weightsPath,
    required this.configPath,
    required this.generationConfigPath,
    required this.tokenizerConfigPath,
    required this.vocabPath,
    required this.mergesPath,
    required this.speechTokenizerDir,
    required this.ttsBosTokenId,
    required this.ttsEosTokenId,
    required this.ttsPadTokenId,
    required this.imStartTokenId,
    required this.imEndTokenId,
    required this.talker,
    required this.quantization,
    required this.decoder,
  });

  factory Qwen3TtsManifest.load(String rawPath) {
    final normalized = p.normalize(rawPath.trim());
    if (normalized.isEmpty) {
      throw StateError('Select a local Qwen3-TTS bundle first.');
    }
    final type = FileSystemEntity.typeSync(normalized);
    if (type == FileSystemEntityType.notFound) {
      throw StateError('Qwen3-TTS bundle not found: $normalized');
    }
    final rootPath =
        type == FileSystemEntityType.directory ? normalized : p.dirname(normalized);
    final manifestFile = File(
      p.basename(normalized) == qwen3TtsManifestName
          ? normalized
          : p.join(rootPath, qwen3TtsManifestName),
    );
    if (!manifestFile.existsSync()) {
      throw StateError('Qwen3-TTS manifest not found: ${manifestFile.path}');
    }
    final decoded = jsonDecode(manifestFile.readAsStringSync());
    if (decoded is! Map<String, Object?>) {
      throw StateError('Invalid Qwen3-TTS manifest: ${manifestFile.path}');
    }
    final format = decoded['format']?.toString().trim();
    if (format != _qwen3TtsFormat) {
      throw StateError(
        'Unsupported Qwen3-TTS bundle format: $format (want $_qwen3TtsFormat)',
      );
    }

    int requireInt(Object? value, String key) {
      if (value is int) return value;
      if (value is num) return value.toInt();
      throw StateError('Qwen3-TTS manifest has invalid "$key".');
    }

    double requireDouble(Object? value, String key) {
      if (value is num) return value.toDouble();
      throw StateError('Qwen3-TTS manifest has invalid "$key".');
    }

    List<int> requireIntList(Object? value, String key) {
      if (value is! List) {
        throw StateError('Qwen3-TTS manifest missing list "$key".');
      }
      return value.map((e) => requireInt(e, key)).toList(growable: false);
    }

    Map<String, int> requireIntMap(Object? value, String key) {
      if (value is! Map) {
        return const <String, int>{};
      }
      final out = <String, int>{};
      for (final entry in value.entries) {
        out[entry.key.toString()] = requireInt(entry.value, key);
      }
      return out;
    }

    String requireRelPath(String key) {
      final raw = decoded[key]?.toString().trim() ?? '';
      if (raw.isEmpty) {
        throw StateError('Qwen3-TTS manifest is missing "$key".');
      }
      return p.normalize(p.join(rootPath, raw));
    }

    final talkerJson = decoded['talker'];
    if (talkerJson is! Map<String, Object?>) {
      throw StateError('Qwen3-TTS manifest missing talker config.');
    }
    final predictorJson = talkerJson['code_predictor'];
    if (predictorJson is! Map<String, Object?>) {
      throw StateError('Qwen3-TTS manifest missing talker.code_predictor config.');
    }
    final quantJson = decoded['quantization'] as Map<String, Object?>? ?? const {};
    final decoderJson = jsonDecode(
          File(p.join(rootPath, 'speech_tokenizer', 'config.json')).readAsStringSync(),
        )
        as Map<String, Object?>;
    final decoderCfg = decoderJson['decoder_config'] as Map<String, Object?>? ?? const {};

    return Qwen3TtsManifest(
      rootPath: rootPath,
      modelId: decoded['model_id']?.toString().trim() ?? '',
      sampleRate: requireInt(decoded['sample_rate'], 'sample_rate'),
      weightsPath: requireRelPath('weights'),
      configPath: requireRelPath('config'),
      generationConfigPath: p.join(rootPath, 'generation_config.json'),
      tokenizerConfigPath: requireRelPath('tokenizer_config'),
      vocabPath: requireRelPath('vocab'),
      mergesPath: requireRelPath('merges'),
      speechTokenizerDir: p.join(rootPath, decoded['speech_tokenizer_dir']!.toString()),
      ttsBosTokenId: requireInt(decoded['tts_bos_token_id'], 'tts_bos_token_id'),
      ttsEosTokenId: requireInt(decoded['tts_eos_token_id'], 'tts_eos_token_id'),
      ttsPadTokenId: requireInt(decoded['tts_pad_token_id'], 'tts_pad_token_id'),
      imStartTokenId: requireInt(decoded['im_start_token_id'], 'im_start_token_id'),
      imEndTokenId: requireInt(decoded['im_end_token_id'], 'im_end_token_id'),
      talker: Qwen3TtsTalkerConfig(
        hiddenSize: requireInt(talkerJson['hidden_size'], 'talker.hidden_size'),
        textHiddenSize: requireInt(
          talkerJson['text_hidden_size'],
          'talker.text_hidden_size',
        ),
        intermediateSize: requireInt(
          talkerJson['intermediate_size'],
          'talker.intermediate_size',
        ),
        numHiddenLayers: requireInt(
          talkerJson['num_hidden_layers'],
          'talker.num_hidden_layers',
        ),
        numAttentionHeads: requireInt(
          talkerJson['num_attention_heads'],
          'talker.num_attention_heads',
        ),
        numKeyValueHeads: requireInt(
          talkerJson['num_key_value_heads'],
          'talker.num_key_value_heads',
        ),
        headDim: requireInt(talkerJson['head_dim'], 'talker.head_dim'),
        vocabSize: requireInt(talkerJson['vocab_size'], 'talker.vocab_size'),
        textVocabSize: requireInt(
          talkerJson['text_vocab_size'],
          'talker.text_vocab_size',
        ),
        numCodeGroups: requireInt(
          talkerJson['num_code_groups'],
          'talker.num_code_groups',
        ),
        rmsNormEps: requireDouble(
          talkerJson['rms_norm_eps'],
          'talker.rms_norm_eps',
        ),
        ropeTheta: requireDouble(talkerJson['rope_theta'], 'talker.rope_theta'),
        codecBosId: requireInt(talkerJson['codec_bos_id'], 'talker.codec_bos_id'),
        codecEosTokenId: requireInt(
          talkerJson['codec_eos_token_id'],
          'talker.codec_eos_token_id',
        ),
        codecPadId: requireInt(talkerJson['codec_pad_id'], 'talker.codec_pad_id'),
        codecThinkId: requireInt(
          talkerJson['codec_think_id'],
          'talker.codec_think_id',
        ),
        codecNoThinkId: requireInt(
          talkerJson['codec_nothink_id'],
          'talker.codec_nothink_id',
        ),
        codecThinkBosId: requireInt(
          talkerJson['codec_think_bos_id'],
          'talker.codec_think_bos_id',
        ),
        codecThinkEosId: requireInt(
          talkerJson['codec_think_eos_id'],
          'talker.codec_think_eos_id',
        ),
        codecLanguageId: requireIntMap(
          talkerJson['codec_language_id'],
          'talker.codec_language_id',
        ),
        codePredictor: Qwen3TtsCodePredictorConfig(
          hiddenSize: requireInt(
            predictorJson['hidden_size'],
            'talker.code_predictor.hidden_size',
          ),
          intermediateSize: requireInt(
            predictorJson['intermediate_size'],
            'talker.code_predictor.intermediate_size',
          ),
          numHiddenLayers: requireInt(
            predictorJson['num_hidden_layers'],
            'talker.code_predictor.num_hidden_layers',
          ),
          numAttentionHeads: requireInt(
            predictorJson['num_attention_heads'],
            'talker.code_predictor.num_attention_heads',
          ),
          numKeyValueHeads: requireInt(
            predictorJson['num_key_value_heads'],
            'talker.code_predictor.num_key_value_heads',
          ),
          headDim: requireInt(
            predictorJson['head_dim'],
            'talker.code_predictor.head_dim',
          ),
          vocabSize: requireInt(
            predictorJson['vocab_size'],
            'talker.code_predictor.vocab_size',
          ),
          numCodeGroups: requireInt(
            predictorJson['num_code_groups'],
            'talker.code_predictor.num_code_groups',
          ),
          rmsNormEps: requireDouble(
            predictorJson['rms_norm_eps'],
            'talker.code_predictor.rms_norm_eps',
          ),
          ropeTheta: requireDouble(
            predictorJson['rope_theta'],
            'talker.code_predictor.rope_theta',
          ),
        ),
      ),
      quantization: Qwen3TtsQuantConfig(
        bits: requireInt(quantJson['bits'], 'quantization.bits'),
        groupSize: requireInt(quantJson['group_size'], 'quantization.group_size'),
        mode: quantJson['mode']?.toString() ?? 'affine',
      ),
      decoder: Qwen3TtsDecoderConfig(
        latentDim: requireInt(decoderCfg['latent_dim'], 'decoder.latent_dim'),
        codebookDim: requireInt(decoderCfg['codebook_dim'], 'decoder.codebook_dim'),
        codebookSize: requireInt(
          decoderCfg['codebook_size'],
          'decoder.codebook_size',
        ),
        decoderDim: requireInt(decoderCfg['decoder_dim'], 'decoder.decoder_dim'),
        hiddenSize: requireInt(decoderCfg['hidden_size'], 'decoder.hidden_size'),
        intermediateSize: requireInt(
          decoderCfg['intermediate_size'],
          'decoder.intermediate_size',
        ),
        layerScaleInitialScale: requireDouble(
          decoderCfg['layer_scale_initial_scale'],
          'decoder.layer_scale_initial_scale',
        ),
        maxPositionEmbeddings: requireInt(
          decoderCfg['max_position_embeddings'],
          'decoder.max_position_embeddings',
        ),
        headDim: requireInt(decoderCfg['head_dim'], 'decoder.head_dim'),
        numAttentionHeads: requireInt(
          decoderCfg['num_attention_heads'],
          'decoder.num_attention_heads',
        ),
        numHiddenLayers: requireInt(
          decoderCfg['num_hidden_layers'],
          'decoder.num_hidden_layers',
        ),
        numKeyValueHeads: requireInt(
          decoderCfg['num_key_value_heads'],
          'decoder.num_key_value_heads',
        ),
        numQuantizers: requireInt(
          decoderCfg['num_quantizers'],
          'decoder.num_quantizers',
        ),
        numSemanticQuantizers: requireInt(
          decoderCfg['num_semantic_quantizers'],
          'decoder.num_semantic_quantizers',
        ),
        rmsNormEps: requireDouble(
          decoderCfg['rms_norm_eps'],
          'decoder.rms_norm_eps',
        ),
        ropeTheta: requireDouble(decoderCfg['rope_theta'], 'decoder.rope_theta'),
        upsampleRates: requireIntList(
          decoderCfg['upsample_rates'],
          'decoder.upsample_rates',
        ),
        upsamplingRatios: requireIntList(
          decoderCfg['upsampling_ratios'],
          'decoder.upsampling_ratios',
        ),
      ),
    );
  }

  final String rootPath;
  final String modelId;
  final int sampleRate;
  final String weightsPath;
  final String configPath;
  final String generationConfigPath;
  final String tokenizerConfigPath;
  final String vocabPath;
  final String mergesPath;
  final String speechTokenizerDir;
  final int ttsBosTokenId;
  final int ttsEosTokenId;
  final int ttsPadTokenId;
  final int imStartTokenId;
  final int imEndTokenId;
  final Qwen3TtsTalkerConfig talker;
  final Qwen3TtsQuantConfig quantization;
  final Qwen3TtsDecoderConfig decoder;
}

final class Qwen3TtsBundle {
  Qwen3TtsBundle._({
    required this.manifest,
    required this.tensors,
    required this.decoderTensors,
    required this.generationConfig,
  });

  factory Qwen3TtsBundle.load(String bundlePath) {
    final manifest = Qwen3TtsManifest.load(bundlePath);
    final tensors = mx.io.loadSafetensors(manifest.weightsPath).tensors;
    final generationConfig =
        jsonDecode(File(manifest.generationConfigPath).readAsStringSync())
            as Map<String, Object?>;
    final rawDecoder = loadTensorMap(manifest.speechTokenizerDir);
    final decoderTensors = _sanitizeDecoderTensors(rawDecoder);
    return Qwen3TtsBundle._(
      manifest: manifest,
      tensors: tensors,
      decoderTensors: decoderTensors,
      generationConfig: generationConfig,
    );
  }

  final Qwen3TtsManifest manifest;
  final Map<String, MlxArray> tensors;
  final Map<String, MlxArray> decoderTensors;
  final Map<String, Object?> generationConfig;

  MlxArray require(String key) {
    final value = tensors[key];
    if (value == null) {
      throw StateError('Missing Qwen3-TTS tensor: $key');
    }
    return value;
  }

  MlxArray requireDecoder(String key) {
    final value = decoderTensors[key];
    if (value == null) {
      throw StateError('Missing Qwen3-TTS decoder tensor: $key');
    }
    return value;
  }

  void close() {
    for (final tensor in tensors.values) {
      tensor.close();
    }
    for (final tensor in decoderTensors.values) {
      tensor.close();
    }
  }
}

Map<String, MlxArray> _sanitizeDecoderTensors(Map<String, MlxArray> raw) {
  final out = <String, MlxArray>{};
  final codebookData = <String, ({MlxArray? usage, MlxArray? sum})>{};
  const eps = 1e-5;

  bool isTransposeConv(String key) {
    final upsample = RegExp(r'^decoder\.upsample\.\d+\.0\.conv\.weight$');
    final block = RegExp(r'^decoder\.decoder\.\d+\.block\.1\.conv\.weight$');
    return upsample.hasMatch(key) || block.hasMatch(key);
  }

  for (final entry in raw.entries) {
    final key = entry.key;
    var value = entry.value;
    if (!key.startsWith('decoder.')) {
      value.close();
      continue;
    }
    if (key.endsWith('._codebook.cluster_usage')) {
      final prefix = key.replaceFirst('._codebook.cluster_usage', '');
      final item = codebookData[prefix];
      codebookData[prefix] = (usage: value, sum: item?.sum);
      continue;
    }
    if (key.endsWith('._codebook.embedding_sum')) {
      final prefix = key.replaceFirst('._codebook.embedding_sum', '');
      final item = codebookData[prefix];
      codebookData[prefix] = (usage: item?.usage, sum: value);
      continue;
    }
    if (key.endsWith('._codebook.initialized')) {
      value.close();
      continue;
    }
    if (value.shape.length == 3 && key.endsWith('.weight')) {
      if (isTransposeConv(key)) {
        final transposed = value.transposeAxes([1, 2, 0]);
        value.close();
        value = transposed;
      } else if (key.contains('.conv.weight') || key.contains('_proj.weight')) {
        final transposed = value.transposeAxes([0, 2, 1]);
        value.close();
        value = transposed;
      }
    }
    out[key] = value;
  }

  for (final entry in codebookData.entries) {
    final usage = entry.value.usage;
    final sum = entry.value.sum;
    if (usage == null || sum == null) {
      usage?.close();
      sum?.close();
      continue;
    }
    final denom = mx.maximum(
      usage.reshape([usage.shape[0], 1]),
      MlxArray.full([], eps),
    );
    final embed = sum / denom;
    denom.close();
    usage.close();
    sum.close();
    out['${entry.key}.codebook.embed.weight'] = embed;
  }

  return out;
}
