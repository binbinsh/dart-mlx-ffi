enum TtsBackendReadiness { ready, partial, blocked, remoteApi }

final class TtsBackendOnnxTarget {
  const TtsBackendOnnxTarget({
    required this.name,
    required this.role,
    required this.path,
    this.requiredForSynthesis = true,
    this.sourceNames = const [],
  });

  final String name;
  final String role;
  final String path;
  final bool requiredForSynthesis;
  final List<String> sourceNames;

  String get basename {
    final slash = path.lastIndexOf('/');
    return slash < 0 ? path : path.substring(slash + 1);
  }

  Map<String, Object?> toJson() => {
    'name': name,
    'role': role,
    'path': path,
    'format': 'onnx',
    'requiredForSynthesis': requiredForSynthesis,
    if (sourceNames.isNotEmpty) 'sourceNames': sourceNames,
  };
}

final class TtsBackendSourceAsset {
  const TtsBackendSourceAsset({
    required this.name,
    required this.role,
    required this.format,
    this.path,
    this.basename,
    this.requiredForExport = true,
  }) : assert(path != null || basename != null);

  final String name;
  final String role;
  final String format;
  final String? path;
  final String? basename;
  final bool requiredForExport;

  String get locator => path ?? basename!;

  Map<String, Object?> toJson() => {
    'name': name,
    'role': role,
    'format': format,
    if (path != null) 'path': path,
    if (basename != null) 'basename': basename,
    'requiredForExport': requiredForExport,
  };
}

final class TtsBackendCapability {
  const TtsBackendCapability({
    required this.provider,
    required this.label,
    required this.defaultFormat,
    required this.readiness,
    required this.runtime,
    this.supportsAudioPrompt = false,
    this.supportsStreaming = false,
    this.supportsTextInstructions = false,
    this.requiresApiKey = false,
    this.languages = const [],
    this.localOnnxAssets = const [],
    this.onnxTargets = const [],
    this.sourceAssets = const [],
    this.blockers = const [],
    this.notes = const [],
  });

  final String provider;
  final String label;
  final String defaultFormat;
  final TtsBackendReadiness readiness;
  final String runtime;
  final bool supportsAudioPrompt;
  final bool supportsStreaming;
  final bool supportsTextInstructions;
  final bool requiresApiKey;
  final List<String> languages;
  final List<String> localOnnxAssets;
  final List<TtsBackendOnnxTarget> onnxTargets;
  final List<TtsBackendSourceAsset> sourceAssets;
  final List<String> blockers;
  final List<String> notes;

  bool get isLocalDartOnnxReady =>
      readiness == TtsBackendReadiness.ready && runtime == 'dart_onnx';

  Map<String, Object?> toJson() => {
    'provider': provider,
    'label': label,
    'defaultFormat': defaultFormat,
    'readiness': readiness.name,
    'runtime': runtime,
    'supportsAudioPrompt': supportsAudioPrompt,
    'supportsStreaming': supportsStreaming,
    'supportsTextInstructions': supportsTextInstructions,
    'requiresApiKey': requiresApiKey,
    'languages': languages,
    'localOnnxAssets': localOnnxAssets,
    if (onnxTargets.isNotEmpty)
      'onnxTargets': [for (final target in onnxTargets) target.toJson()],
    if (sourceAssets.isNotEmpty)
      'sourceAssets': [for (final source in sourceAssets) source.toJson()],
    'blockers': blockers,
    'notes': notes,
  };

  Map<String, Object?> toProviderCard() => {
    'provider': provider,
    'displayName': label,
    'defaultFormat': defaultFormat,
    'supportsAudioPrompt': supportsAudioPrompt,
    'supportsStreaming': supportsStreaming,
    'supportsTextInstructions': supportsTextInstructions,
    'requiresApiKey': requiresApiKey,
    'readiness': {
      'ok': readiness == TtsBackendReadiness.ready,
      'state': readiness.name,
      'runtime': runtime,
      'python': false,
      if (blockers.isNotEmpty) 'blockers': blockers,
      if (notes.isNotEmpty) 'notes': notes,
    },
  };
}

final class TtsBackendCatalog {
  const TtsBackendCatalog._();

  static const all = <TtsBackendCapability>[
    TtsBackendCapability(
      provider: 'kokoro',
      label: 'Kokoro ONNX',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.ready,
      runtime: 'dart_onnx',
      languages: ['en', 'zh', 'ja', 'es', 'fr', 'hi', 'it', 'pt'],
      localOnnxAssets: ['kokoro-v1.0.onnx', 'voices.npz', 'config.json'],
      notes: [
        'UniFrontend structured ONNX, Kokoro ONNX, and eSpeak-NG phonemization are wired through Dart FFI/control flow.',
      ],
    ),
    TtsBackendCapability(
      provider: 'cosyvoice2',
      label: 'CosyVoice2-0.5B',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.partial,
      runtime: 'dart_onnx_partial',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      supportsTextInstructions: true,
      languages: ['zh', 'en', 'ja', 'ko', 'mix'],
      localOnnxAssets: [
        'campplus.onnx',
        'speech_tokenizer_v2.onnx',
        'flow.decoder.estimator.fp32.onnx',
        'flow.encoder.fp32.onnx',
        'llm_prefill.onnx',
        'llm_decode.onnx',
        'llm_decoder_head.onnx',
        'llm_embeddings.npz',
        'hift.onnx',
        'hift_streaming.onnx',
      ],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 'campplus',
          role: 'speaker_embedding',
          path: 'models/CosyVoice2-0.5B/campplus.onnx',
        ),
        TtsBackendOnnxTarget(
          name: 'speech_tokenizer_v2',
          role: 'prompt_speech_tokenizer',
          path: 'models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx',
        ),
        TtsBackendOnnxTarget(
          name: 'flow_decoder_estimator_fp32',
          role: 'diffusion_flow_decoder_estimator',
          path: 'models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.onnx',
        ),
        TtsBackendOnnxTarget(
          name: 'flow_encoder_fp32',
          role: 'flow_token_encoder',
          path: 'models/CosyVoice2-0.5B/flow.encoder.fp32.onnx',
          sourceNames: ['flow_encoder_fp32'],
        ),
        TtsBackendOnnxTarget(
          name: 'llm_prefill',
          role: 'semantic_speech_token_generator_prefill',
          path: 'models/CosyVoice2-0.5B/llm_prefill.onnx',
          sourceNames: ['llm'],
        ),
        TtsBackendOnnxTarget(
          name: 'llm_decode',
          role: 'semantic_speech_token_generator_decode',
          path: 'models/CosyVoice2-0.5B/llm_decode.onnx',
          sourceNames: ['llm'],
        ),
        TtsBackendOnnxTarget(
          name: 'llm_decoder_head',
          role: 'semantic_speech_token_generator_head',
          path: 'models/CosyVoice2-0.5B/llm_decoder_head.onnx',
          sourceNames: ['llm'],
        ),
        TtsBackendOnnxTarget(
          name: 'hift',
          role: 'vocoder',
          path: 'models/CosyVoice2-0.5B/hift.onnx',
          sourceNames: ['hift'],
        ),
        TtsBackendOnnxTarget(
          name: 'hift_streaming',
          role: 'vocoder_streaming',
          path: 'models/CosyVoice2-0.5B/hift_streaming.onnx',
          sourceNames: ['hift'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 'flow_encoder_fp32',
          role: 'flow_token_encoder_source',
          format: 'torchscript_zip',
          path: 'models/CosyVoice2-0.5B/flow.encoder.fp32.zip',
        ),
        TtsBackendSourceAsset(
          name: 'llm',
          role: 'semantic_speech_token_generator_source',
          format: 'torch_checkpoint',
          path: 'models/CosyVoice2-0.5B/llm.pt',
        ),
        TtsBackendSourceAsset(
          name: 'hift',
          role: 'vocoder_source',
          format: 'torch_checkpoint',
          path: 'models/CosyVoice2-0.5B/hift.pt',
        ),
      ],
      blockers: [
        'All ONNX graphs are exported (campplus, speech_tokenizer_v2, flow.decoder.estimator, flow.encoder, llm_prefill, llm_decode, llm_decoder_head, hift, hift_streaming) plus llm_embeddings.npz for token-table lookups. Remaining work: Zig glue for the autoregressive decode loop (prefill -> head -> sample -> embed -> decode -> head -> sample -> ...) using ras_sampling, then wire hift_streaming for cross-chunk streaming.',
      ],
    ),
    TtsBackendCapability(
      provider: 'chatterbox',
      label: 'Chatterbox Turbo (ResembleAI)',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsAudioPrompt: true,
      languages: ['en', 'zh', 'ja', 'ko', 'fr', 'de', 'es', 'it'],
      localOnnxAssets: ['t3_mtl23ls_v2.onnx', 's3gen.onnx', 've.onnx'],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 't3_mtl23ls_v2',
          role: 'text_to_speech_token_generator',
          path: 'models/onnx/t3_mtl23ls_v2.onnx',
          sourceNames: ['t3_mtl23ls_v2'],
        ),
        TtsBackendOnnxTarget(
          name: 's3gen',
          role: 'speech_token_to_waveform_generator',
          path: 'models/onnx/s3gen.onnx',
          sourceNames: ['s3gen'],
        ),
        TtsBackendOnnxTarget(
          name: 've',
          role: 'voice_encoder',
          path: 'models/onnx/ve.onnx',
          sourceNames: ['ve'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 't3_mtl23ls_v2',
          role: 'text_to_speech_token_generator_source',
          format: 'safetensors',
          basename: 't3_mtl23ls_v2.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 's3gen',
          role: 'speech_token_to_waveform_generator_source',
          format: 'torch_checkpoint',
          basename: 's3gen.pt',
        ),
        TtsBackendSourceAsset(
          name: 've',
          role: 'voice_encoder_source',
          format: 'torch_checkpoint',
          basename: 've.pt',
        ),
        TtsBackendSourceAsset(
          name: 'conds',
          role: 'built_in_conditionals',
          format: 'torch_checkpoint',
          basename: 'conds.pt',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'grapheme_tokenizer',
          role: 'text_tokenizer',
          format: 'json',
          basename: 'grapheme_mtl_merged_expanded_v1.json',
          requiredForExport: false,
        ),
      ],
      blockers: [
        'Dart/Zig ONNX targets are defined for the Chatterbox T3 generator, S3Gen decoder, and voice encoder, but the local provider still only has PyTorch/Hugging Face source weights.',
      ],
    ),
    TtsBackendCapability(
      provider: 'glm-tts',
      label: 'GLM-TTS (Z.AI)',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      languages: ['zh', 'en'],
      blockers: ['No local complete ONNX runtime assets are present.'],
    ),
    TtsBackendCapability(
      provider: 'cosyvoice3',
      label: 'Fun-CosyVoice3-0.5B',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      supportsTextInstructions: true,
      languages: ['zh', 'en', 'ja', 'ko', 'mix'],
      blockers: ['No local complete ONNX runtime assets are present.'],
    ),
    TtsBackendCapability(
      provider: 'vibevoice-rt',
      label: 'VibeVoice-Realtime-0.5B',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsStreaming: true,
      languages: ['en'],
      localOnnxAssets: ['model.onnx'],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 'streaming_tts_lm',
          role: 'streaming_text_to_audio_token_generator',
          path: 'models/onnx/model.onnx',
          sourceNames: ['model'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 'model',
          role: 'streaming_text_to_audio_token_generator_source',
          format: 'safetensors',
          basename: 'model.safetensors',
        ),
      ],
      blockers: ['No local complete ONNX runtime assets are present.'],
    ),
    TtsBackendCapability(
      provider: 'dia2',
      label: 'Dia2-2B',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      languages: ['en'],
      localOnnxAssets: ['model.onnx', 'mimi.onnx'],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 'dia2',
          role: 'text_audio_token_generator',
          path: 'models/Dia2-2B/model.onnx',
          sourceNames: ['dia2'],
        ),
        TtsBackendOnnxTarget(
          name: 'mimi',
          role: 'audio_codec',
          path: 'models/onnx/mimi.onnx',
          sourceNames: ['mimi'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 'dia2',
          role: 'text_audio_token_generator_source',
          format: 'safetensors',
          path: 'models/Dia2-2B/model.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 'mimi',
          role: 'audio_codec_source',
          format: 'safetensors',
          basename: 'model.safetensors',
          requiredForExport: false,
        ),
      ],
      blockers: [
        'Dart/Zig ONNX targets are defined for the Dia2 model and audio codec, but the local provider still only has safetensors source weights.',
      ],
    ),
    TtsBackendCapability(
      provider: 'elevenlabs3',
      label: 'ElevenLabs-v3 via Poe',
      defaultFormat: 'mp3',
      readiness: TtsBackendReadiness.remoteApi,
      runtime: 'remote_api',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      supportsTextInstructions: true,
      requiresApiKey: true,
      blockers: [
        'Remote provider; there is no local model to migrate to Dart ONNX.',
      ],
    ),
    TtsBackendCapability(
      provider: 'indextts2',
      label: 'Bilibili IndexTTS2',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      languages: ['auto', 'en', 'zh'],
      localOnnxAssets: [
        'gpt.onnx',
        's2mel.onnx',
        'campplus.onnx',
        'semantic_codec.onnx',
        'bigvgan.onnx',
      ],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 'gpt',
          role: 'semantic_token_generator',
          path: 'models/onnx/gpt.onnx',
          sourceNames: ['gpt'],
        ),
        TtsBackendOnnxTarget(
          name: 's2mel',
          role: 'semantic_to_mel_generator',
          path: 'models/onnx/s2mel.onnx',
          sourceNames: ['s2mel'],
        ),
        TtsBackendOnnxTarget(
          name: 'campplus',
          role: 'speaker_embedding',
          path: 'models/onnx/campplus.onnx',
          sourceNames: ['campplus'],
        ),
        TtsBackendOnnxTarget(
          name: 'semantic_codec',
          role: 'prompt_semantic_codec',
          path: 'models/onnx/semantic_codec.onnx',
          sourceNames: ['semantic_codec'],
        ),
        TtsBackendOnnxTarget(
          name: 'bigvgan',
          role: 'vocoder',
          path: 'models/onnx/bigvgan.onnx',
          sourceNames: ['bigvgan'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 'gpt',
          role: 'semantic_token_generator_source',
          format: 'torch_checkpoint',
          path: 'models/gpt.pth',
        ),
        TtsBackendSourceAsset(
          name: 's2mel',
          role: 'semantic_to_mel_generator_source',
          format: 'torch_checkpoint',
          path: 'models/s2mel.pth',
        ),
        TtsBackendSourceAsset(
          name: 'campplus',
          role: 'speaker_embedding_source',
          format: 'torch_binary',
          basename: 'campplus_cn_common.bin',
        ),
        TtsBackendSourceAsset(
          name: 'semantic_codec',
          role: 'prompt_semantic_codec_source',
          format: 'safetensors',
          basename: 'model.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 'bigvgan',
          role: 'vocoder_source',
          format: 'torch_checkpoint',
          basename: 'bigvgan_generator.pt',
        ),
        TtsBackendSourceAsset(
          name: 'qwen0_6b_emo',
          role: 'emotion_text_lm_source',
          format: 'safetensors',
          path: 'models/qwen0.6bemo4-merge/model.safetensors',
          requiredForExport: false,
        ),
      ],
      blockers: [
        'Dart/Zig ONNX targets are defined for IndexTTS2 GPT, S2Mel, speaker encoder, semantic codec, and vocoder, but the local provider still only has PyTorch/safetensors source weights.',
      ],
    ),
    TtsBackendCapability(
      provider: 'neutts-air',
      label: 'NeuTTS Air',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      languages: ['en'],
      localOnnxAssets: ['neutts_air_lm.onnx', 'neucodec_decoder.onnx'],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 'neutts_air_lm',
          role: 'text_to_codec_token_generator',
          path: 'models/onnx/neutts_air_lm.onnx',
          sourceNames: ['neutts_air_lm', 'neutts_air_gguf'],
        ),
        TtsBackendOnnxTarget(
          name: 'neucodec_decoder',
          role: 'codec_token_to_waveform_decoder',
          path: 'models/onnx/neucodec_decoder.onnx',
          sourceNames: ['neucodec_decoder'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 'neutts_air_lm',
          role: 'text_to_codec_token_generator_source',
          format: 'safetensors',
          path: 'models/model.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 'neutts_air_gguf',
          role: 'text_to_codec_token_generator_quantized_source',
          format: 'gguf',
          path: 'models/neutss-air-BF16.gguf',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'neucodec_decoder',
          role: 'codec_token_to_waveform_decoder_source',
          format: 'safetensors',
          basename: 'pytorch_model.bin',
          requiredForExport: false,
        ),
      ],
      blockers: [
        'Dart/Zig ONNX targets are defined for the NeuTTS LM and NeuCodec decoder, but the local provider still only has safetensors/GGUF/source codec assets.',
      ],
    ),
    TtsBackendCapability(
      provider: 'qwen3-tts',
      label: 'Qwen3-TTS (Alibaba)',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.blocked,
      runtime: 'not_migrated',
      supportsAudioPrompt: true,
      supportsStreaming: true,
      supportsTextInstructions: true,
      languages: ['auto', 'zh', 'en', 'ja', 'ko'],
      localOnnxAssets: ['qwen3_tts.onnx', 'speech_tokenizer.onnx'],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 'qwen3_tts',
          role: 'text_to_speech_token_generator',
          path: 'models/onnx/qwen3_tts.onnx',
          sourceNames: ['qwen3_tts'],
        ),
        TtsBackendOnnxTarget(
          name: 'speech_tokenizer',
          role: 'speech_tokenizer',
          path: 'models/onnx/speech_tokenizer.onnx',
          sourceNames: ['speech_tokenizer'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 'qwen3_tts',
          role: 'text_to_speech_token_generator_source',
          format: 'safetensors',
          basename: 'model.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 'speech_tokenizer',
          role: 'speech_tokenizer_source',
          format: 'safetensors',
          basename: 'model.safetensors',
          requiredForExport: false,
        ),
      ],
      blockers: ['No local complete ONNX runtime assets are present.'],
    ),
    TtsBackendCapability(
      provider: 'sonic3',
      label: 'Sonic 3.0 via Poe',
      defaultFormat: 'mp3',
      readiness: TtsBackendReadiness.remoteApi,
      runtime: 'remote_api',
      supportsStreaming: true,
      supportsTextInstructions: true,
      requiresApiKey: true,
      blockers: [
        'Remote provider; there is no local model to migrate to Dart ONNX.',
      ],
    ),
  ];

  static List<TtsBackendCapability> get localDartOnnxReady =>
      all.where((backend) => backend.isLocalDartOnnxReady).toList();

  static TtsBackendCapability? byProvider(String provider) {
    for (final backend in all) {
      if (backend.provider == provider) {
        return backend;
      }
    }
    return null;
  }

  static List<Map<String, Object?>> providerCards({
    bool includeUnavailable = true,
  }) {
    final providers = includeUnavailable
        ? all
        : all.where(
            (backend) => backend.readiness == TtsBackendReadiness.ready,
          );
    return [for (final backend in providers) backend.toProviderCard()];
  }

  static Map<String, Object?> toJson() => {
    'runtime': 'dart',
    'python': false,
    'providers': [for (final backend in all) backend.toJson()],
  };
}
