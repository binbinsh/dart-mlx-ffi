import 'catalog.dart';

final class TtsBackendNativePlan {
  const TtsBackendNativePlan({
    required this.providers,
    required this.reuseGroups,
  });

  factory TtsBackendNativePlan.fromCatalog({
    List<TtsBackendCapability> capabilities = TtsBackendCatalog.all,
  }) {
    final providers = [
      for (final capability in capabilities)
        TtsBackendNativeEmbedding.fromCapability(capability),
    ];
    final primitives = <String>{};
    for (final provider in providers) {
      primitives.addAll(provider.nativePrimitives);
    }
    final groups = [
      for (final primitive in primitives.toList()..sort())
        TtsNativeReuseGroup._fromPrimitive(primitive, providers),
    ];
    return TtsBackendNativePlan(providers: providers, reuseGroups: groups);
  }

  final List<TtsBackendNativeEmbedding> providers;
  final List<TtsNativeReuseGroup> reuseGroups;

  Map<String, Object?> toJson() => {
    'providers': [for (final provider in providers) provider.toJson()],
    'reuseGroups': [for (final group in reuseGroups) group.toJson()],
  };
}

final class TtsBackendNativeEmbedding {
  const TtsBackendNativeEmbedding({
    required this.provider,
    required this.readiness,
    required this.runtime,
    required this.onnxTargetCount,
    required this.sourceAssetCount,
    required this.nativePrimitives,
    required this.currentNativePrimitives,
  });

  factory TtsBackendNativeEmbedding.fromCapability(
    TtsBackendCapability capability,
  ) {
    final primitives = _nativePrimitives(capability);
    final current = _currentNativePrimitives(capability.provider, primitives);
    return TtsBackendNativeEmbedding(
      provider: capability.provider,
      readiness: capability.readiness.name,
      runtime: capability.runtime,
      onnxTargetCount: capability.onnxTargets.length,
      sourceAssetCount: capability.sourceAssets.length,
      nativePrimitives: primitives,
      currentNativePrimitives: current,
    );
  }

  final String provider;
  final String readiness;
  final String runtime;
  final int onnxTargetCount;
  final int sourceAssetCount;
  final List<String> nativePrimitives;
  final List<String> currentNativePrimitives;

  List<String> get pendingNativePrimitives => [
    for (final primitive in nativePrimitives)
      if (!currentNativePrimitives.contains(primitive)) primitive,
  ];

  Map<String, Object?> toJson() => {
    'provider': provider,
    'readiness': readiness,
    'runtime': runtime,
    'onnxTargetCount': onnxTargetCount,
    'sourceAssetCount': sourceAssetCount,
    'nativePrimitives': nativePrimitives,
    'currentNativePrimitives': currentNativePrimitives,
    'pendingNativePrimitives': pendingNativePrimitives,
  };
}

final class TtsNativeReuseGroup {
  const TtsNativeReuseGroup({
    required this.primitive,
    required this.providers,
    required this.currentProviders,
    required this.pendingProviders,
    required this.status,
    required this.nextStep,
  });

  factory TtsNativeReuseGroup._fromPrimitive(
    String primitive,
    List<TtsBackendNativeEmbedding> embeddings,
  ) {
    final providers = [
      for (final embedding in embeddings)
        if (embedding.nativePrimitives.contains(primitive)) embedding.provider,
    ];
    final currentProviders = [
      for (final embedding in embeddings)
        if (embedding.currentNativePrimitives.contains(primitive))
          embedding.provider,
    ];
    final pendingProviders = [
      for (final provider in providers)
        if (!currentProviders.contains(provider)) provider,
    ];
    final status = pendingProviders.isEmpty
        ? 'covered'
        : currentProviders.isEmpty
        ? 'planned'
        : 'partial';
    return TtsNativeReuseGroup(
      primitive: primitive,
      providers: providers,
      currentProviders: currentProviders,
      pendingProviders: pendingProviders,
      status: status,
      nextStep:
          _nextStepByPrimitive[primitive] ?? 'Define a shared native FFI ABI.',
    );
  }

  final String primitive;
  final List<String> providers;
  final List<String> currentProviders;
  final List<String> pendingProviders;
  final String status;
  final String nextStep;

  Map<String, Object?> toJson() => {
    'primitive': primitive,
    'providers': providers,
    'currentProviders': currentProviders,
    'pendingProviders': pendingProviders,
    'status': status,
    'nextStep': nextStep,
  };
}

List<String> _nativePrimitives(TtsBackendCapability capability) {
  final text = _capabilitySearchText(capability);
  final primitives = <String>{};
  if (capability.defaultFormat == 'wav') {
    primitives.add('wav_pcm16_io');
  }
  if (capability.onnxTargets.isNotEmpty || capability.isLocalDartOnnxReady) {
    primitives.add('onnx_component_loader');
  }
  if (text.contains('qwen2') || text.contains('qwen3')) {
    primitives.add('qwen_bpe_tokenizer');
  }
  if (text.contains('sentencepiece') || text.contains('llama_tokenizer')) {
    primitives.add('unigram_tokenizer');
  }
  if (text.contains('semantic_') ||
      text.contains('speech_token') ||
      text.contains('codec_token') ||
      text.contains('semantic_codec') ||
      text.contains('audio_token')) {
    primitives.add('token_text_codec');
  }
  if (text.contains('llm_embeddings') ||
      text.contains('token_embedding') ||
      text.contains('text_embedding') ||
      text.contains('speech_embedding') ||
      text.contains('input_embedding') ||
      text.contains('embed_tokens')) {
    primitives.add('embedding_lookup');
  }
  if (text.contains('campplus') || text.contains('speaker_embedding')) {
    primitives.add('speaker_embedding_prompt');
  }
  if (capability.supportsAudioPrompt ||
      text.contains('prompt_speech') ||
      text.contains('mel') ||
      text.contains('fbank') ||
      text.contains('campplus') ||
      text.contains('voice_encoder')) {
    primitives.add('audio_prompt_features');
    primitives.add('tensor_signal_ops');
  }
  if (text.contains('flow') ||
      text.contains('diffusion') ||
      text.contains('s2mel') ||
      text.contains('semantic_to_mel')) {
    primitives.add('flow_diffusion_ops');
  }
  if (text.contains('token_generator') ||
      text.contains('llm_') ||
      text.contains(' gpt') ||
      text.contains('semantic_token_generator') ||
      text.contains('tts_lm') ||
      capability.provider == 'vibevoice-rt' ||
      capability.provider == 'dia2') {
    primitives.add('autoregressive_decode_loop');
    primitives.add('sampling_filters');
  }
  if (text.contains('vocoder') ||
      text.contains('waveform') ||
      text.contains('codec') ||
      text.contains('bigvgan') ||
      text.contains('hift') ||
      text.contains('mimi')) {
    primitives.add('codec_vocoder_decode');
  }
  if (capability.supportsStreaming) {
    primitives.add('streaming_state_cache');
  }
  final sorted = primitives.toList()..sort();
  return sorted;
}

String _capabilitySearchText(TtsBackendCapability capability) {
  final values = <String>[
    capability.provider,
    capability.label,
    capability.runtime,
    ...capability.localOnnxAssets,
    for (final target in capability.onnxTargets) ...[
      target.name,
      target.role,
      target.path,
      ...target.sourceNames,
    ],
    for (final source in capability.sourceAssets) ...[
      source.name,
      source.role,
      source.format,
      ?source.path,
      ...source.paths,
      ?source.basename,
    ],
  ];
  return values.join(' ').toLowerCase();
}

List<String> _currentNativePrimitives(
  String provider,
  List<String> candidates,
) {
  final current = _implementedPrimitivesByProvider[provider] ?? const {};
  return [
    for (final primitive in candidates)
      if (current.contains(primitive)) primitive,
  ];
}

const _implementedPrimitivesByProvider = <String, Set<String>>{
  'kokoro': {'onnx_component_loader', 'token_text_codec', 'wav_pcm16_io'},
  'cosyvoice2': {
    'audio_prompt_features',
    'codec_vocoder_decode',
    'embedding_lookup',
    'flow_diffusion_ops',
    'onnx_component_loader',
    'qwen_bpe_tokenizer',
    'sampling_filters',
    'speaker_embedding_prompt',
    'streaming_state_cache',
    'tensor_signal_ops',
    'wav_pcm16_io',
  },
  'sarashina2-tts': {
    'audio_prompt_features',
    'codec_vocoder_decode',
    'embedding_lookup',
    'flow_diffusion_ops',
    'onnx_component_loader',
    'sampling_filters',
    'speaker_embedding_prompt',
    'tensor_signal_ops',
    'token_text_codec',
    'unigram_tokenizer',
    'wav_pcm16_io',
  },
  'neutts-air': {
    'codec_vocoder_decode',
    'onnx_component_loader',
    'qwen_bpe_tokenizer',
    'sampling_filters',
    'token_text_codec',
    'wav_pcm16_io',
  },
};

const _nextStepByPrimitive = <String, String>{
  'audio_prompt_features':
      'Keep mel/fbank/resample/prompt packing in native and share the prompt ABI across speaker-conditioned models.',
  'autoregressive_decode_loop':
      'Move per-step decode input assembly, sampled-token append, and KV bookkeeping behind provider-specific native drivers.',
  'codec_vocoder_decode':
      'Standardize codec-token and mel-to-wave decoder wrappers around native tensors plus shared WAV chunk output.',
  'embedding_lookup':
      'Use the shared native row-embedding lookup for text, speech, and codec token tables.',
  'flow_diffusion_ops':
      'Share flow/diffusion timestep, channel-major mel packing, estimator static inputs, and CFG update loops.',
  'onnx_component_loader':
      'Use the generic TtsOnnxComponentBundle first, then add a provider-level Dart orchestrator before marking ready.',
  'qwen_bpe_tokenizer':
      'Route all Qwen-family BPE tokenizers through the existing native sidecar/handle ABI.',
  'sampling_filters':
      'Reuse native top-k/top-p/repetition/frequency-penalty helpers instead of Dart typed-list sampling loops.',
  'speaker_embedding_prompt':
      'Share CampPlus/voice-encoder prompt extraction and embedding cache contracts.',
  'streaming_state_cache':
      'Move chunk caches, overlap windows, and streaming decoder state to provider-native structs.',
  'tensor_signal_ops':
      'Share resample, transpose, CMN, slice, mask, and small tensor packing loops in native.',
  'token_text_codec':
      'Use native parsers/formatters for speech, semantic, and codec token text surfaces.',
  'unigram_tokenizer':
      'Keep SentencePiece/Unigram-compatible tokenization behind a Dart tokenizer handle.',
  'wav_pcm16_io':
      'Use the shared native audio WAV/concat helpers for every local WAV provider.',
};
