enum TtsBackendReadiness { ready, partial, blocked, remoteApi }

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
      ],
      blockers: [
        'llm.pt, flow.pt, and hift.pt still need complete ONNX export or a Dart-native implementation before end-to-end synthesis can run without Python.',
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
      blockers: [
        'Current local assets are PyTorch/Hugging Face blobs, not a complete ONNX graph set.',
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
      blockers: [
        'Current local assets are safetensors plus codec assets, not a complete ONNX graph set.',
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
      blockers: [
        'Current local assets are PyTorch .pth/.pt checkpoints, not a complete ONNX graph set.',
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
      blockers: [
        'Current local asset is model.safetensors, not a complete ONNX graph set.',
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
