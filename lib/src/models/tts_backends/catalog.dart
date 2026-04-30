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
    this.paths = const [],
    this.basename,
    this.requiredForExport = true,
  });

  final String name;
  final String role;
  final String format;
  final String? path;
  final List<String> paths;
  final String? basename;
  final bool requiredForExport;

  String get locator {
    if (path != null) {
      return path!;
    }
    if (paths.isNotEmpty) {
      return paths.join('|');
    }
    return basename!;
  }

  Map<String, Object?> toJson() => {
    'name': name,
    'role': role,
    'format': format,
    if (path != null) 'path': path,
    if (paths.isNotEmpty) 'paths': paths,
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
      readiness == TtsBackendReadiness.ready &&
      (runtime == 'dart_onnx' ||
          runtime == 'dart_ffi_onnx' ||
          runtime == 'dart_ffi_onnx_decoder');

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
      readiness: TtsBackendReadiness.ready,
      runtime: 'dart_onnx',
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
        'CosyVoice-BlankEN/tokenizer.qwen2bpe',
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
        TtsBackendSourceAsset(
          name: 'flow_support',
          role: 'flow_runtime_sidecar',
          format: 'npz',
          path: 'models/CosyVoice2-0.5B/flow_support.npz',
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_sidecar',
          role: 'qwen2_bpe_tokenizer_sidecar',
          format: 'qwen2bpe',
          path: 'models/CosyVoice2-0.5B/CosyVoice-BlankEN/tokenizer.qwen2bpe',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_vocab',
          role: 'qwen2_bpe_vocab',
          format: 'json',
          path: 'models/CosyVoice2-0.5B/CosyVoice-BlankEN/vocab.json',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_merges',
          role: 'qwen2_bpe_merges',
          format: 'text',
          path: 'models/CosyVoice2-0.5B/CosyVoice-BlankEN/merges.txt',
          requiredForExport: false,
        ),
      ],
      notes: [
        'Provider-level Dart synthesis is wired end-to-end through Qwen2 native tokenization, split LLM ONNX, RAS sampling, flow decoding, HiFT, and shared WAV output. CUDA smoke passes for short no-prompt synthesis with both hift.onnx and hift_streaming.onnx. Remaining work is parity/performance hardening, not Python runtime dependency removal.',
      ],
    ),
    TtsBackendCapability(
      provider: 'sarashina2-tts',
      label: 'sarashina2.2-tts',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.ready,
      runtime: 'dart_ffi_onnx_decoder',
      supportsAudioPrompt: true,
      supportsStreaming: false,
      languages: ['ja', 'en', 'mix'],
      localOnnxAssets: [
        'speech_tokenizer_v2.onnx',
        'campplus.onnx',
        'llm_prefill.onnx',
        'llm_decode.onnx',
        'llm_decoder_head.onnx',
        'llm_embeddings.npz',
        'flow.encoder.fp32.onnx',
        'flow.decoder.estimator.fp32.onnx',
        'flow.decoder.step.fp32.onnx',
        'flow_support.npz',
        'hift.onnx',
      ],
      onnxTargets: [
        TtsBackendOnnxTarget(
          name: 'speech_tokenizer_v2',
          role: 'prompt_speech_tokenizer',
          path: 'models/sarashina2.2-tts/speech_tokenizer_v2.onnx',
        ),
        TtsBackendOnnxTarget(
          name: 'campplus',
          role: 'speaker_embedding',
          path: 'models/sarashina2.2-tts/campplus.onnx',
        ),
        TtsBackendOnnxTarget(
          name: 'llm_prefill',
          role: 'semantic_speech_token_generator_prefill',
          path: 'models/sarashina2.2-tts/llm_prefill.onnx',
          sourceNames: ['llm'],
        ),
        TtsBackendOnnxTarget(
          name: 'llm_decode',
          role: 'semantic_speech_token_generator_decode',
          path: 'models/sarashina2.2-tts/llm_decode.onnx',
          sourceNames: ['llm'],
        ),
        TtsBackendOnnxTarget(
          name: 'llm_decoder_head',
          role: 'semantic_speech_token_generator_head',
          path: 'models/sarashina2.2-tts/llm_decoder_head.onnx',
          sourceNames: ['llm'],
        ),
        TtsBackendOnnxTarget(
          name: 'flow_encoder_fp32',
          role: 'flow_token_encoder',
          path: 'models/sarashina2.2-tts/flow.encoder.fp32.onnx',
          sourceNames: ['flow'],
        ),
        TtsBackendOnnxTarget(
          name: 'flow_decoder_estimator_fp32',
          role: 'diffusion_flow_decoder_estimator',
          path: 'models/sarashina2.2-tts/flow.decoder.estimator.fp32.onnx',
          sourceNames: ['flow'],
        ),
        TtsBackendOnnxTarget(
          name: 'flow_decoder_step_fp32',
          role: 'diffusion_flow_decoder_guidance_step',
          path: 'models/sarashina2.2-tts/flow.decoder.step.fp32.onnx',
          requiredForSynthesis: false,
          sourceNames: ['flow_decoder_estimator_fp32'],
        ),
        TtsBackendOnnxTarget(
          name: 'flow_decoder_step_final_fp32',
          role: 'diffusion_flow_decoder_guidance_final_step',
          path: 'models/sarashina2.2-tts/flow.decoder.step.fp32.onnx',
          requiredForSynthesis: false,
          sourceNames: ['flow_decoder_estimator_fp32'],
        ),
        TtsBackendOnnxTarget(
          name: 'hift',
          role: 'vocoder',
          path: 'models/sarashina2.2-tts/hift.onnx',
          sourceNames: ['hift'],
        ),
      ],
      sourceAssets: [
        TtsBackendSourceAsset(
          name: 'llm',
          role: 'llama_semantic_speech_token_generator_source',
          format: 'safetensors',
          path: 'models/sarashina2.2-tts/model.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 'flow',
          role: 'flow_decoder_source',
          format: 'torch_checkpoint',
          path: 'models/sarashina2.2-tts/flow.pt',
        ),
        TtsBackendSourceAsset(
          name: 'hift',
          role: 'vocoder_source',
          format: 'torch_checkpoint',
          path: 'models/sarashina2.2-tts/hift.pt',
        ),
        TtsBackendSourceAsset(
          name: 'campplus',
          role: 'speaker_embedding_source',
          format: 'torch_binary',
          path: 'models/sarashina2.2-tts/campplus_cn_common.bin',
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_json',
          role: 'llama_tokenizer',
          format: 'json',
          path: 'models/sarashina2.2-tts/tokenizer.json',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_model',
          role: 'llama_sentencepiece_tokenizer',
          format: 'sentencepiece',
          path: 'models/sarashina2.2-tts/tokenizer.model',
          requiredForExport: false,
        ),
      ],
      notes: [
        'Provider-level Dart synthesis is wired through the Sarashina Dart tokenizer, native-backed Llama prefill/decode/head ONNX graphs, native semantic sampling, CosyVoice2-style flow decoding, HiFT, and shared WAV output. CUDA smoke passes for external semantic tokens, text-generated semantic tokens, and prompt-conditioned text generation. Remaining work is Python parity, long-form quality/performance measurement, and optional SilentCipher watermark embedding.',
        'The upstream model is a Japanese/English zero-shot TTS Llama checkpoint that emits <|semantic_N|> speech tokens and uses CosyVoice2-style flow + HiFT decoding.',
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
          paths: [
            'models/models--ResembleAI--chatterbox/snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9/t3_mtl23ls_v2.safetensors',
          ],
        ),
        TtsBackendSourceAsset(
          name: 's3gen',
          role: 'speech_token_to_waveform_generator_source',
          format: 'torch_checkpoint',
          paths: [
            'models/models--ResembleAI--chatterbox/snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9/s3gen.pt',
          ],
        ),
        TtsBackendSourceAsset(
          name: 've',
          role: 'voice_encoder_source',
          format: 'safetensors',
          paths: [
            'models/models--ResembleAI--chatterbox/snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9/ve.safetensors',
            'models/models--ResembleAI--chatterbox/snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9/ve.pt',
          ],
        ),
        TtsBackendSourceAsset(
          name: 'conds',
          role: 'built_in_conditionals',
          format: 'torch_checkpoint',
          paths: [
            'models/models--ResembleAI--chatterbox/snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9/conds.pt',
          ],
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'grapheme_tokenizer',
          role: 'text_tokenizer',
          format: 'json',
          paths: [
            'models/models--ResembleAI--chatterbox/snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9/grapheme_mtl_merged_expanded_v1.json',
          ],
          requiredForExport: false,
        ),
      ],
      blockers: [
        'Chatterbox ONNX targets can be exported with egs/chatterbox/setup/export_onnx.sh --component ve|t3|s3gen, but provider-level Dart synthesis is not wired yet. T3 is currently a no-cache prefix-logits graph, and S3Gen is a fixed 4-token chunk graph with an ONNX-only real-valued HiFT export patch that simplifies the source-fusion STFT branch for export; tokenizer, CFG sampling, chunk orchestration, WAV assembly, and S3Gen parity hardening still need a Dart/FFI provider backend before marking ready.',
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
          path: 'models/VibeVoice-Realtime-0.5B/model.safetensors',
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
          path:
              'models/models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7/model.safetensors',
          requiredForExport: false,
        ),
      ],
      blockers: [
        'Dart/FFI ONNX targets are defined for the Dia2 model and audio codec, but the local provider still only has safetensors source weights.',
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
          path:
              'models/models--funasr--campplus/snapshots/fb71fe990cbf6031ae6987a2d76fe64f94377b7e/campplus_cn_common.bin',
        ),
        TtsBackendSourceAsset(
          name: 'semantic_codec',
          role: 'prompt_semantic_codec_source',
          format: 'safetensors',
          path:
              'models/models--amphion--MaskGCT/snapshots/265c6cef07625665d0c28d2faafb1415562379dc/semantic_codec/model.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 'bigvgan',
          role: 'vocoder_source',
          format: 'torch_checkpoint',
          path:
              'models/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513/bigvgan_generator.pt',
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
        'Dart/FFI ONNX targets are defined for IndexTTS2 GPT, S2Mel, speaker encoder, semantic codec, and vocoder, but the local provider still only has PyTorch/safetensors source weights.',
      ],
    ),
    TtsBackendCapability(
      provider: 'neutts-air',
      label: 'NeuTTS Air',
      defaultFormat: 'wav',
      readiness: TtsBackendReadiness.ready,
      runtime: 'dart_ffi_onnx',
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
          format: 'torch_binary',
          path:
              'models/models--neuphonic--neucodec/snapshots/c92ba97d538f2a0baa9118c21ea5de4cdad4e02a/pytorch_model.bin',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_json',
          role: 'qwen2_bpe_tokenizer',
          format: 'json',
          path: 'models/tokenizer.json',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_config',
          role: 'qwen2_added_tokens',
          format: 'json',
          path: 'models/tokenizer_config.json',
          requiredForExport: false,
        ),
        TtsBackendSourceAsset(
          name: 'tokenizer_sidecar',
          role: 'qwen2_bpe_tokenizer_sidecar',
          format: 'qwen2bpe',
          path: 'models/tokenizer.qwen2bpe',
          requiredForExport: false,
        ),
      ],
      blockers: [],
      notes: [
        'Dart/FFI prompt/token helpers, tokenizer.json loading, registry opt-in, and the NeuCodec decoder wrapper are in place.',
        'neucodec_decoder.onnx exports from the local NeuCodec checkpoint with egs/neutts_air/setup/export_decoder_onnx.sh; current exporter supports fixed-frame decoder graphs.',
        'neutts_air_lm.onnx exports from the local Qwen2 checkpoint with egs/neutts_air/setup/export_lm_onnx.sh as a functional no-cache input_ids-to-logits graph; split prefill/decode KV graphs remain the performance target.',
        'Provider-level text synthesis is wired through Dart/FFI prompt/token helpers, native int64 decode-token buffers, no-cache Qwen2 ONNX logits, native top-k/top-p speech-token sampling, NeuCodec ONNX, and shared WAV output. CUDA smoke passes with the current 4-frame decoder export; parity and split-KV performance hardening remain.',
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
          path:
              'models/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe/model.safetensors',
        ),
        TtsBackendSourceAsset(
          name: 'speech_tokenizer',
          role: 'speech_tokenizer_source',
          format: 'safetensors',
          path:
              'models/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe/speech_tokenizer/model.safetensors',
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
