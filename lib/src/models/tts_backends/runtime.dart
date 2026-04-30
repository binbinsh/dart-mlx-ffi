import 'dart:typed_data';

import '../cosyvoice2/cosyvoice2_audio.dart';
import '../cosyvoice2/cosyvoice2_runtime.dart';
import '../neutts_air/neutts_air.dart';
import '../neutts_air/neutts_air_runtime.dart';
import '../sarashina2/sarashina2.dart';
import '../sarashina2/sarashina2_runtime.dart';
import '../unifrontend/tts.dart';
import 'catalog.dart';

final class DartTtsSynthesisRequest {
  const DartTtsSynthesisRequest({
    required this.provider,
    this.text = '',
    this.phonemes = '',
    this.voice = 'zf_xiaoni',
    this.speed = 1.0,
    this.promptAudioBytes,
    this.promptText = '',
    this.maxGeneratedTokens,
    this.rasSeed = 0,
    this.useStreamingHift = false,
    this.semanticTokenText = '',
    this.semanticTokens = const [],
    this.codecTokenText = '',
    this.codecTokens = const [],
    this.referencePhones = '',
    this.inputPhones = '',
    this.referenceCodes = const [],
    this.promptTokenIds = const [],
    this.latencyTokens = 1,
    this.temperature = sarashina2DefaultTemperature,
    this.topP = sarashina2DefaultTopP,
    this.frequencyPenalty = sarashina2DefaultFrequencyPenalty,
  });

  final String provider;
  final String text;
  final String phonemes;
  final String voice;
  final double speed;
  final Uint8List? promptAudioBytes;
  final String promptText;
  final int? maxGeneratedTokens;
  final int rasSeed;
  final bool useStreamingHift;
  final String semanticTokenText;
  final List<int> semanticTokens;
  final String codecTokenText;
  final List<int> codecTokens;
  final String referencePhones;
  final String inputPhones;
  final List<int> referenceCodes;
  final List<int> promptTokenIds;
  final int latencyTokens;
  final double temperature;
  final double topP;
  final double frequencyPenalty;
}

final class DartTtsSynthesisResult {
  const DartTtsSynthesisResult({
    required this.provider,
    required this.runtime,
    required this.python,
    required this.text,
    required this.frontendText,
    this.frontendSsml = '',
    required this.phonemes,
    required this.audioBytes,
    required this.audioFormat,
    required this.frontendElapsedMicroseconds,
    required this.ttsElapsedMicroseconds,
    required this.frontendProvider,
    required this.phonemizerBackend,
    required this.requestedVoice,
    required this.resolvedVoice,
    required this.phonemeTokenCount,
    required this.phonemeChunkCount,
    required this.warnings,
    this.metadata = const {},
  });

  factory DartTtsSynthesisResult.fromUniFrontendKokoro(
    UniFrontendKokoroTtsResult result, {
    required String phonemizerBackend,
  }) {
    return DartTtsSynthesisResult(
      provider: 'kokoro',
      runtime: 'dart',
      python: false,
      text: result.text,
      frontendText: result.frontendText,
      phonemes: result.phonemes,
      audioBytes: result.audioWav,
      audioFormat: 'wav',
      frontendElapsedMicroseconds: result.frontendElapsedMicroseconds,
      ttsElapsedMicroseconds: result.ttsElapsedMicroseconds,
      frontendProvider: result.frontendProvider,
      phonemizerBackend: phonemizerBackend,
      requestedVoice: result.requestedVoice,
      resolvedVoice: result.resolvedVoice,
      phonemeTokenCount: result.phonemeTokenCount,
      phonemeChunkCount: result.phonemeChunkCount,
      warnings: result.warnings,
    );
  }

  final String provider;
  final String runtime;
  final bool python;
  final String text;
  final String frontendText;
  final String frontendSsml;
  final String phonemes;
  final Uint8List audioBytes;
  final String audioFormat;
  final int frontendElapsedMicroseconds;
  final int ttsElapsedMicroseconds;
  final String frontendProvider;
  final String phonemizerBackend;
  final String requestedVoice;
  final String resolvedVoice;
  final int phonemeTokenCount;
  final int phonemeChunkCount;
  final List<String> warnings;
  final Map<String, Object?> metadata;
}

final class CosyVoice2TtsBackend implements DartTtsBackend {
  const CosyVoice2TtsBackend(this.runtime);

  final CosyVoice2DartRuntime runtime;

  @override
  TtsBackendCapability get capability =>
      TtsBackendCatalog.byProvider('cosyvoice2')!;

  @override
  List<String> get voiceNames => const [];

  @override
  Map<String, Object?> runtimeHealth() => {
    'provider': capability.provider,
    'runtime': 'dart',
    'python': false,
    'loadedComponents': runtime.loadedComponentNames,
    'selectedProviders': runtime.selectedProviders,
  };

  @override
  Future<DartTtsSynthesisResult> synthesize(
    DartTtsSynthesisRequest request,
  ) async {
    final prompt = request.promptAudioBytes == null
        ? null
        : decodeWav(request.promptAudioBytes!);
    final result = await runtime.synthesize(
      text: request.text,
      promptAudio: prompt,
      promptText: request.promptText,
      maxGeneratedTokens: request.maxGeneratedTokens,
      rasSeed: request.rasSeed,
      useStreamingHift: request.useStreamingHift,
      includeFloatOutputs: false,
      includeTokenMetadata: false,
    );
    return DartTtsSynthesisResult(
      provider: 'cosyvoice2',
      runtime: 'dart',
      python: false,
      text: result.text,
      frontendText: result.text,
      phonemes: '',
      audioBytes: result.audioWavBytes,
      audioFormat: 'wav',
      frontendElapsedMicroseconds: result.promptElapsedMicroseconds,
      ttsElapsedMicroseconds:
          result.llmElapsedMicroseconds + result.flowElapsedMicroseconds,
      frontendProvider: 'none',
      phonemizerBackend: 'none',
      requestedVoice: request.voice,
      resolvedVoice: result.usedPrompt ? 'prompt' : 'zero_shot',
      phonemeTokenCount: result.generatedSpeechTokenCount,
      phonemeChunkCount: (result.generatedSpeechTokenCount / 25).ceil(),
      warnings: [
        if (!result.usedPrompt)
          'CosyVoice2 ran without a prompt audio; output voice is uncontrolled.',
        if (result.usedStreamingHift) 'CosyVoice2 used hift_streaming.onnx.',
      ],
      metadata: {
        if (result.generatedSpeechTokens.isNotEmpty)
          'generatedSpeechTokens': result.generatedSpeechTokens,
        'generatedSpeechTokenCount': result.generatedSpeechTokenCount,
        'audioSamples': result.audioSampleCount,
        'melFrames': result.melFrames,
        'llmElapsedMs': result.llmElapsedMicroseconds / 1000.0,
        'flowElapsedMs': result.flowElapsedMicroseconds / 1000.0,
        'usedPrompt': result.usedPrompt,
        'usedStreamingHift': result.usedStreamingHift,
      },
    );
  }

  @override
  void close() {
    runtime.close();
  }
}

final class NeuttsAirTtsBackend implements DartTtsBackend {
  const NeuttsAirTtsBackend(this.runtime);

  final NeuttsAirDartRuntime runtime;

  @override
  TtsBackendCapability get capability =>
      TtsBackendCatalog.byProvider(neuttsAirProvider)!;

  @override
  List<String> get voiceNames => const [];

  @override
  Map<String, Object?> runtimeHealth() => {
    'provider': capability.provider,
    'runtime': runtime.lmLoaded && runtime.codecDecoderLoaded
        ? 'dart_ffi_onnx'
        : 'dart_ffi_onnx_partial',
    'python': false,
    'loadedComponents': runtime.loadedComponentNames,
    'selectedProviders': runtime.selectedProviders,
    'tokenizerLoaded': runtime.tokenizerLoaded,
    'codecDecoderLoaded': runtime.codecDecoderLoaded,
    'lmLoaded': runtime.lmLoaded,
  };

  @override
  Future<DartTtsSynthesisResult> synthesize(
    DartTtsSynthesisRequest request,
  ) async {
    final tokens = codecTokenSourceFromNeuttsAirRequest(
      codecTokens: request.codecTokens,
      codecTokenText: request.codecTokenText,
      includeTokenMetadata: false,
    );
    late final NeuttsAirDecodeResult result;
    try {
      if (tokens.isEmpty) {
        result = await runtime.synthesizeText(
          text: request.text,
          referencePhones: request.referencePhones,
          inputPhones: request.inputPhones,
          referenceCodes: request.referenceCodes,
          maxGeneratedTokens: request.maxGeneratedTokens,
          seed: request.rasSeed,
          temperature: request.temperature,
          topP: request.topP,
          includeCodecMetadata: false,
          includePromptMetadata: false,
        );
      } else {
        result = await runtime.decodeCodecTokens(
          text: request.text,
          codecTokens: tokens.source,
          codecTokenValues: tokens.tokens,
          referencePhones: request.referencePhones,
          inputPhones: request.inputPhones,
          referenceCodes: request.referenceCodes,
          includeCodecMetadata: false,
          includePromptMetadata: false,
          validateCodecTokens: false,
        );
      }
    } finally {
      tokens.close();
    }
    return DartTtsSynthesisResult(
      provider: neuttsAirProvider,
      runtime: runtime.lmLoaded && runtime.codecDecoderLoaded
          ? 'dart_ffi_onnx'
          : 'dart_ffi_onnx_partial',
      python: false,
      text: result.text,
      frontendText: result.text,
      phonemes: '',
      audioBytes: result.audioWavBytes,
      audioFormat: 'wav',
      frontendElapsedMicroseconds: result.promptElapsedMicroseconds,
      ttsElapsedMicroseconds: result.decodeElapsedMicroseconds,
      frontendProvider: 'none',
      phonemizerBackend: 'none',
      requestedVoice: request.voice,
      resolvedVoice: result.promptTokenCount == 0 ? 'codec_tokens' : 'prompt',
      phonemeTokenCount: result.codecTokenCount,
      phonemeChunkCount: (result.codecTokenCount / 50).ceil(),
      warnings: result.generatedFromLm
          ? const [
              'NeuTTS Air text synthesis used the no-cache LM ONNX path; split KV decode is not optimized yet.',
            ]
          : const [
              'NeuTTS Air decoded external codec tokens; LM text generation was bypassed.',
            ],
      metadata: {
        if (result.codecTokens.isNotEmpty) 'codecTokens': result.codecTokens,
        if (result.codecTokens.isNotEmpty)
          'codecTokenText': neuttsAirSpeechTokensToText(result.codecTokens),
        'codecTokenCount': result.codecTokenCount,
        'generatedFromLm': result.generatedFromLm,
        if (result.lmProvider != null) 'lmProvider': result.lmProvider,
        if (result.lmElapsedMicroseconds > 0)
          'lmElapsedMs': result.lmElapsedMicroseconds / 1000.0,
        if (result.lmInputTokenCount > 0)
          'lmInputTokenCount': result.lmInputTokenCount,
        'codecDecoderProvider': result.codecDecoderProvider,
        if (result.codecDecoderFrameCapacity != null)
          'codecDecoderFrameCapacity': result.codecDecoderFrameCapacity,
        if (result.promptTokenIds.isNotEmpty)
          'promptTokenIds': result.promptTokenIds,
        'promptTokenCount': result.promptTokenCount,
        'sampleRate': result.sampleRate,
        'promptElapsedMs': result.promptElapsedMicroseconds / 1000.0,
        'decodeElapsedMs': result.decodeElapsedMicroseconds / 1000.0,
      },
    );
  }

  @override
  void close() {
    runtime.close();
  }
}

final class Sarashina2TtsBackend implements DartTtsBackend {
  Sarashina2TtsBackend(this.runtime, {int promptCacheSize = 4})
    : _promptCache = _Sarashina2PromptCache(maxEntries: promptCacheSize);

  final Sarashina2DartRuntime runtime;
  final _Sarashina2PromptCache _promptCache;

  @override
  TtsBackendCapability get capability =>
      TtsBackendCatalog.byProvider(sarashina2Provider)!;

  @override
  List<String> get voiceNames => const [];

  @override
  Map<String, Object?> runtimeHealth() => {
    'provider': capability.provider,
    'runtime': 'dart_ffi_onnx_decoder',
    'python': false,
    'loadedComponents': runtime.loadedComponentNames,
    'selectedProviders': runtime.selectedProviders,
    'llmPrecision': runtime.llmPrecision,
    'semanticTokenGeneration': runtime.llmLoaded && runtime.tokenizerLoaded
        ? 'dart_tokenizer'
        : runtime.llmLoaded
        ? 'prompt_token_ids'
        : 'external',
  };

  @override
  Future<DartTtsSynthesisResult> synthesize(
    DartTtsSynthesisRequest request,
  ) async {
    final promptLookup = request.promptAudioBytes == null
        ? null
        : _promptCache.getOrExtract(
            runtime: runtime,
            promptAudioBytes: request.promptAudioBytes!,
          );
    final result = await runtime.synthesize(
      Sarashina2SynthesisRequest(
        text: request.text,
        prompt: promptLookup?.prompt,
        promptText: request.promptText,
        semanticTokenText: request.semanticTokenText,
        semanticTokens: request.semanticTokens,
        promptTokenIds: request.promptTokenIds,
        maxGeneratedTokens: request.maxGeneratedTokens ?? 2048,
        latencyTokens: request.latencyTokens,
        seed: request.rasSeed,
        temperature: request.temperature,
        topP: request.topP,
        frequencyPenalty: request.frequencyPenalty,
        includeFloatOutputs: false,
        includeTokenMetadata: false,
      ),
    );
    final decode = result.decode;
    return DartTtsSynthesisResult(
      provider: sarashina2Provider,
      runtime: 'dart_ffi_onnx_decoder',
      python: false,
      text: decode.text,
      frontendText: decode.text,
      phonemes: '',
      audioBytes: result.audioWavBytes,
      audioFormat: 'wav',
      frontendElapsedMicroseconds:
          decode.promptElapsedMicroseconds +
          (promptLookup?.elapsedMicroseconds ?? 0),
      ttsElapsedMicroseconds: result.ttsElapsedMicroseconds,
      frontendProvider: 'none',
      phonemizerBackend: 'none',
      requestedVoice: request.voice,
      resolvedVoice: decode.usedPrompt ? 'prompt' : 'zero_shot',
      phonemeTokenCount: decode.decodedSemanticTokenCount,
      phonemeChunkCount: (decode.decodedSemanticTokenCount / 25).ceil(),
      warnings: [
        if (result.semanticSource == Sarashina2SemanticSource.external)
          'Sarashina2 decoded external semantic tokens.',
        if (result.semanticSource == Sarashina2SemanticSource.promptTokenIds)
          'Sarashina2 generated semantic tokens from pre-tokenized prompt ids.',
        if (result.semanticSource == Sarashina2SemanticSource.text)
          'Sarashina2 generated semantic tokens from text with the Dart tokenizer.',
        if (result.semanticSource == Sarashina2SemanticSource.text &&
            request.promptAudioBytes != null &&
            request.promptText.trim().isEmpty)
          'Sarashina2 used prompt audio for flow decoding only; pass promptText to condition semantic generation.',
        if (!decode.usedPrompt)
          'Sarashina2 ran without prompt audio; output voice is uncontrolled.',
        if (request.latencyTokens != 1)
          'Sarashina2 used latencyTokens=${request.latencyTokens}.',
        'Sarashina2 Dart/FFI runtime does not embed the upstream SilentCipher watermark.',
      ],
      metadata: {
        ...result.toJson(),
        'llmPrecision': runtime.llmPrecision,
        if (promptLookup != null) ...{
          'promptCacheHit': promptLookup.cacheHit,
          'promptCacheElapsedMs': promptLookup.elapsedMicroseconds / 1000.0,
        },
      },
    );
  }

  @override
  void close() {
    _promptCache.close();
    runtime.close();
  }
}

final class _Sarashina2PromptCache {
  _Sarashina2PromptCache({required int maxEntries})
    : maxEntries = maxEntries < 1 ? 1 : maxEntries;

  final int maxEntries;
  final _entries = <String, Sarashina2Prompt>{};

  _Sarashina2PromptLookup getOrExtract({
    required Sarashina2DartRuntime runtime,
    required Uint8List promptAudioBytes,
  }) {
    final timer = Stopwatch()..start();
    if (maxEntries <= 0) {
      final prompt = runtime.extractPrompt(decodeWav(promptAudioBytes));
      timer.stop();
      return _Sarashina2PromptLookup(
        prompt: prompt,
        cacheHit: false,
        elapsedMicroseconds: timer.elapsedMicroseconds,
      );
    }
    final key = _promptAudioKey(promptAudioBytes);
    final cached = _entries.remove(key);
    if (cached != null) {
      _entries[key] = cached;
      timer.stop();
      return _Sarashina2PromptLookup(
        prompt: cached,
        cacheHit: true,
        elapsedMicroseconds: timer.elapsedMicroseconds,
      );
    }
    final prompt = runtime.extractPrompt(decodeWav(promptAudioBytes));
    _entries[key] = prompt;
    while (_entries.length > maxEntries) {
      final oldestKey = _entries.keys.first;
      _entries.remove(oldestKey)?.close();
    }
    timer.stop();
    return _Sarashina2PromptLookup(
      prompt: prompt,
      cacheHit: false,
      elapsedMicroseconds: timer.elapsedMicroseconds,
    );
  }

  void close() {
    for (final prompt in _entries.values) {
      prompt.close();
    }
    _entries.clear();
  }
}

final class _Sarashina2PromptLookup {
  const _Sarashina2PromptLookup({
    required this.prompt,
    required this.cacheHit,
    required this.elapsedMicroseconds,
  });

  final Sarashina2Prompt prompt;
  final bool cacheHit;
  final int elapsedMicroseconds;
}

String _promptAudioKey(Uint8List bytes) {
  const mask = 0xffffffffffffffff;
  var hash = 0xcbf29ce484222325;
  for (final byte in bytes) {
    hash ^= byte;
    hash = (hash * 0x100000001b3) & mask;
  }
  return '${bytes.length}:$hash';
}

abstract interface class DartTtsBackend {
  TtsBackendCapability get capability;

  List<String> get voiceNames;

  Map<String, Object?> runtimeHealth();

  Future<DartTtsSynthesisResult> synthesize(DartTtsSynthesisRequest request);

  void close();
}

final class KokoroTtsBackend implements DartTtsBackend {
  const KokoroTtsBackend(this.runtime);

  final UniFrontendKokoroTtsRuntime runtime;

  @override
  TtsBackendCapability get capability =>
      TtsBackendCatalog.byProvider('kokoro')!;

  @override
  List<String> get voiceNames => runtime.voiceNames;

  @override
  Map<String, Object?> runtimeHealth() => {
    'provider': capability.provider,
    'runtime': 'dart',
    'python': false,
    'structuredOnnxProvider': runtime.frontendProvider,
    'kokoroOnnxProvider': runtime.kokoroProvider,
    'phonemizerBackend': runtime.phonemizerBackend,
  };

  @override
  Future<DartTtsSynthesisResult> synthesize(
    DartTtsSynthesisRequest request,
  ) async {
    final result = await runtime.synthesize(
      text: request.text,
      phonemes: request.phonemes,
      voice: request.voice,
      speed: request.speed,
    );
    return DartTtsSynthesisResult.fromUniFrontendKokoro(
      result,
      phonemizerBackend: runtime.phonemizerBackend,
    );
  }

  @override
  void close() {
    runtime.close();
  }
}

final class DartTtsBackendRegistry {
  DartTtsBackendRegistry({required Iterable<DartTtsBackend> backends})
    : _backends = {
        for (final backend in backends) backend.capability.provider: backend,
      };

  final Map<String, DartTtsBackend> _backends;
  bool _closed = false;

  List<String> get providerNames => _backends.keys.toList(growable: false);

  DartTtsBackend? byProvider(String provider) => _backends[provider];

  DartTtsBackend requireProvider(String provider) {
    final backend = byProvider(provider);
    if (backend == null) {
      throw UnsupportedError('unsupported TTS provider: $provider');
    }
    return backend;
  }

  List<Map<String, Object?>> providerCards({bool includeUnavailable = true}) {
    final cards = TtsBackendCatalog.providerCards(
      includeUnavailable: includeUnavailable,
    );
    final readyProviders = _backends.keys.toSet();
    return [
      for (final card in cards)
        {...card, 'loaded': readyProviders.contains(card['provider'])},
    ];
  }

  Map<String, Object?> runtimeHealth() => {
    'ok': true,
    'runtime': 'dart',
    'python': false,
    'providers': providerNames,
    'backends': {
      for (final entry in _backends.entries)
        entry.key: entry.value.runtimeHealth(),
    },
  };

  Future<DartTtsSynthesisResult> synthesize(DartTtsSynthesisRequest request) {
    if (_closed) {
      throw StateError('TTS backend registry is closed.');
    }
    return requireProvider(request.provider).synthesize(request);
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    for (final backend in _backends.values) {
      backend.close();
    }
  }
}
