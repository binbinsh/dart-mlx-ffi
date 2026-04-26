import 'dart:typed_data';

import '../unifrontend/tts.dart';
import 'catalog.dart';

final class DartTtsSynthesisRequest {
  const DartTtsSynthesisRequest({
    required this.provider,
    this.text = '',
    this.phonemes = '',
    this.voice = 'zf_xiaoni',
    this.speed = 1.0,
  });

  final String provider;
  final String text;
  final String phonemes;
  final String voice;
  final double speed;
}

final class DartTtsSynthesisResult {
  const DartTtsSynthesisResult({
    required this.provider,
    required this.runtime,
    required this.python,
    required this.text,
    required this.frontendText,
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
    'phonemizerProcessFallbackAllowed':
        runtime.phonemizerProcessFallbackAllowed,
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
