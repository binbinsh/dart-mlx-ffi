import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:test/test.dart';

void main() {
  test('TTS backend catalog exposes Kokoro as the local Dart ONNX backend', () {
    final kokoro = TtsBackendCatalog.byProvider('kokoro');

    expect(kokoro, isNotNull);
    expect(kokoro!.isLocalDartOnnxReady, isTrue);
    expect(kokoro.runtime, 'dart_onnx');
    expect(kokoro.localOnnxAssets, contains('kokoro-v1.0.onnx'));
  });

  test('TTS backend catalog records unavailable providers explicitly', () {
    final providers = {
      for (final backend in TtsBackendCatalog.all) backend.provider: backend,
    };

    expect(providers.keys, containsAll(['cosyvoice2', 'indextts2', 'sonic3']));
    expect(providers['cosyvoice2']!.readiness, TtsBackendReadiness.partial);
    expect(providers['indextts2']!.blockers, isNotEmpty);
    expect(providers['sonic3']!.readiness, TtsBackendReadiness.remoteApi);
  });

  test(
    'TTS backend registry routes synthesis through loaded backends',
    () async {
      final registry = DartTtsBackendRegistry(backends: [_FakeTtsBackend()]);

      expect(registry.providerNames, ['kokoro']);
      expect(
        registry.providerCards().singleWhere(
          (card) => card['provider'] == 'kokoro',
        )['loaded'],
        isTrue,
      );

      final result = await registry.synthesize(
        const DartTtsSynthesisRequest(provider: 'kokoro', text: 'hello'),
      );

      expect(result.runtime, 'dart');
      expect(result.python, isFalse);
      expect(result.audioFormat, 'wav');
      expect(result.audioBytes, [1, 2, 3]);

      registry.close();
    },
  );

  test('TTS backend registry closes loaded backends', () {
    final backend = _FakeTtsBackend();
    final registry = DartTtsBackendRegistry(backends: [backend]);

    registry.close();
    registry.close();

    expect(backend.closeCount, 1);
    expect(
      () => registry.synthesize(
        const DartTtsSynthesisRequest(provider: 'kokoro', text: 'hello'),
      ),
      throwsStateError,
    );
  });

  test('UniFrontend TTS path defaults are rooted in the caller repo', () {
    final paths = DartUniFrontendTtsPaths.fromUniFrontendRoot('/repo/');

    expect(
      paths.kokoroModelPath,
      '/repo/src/ttsbackends/providers/kokoro/models/kokoro-v1.0.onnx',
    );
    expect(
      paths.structuredModelPath,
      contains('/repo/artifacts/onnx/structured-mmbert-focus-v2-step-20000'),
    );
    expect(
      paths.structuredTokenizerPath,
      contains('/repo/src/ttsbackends/providers/kokoro/models/models--'),
    );
  });

  test('TTS runtime options default to strict GPU runtime', () {
    const options = DartTtsRuntimeOptions();

    expect(options.provider, 'cuda');
    expect(options.deviceId, 0);
    expect(options.requireProvider, isTrue);
    expect(options.cudaMemoryLimitMb, 16384);
    expect(options.allowEspeakProcessFallback, isFalse);
    expect(options.preloadLibraries, isEmpty);

    final copied = options.copyWith(preloadLibraries: ['/cuda/libcudart.so']);
    expect(copied.preloadLibraries, ['/cuda/libcudart.so']);
  });
}

final class _FakeTtsBackend implements DartTtsBackend {
  int closeCount = 0;

  @override
  TtsBackendCapability get capability =>
      TtsBackendCatalog.byProvider('kokoro')!;

  @override
  List<String> get voiceNames => const ['zf_xiaoni'];

  @override
  Map<String, Object?> runtimeHealth() => const {
    'provider': 'kokoro',
    'runtime': 'dart',
    'python': false,
  };

  @override
  Future<DartTtsSynthesisResult> synthesize(
    DartTtsSynthesisRequest request,
  ) async {
    return DartTtsSynthesisResult(
      provider: request.provider,
      runtime: 'dart',
      python: false,
      text: request.text,
      frontendText: request.text,
      phonemes: 'həlˈoʊ',
      audioBytes: Uint8List.fromList([1, 2, 3]),
      audioFormat: 'wav',
      frontendElapsedMicroseconds: 1,
      ttsElapsedMicroseconds: 2,
      frontendProvider: 'CUDAExecutionProvider',
      phonemizerBackend: 'injected',
      requestedVoice: request.voice,
      resolvedVoice: request.voice,
      phonemeTokenCount: 5,
      phonemeChunkCount: 1,
      warnings: const [],
    );
  }

  @override
  void close() {
    closeCount += 1;
  }
}
