import 'dart:typed_data';
import 'dart:io';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';
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
    expect(
      providers['indextts2']!.onnxTargets.map((target) => target.name),
      containsAll(['gpt', 's2mel', 'bigvgan']),
    );
    expect(
      providers['chatterbox']!.sourceAssets.map((source) => source.name),
      containsAll(['t3_mtl23ls_v2', 's3gen', 've']),
    );
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
    expect(
      paths.cosyVoice2Paths.campplusOnnx,
      '/repo/src/ttsbackends/providers/cosyvoice2/models/CosyVoice2-0.5B/campplus.onnx',
    );
  });

  test('CosyVoice2 partial bundle records ONNX components and blockers', () {
    final paths = CosyVoice2Paths(modelDir: '/repo/cosyvoice2');
    final bundle = CosyVoice2PartialOnnxBundle.inspect(paths: paths);

    expect(
      bundle.statuses.map((status) => status.file.name),
      containsAll([
        'campplus',
        'speech_tokenizer_v2',
        'flow_decoder_estimator_fp32',
        'llm',
        'hift',
      ]),
    );
    expect(
      bundle.statuses
          .where((status) => status.file.loadableByDartOnnx)
          .map((status) => status.file.name),
      [
        'campplus',
        'speech_tokenizer_v2',
        'flow_decoder_estimator_fp32',
        'flow_encoder_fp32',
        'flow_encoder_fp16',
        'llm',
        'flow',
        'hift',
        'blank_en_llm',
        'vllm',
      ],
    );
    final flowEncoder = bundle.statuses.singleWhere(
      (status) => status.file.name == 'flow_encoder_fp32',
    );
    expect(flowEncoder.file.path, '/repo/cosyvoice2/flow.encoder.fp32.onnx');
    expect(
      flowEncoder.file.sourcePath,
      '/repo/cosyvoice2/flow.encoder.fp32.zip',
    );
    expect(flowEncoder.file.sourceFormat, 'torchscript_zip');
    expect(bundle.hasRequiredBlockedComponents, isTrue);
    expect(bundle.blockers, contains(contains('flow_encoder_fp32 ONNX')));
    expect(bundle.loadedComponentNames, isEmpty);
    expect(() => bundle.requireLoadedComponent('campplus'), throwsStateError);
    expect(bundle.toJson()['runtime'], 'dart_inference_onnx_partial');
  });

  test('TTS backend asset audit resolves ONNX targets and source assets', () {
    final root = Directory.systemTemp.createTempSync('tts-audit-root-');
    try {
      final providerDir = Directory(
        '${root.path}/src/ttsbackends/providers/chatterbox',
      )..createSync(recursive: true);
      Directory('${providerDir.path}/models/onnx').createSync(recursive: true);
      File('${providerDir.path}/models/onnx/t3_mtl23ls_v2.onnx')
        ..createSync(recursive: true)
        ..writeAsBytesSync(const [1, 2, 3]);
      final snapshot = Directory(
        '${providerDir.path}/models/models--ResembleAI--chatterbox/snapshots/demo',
      )..createSync(recursive: true);
      File('${snapshot.path}/t3_mtl23ls_v2.safetensors')
        ..createSync()
        ..writeAsBytesSync(const [1]);
      File('${snapshot.path}/s3gen.pt')
        ..createSync()
        ..writeAsBytesSync(const [2]);

      final audit = TtsBackendAssetAudit.audit(root.path);
      final chatterbox = audit.providers['chatterbox']!;

      expect(chatterbox.exists, isTrue);
      expect(
        chatterbox.missingRequiredOnnxTargets,
        containsAll(['models/onnx/s3gen.onnx', 'models/onnx/ve.onnx']),
      );
      expect(
        chatterbox.sourceAssets.singleWhere(
          (source) => source['name'] == 't3_mtl23ls_v2',
        )['exists'],
        isTrue,
      );
      expect(chatterbox.toJson()['missingRequiredOnnxTargets'], isNotEmpty);
    } finally {
      root.deleteSync(recursive: true);
    }
  });

  test('generic TTS ONNX component bundle inspects manifest targets', () {
    final root = Directory.systemTemp.createTempSync('tts-component-root-');
    try {
      final providerDir = Directory('${root.path}/chatterbox')
        ..createSync(recursive: true);
      Directory('${providerDir.path}/models/onnx').createSync(recursive: true);
      File('${providerDir.path}/models/onnx/t3_mtl23ls_v2.onnx')
        ..createSync(recursive: true)
        ..writeAsBytesSync(const [1, 2, 3]);
      final snapshot = Directory('${providerDir.path}/models/snapshot')
        ..createSync(recursive: true);
      File('${snapshot.path}/t3_mtl23ls_v2.safetensors')
        ..createSync()
        ..writeAsBytesSync(const [1]);
      File('${snapshot.path}/s3gen.pt')
        ..createSync()
        ..writeAsBytesSync(const [2]);

      final capability = TtsBackendCatalog.byProvider('chatterbox')!;
      final bundle = TtsOnnxComponentBundle.inspect(
        capability: capability,
        providerDir: providerDir.path,
      );

      final t3 = bundle.statuses.singleWhere(
        (status) => status.target.name == 't3_mtl23ls_v2',
      );
      expect(t3.exists, isTrue);
      expect(t3.sources.single['exists'], isTrue);
      expect(bundle.loadedComponentNames, isEmpty);
      expect(bundle.hasRequiredBlockedComponents, isTrue);
      expect(bundle.blockers, contains(contains('s3gen ONNX is missing')));
      expect(() => bundle.requireLoadedComponent('s3gen'), throwsStateError);
      expect(bundle.toJson()['runtime'], 'dart_inference_onnx_components');
    } finally {
      root.deleteSync(recursive: true);
    }
  });

  test('CosyVoice2 component status reports smoke benchmark timings', () {
    const file = CosyVoice2ComponentFile.onnx(
      name: 'campplus',
      role: 'speaker_embedding',
      path: '/repo/cosyvoice2/campplus.onnx',
      requiredForSynthesis: true,
    );
    const status = CosyVoice2ComponentStatus(
      file: file,
      exists: true,
      sizeBytes: 128,
      loaded: true,
      smokeRan: true,
      smokeElapsedMicroseconds: 9000,
      smokeWarmupIterations: 1,
      smokeIterations: 5,
      smokeTotalElapsedMicroseconds: 45000,
      smokeMinElapsedMicroseconds: 8000,
      smokeMaxElapsedMicroseconds: 10000,
      smokeOutputs: [
        {
          'name': 'output',
          'dtype': 'float32',
          'shape': [1, 192],
          'byteLength': 768,
        },
      ],
    );

    final json = status.toJson();

    expect(json['smokeRan'], isTrue);
    expect(json['smokeElapsedMs'], 9.0);
    expect(json['smokeWarmupIterations'], 1);
    expect(json['smokeIterations'], 5);
    expect(json['smokeTotalElapsedMs'], 45.0);
    expect(json['smokeMinElapsedMs'], 8.0);
    expect(json['smokeMaxElapsedMs'], 10.0);
    expect(json['smokeOutputs'], isNotEmpty);
  });

  test('TTS runtime options default to strict GPU runtime', () {
    const options = DartTtsRuntimeOptions();

    expect(options.provider, 'cuda');
    expect(options.deviceId, 0);
    expect(options.requireProvider, isTrue);
    expect(options.cudaMemoryLimitMb, 16384);
    expect(options.preloadLibraries, isEmpty);

    final copied = options.copyWith(preloadLibraries: ['/cuda/libcudart.so']);
    expect(copied.preloadLibraries, ['/cuda/libcudart.so']);
  });

  test('runtime dependency audit reports missing TensorRT libraries', () {
    final audit = RuntimeDependencyAudit.inspect(
      root: '/path/that/does/not/exist',
      provider: 'tensorrt',
      environment: const {},
      includeSystemDirs: false,
    );

    expect(audit.tensorrtRequested, isTrue);
    expect(audit.tensorrtReady, isFalse);
    expect(audit.tensorrt10.missing, contains('libnvinfer.so.10'));
    expect(audit.skipReason, contains('TensorRT 10 missing'));
    expect(audit.toJson()['tensorrtReady'], isFalse);
  });

  test('runtime dependency audit accepts complete TensorRT 10 set', () {
    final dir = Directory.systemTemp.createTempSync('trt-audit-test-');
    try {
      for (final name in RuntimeDependencyAudit.tensorRt10Libraries) {
        File('${dir.path}/$name').writeAsBytesSync(const []);
      }

      final audit = RuntimeDependencyAudit.inspect(
        root: '/path/that/does/not/exist',
        provider: 'trt',
        environment: const {},
        extraSearchDirs: [dir.path],
        includeSystemDirs: false,
      );

      expect(audit.tensorrtReady, isTrue);
      expect(audit.tensorrt10.ready, isTrue);
      expect(audit.skipReason, isNull);
      expect(audit.toJson()['tensorrtReady'], isTrue);
    } finally {
      dir.deleteSync(recursive: true);
    }
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
