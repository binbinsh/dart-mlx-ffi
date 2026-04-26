import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_inference/runtime.dart';

void main() {
  group('RuntimeResolver', () {
    final spec = ModelSpec(
      id: 'demo',
      family: 'Demo',
      modalities: const [ModelModality.textGeneration],
      platformArtifacts: const {
        RuntimeEngine.mlx: RuntimeArtifact(
          engine: RuntimeEngine.mlx,
          path: 'mlx',
          targetPlatforms: ['ios', 'macos'],
        ),
        RuntimeEngine.coreml: RuntimeArtifact(
          engine: RuntimeEngine.coreml,
          path: 'coreml',
          targetPlatforms: ['ios', 'macos'],
          accelerators: [Accelerator.ane],
        ),
        RuntimeEngine.onnx: RuntimeArtifact(
          engine: RuntimeEngine.onnx,
          path: 'model.onnx',
          targetPlatforms: ['windows', 'linux'],
        ),
        RuntimeEngine.litert: RuntimeArtifact(
          engine: RuntimeEngine.litert,
          path: 'model.tflite',
          targetPlatforms: ['android'],
        ),
      },
    );

    test('prefers Core ML on iOS', () {
      final resolver = RuntimeResolver(hostPlatform: RuntimePlatform.ios);
      final resolution = resolver.resolve(spec);
      expect(resolution.engine, RuntimeEngine.coreml);
      expect(resolution.accelerators.first, Accelerator.ane);
    });

    test('prefers Core ML over preview MLX safetensors on macOS', () {
      final resolver = RuntimeResolver(hostPlatform: RuntimePlatform.macos);
      expect(resolver.resolve(spec).engine, RuntimeEngine.coreml);
      expect(
        resolver
            .resolve(spec, const RuntimeOptions(prefer: [Accelerator.ane]))
            .engine,
        RuntimeEngine.coreml,
      );
    });

    test('selects registered MLX function bundles on Apple platforms', () {
      const mlxFunctionOnly = ModelSpec(
        id: 'mlxfn',
        family: 'MLX function',
        modalities: [ModelModality.textGeneration],
        platformArtifacts: {
          RuntimeEngine.mlx: RuntimeArtifact(
            engine: RuntimeEngine.mlx,
            path: 'function.mlxfn',
            format: 'mlx-function',
            targetPlatforms: ['ios', 'macos'],
          ),
          RuntimeEngine.onnx: RuntimeArtifact(
            engine: RuntimeEngine.onnx,
            path: 'model.onnx',
            targetPlatforms: ['macos'],
          ),
        },
      );

      final resolver = RuntimeResolver(hostPlatform: RuntimePlatform.macos);
      expect(resolver.resolve(mlxFunctionOnly).engine, RuntimeEngine.mlx);
    });

    test('selects platform engines for desktop and Android', () {
      expect(
        RuntimeResolver(
          hostPlatform: RuntimePlatform.windows,
        ).resolve(spec).engine,
        RuntimeEngine.onnx,
      );
      expect(
        RuntimeResolver(
          hostPlatform: RuntimePlatform.linux,
        ).resolve(spec).engine,
        RuntimeEngine.onnx,
      );
      expect(
        RuntimeResolver(
          hostPlatform: RuntimePlatform.android,
        ).resolve(spec).engine,
        RuntimeEngine.litert,
      );
    });

    test('falls back from unavailable explicit engine when allowed', () {
      final resolver = RuntimeResolver(hostPlatform: RuntimePlatform.android);
      final resolution = resolver.resolve(
        spec,
        const RuntimeOptions(engine: RuntimeEngine.coreml),
      );
      expect(resolution.engine, RuntimeEngine.litert);
      expect(resolution.fallbackReason, contains('coreml'));
    });

    test(
      'falls back to ONNX where native platform engines are unavailable',
      () {
        const onnxOnly = ModelSpec(
          id: 'onnx_only',
          family: 'ONNX only',
          modalities: [ModelModality.textGeneration],
          platformArtifacts: {
            RuntimeEngine.onnx: RuntimeArtifact(
              engine: RuntimeEngine.onnx,
              path: 'model.onnx',
              targetPlatforms: ['ios', 'macos', 'android'],
            ),
          },
        );

        expect(
          RuntimeResolver(
            hostPlatform: RuntimePlatform.android,
          ).resolve(onnxOnly).engine,
          RuntimeEngine.onnx,
        );
        expect(
          RuntimeResolver(
            hostPlatform: RuntimePlatform.ios,
          ).resolve(onnxOnly).engine,
          RuntimeEngine.onnx,
        );
      },
    );

    test('throws for unavailable explicit engine without fallback', () {
      final resolver = RuntimeResolver(hostPlatform: RuntimePlatform.android);
      expect(
        () => resolver.resolve(
          spec,
          const RuntimeOptions(
            engine: RuntimeEngine.coreml,
            allowFallback: false,
          ),
        ),
        throwsStateError,
      );
    });

    test('registry falls back when selected engine is not registered', () {
      final registry = RuntimeRegistry(
        resolver: const RuntimeResolver(hostPlatform: RuntimePlatform.macos),
      )..register(_FakeRuntime(RuntimeEngine.coreml));

      final session = registry.load(spec, rootPath: '/tmp');
      final outputs = session.run(const ModelInputs({}));

      expect(outputs.diagnostics['engine'], 'coreml');
      session.close();
    });

    test('registry resolves HF artifacts before async load', () async {
      const spec = ModelSpec(
        id: 'hf_demo',
        family: 'HF demo',
        modalities: [ModelModality.textGeneration],
        platformArtifacts: {
          RuntimeEngine.onnx: RuntimeArtifact(
            engine: RuntimeEngine.onnx,
            path: 'hf://acme/demo/onnx/model.onnx',
            targetPlatforms: ['linux'],
            metadata: {'repo': 'acme/demo', 'artifact': 'onnx/model.onnx'},
          ),
        },
      );
      final registry = RuntimeRegistry(
        resolver: const RuntimeResolver(hostPlatform: RuntimePlatform.linux),
      )..register(const _PathRuntime(RuntimeEngine.onnx));

      final session = await registry.loadAsync(
        spec,
        options: const RuntimeOptions(
          artifactResolver: _FakeArtifactResolver('/tmp/model.onnx'),
        ),
      );
      final outputs = session.run(const ModelInputs({}));

      expect(outputs.diagnostics['artifactPath'], '/tmp/model.onnx');
      session.close();
    });
  });

  group('RuntimeTensor', () {
    test('wraps typed data and exposes typed views', () {
      final source = Float32List.fromList([1, 2, 3]);
      final tensor = RuntimeTensor.float32([3], source);
      source[0] = 99;

      expect(tensor.dtype, RuntimeTensorDataType.float32);
      expect(tensor.shape, [3]);
      expect(tensor.asFloat32List(), [99, 2, 3]);
    });

    test('allocates Zig-owned native input buffers', () {
      final buffer = NativeTensorBuffer.float32([3]);
      try {
        buffer.asFloat32List().setAll(0, [1, 2, 3]);
        final tensor = buffer.tensor;

        expect(tensor.dtype, RuntimeTensorDataType.float32);
        expect(tensor.shape, [3]);
        expect(tensor.asFloat32List(), [1, 2, 3]);
      } finally {
        buffer.close();
      }
    });

    test('uses Zig tensor allocation validation', () {
      final empty = NativeTensorBuffer.float32([0, 3]);
      try {
        expect(empty.byteLength, 0);
        expect(empty.bytes, isEmpty);
      } finally {
        empty.close();
      }

      expect(
        () => NativeTensorBuffer.float32([-1]),
        throwsA(
          isA<StateError>().having(
            (error) => error.message,
            'message',
            contains('invalid shape'),
          ),
        ),
      );
    });
  });

  group('runtime metadata JSON', () {
    test('round trips artifacts and validation status', () {
      const original = ModelSpec(
        id: 'prod',
        family: 'Production',
        modalities: [ModelModality.embedding],
        supportLevel: SupportLevel.production,
        platformArtifacts: {
          RuntimeEngine.coreml: RuntimeArtifact(
            engine: RuntimeEngine.coreml,
            path: 'hf://org/repo/model.mlmodelc',
            format: 'mlmodelc',
            sourceUri: 'hf://org/repo/model.mlmodelc',
            targetPlatforms: ['ios'],
            accelerators: [Accelerator.ane],
          ),
        },
        validationStatus: {
          'ios': RuntimeValidationStatus(
            platform: 'ios',
            engine: RuntimeEngine.coreml,
            correctnessPassed: true,
            speedPassed: true,
            peakMemoryPassed: true,
            deviceProfilePassed: true,
            speedRatio: 0.95,
            peakMemoryRatio: 1.02,
          ),
        },
      );

      final json = (jsonDecode(jsonEncode(original.toJson())) as Map)
          .cast<String, Object?>();
      final manifest = ModelManifest.fromJson({
        'version': 1,
        'models': [json],
      });
      final restored = manifest.models.single;

      expect(restored.supportLevel, SupportLevel.production);
      expect(
        restored.platformArtifacts[RuntimeEngine.coreml]?.format,
        'mlmodelc',
      );
      expect(
        restored.platformArtifacts[RuntimeEngine.coreml]?.sourceUri,
        'hf://org/repo/model.mlmodelc',
      );
      expect(restored.validationStatus['ios']?.passed, isTrue);
      expect(restored.validationStatus['ios']?.deviceProfilePassed, isTrue);
      expect(manifest.productionModels, hasLength(1));
    });
  });

  group('CoreMlBundleLayout', () {
    test('discovers CoreML-LLM chunked layout', () {
      final dir = Directory.systemTemp.createTempSync('coreml_layout_');
      try {
        File('${dir.path}/model_config.json').writeAsStringSync('{}');
        Directory('${dir.path}/chunk2.mlmodelc').createSync();
        Directory('${dir.path}/chunk1.mlmodelc').createSync();
        Directory('${dir.path}/chunk_3.mlpackage').createSync();
        Directory('${dir.path}/chunk_head.mlpackage').createSync();
        Directory('${dir.path}/chunk_0_vision.mlpackage').createSync();
        Directory('${dir.path}/prefill_chunk1.mlmodelc').createSync();
        Directory('${dir.path}/prefill_chunk_2.mlpackage').createSync();
        File('${dir.path}/embed_weight.bin').writeAsBytesSync([0, 1, 2]);

        final layout = CoreMlBundleLayout.discover(dir.path);
        expect(layout.isLoadable, isTrue);
        expect(layout.isChunked, isTrue);
        expect(layout.decodeChunks.map((p) => p.split('/').last), [
          'chunk1.mlmodelc',
          'chunk2.mlmodelc',
          'chunk_3.mlpackage',
          'chunk_head.mlpackage',
        ]);
        expect(layout.prefillChunks.map((p) => p.split('/').last), [
          'prefill_chunk1.mlmodelc',
          'prefill_chunk_2.mlpackage',
        ]);
        expect(layout.sidecars.single.endsWith('embed_weight.bin'), isTrue);
      } finally {
        dir.deleteSync(recursive: true);
      }
    });

    test('discovers monolithic Core ML layout', () {
      final dir = Directory.systemTemp.createTempSync('coreml_layout_');
      try {
        File('${dir.path}/model_config.json').writeAsStringSync('{}');
        Directory('${dir.path}/model.mlpackage').createSync();

        final layout = CoreMlBundleLayout.discover(dir.path);
        expect(layout.isLoadable, isTrue);
        expect(layout.isMonolithic, isTrue);
        expect(layout.isChunked, isFalse);
      } finally {
        dir.deleteSync(recursive: true);
      }
    });

    test('discovers Core ML pipeline spec', () {
      final dir = Directory.systemTemp.createTempSync('coreml_layout_');
      try {
        final spec = File('${dir.path}/pipeline.json')
          ..writeAsStringSync('{"format":"dart_inference.coreml_pipeline.v1"}');

        final layout = CoreMlBundleLayout.discover(spec.path);
        expect(layout.isLoadable, isTrue);
        expect(layout.isPipeline, isTrue);
        expect(layout.pipelineSpecPath, spec.path);
        expect(layout.isChunked, isFalse);
        expect(layout.isMonolithic, isFalse);
      } finally {
        dir.deleteSync(recursive: true);
      }
    });

    test('treats HF chunk directories without model_config as loadable', () {
      final dir = Directory.systemTemp.createTempSync('coreml_layout_');
      try {
        Directory('${dir.path}/chunk_0.mlpackage').createSync();
        Directory('${dir.path}/chunk_head.mlpackage').createSync();

        final layout = CoreMlBundleLayout.discover(dir.path);
        expect(layout.modelConfigPath, isNull);
        expect(layout.isLoadable, isTrue);
        expect(layout.decodeChunks.map((p) => p.split('/').last), [
          'chunk_0.mlpackage',
          'chunk_head.mlpackage',
        ]);
      } finally {
        dir.deleteSync(recursive: true);
      }
    });
  });

  group('HuggingFaceArtifactCache', () {
    test('downloads single-file artifacts and reuses cache', () async {
      final server = await _startHfServer({
        '/acme/demo/resolve/main/onnx/model.onnx': 'onnx-bytes',
      });
      final cacheDir = Directory.systemTemp.createTempSync('hf_cache_');
      try {
        final cache = HuggingFaceArtifactCache(
          cacheRoot: cacheDir.path,
          endpoint: 'http://${server.address.host}:${server.port}',
        );
        const artifact = RuntimeArtifact(
          engine: RuntimeEngine.onnx,
          path: 'hf://acme/demo/onnx/model.onnx',
          metadata: {'repo': 'acme/demo', 'artifact': 'onnx/model.onnx'},
        );

        final resolved = await cache.resolve(artifact);
        expect(File(resolved.path).readAsStringSync(), 'onnx-bytes');
        expect(resolved.sourceUri, artifact.path);
        expect(resolved.metadata['resolvedSourceUri'], artifact.path);
        expect(cache.resolveCached(artifact).path, resolved.path);
      } finally {
        await server.close(force: true);
        cacheDir.deleteSync(recursive: true);
      }
    });

    test('downloads directory artifacts from repository tree', () async {
      final server = await _startHfServer({
        '/api/models/acme/demo/tree/main/bundle.mlmodelc': jsonEncode([
          {'type': 'directory', 'path': 'bundle.mlmodelc/Data'},
          {'type': 'file', 'path': 'bundle.mlmodelc/Manifest.json'},
          {'type': 'file', 'path': 'bundle.mlmodelc/Data/model.bin'},
        ]),
        '/acme/demo/resolve/main/bundle.mlmodelc/Manifest.json': '{}',
        '/acme/demo/resolve/main/bundle.mlmodelc/Data/model.bin': 'weights',
      });
      final cacheDir = Directory.systemTemp.createTempSync('hf_cache_');
      try {
        final cache = HuggingFaceArtifactCache(
          cacheRoot: cacheDir.path,
          endpoint: 'http://${server.address.host}:${server.port}',
        );
        const artifact = RuntimeArtifact(
          engine: RuntimeEngine.coreml,
          path: 'hf://acme/demo/bundle.mlmodelc',
          metadata: {'repo': 'acme/demo', 'artifact': 'bundle.mlmodelc'},
        );

        final resolved = await cache.resolve(artifact);
        expect(Directory(resolved.path).existsSync(), isTrue);
        expect(
          File('${resolved.path}/Data/model.bin').readAsStringSync(),
          'weights',
        );
      } finally {
        await server.close(force: true);
        cacheDir.deleteSync(recursive: true);
      }
    });
  });
}

final class _FakeRuntime implements ModelRuntime {
  const _FakeRuntime(this.engine);

  final RuntimeEngine engine;

  @override
  RuntimeCapabilities get capabilities =>
      RuntimeCapabilities(engine: engine, platform: RuntimePlatform.macos);

  @override
  ModelSession load(ModelBundle bundle, RuntimeOptions options) {
    return _FakeSession(engine.name);
  }
}

final class _FakeSession implements ModelSession {
  const _FakeSession(this.engine);

  final String engine;

  @override
  Map<String, Object?> get diagnostics => {'engine': engine};

  @override
  void close() {}

  @override
  ModelOutputs run(ModelInputs inputs) {
    return ModelOutputs(const {}, diagnostics: {'engine': engine});
  }

  @override
  Stream<ModelOutputs> stream(ModelInputs inputs) async* {
    yield run(inputs);
  }
}

final class _FakeArtifactResolver implements RuntimeArtifactResolver {
  const _FakeArtifactResolver(this.path);

  final String path;

  @override
  RuntimeArtifact resolveCached(RuntimeArtifact artifact) {
    return artifact.copyWith(path: path);
  }

  @override
  Future<RuntimeArtifact> resolve(RuntimeArtifact artifact) async {
    return resolveCached(artifact);
  }
}

final class _PathRuntime implements ModelRuntime {
  const _PathRuntime(this.engine);

  final RuntimeEngine engine;

  @override
  RuntimeCapabilities get capabilities =>
      RuntimeCapabilities(engine: engine, platform: RuntimePlatform.linux);

  @override
  ModelSession load(ModelBundle bundle, RuntimeOptions options) {
    return _PathSession(bundle.artifactPath);
  }
}

final class _PathSession implements ModelSession {
  const _PathSession(this.artifactPath);

  final String artifactPath;

  @override
  Map<String, Object?> get diagnostics => {'artifactPath': artifactPath};

  @override
  void close() {}

  @override
  ModelOutputs run(ModelInputs inputs) {
    return ModelOutputs(const {}, diagnostics: {'artifactPath': artifactPath});
  }

  @override
  Stream<ModelOutputs> stream(ModelInputs inputs) async* {
    yield run(inputs);
  }
}

Future<HttpServer> _startHfServer(Map<String, String> responses) async {
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  server.listen((request) {
    final key = request.uri.path;
    final body = responses[key];
    if (body == null) {
      request.response.statusCode = HttpStatus.notFound;
      request.response.write('missing $key');
    } else {
      request.response.write(body);
    }
    unawaited(request.response.close());
  });
  return server;
}
