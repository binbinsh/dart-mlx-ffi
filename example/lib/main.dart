import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'package:dart_inference/dart_mlx_ffi.dart';
import 'package:dart_inference/models.dart'
    show Qwen3AsrCoreMlRunner, Qwen3AsrNativeRunner;
import 'package:dart_inference/runtime.dart';

const _runtimeSmokeEnabled = bool.fromEnvironment(
  'DMF_RUNTIME_SMOKE',
  defaultValue: false,
);
const _runtimeSmokeModelId = String.fromEnvironment(
  'DMF_RUNTIME_SMOKE_MODEL',
  defaultValue: 'silero_vad',
);
const _runtimeSmokeEngineName = String.fromEnvironment(
  'DMF_RUNTIME_SMOKE_ENGINE',
  defaultValue: 'coreml',
);
const _runtimeSmokeArtifact = String.fromEnvironment(
  'DMF_RUNTIME_SMOKE_ARTIFACT',
  defaultValue: '',
);
const _runtimeSmokeWaitForArtifactSeconds = int.fromEnvironment(
  'DMF_RUNTIME_SMOKE_WAIT_FOR_ARTIFACT_SECONDS',
  defaultValue: 0,
);
const _runtimeSmokeCoreMlComputeUnits = String.fromEnvironment(
  'DMF_RUNTIME_SMOKE_COREML_COMPUTE_UNITS',
  defaultValue: '',
);
const _runtimeSmokeProvider = String.fromEnvironment(
  'DMF_RUNTIME_SMOKE_PROVIDER',
  defaultValue: '',
);
const _runtimeSmokeRequireProvider = bool.fromEnvironment(
  'DMF_RUNTIME_SMOKE_REQUIRE_PROVIDER',
  defaultValue: false,
);
const _runtimeSmokeDelegate = String.fromEnvironment(
  'DMF_RUNTIME_SMOKE_DELEGATE',
  defaultValue: '',
);
const _runtimeSmokeRequireDelegate = bool.fromEnvironment(
  'DMF_RUNTIME_SMOKE_REQUIRE_DELEGATE',
  defaultValue: false,
);
const _runtimeSmokeLiteRtSectionIndex = int.fromEnvironment(
  'DMF_RUNTIME_SMOKE_LITERT_SECTION_INDEX',
  defaultValue: -1,
);
const _runtimeSmokeLogTag = 'DMF_RUNTIME_SMOKE';
const _runtimeSmokeLogMethod = MethodChannel('dart_mlx_ffi/runtime_smoke');

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  if (_runtimeSmokeEnabled) {
    await _runRuntimeSmoke();
    runApp(const _RuntimeSmokeApp());
    return;
  }
  runApp(const DemoApp());
}

Future<void> _runRuntimeSmoke() async {
  await _emitRuntimeSmokeLogLine('$_runtimeSmokeLogTag:BOOT');
  final payload = <String, Object?>{
    'mode': 'runtime_smoke',
    'model_id': _runtimeSmokeModelId,
    'engine': _runtimeSmokeEngineName,
    'platform': RuntimePlatformCurrent.current().name,
    'started_at': DateTime.now().toUtc().toIso8601String(),
  };
  try {
    final engine = _runtimeEngine(_runtimeSmokeEngineName);
    final resolvedArtifact = _resolveRuntimeSmokeArtifactPath(
      _runtimeSmokeArtifact,
    );
    final manifest = ModelManifest.builtIn();
    final baseSpec =
        manifest[_runtimeSmokeModelId] ??
        _runtimeSmokeAdHocSpec(engine: engine, artifactPath: resolvedArtifact);
    if (baseSpec == null) {
      throw StateError('Unknown built-in model id: $_runtimeSmokeModelId');
    }
    if (resolvedArtifact.isNotEmpty) {
      await _waitForRuntimeSmokeArtifact(
        resolvedArtifact,
        _runtimeSmokeWaitForArtifactSeconds,
      );
    }
    final spec = resolvedArtifact.isEmpty
        ? baseSpec
        : baseSpec.copyWith(
            platformArtifacts: {
              engine: RuntimeArtifact(
                engine: engine,
                path: resolvedArtifact,
                sourceUri: resolvedArtifact.contains('://')
                    ? resolvedArtifact
                    : null,
                targetPlatforms: [RuntimePlatformCurrent.current().name],
              ),
            },
          );
    if (await _tryRunQwen3AsrRuntimeSmoke(
      payload: payload,
      baseSpec: baseSpec,
      engine: engine,
      resolvedArtifact: resolvedArtifact,
    )) {
      payload['finished_at'] = DateTime.now().toUtc().toIso8601String();
      await _emitRuntimeSmokePayload(payload);
      return;
    }
    final before = NativeRuntimeMemory.snapshot();
    final registry = RuntimeRegistry.native();
    final session = await registry.loadAsync(
      spec,
      options: RuntimeOptions(
        engine: engine,
        diagnostics: true,
        backendOptions: _runtimeSmokeBackendOptions(engine),
      ),
    );
    try {
      final after = NativeRuntimeMemory.snapshot();
      payload['passed'] = true;
      payload['memory_before'] = before;
      payload['memory_after'] = after;
      payload['diagnostics'] = session.diagnostics;
    } finally {
      session.close();
    }
  } catch (error, stackTrace) {
    payload['passed'] = false;
    payload['error'] = error.toString();
    payload['stack'] = stackTrace.toString();
  }
  payload['finished_at'] = DateTime.now().toUtc().toIso8601String();
  await _emitRuntimeSmokePayload(payload);
}

ModelSpec? _runtimeSmokeAdHocSpec({
  required RuntimeEngine engine,
  required String artifactPath,
}) {
  if (artifactPath.isEmpty) return null;
  return ModelSpec(
    id: _runtimeSmokeModelId,
    family: 'Runtime smoke ad-hoc artifact',
    modalities: const [ModelModality.textGeneration],
    description: 'Temporary model spec for explicit runtime smoke artifacts.',
    platformArtifacts: {
      engine: RuntimeArtifact(
        engine: engine,
        path: artifactPath,
        sourceUri: artifactPath.contains('://') ? artifactPath : null,
        targetPlatforms: [RuntimePlatformCurrent.current().name],
      ),
    },
  );
}

Future<bool> _tryRunQwen3AsrRuntimeSmoke({
  required Map<String, Object?> payload,
  required ModelSpec baseSpec,
  required RuntimeEngine engine,
  required String resolvedArtifact,
}) async {
  if (_runtimeSmokeModelId != 'qwen3_asr') {
    return false;
  }
  final artifact = _qwen3AsrRuntimeSmokeArtifact(
    baseSpec: baseSpec,
    engine: engine,
    resolvedArtifact: resolvedArtifact,
  );
  if (artifact == null) {
    throw StateError('qwen3_asr has no ${engine.name} artifact.');
  }
  final before = NativeRuntimeMemory.snapshot();
  final bundlePath = await _resolveQwen3AsrBundlePath(artifact);
  final tokenizerPath = engine == RuntimeEngine.coreml
      ? await _resolveQwen3AsrTokenizerPath(bundlePath)
      : null;
  final runner = _loadQwen3AsrRunner(
    engine: engine,
    bundlePath: bundlePath,
    tokenizerPath: tokenizerPath,
  );
  try {
    final after = NativeRuntimeMemory.snapshot();
    payload['passed'] = true;
    payload['memory_before'] = before;
    payload['memory_after'] = after;
    payload['diagnostics'] = {
      ..._qwen3AsrRuntimeDiagnostics(runner: runner, engine: engine),
      'bundle_path': bundlePath,
      ...tokenizerPath == null
          ? const <String, Object?>{}
          : {'tokenizer_path': tokenizerPath},
      if (_runtimeSmokeCoreMlComputeUnits.trim().isNotEmpty)
        'requested_coreml_compute_units': _runtimeSmokeCoreMlComputeUnits
            .trim(),
    };
    return true;
  } finally {
    runner.close();
  }
}

RuntimeArtifact? _qwen3AsrRuntimeSmokeArtifact({
  required ModelSpec baseSpec,
  required RuntimeEngine engine,
  required String resolvedArtifact,
}) {
  if (resolvedArtifact.isNotEmpty) {
    return RuntimeArtifact(
      engine: engine,
      path: resolvedArtifact,
      sourceUri: resolvedArtifact.contains('://') ? resolvedArtifact : null,
      targetPlatforms: [RuntimePlatformCurrent.current().name],
    );
  }
  return baseSpec.platformArtifacts[engine];
}

dynamic _loadQwen3AsrRunner({
  required RuntimeEngine engine,
  required String bundlePath,
  required String? tokenizerPath,
}) {
  final options = RuntimeOptions(
    engine: engine,
    diagnostics: true,
    prefer: engine == RuntimeEngine.onnx && Platform.isAndroid
        ? const [Accelerator.npu, Accelerator.gpu, Accelerator.cpu]
        : const [Accelerator.gpu, Accelerator.cpu],
    backendOptions: _runtimeSmokeBackendOptions(engine),
  );
  return switch (engine) {
    RuntimeEngine.coreml => Qwen3AsrCoreMlRunner.loadCoreMlBundle(
      bundlePath,
      tokenizerPath: tokenizerPath!,
      options: RuntimeOptions(
        engine: engine,
        diagnostics: true,
        prefer: const [Accelerator.ane, Accelerator.gpu, Accelerator.cpu],
        backendOptions: _runtimeSmokeBackendOptions(engine),
      ),
    ),
    RuntimeEngine.onnx => Qwen3AsrNativeRunner.loadOnnxBundle(
      bundlePath,
      options: options,
    ),
    RuntimeEngine.litert => Qwen3AsrNativeRunner.loadLiteRtBundle(
      bundlePath,
      options: options,
    ),
    RuntimeEngine.mlx => throw StateError(
      'qwen3_asr runtime smoke does not support MLX in Flutter example.',
    ),
  };
}

Map<String, Object?> _qwen3AsrRuntimeDiagnostics({
  required dynamic runner,
  required RuntimeEngine engine,
}) {
  if (runner is Qwen3AsrCoreMlRunner) {
    return runner.componentDiagnostics(includeDecoder: true);
  }
  if (runner is Qwen3AsrNativeRunner) {
    return runner.componentDiagnostics();
  }
  return <String, Object?>{'engine': engine.name};
}

Map<String, Object?> _runtimeSmokeBackendOptions(RuntimeEngine engine) {
  final options = <String, Object?>{};
  final computeUnits = _runtimeSmokeCoreMlComputeUnits.trim();
  if (engine == RuntimeEngine.coreml && computeUnits.isNotEmpty) {
    options['coremlComputeUnits'] = computeUnits;
  }
  final provider = _runtimeSmokeProvider.trim();
  if (engine == RuntimeEngine.onnx && provider.isNotEmpty) {
    options['provider'] = provider;
  }
  if (engine == RuntimeEngine.onnx && _runtimeSmokeRequireProvider) {
    options['requireProvider'] = true;
  }
  final delegate = _runtimeSmokeDelegate.trim();
  if (engine == RuntimeEngine.litert && delegate.isNotEmpty) {
    options['delegate'] = delegate;
  }
  if (engine == RuntimeEngine.litert && _runtimeSmokeRequireDelegate) {
    options['requireDelegate'] = true;
  }
  if (engine == RuntimeEngine.litert && _runtimeSmokeLiteRtSectionIndex >= 0) {
    options['litertSectionIndex'] = _runtimeSmokeLiteRtSectionIndex;
  }
  return options;
}

Future<String> _resolveQwen3AsrBundlePath(RuntimeArtifact artifact) async {
  if (artifact.path.contains('://')) {
    final rootArtifact = artifact.copyWith(
      path: _hfRepositoryRootUri(artifact.path),
    );
    final resolved = await HuggingFaceArtifactCache().resolve(rootArtifact);
    return resolved.path;
  }
  final path = artifact.path;
  if (path.endsWith('.mlmodelc') || path.endsWith('.mlpackage')) {
    return File(path).parent.path;
  }
  return path;
}

Future<String> _resolveQwen3AsrTokenizerPath(String bundlePath) async {
  if (File('$bundlePath/tokenizer.json').existsSync() ||
      File('$bundlePath/vocab.json').existsSync()) {
    return bundlePath;
  }
  final resolved = await HuggingFaceArtifactCache().resolve(
    const RuntimeArtifact(
      engine: RuntimeEngine.onnx,
      path: 'hf://andrewleech/qwen3-asr-1.7b-onnx/tokenizer.json',
    ),
  );
  return File(resolved.path).parent.path;
}

String _hfRepositoryRootUri(String artifact) {
  final uri = Uri.parse(artifact);
  if (uri.scheme != 'hf' || uri.host.isEmpty || uri.pathSegments.isEmpty) {
    return artifact;
  }
  return 'hf://${uri.host}/${uri.pathSegments.first}/.';
}

String _resolveRuntimeSmokeArtifactPath(String value) {
  final artifact = value.trim();
  if (artifact.isEmpty) {
    return '';
  }
  if (artifact.contains('://') || artifact.startsWith('/')) {
    return artifact;
  }
  if (artifact.startsWith('Documents/') ||
      artifact.startsWith('Library/') ||
      artifact.startsWith('tmp/')) {
    final home = _runtimeSandboxHomePath();
    if (home != null && home.isNotEmpty) {
      return '$home/$artifact';
    }
  }
  return artifact;
}

Future<void> _waitForRuntimeSmokeArtifact(
  String path,
  int timeoutSeconds,
) async {
  if (timeoutSeconds <= 0) {
    return;
  }
  if (path.contains('://')) {
    return;
  }
  final deadline = DateTime.now().add(Duration(seconds: timeoutSeconds));
  while (DateTime.now().isBefore(deadline)) {
    final type = FileSystemEntity.typeSync(path);
    if (_isRuntimeSmokeArtifactReady(path, type)) {
      return;
    }
    await Future<void>.delayed(const Duration(milliseconds: 250));
  }
}

bool _isRuntimeSmokeArtifactReady(String path, FileSystemEntityType type) {
  if (type == FileSystemEntityType.notFound) {
    return false;
  }
  if (type == FileSystemEntityType.directory) {
    try {
      final entries = Directory(path).listSync(followLinks: false);
      return entries.isNotEmpty;
    } catch (_) {
      return false;
    }
  }
  if (type != FileSystemEntityType.file) {
    return true;
  }
  final file = File(path);
  final lower = path.toLowerCase();
  try {
    if (file.lengthSync() <= 0) {
      return false;
    }
    if (!lower.endsWith('.json')) {
      return true;
    }
    final decoded = jsonDecode(file.readAsStringSync());
    if (decoded is! Map<String, dynamic>) {
      return false;
    }
    return _isPipelineSpecReady(path, decoded);
  } catch (_) {
    return false;
  }
}

bool _isPipelineSpecReady(String specPath, Map<String, dynamic> decoded) {
  final stages = decoded['stages'];
  if (stages is! List) {
    return true;
  }
  final specDir = File(specPath).parent.path;
  for (final stage in stages) {
    if (stage is! Map) {
      continue;
    }
    final modelValue = stage['model'];
    if (modelValue is! String || modelValue.trim().isEmpty) {
      continue;
    }
    final modelPath = _resolvePipelineModelPath(specDir, modelValue);
    final modelType = FileSystemEntity.typeSync(modelPath);
    if (modelType == FileSystemEntityType.notFound) {
      return false;
    }
    if (modelType == FileSystemEntityType.directory) {
      final coremlData = File('$modelPath/coremldata.bin');
      if (!coremlData.existsSync() || coremlData.lengthSync() <= 0) {
        return false;
      }
      continue;
    }
    if (modelType == FileSystemEntityType.file) {
      final modelFile = File(modelPath);
      if (modelFile.lengthSync() <= 0) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

String _resolvePipelineModelPath(String specDir, String modelValue) {
  final model = modelValue.trim();
  if (model.startsWith('/')) {
    return model;
  }
  return '$specDir/$model';
}

String? _runtimeSandboxHomePath() {
  final home = Platform.environment['HOME'];
  if (home != null && home.isNotEmpty) {
    return home;
  }
  final tempPath = Directory.systemTemp.path;
  if (tempPath.isEmpty) {
    return null;
  }
  try {
    return Directory(tempPath).parent.path;
  } catch (_) {
    return null;
  }
}

Future<void> _emitRuntimeSmokePayload(Map<String, Object?> payload) async {
  final encoded = base64Encode(utf8.encode(jsonEncode(payload)));
  const chunkSize = 700;
  final totalChunks = math.max(1, (encoded.length / chunkSize).ceil());
  await _emitRuntimeSmokeLogLine('DMF_RUNTIME_SMOKE_RESULT_BEGIN:$totalChunks');
  for (var index = 0; index < totalChunks; index++) {
    final start = index * chunkSize;
    final end = math.min(start + chunkSize, encoded.length);
    final chunk = encoded.substring(start, end);
    await _emitRuntimeSmokeLogLine(
      'DMF_RUNTIME_SMOKE_RESULT_CHUNK:${index + 1}/$totalChunks:$chunk',
    );
  }
  await _emitRuntimeSmokeLogLine('DMF_RUNTIME_SMOKE_RESULT_END');
}

Future<void> _emitRuntimeSmokeLogLine(String line) async {
  // ignore: avoid_print - runtime smoke markers are consumed from Flutter logs.
  print(line);
  if (!_isAndroidRuntime()) {
    return;
  }
  try {
    await _runtimeSmokeLogMethod.invokeMethod<void>('logLine', line);
  } catch (_) {
    // Best-effort bridge for release-mode Android logcat capture.
  }
}

bool _isAndroidRuntime() {
  return !kIsWeb && defaultTargetPlatform == TargetPlatform.android;
}

RuntimeEngine _runtimeEngine(String value) {
  for (final engine in RuntimeEngine.values) {
    if (engine.name == value) {
      return engine;
    }
  }
  throw ArgumentError.value(value, 'DMF_RUNTIME_SMOKE_ENGINE');
}

class _RuntimeSmokeApp extends StatelessWidget {
  const _RuntimeSmokeApp();

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(body: Center(child: Text('Runtime smoke mode'))),
    );
  }
}

final class DemoSnapshot {
  const DemoSnapshot({
    required this.version,
    required this.defaultDevice,
    required this.metalAvailable,
    required this.activeBytes,
    required this.cacheBytes,
    required this.peakBytes,
    required this.addResult,
    required this.matmulResult,
  });

  final String version;
  final String defaultDevice;
  final bool metalAvailable;
  final int activeBytes;
  final int cacheBytes;
  final int peakBytes;
  final List<Object> addResult;
  final List<Object> matmulResult;
}

class DemoApp extends StatelessWidget {
  const DemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'dart_mlx_ffi Demo',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const DemoScreen(),
    );
  }
}

class DemoScreen extends StatefulWidget {
  const DemoScreen({super.key});

  @override
  State<DemoScreen> createState() => _DemoScreenState();
}

class _DemoScreenState extends State<DemoScreen> {
  DemoSnapshot? _snapshot;
  Object? _error;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _runDemo();
  }

  Future<void> _runDemo() async {
    if (_loading) return;
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
      final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
      final add = mx.add(a, b);
      final matmul = mx.matmul(a, b);

      try {
        MlxRuntime.evalAll([add, matmul]);
        final device = MlxDevice.defaultDevice();
        try {
          final snapshot = DemoSnapshot(
            version: MlxVersion.current(),
            defaultDevice:
                '${device.type.name.toLowerCase()}:${device.index}'
                ' available=${device.isAvailable}',
            metalAvailable: MlxMetal.isAvailable(),
            activeBytes: MlxMemory.activeBytes(),
            cacheBytes: MlxMemory.cacheBytes(),
            peakBytes: MlxMemory.peakBytes(),
            addResult: add.toList(),
            matmulResult: matmul.toList(),
          );
          if (!mounted) return;
          setState(() {
            _snapshot = snapshot;
          });
        } finally {
          device.close();
        }
      } finally {
        matmul.close();
        add.close();
        b.close();
        a.close();
      }
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _error = error;
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final snapshot = _snapshot;
    return Scaffold(
      appBar: AppBar(title: const Text('dart_mlx_ffi Demo')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          FilledButton(
            onPressed: _loading ? null : _runDemo,
            child: Text(_loading ? 'Running…' : 'Run Demo'),
          ),
          const SizedBox(height: 16),
          if (_error != null)
            _InfoCard(title: 'Error', body: _error.toString())
          else if (snapshot == null)
            const _InfoCard(title: 'Status', body: 'No snapshot yet.')
          else ...[
            _InfoCard(title: 'MLX Version', body: snapshot.version),
            _InfoCard(title: 'Default Device', body: snapshot.defaultDevice),
            _InfoCard(
              title: 'Metal Available',
              body: snapshot.metalAvailable.toString(),
            ),
            _InfoCard(
              title: 'Memory',
              body:
                  'active=${snapshot.activeBytes} bytes\n'
                  'cache=${snapshot.cacheBytes} bytes\n'
                  'peak=${snapshot.peakBytes} bytes',
            ),
            _InfoCard(title: 'Add Result', body: snapshot.addResult.join(', ')),
            _InfoCard(
              title: 'Matmul Result',
              body: snapshot.matmulResult.join(', '),
            ),
          ],
        ],
      ),
    );
  }
}

class _InfoCard extends StatelessWidget {
  const _InfoCard({required this.title, required this.body});

  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            SelectableText(body),
          ],
        ),
      ),
    );
  }
}
