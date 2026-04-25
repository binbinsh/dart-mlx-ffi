import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/runtime.dart';

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
    final manifest = ModelManifest.builtIn();
    final baseSpec = manifest[_runtimeSmokeModelId];
    if (baseSpec == null) {
      throw StateError('Unknown built-in model id: $_runtimeSmokeModelId');
    }
    final engine = _runtimeEngine(_runtimeSmokeEngineName);
    final spec = _runtimeSmokeArtifact.isEmpty
        ? baseSpec
        : baseSpec.copyWith(
            platformArtifacts: {
              engine: RuntimeArtifact(
                engine: engine,
                path: _runtimeSmokeArtifact,
                sourceUri: _runtimeSmokeArtifact.contains('://')
                    ? _runtimeSmokeArtifact
                    : null,
                targetPlatforms: [RuntimePlatformCurrent.current().name],
              ),
            },
          );
    final before = NativeRuntimeMemory.snapshot();
    final registry = RuntimeRegistry.native();
    final session = await registry.loadAsync(
      spec,
      options: RuntimeOptions(engine: engine, diagnostics: true),
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
