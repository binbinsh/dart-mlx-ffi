import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

Future<void> main(List<String> args) async {
  final parsed = _Args(args);
  final modelPath = parsed.option('model', required: true)!;
  final provider = parsed.option('provider') ?? 'cuda';
  final deviceId = int.tryParse(parsed.option('device-id') ?? '1') ?? 1;
  final requireProvider = parsed.flag('require-provider');
  final trtFp16 = parsed.flag('trt-fp16');
  final preferCpu = parsed.flag('prefer-cpu');
  final numThreads = int.tryParse(parsed.option('num-threads') ?? '0') ?? 0;
  final warmupIters = int.tryParse(parsed.option('warmup-iters') ?? '5') ?? 5;
  final iters = int.tryParse(parsed.option('iters') ?? '30') ?? 30;
  final trtCacheDir = parsed.option('trt-cache-dir');
  final preloadLibraries = _preloadLibrariesFromArgs(parsed, provider);
  if (requireProvider) {
    final audit = RuntimeDependencyAudit.inspect(
      root: _runtimeSearchRoot(parsed),
      provider: provider,
      extraSearchDirs: _runtimeLibraryDirsFromArgs(parsed),
    );
    final dependencyError = audit.skipReason;
    if (dependencyError != null) {
      stdout.writeln(
        jsonEncode({
          'ok': false,
          'stage': 'preflight',
          'error': dependencyError,
          'runtimeDependencyAudit': audit.toJson(),
        }),
      );
      exitCode = 78;
      return;
    }
  }

  final batchSize = int.tryParse(parsed.option('batch-size') ?? '8') ?? 8;
  final tokenLength =
      int.tryParse(parsed.option('token-length') ?? '512') ?? 512;
  final charLength =
      int.tryParse(parsed.option('char-length') ?? '1024') ?? 1024;
  final homographTargets =
      int.tryParse(parsed.option('homograph-targets') ?? '16') ?? 16;
  final polyphoneTargets =
      int.tryParse(parsed.option('polyphone-targets') ?? '16') ?? 16;
  final homographClasses =
      int.tryParse(parsed.option('homograph-classes') ?? '326') ?? 326;
  final polyphoneClasses =
      int.tryParse(parsed.option('polyphone-classes') ?? '658') ?? 658;

  late final DartOnnxSession session;
  try {
    session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: modelPath,
        id: 'unifrontend_onnx_smoke',
        family: 'unifrontend_onnx_smoke',
        provider: provider,
        deviceId: deviceId,
        requireProvider: requireProvider,
        preferCpu: preferCpu,
        numThreads: numThreads > 0 ? numThreads : 0,
        backendOptions: {
          if (trtFp16) 'trtFp16': true,
          if (trtCacheDir != null && trtCacheDir.isNotEmpty)
            'trtCacheDir': trtCacheDir,
          if (parsed.option('trt-workspace-mb') != null)
            'trtWorkspaceMemoryLimitMb': int.tryParse(
              parsed.option('trt-workspace-mb')!,
            ),
          if (parsed.option('trt-min-subgraph-size') != null)
            'trtMinSubgraphSize': int.tryParse(
              parsed.option('trt-min-subgraph-size')!,
            ),
          if (preloadLibraries.isNotEmpty)
            'preloadLibraries': encodeOnnxRuntimePreloadLibraries(
              preloadLibraries,
            ),
        },
      ),
    );
  } catch (error, stack) {
    stdout.writeln(
      jsonEncode({
        'ok': false,
        'stage': 'load',
        'error': '$error',
        'stack': '$stack',
      }),
    );
    exitCode = 1;
    return;
  }

  try {
    final inputs = {
      'input_ids': int64Tensor(Int64List(batchSize * tokenLength), [
        batchSize,
        tokenLength,
      ]),
      'attention_mask': int64Tensor(
        Int64List.fromList(List<int>.filled(batchSize * tokenLength, 1)),
        [batchSize, tokenLength],
      ),
      'char_ids': int64Tensor(Int64List(batchSize * charLength), [
        batchSize,
        charLength,
      ]),
      'char_attention_mask': int64Tensor(
        Int64List.fromList(List<int>.filled(batchSize * charLength, 1)),
        [batchSize, charLength],
      ),
      'homograph_target_masks': boolTensor(
        Uint8List(batchSize * homographTargets * tokenLength),
        [batchSize, homographTargets, tokenLength],
      ),
      'homograph_candidate_masks': boolTensor(
        Uint8List.fromList(
          List<int>.filled(batchSize * homographTargets * homographClasses, 1),
        ),
        [batchSize, homographTargets, homographClasses],
      ),
      'polyphone_target_char_masks': boolTensor(
        Uint8List(batchSize * polyphoneTargets * charLength),
        [batchSize, polyphoneTargets, charLength],
      ),
      'polyphone_candidate_masks': boolTensor(
        Uint8List.fromList(
          List<int>.filled(batchSize * polyphoneTargets * polyphoneClasses, 1),
        ),
        [batchSize, polyphoneTargets, polyphoneClasses],
      ),
    };
    var warmupTotalMicros = 0;
    for (var i = 0; i < warmupIters; i++) {
      final sw = Stopwatch()..start();
      session.run(inputs);
      sw.stop();
      warmupTotalMicros += sw.elapsedMicroseconds;
    }

    final latenciesMs = <double>[];
    DartOnnxResult? lastOutputs;
    for (var i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      lastOutputs = session.run(inputs);
      sw.stop();
      latenciesMs.add(sw.elapsedMicroseconds / 1000.0);
    }
    final outputs = lastOutputs;
    if (outputs == null) {
      throw StateError('no inference outputs produced');
    }

    final outputShapes = <String, List<int>>{};
    for (final entry in outputs.outputs.entries) {
      final value = entry.value;
      if (value is RuntimeTensor) {
        outputShapes[entry.key] = List<int>.from(value.shape);
      }
    }
    final sorted = List<double>.from(latenciesMs)..sort();
    final meanMs = latenciesMs.isEmpty
        ? 0.0
        : latenciesMs.reduce((a, b) => a + b) / latenciesMs.length;
    final totalMs = latenciesMs.fold<double>(0.0, (a, b) => a + b);
    final p50Ms = _percentile(sorted, 0.50);
    final p90Ms = _percentile(sorted, 0.90);
    final p99Ms = _percentile(sorted, 0.99);
    final samplesPerSec = meanMs <= 0.0 ? 0.0 : (batchSize * 1000.0 / meanMs);

    stdout.writeln(
      jsonEncode({
        'ok': true,
        'provider': outputs.providerOr(session.selectedProvider),
        'provider_appended':
            (outputs.diagnostics['provider_appended'] ??
            session.diagnostics['provider_appended']),
        'available_providers':
            (session.diagnostics['available_providers'] ?? const []),
        'num_threads': numThreads > 0 ? numThreads : null,
        'warmup_iters': warmupIters,
        'iters': iters,
        'warmup_total_ms': warmupTotalMicros / 1000.0,
        'latency_ms': {
          'mean': meanMs,
          'p50': p50Ms,
          'p90': p90Ms,
          'p99': p99Ms,
          'total': totalMs,
        },
        'throughput': {
          'samples_per_sec': samplesPerSec,
          'batch_size': batchSize,
        },
        'output_shapes': outputShapes,
      }),
    );
  } catch (error, stack) {
    stdout.writeln(
      jsonEncode({
        'ok': false,
        'stage': 'run',
        'error': '$error',
        'stack': '$stack',
      }),
    );
    exitCode = 1;
  } finally {
    session.close();
  }
}

final class _Args {
  _Args(List<String> args) : this._(_parseAll(args));

  _Args._((Map<String, String?>, Map<String, List<String>>) parsed)
    : _values = parsed.$1,
      _allValues = parsed.$2;

  final Map<String, String?> _values;
  final Map<String, List<String>> _allValues;

  bool flag(String name) => _values.containsKey(name);

  String? option(String name, {bool required = false}) {
    final value = _values[name];
    if (required && (value == null || value.isEmpty)) {
      throw ArgumentError('Missing --$name');
    }
    return value;
  }

  List<String> values(String name) => _allValues[name] ?? const [];

  static (Map<String, String?>, Map<String, List<String>>) _parseAll(
    List<String> args,
  ) {
    final values = <String, String?>{};
    final allValues = <String, List<String>>{};
    for (var i = 0; i < args.length; i++) {
      final arg = args[i];
      if (!arg.startsWith('--')) {
        throw ArgumentError('Unexpected positional argument: $arg');
      }
      final name = arg.substring(2);
      if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
        final value = args[++i];
        values[name] = value;
        allValues.putIfAbsent(name, () => []).add(value);
      } else {
        values[name] = null;
        allValues.putIfAbsent(name, () => []);
      }
    }
    return (values, allValues);
  }
}

double _percentile(List<double> sortedValues, double ratio) {
  if (sortedValues.isEmpty) {
    return 0.0;
  }
  final clamped = ratio.clamp(0.0, 1.0);
  final index = ((sortedValues.length - 1) * clamped).round();
  return sortedValues[index];
}

List<String> _preloadLibrariesFromArgs(_Args parsed, String provider) {
  return discoverDefaultOnnxRuntimePreloadLibraries(
    explicitLibraries: parsed.values('preload-library'),
    libraryDirectories: [
      ...parsed.values('cuda-library-dir'),
      ...parsed.values('native-library-dir'),
    ],
    libraryNames: onnxRuntimePreloadLibraryNamesForProvider(provider),
  );
}

String? _runtimeSearchRoot(_Args parsed) {
  for (final value in [
    parsed.option('runtime-root'),
    parsed.option('root'),
    Platform.environment['UNIFRONTEND_ROOT'],
    Directory.current.path,
  ]) {
    final trimmed = value?.trim();
    if (trimmed != null && trimmed.isNotEmpty) {
      return trimmed;
    }
  }
  return null;
}

List<String> _runtimeLibraryDirsFromArgs(_Args parsed) => [
  ...parsed.values('cuda-library-dir'),
  ...parsed.values('native-library-dir'),
  ...parsed.values('library-dir'),
];
