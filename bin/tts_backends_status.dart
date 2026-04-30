import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';

Future<void> main(List<String> args) async {
  final parsed = _Args(args);
  final pretty = parsed.flag('pretty');
  final root = _discoverProjectRoot(parsed.option('root'));
  final smokeOnnx = parsed.flag('smoke-onnx');
  final loadOnnx = parsed.flag('load-onnx') || smokeOnnx;
  final smokeWarmupIterations =
      int.tryParse(parsed.option('smoke-warmup') ?? '1') ?? 1;
  final smokeIterations =
      int.tryParse(parsed.option('smoke-iters') ?? '5') ?? 5;
  final provider = parsed.option('provider') ?? 'cuda';
  final deviceId = int.tryParse(parsed.option('device-id') ?? '0') ?? 0;
  final requireProvider = !parsed.flag('allow-cpu-fallback');
  final numThreads = int.tryParse(parsed.option('num-threads') ?? '4') ?? 4;
  final componentNames = parsed
      .option('cosyvoice2-components')
      ?.split(',')
      .map((value) => value.trim())
      .where((value) => value.isNotEmpty)
      .toList(growable: false);
  final runtimeDependencyAudit = root == null
      ? null
      : RuntimeDependencyAudit.inspect(root: root, provider: provider);
  final assetAudit = root == null ? null : TtsBackendAssetAudit.audit(root);
  final dependencySkipReason = requireProvider
      ? runtimeDependencyAudit?.skipReason
      : null;
  final payload = {
    ...TtsBackendCatalog.toJson(),
    'nativeReusePlan': TtsBackendNativePlan.fromCatalog().toJson(),
    'loadOnnx': loadOnnx,
    'smokeOnnx': smokeOnnx,
    'smokeWarmupIterations': smokeWarmupIterations,
    'smokeIterations': smokeIterations,
    'requestedProvider': provider,
    'deviceId': deviceId,
    if (root != null) ...{
      'assetAudit': assetAudit!.toJson(),
      'runtimeDependencyAudit': runtimeDependencyAudit?.toJson(),
      'models': await _loadModelStatuses(
        root: root,
        assetAudit: assetAudit,
        provider: provider,
        deviceId: deviceId,
        requireProvider: requireProvider,
        numThreads: numThreads,
        loadOnnx: loadOnnx,
        smokeOnnx: smokeOnnx,
        smokeWarmupIterations: smokeWarmupIterations,
        smokeIterations: smokeIterations,
        componentNames: componentNames,
        parsed: parsed,
        dependencySkipReason: dependencySkipReason,
      ),
    },
  };
  final encoder = pretty
      ? const JsonEncoder.withIndent('  ')
      : const JsonEncoder();
  stdout.writeln(encoder.convert(payload));
}

Future<List<Map<String, Object?>>> _loadModelStatuses({
  required String root,
  required TtsBackendAssetAudit assetAudit,
  required String provider,
  required int deviceId,
  required bool requireProvider,
  required int numThreads,
  required bool loadOnnx,
  required bool smokeOnnx,
  required int smokeWarmupIterations,
  required int smokeIterations,
  required List<String>? componentNames,
  required _Args parsed,
  required String? dependencySkipReason,
}) async {
  final paths = DartUniFrontendTtsPaths.fromUniFrontendRoot(root);
  final backendOptions = _backendOptions(root, provider, parsed);
  final closeables = <Object>[];
  try {
    return [
      await _kokoroStatus(
        paths: paths,
        provider: provider,
        deviceId: deviceId,
        requireProvider: requireProvider,
        numThreads: numThreads,
        backendOptions: backendOptions,
        loadOnnx: loadOnnx,
        smokeOnnx: smokeOnnx,
        smokeWarmupIterations: smokeWarmupIterations,
        smokeIterations: smokeIterations,
        dependencySkipReason: dependencySkipReason,
        closeables: closeables,
      ),
      _cosyVoice2Status(
        paths: paths.cosyVoice2Paths,
        provider: provider,
        deviceId: deviceId,
        requireProvider: requireProvider,
        numThreads: numThreads,
        backendOptions: backendOptions,
        loadOnnx: loadOnnx,
        smokeOnnx: smokeOnnx,
        smokeWarmupIterations: smokeWarmupIterations,
        smokeIterations: smokeIterations,
        componentNames: componentNames,
        dependencySkipReason: dependencySkipReason,
        closeables: closeables,
      ),
      ..._genericOnnxComponentStatuses(
        assetAudit: assetAudit,
        provider: provider,
        deviceId: deviceId,
        requireProvider: requireProvider,
        numThreads: numThreads,
        backendOptions: backendOptions,
        loadOnnx: loadOnnx,
        smokeOnnx: smokeOnnx,
        smokeWarmupIterations: smokeWarmupIterations,
        smokeIterations: smokeIterations,
        dependencySkipReason: dependencySkipReason,
        closeables: closeables,
      ),
    ];
  } finally {
    for (final closeable in closeables.reversed) {
      if (closeable is KokoroDartRuntime) {
        closeable.close();
      } else if (closeable is CosyVoice2PartialOnnxBundle) {
        closeable.close();
      } else if (closeable is TtsOnnxComponentBundle) {
        closeable.close();
      }
    }
  }
}

Future<Map<String, Object?>> _kokoroStatus({
  required DartUniFrontendTtsPaths paths,
  required String provider,
  required int deviceId,
  required bool requireProvider,
  required int numThreads,
  required Map<String, Object?> backendOptions,
  required bool loadOnnx,
  required bool smokeOnnx,
  required int smokeWarmupIterations,
  required int smokeIterations,
  required String? dependencySkipReason,
  required List<Object> closeables,
}) async {
  final files = {
    'model': paths.kokoroModelPath,
    'voices': paths.kokoroVoicesPath,
    'config': paths.kokoroConfigPath,
  };
  final status = <String, Object?>{
    'provider': 'kokoro',
    'runtime': 'dart_inference_onnx',
    'python': false,
    'readyForSynthesis': files.values.every((path) => File(path).existsSync()),
    'files': {
      for (final entry in files.entries) entry.key: _fileStatus(entry.value),
    },
  };
  if (!loadOnnx) {
    return status;
  }
  if (dependencySkipReason != null) {
    return status
      ..['loaded'] = false
      ..['skipped'] = true
      ..['skipReason'] = dependencySkipReason;
  }
  final stopwatch = Stopwatch()..start();
  try {
    final runtime = await KokoroDartRuntime.load(
      modelPath: paths.kokoroModelPath,
      voicesPath: paths.kokoroVoicesPath,
      configPath: paths.kokoroConfigPath,
      provider: provider,
      deviceId: deviceId,
      requireProvider: requireProvider,
      numThreads: numThreads,
      backendOptions: backendOptions,
    );
    stopwatch.stop();
    closeables.add(runtime);
    status
      ..['loaded'] = true
      ..['loadElapsedMs'] = stopwatch.elapsedMicroseconds / 1000.0
      ..['selectedProvider'] = runtime.selectedProvider
      ..['voiceCount'] = runtime.voiceNames.length
      ..['diagnostics'] = runtime.session.diagnostics;
    if (smokeOnnx) {
      status['smoke'] = _kokoroSmokeStatus(
        runtime: runtime,
        warmupIterations: smokeWarmupIterations,
        iterations: smokeIterations,
      );
    }
  } catch (error) {
    stopwatch.stop();
    status
      ..['loaded'] = false
      ..['loadElapsedMs'] = stopwatch.elapsedMicroseconds / 1000.0
      ..['error'] = '$error';
  }
  return status;
}

Map<String, Object?> _kokoroSmokeStatus({
  required KokoroDartRuntime runtime,
  required int warmupIterations,
  required int iterations,
}) {
  const phonemes = 'həlˈoʊ';
  const voice = 'zf_xiaoni';
  final warmups = warmupIterations < 0 ? 0 : warmupIterations;
  final runs = iterations < 1 ? 1 : iterations;
  final failureTimer = Stopwatch()..start();
  try {
    for (var i = 0; i < warmups; i++) {
      runtime.synthesizePhonemes(phonemes: phonemes, voice: voice, speed: 1.0);
    }

    final timer = Stopwatch();
    var totalElapsedMicroseconds = 0;
    var minElapsedMicroseconds = 0;
    var maxElapsedMicroseconds = 0;
    var audioBytes = 0;
    for (var i = 0; i < runs; i++) {
      timer
        ..reset()
        ..start();
      final audio = runtime.synthesizePhonemes(
        phonemes: phonemes,
        voice: voice,
        speed: 1.0,
      );
      timer.stop();
      audioBytes = audio.length;
      final elapsed = timer.elapsedMicroseconds;
      totalElapsedMicroseconds += elapsed;
      if (i == 0 || elapsed < minElapsedMicroseconds) {
        minElapsedMicroseconds = elapsed;
      }
      if (elapsed > maxElapsedMicroseconds) {
        maxElapsedMicroseconds = elapsed;
      }
    }
    return {
      'ran': true,
      'warmupIterations': warmups,
      'iterations': runs,
      'elapsedMs': (totalElapsedMicroseconds ~/ runs) / 1000.0,
      'totalElapsedMs': totalElapsedMicroseconds / 1000.0,
      'minElapsedMs': minElapsedMicroseconds / 1000.0,
      'maxElapsedMs': maxElapsedMicroseconds / 1000.0,
      'audioFormat': 'wav',
      'audioBytes': audioBytes,
      'phonemeTokenCount': runtime.phonemeTokenCount(phonemes),
      'phonemeChunkCount': runtime.phonemeChunkCount(phonemes),
    };
  } catch (error) {
    failureTimer.stop();
    return {
      'ran': true,
      'warmupIterations': warmups,
      'iterations': runs,
      'elapsedMs': failureTimer.elapsedMicroseconds / 1000.0,
      'error': '$error',
    };
  }
}

Map<String, Object?> _cosyVoice2Status({
  required CosyVoice2Paths paths,
  required String provider,
  required int deviceId,
  required bool requireProvider,
  required int numThreads,
  required Map<String, Object?> backendOptions,
  required bool loadOnnx,
  required bool smokeOnnx,
  required int smokeWarmupIterations,
  required int smokeIterations,
  required List<String>? componentNames,
  required String? dependencySkipReason,
  required List<Object> closeables,
}) {
  if (loadOnnx && dependencySkipReason != null) {
    final inspected = CosyVoice2PartialOnnxBundle.inspect(paths: paths);
    closeables.add(inspected);
    return inspected.toJson()
      ..['skipped'] = true
      ..['skipReason'] = dependencySkipReason;
  }
  final bundle = loadOnnx
      ? CosyVoice2PartialOnnxBundle.load(
          paths: paths,
          provider: provider,
          deviceId: deviceId,
          requireProvider: requireProvider,
          numThreads: numThreads,
          backendOptions: backendOptions,
          componentNames: componentNames,
          smoke: smokeOnnx,
          smokeWarmupIterations: smokeWarmupIterations,
          smokeIterations: smokeIterations,
        )
      : CosyVoice2PartialOnnxBundle.inspect(paths: paths);
  closeables.add(bundle);
  return bundle.toJson();
}

List<Map<String, Object?>> _genericOnnxComponentStatuses({
  required TtsBackendAssetAudit assetAudit,
  required String provider,
  required int deviceId,
  required bool requireProvider,
  required int numThreads,
  required Map<String, Object?> backendOptions,
  required bool loadOnnx,
  required bool smokeOnnx,
  required int smokeWarmupIterations,
  required int smokeIterations,
  required String? dependencySkipReason,
  required List<Object> closeables,
}) {
  final statuses = <Map<String, Object?>>[];
  for (final capability in TtsBackendCatalog.all) {
    if (capability.provider == 'kokoro' ||
        capability.provider == 'cosyvoice2' ||
        capability.onnxTargets.isEmpty) {
      continue;
    }
    final audit = assetAudit.providers[capability.provider];
    final providerDir = audit?.providerDir;
    final bundle = loadOnnx && dependencySkipReason == null
        ? TtsOnnxComponentBundle.load(
            capability: capability,
            providerDir: providerDir,
            provider: provider,
            deviceId: deviceId,
            requireProvider: requireProvider,
            numThreads: numThreads,
            backendOptions: backendOptions,
            smoke: smokeOnnx,
            smokeWarmupIterations: smokeWarmupIterations,
            smokeIterations: smokeIterations,
          )
        : TtsOnnxComponentBundle.inspect(
            capability: capability,
            providerDir: providerDir,
          );
    closeables.add(bundle);
    final json = bundle.toJson();
    if (loadOnnx && dependencySkipReason != null) {
      json
        ..['skipped'] = true
        ..['skipReason'] = dependencySkipReason;
    }
    statuses.add(json);
  }
  return statuses;
}

Map<String, Object?> _fileStatus(String path) {
  final file = File(path);
  final exists = file.existsSync();
  return {
    'path': path,
    'exists': exists,
    if (exists) 'sizeBytes': file.lengthSync(),
  };
}

Map<String, Object?> _backendOptions(
  String root,
  String provider,
  _Args parsed,
) {
  final preloadLibraries = discoverDefaultOnnxRuntimePreloadLibraries(
    libraryDirectories: [
      '$root/artifacts/runtime/onnxruntime/lib',
      '$root/artifacts/runtime/cuda/lib',
      '$root/artifacts/runtime/tensorrt/lib',
    ],
    libraryNames: onnxRuntimePreloadLibraryNamesForProvider(provider),
    runtimeEnvSearchRoots: [root],
  );
  final trtCacheDir = parsed.option('trt-cache-dir');
  return {
    'cudaMemoryLimitMb':
        int.tryParse(parsed.option('cuda-memory-limit-mb') ?? '16384') ?? 16384,
    if (parsed.flag('trt-fp16')) 'trtFp16': true,
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
    if (parsed.option('trt-max-partition-iterations') != null)
      'trtMaxPartitionIterations': int.tryParse(
        parsed.option('trt-max-partition-iterations')!,
      ),
    if (parsed.flag('trt-force-sequential-engine-build'))
      'trtForceSequentialEngineBuild': true,
    if (preloadLibraries.isNotEmpty)
      'preloadLibraries': encodeOnnxRuntimePreloadLibraries(preloadLibraries),
  };
}

String? _discoverProjectRoot(String? explicitRoot) {
  final envRoot = Platform.environment['UNIFRONTEND_ROOT'];
  for (final value in [explicitRoot, envRoot]) {
    if (value == null || value.isEmpty) {
      continue;
    }
    final root = Directory(value).absolute;
    if (_looksLikeUniFrontendRoot(root)) {
      return root.path;
    }
  }
  return null;
}

bool _looksLikeUniFrontendRoot(Directory directory) {
  return File('${directory.path}/src/ttsbackends/registry.toml').existsSync() &&
      Directory('${directory.path}/src/unifrontend').existsSync();
}

final class _Args {
  _Args(this.args);

  final List<String> args;

  bool flag(String name) => args.contains('--$name');

  String? option(String name) {
    final prefix = '--$name=';
    for (var i = 0; i < args.length; i++) {
      final value = args[i];
      if (value.startsWith(prefix)) {
        return value.substring(prefix.length);
      }
      if (value == '--$name' && i + 1 < args.length) {
        return args[i + 1];
      }
    }
    return null;
  }
}
