import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';

Future<void> main(List<String> args) async {
  late final _Opts opts;
  try {
    opts = _parseArgs(args);
  } catch (error, stack) {
    _writeFailure(args.contains('--json'), error, stack, exit: 64);
    return;
  }

  final dependencyError = _runtimeDependencyError(opts);
  if (dependencyError != null) {
    _writeFailure(
      opts.json,
      StateError(dependencyError),
      StackTrace.current,
      exit: 78,
    );
    return;
  }

  final contextFile = File(opts.contextPath);
  if (contextFile.existsSync() && !opts.force) {
    _writeSuccess(opts, loadElapsedMs: 0, skipped: true);
    return;
  }

  await Directory(opts.artifactDir).create(recursive: true);
  if (opts.force) {
    await contextFile.deleteIfExists();
    await Directory('${opts.artifactDir}/engines').deleteIfExists();
    await Directory('${opts.artifactDir}/timing').deleteIfExists();
  }
  await Directory('${opts.artifactDir}/timing').create(recursive: true);

  final timer = Stopwatch()..start();
  Sarashina2DartRuntime? runtime;
  try {
    runtime = await Sarashina2DartRuntime.load(
      paths: Sarashina2TtsPaths(modelDir: opts.modelDir),
      provider: 'tensorrt',
      deviceId: opts.deviceId,
      requireProvider: true,
      numThreads: opts.numThreads,
      backendOptions: {
        if (opts.cudaMemoryLimitMb > 0)
          'cudaMemoryLimitMb': opts.cudaMemoryLimitMb,
        if (opts.preloadLibraries.isNotEmpty)
          'preloadLibraries': encodeOnnxRuntimePreloadLibraries(
            opts.preloadLibraries,
          ),
        'sarashina2TensorRtUseFlowStepContext': false,
        'trtEngineCachePath': 'engines',
        'trtTimingCachePath': '${opts.artifactDir}/timing',
        'trtDumpEpContextModel': true,
        'trtEpContextFilePath': opts.contextPath,
      },
      loadLlm: false,
    );
    timer.stop();
    if (!contextFile.existsSync()) {
      throw StateError(
        'TensorRT EPContext was not written: ${opts.contextPath}',
      );
    }
    _writeSuccess(opts, loadElapsedMs: timer.elapsedMicroseconds / 1000.0);
  } catch (error, stack) {
    _writeFailure(opts.json, error, stack, exit: 1);
  } finally {
    runtime?.close();
  }
}

final class _Opts {
  const _Opts({
    required this.modelDir,
    required this.artifactDir,
    required this.contextPath,
    required this.deviceId,
    required this.numThreads,
    required this.cudaMemoryLimitMb,
    required this.preloadLibraries,
    required this.force,
    required this.json,
  });

  final String modelDir;
  final String artifactDir;
  final String contextPath;
  final int deviceId;
  final int numThreads;
  final int cudaMemoryLimitMb;
  final List<String> preloadLibraries;
  final bool force;
  final bool json;
}

_Opts _parseArgs(List<String> args) {
  String? root;
  String? modelDir;
  String? artifactDir;
  var deviceId = 0;
  var numThreads = 4;
  var cudaMemoryLimitMb = 16384;
  var force = false;
  var json = false;
  final preloadLibraries = <String>[];
  final libraryDirectories = <String>[];

  for (var i = 0; i < args.length; i += 1) {
    final arg = args[i];
    String next() {
      if (i + 1 >= args.length) {
        throw ArgumentError('Missing value for $arg');
      }
      i += 1;
      return args[i];
    }

    switch (arg) {
      case '--root':
        root = next();
      case '--model-dir':
        modelDir = next();
      case '--artifact-dir':
        artifactDir = next();
      case '--device-id':
        deviceId = int.parse(next());
      case '--num-threads':
        numThreads = int.parse(next());
      case '--cuda-memory-limit-mb':
        cudaMemoryLimitMb = int.parse(next());
      case '--preload-library':
        preloadLibraries.add(next());
      case '--cuda-library-dir':
      case '--native-library-dir':
        libraryDirectories.add(next());
      case '--force':
        force = true;
      case '--json':
        json = true;
      case '-h':
      case '--help':
        _printUsage();
        exit(0);
      default:
        throw ArgumentError('Unknown flag: $arg');
    }
  }

  final resolvedModelDir = _resolveModelDir(modelDir: modelDir, root: root);
  final resolvedRoot = root ?? _uniFrontendRoot(resolvedModelDir);
  final resolvedArtifactDir =
      artifactDir ??
      (resolvedRoot == null
          ? '${Directory.current.path}/.dart_tool/sarashina2/tensorrt'
          : '$resolvedRoot/artifacts/runtime/sarashina2/tensorrt');
  final libraries = discoverDefaultOnnxRuntimePreloadLibraries(
    explicitLibraries: preloadLibraries,
    libraryDirectories: libraryDirectories,
    libraryNames: onnxRuntimePreloadLibraryNamesForProvider('tensorrt'),
    runtimeEnvSearchRoots: [?resolvedRoot, resolvedModelDir],
  );
  return _Opts(
    modelDir: resolvedModelDir,
    artifactDir: resolvedArtifactDir,
    contextPath: '$resolvedArtifactDir/flow_step_ctx.onnx',
    deviceId: deviceId,
    numThreads: numThreads,
    cudaMemoryLimitMb: cudaMemoryLimitMb,
    preloadLibraries: libraries,
    force: force,
    json: json,
  );
}

String _resolveModelDir({required String? modelDir, required String? root}) {
  if (modelDir != null && modelDir.isNotEmpty) {
    return modelDir;
  }
  final effectiveRoot = root ?? Platform.environment['UNIFRONTEND_ROOT'];
  if (effectiveRoot == null || effectiveRoot.isEmpty) {
    throw ArgumentError('--model-dir is required unless --root is provided');
  }
  return Sarashina2TtsPaths.fromUniFrontendRoot(effectiveRoot).modelDir;
}

String? _uniFrontendRoot(String modelDir) {
  const marker = '/src/ttsbackends/';
  final index = modelDir.indexOf(marker);
  return index > 0 ? modelDir.substring(0, index) : null;
}

String? _runtimeDependencyError(_Opts opts) {
  final audit = RuntimeDependencyAudit.inspect(
    root: opts.modelDir,
    provider: 'tensorrt',
    extraSearchDirs: [
      for (final path in opts.preloadLibraries) File(path).absolute.parent.path,
    ],
  );
  return audit.skipReason;
}

void _writeSuccess(
  _Opts opts, {
  required double loadElapsedMs,
  bool skipped = false,
}) {
  final contextFile = File(opts.contextPath);
  final enginesDir = Directory('${opts.artifactDir}/engines');
  final payload = {
    'ok': true,
    'skipped': skipped,
    'modelDir': opts.modelDir,
    'artifactDir': opts.artifactDir,
    'contextPath': opts.contextPath,
    'contextBytes': contextFile.existsSync() ? contextFile.lengthSync() : 0,
    'engineBytes': _directoryBytes(enginesDir),
    'loadElapsedMs': loadElapsedMs,
  };
  if (opts.json) {
    stdout.writeln(jsonEncode(payload));
  } else {
    stdout.writeln(
      skipped
          ? 'Sarashina2 TensorRT context already exists: ${opts.contextPath}'
          : 'Wrote Sarashina2 TensorRT context: ${opts.contextPath}',
    );
    stdout.writeln('engine bytes: ${payload['engineBytes']}');
  }
}

int _directoryBytes(Directory dir) {
  if (!dir.existsSync()) {
    return 0;
  }
  var total = 0;
  for (final entity in dir.listSync(recursive: true, followLinks: false)) {
    if (entity is File) {
      total += entity.lengthSync();
    }
  }
  return total;
}

void _writeFailure(
  bool json,
  Object error,
  StackTrace stack, {
  required int exit,
}) {
  if (json) {
    stdout.writeln(
      jsonEncode({'ok': false, 'error': '$error', 'stack': '$stack'}),
    );
  } else {
    stderr.writeln('Sarashina2 TensorRT context preparation failed: $error');
    stderr.writeln(stack);
  }
  exitCode = exit;
}

void _printUsage() {
  stdout.writeln('''
Usage:
  dart run tool/prepare_sarashina2_tensorrt_context.dart --root <unifrontend>

Options:
  --model-dir PATH              Sarashina2 model directory.
  --artifact-dir PATH           Default: <root>/artifacts/runtime/sarashina2/tensorrt
  --device-id N                 Default: 0
  --num-threads N               Default: 4
  --cuda-memory-limit-mb N      Default: 16384
  --preload-library PATH
  --cuda-library-dir PATH
  --force                       Regenerate context and engine files.
  --json
''');
}

extension on File {
  Future<void> deleteIfExists() async {
    if (await exists()) {
      await delete();
    }
  }
}

extension on Directory {
  Future<void> deleteIfExists() async {
    if (await exists()) {
      await delete(recursive: true);
    }
  }
}
