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
  if (opts.requireProvider) {
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
  }

  final loadTimer = Stopwatch()..start();
  Sarashina2DartRuntime? runtime;
  try {
    final paths = Sarashina2TtsPaths(modelDir: opts.modelDir);
    runtime = await Sarashina2DartRuntime.load(
      paths: paths,
      provider: opts.provider,
      deviceId: opts.deviceId,
      requireProvider: opts.requireProvider,
      numThreads: opts.numThreads,
      backendOptions: opts.backendOptions,
      loadLlm: opts.needsLlm,
    );
    loadTimer.stop();

    final prompt = opts.promptWav == null
        ? null
        : runtime.extractPrompt(
            decodeWav(File(opts.promptWav!).readAsBytesSync()),
          );
    final runs = <Map<String, Object?>>[];
    final totalTimer = Stopwatch()..start();
    for (var i = 0; i < opts.repeat; i += 1) {
      final run = await _synthesizeOnce(
        runtime: runtime,
        opts: opts,
        prompt: prompt,
        outputWav: _outputPathForIteration(opts.outputWav, i, opts.repeat),
      );
      runs.add(run);
    }
    totalTimer.stop();

    final payload = {
      'ok': true,
      'modelDir': opts.modelDir,
      'mode': opts.mode.name,
      'provider': opts.provider,
      'deviceId': opts.deviceId,
      'numThreads': opts.numThreads,
      'loadLlm': opts.needsLlm,
      'loadedComponents': runtime.loadedComponentNames,
      'selectedProviders': runtime.selectedProviders,
      'loadElapsedMs': loadTimer.elapsedMicroseconds / 1000.0,
      if (prompt != null) ...{
        'promptExtractElapsedMs': prompt.extractElapsedMicroseconds / 1000.0,
        'promptSemanticTokenCount': prompt.promptSemanticTokenCount,
        'promptSpeechFeatFrames': prompt.promptSpeechFeatFrames,
      },
      'totalElapsedMs': totalTimer.elapsedMicroseconds / 1000.0,
      'repeat': opts.repeat,
      'runs': runs,
      'warnings': const [
        'Sarashina2 Dart/FFI runtime does not embed the upstream SilentCipher watermark.',
      ],
    };
    if (opts.json) {
      stdout.writeln(jsonEncode(payload));
    } else {
      for (final run in runs) {
        stdout.writeln(
          'wrote ${run['outputWav']} '
          '(${run['audioBytes']} bytes, '
          '${run['decodedSemanticTokenCount']} decoded semantic tokens)',
        );
      }
    }
  } catch (error, stack) {
    _writeFailure(opts.json, error, stack, exit: 1);
  } finally {
    runtime?.close();
  }
}

Future<Map<String, Object?>> _synthesizeOnce({
  required Sarashina2DartRuntime runtime,
  required _Opts opts,
  required Sarashina2Prompt? prompt,
  required String outputWav,
}) async {
  final result = await runtime.synthesize(
    Sarashina2SynthesisRequest(
      text: opts.text,
      prompt: prompt,
      promptText: opts.promptText,
      semanticTokens: opts.semanticTokens,
      promptTokenIds: opts.promptTokenIds,
      maxGeneratedTokens: opts.maxGeneratedTokens,
      latencyTokens: opts.latencyTokens,
      seed: opts.seed,
      temperature: opts.temperature,
      topP: opts.topP,
      frequencyPenalty: opts.frequencyPenalty,
      includeFloatOutputs: false,
    ),
  );
  final wav = result.audioWavBytes;
  await File(outputWav).parent.create(recursive: true);
  await File(outputWav).writeAsBytes(wav);
  return result.toJson(outputWav: outputWav);
}

final class _Opts {
  const _Opts({
    required this.modelDir,
    required this.text,
    required this.outputWav,
    required this.promptWav,
    required this.promptText,
    required this.semanticTokens,
    required this.promptTokenIds,
    required this.maxGeneratedTokens,
    required this.latencyTokens,
    required this.seed,
    required this.temperature,
    required this.topP,
    required this.frequencyPenalty,
    required this.repeat,
    required this.provider,
    required this.deviceId,
    required this.requireProvider,
    required this.numThreads,
    required this.backendOptions,
    required this.json,
  });

  final String modelDir;
  final String text;
  final String outputWav;
  final String? promptWav;
  final String promptText;
  final List<int> semanticTokens;
  final List<int> promptTokenIds;
  final int maxGeneratedTokens;
  final int latencyTokens;
  final int seed;
  final double temperature;
  final double topP;
  final double frequencyPenalty;
  final int repeat;
  final String provider;
  final int deviceId;
  final bool requireProvider;
  final int numThreads;
  final Map<String, Object?> backendOptions;
  final bool json;

  _Mode get mode {
    if (semanticTokens.isNotEmpty) return _Mode.semanticTokens;
    if (promptTokenIds.isNotEmpty) return _Mode.promptTokenIds;
    return _Mode.text;
  }

  bool get needsLlm => mode != _Mode.semanticTokens;
}

enum _Mode { text, promptTokenIds, semanticTokens }

_Opts _parseArgs(List<String> args) {
  String? modelDir;
  String? root;
  var text = '';
  String? outputWav;
  String? promptWav;
  var promptText = '';
  var semanticTokenText = '';
  var semanticTokens = const <int>[];
  var promptTokenIds = const <int>[];
  var maxGeneratedTokens = 2048;
  var latencyTokens = 1;
  var seed = 0;
  var temperature = sarashina2DefaultTemperature;
  var topP = sarashina2DefaultTopP;
  var frequencyPenalty = sarashina2DefaultFrequencyPenalty;
  var repeat = 1;
  var provider = 'cuda';
  var deviceId = 0;
  var requireProvider = true;
  var numThreads = 4;
  var cudaMemoryLimitMb = 16384;
  var json = false;
  final preloadLibraries = <String>[];
  final libraryDirectories = <String>[];
  final backendOverrides = <String, Object?>{};

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
      case '--model-dir':
        modelDir = next();
      case '--root':
        root = next();
      case '--text':
        text = next();
      case '--output-wav':
        outputWav = next();
      case '--prompt-wav':
        promptWav = next();
      case '--prompt-text':
        promptText = next();
      case '--semantic-token-text':
        semanticTokenText = next();
      case '--semantic-token-file':
        semanticTokenText = File(next()).readAsStringSync();
      case '--semantic-token':
        semanticTokens = [...semanticTokens, int.parse(next())];
      case '--semantic-tokens':
        semanticTokens = [...semanticTokens, ..._parseIntList(next())];
      case '--semantic-tokens-file':
        semanticTokens = [
          ...semanticTokens,
          ..._parseIntList(File(next()).readAsStringSync()),
        ];
      case '--prompt-token-id':
        promptTokenIds = [...promptTokenIds, int.parse(next())];
      case '--prompt-token-ids':
        promptTokenIds = [...promptTokenIds, ..._parseIntList(next())];
      case '--prompt-token-ids-file':
        promptTokenIds = [
          ...promptTokenIds,
          ..._parseIntList(File(next()).readAsStringSync()),
        ];
      case '--max-generated-tokens':
        maxGeneratedTokens = int.parse(next());
      case '--latency-tokens':
        latencyTokens = int.parse(next());
      case '--seed':
        seed = int.parse(next());
      case '--temperature':
        temperature = double.parse(next());
      case '--top-p':
        topP = double.parse(next());
      case '--frequency-penalty':
        frequencyPenalty = double.parse(next());
      case '--repeat':
        repeat = int.parse(next());
      case '--provider':
        provider = next();
      case '--device-id':
        deviceId = int.parse(next());
      case '--allow-cpu-fallback':
        requireProvider = false;
      case '--num-threads':
        numThreads = int.parse(next());
      case '--cuda-memory-limit-mb':
        cudaMemoryLimitMb = int.parse(next());
      case '--preload-library':
        preloadLibraries.add(next());
      case '--cuda-library-dir':
      case '--native-library-dir':
        libraryDirectories.add(next());
      case '--backend-bool':
        final entry = _parseBackendEntry(next());
        backendOverrides[entry.key] = _parseBool(entry.value);
      case '--backend-int':
        final entry = _parseBackendEntry(next());
        backendOverrides[entry.key] = int.parse(entry.value);
      case '--backend-string':
        final entry = _parseBackendEntry(next());
        backendOverrides[entry.key] = entry.value;
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
  final output = outputWav;
  if (output == null || output.isEmpty) {
    throw ArgumentError('--output-wav is required');
  }
  if (repeat < 1) {
    throw ArgumentError('--repeat must be positive');
  }
  if (maxGeneratedTokens < 1) {
    throw ArgumentError('--max-generated-tokens must be positive');
  }
  if (latencyTokens < 0) {
    throw ArgumentError('--latency-tokens must be non-negative');
  }
  if (!temperature.isFinite || temperature < 0) {
    throw ArgumentError('--temperature must be non-negative');
  }
  if (!topP.isFinite || topP <= 0) {
    throw ArgumentError('--top-p must be positive');
  }
  if (!frequencyPenalty.isFinite) {
    throw ArgumentError('--frequency-penalty must be finite');
  }
  final parsedSemanticText = semanticTokenText.trim().isEmpty
      ? const <int>[]
      : parseSarashina2SemanticTokens(
          semanticTokenText,
        ).toList(growable: false);
  if (semanticTokenText.trim().isNotEmpty && parsedSemanticText.isEmpty) {
    throw ArgumentError(
      '--semantic-token-text did not contain Sarashina2 semantic tokens',
    );
  }
  final allSemanticTokens = [...semanticTokens, ...parsedSemanticText];
  validateSarashina2SemanticTokens(allSemanticTokens);
  if (allSemanticTokens.isNotEmpty && promptTokenIds.isNotEmpty) {
    throw ArgumentError(
      'Pass either semantic tokens or prompt token ids, not both.',
    );
  }
  if (allSemanticTokens.isEmpty &&
      promptTokenIds.isEmpty &&
      text.trim().isEmpty) {
    throw ArgumentError('--text must not be empty for raw text synthesis');
  }
  if (promptText.trim().isNotEmpty && promptWav == null) {
    throw ArgumentError('--prompt-text requires --prompt-wav');
  }
  final libraries = discoverDefaultOnnxRuntimePreloadLibraries(
    explicitLibraries: preloadLibraries,
    libraryDirectories: libraryDirectories,
    libraryNames: onnxRuntimePreloadLibraryNamesForProvider(provider),
    runtimeEnvSearchRoots: _runtimeEnvSearchRoots(
      modelDir: resolvedModelDir,
      root: root,
    ),
  );
  final backendOptions = <String, Object?>{
    if (cudaMemoryLimitMb > 0) 'cudaMemoryLimitMb': cudaMemoryLimitMb,
    if (libraries.isNotEmpty)
      'preloadLibraries': encodeOnnxRuntimePreloadLibraries(libraries),
    ...backendOverrides,
  };
  return _Opts(
    modelDir: resolvedModelDir,
    text: text,
    outputWav: output,
    promptWav: promptWav,
    promptText: promptText.trim(),
    semanticTokens: List<int>.unmodifiable(allSemanticTokens),
    promptTokenIds: List<int>.unmodifiable(promptTokenIds),
    maxGeneratedTokens: maxGeneratedTokens,
    latencyTokens: latencyTokens,
    seed: seed,
    temperature: temperature,
    topP: topP,
    frequencyPenalty: frequencyPenalty,
    repeat: repeat,
    provider: provider,
    deviceId: deviceId,
    requireProvider: requireProvider,
    numThreads: numThreads,
    backendOptions: backendOptions,
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

List<String> _runtimeEnvSearchRoots({
  required String modelDir,
  required String? root,
}) {
  final roots = <String>{};
  if (root != null && root.isNotEmpty) {
    roots.add(root);
  }
  void addRootBefore(String marker) {
    final index = modelDir.indexOf(marker);
    if (index > 0) {
      roots.add(modelDir.substring(0, index));
    }
  }

  addRootBefore('/src/ttsbackends/');
  addRootBefore('/artifacts/');
  final modelsIndex = modelDir.indexOf('/models/');
  if (modelsIndex > 0) {
    roots.add(modelDir.substring(0, modelsIndex));
  }
  roots.add(modelDir);
  return roots.toList(growable: false);
}

List<int> _parseIntList(String value) {
  if (value.trim().isEmpty) {
    return const [];
  }
  return value
      .split(RegExp(r'[\s,]+'))
      .where((part) => part.isNotEmpty)
      .map(int.parse)
      .toList(growable: false);
}

MapEntry<String, String> _parseBackendEntry(String value) {
  final index = value.indexOf('=');
  if (index <= 0) {
    throw ArgumentError('Backend option must use key=value: $value');
  }
  final key = value.substring(0, index).trim();
  final optionValue = value.substring(index + 1).trim();
  if (key.isEmpty || optionValue.isEmpty) {
    throw ArgumentError('Backend option must use key=value: $value');
  }
  return MapEntry(key, optionValue);
}

bool _parseBool(String value) {
  final normalized = value.trim().toLowerCase();
  if (normalized == '1' || normalized == 'true' || normalized == 'yes') {
    return true;
  }
  if (normalized == '0' || normalized == 'false' || normalized == 'no') {
    return false;
  }
  throw ArgumentError('Expected boolean value, got: $value');
}

String? _runtimeDependencyError(_Opts opts) {
  final audit = RuntimeDependencyAudit.inspect(
    root: opts.modelDir,
    provider: opts.provider,
    extraSearchDirs: _preloadParentDirs(
      opts.backendOptions['preloadLibraries'],
    ),
  );
  return audit.skipReason;
}

List<String> _preloadParentDirs(Object? value) {
  if (value is! String || value.isEmpty) {
    return const [];
  }
  return [
    for (final path in RuntimeDependencyAudit.splitPathEnv(value))
      File(path).absolute.parent.path,
  ];
}

String _outputPathForIteration(String path, int iteration, int repeat) {
  if (repeat == 1) {
    return path;
  }
  final index = (iteration + 1).toString().padLeft(3, '0');
  if (path.contains('{i}')) {
    return path.replaceAll('{i}', index);
  }
  final slash = path.lastIndexOf(Platform.pathSeparator);
  final dot = path.lastIndexOf('.');
  if (dot > slash) {
    return '${path.substring(0, dot)}_$index${path.substring(dot)}';
  }
  return '${path}_$index';
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
    stderr.writeln('Sarashina2 synthesis failed: $error');
    stderr.writeln(stack);
  }
  exitCode = exit;
}

void _printUsage() {
  stdout.writeln('''
Usage:
  dart run bin/sarashina2_synthesize.dart --model-dir <sarashina2.2-tts> --text "こんにちは" --output-wav out.wav

Direct modes:
  Raw text:
    --text TEXT
  Pre-tokenized prompt ids:
    --prompt-token-ids IDS [--text TEXT]
  Decoder-only semantic tokens:
    --semantic-token-text "<|semantic_1|>..." | --semantic-tokens "1,2,3"

Options:
  --prompt-wav PATH --prompt-text TEXT
  --max-generated-tokens N       Default: 2048
  --latency-tokens N             Default: 1
  --temperature N                Default: 0.9
  --top-p N                      Default: 0.95
  --frequency-penalty N          Default: 1.0
  --seed N                       Default: 0
  --repeat N                     Reuses the loaded runtime in this process.
  --provider cuda|tensorrt|cpu   Default: cuda
  --allow-cpu-fallback
  --device-id N --num-threads N
  --backend-bool KEY=VALUE
  --backend-int KEY=VALUE
  --backend-string KEY=VALUE
  --json
''');
}
