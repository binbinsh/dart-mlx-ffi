import '../kokoro/kokoro.dart';
import '../unifrontend/unifrontend.dart';
import '../../runtime/onnx.dart';
import 'runtime.dart';

final class DartUniFrontendTtsPaths {
  const DartUniFrontendTtsPaths({
    required this.kokoroModelPath,
    required this.kokoroVoicesPath,
    required this.kokoroConfigPath,
    required this.structuredModelPath,
    required this.structuredExportConfigPath,
    required this.structuredConfigPath,
    required this.structuredCharVocabPath,
    required this.structuredLabelSpacePath,
    required this.structuredEnglishTnLexiconPath,
    required this.structuredTokenizerPath,
  });

  factory DartUniFrontendTtsPaths.fromUniFrontendRoot(String root) {
    final normalized = root.endsWith('/')
        ? root.substring(0, root.length - 1)
        : root;
    return DartUniFrontendTtsPaths(
      kokoroModelPath:
          '$normalized/src/ttsbackends/providers/kokoro/models/kokoro-v1.0.onnx',
      kokoroVoicesPath:
          '$normalized/src/ttsbackends/providers/kokoro/models/voices.npz',
      kokoroConfigPath:
          '$normalized/src/ttsbackends/providers/kokoro/models/config.json',
      structuredModelPath:
          '$normalized/artifacts/onnx/structured-mmbert-focus-v2-step-20000.online-multi.fixed8.512x1024.onnx',
      structuredExportConfigPath:
          '$normalized/artifacts/onnx/structured-mmbert-focus-v2-step-20000.online-multi.fixed8.512x1024.json',
      structuredConfigPath:
          '$normalized/artifacts/releases/structured-mmbert-focus-v2-step-20000/structured_config.json',
      structuredCharVocabPath:
          '$normalized/artifacts/releases/structured-mmbert-focus-v2-step-20000/char_vocab.json',
      structuredLabelSpacePath:
          '$normalized/artifacts/releases/structured-mmbert-focus-v2-step-20000/label_space.json',
      structuredEnglishTnLexiconPath:
          '$normalized/artifacts/releases/structured-mmbert-focus-v2-step-20000/english_tn_lexicon.json',
      structuredTokenizerPath:
          '$normalized/src/ttsbackends/providers/kokoro/models/models--jhu-clsp--mmBERT-base/snapshots/c5955035435e2bf121cde7f3c8863ef52ff35d82/tokenizer.json',
    );
  }

  final String kokoroModelPath;
  final String kokoroVoicesPath;
  final String kokoroConfigPath;
  final String structuredModelPath;
  final String structuredExportConfigPath;
  final String structuredConfigPath;
  final String structuredCharVocabPath;
  final String structuredLabelSpacePath;
  final String structuredEnglishTnLexiconPath;
  final String structuredTokenizerPath;

  DartUniFrontendTtsPaths copyWith({
    String? kokoroModelPath,
    String? kokoroVoicesPath,
    String? kokoroConfigPath,
    String? structuredModelPath,
    String? structuredExportConfigPath,
    String? structuredConfigPath,
    String? structuredCharVocabPath,
    String? structuredLabelSpacePath,
    String? structuredEnglishTnLexiconPath,
    String? structuredTokenizerPath,
  }) {
    return DartUniFrontendTtsPaths(
      kokoroModelPath: kokoroModelPath ?? this.kokoroModelPath,
      kokoroVoicesPath: kokoroVoicesPath ?? this.kokoroVoicesPath,
      kokoroConfigPath: kokoroConfigPath ?? this.kokoroConfigPath,
      structuredModelPath: structuredModelPath ?? this.structuredModelPath,
      structuredExportConfigPath:
          structuredExportConfigPath ?? this.structuredExportConfigPath,
      structuredConfigPath: structuredConfigPath ?? this.structuredConfigPath,
      structuredCharVocabPath:
          structuredCharVocabPath ?? this.structuredCharVocabPath,
      structuredLabelSpacePath:
          structuredLabelSpacePath ?? this.structuredLabelSpacePath,
      structuredEnglishTnLexiconPath:
          structuredEnglishTnLexiconPath ?? this.structuredEnglishTnLexiconPath,
      structuredTokenizerPath:
          structuredTokenizerPath ?? this.structuredTokenizerPath,
    );
  }
}

final class DartTtsRuntimeOptions {
  const DartTtsRuntimeOptions({
    this.provider = 'cuda',
    this.deviceId = 0,
    this.requireProvider = true,
    this.numThreads = 4,
    this.cudaMemoryLimitMb = 16384,
    this.allowEspeakProcessFallback = false,
    this.preloadLibraries = const [],
    this.backendOptions = const {},
  });

  final String provider;
  final int deviceId;
  final bool requireProvider;
  final int numThreads;
  final int cudaMemoryLimitMb;
  final bool allowEspeakProcessFallback;
  final List<String> preloadLibraries;
  final Map<String, Object?> backendOptions;

  DartTtsRuntimeOptions copyWith({
    String? provider,
    int? deviceId,
    bool? requireProvider,
    int? numThreads,
    int? cudaMemoryLimitMb,
    bool? allowEspeakProcessFallback,
    List<String>? preloadLibraries,
    Map<String, Object?>? backendOptions,
  }) {
    return DartTtsRuntimeOptions(
      provider: provider ?? this.provider,
      deviceId: deviceId ?? this.deviceId,
      requireProvider: requireProvider ?? this.requireProvider,
      numThreads: numThreads ?? this.numThreads,
      cudaMemoryLimitMb: cudaMemoryLimitMb ?? this.cudaMemoryLimitMb,
      allowEspeakProcessFallback:
          allowEspeakProcessFallback ?? this.allowEspeakProcessFallback,
      preloadLibraries: preloadLibraries ?? this.preloadLibraries,
      backendOptions: backendOptions ?? this.backendOptions,
    );
  }
}

Future<DartTtsBackendRegistry> loadUniFrontendKokoroTtsRegistry({
  required DartUniFrontendTtsPaths paths,
  DartTtsRuntimeOptions options = const DartTtsRuntimeOptions(),
}) async {
  final preloadLibraries = options.preloadLibraries.isNotEmpty
      ? options.preloadLibraries
      : discoverDefaultOnnxRuntimePreloadLibraries(
          libraryDirectories: _runtimeLibraryDirectories(paths),
          runtimeEnvSearchRoots: _runtimeEnvSearchRoots(paths),
        );
  final backendOptions = {
    if (options.cudaMemoryLimitMb > 0)
      'cudaMemoryLimitMb': options.cudaMemoryLimitMb,
    if (preloadLibraries.isNotEmpty)
      'preloadLibraries': encodeOnnxRuntimePreloadLibraries(preloadLibraries),
    ...options.backendOptions,
  };
  final frontend = await DartStructuredFrontendRuntime.load(
    modelPath: paths.structuredModelPath,
    exportConfigPath: paths.structuredExportConfigPath,
    structuredConfigPath: paths.structuredConfigPath,
    tokenizerJsonPath: paths.structuredTokenizerPath,
    charVocabPath: paths.structuredCharVocabPath,
    labelSpacePath: paths.structuredLabelSpacePath,
    englishTnLexiconPath: paths.structuredEnglishTnLexiconPath,
    provider: options.provider,
    deviceId: options.deviceId,
    requireProvider: options.requireProvider,
    numThreads: options.numThreads,
    backendOptions: backendOptions,
  );
  try {
    final kokoro = await KokoroDartRuntime.load(
      modelPath: paths.kokoroModelPath,
      voicesPath: paths.kokoroVoicesPath,
      configPath: paths.kokoroConfigPath,
      provider: options.provider,
      deviceId: options.deviceId,
      requireProvider: options.requireProvider,
      numThreads: options.numThreads,
      backendOptions: backendOptions,
    );
    final tts = UniFrontendKokoroTtsRuntime(
      frontend: frontend,
      kokoro: kokoro,
      phonemizer: KokoroPhonemizer(
        allowProcessFallback: options.allowEspeakProcessFallback,
      ),
    );
    return DartTtsBackendRegistry(backends: [KokoroTtsBackend(tts)]);
  } catch (_) {
    frontend.close();
    rethrow;
  }
}

List<String> _runtimeEnvSearchRoots(DartUniFrontendTtsPaths paths) {
  final roots = <String>{};
  void addRootBefore(String path, String marker) {
    final index = path.indexOf(marker);
    if (index > 0) {
      roots.add(path.substring(0, index));
    }
  }

  for (final path in [
    paths.kokoroModelPath,
    paths.structuredModelPath,
    paths.structuredConfigPath,
  ]) {
    addRootBefore(path, '/src/ttsbackends/');
    addRootBefore(path, '/artifacts/');
  }
  return roots.toList(growable: false);
}

List<String> _runtimeLibraryDirectories(DartUniFrontendTtsPaths paths) {
  return [
    for (final root in _runtimeEnvSearchRoots(paths)) ...[
      '$root/artifacts/runtime/onnxruntime/lib',
      '$root/artifacts/runtime/cuda/lib',
      '$root/artifacts/runtime/tensorrt/lib',
    ],
  ];
}
