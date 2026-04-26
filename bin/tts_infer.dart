import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';

Future<void> main(List<String> args) async {
  final parsed = _Args(args);
  final root = _discoverProjectRoot(parsed.option('root'));
  final provider = parsed.option('provider') ?? 'cuda';
  final deviceId = int.tryParse(parsed.option('device-id') ?? '0') ?? 0;
  final requireProvider = !parsed.flag('allow-cpu-fallback');
  final numThreads = int.tryParse(parsed.option('num-threads') ?? '4') ?? 4;
  final cudaMemoryLimitMb =
      int.tryParse(parsed.option('cuda-memory-limit-mb') ?? '') ?? 16384;
  final allowEspeakProcessFallback = parsed.flag(
    'allow-espeak-process-fallback',
  );
  final trtCacheDir = parsed.option('trt-cache-dir');
  final preloadLibraries = _preloadLibrariesFromArgs(parsed, root: root);
  final backendOptions = <String, Object?>{
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
  };
  final paths = DartUniFrontendTtsPaths.fromUniFrontendRoot(root).copyWith(
    kokoroModelPath: parsed.option('kokoro-model'),
    kokoroVoicesPath: parsed.option('kokoro-voices'),
    kokoroConfigPath: parsed.option('kokoro-config'),
    structuredModelPath: parsed.option('structured-model'),
    structuredExportConfigPath: parsed.option('structured-export-config'),
    structuredConfigPath: parsed.option('structured-config'),
    structuredCharVocabPath: parsed.option('structured-char-vocab'),
    structuredLabelSpacePath: parsed.option('structured-label-space'),
    structuredEnglishTnLexiconPath: parsed.option(
      'structured-english-tn-lexicon',
    ),
    structuredTokenizerPath: parsed.option('structured-tokenizer'),
  );
  final registry = await loadUniFrontendKokoroTtsRegistry(
    paths: paths,
    options: DartTtsRuntimeOptions(
      provider: provider,
      deviceId: deviceId,
      requireProvider: requireProvider,
      numThreads: numThreads,
      cudaMemoryLimitMb: cudaMemoryLimitMb,
      allowEspeakProcessFallback: allowEspeakProcessFallback,
      preloadLibraries: preloadLibraries,
      backendOptions: backendOptions,
    ),
  );

  try {
    final request = _requestFromArgs(parsed);
    final result = await registry.synthesize(request);
    final outputPath = parsed.option('output-wav');
    if (outputPath != null && outputPath.isNotEmpty) {
      await File(outputPath).writeAsBytes(result.audioBytes);
    }

    stdout.writeln(
      jsonEncode({
        'runtime': result.runtime,
        'python': result.python,
        'provider': result.provider,
        'frontendProvider': result.frontendProvider,
        'phonemizerBackend': result.phonemizerBackend,
        'requestedVoice': result.requestedVoice,
        'voice': result.resolvedVoice,
        'audioFormat': result.audioFormat,
        'audioBytes': result.audioBytes.length,
        'outputWav': outputPath,
        'phonemeTokenCount': result.phonemeTokenCount,
        'phonemeChunkCount': result.phonemeChunkCount,
        'frontendElapsedMs': result.frontendElapsedMicroseconds / 1000.0,
        'ttsElapsedMs': result.ttsElapsedMicroseconds / 1000.0,
        'warnings': result.warnings,
        'phonemes': result.phonemes,
      }),
    );
  } finally {
    registry.close();
  }
}

DartTtsSynthesisRequest _requestFromArgs(_Args parsed) {
  final provider = parsed.option('tts-provider') ?? 'kokoro';
  final text = parsed.option('text') ?? '';
  final phonemes = parsed.option('phonemes') ?? '';
  final voice = parsed.option('voice') ?? 'zf_xiaoni';
  final speed = double.tryParse(parsed.option('speed') ?? '1.0') ?? 1.0;
  return DartTtsSynthesisRequest(
    provider: provider,
    text: text,
    phonemes: phonemes,
    voice: voice,
    speed: speed,
  );
}

final class _Args {
  _Args(this.args);

  final List<String> args;

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

  bool flag(String name) => args.contains('--$name');

  List<String> values(String name) {
    final values = <String>[];
    final prefix = '--$name=';
    for (var i = 0; i < args.length; i++) {
      final value = args[i];
      if (value.startsWith(prefix)) {
        values.add(value.substring(prefix.length));
      } else if (value == '--$name' && i + 1 < args.length) {
        values.add(args[++i]);
      }
    }
    return values;
  }
}

String _discoverProjectRoot(String? explicitRoot) {
  final envRoot = Platform.environment['UNIFRONTEND_ROOT'];
  for (final value in [explicitRoot, envRoot, Directory.current.path]) {
    if (value == null || value.isEmpty) {
      continue;
    }
    final root = Directory(value).absolute;
    if (_looksLikeUniFrontendRoot(root)) {
      return root.path;
    }
    for (var current = root; current.parent.path != current.path;) {
      if (_looksLikeUniFrontendRoot(current)) {
        return current.path;
      }
      current = current.parent;
    }
  }
  throw StateError(
    'Could not locate unifrontend root. Pass --root or set UNIFRONTEND_ROOT.',
  );
}

bool _looksLikeUniFrontendRoot(Directory directory) {
  return File('${directory.path}/src/ttsbackends/registry.toml').existsSync() &&
      Directory('${directory.path}/src/unifrontend').existsSync();
}

List<String> _preloadLibrariesFromArgs(_Args parsed, {required String root}) {
  return discoverDefaultOnnxRuntimePreloadLibraries(
    explicitLibraries: parsed.values('preload-library'),
    libraryDirectories: [
      ...parsed.values('cuda-library-dir'),
      ...parsed.values('native-library-dir'),
    ],
    runtimeEnvSearchRoots: [root],
  );
}
