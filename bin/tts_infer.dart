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
  final trtCacheDir = parsed.option('trt-cache-dir');
  if (requireProvider) {
    final dependencyError = _runtimeDependencyError(root, provider, parsed);
    if (dependencyError != null) {
      stderr.writeln(dependencyError);
      exitCode = 78;
      return;
    }
  }
  final preloadLibraries = _preloadLibrariesFromArgs(
    parsed,
    root: root,
    provider: provider,
  );
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
    ..._backendOptionsFromArgs(parsed),
  };
  final ttsProvider = parsed.option('tts-provider') ?? 'kokoro';
  final defaultPaths = DartUniFrontendTtsPaths.fromUniFrontendRoot(root);
  final sarashina2ModelDir = parsed.option('sarashina2-model-dir');
  final neuttsAirProviderDir = parsed.option('neutts-air-provider-dir');
  final needsSarashina2TextGeneration =
      ttsProvider == sarashina2Provider && !_hasSemanticTokenInputs(parsed);
  final loadSarashina2Llm =
      parsed.flag('enable-sarashina2-llm') ||
      _hasPromptTokenIds(parsed) ||
      needsSarashina2TextGeneration;
  final paths = defaultPaths.copyWith(
    kokoroModelPath: parsed.option('kokoro-model'),
    kokoroVoicesPath: parsed.option('kokoro-voices'),
    kokoroConfigPath: parsed.option('kokoro-config'),
    sarashina2Paths: sarashina2ModelDir == null || sarashina2ModelDir.isEmpty
        ? null
        : Sarashina2TtsPaths(modelDir: sarashina2ModelDir),
    neuttsAirPaths: neuttsAirProviderDir == null || neuttsAirProviderDir.isEmpty
        ? null
        : NeuttsAirPaths(providerDir: neuttsAirProviderDir),
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
  final registry = await loadUniFrontendTtsRegistry(
    paths: paths,
    options: DartTtsRuntimeOptions(
      provider: provider,
      deviceId: deviceId,
      requireProvider: requireProvider,
      numThreads: numThreads,
      cudaMemoryLimitMb: cudaMemoryLimitMb,
      preloadLibraries: preloadLibraries,
      backendOptions: backendOptions,
    ),
    includeKokoro:
        (ttsProvider != 'cosyvoice2' &&
            ttsProvider != neuttsAirProvider &&
            ttsProvider != sarashina2Provider) ||
        parsed.flag('include-kokoro'),
    includeCosyVoice2:
        ttsProvider == 'cosyvoice2' || parsed.flag('enable-cosyvoice2'),
    includeNeuttsAir:
        ttsProvider == neuttsAirProvider || parsed.flag('enable-neutts-air'),
    includeSarashina2:
        ttsProvider == sarashina2Provider || parsed.flag('enable-sarashina2'),
    loadSarashina2Llm: loadSarashina2Llm,
    loadCosyVoice2StreamingHift: !parsed.flag('no-cosyvoice2-streaming-hift'),
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
        if (result.metadata.isNotEmpty) 'metadata': result.metadata,
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
  final promptWav = parsed.option('prompt-wav');
  return DartTtsSynthesisRequest(
    provider: provider,
    text: text,
    phonemes: phonemes,
    voice: voice,
    speed: speed,
    promptAudioBytes: promptWav == null || promptWav.isEmpty
        ? null
        : File(promptWav).readAsBytesSync(),
    promptText: parsed.option('prompt-text') ?? '',
    maxGeneratedTokens: int.tryParse(
      parsed.option('max-generated-tokens') ?? '',
    ),
    rasSeed: int.tryParse(parsed.option('ras-seed') ?? '0') ?? 0,
    useStreamingHift: parsed.flag('use-streaming-hift'),
    semanticTokenText: _semanticTokenTextFromArgs(parsed),
    semanticTokens: _semanticTokensFromArgs(parsed),
    codecTokenText: _codecTokenTextFromArgs(parsed),
    codecTokens: _codecTokensFromArgs(parsed),
    referencePhones: parsed.option('reference-phones') ?? '',
    inputPhones: parsed.option('input-phones') ?? '',
    referenceCodes: _referenceCodesFromArgs(parsed),
    promptTokenIds: _promptTokenIdsFromArgs(parsed),
    latencyTokens: int.tryParse(parsed.option('latency-tokens') ?? '1') ?? 1,
    temperature: double.tryParse(parsed.option('temperature') ?? '0.9') ?? 0.9,
    topP: double.tryParse(parsed.option('top-p') ?? '0.95') ?? 0.95,
    frequencyPenalty:
        double.tryParse(parsed.option('frequency-penalty') ?? '1.0') ?? 1.0,
  );
}

String _codecTokenTextFromArgs(_Args parsed) {
  final inline = parsed.option('codec-token-text');
  if (inline != null && inline.isNotEmpty) {
    return inline;
  }
  final path = parsed.option('codec-token-file');
  if (path == null || path.isEmpty) {
    return '';
  }
  return File(path).readAsStringSync();
}

List<int> _codecTokensFromArgs(_Args parsed) {
  final values = <String>[...parsed.values('codec-token')];
  final inline = parsed.option('codec-tokens');
  if (inline != null && inline.isNotEmpty) {
    values.add(inline);
  }
  final path = parsed.option('codec-tokens-file');
  if (path != null && path.isNotEmpty) {
    values.add(File(path).readAsStringSync());
  }
  return _parseIntList(values.join(','));
}

List<int> _referenceCodesFromArgs(_Args parsed) {
  final values = <String>[...parsed.values('reference-code')];
  final inline = parsed.option('reference-codes');
  if (inline != null && inline.isNotEmpty) {
    values.add(inline);
  }
  final path = parsed.option('reference-codes-file');
  if (path != null && path.isNotEmpty) {
    values.add(File(path).readAsStringSync());
  }
  return _parseIntList(values.join(','));
}

String _semanticTokenTextFromArgs(_Args parsed) {
  final inline = parsed.option('semantic-token-text');
  if (inline != null && inline.isNotEmpty) {
    return inline;
  }
  final path = parsed.option('semantic-token-file');
  if (path == null || path.isEmpty) {
    return '';
  }
  return File(path).readAsStringSync();
}

List<int> _semanticTokensFromArgs(_Args parsed) {
  final values = <String>[...parsed.values('semantic-token')];
  final inline = parsed.option('semantic-tokens');
  if (inline != null && inline.isNotEmpty) {
    values.add(inline);
  }
  final path = parsed.option('semantic-tokens-file');
  if (path != null && path.isNotEmpty) {
    values.add(File(path).readAsStringSync());
  }
  return _parseIntList(values.join(','));
}

List<int> _promptTokenIdsFromArgs(_Args parsed) {
  final values = <String>[...parsed.values('prompt-token-id')];
  final inline = parsed.option('prompt-token-ids');
  if (inline != null && inline.isNotEmpty) {
    values.add(inline);
  }
  final path = parsed.option('prompt-token-ids-file');
  if (path != null && path.isNotEmpty) {
    values.add(File(path).readAsStringSync());
  }
  return _parseIntList(values.join(','));
}

bool _hasPromptTokenIds(_Args parsed) {
  return parsed.values('prompt-token-id').isNotEmpty ||
      (parsed.option('prompt-token-ids') ?? '').isNotEmpty ||
      (parsed.option('prompt-token-ids-file') ?? '').isNotEmpty;
}

bool _hasSemanticTokenInputs(_Args parsed) {
  return parsed.values('semantic-token').isNotEmpty ||
      (parsed.option('semantic-tokens') ?? '').isNotEmpty ||
      (parsed.option('semantic-tokens-file') ?? '').isNotEmpty ||
      (parsed.option('semantic-token-text') ?? '').isNotEmpty ||
      (parsed.option('semantic-token-file') ?? '').isNotEmpty;
}

List<int> _parseIntList(String value) {
  if (value.trim().isEmpty) {
    return const [];
  }
  return value
      .split(RegExp(r'[\s,]+'))
      .where((part) => part.isNotEmpty)
      .map((part) => int.parse(part))
      .toList(growable: false);
}

Map<String, Object?> _backendOptionsFromArgs(_Args parsed) {
  final options = <String, Object?>{};
  for (final entry in parsed.values('backend-bool')) {
    final parsedEntry = _parseBackendEntry(entry);
    options[parsedEntry.key] = _parseBool(parsedEntry.value);
  }
  for (final entry in parsed.values('backend-int')) {
    final parsedEntry = _parseBackendEntry(entry);
    options[parsedEntry.key] = int.parse(parsedEntry.value);
  }
  for (final entry in parsed.values('backend-string')) {
    final parsedEntry = _parseBackendEntry(entry);
    options[parsedEntry.key] = parsedEntry.value;
  }
  return options;
}

MapEntry<String, String> _parseBackendEntry(String value) {
  final index = value.indexOf('=');
  if (index <= 0) {
    throw ArgumentError('Expected KEY=VALUE, got "$value"');
  }
  return MapEntry(value.substring(0, index), value.substring(index + 1));
}

bool _parseBool(String value) {
  final normalized = value.trim().toLowerCase();
  if (normalized == '1' || normalized == 'true' || normalized == 'yes') {
    return true;
  }
  if (normalized == '0' || normalized == 'false' || normalized == 'no') {
    return false;
  }
  throw ArgumentError('Expected boolean value, got "$value"');
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

List<String> _preloadLibrariesFromArgs(
  _Args parsed, {
  required String root,
  required String provider,
}) {
  return discoverDefaultOnnxRuntimePreloadLibraries(
    explicitLibraries: parsed.values('preload-library'),
    libraryDirectories: [
      ...parsed.values('cuda-library-dir'),
      ...parsed.values('native-library-dir'),
    ],
    libraryNames: onnxRuntimePreloadLibraryNamesForProvider(provider),
    runtimeEnvSearchRoots: [root],
  );
}

String? _runtimeDependencyError(String root, String provider, _Args parsed) {
  final audit = RuntimeDependencyAudit.inspect(
    root: root,
    provider: provider,
    extraSearchDirs: [
      ...parsed.values('cuda-library-dir'),
      ...parsed.values('native-library-dir'),
    ],
  );
  return audit.skipReason;
}
