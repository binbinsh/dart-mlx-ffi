import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';

Future<void> main(List<String> args) async {
  final parsed = _Args(args);
  final host = parsed.option('host') ?? '127.0.0.1';
  final port = int.tryParse(parsed.option('port') ?? '') ?? 8020;
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
  };
  final defaultPaths = DartUniFrontendTtsPaths.fromUniFrontendRoot(root);
  final sarashina2ModelDir = parsed.option('sarashina2-model-dir');
  final neuttsAirProviderDir = parsed.option('neutts-air-provider-dir');
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
  final includeCosyVoice2 =
      parsed.flag('enable-cosyvoice2') ||
      _envFlag('DART_TTS_ENABLE_COSYVOICE2');
  final cosyVoice2Only =
      parsed.flag('cosyvoice2-only') || _envFlag('DART_TTS_COSYVOICE2_ONLY');
  final includeSarashina2 =
      parsed.flag('enable-sarashina2') ||
      _envFlag('DART_TTS_ENABLE_SARASHINA2');
  final sarashina2Only =
      parsed.flag('sarashina2-only') || _envFlag('DART_TTS_SARASHINA2_ONLY');
  final includeNeuttsAir =
      parsed.flag('enable-neutts-air') ||
      _envFlag('DART_TTS_ENABLE_NEUTTS_AIR');
  final neuttsAirOnly =
      parsed.flag('neutts-air-only') || _envFlag('DART_TTS_NEUTTS_AIR_ONLY');
  final loadSarashina2Llm =
      parsed.flag('enable-sarashina2-llm') ||
      _envFlag('DART_TTS_ENABLE_SARASHINA2_LLM');
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
    includeKokoro: !cosyVoice2Only && !neuttsAirOnly && !sarashina2Only,
    includeCosyVoice2: includeCosyVoice2 || cosyVoice2Only,
    includeNeuttsAir: includeNeuttsAir || neuttsAirOnly,
    includeSarashina2: includeSarashina2 || sarashina2Only,
    loadSarashina2Llm: loadSarashina2Llm,
    loadCosyVoice2StreamingHift: !parsed.flag('no-cosyvoice2-streaming-hift'),
  );

  HttpServer? server;
  try {
    server = await HttpServer.bind(host, port);
    final health = registry.runtimeHealth();
    final backends = health['backends'] as Map<String, Object?>;
    final kokoroHealth = backends['kokoro'] as Map<String, Object?>?;
    stderr.writeln(
      'dart_tts_server listening on http://$host:$port '
      'providers=${registry.providerNames.join(',')} '
      'structured=${kokoroHealth?['structuredOnnxProvider']} '
      'kokoro=${kokoroHealth?['kokoroOnnxProvider']}',
    );
    await for (final req in server) {
      await _handleRequest(req, registry);
    }
  } finally {
    await server?.close(force: true);
    registry.close();
  }
}

Future<void> _handleRequest(
  HttpRequest req,
  DartTtsBackendRegistry registry,
) async {
  try {
    final path = req.uri.path;
    if (req.method == 'GET' && path == '/health') {
      final health = registry.runtimeHealth();
      final backends = health['backends'] as Map<String, Object?>;
      final kokoro = backends['kokoro'] as Map<String, Object?>?;
      await _json(req, {
        ...health,
        if (kokoro != null) ...{
          'structuredOnnxProvider': kokoro['structuredOnnxProvider'],
          'kokoroOnnxProvider': kokoro['kokoroOnnxProvider'],
          'phonemizerBackend': kokoro['phonemizerBackend'],
        },
      });
      return;
    }
    if (req.method == 'GET' && path == '/providers') {
      await _json(req, registry.providerCards());
      return;
    }
    if (req.method == 'GET' && path == '/references') {
      await _json(req, {'references': <Object>[], 'default': null});
      return;
    }
    if (req.method == 'GET' && path == '/providers/kokoro/voices') {
      final kokoro = registry.byProvider('kokoro');
      await _json(req, {
        'voices': kokoro?.voiceNames ?? const <String>[],
        'languages': ['zh', 'en'],
        'voiceLanguages': {
          for (final voice in kokoro?.voiceNames ?? const <String>[])
            voice: voice.startsWith('zf_') || voice.startsWith('zm_')
                ? 'zh'
                : 'en',
        },
        'languageLabels': {'zh': 'Chinese', 'en': 'English'},
      });
      return;
    }
    if (req.method == 'POST' && path == '/synthesize') {
      final body = await utf8.decoder.bind(req).join();
      final decoded = jsonDecode(body);
      if (decoded is! Map) {
        throw FormatException('request body must be a JSON object');
      }
      final payload = decoded.map((k, v) => MapEntry(k.toString(), v));
      final provider = (payload['provider'] ?? 'kokoro').toString();
      if (registry.byProvider(provider) == null) {
        final catalog = TtsBackendCatalog.byProvider(provider);
        await _json(req, {
          'error': 'unsupported_provider',
          'detail': catalog?.blockers.join(' '),
          'providers': registry.providerCards(),
        }, statusCode: 400);
        return;
      }
      final phonemes = (payload['phonemes'] ?? '').toString().trim();
      final text = (payload['text'] ?? '').toString().trim();
      final voice = (payload['voice'] ?? 'zf_xiaoni').toString();
      final speed = double.tryParse('${payload['speed'] ?? '1.0'}') ?? 1.0;
      final extra = payload['extra'] is Map
          ? (payload['extra'] as Map).map((k, v) => MapEntry('$k', v))
          : const <String, Object?>{};
      final result = await registry.synthesize(
        DartTtsSynthesisRequest(
          provider: provider,
          text: text,
          phonemes: phonemes,
          voice: voice,
          speed: speed,
          promptAudioBytes: decodeAudioDataUrl(payload['promptAudioBase64']),
          promptText: (payload['promptText'] ?? extra['promptText'] ?? '')
              .toString(),
          maxGeneratedTokens: _intValue(
            payload['maxGeneratedTokens'] ?? extra['maxGeneratedTokens'],
          ),
          rasSeed: _intValue(payload['rasSeed'] ?? extra['rasSeed']) ?? 0,
          useStreamingHift: _boolValue(
            payload['useStreamingHift'] ?? extra['useStreamingHift'],
          ),
          semanticTokenText:
              (payload['semanticTokenText'] ?? extra['semanticTokenText'] ?? '')
                  .toString(),
          semanticTokens: _intListValue(
            payload['semanticTokens'] ?? extra['semanticTokens'],
          ),
          codecTokenText:
              (payload['codecTokenText'] ?? extra['codecTokenText'] ?? '')
                  .toString(),
          codecTokens: _intListValue(
            payload['codecTokens'] ?? extra['codecTokens'],
          ),
          referencePhones:
              (payload['referencePhones'] ?? extra['referencePhones'] ?? '')
                  .toString(),
          inputPhones: (payload['inputPhones'] ?? extra['inputPhones'] ?? '')
              .toString(),
          referenceCodes: _intListValue(
            payload['referenceCodes'] ?? extra['referenceCodes'],
          ),
          promptTokenIds: _intListValue(
            payload['promptTokenIds'] ?? extra['promptTokenIds'],
          ),
          latencyTokens:
              _intValue(payload['latencyTokens'] ?? extra['latencyTokens']) ??
              1,
          temperature:
              _doubleValue(payload['temperature'] ?? extra['temperature']) ??
              0.9,
          topP: _doubleValue(payload['topP'] ?? extra['topP']) ?? 0.95,
          frequencyPenalty:
              _doubleValue(
                payload['frequencyPenalty'] ?? extra['frequencyPenalty'],
              ) ??
              1.0,
        ),
      );
      await _json(req, {
        'text': text,
        'frontendText': result.frontendText,
        'frontendSsml': result.frontendSsml,
        'frontendElapsedMs': result.frontendElapsedMicroseconds / 1000.0,
        'frontendProvider': result.frontendProvider,
        'phonemes': result.phonemes,
        'phonemeTokenCount': result.phonemeTokenCount,
        'phonemeChunkCount': result.phonemeChunkCount,
        'provider': result.provider,
        'runtime': result.runtime,
        'python': result.python,
        'phonemizerBackend': result.phonemizerBackend,
        'requestedVoice': result.requestedVoice,
        'voice': result.resolvedVoice,
        'audioFormat': result.audioFormat,
        'audioBase64': base64Encode(result.audioBytes),
        'warnings': result.warnings,
        if (result.metadata.isNotEmpty) 'metadata': result.metadata,
        'ttsElapsedMs': result.ttsElapsedMicroseconds / 1000.0,
      });
      return;
    }
    await _json(req, {'error': 'not_found'}, statusCode: 404);
  } on FormatException catch (error, stack) {
    await _json(req, {
      'error': 'bad_request',
      'detail': '$error',
      'stack': '$stack',
    }, statusCode: 400);
  } on ArgumentError catch (error, stack) {
    await _json(req, {
      'error': 'bad_request',
      'detail': '$error',
      'stack': '$stack',
    }, statusCode: 400);
  } catch (error, stack) {
    await _json(req, {'error': '$error', 'stack': '$stack'}, statusCode: 500);
  }
}

Future<void> _json(
  HttpRequest req,
  Object payload, {
  int statusCode = 200,
}) async {
  req.response.statusCode = statusCode;
  req.response.headers.contentType = ContentType.json;
  req.response.headers.set('Access-Control-Allow-Origin', '*');
  req.response.write(jsonEncode(payload));
  await req.response.close();
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

bool _envFlag(String name) {
  final value = Platform.environment[name]?.toLowerCase();
  return value == '1' || value == 'true' || value == 'yes';
}

int? _intValue(Object? value) {
  if (value == null) {
    return null;
  }
  if (value is num) {
    return value.toInt();
  }
  return int.tryParse('$value');
}

double? _doubleValue(Object? value) {
  if (value == null) {
    return null;
  }
  if (value is num) {
    return value.toDouble();
  }
  return double.tryParse('$value');
}

List<int> _intListValue(Object? value) {
  if (value == null) {
    return const [];
  }
  if (value is Iterable) {
    final parsed = <int>[];
    for (final item in value) {
      parsed.add(_strictIntValue(item));
    }
    return parsed;
  }
  final text = '$value'.trim();
  if (text.isEmpty) {
    return const [];
  }
  return [
    for (final part in text.split(RegExp(r'[\s,]+')))
      if (part.isNotEmpty) int.parse(part),
  ];
}

int _strictIntValue(Object? value) {
  if (value is int) {
    return value;
  }
  if (value is num) {
    if (value.isFinite && value == value.truncateToDouble()) {
      return value.toInt();
    }
    throw FormatException('expected integer token id, got $value');
  }
  final text = '$value'.trim();
  if (text.isEmpty) {
    throw const FormatException('expected integer token id, got empty value');
  }
  return int.parse(text);
}

bool _boolValue(Object? value) {
  if (value == null) {
    return false;
  }
  if (value is bool) {
    return value;
  }
  final text = '$value'.toLowerCase();
  return text == '1' || text == 'true' || text == 'yes';
}
