import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';

Future<void> main(List<String> args) async {
  final parsed = _Args(args);
  final root = _discoverProjectRoot(parsed.option('root'));
  final texts = parsed.values('text');
  if (texts.isEmpty) {
    stderr.writeln('Missing --text. Pass one or more --text values.');
    exitCode = 64;
    return;
  }

  final provider = parsed.option('provider') ?? 'cuda';
  final deviceId = int.tryParse(parsed.option('device-id') ?? '0') ?? 0;
  final requireProvider = !parsed.flag('allow-cpu-fallback');
  final numThreads = int.tryParse(parsed.option('num-threads') ?? '4') ?? 4;
  final trtCacheDir = parsed.option('trt-cache-dir');
  if (requireProvider) {
    final dependencyError = _runtimeDependencyError(root, provider, parsed);
    if (dependencyError != null) {
      stderr.writeln(dependencyError);
      exitCode = 78;
      return;
    }
  }
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
    if (preloadLibraries.isNotEmpty)
      'preloadLibraries': encodeOnnxRuntimePreloadLibraries(preloadLibraries),
  };

  final runtime = await DartStructuredFrontendRuntime.load(
    modelPath:
        parsed.option('model') ??
        '$root/artifacts/onnx/structured-mmbert-focus-v2-step-20000.online-multi.fixed8.512x1024.onnx',
    exportConfigPath:
        parsed.option('export-config') ??
        '$root/artifacts/onnx/structured-mmbert-focus-v2-step-20000.online-multi.fixed8.512x1024.json',
    structuredConfigPath:
        parsed.option('structured-config') ??
        '$root/artifacts/releases/structured-mmbert-focus-v2-step-20000/structured_config.json',
    tokenizerJsonPath:
        parsed.option('tokenizer') ??
        '$root/src/ttsbackends/providers/kokoro/models/models--jhu-clsp--mmBERT-base/snapshots/c5955035435e2bf121cde7f3c8863ef52ff35d82/tokenizer.json',
    charVocabPath:
        parsed.option('char-vocab') ??
        '$root/artifacts/releases/structured-mmbert-focus-v2-step-20000/char_vocab.json',
    labelSpacePath:
        parsed.option('label-space') ??
        '$root/artifacts/releases/structured-mmbert-focus-v2-step-20000/label_space.json',
    englishTnLexiconPath:
        parsed.option('english-tn-lexicon') ??
        '$root/artifacts/releases/structured-mmbert-focus-v2-step-20000/english_tn_lexicon.json',
    provider: provider,
    deviceId: deviceId,
    requireProvider: requireProvider,
    numThreads: numThreads,
    backendOptions: backendOptions,
  );

  try {
    final results = [
      for (final result in runtime.processBatch(texts)) result.toJson(),
    ];
    final payload = {
      'runtime': 'dart',
      'python': false,
      'provider': runtime.selectedProvider,
      'results': results,
    };
    final encoder = parsed.flag('pretty')
        ? const JsonEncoder.withIndent('  ')
        : const JsonEncoder();
    stdout.writeln(encoder.convert(payload));
  } finally {
    runtime.close();
  }
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

  bool flag(String name) => args.contains('--$name');
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
