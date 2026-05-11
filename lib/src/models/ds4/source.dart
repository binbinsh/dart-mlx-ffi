library;

import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:path/path.dart' as p;

const String ds4BundledSourceCommit =
    '8e7575be0ef44bd97c5ebaccf49ef85e05048b7b';

Future<String?> resolveDs4SourceDir({String explicitSourceDir = ''}) async {
  final explicit = explicitSourceDir.trim();
  if (explicit.isNotEmpty) {
    final dir = Directory(explicit);
    if (!await dir.exists()) {
      throw StateError('ds4 source directory does not exist: ${dir.path}');
    }
    await _validateDs4Source(dir.path);
    return p.normalize(dir.path);
  }

  for (final candidate in await ds4BundledSourceCandidates()) {
    if (await _looksLikeDs4Source(candidate)) {
      return p.normalize(candidate);
    }
  }
  return null;
}

Future<List<String>> ds4BundledSourceCandidates() async {
  final candidates = <String>[];
  final packageLib = await Isolate.resolvePackageUri(
    Uri.parse('package:dart_inference/models.dart'),
  );
  if (packageLib != null && packageLib.isScheme('file')) {
    final libFile = packageLib.toFilePath();
    candidates.add(
      p.normalize(p.join(p.dirname(libFile), '..', 'third_party', 'ds4')),
    );
  }

  var current = Directory.current.absolute.path;
  for (var i = 0; i < 5; i += 1) {
    candidates.add(p.join(current, 'third_party', 'ds4'));
    final parent = p.dirname(current);
    if (parent == current) {
      break;
    }
    current = parent;
  }

  final seen = <String>{};
  return <String>[
    for (final candidate in candidates)
      if (seen.add(p.normalize(candidate))) p.normalize(candidate),
  ];
}

Future<String> buildDs4DynamicLibrary(String sourceDir) async {
  final root = Directory(sourceDir.trim());
  if (!await root.exists()) {
    throw StateError('ds4 source directory does not exist: ${root.path}');
  }
  await _validateDs4Source(root.path);
  final buildDir = Directory(
    p.join(_ds4BuildCacheRoot(), _stableHex(await _sourceStamp(root.path))),
  );
  await buildDir.create(recursive: true);
  final output = p.join(
    buildDir.path,
    Platform.isMacOS ? 'libds4.dylib' : 'libds4.so',
  );
  if (await File(output).exists()) {
    return output;
  }

  if (Platform.isMacOS) {
    await _runBuild(root.path, 'cc', <String>[
      '-O3',
      '-ffast-math',
      '-mcpu=native',
      '-Wall',
      '-Wextra',
      '-std=c99',
      '-fPIC',
      '-c',
      'ds4.c',
      '-o',
      p.join(buildDir.path, 'ds4.o'),
    ]);
    await _runBuild(root.path, 'cc', <String>[
      '-O3',
      '-ffast-math',
      '-mcpu=native',
      '-Wall',
      '-Wextra',
      '-fobjc-arc',
      '-fPIC',
      '-c',
      'ds4_metal.m',
      '-o',
      p.join(buildDir.path, 'ds4_metal.o'),
    ]);
    await _runBuild(root.path, 'cc', <String>[
      '-dynamiclib',
      '-o',
      output,
      p.join(buildDir.path, 'ds4.o'),
      p.join(buildDir.path, 'ds4_metal.o'),
      '-lm',
      '-pthread',
      '-framework',
      'Foundation',
      '-framework',
      'Metal',
    ]);
  } else {
    await _runBuild(root.path, 'cc', <String>[
      '-shared',
      '-O3',
      '-ffast-math',
      '-Wall',
      '-Wextra',
      '-std=c99',
      '-DDS4_NO_METAL',
      '-fPIC',
      'ds4.c',
      '-o',
      output,
      '-lm',
      '-pthread',
    ]);
  }
  return output;
}

Map<String, String> ds4MetalSourceEnvironment(String sourceDir) {
  final root = sourceDir.trim();
  return <String, String>{
    'DS4_METAL_FLASH_ATTN_SOURCE': p.join(root, 'metal', 'flash_attn.metal'),
    'DS4_METAL_DENSE_SOURCE': p.join(root, 'metal', 'dense.metal'),
    'DS4_METAL_MOE_SOURCE': p.join(root, 'metal', 'moe.metal'),
    'DS4_METAL_DSV4_HC_SOURCE': p.join(root, 'metal', 'dsv4_hc.metal'),
    'DS4_METAL_UNARY_SOURCE': p.join(root, 'metal', 'unary.metal'),
    'DS4_METAL_DSV4_KV_SOURCE': p.join(root, 'metal', 'dsv4_kv.metal'),
    'DS4_METAL_DSV4_ROPE_SOURCE': p.join(root, 'metal', 'dsv4_rope.metal'),
    'DS4_METAL_DSV4_MISC_SOURCE': p.join(root, 'metal', 'dsv4_misc.metal'),
    'DS4_METAL_ARGSORT_SOURCE': p.join(root, 'metal', 'argsort.metal'),
    'DS4_METAL_CPY_SOURCE': p.join(root, 'metal', 'cpy.metal'),
    'DS4_METAL_CONCAT_SOURCE': p.join(root, 'metal', 'concat.metal'),
    'DS4_METAL_GET_ROWS_SOURCE': p.join(root, 'metal', 'get_rows.metal'),
    'DS4_METAL_SUM_ROWS_SOURCE': p.join(root, 'metal', 'sum_rows.metal'),
    'DS4_METAL_SOFTMAX_SOURCE': p.join(root, 'metal', 'softmax.metal'),
    'DS4_METAL_REPEAT_SOURCE': p.join(root, 'metal', 'repeat.metal'),
    'DS4_METAL_GLU_SOURCE': p.join(root, 'metal', 'glu.metal'),
    'DS4_METAL_NORM_SOURCE': p.join(root, 'metal', 'norm.metal'),
    'DS4_METAL_BIN_SOURCE': p.join(root, 'metal', 'bin.metal'),
    'DS4_METAL_SET_ROWS_SOURCE': p.join(root, 'metal', 'set_rows.metal'),
  };
}

Future<bool> _looksLikeDs4Source(String path) async {
  return await File(p.join(path, 'ds4.c')).exists() &&
      await File(p.join(path, 'ds4.h')).exists() &&
      await File(p.join(path, 'ds4_metal.m')).exists();
}

Future<void> _validateDs4Source(String path) async {
  if (await _looksLikeDs4Source(path)) {
    return;
  }
  throw StateError(
    'ds4 source directory is missing ds4.c, ds4.h, or ds4_metal.m: $path',
  );
}

Future<String> _sourceStamp(String root) async {
  final paths = <String>[
    p.join(root, 'ds4.c'),
    p.join(root, 'ds4.h'),
    p.join(root, 'ds4_metal.h'),
    p.join(root, 'ds4_metal.m'),
  ];
  final metalDir = Directory(p.join(root, 'metal'));
  if (await metalDir.exists()) {
    await for (final entity in metalDir.list(followLinks: false)) {
      if (entity is File && p.extension(entity.path) == '.metal') {
        paths.add(entity.path);
      }
    }
  }
  paths.sort();

  final stamp = <String>[p.normalize(root)];
  for (final path in paths) {
    final file = File(path);
    if (!await file.exists()) {
      continue;
    }
    final stat = await file.stat();
    stamp.add(
      [
        p.relative(path, from: root),
        stat.size,
        stat.modified.millisecondsSinceEpoch,
      ].join(':'),
    );
  }
  return stamp.join('|');
}

String _ds4BuildCacheRoot() {
  final override = Platform.environment['DART_INFERENCE_DS4_BUILD_CACHE'];
  if (override != null && override.trim().isNotEmpty) {
    return override.trim();
  }
  final home = Platform.environment['HOME'];
  if (home != null && home.trim().isNotEmpty) {
    if (Platform.isMacOS) {
      return p.join(home, 'Library', 'Caches', 'dart_inference', 'ds4');
    }
    return p.join(home, '.cache', 'dart_inference', 'ds4');
  }
  return p.join(Directory.systemTemp.path, 'dart_inference', 'ds4');
}

String _stableHex(String value) {
  var hash = 0xcbf29ce484222325;
  for (final byte in utf8.encode(value)) {
    hash ^= byte;
    hash = (hash * 0x100000001b3) & 0xffffffffffffffff;
  }
  return hash.toRadixString(16).padLeft(16, '0');
}

Future<void> _runBuild(
  String workingDirectory,
  String executable,
  List<String> args,
) async {
  final result = await Process.run(
    executable,
    args,
    workingDirectory: workingDirectory,
  );
  if (result.exitCode != 0) {
    throw StateError(
      'Failed to build ds4 dynamic library with `$executable ${args.join(' ')}` '
      'in $workingDirectory:\n${result.stdout}\n${result.stderr}',
    );
  }
}
