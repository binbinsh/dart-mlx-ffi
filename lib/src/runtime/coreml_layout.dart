/// Core ML artifact layout helpers.
library;

import 'dart:io';

/// Discovered CoreML-LLM-style model bundle layout.
final class CoreMlBundleLayout {
  const CoreMlBundleLayout({
    required this.rootPath,
    this.pipelineSpecPath,
    this.modelConfigPath,
    this.monolithicModelPath,
    this.decodeChunks = const <String>[],
    this.prefillChunks = const <String>[],
    this.sidecars = const <String>[],
  });

  final String rootPath;
  final String? pipelineSpecPath;
  final String? modelConfigPath;
  final String? monolithicModelPath;
  final List<String> decodeChunks;
  final List<String> prefillChunks;
  final List<String> sidecars;

  /// True for CoreML-LLM-style chunked decode/prefill bundles.
  bool get isChunked => decodeChunks.isNotEmpty;

  /// True for a single `.mlmodelc`/`.mlpackage` bundle.
  bool get isMonolithic => monolithicModelPath != null;

  /// True for a `dart_inference.coreml_pipeline.v1` JSON spec.
  bool get isPipeline => pipelineSpecPath != null;

  /// Whether the bundle has enough structure for a Core ML runtime to load.
  bool get isLoadable => isChunked || isMonolithic || isPipeline;

  /// Discover a Core ML bundle rooted at [rootPath].
  factory CoreMlBundleLayout.discover(String rootPath) {
    final file = File(rootPath);
    if (file.existsSync() && rootPath.endsWith('.json')) {
      return CoreMlBundleLayout(rootPath: rootPath, pipelineSpecPath: rootPath);
    }
    final root = Directory(rootPath);
    if (!root.existsSync()) {
      return CoreMlBundleLayout(rootPath: rootPath);
    }

    final entries = root.listSync().toList();
    final config = File('${root.path}/model_config.json');
    final decode = <_IndexedPath>[];
    final prefill = <_IndexedPath>[];
    final sidecars = <String>[];
    String? monolithic;

    for (final entry in entries) {
      final name = _basename(entry.path);
      if (_isCoreMlBundle(name)) {
        if (name == 'model.mlmodelc' || name == 'model.mlpackage') {
          monolithic = entry.path;
          continue;
        }
        final decodeIndex = _chunkIndex(name, 'chunk');
        if (decodeIndex != null) {
          decode.add(_IndexedPath(decodeIndex, entry.path));
          continue;
        }
        final prefillIndex = _chunkIndex(name, 'prefill_chunk');
        if (prefillIndex != null) {
          prefill.add(_IndexedPath(prefillIndex, entry.path));
          continue;
        }
      }
      if (entry is File && name != 'model_config.json') {
        sidecars.add(entry.path);
      }
    }

    decode.sort();
    prefill.sort();
    sidecars.sort();
    return CoreMlBundleLayout(
      rootPath: root.path,
      modelConfigPath: config.existsSync() ? config.path : null,
      monolithicModelPath: monolithic,
      decodeChunks: decode.map((chunk) => chunk.path).toList(),
      prefillChunks: prefill.map((chunk) => chunk.path).toList(),
      sidecars: sidecars,
    );
  }
}

final class _IndexedPath implements Comparable<_IndexedPath> {
  const _IndexedPath(this.index, this.path);

  final int index;
  final String path;

  @override
  int compareTo(_IndexedPath other) => index.compareTo(other.index);
}

bool _isCoreMlBundle(String name) =>
    name.endsWith('.mlmodelc') || name.endsWith('.mlpackage');

int? _chunkIndex(String name, String prefix) {
  final stem = name
      .replaceFirst(RegExp(r'\.mlmodelc$'), '')
      .replaceFirst(RegExp(r'\.mlpackage$'), '');
  if (prefix == 'chunk' && stem == 'chunk_head') {
    return 1000000;
  }
  final match = RegExp('^${RegExp.escape(prefix)}_?(\\d+)\$').firstMatch(stem);
  if (match == null) return null;
  return int.parse(match.group(1)!);
}

String _basename(String path) {
  final normalized = path.replaceAll('\\', '/');
  final index = normalized.lastIndexOf('/');
  return index < 0 ? normalized : normalized.substring(index + 1);
}
