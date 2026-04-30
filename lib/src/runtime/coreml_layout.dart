/// Core ML artifact layout helpers.
library;

import 'dart:ffi' as ffi;

import 'native_ffi.dart' as dz;
import 'package:ffi/ffi.dart';

import 'native_bindings.dart' as native;

const _fieldSep = '\x1f';
const _listSep = '\x1e';

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
    final path = dz.NativeUtf8CString.utf8(rootPath);
    ffi.Pointer<ffi.Char> result = ffi.nullptr;
    try {
      result = native.coremlLayout(path.pointer);
      if (result == ffi.nullptr) {
        return CoreMlBundleLayout(rootPath: rootPath);
      }
      final fields = result.cast<Utf8>().toDartString().split(_fieldSep);
      if (fields.length < 7) {
        return CoreMlBundleLayout(rootPath: rootPath);
      }
      return CoreMlBundleLayout(
        rootPath: _string(fields[0]) ?? rootPath,
        pipelineSpecPath: _string(fields[1]),
        modelConfigPath: _string(fields[2]),
        monolithicModelPath: _string(fields[3]),
        decodeChunks: _strings(fields[4]),
        prefillChunks: _strings(fields[5]),
        sidecars: _strings(fields[6]),
      );
    } finally {
      if (result != ffi.nullptr) {
        native.freeStr(result);
      }
      path.close();
    }
  }
}

String? _string(Object? value) =>
    value is String && value.isNotEmpty ? value : null;

List<String> _strings(String value) {
  if (value.isEmpty) {
    return const [];
  }
  return [
    for (final item in value.split(_listSep))
      if (item.isNotEmpty) item,
  ];
}
