/// Core ML artifact layout helpers.
library;

import 'dart:convert';
import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'native_bindings.dart' as native;

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
    final path = rootPath.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    ffi.Pointer<ffi.Char> result = ffi.nullptr;
    try {
      result = native.coremlLayoutJson(path);
      if (result == ffi.nullptr) {
        return CoreMlBundleLayout(rootPath: rootPath);
      }
      final decoded = jsonDecode(result.cast<Utf8>().toDartString());
      if (decoded is! Map<String, Object?>) {
        return CoreMlBundleLayout(rootPath: rootPath);
      }
      return CoreMlBundleLayout(
        rootPath: _string(decoded['root_path']) ?? rootPath,
        pipelineSpecPath: _string(decoded['pipeline_spec_path']),
        modelConfigPath: _string(decoded['model_config_path']),
        monolithicModelPath: _string(decoded['monolithic_model_path']),
        decodeChunks: _strings(decoded['decode_chunks']),
        prefillChunks: _strings(decoded['prefill_chunks']),
        sidecars: _strings(decoded['sidecars']),
      );
    } finally {
      if (result != ffi.nullptr) {
        native.freeStr(result);
      }
      calloc.free(path);
    }
  }
}

String? _string(Object? value) =>
    value is String && value.isNotEmpty ? value : null;

List<String> _strings(Object? value) {
  if (value is! List) {
    return const [];
  }
  return [
    for (final item in value)
      if (item is String && item.isNotEmpty) item,
  ];
}
