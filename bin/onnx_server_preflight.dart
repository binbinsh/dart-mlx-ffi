import 'dart:io';

import 'package:dart_inference/runtime.dart';

final class OnnxServerPreflightException implements Exception {
  const OnnxServerPreflightException(this.message, this.audit);

  final String message;
  final RuntimeDependencyAudit audit;

  @override
  String toString() => message;
}

Map<String, Object?> onnxServerFatalPayload(Object error, StackTrace stack) =>
    error is OnnxServerPreflightException
    ? {
        'type': 'fatal',
        'stage': 'preflight',
        'error': error.message,
        'runtimeDependencyAudit': error.audit.toJson(),
      }
    : {'type': 'fatal', 'error': '$error', 'stack': '$stack'};

Map<String, Object?> onnxServerErrorPayload(
  Object? requestId,
  Object error,
  StackTrace stack,
) => error is OnnxServerPreflightException
    ? {
        'type': 'error',
        'id': requestId,
        'stage': 'preflight',
        'error': error.message,
        'runtimeDependencyAudit': error.audit.toJson(),
      }
    : {'type': 'error', 'id': requestId, 'error': '$error', 'stack': '$stack'};

void preflightOnnxProvider({
  required String? provider,
  required bool requireProvider,
  required String? runtimeRoot,
  required List<String> dependencySearchDirs,
}) {
  if (!requireProvider || provider == null || provider.isEmpty) {
    return;
  }
  final audit = RuntimeDependencyAudit.inspect(
    root: runtimeRoot,
    provider: provider,
    extraSearchDirs: dependencySearchDirs,
  );
  final reason = audit.skipReason;
  if (reason != null) {
    throw OnnxServerPreflightException(reason, audit);
  }
}

String? onnxRuntimeRoot({String? explicitRuntimeRoot, String? explicitRoot}) {
  for (final value in [
    explicitRuntimeRoot,
    explicitRoot,
    Platform.environment['UNIFRONTEND_ROOT'],
    Directory.current.path,
  ]) {
    final trimmed = value?.trim();
    if (trimmed != null && trimmed.isNotEmpty) {
      return trimmed;
    }
  }
  return null;
}

List<String> onnxDependencySearchDirs({
  Iterable<String> cudaLibraryDirs = const [],
  Iterable<String> nativeLibraryDirs = const [],
  Iterable<String> libraryDirs = const [],
}) => [...cudaLibraryDirs, ...nativeLibraryDirs, ...libraryDirs];

String? onnxRuntimeRootFromPayload(
  Map<String, dynamic> payload,
  String? fallback,
) {
  final value = payload['runtime_root'] ?? payload['runtimeRoot'];
  final trimmed = value?.toString().trim();
  return trimmed == null || trimmed.isEmpty ? fallback : trimmed;
}

List<String> onnxDependencySearchDirsFromPayload(Map<String, dynamic> payload) {
  final preload = payload['preload_libraries'] ?? payload['preloadLibraries'];
  return [
    ..._stringList(payload['cuda_library_dir']),
    ..._stringList(payload['cudaLibraryDir']),
    ..._stringList(payload['native_library_dir']),
    ..._stringList(payload['nativeLibraryDir']),
    ..._stringList(payload['library_dir']),
    ..._stringList(payload['libraryDir']),
    ..._parentDirs(_preloadLibraries(preload)),
  ];
}

List<String> _preloadLibraries(Object? value) {
  if (value is String) {
    return value
        .split(RegExp(r'[:,;\n\r]+'))
        .map((part) => part.trim())
        .where((part) => part.isNotEmpty)
        .toList(growable: false);
  }
  if (value is List) {
    return value
        .map((item) => item.toString().trim())
        .where((part) => part.isNotEmpty)
        .toList(growable: false);
  }
  return const [];
}

List<String> _stringList(Object? value) {
  if (value is List) {
    return value
        .map((item) => item.toString().trim())
        .where((item) => item.isNotEmpty)
        .toList(growable: false);
  }
  final trimmed = value?.toString().trim();
  return trimmed == null || trimmed.isEmpty ? const [] : [trimmed];
}

List<String> _parentDirs(List<String> paths) {
  final dirs = <String>{};
  for (final path in paths) {
    final trimmed = path.trim();
    if (trimmed.isNotEmpty) {
      dirs.add(File(trimmed).absolute.parent.path);
    }
  }
  return dirs.toList(growable: false);
}
