import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../models/shared/model_spec.dart';
import '../models/shared/runtime_metadata.dart';
import 'native_tensor_buffers.dart';
import 'runtime.dart';
import 'runtime_deps.dart';
import 'runtime_library_dirs.dart';

const _disabledRuntimeEnvFile = '<dart_inference:no-runtime-env>';

/// Returns the ONNX Runtime execution-provider name for common user aliases.
String? canonicalOnnxExecutionProvider(String value) {
  final normalized = value.trim().toLowerCase();
  return switch (normalized) {
    'cpu' || 'cpuexecutionprovider' => 'CPUExecutionProvider',
    'coreml' || 'coremlexecutionprovider' => 'CoreMLExecutionProvider',
    'cuda' || 'cudaexecutionprovider' => 'CUDAExecutionProvider',
    'trt' ||
    'tensorrt' ||
    'tensorrtexecutionprovider' => 'TensorrtExecutionProvider',
    'directml' || 'dml' || 'dmlexecutionprovider' => 'DmlExecutionProvider',
    'openvino' || 'openvinoexecutionprovider' => 'OpenVINOExecutionProvider',
    'rocm' || 'rocmexecutionprovider' => 'ROCMExecutionProvider',
    'qnn' || 'npu' || 'qnnexecutionprovider' => 'QNNExecutionProvider',
    'xnnpack' || 'xnnpackexecutionprovider' => 'XnnpackExecutionProvider',
    _ => null,
  };
}

/// Convenience configuration for loading an ONNX Runtime session through the
/// package's native runtime bridge.
final class DartOnnxConfig {
  const DartOnnxConfig({
    required this.modelPath,
    required this.id,
    required this.family,
    this.provider = 'cuda',
    this.deviceId = 0,
    this.requireProvider = true,
    this.numThreads = 4,
    this.preferCpu = false,
    this.backendOptions = const {},
    this.runDiagnostics = false,
  });

  final String modelPath;
  final String id;
  final String family;
  final String provider;
  final int deviceId;
  final bool requireProvider;
  final int numThreads;
  final bool preferCpu;
  final Map<String, Object?> backendOptions;
  final bool runDiagnostics;
}

/// Thin Dart ONNX Runtime session wrapper.
///
/// This keeps model packages from repeating [ModelSpec], [RuntimeOptions], and
/// tensor boilerplate while still using the shared `dart_inference` runtime API.
final class DartOnnxSession {
  DartOnnxSession._(this._session);

  final ModelSession _session;
  Map<String, Object?>? _diagnostics;

  Map<String, Object?> get diagnostics => _diagnostics ??= _session.diagnostics;

  String get selectedProvider => '${diagnostics['provider'] ?? 'unknown'}';

  static DartOnnxSession load(DartOnnxConfig config) {
    try {
      return _loadNative(config);
    } catch (error) {
      final requestedProvider = canonicalOnnxExecutionProvider(config.provider);
      if (config.requireProvider ||
          requestedProvider == 'CPUExecutionProvider') {
        rethrow;
      }
      try {
        return _loadNative(_cpuFallbackConfig(config));
      } catch (fallbackError) {
        throw StateError(
          'Failed to load ${config.id} with ${config.provider}; '
          'CPU fallback also failed. Provider error: $error. '
          'CPU error: $fallbackError',
        );
      }
    }
  }

  static DartOnnxSession _loadNative(DartOnnxConfig config) {
    // Send the canonical provider name (e.g. 'CoreMLExecutionProvider')
    // to the native bridge so its provider-availability gate matches
    // values returned by GetAvailableProviders. Falls back to the raw
    // string for unknown aliases.
    final canonicalProvider =
        canonicalOnnxExecutionProvider(config.provider) ?? config.provider;
    final backendOptions = <String, Object?>{
      'provider': canonicalProvider,
      'deviceId': config.deviceId,
      if (config.requireProvider) 'requireProvider': true,
      ...config.backendOptions,
    };
    if (!backendOptions.containsKey('preloadLibraries')) {
      final preloadLibraries = discoverDefaultOnnxRuntimePreloadLibraries(
        libraryNames: onnxRuntimePreloadLibraryNamesForProvider(
          config.provider,
        ),
      );
      if (preloadLibraries.isNotEmpty) {
        backendOptions['preloadLibraries'] = encodeOnnxRuntimePreloadLibraries(
          preloadLibraries,
        );
      }
    }
    final spec = ModelSpec(
      id: config.id,
      family: config.family,
      modalities: const [ModelModality.textGeneration],
      platformArtifacts: {
        RuntimeEngine.onnx: RuntimeArtifact(
          engine: RuntimeEngine.onnx,
          path: config.modelPath,
          targetPlatforms: [RuntimePlatformCurrent.current().name],
        ),
      },
    );
    final session = RuntimeRegistry.native().load(
      spec,
      options: RuntimeOptions(
        engine: RuntimeEngine.onnx,
        prefer: config.preferCpu
            ? const [Accelerator.cpu]
            : const [Accelerator.gpu, Accelerator.cpu],
        diagnostics: config.runDiagnostics,
        numThreads: config.numThreads > 0 ? config.numThreads : null,
        backendOptions: backendOptions,
      ),
    );
    return DartOnnxSession._(session);
  }

  DartOnnxResult run(Map<String, Object?> inputs) {
    final outputs = _session.run(ModelInputs(inputs));
    return DartOnnxResult(
      outputs.values,
      outputs.diagnostics,
      release: outputs.close,
    );
  }

  void close() {
    _session.close();
  }
}

DartOnnxConfig _cpuFallbackConfig(DartOnnxConfig config) {
  return DartOnnxConfig(
    modelPath: config.modelPath,
    id: config.id,
    family: config.family,
    provider: 'cpu',
    deviceId: 0,
    requireProvider: false,
    numThreads: config.numThreads,
    preferCpu: true,
    backendOptions: _cpuFallbackBackendOptions(config.backendOptions),
    runDiagnostics: config.runDiagnostics,
  );
}

Map<String, Object?> _cpuFallbackBackendOptions(Map<String, Object?> options) {
  const providerOnlyKeys = {
    'cudaMemoryLimitMb',
    'gpuMemoryLimitMb',
    'cudaGraph',
    'cudaGraphId',
    'deviceOutputNames',
    'inputDeviceNames',
    'preferredOutputDevice',
    'preloadLibraries',
    'prepackedWeightsKey',
    'syncInputs',
    'syncOutputs',
    'trtEngineCacheEnable',
    'trtEngineCachePath',
    'trtCacheDir',
    'trtFp16',
    'trtMaxWorkspaceSizeMb',
    'trtProfileMaxShapes',
    'trtProfileMinShapes',
    'trtProfileOptShapes',
    'trtWorkspaceMemoryLimitMb',
    'useEnvAllocators',
    'useIoBinding',
    'useOutputViews',
  };
  final sanitized = <String, Object?>{};
  for (final entry in options.entries) {
    if (!providerOnlyKeys.contains(entry.key)) {
      sanitized[entry.key] = entry.value;
    }
  }
  return sanitized;
}

final class DartOnnxResult {
  const DartOnnxResult(
    this.outputs,
    this.diagnostics, {
    void Function()? release,
  }) : _release = release;

  final Map<String, Object?> outputs;
  final Map<String, Object?> diagnostics;
  final void Function()? _release;

  String providerOr(String fallback) =>
      '${diagnostics['provider'] ?? fallback}';

  void close() {
    _release?.call();
  }
}

RuntimeTensor int64Tensor(Int64List values, List<int> shape) {
  return nativeInt64Buffer(values, shape: shape).tensor;
}

RuntimeTensor float32Tensor(Float32List values, List<int> shape) {
  return nativeFloat32Buffer(values, shape: shape).tensor;
}

RuntimeTensor boolTensor(Uint8List values, List<int> shape) {
  return nativeBooleanBuffer(values, shape: shape).tensor;
}

Float32List float32View(RuntimeTensor tensor) {
  if (tensor.isNativeHandle) {
    throw StateError('Runtime tensor is backed by a native runtime handle.');
  }
  return Float32List.view(
    tensor.bytes.buffer,
    tensor.bytes.offsetInBytes,
    tensor.byteLength ~/ 4,
  );
}

List<String> discoverDefaultOnnxRuntimePreloadLibraries({
  Iterable<String> explicitLibraries = const [],
  Iterable<String> libraryDirectories = const [],
  Iterable<String>? libraryNames,
  String? runtimeEnvFile,
  Iterable<String> runtimeEnvSearchRoots = const [],
}) {
  return _ortLibs(
    runtimeEnvFile: runtimeEnvFile,
    runtimeEnvSearchRoots: runtimeEnvSearchRoots,
    explicitLibraries: explicitLibraries,
    libraryDirectories: libraryDirectories,
    libraryNames: libraryNames,
  );
}

List<String> discoverOnnxRuntimePreloadLibraries({
  Iterable<String> explicitLibraries = const [],
  Iterable<String> libraryDirectories = const [],
  Iterable<String>? libraryNames,
  Iterable<String> runtimeEnvSearchRoots = const [],
}) => _ortLibs(
  runtimeEnvFile: _disabledRuntimeEnvFile,
  runtimeEnvSearchRoots: runtimeEnvSearchRoots,
  explicitLibraries: explicitLibraries,
  libraryDirectories: libraryDirectories,
  libraryNames: libraryNames,
);

List<String> onnxRuntimePreloadLibraryNamesForProvider(String provider) =>
    RuntimeDependencyAudit.preloadLibrariesForProvider(provider);

String encodeOnnxRuntimePreloadLibraries(Iterable<String> libraries) {
  return libraries.where((path) => path.isNotEmpty).join(':');
}

List<String> _ortLibs({
  String? runtimeEnvFile,
  Iterable<String> runtimeEnvSearchRoots = const [],
  Iterable<String> explicitLibraries = const [],
  Iterable<String> libraryDirectories = const [],
  Iterable<String>? libraryNames,
}) {
  final roots = cleanRuntimeLibraryPaths(runtimeEnvSearchRoots);
  final requestedNames = libraryNames
      ?.where((value) => value.trim().isNotEmpty)
      .toList(growable: false);
  if (requestedNames != null && requestedNames.isEmpty) {
    return explicitLibraries
        .where((value) => value.trim().isNotEmpty)
        .toList(growable: false);
  }
  final dirs = [
    ...libraryDirectories,
    ..._runtimeEnvLibraryDirectories(runtimeEnvFile),
    ...runtimeLibraryDirectories(roots),
    ...pythonNvidiaLibraryDirectories(roots),
  ];
  final out = <String>[];
  final seen = <String>{};
  for (final path in explicitLibraries) {
    final trimmed = path.trim();
    if (trimmed.isNotEmpty && File(trimmed).existsSync()) {
      final absolute = File(trimmed).absolute.path;
      if (seen.add(absolute)) out.add(absolute);
    }
  }
  final namesToFind = requestedNames ?? RuntimeDependencyAudit.cudaLibraries;
  for (final name in namesToFind) {
    for (final dir in dirs) {
      final candidate = File('$dir/$name');
      if (candidate.existsSync()) {
        final absolute = candidate.absolute.path;
        if (seen.add(absolute)) out.add(absolute);
        break;
      }
    }
  }
  return out;
}

List<String> _runtimeEnvLibraryDirectories(String? runtimeEnvFile) {
  if (runtimeEnvFile == null || runtimeEnvFile == _disabledRuntimeEnvFile) {
    return const [];
  }
  final file = File(runtimeEnvFile);
  if (!file.existsSync()) {
    return const [];
  }
  final dirs = <String>{};
  try {
    final decoded = jsonDecode(file.readAsStringSync());
    if (decoded is! Map) return const [];
    final ort = decoded['DART_INFERENCE_ORT_RUNTIME_LIBRARY'];
    if (ort is String && ort.trim().isNotEmpty) {
      final ortFile = File(ort);
      final runtimeRoot = ortFile.absolute.parent.parent.parent.path;
      addExistingRuntimeLibraryDir(dirs, ortFile.absolute.parent.path);
      addExistingRuntimeLibraryDir(dirs, '$runtimeRoot/cuda/lib');
      addExistingRuntimeLibraryDir(dirs, '$runtimeRoot/tensorrt/lib');
    }
  } catch (_) {
    return const [];
  }
  return dirs.toList(growable: false);
}
