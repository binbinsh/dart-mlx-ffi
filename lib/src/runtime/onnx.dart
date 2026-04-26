import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../models/shared/model_spec.dart';
import '../models/shared/runtime_metadata.dart';
import 'runtime.dart';

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
}

/// Thin Dart ONNX Runtime session wrapper.
///
/// This keeps model packages from repeating [ModelSpec], [RuntimeOptions], and
/// tensor boilerplate while still using the shared `dart_inference` runtime API.
final class DartOnnxSession {
  DartOnnxSession._(this._session);

  final ModelSession _session;

  Map<String, Object?> get diagnostics => _session.diagnostics;

  String get selectedProvider =>
      '${_session.diagnostics['provider'] ?? 'unknown'}';

  static DartOnnxSession load(DartOnnxConfig config) {
    final backendOptions = <String, Object?>{
      'provider': config.provider,
      'deviceId': config.deviceId,
      if (config.requireProvider) 'requireProvider': true,
      ...config.backendOptions,
    };
    if (!backendOptions.containsKey('preloadLibraries')) {
      final preloadLibraries = discoverDefaultOnnxRuntimePreloadLibraries();
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
        diagnostics: true,
        numThreads: config.numThreads > 0 ? config.numThreads : null,
        backendOptions: backendOptions,
      ),
    );
    return DartOnnxSession._(session);
  }

  DartOnnxResult run(Map<String, Object?> inputs) {
    final outputs = _session.run(ModelInputs(inputs));
    return DartOnnxResult(outputs.values, outputs.diagnostics);
  }

  void close() {
    _session.close();
  }
}

final class DartOnnxResult {
  const DartOnnxResult(this.outputs, this.diagnostics);

  final Map<String, Object?> outputs;
  final Map<String, Object?> diagnostics;

  String providerOr(String fallback) =>
      '${diagnostics['provider'] ?? fallback}';
}

RuntimeTensor int64Tensor(Int64List values, List<int> shape) => RuntimeTensor(
  dtype: RuntimeTensorDataType.int64,
  shape: shape,
  bytes: Uint8List.view(values.buffer),
);

RuntimeTensor float32Tensor(Float32List values, List<int> shape) =>
    RuntimeTensor(
      dtype: RuntimeTensorDataType.float32,
      shape: shape,
      bytes: Uint8List.view(values.buffer),
    );

RuntimeTensor boolTensor(Uint8List values, List<int> shape) => RuntimeTensor(
  dtype: RuntimeTensorDataType.boolean,
  shape: shape,
  bytes: values,
);

Float32List float32View(RuntimeTensor tensor) => Float32List.view(
  tensor.bytes.buffer,
  tensor.bytes.offsetInBytes,
  tensor.bytes.lengthInBytes ~/ 4,
);

const List<String> onnxCudaPreloadLibraryNames = [
  'libcudart.so.12',
  'libcublas.so.12',
  'libcublasLt.so.12',
  'libcurand.so.10',
  'libcufft.so.11',
  'libcudnn.so.9',
  'libcudnn_ops.so.9',
  'libcudnn_cnn.so.9',
  'libcudnn_adv.so.9',
  'libcudnn_graph.so.9',
  'libcudnn_heuristic.so.9',
  'libcudnn_engines_precompiled.so.9',
  'libcudnn_engines_runtime_compiled.so.9',
  'libnvinfer.so.10',
  'libnvinfer_plugin.so.10',
  'libnvonnxparser.so.10',
  'libnvinfer.so.9',
  'libnvinfer_plugin.so.9',
  'libnvonnxparser.so.9',
];

/// Runtime environment values used by the build hook and Dart ONNX helpers.
///
/// Values are loaded from `DART_INFERENCE_RUNTIME_ENV_FILE` when present, otherwise
/// from `.dart_inference_runtime_env.json` in the current working directory or one of
/// the caller-provided roots. Process environment values take precedence over
/// file values.
final class DartInferenceRuntimeEnv {
  const DartInferenceRuntimeEnv._(this.fileValues, {this.filePath});

  final Map<String, String> fileValues;
  final String? filePath;

  static DartInferenceRuntimeEnv load({
    String? runtimeEnvFile,
    Iterable<String> searchRoots = const [],
  }) {
    final file = _resolveRuntimeEnvFile(runtimeEnvFile, searchRoots);
    final values = <String, String>{};
    if (file != null && file.existsSync()) {
      try {
        final decoded = jsonDecode(file.readAsStringSync());
        if (decoded is Map) {
          for (final entry in decoded.entries) {
            final key = entry.key;
            final value = entry.value;
            if (key is String && value is String && value.isNotEmpty) {
              values[key] = value;
            }
          }
        }
      } catch (_) {
        // Malformed local env files should not prevent explicit CLI/env config.
      }
    }
    return DartInferenceRuntimeEnv._(values, filePath: file?.path);
  }

  String? value(String name) {
    final envValue = Platform.environment[name];
    if (envValue != null && envValue.isNotEmpty) {
      return envValue;
    }
    final fileValue = fileValues[name];
    if (fileValue != null && fileValue.isNotEmpty) {
      return fileValue;
    }
    return null;
  }

  List<String> splitPaths(String name) => _splitRuntimePathList(value(name));

  List<String> onnxPreloadLibraryDirectories({
    Iterable<String> extraDirectories = const [],
  }) {
    final dirs = <String>[
      ...extraDirectories,
      ...splitPaths('DART_INFERENCE_NATIVE_LIBRARY_DIRS'),
      ...splitPaths('DART_INFERENCE_CUDA_LIBRARY_DIRS'),
      ...splitPaths('DART_INFERENCE_CUDA_LIB_DIRS'),
      ...splitPaths('DART_INFERENCE_TENSORRT_LIBRARY_DIRS'),
      ?value('DART_INFERENCE_CUDA_LIB_DIR'),
      ?value('DART_INFERENCE_TENSORRT_LIB_DIR'),
    ];
    for (final name in const [
      'DART_INFERENCE_ORT_RUNTIME_LIBRARY',
      'DART_INFERENCE_ORT_LIBRARY',
    ]) {
      final library = value(name);
      if (library == null || library.isEmpty) {
        continue;
      }
      final file = File(library).absolute;
      final libDir = file.parent;
      dirs.add(libDir.path);
      final ortRoot = libDir.parent;
      final runtimeRoot = ortRoot.parent;
      dirs.add('${runtimeRoot.path}/cuda/lib');
      dirs.add('${runtimeRoot.path}/tensorrt/lib');
    }
    return _dedupeExistingDirectories(dirs);
  }
}

List<String> discoverDefaultOnnxRuntimePreloadLibraries({
  Iterable<String> explicitLibraries = const [],
  Iterable<String> libraryDirectories = const [],
  Iterable<String> libraryNames = onnxCudaPreloadLibraryNames,
  String? runtimeEnvFile,
  Iterable<String> runtimeEnvSearchRoots = const [],
}) {
  final runtimeEnv = DartInferenceRuntimeEnv.load(
    runtimeEnvFile: runtimeEnvFile,
    searchRoots: runtimeEnvSearchRoots,
  );
  return discoverOnnxRuntimePreloadLibraries(
    explicitLibraries: explicitLibraries,
    libraryDirectories: runtimeEnv.onnxPreloadLibraryDirectories(
      extraDirectories: libraryDirectories,
    ),
    libraryNames: libraryNames,
  );
}

List<String> discoverOnnxRuntimePreloadLibraries({
  Iterable<String> explicitLibraries = const [],
  Iterable<String> libraryDirectories = const [],
  Iterable<String> libraryNames = onnxCudaPreloadLibraryNames,
}) {
  final seen = <String>{};
  final out = <String>[];
  void add(String path) {
    if (path.isEmpty) {
      return;
    }
    final absolute = File(path).absolute.path;
    if (seen.add(absolute)) {
      out.add(absolute);
    }
  }

  for (final path in explicitLibraries) {
    if (File(path).existsSync()) {
      add(path);
    }
  }
  for (final directory in libraryDirectories) {
    if (directory.isEmpty || !Directory(directory).existsSync()) {
      continue;
    }
    for (final name in libraryNames) {
      final path = '$directory/$name';
      if (File(path).existsSync()) {
        add(path);
      }
    }
  }
  return out;
}

String encodeOnnxRuntimePreloadLibraries(Iterable<String> libraries) {
  return libraries.where((path) => path.isNotEmpty).join(':');
}

File? _resolveRuntimeEnvFile(
  String? runtimeEnvFile,
  Iterable<String> searchRoots,
) {
  for (final candidate in [
    runtimeEnvFile,
    Platform.environment['DART_INFERENCE_RUNTIME_ENV_FILE'],
    '${Directory.current.path}/.dart_inference_runtime_env.json',
    for (final root in searchRoots) '$root/.dart_inference_runtime_env.json',
  ]) {
    if (candidate == null || candidate.isEmpty) {
      continue;
    }
    final file = File(candidate).absolute;
    if (file.existsSync()) {
      return file;
    }
  }
  return null;
}

List<String> _splitRuntimePathList(String? raw) {
  if (raw == null || raw.isEmpty) {
    return const [];
  }
  final separators = Platform.isWindows
      ? RegExp(r'[;,\n\r]+')
      : RegExp(r'[:,;\n\r]+');
  final out = <String>[];
  final seen = <String>{};
  for (final part in raw.split(separators)) {
    final path = part.trim();
    if (path.isEmpty || !seen.add(path)) {
      continue;
    }
    out.add(path);
  }
  return out;
}

List<String> _dedupeExistingDirectories(Iterable<String> dirs) {
  final out = <String>[];
  final seen = <String>{};
  for (final dir in dirs) {
    if (dir.isEmpty) {
      continue;
    }
    final absolute = Directory(dir).absolute.path;
    if (seen.add(absolute) && Directory(absolute).existsSync()) {
      out.add(absolute);
    }
  }
  return out;
}
