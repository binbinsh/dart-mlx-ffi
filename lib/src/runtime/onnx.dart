import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../models/shared/model_spec.dart';
import '../models/shared/runtime_metadata.dart';
import 'native_bindings.dart' as native;
import 'native_runtime.dart';
import 'runtime.dart';

const _disabledRuntimeEnvFile = '<dart_inference:no-runtime-env>';

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
  final buffer = NativeTensorBuffer.int64(shape);
  buffer.copyFrom(values);
  return buffer.tensor;
}

RuntimeTensor float32Tensor(Float32List values, List<int> shape) {
  final buffer = NativeTensorBuffer.float32(shape);
  buffer.copyFrom(values);
  return buffer.tensor;
}

RuntimeTensor boolTensor(Uint8List values, List<int> shape) {
  final buffer = NativeTensorBuffer.boolean(shape);
  buffer.copyFrom(values);
  return buffer.tensor;
}

Float32List float32View(RuntimeTensor tensor) => Float32List.view(
  tensor.bytes.buffer,
  tensor.bytes.offsetInBytes,
  tensor.bytes.lengthInBytes ~/ 4,
);

List<String> discoverDefaultOnnxRuntimePreloadLibraries({
  Iterable<String> explicitLibraries = const [],
  Iterable<String> libraryDirectories = const [],
  Iterable<String> libraryNames = const [],
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
  Iterable<String> libraryNames = const [],
}) => _ortLibs(
  runtimeEnvFile: _disabledRuntimeEnvFile,
  explicitLibraries: explicitLibraries,
  libraryDirectories: libraryDirectories,
  libraryNames: libraryNames,
);

String encodeOnnxRuntimePreloadLibraries(Iterable<String> libraries) {
  return libraries.where((path) => path.isNotEmpty).join(':');
}

List<String> _ortLibs({
  String? runtimeEnvFile,
  Iterable<String> runtimeEnvSearchRoots = const [],
  Iterable<String> explicitLibraries = const [],
  Iterable<String> libraryDirectories = const [],
  Iterable<String> libraryNames = const [],
}) {
  final envFile = _nativeText(runtimeEnvFile ?? '');
  final roots = _nativeText(_pack(runtimeEnvSearchRoots));
  final explicit = _nativeText(_pack(explicitLibraries));
  final dirs = _nativeText(_pack(libraryDirectories));
  final names = _nativeText(_pack(libraryNames));
  ffi.Pointer<ffi.Char> result = ffi.nullptr;
  try {
    result = native.ortLibs(envFile, roots, explicit, dirs, names);
    if (result == ffi.nullptr) {
      return const [];
    }
    final text = result.cast<Utf8>().toDartString();
    if (text.isEmpty) {
      return const [];
    }
    return [
      for (final value in text.split('\n'))
        if (value.isNotEmpty) value,
    ];
  } finally {
    if (result != ffi.nullptr) {
      native.freeStr(result);
    }
    calloc
      ..free(envFile)
      ..free(roots)
      ..free(explicit)
      ..free(dirs)
      ..free(names);
  }
}

ffi.Pointer<ffi.Char> _nativeText(String value) {
  return value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
}

String _pack(Iterable<String> values) {
  return values.where((value) => value.isNotEmpty).join('\n');
}
