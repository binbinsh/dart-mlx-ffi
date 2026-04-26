/// Model-level runtime abstractions and resolution.
library;

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../models/shared/model_spec.dart';
import '../models/shared/runtime_metadata.dart';
import 'artifact_resolver.dart';
import 'native_bindings.dart' as native;
import 'native_runtime.dart';

/// Host platforms used by runtime resolution.
enum RuntimePlatform { ios, macos, windows, linux, android, unknown }

extension RuntimePlatformCurrent on RuntimePlatform {
  /// Detect the current Dart VM platform.
  static RuntimePlatform current() => _platformById(native.platformId());
}

RuntimePlatform _platformById(int id) => switch (id) {
  0 => RuntimePlatform.ios,
  1 => RuntimePlatform.macos,
  2 => RuntimePlatform.windows,
  3 => RuntimePlatform.linux,
  4 => RuntimePlatform.android,
  _ => RuntimePlatform.unknown,
};

/// User-selected runtime preferences.
final class RuntimeOptions {
  const RuntimeOptions({
    this.engine,
    this.prefer = const <Accelerator>[],
    this.allowFallback = true,
    this.diagnostics = false,
    this.numThreads,
    this.backendOptions = const <String, Object?>{},
    this.artifactResolver,
  });

  /// Explicit runtime engine, if the caller wants to override defaults.
  final RuntimeEngine? engine;

  /// Preferred accelerators in priority order.
  final List<Accelerator> prefer;

  /// Whether the resolver may select another engine when the preferred one
  /// has no compatible artifact.
  final bool allowFallback;

  /// Whether runtime implementations should collect detailed diagnostics.
  final bool diagnostics;

  /// Optional CPU worker thread count for native runtimes.
  final int? numThreads;

  /// Backend-specific native runtime options.
  ///
  /// Common keys understood by bundled backends include `provider` for ONNX
  /// Runtime, `delegate` for LiteRT, and `litertSectionIndex` for selecting
  /// a TFLite flatbuffer inside multi-section `.task` / `.litertlm` files.
  final Map<String, Object?> backendOptions;

  /// Optional resolver for remote artifact URIs such as `hf://...`.
  ///
  /// Synchronous [RuntimeRegistry.load] uses this only for cache hits. Use
  /// [RuntimeRegistry.loadAsync] to allow network downloads.
  final RuntimeArtifactResolver? artifactResolver;
}

/// Runtime capabilities reported by an implementation.
final class RuntimeCapabilities {
  const RuntimeCapabilities({
    required this.engine,
    required this.platform,
    this.available = true,
    this.accelerators = const <Accelerator>[],
    this.details = const <String, Object?>{},
  });

  final RuntimeEngine engine;
  final RuntimePlatform platform;
  final bool available;
  final List<Accelerator> accelerators;
  final Map<String, Object?> details;
}

/// Resolved artifact and engine for a model load request.
final class RuntimeResolution {
  const RuntimeResolution({
    required this.platform,
    required this.engine,
    required this.artifact,
    required this.accelerators,
    this.fallbackReason,
  });

  final RuntimePlatform platform;
  final RuntimeEngine engine;
  final RuntimeArtifact artifact;
  final List<Accelerator> accelerators;
  final String? fallbackReason;
}

/// Model artifact root plus the selected runtime artifact.
final class ModelBundle {
  const ModelBundle({
    required this.spec,
    required this.rootPath,
    required this.artifact,
  });

  final ModelSpec spec;
  final String rootPath;
  final RuntimeArtifact artifact;

  /// Resolve the artifact path relative to [rootPath] when needed.
  String get artifactPath {
    final path = artifact.path;
    if (path.contains('://')) return path;
    if (path.startsWith('/')) return path;
    if (rootPath.isEmpty) return path;
    if (rootPath.endsWith('/')) return '$rootPath$path';
    return '$rootPath/$path';
  }
}

/// Runtime input map.
final class ModelInputs {
  const ModelInputs(this.values);

  final Map<String, Object?> values;
}

/// Runtime output map plus optional diagnostics.
final class ModelOutputs {
  const ModelOutputs(
    this.values, {
    this.diagnostics = const <String, Object?>{},
    void Function()? release,
  }) : _release = release;

  final Map<String, Object?> values;
  final Map<String, Object?> diagnostics;
  final void Function()? _release;

  /// Release native output buffers immediately.
  ///
  /// Output tensors backed by native memory must not be read after this call.
  void close() {
    _release?.call();
  }
}

/// Tensor dtypes supported by native runtime backends.
enum RuntimeTensorDataType {
  float32,
  int32,
  int64,
  uint8,
  float64,
  float16,
  boolean,
}

/// Named-tensor value passed to model-level runtimes.
final class RuntimeTensor {
  const RuntimeTensor({
    required this.dtype,
    required this.shape,
    required this.bytes,
    Object? owner,
  }) : _owner = owner;

  factory RuntimeTensor.float32(List<int> shape, Float32List data) {
    return RuntimeTensor(
      dtype: RuntimeTensorDataType.float32,
      shape: shape,
      bytes: _copyBytes(data),
    );
  }

  factory RuntimeTensor.int32(List<int> shape, Int32List data) {
    return RuntimeTensor(
      dtype: RuntimeTensorDataType.int32,
      shape: shape,
      bytes: _copyBytes(data),
    );
  }

  factory RuntimeTensor.int64(List<int> shape, Int64List data) {
    return RuntimeTensor(
      dtype: RuntimeTensorDataType.int64,
      shape: shape,
      bytes: _copyBytes(data),
    );
  }

  factory RuntimeTensor.uint8(List<int> shape, Uint8List data) {
    return RuntimeTensor(
      dtype: RuntimeTensorDataType.uint8,
      shape: shape,
      bytes: _copyBytes(data),
    );
  }

  factory RuntimeTensor.boolean(List<int> shape, Uint8List data) {
    return RuntimeTensor(
      dtype: RuntimeTensorDataType.boolean,
      shape: shape,
      bytes: _copyBytes(data),
    );
  }

  factory RuntimeTensor.float64(List<int> shape, Float64List data) {
    return RuntimeTensor(
      dtype: RuntimeTensorDataType.float64,
      shape: shape,
      bytes: _copyBytes(data),
    );
  }

  final RuntimeTensorDataType dtype;
  final List<int> shape;
  final Uint8List bytes;
  // Keeps native output owners alive for external typed-data backed tensors.
  // ignore: unused_field
  final Object? _owner;

  Float32List asFloat32List() => _view<Float32List>(
    () => bytes.buffer.asFloat32List(
      bytes.offsetInBytes,
      bytes.lengthInBytes ~/ 4,
    ),
  );

  Int32List asInt32List() => _view<Int32List>(
    () =>
        bytes.buffer.asInt32List(bytes.offsetInBytes, bytes.lengthInBytes ~/ 4),
  );

  Int64List asInt64List() => _view<Int64List>(
    () =>
        bytes.buffer.asInt64List(bytes.offsetInBytes, bytes.lengthInBytes ~/ 8),
  );

  Float64List asFloat64List() => _view<Float64List>(
    () => bytes.buffer.asFloat64List(
      bytes.offsetInBytes,
      bytes.lengthInBytes ~/ 8,
    ),
  );

  Uint8List asUint8List() => bytes;

  T _view<T extends TypedData>(T Function() create) => create();
}

/// Runtime implementation contract.
abstract interface class ModelRuntime {
  RuntimeCapabilities get capabilities;

  ModelSession load(ModelBundle bundle, RuntimeOptions options);
}

/// Loaded model session.
abstract interface class ModelSession {
  Map<String, Object?> get diagnostics;

  ModelOutputs run(ModelInputs inputs);

  Stream<ModelOutputs> stream(ModelInputs inputs);

  void close();
}

/// Registry for concrete runtime backends.
final class RuntimeRegistry {
  RuntimeRegistry({this.resolver = const RuntimeResolver()});

  final RuntimeResolver resolver;
  final Map<RuntimeEngine, ModelRuntime> _runtimes = {};

  /// Registry containing the bundled native backends.
  factory RuntimeRegistry.native({
    RuntimeResolver resolver = const RuntimeResolver(),
  }) {
    final registry = RuntimeRegistry(resolver: resolver);
    registry
      ..register(NativeModelRuntime(RuntimeEngine.mlx))
      ..register(NativeModelRuntime(RuntimeEngine.coreml))
      ..register(NativeModelRuntime(RuntimeEngine.onnx))
      ..register(NativeModelRuntime(RuntimeEngine.litert));
    return registry;
  }

  void register(ModelRuntime runtime) {
    _runtimes[runtime.capabilities.engine] = runtime;
  }

  RuntimeResolution resolve(
    ModelSpec spec, [
    RuntimeOptions options = const RuntimeOptions(),
  ]) {
    return resolver.resolve(spec, options);
  }

  ModelSession load(
    ModelSpec spec, {
    String rootPath = '',
    RuntimeOptions options = const RuntimeOptions(),
  }) {
    final resolution = resolve(spec, options);
    final selected = _runtimeFor(spec, resolution, options);
    final bundle = _resolveCachedBundle(
      ModelBundle(
        spec: spec,
        rootPath: rootPath,
        artifact: selected.resolution.artifact,
      ),
      options,
    );
    return selected.runtime.load(bundle, options);
  }

  /// Resolve remote artifacts if needed, then load the selected backend.
  Future<ModelSession> loadAsync(
    ModelSpec spec, {
    String rootPath = '',
    RuntimeOptions options = const RuntimeOptions(),
  }) async {
    final resolution = resolve(spec, options);
    final selected = _runtimeFor(spec, resolution, options);
    final bundle = await _resolveBundle(
      ModelBundle(
        spec: spec,
        rootPath: rootPath,
        artifact: selected.resolution.artifact,
      ),
      options,
    );
    return selected.runtime.load(bundle, options);
  }

  _SelectedRuntime _runtimeFor(
    ModelSpec spec,
    RuntimeResolution resolution,
    RuntimeOptions options,
  ) {
    var effective = resolution;
    var runtime = _runtimes[effective.engine];
    if (runtime == null && options.allowFallback) {
      final fallback = _registeredFallback(spec, effective.platform, options);
      if (fallback != null) {
        effective = fallback;
        runtime = _runtimes[effective.engine];
      }
    }
    if (runtime == null) {
      throw UnsupportedError(
        'No ${effective.engine.name} runtime backend is registered. '
        'Selected artifact: ${effective.artifact.path}',
      );
    }
    return _SelectedRuntime(effective, runtime);
  }

  ModelBundle _resolveCachedBundle(ModelBundle bundle, RuntimeOptions options) {
    final resolver = options.artifactResolver;
    if (resolver == null) return bundle;
    final artifact = resolver.resolveCached(bundle.artifact);
    return ModelBundle(
      spec: bundle.spec,
      rootPath: bundle.rootPath,
      artifact: artifact,
    );
  }

  Future<ModelBundle> _resolveBundle(
    ModelBundle bundle,
    RuntimeOptions options,
  ) async {
    final resolver = options.artifactResolver ?? HuggingFaceArtifactCache();
    final artifact = await resolver.resolve(bundle.artifact);
    return ModelBundle(
      spec: bundle.spec,
      rootPath: bundle.rootPath,
      artifact: artifact,
    );
  }

  RuntimeResolution? _registeredFallback(
    ModelSpec spec,
    RuntimePlatform platform,
    RuntimeOptions options,
  ) {
    for (final entry in spec.platformArtifacts.entries) {
      if (!_runtimes.containsKey(entry.key)) continue;
      final artifact = entry.value;
      if (artifact.targetPlatforms.isNotEmpty &&
          !artifact.targetPlatforms.contains(platform.name)) {
        continue;
      }
      return RuntimeResolution(
        platform: platform,
        engine: entry.key,
        artifact: artifact,
        accelerators: options.prefer,
        fallbackReason: 'Selected engine has no registered runtime backend.',
      );
    }
    return null;
  }
}

final class _SelectedRuntime {
  const _SelectedRuntime(this.resolution, this.runtime);

  final RuntimeResolution resolution;
  final ModelRuntime runtime;
}

Uint8List _copyBytes(TypedData data) {
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

/// Runtime selection policy.
final class RuntimeResolver {
  const RuntimeResolver({RuntimePlatform? hostPlatform})
    : _hostPlatform = hostPlatform;

  final RuntimePlatform? _hostPlatform;

  RuntimePlatform get hostPlatform =>
      _hostPlatform ?? RuntimePlatformCurrent.current();

  RuntimeResolution resolve(
    ModelSpec spec, [
    RuntimeOptions options = const RuntimeOptions(),
  ]) {
    final platform = hostPlatform;
    final requested = options.engine;
    if (requested != null) {
      final artifact = _artifactFor(
        spec,
        requested,
        platform,
        allowPreviewMlx: true,
      );
      if (artifact != null) {
        return RuntimeResolution(
          platform: platform,
          engine: requested,
          artifact: artifact,
          accelerators: _acceleratorsFor(requested, options),
        );
      }
      if (!options.allowFallback) {
        throw StateError(
          'Model ${spec.id} has no ${requested.name} artifact for '
          '${platform.name}.',
        );
      }
    }

    final fallbackReason = requested == null
        ? null
        : 'Requested ${requested.name} has no compatible artifact.';
    for (final engine in _defaultOrder(platform, options)) {
      final artifact = _artifactFor(spec, engine, platform);
      if (artifact == null) continue;
      return RuntimeResolution(
        platform: platform,
        engine: engine,
        artifact: artifact,
        accelerators: _acceleratorsFor(engine, options),
        fallbackReason: fallbackReason,
      );
    }
    throw StateError(
      'Model ${spec.id} has no runtime artifact for ${platform.name}.',
    );
  }

  RuntimeArtifact? _artifactFor(
    ModelSpec spec,
    RuntimeEngine engine,
    RuntimePlatform platform, {
    bool allowPreviewMlx = false,
  }) {
    final artifact = spec.platformArtifacts[engine];
    if (artifact == null) return null;
    final targets = artifact.targetPlatforms
        .join('\n')
        .toNativeUtf8(allocator: calloc)
        .cast<ffi.Char>();
    final format = (artifact.format ?? '')
        .toNativeUtf8(allocator: calloc)
        .cast<ffi.Char>();
    final path = artifact.path.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    try {
      final matches =
          native.artifactMatches(
            _engineId(engine),
            _platformId(platform),
            targets,
            format,
            path,
            allowPreviewMlx ? 1 : 0,
          ) !=
          0;
      return matches ? artifact : null;
    } finally {
      calloc
        ..free(targets)
        ..free(format)
        ..free(path);
    }
  }

  List<RuntimeEngine> _defaultOrder(
    RuntimePlatform platform,
    RuntimeOptions options,
  ) {
    final resolved = _enginesJson(
      native.engineOrderJson(_platformId(platform)),
    );
    if (resolved.isNotEmpty) {
      return resolved;
    }
    return const [RuntimeEngine.onnx];
  }

  List<Accelerator> _acceleratorsFor(
    RuntimeEngine engine,
    RuntimeOptions options,
  ) {
    if (options.prefer.isNotEmpty) return options.prefer;
    final resolved = _acceleratorsJson(
      native.engineAccelsJson(_engineId(engine)),
    );
    if (resolved.isNotEmpty) {
      return resolved;
    }
    return const [Accelerator.cpu];
  }
}

int _platformId(RuntimePlatform platform) => switch (platform) {
  RuntimePlatform.ios => 0,
  RuntimePlatform.macos => 1,
  RuntimePlatform.windows => 2,
  RuntimePlatform.linux => 3,
  RuntimePlatform.android => 4,
  RuntimePlatform.unknown => 5,
};

int _engineId(RuntimeEngine engine) => switch (engine) {
  RuntimeEngine.mlx => 0,
  RuntimeEngine.coreml => 1,
  RuntimeEngine.onnx => 2,
  RuntimeEngine.litert => 3,
};

List<RuntimeEngine> _enginesJson(ffi.Pointer<ffi.Char> ptr) {
  if (ptr == ffi.nullptr) {
    return const [];
  }
  try {
    final decoded = jsonDecode(ptr.cast<Utf8>().toDartString());
    if (decoded is! List) {
      return const [];
    }
    return [
      for (final item in decoded)
        switch ('$item') {
          'mlx' => RuntimeEngine.mlx,
          'coreml' => RuntimeEngine.coreml,
          'onnx' => RuntimeEngine.onnx,
          'litert' => RuntimeEngine.litert,
          _ => null,
        },
    ].whereType<RuntimeEngine>().toList(growable: false);
  } finally {
    native.freeStr(ptr);
  }
}

List<Accelerator> _acceleratorsJson(ffi.Pointer<ffi.Char> ptr) {
  if (ptr == ffi.nullptr) {
    return const [];
  }
  try {
    final decoded = jsonDecode(ptr.cast<Utf8>().toDartString());
    if (decoded is! List) {
      return const [];
    }
    return [
      for (final item in decoded)
        switch ('$item') {
          'ane' => Accelerator.ane,
          'gpu' => Accelerator.gpu,
          'npu' => Accelerator.npu,
          'cpu' => Accelerator.cpu,
          _ => null,
        },
    ].whereType<Accelerator>().toList(growable: false);
  } finally {
    native.freeStr(ptr);
  }
}
