/// Model-level runtime abstractions and resolution.
library;

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
    final root = rootPath.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final path = artifact.path.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    ffi.Pointer<ffi.Char> result = ffi.nullptr;
    try {
      result = native.artifactPath(root, path);
      if (result == ffi.nullptr) {
        return artifact.path;
      }
      final resolved = result.cast<Utf8>().toDartString();
      return resolved.isEmpty ? artifact.path : resolved;
    } finally {
      if (result != ffi.nullptr) {
        native.freeStr(result);
      }
      calloc
        ..free(root)
        ..free(path);
    }
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
    final artifacts = _ArtifactArena(spec);
    final registered = _IntArena(_runtimes.keys.map(_engineId));
    try {
      final engineId = native.fallback(
        _platformId(platform),
        registered.pointer,
        registered.count,
        artifacts.pointer,
        artifacts.count,
      );
      if (engineId < 0) {
        return null;
      }
      final engine = _engineById(engineId);
      final artifact = spec.platformArtifacts[engine];
      if (artifact == null) {
        return null;
      }
      return RuntimeResolution(
        platform: platform,
        engine: engine,
        artifact: artifact,
        accelerators: options.prefer,
        fallbackReason: 'Selected engine has no registered runtime backend.',
      );
    } finally {
      registered.close();
      artifacts.close();
    }
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
    final artifacts = _ArtifactArena(spec);
    final result = calloc<native.ResolveResultAbi>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    final modelId = _nativeText(spec.id);
    try {
      final status = native.resolve(
        modelId,
        _platformId(platform),
        requested == null ? -1 : _engineId(requested),
        options.allowFallback ? 1 : 0,
        _preferMask(options.prefer),
        artifacts.pointer,
        artifacts.count,
        result,
        error,
      );
      if (status != 0) {
        throw StateError(
          _takeError(error, 'Native runtime resolver failed for ${spec.id}.'),
        );
      }
      final resolved = result.ref;
      final engine = _engineById(resolved.engine);
      final artifact = spec.platformArtifacts[engine];
      if (artifact == null) {
        throw StateError(
          'Native runtime resolver selected missing ${engine.name} artifact.',
        );
      }
      final fallbackEngine = resolved.fallbackEngine >= 0
          ? _engineById(resolved.fallbackEngine)
          : null;
      return RuntimeResolution(
        platform: platform,
        engine: engine,
        artifact: artifact,
        accelerators: options.prefer.isNotEmpty
            ? options.prefer
            : _accelerators(resolved.accelMask),
        fallbackReason: fallbackEngine == null
            ? null
            : 'Requested ${fallbackEngine.name} has no compatible artifact.',
      );
    } finally {
      artifacts.close();
      calloc
        ..free(modelId)
        ..free(result)
        ..free(error);
    }
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

RuntimeEngine _engineById(int id) => switch (id) {
  0 => RuntimeEngine.mlx,
  1 => RuntimeEngine.coreml,
  2 => RuntimeEngine.onnx,
  3 => RuntimeEngine.litert,
  _ => throw StateError('Unsupported native runtime engine id: $id'),
};

int _preferMask(List<Accelerator> values) {
  var mask = 0;
  for (final value in values) {
    mask |= switch (value) {
      Accelerator.cpu => 1,
      Accelerator.gpu => 2,
      Accelerator.ane => 4,
      Accelerator.npu => 8,
    };
  }
  return mask;
}

List<Accelerator> _accelerators(int mask) => [
  if ((mask & 4) != 0) Accelerator.ane,
  if ((mask & 2) != 0) Accelerator.gpu,
  if ((mask & 8) != 0) Accelerator.npu,
  if ((mask & 1) != 0) Accelerator.cpu,
];

ffi.Pointer<ffi.Char> _nativeText(String value) {
  return value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
}

String _takeError(ffi.Pointer<ffi.Pointer<ffi.Char>> error, String fallback) {
  final value = error.value;
  if (value == ffi.nullptr) {
    return fallback;
  }
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
  }
}

final class _ArtifactArena {
  _ArtifactArena(ModelSpec spec)
    : count = spec.platformArtifacts.length,
      pointer = spec.platformArtifacts.isEmpty
          ? ffi.nullptr
          : calloc<native.ResolveArtifactAbi>(spec.platformArtifacts.length) {
    var index = 0;
    for (final entry in spec.platformArtifacts.entries) {
      final artifact = entry.value;
      pointer[index]
        ..engine = _engineId(entry.key)
        ..path = _add(artifact.path)
        ..format = _add(artifact.format ?? '')
        ..targetPlatforms = _add(artifact.targetPlatforms.join('\n'));
      index += 1;
    }
  }

  final ffi.Pointer<native.ResolveArtifactAbi> pointer;
  final int count;
  final List<ffi.Pointer<ffi.Char>> _strings = [];

  ffi.Pointer<ffi.Char> _add(String value) {
    final pointer = _nativeText(value);
    _strings.add(pointer);
    return pointer;
  }

  void close() {
    for (final value in _strings) {
      calloc.free(value);
    }
    _strings.clear();
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

final class _IntArena {
  _IntArena(Iterable<int> values) {
    final list = values.toList(growable: false);
    count = list.length;
    pointer = count == 0 ? ffi.nullptr : calloc<ffi.Int32>(count);
    if (count > 0) {
      pointer.asTypedList(count).setAll(0, list);
    }
  }

  late final ffi.Pointer<ffi.Int32> pointer;
  late final int count;

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}
