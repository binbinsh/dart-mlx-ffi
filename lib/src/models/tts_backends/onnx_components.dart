import 'dart:io';
import 'dart:typed_data';

import '../../runtime/native_runtime.dart';
import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart';
import 'catalog.dart';

final class TtsOnnxComponentStatus {
  const TtsOnnxComponentStatus({
    required this.capability,
    required this.target,
    required this.path,
    required this.exists,
    required this.sizeBytes,
    this.sources = const [],
    this.loaded = false,
    this.loadElapsedMicroseconds,
    this.selectedProvider,
    this.diagnostics = const {},
    this.error,
    this.smokeRan = false,
    this.smokeElapsedMicroseconds,
    this.smokeWarmupIterations,
    this.smokeIterations,
    this.smokeTotalElapsedMicroseconds,
    this.smokeMinElapsedMicroseconds,
    this.smokeMaxElapsedMicroseconds,
    this.smokeOutputs = const [],
    this.smokeError,
  });

  final TtsBackendCapability capability;
  final TtsBackendOnnxTarget target;
  final String path;
  final bool exists;
  final int? sizeBytes;
  final List<Map<String, Object?>> sources;
  final bool loaded;
  final int? loadElapsedMicroseconds;
  final String? selectedProvider;
  final Map<String, Object?> diagnostics;
  final String? error;
  final bool smokeRan;
  final int? smokeElapsedMicroseconds;
  final int? smokeWarmupIterations;
  final int? smokeIterations;
  final int? smokeTotalElapsedMicroseconds;
  final int? smokeMinElapsedMicroseconds;
  final int? smokeMaxElapsedMicroseconds;
  final List<Map<String, Object?>> smokeOutputs;
  final String? smokeError;

  Map<String, Object?> toJson() => {
    ...target.toJson(),
    'path': path,
    'exists': exists,
    'sizeBytes': sizeBytes,
    if (sources.isNotEmpty) 'sources': sources,
    'loaded': loaded,
    if (loadElapsedMicroseconds != null)
      'loadElapsedMs': loadElapsedMicroseconds! / 1000.0,
    if (selectedProvider != null) 'provider': selectedProvider,
    if (diagnostics.isNotEmpty) 'diagnostics': diagnostics,
    if (error != null) 'error': error,
    if (smokeRan) 'smokeRan': true,
    if (smokeElapsedMicroseconds != null)
      'smokeElapsedMs': smokeElapsedMicroseconds! / 1000.0,
    if (smokeWarmupIterations != null)
      'smokeWarmupIterations': smokeWarmupIterations,
    if (smokeIterations != null) 'smokeIterations': smokeIterations,
    if (smokeTotalElapsedMicroseconds != null)
      'smokeTotalElapsedMs': smokeTotalElapsedMicroseconds! / 1000.0,
    if (smokeMinElapsedMicroseconds != null)
      'smokeMinElapsedMs': smokeMinElapsedMicroseconds! / 1000.0,
    if (smokeMaxElapsedMicroseconds != null)
      'smokeMaxElapsedMs': smokeMaxElapsedMicroseconds! / 1000.0,
    if (smokeOutputs.isNotEmpty) 'smokeOutputs': smokeOutputs,
    if (smokeError != null) 'smokeError': smokeError,
  };
}

final class TtsOnnxComponentSmokeResult {
  const TtsOnnxComponentSmokeResult({
    required this.elapsedMicroseconds,
    required this.warmupIterations,
    required this.iterations,
    required this.totalElapsedMicroseconds,
    required this.minElapsedMicroseconds,
    required this.maxElapsedMicroseconds,
    required this.outputs,
    this.error,
  });

  final int elapsedMicroseconds;
  final int warmupIterations;
  final int iterations;
  final int totalElapsedMicroseconds;
  final int minElapsedMicroseconds;
  final int maxElapsedMicroseconds;
  final List<Map<String, Object?>> outputs;
  final String? error;
}

final class TtsLoadedOnnxComponent {
  const TtsLoadedOnnxComponent({
    required this.capability,
    required this.target,
    required this.path,
    required this.session,
  });

  final TtsBackendCapability capability;
  final TtsBackendOnnxTarget target;
  final String path;
  final DartOnnxSession session;

  String get name => target.name;

  String get selectedProvider => session.selectedProvider;

  DartOnnxResult run(Map<String, Object?> inputs) => session.run(inputs);
}

final class TtsOnnxComponentBundle {
  const TtsOnnxComponentBundle._({
    required this.capability,
    required this.providerDir,
    required this.statuses,
    required this.loadedComponents,
  });

  final TtsBackendCapability capability;
  final String? providerDir;
  final List<TtsOnnxComponentStatus> statuses;
  final List<TtsLoadedOnnxComponent> loadedComponents;

  static TtsOnnxComponentBundle inspect({
    required TtsBackendCapability capability,
    required String? providerDir,
  }) {
    return TtsOnnxComponentBundle._(
      capability: capability,
      providerDir: providerDir,
      statuses: [
        for (final target in capability.onnxTargets)
          _inspectTarget(capability, providerDir, target),
      ],
      loadedComponents: const [],
    );
  }

  static TtsOnnxComponentBundle load({
    required TtsBackendCapability capability,
    required String? providerDir,
    required String provider,
    required int deviceId,
    required bool requireProvider,
    required int numThreads,
    Map<String, Object?> backendOptions = const {},
    Iterable<String>? componentNames,
    bool smoke = false,
    int smokeWarmupIterations = 1,
    int smokeIterations = 5,
  }) {
    final requested = componentNames?.toSet();
    final statuses = <TtsOnnxComponentStatus>[];
    final loaded = <TtsLoadedOnnxComponent>[];
    for (final target in capability.onnxTargets) {
      final inspected = _inspectTarget(capability, providerDir, target);
      if (requested != null && !requested.contains(target.name)) {
        statuses.add(inspected);
        continue;
      }
      if (providerDir == null || !inspected.exists) {
        statuses.add(inspected);
        continue;
      }

      final stopwatch = Stopwatch()..start();
      try {
        final session = DartOnnxSession.load(
          DartOnnxConfig(
            modelPath: inspected.path,
            id: '${capability.provider}_${target.name}',
            family: capability.provider,
            provider: provider,
            deviceId: deviceId,
            requireProvider: requireProvider,
            numThreads: numThreads,
            backendOptions: backendOptions,
          ),
        );
        stopwatch.stop();
        final smokeResult = smoke
            ? smokeOnnxComponent(
                session: session,
                warmupIterations: smokeWarmupIterations,
                iterations: smokeIterations,
              )
            : null;
        loaded.add(
          TtsLoadedOnnxComponent(
            capability: capability,
            target: target,
            path: inspected.path,
            session: session,
          ),
        );
        statuses.add(
          TtsOnnxComponentStatus(
            capability: capability,
            target: target,
            path: inspected.path,
            exists: true,
            sizeBytes: inspected.sizeBytes,
            sources: inspected.sources,
            loaded: true,
            loadElapsedMicroseconds: stopwatch.elapsedMicroseconds,
            selectedProvider: session.selectedProvider,
            diagnostics: session.diagnostics,
            smokeRan: smokeResult != null,
            smokeElapsedMicroseconds: smokeResult?.elapsedMicroseconds,
            smokeWarmupIterations: smokeResult?.warmupIterations,
            smokeIterations: smokeResult?.iterations,
            smokeTotalElapsedMicroseconds:
                smokeResult?.totalElapsedMicroseconds,
            smokeMinElapsedMicroseconds: smokeResult?.minElapsedMicroseconds,
            smokeMaxElapsedMicroseconds: smokeResult?.maxElapsedMicroseconds,
            smokeOutputs: smokeResult?.outputs ?? const [],
            smokeError: smokeResult?.error,
          ),
        );
      } catch (error) {
        stopwatch.stop();
        statuses.add(
          TtsOnnxComponentStatus(
            capability: capability,
            target: target,
            path: inspected.path,
            exists: true,
            sizeBytes: inspected.sizeBytes,
            sources: inspected.sources,
            loadElapsedMicroseconds: stopwatch.elapsedMicroseconds,
            error: '$error',
          ),
        );
      }
    }
    return TtsOnnxComponentBundle._(
      capability: capability,
      providerDir: providerDir,
      statuses: statuses,
      loadedComponents: loaded,
    );
  }

  List<String> get loadedComponentNames => [
    for (final component in loadedComponents) component.name,
  ];

  TtsLoadedOnnxComponent? loadedComponent(String name) {
    for (final component in loadedComponents) {
      if (component.name == name) {
        return component;
      }
    }
    return null;
  }

  TtsLoadedOnnxComponent requireLoadedComponent(String name) {
    final component = loadedComponent(name);
    if (component == null) {
      throw StateError(
        '${capability.provider} ONNX component is not loaded: $name',
      );
    }
    return component;
  }

  DartOnnxResult runComponent(String name, Map<String, Object?> inputs) {
    return requireLoadedComponent(name).run(inputs);
  }

  bool get hasRequiredBlockedComponents => statuses.any(
    (status) =>
        status.target.requiredForSynthesis &&
        (!status.exists ||
            status.error != null ||
            (status.smokeRan && status.smokeError != null)),
  );

  List<String> get blockers {
    final blockers = <String>[];
    for (final status in statuses) {
      if (!status.target.requiredForSynthesis) {
        continue;
      }
      if (!status.exists) {
        blockers.add(_missingBlocker(status));
      } else if (status.error != null) {
        blockers.add(
          '${status.target.name} ONNX did not load: ${status.error}',
        );
      } else if (status.smokeRan && status.smokeError != null) {
        blockers.add(
          '${status.target.name} ONNX smoke failed: ${status.smokeError}',
        );
      }
    }
    return blockers;
  }

  Map<String, Object?> toJson() => {
    'provider': capability.provider,
    'runtime': 'dart_inference_onnx_components',
    'python': false,
    'providerDir': providerDir,
    'readyForSynthesis': !hasRequiredBlockedComponents,
    'loadedOnnxComponents': loadedComponentNames,
    'blockers': blockers,
    'components': [for (final status in statuses) status.toJson()],
  };

  void close() {
    for (final component in loadedComponents) {
      component.session.close();
    }
  }
}

TtsOnnxComponentSmokeResult smokeOnnxComponent({
  required DartOnnxSession session,
  int warmupIterations = 1,
  int iterations = 5,
}) {
  final warmups = warmupIterations < 0 ? 0 : warmupIterations;
  final runs = iterations < 1 ? 1 : iterations;
  final buffers = <NativeTensorBuffer>[];
  final failureTimer = Stopwatch()..start();
  try {
    final inputs = smokeInputsFromOnnxMetadata(session.diagnostics, buffers);
    for (var i = 0; i < warmups; i++) {
      final result = session.run(inputs);
      result.close();
    }

    final timer = Stopwatch();
    var totalElapsedMicroseconds = 0;
    var minElapsedMicroseconds = 0;
    var maxElapsedMicroseconds = 0;
    var outputs = const <Map<String, Object?>>[];
    for (var i = 0; i < runs; i++) {
      timer
        ..reset()
        ..start();
      final result = session.run(inputs);
      timer.stop();
      final elapsed = timer.elapsedMicroseconds;
      totalElapsedMicroseconds += elapsed;
      if (i == 0 || elapsed < minElapsedMicroseconds) {
        minElapsedMicroseconds = elapsed;
      }
      if (elapsed > maxElapsedMicroseconds) {
        maxElapsedMicroseconds = elapsed;
      }
      try {
        if (i == runs - 1) {
          outputs = outputSummaries(result.outputs);
        }
      } finally {
        result.close();
      }
    }

    return TtsOnnxComponentSmokeResult(
      elapsedMicroseconds: totalElapsedMicroseconds ~/ runs,
      warmupIterations: warmups,
      iterations: runs,
      totalElapsedMicroseconds: totalElapsedMicroseconds,
      minElapsedMicroseconds: minElapsedMicroseconds,
      maxElapsedMicroseconds: maxElapsedMicroseconds,
      outputs: outputs,
    );
  } catch (error) {
    failureTimer.stop();
    final elapsed = failureTimer.elapsedMicroseconds;
    return TtsOnnxComponentSmokeResult(
      elapsedMicroseconds: elapsed,
      warmupIterations: warmups,
      iterations: runs,
      totalElapsedMicroseconds: elapsed,
      minElapsedMicroseconds: elapsed,
      maxElapsedMicroseconds: elapsed,
      outputs: const [],
      error: '$error',
    );
  } finally {
    for (final buffer in buffers.reversed) {
      buffer.close();
    }
  }
}

Map<String, Object?> smokeInputsFromOnnxMetadata(
  Map<String, Object?> diagnostics,
  List<NativeTensorBuffer> buffers,
) {
  final metadata = diagnostics['input_metadata'];
  if (metadata is! List || metadata.isEmpty) {
    throw UnsupportedError('ONNX input metadata is unavailable.');
  }
  final inputs = <String, Object?>{};
  for (final raw in metadata) {
    if (raw is! Map) {
      continue;
    }
    final item = raw.map((key, value) => MapEntry(key.toString(), value));
    if ((item['onnx_type'] ?? 'tensor').toString() != 'tensor') {
      throw UnsupportedError(
        'TTS ONNX smoke only supports tensor inputs: ${item['name']}',
      );
    }
    final name = (item['name'] ?? '').toString();
    if (name.isEmpty) {
      throw UnsupportedError('ONNX input metadata is missing an input name.');
    }
    final dtype = _metadataDtype(item);
    final shape = _metadataShape(item);
    _checkSmokeElementCount(name, shape);
    inputs[name] = _metadataTensor(name, dtype, shape, buffers);
  }
  if (inputs.isEmpty) {
    throw UnsupportedError(
      'ONNX input metadata did not contain tensor inputs.',
    );
  }
  return inputs;
}

List<Map<String, Object?>> outputSummaries(Map<String, Object?> outputs) => [
  for (final entry in outputs.entries) _outputSummary(entry.key, entry.value),
];

TtsOnnxComponentStatus _inspectTarget(
  TtsBackendCapability capability,
  String? providerDir,
  TtsBackendOnnxTarget target,
) {
  final path = providerDir == null
      ? target.path
      : '$providerDir/${target.path}';
  final file = File(path);
  final exists = file.existsSync();
  return TtsOnnxComponentStatus(
    capability: capability,
    target: target,
    path: path,
    exists: exists,
    sizeBytes: exists ? file.lengthSync() : null,
    sources: _targetSources(capability, providerDir, target),
  );
}

List<Map<String, Object?>> _targetSources(
  TtsBackendCapability capability,
  String? providerDir,
  TtsBackendOnnxTarget target,
) {
  if (target.sourceNames.isEmpty) {
    return const [];
  }
  return [
    for (final source in capability.sourceAssets)
      if (target.sourceNames.contains(source.name))
        _sourceStatus(providerDir, source),
  ];
}

Map<String, Object?> _sourceStatus(
  String? providerDir,
  TtsBackendSourceAsset source,
) {
  final path = providerDir == null
      ? null
      : _findSourcePath(providerDir, source);
  if (providerDir != null && path != null) {
    final file = File('$providerDir/$path');
    final exists = file.existsSync();
    return {
      ...source.toJson(),
      'locator': source.locator,
      'resolvedPath': path,
      'exists': exists,
      if (exists) 'sizeBytes': file.lengthSync(),
    };
  }
  return {...source.toJson(), 'locator': source.locator, 'exists': false};
}

String? _findSourcePath(String providerDir, TtsBackendSourceAsset source) {
  final path = source.path;
  if (path != null && File('$providerDir/$path').existsSync()) {
    return path;
  }
  final basename = source.basename;
  if (basename == null) {
    return null;
  }
  final root = Directory(providerDir);
  if (!root.existsSync()) {
    return null;
  }
  for (final entity in root.listSync(recursive: true).whereType<File>()) {
    final relative = entity.path.substring(providerDir.length + 1);
    if (!relative.startsWith('models/')) {
      continue;
    }
    if (_basename(relative) == basename) {
      return relative;
    }
  }
  return null;
}

String _missingBlocker(TtsOnnxComponentStatus status) {
  final existingSource = status.sources.cast<Map<String, Object?>>().where(
    (source) => source['exists'] == true,
  );
  if (existingSource.isNotEmpty) {
    final source = existingSource.first;
    return '${status.target.name} ONNX is missing at ${status.path}; '
        'current source is ${source['format']} at '
        '${source['resolvedPath'] ?? source['locator']}.';
  }
  return '${status.target.name} ONNX is missing at ${status.path}';
}

String _basename(String path) {
  final slash = path.lastIndexOf('/');
  return slash < 0 ? path : path.substring(slash + 1);
}

RuntimeTensorDataType _metadataDtype(Map<String, Object?> item) {
  final dtype = (item['dtype'] ?? '').toString().toLowerCase();
  switch (dtype) {
    case 'float':
    case 'float32':
      return RuntimeTensorDataType.float32;
    case 'double':
    case 'float64':
      return RuntimeTensorDataType.float64;
    case 'float16':
      return RuntimeTensorDataType.float16;
    case 'int32':
      return RuntimeTensorDataType.int32;
    case 'int64':
      return RuntimeTensorDataType.int64;
    case 'uint8':
      return RuntimeTensorDataType.uint8;
    case 'bool':
    case 'boolean':
      return RuntimeTensorDataType.boolean;
  }
  final dtypeId = int.tryParse('${item['dtype_id'] ?? ''}') ?? 0;
  switch (dtypeId) {
    case 1:
      return RuntimeTensorDataType.float32;
    case 2:
      return RuntimeTensorDataType.int32;
    case 3:
      return RuntimeTensorDataType.int64;
    case 4:
      return RuntimeTensorDataType.uint8;
    case 5:
      return RuntimeTensorDataType.float64;
    case 6:
      return RuntimeTensorDataType.float16;
    case 7:
      return RuntimeTensorDataType.boolean;
  }
  throw UnsupportedError('Unsupported ONNX input dtype: ${item['dtype']}');
}

List<int> _metadataShape(Map<String, Object?> item) {
  final rawShape = item['shape'];
  if (rawShape is! List) {
    return const [];
  }
  final signature = item['shape_signature'] is List
      ? item['shape_signature'] as List
      : const <Object?>[];
  final name = (item['name'] ?? '').toString();
  return [
    for (var i = 0; i < rawShape.length; i++)
      _metadataDim(
        int.tryParse('${rawShape[i]}') ?? -1,
        i < signature.length ? signature[i].toString() : '',
        inputName: name,
        axis: i,
      ),
  ];
}

int _metadataDim(
  int dim,
  String signature, {
  required String inputName,
  required int axis,
}) {
  if (dim > 0) {
    return dim;
  }
  final value = '${inputName}_$signature'.toLowerCase();
  if (value.contains('batch')) {
    return 1;
  }
  if (value.contains('seq') ||
      value.contains('token') ||
      value.contains('length') ||
      value.contains('frame') ||
      value.contains('time')) {
    return 16;
  }
  if (axis == 0) {
    return 1;
  }
  return 1;
}

void _checkSmokeElementCount(String name, List<int> shape) {
  var elements = 1;
  for (final dim in shape) {
    elements *= dim;
    if (elements > 4 * 1024 * 1024) {
      throw UnsupportedError(
        'TTS ONNX smoke input $name is too large for synthetic allocation: '
        '$shape',
      );
    }
  }
}

RuntimeTensor _metadataTensor(
  String name,
  RuntimeTensorDataType dtype,
  List<int> shape,
  List<NativeTensorBuffer> buffers,
) {
  final buffer = NativeTensorBuffer.allocate(dtype: dtype, shape: shape);
  buffers.add(buffer);
  final normalizedName = name.toLowerCase();
  switch (dtype) {
    case RuntimeTensorDataType.float32:
      buffer.asFloat32List().fillRange(
        0,
        buffer.byteLength ~/ 4,
        _floatFillValue(normalizedName),
      );
      break;
    case RuntimeTensorDataType.float64:
      buffer.asFloat64List().fillRange(
        0,
        buffer.byteLength ~/ 8,
        _floatFillValue(normalizedName),
      );
      break;
    case RuntimeTensorDataType.float16:
      _fillFloat16(buffer, _floatFillValue(normalizedName));
      break;
    case RuntimeTensorDataType.int32:
      buffer.asInt32List().fillRange(
        0,
        buffer.byteLength ~/ 4,
        _intFillValue(normalizedName),
      );
      break;
    case RuntimeTensorDataType.int64:
      buffer.asInt64List().fillRange(
        0,
        buffer.byteLength ~/ 8,
        _intFillValue(normalizedName),
      );
      break;
    case RuntimeTensorDataType.uint8:
    case RuntimeTensorDataType.boolean:
      buffer.asUint8List().fillRange(
        0,
        buffer.byteLength,
        _boolLikeFillValue(normalizedName),
      );
      break;
  }
  return buffer.tensor;
}

double _floatFillValue(String name) {
  if (name.contains('mask')) {
    return 1.0;
  }
  if (name == 't' || name.endsWith('_t')) {
    return 0.5;
  }
  if (name.contains('speed')) {
    return 1.0;
  }
  return 0.0;
}

int _intFillValue(String name) {
  if (name.contains('length') || name.endsWith('_len')) {
    return 16;
  }
  if (name.contains('mask')) {
    return 1;
  }
  return 0;
}

int _boolLikeFillValue(String name) => name.contains('mask') ? 1 : 0;

void _fillFloat16(NativeTensorBuffer buffer, double value) {
  final bits = value == 1.0
      ? 0x3c00
      : value == 0.5
      ? 0x3800
      : 0;
  final bytes = buffer.asUint8List();
  for (var i = 0; i + 1 < bytes.length; i += 2) {
    bytes[i] = bits & 0xff;
    bytes[i + 1] = (bits >> 8) & 0xff;
  }
}

Map<String, Object?> _outputSummary(String name, Object? value) {
  if (value is RuntimeTensor) {
    return {
      'name': name,
      'dtype': value.dtype.name,
      'shape': value.shape,
      'byteLength': value.bytes.lengthInBytes,
    };
  }
  if (value is TypedData) {
    return {
      'name': name,
      'dartType': value.runtimeType.toString(),
      'byteLength': value.lengthInBytes,
    };
  }
  return {'name': name, 'dartType': value.runtimeType.toString()};
}
