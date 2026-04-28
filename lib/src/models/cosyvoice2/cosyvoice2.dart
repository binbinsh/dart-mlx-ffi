import 'dart:io';
import 'dart:typed_data';

import '../../runtime/native_runtime.dart';
import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart';

final class CosyVoice2Paths {
  const CosyVoice2Paths({required this.modelDir});

  factory CosyVoice2Paths.fromUniFrontendRoot(String root) {
    final normalized = root.endsWith('/')
        ? root.substring(0, root.length - 1)
        : root;
    return CosyVoice2Paths(
      modelDir:
          '$normalized/src/ttsbackends/providers/cosyvoice2/models/CosyVoice2-0.5B',
    );
  }

  final String modelDir;

  String get campplusOnnx => '$modelDir/campplus.onnx';
  String get speechTokenizerOnnx => '$modelDir/speech_tokenizer_v2.onnx';
  String get flowDecoderEstimatorOnnx =>
      '$modelDir/flow.decoder.estimator.fp32.onnx';
  String get flowEncoderFp32Onnx => '$modelDir/flow.encoder.fp32.onnx';
  String get flowEncoderFp16Onnx => '$modelDir/flow.encoder.fp16.onnx';
  String get llmOnnx => '$modelDir/llm.onnx';
  String get flowOnnx => '$modelDir/flow.onnx';
  String get hiftOnnx => '$modelDir/hift.onnx';
  String get blankEnOnnx => '$modelDir/CosyVoice-BlankEN/model.onnx';
  String get vllmOnnx => '$modelDir/vllm/model.onnx';
  String get flowEncoderFp32Archive => '$modelDir/flow.encoder.fp32.zip';
  String get flowEncoderFp16Archive => '$modelDir/flow.encoder.fp16.zip';
  String get llmCheckpoint => '$modelDir/llm.pt';
  String get flowCheckpoint => '$modelDir/flow.pt';
  String get hiftCheckpoint => '$modelDir/hift.pt';
  String get blankEnSafetensors =>
      '$modelDir/CosyVoice-BlankEN/model.safetensors';
  String get vllmSafetensors => '$modelDir/vllm/model.safetensors';
  String get configPath => '$modelDir/cosyvoice2.yaml';

  List<CosyVoice2ComponentFile> componentFiles() => [
    CosyVoice2ComponentFile.onnx(
      name: 'campplus',
      role: 'speaker_embedding',
      path: campplusOnnx,
      requiredForSynthesis: true,
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'speech_tokenizer_v2',
      role: 'prompt_speech_tokenizer',
      path: speechTokenizerOnnx,
      requiredForSynthesis: true,
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'flow_decoder_estimator_fp32',
      role: 'diffusion_flow_decoder_estimator',
      path: flowDecoderEstimatorOnnx,
      requiredForSynthesis: true,
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'flow_encoder_fp32',
      role: 'flow_token_encoder',
      path: flowEncoderFp32Onnx,
      requiredForSynthesis: true,
      sourcePath: flowEncoderFp32Archive,
      sourceFormat: 'torchscript_zip',
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'flow_encoder_fp16',
      role: 'flow_token_encoder',
      path: flowEncoderFp16Onnx,
      requiredForSynthesis: false,
      sourcePath: flowEncoderFp16Archive,
      sourceFormat: 'torchscript_zip',
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'llm',
      role: 'semantic_speech_token_generator',
      path: llmOnnx,
      requiredForSynthesis: true,
      sourcePath: llmCheckpoint,
      sourceFormat: 'torch_checkpoint',
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'flow',
      role: 'flow_wrapper_checkpoint',
      path: flowOnnx,
      requiredForSynthesis: false,
      sourcePath: flowCheckpoint,
      sourceFormat: 'torch_checkpoint',
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'hift',
      role: 'vocoder',
      path: hiftOnnx,
      requiredForSynthesis: true,
      sourcePath: hiftCheckpoint,
      sourceFormat: 'torch_checkpoint',
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'blank_en_llm',
      role: 'text_llm_checkpoint',
      path: blankEnOnnx,
      requiredForSynthesis: false,
      sourcePath: blankEnSafetensors,
      sourceFormat: 'safetensors',
    ),
    CosyVoice2ComponentFile.onnx(
      name: 'vllm',
      role: 'vllm_checkpoint',
      path: vllmOnnx,
      requiredForSynthesis: false,
      sourcePath: vllmSafetensors,
      sourceFormat: 'safetensors',
    ),
  ];
}

final class CosyVoice2ComponentFile {
  const CosyVoice2ComponentFile({
    required this.name,
    required this.role,
    required this.path,
    required this.format,
    required this.requiredForSynthesis,
    required this.loadableByDartOnnx,
    this.sourcePath,
    this.sourceFormat,
  });

  const CosyVoice2ComponentFile.onnx({
    required this.name,
    required this.role,
    required this.path,
    required this.requiredForSynthesis,
    this.sourcePath,
    this.sourceFormat,
  }) : format = 'onnx',
       loadableByDartOnnx = true;

  final String name;
  final String role;
  final String path;
  final String format;
  final bool requiredForSynthesis;
  final bool loadableByDartOnnx;
  final String? sourcePath;
  final String? sourceFormat;

  File get file => File(path);

  bool get exists => file.existsSync();

  int? get sizeBytes => exists ? file.lengthSync() : null;

  File? get sourceFile => sourcePath == null ? null : File(sourcePath!);

  bool get sourceExists => sourceFile?.existsSync() ?? false;

  int? get sourceSizeBytes => sourceExists ? sourceFile!.lengthSync() : null;
}

final class CosyVoice2ComponentStatus {
  const CosyVoice2ComponentStatus({
    required this.file,
    required this.exists,
    required this.sizeBytes,
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

  final CosyVoice2ComponentFile file;
  final bool exists;
  final int? sizeBytes;
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
    'name': file.name,
    'role': file.role,
    'format': file.format,
    'path': file.path,
    'exists': exists,
    'sizeBytes': sizeBytes,
    'requiredForSynthesis': file.requiredForSynthesis,
    'loadableByDartOnnx': file.loadableByDartOnnx,
    if (file.sourcePath != null)
      'source': {
        'path': file.sourcePath,
        'format': file.sourceFormat,
        'exists': file.sourceExists,
        if (file.sourceSizeBytes != null) 'sizeBytes': file.sourceSizeBytes,
      },
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

final class CosyVoice2SmokeResult {
  const CosyVoice2SmokeResult({
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

final class CosyVoice2LoadedComponent {
  const CosyVoice2LoadedComponent({required this.file, required this.session});

  final CosyVoice2ComponentFile file;
  final DartOnnxSession session;

  String get name => file.name;

  String get selectedProvider => session.selectedProvider;

  DartOnnxResult run(Map<String, Object?> inputs) => session.run(inputs);
}

final class CosyVoice2PartialOnnxBundle {
  CosyVoice2PartialOnnxBundle._({
    required this.paths,
    required this.statuses,
    required this.loadedComponents,
  });

  final CosyVoice2Paths paths;
  final List<CosyVoice2ComponentStatus> statuses;
  final List<CosyVoice2LoadedComponent> loadedComponents;

  List<String> get loadedComponentNames => [
    for (final component in loadedComponents) component.name,
  ];

  CosyVoice2LoadedComponent? loadedComponent(String name) {
    for (final component in loadedComponents) {
      if (component.name == name) {
        return component;
      }
    }
    return null;
  }

  CosyVoice2LoadedComponent requireLoadedComponent(String name) {
    final component = loadedComponent(name);
    if (component == null) {
      throw StateError('CosyVoice2 ONNX component is not loaded: $name');
    }
    return component;
  }

  DartOnnxResult runComponent(String name, Map<String, Object?> inputs) {
    return requireLoadedComponent(name).run(inputs);
  }

  static CosyVoice2PartialOnnxBundle inspect({required CosyVoice2Paths paths}) {
    return CosyVoice2PartialOnnxBundle._(
      paths: paths,
      statuses: [
        for (final file in paths.componentFiles())
          CosyVoice2ComponentStatus(
            file: file,
            exists: file.exists,
            sizeBytes: file.sizeBytes,
          ),
      ],
      loadedComponents: const [],
    );
  }

  static CosyVoice2PartialOnnxBundle load({
    required CosyVoice2Paths paths,
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
    final statuses = <CosyVoice2ComponentStatus>[];
    final loaded = <CosyVoice2LoadedComponent>[];

    for (final file in paths.componentFiles()) {
      if (requested != null && !requested.contains(file.name)) {
        statuses.add(
          CosyVoice2ComponentStatus(
            file: file,
            exists: file.exists,
            sizeBytes: file.sizeBytes,
          ),
        );
        continue;
      }
      if (!file.exists || !file.loadableByDartOnnx) {
        statuses.add(
          CosyVoice2ComponentStatus(
            file: file,
            exists: file.exists,
            sizeBytes: file.sizeBytes,
          ),
        );
        continue;
      }
      final stopwatch = Stopwatch()..start();
      try {
        final session = DartOnnxSession.load(
          DartOnnxConfig(
            modelPath: file.path,
            id: 'cosyvoice2_${file.name}',
            family: 'cosyvoice2',
            provider: provider,
            deviceId: deviceId,
            requireProvider: requireProvider,
            numThreads: numThreads,
            backendOptions: backendOptions,
          ),
        );
        stopwatch.stop();
        final smokeResult = smoke
            ? _smokeComponent(
                file: file,
                session: session,
                warmupIterations: smokeWarmupIterations,
                iterations: smokeIterations,
              )
            : null;
        loaded.add(CosyVoice2LoadedComponent(file: file, session: session));
        statuses.add(
          CosyVoice2ComponentStatus(
            file: file,
            exists: true,
            sizeBytes: file.sizeBytes,
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
          CosyVoice2ComponentStatus(
            file: file,
            exists: true,
            sizeBytes: file.sizeBytes,
            loadElapsedMicroseconds: stopwatch.elapsedMicroseconds,
            error: '$error',
          ),
        );
      }
    }

    return CosyVoice2PartialOnnxBundle._(
      paths: paths,
      statuses: statuses,
      loadedComponents: loaded,
    );
  }

  bool get hasRequiredBlockedComponents => statuses.any(
    (status) =>
        status.file.requiredForSynthesis &&
        (!status.exists ||
            !status.file.loadableByDartOnnx ||
            status.error != null ||
            (status.smokeRan && status.smokeError != null)),
  );

  List<String> get blockers {
    final blockers = <String>[];
    for (final status in statuses) {
      if (!status.file.requiredForSynthesis) {
        continue;
      }
      if (!status.exists) {
        final source = status.file.sourcePath;
        final sourceFormat = status.file.sourceFormat;
        if (source != null && status.file.sourceExists) {
          blockers.add(
            '${status.file.name} ONNX is missing at ${status.file.path}; '
            'current source is $sourceFormat at $source.',
          );
        } else {
          blockers.add(
            '${status.file.name} ONNX is missing at ${status.file.path}',
          );
        }
      } else if (!status.file.loadableByDartOnnx) {
        blockers.add(
          '${status.file.name} is ${status.file.format}, not an ONNX graph.',
        );
      } else if (status.error != null) {
        blockers.add('${status.file.name} ONNX did not load: ${status.error}');
      } else if (status.smokeRan && status.smokeError != null) {
        blockers.add(
          '${status.file.name} ONNX smoke failed: ${status.smokeError}',
        );
      }
    }
    return blockers;
  }

  Map<String, Object?> toJson() => {
    'provider': 'cosyvoice2',
    'runtime': 'dart_inference_onnx_partial',
    'python': false,
    'modelDir': paths.modelDir,
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

CosyVoice2SmokeResult _smokeComponent({
  required CosyVoice2ComponentFile file,
  required DartOnnxSession session,
  required int warmupIterations,
  required int iterations,
}) {
  final warmups = warmupIterations < 0 ? 0 : warmupIterations;
  final runs = iterations < 1 ? 1 : iterations;
  final buffers = <NativeTensorBuffer>[];
  final stopwatch = Stopwatch();
  try {
    final inputs = _smokeInputs(file.name, session.diagnostics, buffers);
    for (var i = 0; i < warmups; i++) {
      final result = session.run(inputs);
      result.close();
    }

    var totalElapsedMicroseconds = 0;
    var minElapsedMicroseconds = 0;
    var maxElapsedMicroseconds = 0;
    var outputs = const <Map<String, Object?>>[];
    for (var i = 0; i < runs; i++) {
      stopwatch
        ..reset()
        ..start();
      final result = session.run(inputs);
      stopwatch.stop();
      final elapsed = stopwatch.elapsedMicroseconds;
      totalElapsedMicroseconds += elapsed;
      if (i == 0 || elapsed < minElapsedMicroseconds) {
        minElapsedMicroseconds = elapsed;
      }
      if (elapsed > maxElapsedMicroseconds) {
        maxElapsedMicroseconds = elapsed;
      }
      try {
        if (i == runs - 1) {
          outputs = _outputSummaries(result.outputs);
        }
      } finally {
        result.close();
      }
    }

    return CosyVoice2SmokeResult(
      elapsedMicroseconds: totalElapsedMicroseconds ~/ runs,
      warmupIterations: warmups,
      iterations: runs,
      totalElapsedMicroseconds: totalElapsedMicroseconds,
      minElapsedMicroseconds: minElapsedMicroseconds,
      maxElapsedMicroseconds: maxElapsedMicroseconds,
      outputs: outputs,
    );
  } catch (error) {
    stopwatch.stop();
    final elapsed = stopwatch.elapsedMicroseconds;
    return CosyVoice2SmokeResult(
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

Map<String, Object?> _smokeInputs(
  String componentName,
  Map<String, Object?> diagnostics,
  List<NativeTensorBuffer> buffers,
) {
  switch (componentName) {
    case 'campplus':
      return {
        'input': _float32Tensor([1, 16, 80], buffers),
      };
    case 'speech_tokenizer_v2':
      return {
        'feats': _float32Tensor([1, 128, 16], buffers),
        'feats_length': _int32Tensor([1], buffers, values: const [16]),
      };
    case 'flow_decoder_estimator_fp32':
      return {
        'x': _float32Tensor([2, 80, 16], buffers),
        'mask': _float32Tensor([2, 1, 16], buffers, value: 1.0),
        'mu': _float32Tensor([2, 80, 16], buffers),
        't': _float32Tensor([2], buffers, value: 0.5),
        'spks': _float32Tensor([2, 80], buffers),
        'cond': _float32Tensor([2, 80, 16], buffers),
      };
    default:
      return _smokeInputsFromMetadata(diagnostics, buffers);
  }
}

Map<String, Object?> _smokeInputsFromMetadata(
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
        'CosyVoice2 smoke only supports tensor inputs: ${item['name']}',
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
        'CosyVoice2 smoke input $name is too large for synthetic allocation: '
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

RuntimeTensor _float32Tensor(
  List<int> shape,
  List<NativeTensorBuffer> buffers, {
  double value = 0.0,
}) {
  final buffer = NativeTensorBuffer.float32(shape);
  buffer.asFloat32List().fillRange(0, buffer.byteLength ~/ 4, value);
  buffers.add(buffer);
  return buffer.tensor;
}

RuntimeTensor _int32Tensor(
  List<int> shape,
  List<NativeTensorBuffer> buffers, {
  required List<int> values,
}) {
  final buffer = NativeTensorBuffer.int32(shape);
  final target = buffer.asInt32List();
  target.fillRange(0, target.length, 0);
  final count = values.length < target.length ? values.length : target.length;
  target.setRange(0, count, values);
  buffers.add(buffer);
  return buffer.tensor;
}

List<Map<String, Object?>> _outputSummaries(Map<String, Object?> outputs) => [
  for (final entry in outputs.entries) _outputSummary(entry.key, entry.value),
];

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
