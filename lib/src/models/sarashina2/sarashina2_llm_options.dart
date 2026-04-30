import 'dart:io';

import 'sarashina2.dart';

const _sarashina2LlmLayers = 24;
const _sarashina2BaseComponents = [
  'campplus',
  'speech_tokenizer_v2',
  'flow_encoder_fp32',
  'flow_decoder_estimator_fp32',
  'hift',
];

String resolveSarashina2LlmPrecision(
  Sarashina2TtsPaths paths,
  Map<String, Object?> backendOptions,
) {
  final requested = (backendOptions['sarashina2LlmPrecision'] ?? 'auto')
      .toString()
      .trim()
      .toLowerCase();
  switch (requested) {
    case '':
    case 'auto':
      // The direct FP16 export currently produces invalid logits on CUDA for
      // this LLM; keep it opt-in and prefer the validated BF16 path.
      if (_precisionFilesExist(paths, 'bf16')) {
        return 'bf16';
      }
      return 'fp32';
    case 'fp32':
    case 'float32':
      return 'fp32';
    case 'fp16':
    case 'float16':
      return 'fp16';
    case 'bf16':
    case 'bfloat16':
      return 'bf16';
    default:
      throw ArgumentError.value(
        requested,
        'sarashina2LlmPrecision',
        'expected auto, fp32, fp16, or bf16',
      );
  }
}

Map<String, String> sarashina2LlmPathOverrides(
  Sarashina2TtsPaths paths,
  String precision, {
  required String provider,
  required int deviceId,
  required Map<String, Object?> backendOptions,
}) {
  final sources = _componentSourcePathsWithOptionalDecodeHead(
    paths,
    precision,
    backendOptions: backendOptions,
  );
  if (!_backendBool(backendOptions, 'sarashina2LlmUseOptimizedGraphs', true)) {
    final needsOverrides =
        precision != 'fp32' || sources.containsKey('llm_decode_head');
    return needsOverrides ? sources : const {};
  }
  final overrides = <String, String>{};
  for (final entry in sources.entries) {
    final cache = _optimizedPath(
      paths: paths,
      component: entry.key,
      sourcePath: entry.value,
      precision: precision,
      provider: provider,
      deviceId: deviceId,
    );
    overrides[entry.key] = _freshCache(cache, entry.value)
        ? cache
        : entry.value;
  }
  return overrides;
}

Map<String, Map<String, Object?>> sarashina2LlmComponentBackendOptions({
  required Sarashina2TtsPaths paths,
  required String provider,
  required int deviceId,
  required String precision,
  required Map<String, Object?> backendOptions,
}) {
  final sources = _componentSourcePathsWithOptionalDecodeHead(
    paths,
    precision,
    backendOptions: backendOptions,
  );
  final useDeviceOutputs = _backendBool(
    backendOptions,
    'sarashina2LlmUseDeviceKvCache',
    true,
  );
  final useDeviceHidden = _backendBool(
    backendOptions,
    'sarashina2LlmUseDeviceHidden',
    true,
  );
  final useCudaGraph = _backendBool(
    backendOptions,
    'sarashina2LlmCudaGraph',
    false,
  );
  final usePrefillDeviceHidden = _backendBool(
    backendOptions,
    'sarashina2LlmUsePrefillDeviceHidden',
    _isCudaProvider(provider),
  );
  final prefillUsesLastHidden = _isPrefillLastHiddenPath(
    paths,
    precision,
    sources['llm_prefill'] ?? '',
  );
  final allDecodeOutputsOnDevice = useDeviceOutputs && useDeviceHidden;
  final decodeSyncOutputs = _backendBool(
    backendOptions,
    'sarashina2LlmDecodeSyncOutputs',
    !allDecodeOutputsOnDevice,
  );
  final decodeCacheBoundOutputs = _backendBool(
    backendOptions,
    'sarashina2LlmDecodeCacheBoundOutputs',
    allDecodeOutputsOnDevice,
  );
  final componentOptions = <String, Object?>{
    if (_backendBool(backendOptions, 'sarashina2LlmUseIoBinding', true))
      'useIoBinding': true,
    if (_backendBool(backendOptions, 'sarashina2LlmUseOutputViews', false))
      'useOutputViews': true,
    if (_backendBool(backendOptions, 'sarashina2LlmSharedPrepacking', true))
      'prepackedWeightsKey':
          'sarashina2:${paths.modelDir}:$provider:$deviceId:$precision',
    if (_backendBool(backendOptions, 'sarashina2LlmDisableMemPattern', true))
      'disableMemPattern': true,
    if (_backendBool(backendOptions, 'sarashina2LlmUseEnvAllocators', true))
      'useEnvAllocators': true,
    if (useCudaGraph) 'cudaEnableGraph': true,
    if (useCudaGraph)
      'cudaGraphId': _backendInt(backendOptions, 'sarashina2LlmCudaGraphId', 0),
    'executionMode': 'sequential',
  };
  final prefillOptions = <String, Object?>{
    ...componentOptions,
    if (useDeviceOutputs) 'useDeviceOutputs': true,
    if (useDeviceOutputs)
      'deviceOutputNames': _joinNames([
        if (useDeviceHidden && usePrefillDeviceHidden && prefillUsesLastHidden)
          'hidden',
        ..._llmKvOutputNames(),
      ]),
  };
  final decodeOptions = <String, Object?>{
    ...componentOptions,
    if (useDeviceOutputs) 'useDeviceOutputs': true,
    if (useDeviceOutputs && useDeviceHidden)
      'deviceOutputNames': _joinNames(_llmDecodeDeviceOutputNames()),
    if (useDeviceOutputs && !useDeviceHidden) 'cpuOutputNames': 'hidden',
    if (!_backendBool(backendOptions, 'sarashina2LlmDecodeSyncInputs', true))
      'syncBoundInputs': false,
    if (!decodeSyncOutputs) 'syncBoundOutputs': false,
    if (decodeCacheBoundOutputs) 'cacheBoundOutputs': true,
  };
  final decodeHeadOptions = <String, Object?>{
    ...componentOptions,
    if (useDeviceOutputs) 'useDeviceOutputs': true,
    if (useDeviceOutputs) 'deviceOutputNames': _joinNames(_llmKvOutputNames()),
    if (!_backendBool(backendOptions, 'sarashina2LlmDecodeSyncInputs', true))
      'syncBoundInputs': false,
    if (!_backendBool(backendOptions, 'sarashina2LlmDecodeSyncOutputs', true))
      'syncBoundOutputs': false,
    if (_backendBool(
      backendOptions,
      'sarashina2LlmDecodeCacheBoundOutputs',
      false,
    ))
      'cacheBoundOutputs': true,
  };
  final useOptimizedGraphs = _backendBool(
    backendOptions,
    'sarashina2LlmUseOptimizedGraphs',
    true,
  );
  return {
    for (final entry in sources.entries)
      entry.key: {
        ...switch (entry.key) {
          'llm_prefill' => prefillOptions,
          'llm_decode' => decodeOptions,
          'llm_decode_head' => decodeHeadOptions,
          _ => componentOptions,
        },
        if (useOptimizedGraphs)
          ..._optimizedOutputOption(
            paths: paths,
            component: entry.key,
            sourcePath: entry.value,
            precision: precision,
            provider: provider,
            deviceId: deviceId,
          ),
      },
  };
}

String sarashina2LlmDecodeHeadPath(Sarashina2TtsPaths paths, String precision) {
  return switch (precision) {
    'fp16' => paths.llmDecodeHeadFp16Onnx,
    'bf16' => paths.llmDecodeHeadBf16Onnx,
    _ => paths.llmDecodeHeadOnnx,
  };
}

bool sarashina2LlmDecodeHeadExists(
  Sarashina2TtsPaths paths,
  String precision,
) => File(sarashina2LlmDecodeHeadPath(paths, precision)).existsSync();

Map<String, Map<String, Object?>> sarashina2BaseComponentBackendOptions(
  Map<String, Object?> backendOptions,
) {
  if (!_backendBool(backendOptions, 'sarashina2UseOutputViews', false)) {
    return const {};
  }
  return {
    for (final name in _sarashina2BaseComponents)
      name: const {'useOutputViews': true},
  };
}

List<String> _llmKvOutputNames() => [
  for (var layer = 0; layer < _sarashina2LlmLayers; layer += 1) ...[
    'present_key_$layer',
    'present_value_$layer',
  ],
];

List<String> _llmDecodeDeviceOutputNames() => [
  'hidden',
  ..._llmKvOutputNames(),
];

String _joinNames(List<String> names) => names.join(',');

bool _precisionFilesExist(Sarashina2TtsPaths paths, String precision) {
  return _componentSourcePaths(
    paths,
    precision,
  ).values.every((path) => File(path).existsSync());
}

Map<String, String> _componentSourcePaths(
  Sarashina2TtsPaths paths,
  String precision, {
  Map<String, Object?> backendOptions = const {},
}) {
  final prefillPath = _prefillPath(paths, precision, backendOptions);
  return switch (precision) {
    'fp16' => {
      'llm_prefill': prefillPath,
      'llm_decode': paths.llmDecodeFp16Onnx,
      'llm_decoder_head': paths.llmDecoderHeadFp16Onnx,
    },
    'bf16' => {
      'llm_prefill': prefillPath,
      'llm_decode': paths.llmDecodeBf16Onnx,
      'llm_decoder_head': paths.llmDecoderHeadBf16Onnx,
    },
    _ => {
      'llm_prefill': prefillPath,
      'llm_decode': paths.llmDecodeOnnx,
      'llm_decoder_head': paths.llmDecoderHeadOnnx,
    },
  };
}

Map<String, String> _componentSourcePathsWithOptionalDecodeHead(
  Sarashina2TtsPaths paths,
  String precision, {
  Map<String, Object?> backendOptions = const {},
}) {
  final sources = _componentSourcePaths(
    paths,
    precision,
    backendOptions: backendOptions,
  );
  final decodeHeadPath = sarashina2LlmDecodeHeadPath(paths, precision);
  if (!File(decodeHeadPath).existsSync()) {
    return sources;
  }
  return {...sources, 'llm_decode_head': decodeHeadPath};
}

String _prefillPath(
  Sarashina2TtsPaths paths,
  String precision,
  Map<String, Object?> backendOptions,
) {
  final base = switch (precision) {
    'fp16' => paths.llmPrefillFp16Onnx,
    'bf16' => paths.llmPrefillBf16Onnx,
    _ => paths.llmPrefillOnnx,
  };
  if (!_backendBool(
    backendOptions,
    'sarashina2LlmUsePrefillLastHidden',
    true,
  )) {
    return base;
  }
  final last = switch (precision) {
    'fp16' => paths.llmPrefillLastFp16Onnx,
    'bf16' => paths.llmPrefillLastBf16Onnx,
    _ => paths.llmPrefillLastOnnx,
  };
  return File(last).existsSync() ? last : base;
}

bool _isPrefillLastHiddenPath(
  Sarashina2TtsPaths paths,
  String precision,
  String path,
) {
  final last = switch (precision) {
    'fp16' => paths.llmPrefillLastFp16Onnx,
    'bf16' => paths.llmPrefillLastBf16Onnx,
    _ => paths.llmPrefillLastOnnx,
  };
  return path == last;
}

Map<String, Object?> _optimizedOutputOption({
  required Sarashina2TtsPaths paths,
  required String component,
  required String sourcePath,
  required String precision,
  required String provider,
  required int deviceId,
}) {
  if (_usesExternalData(sourcePath)) {
    return const {};
  }
  final cache = _optimizedPath(
    paths: paths,
    component: component,
    sourcePath: sourcePath,
    precision: precision,
    provider: provider,
    deviceId: deviceId,
  );
  if (_freshCache(cache, sourcePath)) {
    return const {};
  }
  try {
    Directory(cache).parent.createSync(recursive: true);
  } catch (_) {
    return const {};
  }
  return {'optimizedModelFilePath': cache};
}

String _optimizedPath({
  required Sarashina2TtsPaths paths,
  required String component,
  required String sourcePath,
  required String precision,
  required String provider,
  required int deviceId,
}) {
  final source = File(sourcePath);
  final stamp = source.existsSync()
      ? '${source.lengthSync()}_${source.lastModifiedSync().millisecondsSinceEpoch}'
      : 'missing';
  return '${paths.modelDir}/.dart_inference_ort_cache/'
      '$component.$precision.${_safe(provider)}.$deviceId.$stamp.opt.onnx';
}

bool _freshCache(String cachePath, String sourcePath) {
  if (_usesExternalData(sourcePath)) {
    return false;
  }
  final cache = File(cachePath);
  final source = File(sourcePath);
  if (!cache.existsSync() || !source.existsSync() || cache.lengthSync() == 0) {
    return false;
  }
  return !cache.lastModifiedSync().isBefore(source.lastModifiedSync());
}

bool _usesExternalData(String modelPath) {
  return File('$modelPath.data').existsSync();
}

String _safe(String value) {
  final buffer = StringBuffer();
  for (final code in value.codeUnits) {
    final isDigit = code >= 48 && code <= 57;
    final isUpper = code >= 65 && code <= 90;
    final isLower = code >= 97 && code <= 122;
    buffer.write(
      isDigit || isUpper || isLower ? String.fromCharCode(code) : '_',
    );
  }
  return buffer.isEmpty ? 'provider' : buffer.toString();
}

bool _backendBool(
  Map<String, Object?> backendOptions,
  String key,
  bool fallback,
) {
  final value = backendOptions[key];
  if (value is bool) {
    return value;
  }
  if (value is String) {
    final normalized = value.trim().toLowerCase();
    if (normalized == '1' || normalized == 'true' || normalized == 'yes') {
      return true;
    }
    if (normalized == '0' || normalized == 'false' || normalized == 'no') {
      return false;
    }
  }
  return fallback;
}

bool _isCudaProvider(String provider) {
  final normalized = provider.trim().toLowerCase();
  return normalized == 'cuda' || normalized == 'cudaexecutionprovider';
}

int _backendInt(Map<String, Object?> backendOptions, String key, int fallback) {
  final value = backendOptions[key];
  if (value is int) {
    return value;
  }
  if (value is String) {
    return int.tryParse(value.trim()) ?? fallback;
  }
  return fallback;
}
