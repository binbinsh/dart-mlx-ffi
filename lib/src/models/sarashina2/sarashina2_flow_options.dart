import 'dart:io';

import 'sarashina2.dart';

bool sarashina2UseFusedFlowStep(
  Sarashina2TtsPaths paths,
  Map<String, Object?> backendOptions,
) {
  if (!_backendBool(backendOptions, 'sarashina2UseFusedFlowStep', true)) {
    return false;
  }
  return File(paths.flowDecoderStepOnnx).existsSync();
}

bool sarashina2UseFusedFlowLoop(
  Sarashina2TtsPaths paths,
  Map<String, Object?> backendOptions,
) {
  if (!_backendBool(backendOptions, 'sarashina2UseFusedFlowLoop', false)) {
    return false;
  }
  return File(paths.flowDecoderLoopOnnx).existsSync();
}

String? resolveSarashina2TensorRtFlowStepContextPath(
  Sarashina2TtsPaths paths,
  Map<String, Object?> backendOptions,
) {
  final explicit = _backendString(
    backendOptions,
    'sarashina2TensorRtFlowStepContextPath',
    '',
  );
  if (explicit.isNotEmpty) {
    if (!File(explicit).existsSync()) {
      throw StateError(
        'Missing sarashina2TensorRtFlowStepContextPath: $explicit',
      );
    }
    return explicit;
  }
  if (!_backendBool(
    backendOptions,
    'sarashina2TensorRtUseFlowStepContext',
    true,
  )) {
    return null;
  }
  for (final path in _defaultFlowStepContextPaths(paths)) {
    if (File(path).existsSync()) {
      return path;
    }
  }
  return null;
}

bool resolveSarashina2UseDeviceFlowLoop(
  Map<String, Object?> backendOptions, {
  String provider = '',
}) {
  return _backendBool(backendOptions, 'sarashina2UseDeviceFlowLoop', false);
}

String resolveSarashina2FlowStepPrecision(
  Sarashina2TtsPaths paths,
  Map<String, Object?> backendOptions, {
  String provider = '',
}) {
  final requested = (backendOptions['sarashina2FlowStepPrecision'] ?? 'auto')
      .toString()
      .trim()
      .toLowerCase();
  switch (requested) {
    case '':
    case 'auto':
      if (_isCudaProvider(provider) &&
          File(paths.flowDecoderStepFp16Onnx).existsSync()) {
        return 'fp16';
      }
      return 'fp32';
    case 'fp32':
    case 'float32':
      return 'fp32';
    case 'fp16':
    case 'float16':
      if (!File(paths.flowDecoderStepFp16Onnx).existsSync()) {
        throw StateError('Missing ${paths.flowDecoderStepFp16Onnx}');
      }
      return 'fp16';
    default:
      throw ArgumentError.value(
        requested,
        'sarashina2FlowStepPrecision',
        'expected fp32 or fp16',
      );
  }
}

bool _isCudaProvider(String provider) {
  final normalized = provider.trim().toLowerCase();
  return normalized == 'cuda' || normalized == 'cudaexecutionprovider';
}

int resolveSarashina2FlowSteps(Map<String, Object?> backendOptions) {
  final value = backendOptions['sarashina2FlowSteps'];
  if (value == null) {
    return 10;
  }
  final steps = switch (value) {
    int() => value,
    String() => int.parse(value.trim()),
    _ => throw ArgumentError.value(value, 'sarashina2FlowSteps'),
  };
  if (steps < 1 || steps > 10) {
    throw RangeError.range(steps, 1, 10, 'sarashina2FlowSteps');
  }
  return steps;
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

String _backendString(
  Map<String, Object?> backendOptions,
  String key,
  String fallback,
) {
  final value = backendOptions[key];
  if (value is String) {
    final normalized = value.trim();
    return normalized.isEmpty ? fallback : normalized;
  }
  return fallback;
}

List<String> _defaultFlowStepContextPaths(Sarashina2TtsPaths paths) {
  final out = <String>[paths.flowDecoderStepTensorRtContextOnnx];
  final root = _uniFrontendRoot(paths.modelDir);
  if (root != null) {
    out.add('$root/artifacts/runtime/sarashina2/tensorrt/flow_step_ctx.onnx');
  }
  return out;
}

String? _uniFrontendRoot(String modelDir) {
  const marker = '/src/ttsbackends/';
  final index = modelDir.indexOf(marker);
  return index > 0 ? modelDir.substring(0, index) : null;
}
