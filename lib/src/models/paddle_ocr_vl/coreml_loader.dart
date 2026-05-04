/// CoreML session-loader plumbing shared by the PaddleOCR-VL hybrid runner
/// (commit #8) and any future CoreML-backed stage in this package.
///
/// Extracted from the legacy `coreml_runner.dart` (commit #11 of the hybrid
/// OCR refactor, issue #1) when the 4-stage `PaddleOcrVlCoremlRunner` was
/// retired. The runner is gone; the loader/session abstractions remain
/// because they are how the hybrid runner opens its single
/// `vision_embed.mlpackage` stage.
///
/// This file deliberately exposes nothing PaddleOCR-specific — it is a thin
/// facade over `NativeModelRuntime(RuntimeEngine.coreml)` keyed by a stage
/// name + compute units.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../shared/model_spec.dart';
import '../shared/runtime_metadata.dart';
import '../../runtime/coreml_runtime.dart' as coreml_runtime;
import '../../runtime/native_runtime.dart';
import '../../runtime/runtime.dart';
import 'coreml_pipeline_manifest.dart' show CoremlComputeUnits;

/// Facade over a CoreML mlpackage session. One stage per instance.
///
/// Inputs are name → tensor; tensors are
/// `Float32List` / `Int32List` / `(shape, data)` records — the exact
/// envelope is defined by the underlying `NativeModelRuntime` engine.
abstract interface class CoremlSession {
  /// Run one inference. See class doc for the tensor envelope.
  Map<String, Object> predict(Map<String, Object> inputs);

  /// Release the underlying MLModel + MLState.
  void close();

  /// Drop the cached MLState so the next `predict` materialises a fresh
  /// one. No-op for non-stateful sessions.
  void resetState();
}

/// Loader that opens a single CoreML stage. Implementations are responsible
/// for materialising whatever runtime-spec file the engine needs.
abstract class CoremlLoader {
  CoremlSession loadStage({
    required String packagePath,
    required CoremlComputeUnits computeUnits,
    required bool stateful,
  });
}

/// Hook so callers can be unit-tested with a fake loader. Both the legacy
/// runner (now deleted) and the hybrid runner honour this override.
CoremlLoader? testCoremlLoaderOverride;

/// Public factory for the production CoreML loader used by
/// `PaddleOcrVlHybridRunner` (commit #8) and any other in-tree consumer.
/// Callers that want to inject a fake should set [testCoremlLoaderOverride].
CoremlLoader defaultCoremlLoader() => const _NativeCoremlLoader();

final class _NativeCoremlLoader implements CoremlLoader {
  const _NativeCoremlLoader();

  @override
  CoremlSession loadStage({
    required String packagePath,
    required CoremlComputeUnits computeUnits,
    required bool stateful,
  }) {
    final artifactPath = _stagePipelineSpecPath(
      packagePath: packagePath,
      computeUnits: computeUnits,
      stateful: stateful,
    );
    final artifact = RuntimeArtifact(
      engine: RuntimeEngine.coreml,
      path: artifactPath,
      format: 'coreml-stage-pipeline',
      targetPlatforms: const ['ios', 'macos'],
      accelerators: const [Accelerator.ane, Accelerator.gpu, Accelerator.cpu],
      metadata: {
        'modelId': 'paddle_ocr_vl',
        'stateful': stateful,
        'computeUnits': _computeUnitsOption(computeUnits),
      },
    );
    final spec = ModelSpec(
      id: 'paddle_ocr_vl_coreml_stage',
      family: 'PaddleOCR-VL',
      modalities: const [ModelModality.visionLanguage],
      description: 'PaddleOCR-VL CoreML stage',
      requiredFiles: const [],
      platformArtifacts: {RuntimeEngine.coreml: artifact},
    );
    final session = NativeModelRuntime(RuntimeEngine.coreml).load(
      ModelBundle(spec: spec, rootPath: '', artifact: artifact),
      RuntimeOptions(
        engine: RuntimeEngine.coreml,
        allowFallback: false,
        prefer: const [Accelerator.ane, Accelerator.gpu, Accelerator.cpu],
        backendOptions: {
          'coremlComputeUnits': _computeUnitsOption(computeUnits),
        },
      ),
    );
    return _RuntimeCoremlSession(session, stateful: stateful);
  }
}

final class _RuntimeCoremlSession implements CoremlSession {
  _RuntimeCoremlSession(this._session, {required this.stateful});

  final ModelSession _session;
  final bool stateful;

  @override
  Map<String, Object> predict(Map<String, Object> inputs) {
    final outputs = _session.run(ModelInputs(_runtimeInputs(inputs)));
    try {
      return outputs.values.map(
        (key, value) => MapEntry(key, _runtimeOutput(key, value)),
      );
    } finally {
      outputs.close();
    }
  }

  @override
  void resetState() {
    if (!stateful) return;
    final session = _session;
    if (session is coreml_runtime.CoremlStateResettable) {
      (session as coreml_runtime.CoremlStateResettable).resetCoremlState();
      return;
    }
    throw StateError('CoreML session does not support state reset.');
  }

  @override
  void close() {
    _session.close();
  }

  static Map<String, Object?> _runtimeInputs(Map<String, Object> inputs) {
    return inputs.map(
      (name, value) => MapEntry(name, _runtimeInput(name, value)),
    );
  }

  static Object _runtimeInput(String name, Object value) {
    if (value is (List<int>, Float32List)) {
      return RuntimeTensor.float32(value.$1, value.$2);
    }
    if (value is (List<int>, Int32List)) {
      return RuntimeTensor.int32(value.$1, value.$2);
    }
    if (value is (List<int>, Int64List)) {
      return RuntimeTensor.int64(value.$1, value.$2);
    }
    if (value is (List<int>, Uint8List)) {
      return RuntimeTensor.uint8(value.$1, value.$2);
    }
    if (value is (List<int>, Float64List)) {
      return RuntimeTensor.float64(value.$1, value.$2);
    }
    if (value is TypedData) return value;
    throw ArgumentError.value(value, name, 'Unsupported CoreML input tensor');
  }

  static Object _runtimeOutput(String name, Object? value) {
    if (value is! RuntimeTensor) {
      throw StateError('CoreML output "$name" is not a runtime tensor.');
    }
    return switch (value.dtype) {
      RuntimeTensorDataType.float32 => (
        List<int>.unmodifiable(value.shape),
        Float32List.fromList(value.asFloat32List()),
      ),
      RuntimeTensorDataType.int32 => (
        List<int>.unmodifiable(value.shape),
        Int32List.fromList(value.asInt32List()),
      ),
      RuntimeTensorDataType.int64 => (
        List<int>.unmodifiable(value.shape),
        Int64List.fromList(value.asInt64List()),
      ),
      RuntimeTensorDataType.float64 => (
        List<int>.unmodifiable(value.shape),
        Float64List.fromList(value.asFloat64List()),
      ),
      RuntimeTensorDataType.uint8 || RuntimeTensorDataType.boolean => (
        List<int>.unmodifiable(value.shape),
        Uint8List.fromList(value.asUint8List()),
      ),
      RuntimeTensorDataType.float16 => throw StateError(
        'CoreML output "$name" uses float16; PaddleOCR CoreML loader expects '
        'float32 outputs.',
      ),
    };
  }
}

String _computeUnitsOption(CoremlComputeUnits units) => switch (units) {
  CoremlComputeUnits.cpuOnly => 'cpuOnly',
  CoremlComputeUnits.cpuAndGpu => 'cpuAndGPU',
  CoremlComputeUnits.cpuAndNeuralEngine => 'cpuAndNeuralEngine',
  CoremlComputeUnits.all => 'all',
};

String _pipelineComputeUnits(CoremlComputeUnits units) => switch (units) {
  CoremlComputeUnits.cpuOnly => 'cpu_only',
  CoremlComputeUnits.cpuAndGpu => 'cpu_and_gpu',
  CoremlComputeUnits.cpuAndNeuralEngine => 'cpu_and_neural_engine',
  CoremlComputeUnits.all => 'all',
};

String _stagePipelineSpecPath({
  required String packagePath,
  required CoremlComputeUnits computeUnits,
  required bool stateful,
}) {
  final stageName = _stageName(packagePath);
  final unitName = _pipelineComputeUnits(computeUnits);
  final file = File(
    '${Directory.systemTemp.path}/dart_inference_${stageName}_'
    '${unitName}_${stateful ? "stateful" : "stateless"}_'
    '${packagePath.hashCode.toUnsigned(32)}.coreml_pipeline.json',
  );
  if (!file.existsSync()) {
    file.writeAsStringSync(
      jsonEncode({
        'format': 'dart_inference.coreml_pipeline.v1',
        'stages': [
          {
            'name': stageName,
            'model': packagePath,
            'compute_units': unitName,
            'stateful': stateful,
          },
        ],
      }),
      flush: true,
    );
  }
  return file.path;
}

String _stageName(String packagePath) {
  final name = packagePath.split(Platform.pathSeparator).last;
  if (name.endsWith('.mlpackage')) {
    return name.substring(0, name.length - '.mlpackage'.length);
  }
  if (name.endsWith('.mlmodelc')) {
    return name.substring(0, name.length - '.mlmodelc'.length);
  }
  return name.replaceAll(RegExp(r'[^A-Za-z0-9_]+'), '_');
}
