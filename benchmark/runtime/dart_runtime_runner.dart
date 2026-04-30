import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/models.dart'
    show Qwen3AsrCoreMlRunner, Qwen3AsrNativeRunner;
import 'package:dart_mlx_ffi/runtime.dart';

import 'input_json.dart';

Future<void> main(List<String> args) async {
  final parsed = _Args(args);
  if (parsed.flag('help')) {
    stdout.writeln(_usage);
    return;
  }

  final modelId = parsed.option('model-id', required: true)!;
  final engine = _engine(parsed.option('engine', required: true)!);
  final artifact = parsed.option('artifact', required: true)!;
  final task = parsed.option('task') ?? 'tensor';
  final healthCheck = parsed.flag('health-check');
  final inputPath = parsed.option('input-json', required: !healthCheck);
  final outPath = parsed.option('out');
  final platformName =
      parsed.option('platform') ?? RuntimePlatformCurrent.current().name;
  final warmup = int.parse(parsed.option('warmup') ?? '1');
  final iters = int.parse(parsed.option('iters') ?? '5');
  final maxTokens = int.parse(parsed.option('max-tokens') ?? '64');
  final root = parsed.option('root') ?? Directory.current.path;
  final hfCacheRoot = parsed.option('hf-cache-root');
  final numThreads = parsed.option('num-threads') == null
      ? null
      : int.parse(parsed.option('num-threads')!);
  final backendOptions = <String, Object?>{
    if (parsed.option('provider') != null)
      'provider': parsed.option('provider'),
    if (parsed.option('delegate') != null)
      'delegate': parsed.option('delegate'),
    if (parsed.option('coreml-mode') != null)
      'coremlMode': parsed.option('coreml-mode'),
    if (parsed.option('coreml-compute-units') != null)
      'coremlComputeUnits': parsed.option('coreml-compute-units'),
    if (parsed.option('litert-section-index') != null)
      'litertSectionIndex': int.parse(parsed.option('litert-section-index')!),
    if (parsed.flag('require-provider')) 'requireProvider': true,
    if (parsed.flag('require-delegate')) 'requireDelegate': true,
  };

  if (await _tryRunQwen3AsrModelLevel(
    modelId: modelId,
    task: task,
    engine: engine,
    artifact: artifact,
    root: root,
    hfCacheRoot: hfCacheRoot,
    healthCheck: healthCheck,
    inputPath: inputPath,
    outPath: outPath,
    platformName: platformName,
    warmup: warmup,
    iters: iters,
    maxTokens: maxTokens,
    numThreads: numThreads,
    backendOptions: backendOptions,
  )) {
    return;
  }

  final spec = ModelSpec(
    id: modelId,
    family: modelId,
    modalities: const [ModelModality.textGeneration],
    platformArtifacts: {
      engine: RuntimeArtifact(
        engine: engine,
        path: artifact,
        targetPlatforms: [platformName],
      ),
    },
  );

  final registry = RuntimeRegistry.native(
    resolver: RuntimeResolver(hostPlatform: _platform(platformName)),
  );
  final options = RuntimeOptions(
    engine: engine,
    diagnostics: true,
    numThreads: numThreads,
    backendOptions: backendOptions,
    artifactResolver: hfCacheRoot == null
        ? null
        : HuggingFaceArtifactCache(cacheRoot: hfCacheRoot),
  );
  final session = artifact.startsWith('hf://')
      ? await registry.loadAsync(spec, rootPath: root, options: options)
      : registry.load(spec, rootPath: root, options: options);
  if (healthCheck) {
    try {
      final memoryAfterLoad = NativeRuntimeMemory.snapshot();
      final report = {
        'model_id': modelId,
        'platform': platformName,
        'engine': engine.name,
        'task': task,
        'artifact': artifact,
        'run_config': _runConfig(
          task: task,
          warmup: warmup,
          iters: iters,
          maxTokens: maxTokens,
        ),
        'passed': true,
        'health': {'loaded': true},
        'metrics': {'peak_memory_bytes': _peakMemory(memoryAfterLoad)},
        'device_profile': {
          'runtime': 'dart_mlx_ffi',
          'runtime_diagnostics': session.diagnostics,
          'memory_after': memoryAfterLoad,
          'raw_peak_memory_field': _rawPeakField(memoryAfterLoad),
        },
      };
      _writeReport(report, outPath);
    } finally {
      session.close();
    }
    return;
  }

  final inputPathValue = inputPath!;
  final sourceInputSignature = _readSourceInputSignature(inputPathValue);
  final inputs = ModelInputs(readRuntimeInputsJson(inputPathValue));
  final inputSignature = sourceInputSignature ?? _inputSignature(inputs);
  final memoryBefore = NativeRuntimeMemory.snapshot();
  var peakMemory = _peakMemory(memoryBefore);
  ModelOutputs? last;
  final latenciesMs = <double>[];
  try {
    for (var i = 0; i < warmup; i++) {
      session.run(inputs);
      peakMemory = _maxPeak(peakMemory, NativeRuntimeMemory.snapshot());
    }
    for (var i = 0; i < iters; i++) {
      final watch = Stopwatch()..start();
      last = session.run(inputs);
      watch.stop();
      latenciesMs.add(watch.elapsedMicroseconds / 1000.0);
      peakMemory = _maxPeak(peakMemory, NativeRuntimeMemory.snapshot());
    }
  } finally {
    session.close();
  }

  final memoryAfter = NativeRuntimeMemory.snapshot();
  peakMemory = _maxPeak(peakMemory, memoryAfter);
  final meanLatencyMs = _mean(latenciesMs);
  final report = {
    'model_id': modelId,
    'platform': platformName,
    'engine': engine.name,
    'task': task,
    'artifact': artifact,
    'run_config': _runConfig(
      task: task,
      warmup: warmup,
      iters: iters,
      maxTokens: maxTokens,
    ),
    'input_signature': inputSignature,
    'correctness': _correctness(last?.values ?? const {}),
    'metrics': {
      'end_to_end_ms': meanLatencyMs,
      'latency_ms': {
        'values': latenciesMs,
        'mean': meanLatencyMs,
        'p50': _percentile(latenciesMs, 0.50),
        'p95': _percentile(latenciesMs, 0.95),
      },
      'iteration_count': latenciesMs.length,
      'warmup_count': warmup,
      'peak_memory_bytes': peakMemory,
    },
    'device_profile': {
      'runtime': 'dart_mlx_ffi',
      'runtime_diagnostics': last?.diagnostics ?? const <String, Object?>{},
      'memory_before': memoryBefore,
      'memory_after': memoryAfter,
      'raw_peak_memory_field': _rawPeakField(memoryAfter),
    },
  };

  _writeReport(report, outPath);
}

Future<bool> _tryRunQwen3AsrModelLevel({
  required String modelId,
  required String task,
  required RuntimeEngine engine,
  required String artifact,
  required String root,
  required String? hfCacheRoot,
  required bool healthCheck,
  required String? inputPath,
  required String? outPath,
  required String platformName,
  required int warmup,
  required int iters,
  required int maxTokens,
  required int? numThreads,
  required Map<String, Object?> backendOptions,
}) async {
  if (!_shouldUseQwen3AsrRunner(modelId, task, engine, artifact)) {
    return false;
  }
  final bundlePath = engine == RuntimeEngine.coreml
      ? await _resolveQwen3AsrCoreMlBundlePath(
          artifact: artifact,
          root: root,
          hfCacheRoot: hfCacheRoot,
        )
      : await _resolveQwen3AsrBundlePath(
          artifact: artifact,
          engine: engine,
          root: root,
          hfCacheRoot: hfCacheRoot,
        );
  final tokenizerPath = engine == RuntimeEngine.coreml
      ? await _resolveQwen3AsrCoreMlTokenizerPath(
          bundlePath: bundlePath,
          hfCacheRoot: hfCacheRoot,
        )
      : null;
  final options = RuntimeOptions(
    engine: engine,
    diagnostics: true,
    numThreads: numThreads,
    prefer: _qwen3AsrAccelerators(engine, platformName),
    backendOptions: backendOptions,
  );
  final dynamic runner;
  final runnerName = engine == RuntimeEngine.coreml
      ? 'Qwen3AsrCoreMlRunner'
      : 'Qwen3AsrNativeRunner';
  if (engine == RuntimeEngine.coreml) {
    runner = Qwen3AsrCoreMlRunner.loadCoreMlBundle(
      bundlePath,
      tokenizerPath: tokenizerPath!,
      options: options,
    );
  } else if (engine == RuntimeEngine.litert) {
    runner = Qwen3AsrNativeRunner.loadLiteRtBundle(
      bundlePath,
      options: options,
    );
  } else {
    runner = Qwen3AsrNativeRunner.loadOnnxBundle(bundlePath, options: options);
  }
  try {
    if (healthCheck) {
      final memoryAfterLoad = NativeRuntimeMemory.snapshot();
      final componentDiagnostics = runner is Qwen3AsrCoreMlRunner
          ? runner.componentDiagnostics(includeDecoder: true)
          : (runner as Qwen3AsrNativeRunner).componentDiagnostics();
      _writeReport({
        'model_id': modelId,
        'platform': platformName,
        'engine': engine.name,
        'task': task,
        'artifact': artifact,
        'run_config': _runConfig(
          task: task,
          warmup: warmup,
          iters: iters,
          maxTokens: maxTokens,
        ),
        'passed': true,
        'health': {'loaded': true},
        'metrics': {'peak_memory_bytes': _peakMemory(memoryAfterLoad)},
        'device_profile': {
          'runtime': 'dart_mlx_ffi',
          'runtime_diagnostics': {
            ...componentDiagnostics,
            'bundle_path': bundlePath,
            ...tokenizerPath == null
                ? const <String, Object?>{}
                : {'tokenizer_path': tokenizerPath},
            'text_vocab_size': runner.config.textVocabSize,
            'text_hidden_size': runner.config.textHiddenSize,
          },
          'memory_after': memoryAfterLoad,
          'raw_peak_memory_field': _rawPeakField(memoryAfterLoad),
        },
      }, outPath);
      return true;
    }

    final inputPathValue = inputPath!;
    final sourceInputSignature = _readSourceInputSignature(inputPathValue);
    final inputs = ModelInputs(readRuntimeInputsJson(inputPathValue));
    final audio = _qwen3AsrAudio(inputs);
    final inputSignature = sourceInputSignature ?? _inputSignature(inputs);
    final memoryBefore = NativeRuntimeMemory.snapshot();
    var peakMemory = _peakMemory(memoryBefore);
    var lastIds = const <int>[];
    var lastText = '';
    final latenciesMs = <double>[];
    for (var i = 0; i < warmup; i++) {
      runner.transcribeToIds(audio, maxNewTokens: maxTokens);
      peakMemory = _maxPeak(peakMemory, NativeRuntimeMemory.snapshot());
    }
    for (var i = 0; i < iters; i++) {
      final watch = Stopwatch()..start();
      lastIds = runner.transcribeToIds(audio, maxNewTokens: maxTokens);
      watch.stop();
      lastText = runner.detokenize(lastIds);
      latenciesMs.add(watch.elapsedMicroseconds / 1000.0);
      peakMemory = _maxPeak(peakMemory, NativeRuntimeMemory.snapshot());
    }
    final memoryAfter = NativeRuntimeMemory.snapshot();
    peakMemory = _maxPeak(peakMemory, memoryAfter);
    final meanLatencyMs = _mean(latenciesMs);
    final tokenCount = lastIds.length;
    _writeReport({
      'model_id': modelId,
      'platform': platformName,
      'engine': engine.name,
      'task': task,
      'artifact': artifact,
      'run_config': _runConfig(
        task: task,
        warmup: warmup,
        iters: iters,
        maxTokens: maxTokens,
      ),
      'input_signature': inputSignature,
      'correctness': {
        'token_ids': lastIds,
        'output_text': lastText,
        'output_values': {
          'token_ids': {
            'dtype': 'int32',
            'shape': [tokenCount],
            'values': lastIds,
          },
        },
      },
      'metrics': {
        'end_to_end_ms': meanLatencyMs,
        'latency_ms': {
          'values': latenciesMs,
          'mean': meanLatencyMs,
          'p50': _percentile(latenciesMs, 0.50),
          'p95': _percentile(latenciesMs, 0.95),
        },
        'decode_tok_s': meanLatencyMs <= 0
            ? 0.0
            : tokenCount / (meanLatencyMs / 1000.0),
        'iteration_count': latenciesMs.length,
        'warmup_count': warmup,
        'peak_memory_bytes': peakMemory,
      },
      'device_profile': {
        'runtime': 'dart_mlx_ffi',
        'runtime_diagnostics': {
          'engine': engine.name,
          'model_level_runner': runnerName,
          'bundle_path': bundlePath,
          ...tokenizerPath == null
              ? const <String, Object?>{}
              : {'tokenizer_path': tokenizerPath},
        },
        'memory_before': memoryBefore,
        'memory_after': memoryAfter,
        'raw_peak_memory_field': _rawPeakField(memoryAfter),
      },
    }, outPath);
    return true;
  } finally {
    runner.close();
  }
}

List<Accelerator> _qwen3AsrAccelerators(
  RuntimeEngine engine,
  String platformName,
) {
  if (engine == RuntimeEngine.coreml) {
    return const [Accelerator.ane, Accelerator.gpu, Accelerator.cpu];
  }
  if (engine == RuntimeEngine.onnx && platformName == 'android') {
    return const [Accelerator.npu, Accelerator.gpu, Accelerator.cpu];
  }
  return const [Accelerator.gpu, Accelerator.cpu];
}

bool _shouldUseQwen3AsrRunner(
  String modelId,
  String task,
  RuntimeEngine engine,
  String artifact,
) {
  if (engine != RuntimeEngine.coreml &&
      engine != RuntimeEngine.onnx &&
      engine != RuntimeEngine.litert) {
    return false;
  }
  if (modelId == 'qwen3_asr') return true;
  if ((task == 'audio' || task == 'asr') && artifact.contains('qwen3_asr')) {
    return true;
  }
  return _isQwen3AsrLiteRtBundle(artifact);
}

Future<String> _resolveQwen3AsrCoreMlBundlePath({
  required String artifact,
  required String root,
  required String? hfCacheRoot,
}) async {
  if (artifact.startsWith('hf://')) {
    final resolved = await HuggingFaceArtifactCache(cacheRoot: hfCacheRoot)
        .resolve(
          RuntimeArtifact(
            engine: RuntimeEngine.coreml,
            path: _hfRepositoryRootUri(artifact),
          ),
        );
    return resolved.path;
  }
  final local = _resolveLocalArtifactPath(artifact, root);
  if (local.endsWith('.mlmodelc') || local.endsWith('.mlpackage')) {
    return Directory(local).parent.path;
  }
  return local;
}

Future<String> _resolveQwen3AsrCoreMlTokenizerPath({
  required String bundlePath,
  required String? hfCacheRoot,
}) async {
  if (File('$bundlePath/tokenizer.json').existsSync() ||
      File('$bundlePath/vocab.json').existsSync()) {
    return bundlePath;
  }
  final resolved = await HuggingFaceArtifactCache(cacheRoot: hfCacheRoot)
      .resolve(
        const RuntimeArtifact(
          engine: RuntimeEngine.onnx,
          path: 'hf://andrewleech/qwen3-asr-1.7b-onnx/tokenizer.json',
        ),
      );
  return File(resolved.path).parent.path;
}

String _hfRepositoryRootUri(String artifact) {
  final uri = Uri.parse(artifact);
  if (uri.scheme != 'hf' || uri.host.isEmpty || uri.pathSegments.isEmpty) {
    return artifact;
  }
  return 'hf://${uri.host}/${uri.pathSegments.first}/.';
}

bool _isQwen3AsrLiteRtBundle(String artifact) {
  if (!artifact.endsWith('.json')) return false;
  final file = File(artifact);
  if (!file.existsSync()) return false;
  try {
    final decoded = jsonDecode(file.readAsStringSync());
    return decoded is Map &&
        decoded['format'] == 'dart_mlx_ffi.qwen3_asr_litert_bundle.v1';
  } on FormatException {
    return false;
  } on FileSystemException {
    return false;
  }
}

Future<String> _resolveQwen3AsrBundlePath({
  required String artifact,
  required RuntimeEngine engine,
  required String root,
  required String? hfCacheRoot,
}) async {
  var local = _resolveLocalArtifactPath(artifact, root);
  if (artifact.startsWith('hf://')) {
    final resolved = await HuggingFaceArtifactCache(
      cacheRoot: hfCacheRoot,
    ).resolve(RuntimeArtifact(engine: engine, path: artifact));
    local = resolved.path;
  }
  if (local.endsWith('.json') && File(local).existsSync()) {
    return File(local).parent.path;
  }
  if (FileSystemEntity.isFileSync(local)) {
    return File(local).parent.path;
  }
  return local;
}

String _resolveLocalArtifactPath(String artifact, String root) {
  if (artifact.contains('://') || File(artifact).isAbsolute) return artifact;
  if (FileSystemEntity.typeSync(artifact) != FileSystemEntityType.notFound) {
    return artifact;
  }
  if (root.isEmpty) return artifact;
  return root.endsWith('/') ? '$root$artifact' : '$root/$artifact';
}

Float32List _qwen3AsrAudio(ModelInputs inputs) {
  const preferred = ['audio', 'waveform', 'input_values', 'input'];
  for (final name in preferred) {
    final value = inputs.values[name];
    if (value is RuntimeTensor &&
        value.dtype == RuntimeTensorDataType.float32) {
      return Float32List.fromList(value.asFloat32List());
    }
  }
  for (final value in inputs.values.values) {
    if (value is RuntimeTensor &&
        value.dtype == RuntimeTensorDataType.float32) {
      return Float32List.fromList(value.asFloat32List());
    }
  }
  throw StateError(
    'Qwen3-ASR benchmark input must include float32 audio, waveform, '
    'input_values, or input.',
  );
}

void _writeReport(Map<String, Object?> report, String? outPath) {
  final text = const JsonEncoder.withIndent('  ').convert(report);
  if (outPath != null) {
    File(outPath).createSync(recursive: true);
    File(outPath).writeAsStringSync('$text\n');
  }
  stdout.writeln(text);
}

Map<String, Object?> _correctness(Map<String, Object?> outputs) {
  final outputValues = <String, Object?>{};
  final outputSummaries = <String, Object?>{};
  for (final entry in outputs.entries) {
    final tensor = entry.value;
    if (tensor is! RuntimeTensor) continue;
    final values = _tensorValues(tensor);
    final summary = <String, Object?>{
      'dtype': tensor.dtype.name,
      'shape': tensor.shape,
      'preview': values.take(16).toList(),
      'num_values': values.length,
    };
    final topK = _topK(values, 10);
    if (topK.isNotEmpty) {
      summary['top_k'] = topK;
    }
    outputSummaries[entry.key] = summary;
    if (values.length <= 4096) {
      outputValues[entry.key] = {
        'dtype': tensor.dtype.name,
        'shape': tensor.shape,
        'values': values,
      };
    }
  }
  return {
    'output_summaries': outputSummaries,
    if (outputValues.isNotEmpty) 'output_values': outputValues,
  };
}

Map<String, Object?> _inputSignature(ModelInputs inputs) {
  final tensors = <Map<String, Object?>>[];
  var rolling = _Fnv64();
  for (final name in inputs.values.keys.toList()..sort()) {
    final value = inputs.values[name];
    if (value is! RuntimeTensor) continue;
    final tensorHash = _Fnv64()
      ..addString(name)
      ..addString(value.dtype.name)
      ..addString(value.shape.join(','))
      ..addBytes(value.bytes);
    final digest = tensorHash.hex();
    rolling
      ..addString(name)
      ..addString(value.dtype.name)
      ..addString(value.shape.join(','))
      ..addString(digest);
    tensors.add({
      'name': name,
      'dtype': value.dtype.name,
      'shape': value.shape,
      'byte_length': value.bytes.lengthInBytes,
      'digest': digest,
    });
  }
  return {
    'format': 'dart_mlx_ffi.input_signature.v1',
    'digest': rolling.hex(),
    'tensors': tensors,
  };
}

Map<String, Object?> _runConfig({
  required String task,
  required int warmup,
  required int iters,
  required int maxTokens,
}) {
  return {
    'format': 'dart_mlx_ffi.run_config.v1',
    'task': task,
    'warmup': warmup,
    'iters': iters,
    'max_tokens': maxTokens,
    'sampling_strategy': 'greedy',
  };
}

Map<String, Object?>? _readSourceInputSignature(String inputPath) {
  final decoded = jsonDecode(File(inputPath).readAsStringSync());
  if (decoded is! Map) return null;
  final metadata = decoded['metadata'];
  if (metadata is! Map) return null;
  final signature = metadata['input_signature'];
  if (signature is! Map) return null;
  return Map<String, Object?>.from(signature);
}

List<Object?> _tensorValues(RuntimeTensor tensor) {
  return switch (tensor.dtype) {
    RuntimeTensorDataType.float32 => tensor.asFloat32List().toList(),
    RuntimeTensorDataType.float64 => tensor.asFloat64List().toList(),
    RuntimeTensorDataType.int32 => tensor.asInt32List().toList(),
    RuntimeTensorDataType.int64 => tensor.asInt64List().toList(),
    RuntimeTensorDataType.uint8 => tensor.asUint8List().toList(),
    RuntimeTensorDataType.boolean =>
      tensor.asUint8List().map((value) => value != 0).toList(),
    RuntimeTensorDataType.float16 => _float16Values(tensor),
  };
}

List<double> _float16Values(RuntimeTensor tensor) {
  final halves = tensor.bytes.buffer.asUint16List(
    tensor.bytes.offsetInBytes,
    tensor.bytes.lengthInBytes ~/ 2,
  );
  return halves.map(_halfToDouble).toList(growable: false);
}

double _halfToDouble(int value) {
  final sign = (value & 0x8000) == 0 ? 1.0 : -1.0;
  final exponent = (value >> 10) & 0x1f;
  final fraction = value & 0x03ff;
  if (exponent == 0) {
    if (fraction == 0) return sign == 1.0 ? 0.0 : -0.0;
    return sign * (fraction / 1024.0) * 0.00006103515625;
  }
  if (exponent == 0x1f) {
    if (fraction == 0) return sign * double.infinity;
    return double.nan;
  }
  return sign * (1.0 + fraction / 1024.0) * _pow2(exponent - 15);
}

double _pow2(int exponent) {
  var value = 1.0;
  if (exponent >= 0) {
    for (var i = 0; i < exponent; i++) {
      value *= 2.0;
    }
  } else {
    for (var i = 0; i < -exponent; i++) {
      value *= 0.5;
    }
  }
  return value;
}

List<Map<String, Object?>> _topK(List<Object?> values, int k) {
  final indexed = <({int index, num value})>[];
  for (var i = 0; i < values.length; i++) {
    final value = values[i];
    if (value is num && !value.isNaN) {
      indexed.add((index: i, value: value));
    }
  }
  indexed.sort((a, b) => b.value.compareTo(a.value));
  return indexed
      .take(k)
      .map((item) => {'index': item.index, 'value': item.value})
      .toList(growable: false);
}

double _mean(List<double> values) {
  if (values.isEmpty) return 0.0;
  return values.reduce((left, right) => left + right) / values.length;
}

double _percentile(List<double> values, double percentile) {
  if (values.isEmpty) return 0.0;
  final sorted = values.toList(growable: false)..sort();
  if (sorted.length == 1) return sorted.first;
  final rank = percentile.clamp(0.0, 1.0) * (sorted.length - 1);
  final lower = rank.floor();
  final upper = rank.ceil();
  if (lower == upper) return sorted[lower];
  final fraction = rank - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

RuntimeEngine _engine(String value) {
  return RuntimeEngine.values.firstWhere(
    (engine) => engine.name == value,
    orElse: () => throw ArgumentError('Unsupported engine: $value'),
  );
}

RuntimePlatform _platform(String value) {
  return RuntimePlatform.values.firstWhere(
    (platform) => platform.name == value,
    orElse: () => RuntimePlatformCurrent.current(),
  );
}

int _peakMemory(Map<String, Object?> snapshot) =>
    (snapshot['peak_memory_bytes'] as num?)?.toInt() ??
    (snapshot['resident_size'] as num?)?.toInt() ??
    (snapshot['vm_rss'] as num?)?.toInt() ??
    ProcessInfo.currentRss;

int _maxPeak(int current, Map<String, Object?> snapshot) =>
    current > _peakMemory(snapshot) ? current : _peakMemory(snapshot);

String _rawPeakField(Map<String, Object?> snapshot) {
  if (snapshot.containsKey('phys_footprint')) return 'phys_footprint';
  if (snapshot.containsKey('peak_working_set')) return 'peak_working_set';
  if (snapshot.containsKey('android_peak_pss')) return 'android_peak_pss';
  if (snapshot.containsKey('vm_hwm')) return 'VmHWM';
  return 'ProcessInfo.currentRss';
}

final class _Fnv64 {
  static const _offset = 0xcbf29ce484222325;
  static const _prime = 0x100000001b3;
  static const _mask = 0xffffffffffffffff;

  int _value = _offset;

  void addString(String value) {
    addBytes(utf8.encode(value));
    addByte(0);
  }

  void addBytes(List<int> bytes) {
    for (final byte in bytes) {
      addByte(byte);
    }
  }

  void addByte(int byte) {
    _value ^= byte & 0xff;
    _value = (_value * _prime) & _mask;
  }

  String hex() => _value.toRadixString(16).padLeft(16, '0');
}

final class _Args {
  _Args(List<String> args) : _values = _parse(args);

  final Map<String, String?> _values;

  bool flag(String name) => _values.containsKey(name);

  String? option(String name, {bool required = false}) {
    final value = _values[name];
    if (required && (value == null || value.isEmpty)) {
      throw ArgumentError('Missing --$name');
    }
    return value;
  }

  static Map<String, String?> _parse(List<String> args) {
    final values = <String, String?>{};
    for (var i = 0; i < args.length; i++) {
      final arg = args[i];
      if (!arg.startsWith('--')) {
        throw ArgumentError('Unexpected positional argument: $arg');
      }
      final name = arg.substring(2);
      if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
        values[name] = args[++i];
      } else {
        values[name] = null;
      }
    }
    return values;
  }
}

const _usage = '''
Usage:
  dart run benchmark/runtime/dart_runtime_runner.dart \\
    --model-id <id> \\
    --engine <coreml|onnx|litert> \\
    --artifact <path> \\
    [--task <text|vlm|embedding|function|audio|tts|vad|tensor>] \\
    --input-json <inputs.json> \\
    [--root <artifact-root>] [--warmup 1] [--iters 5] [--out report.json]
    [--num-threads N] [--provider ORT_EP] [--delegate xnnpack]
    [--coreml-mode decode|prefill] [--coreml-compute-units cpuAndNeuralEngine|cpuAndGPU|cpuOnly|all]
    [--litert-section-index N]
    [--hf-cache-root <dir>] [--health-check]

When --artifact starts with hf://, the runner resolves it through
HuggingFaceArtifactCache before native execution.
When --health-check is set, the runner only loads the native session and writes
diagnostics; --input-json is not required and no inference is executed.

Input JSON:
{
  "input": {"dtype": "float32", "shape": [1, 4], "values": [1, 2, 3, 4]}
}

The runner also accepts {"inputs": {...}}, nested numeric values, base64 raw
tensor bytes, or file/path fields relative to the JSON file.
''';
