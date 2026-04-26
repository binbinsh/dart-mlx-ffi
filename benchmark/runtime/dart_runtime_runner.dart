import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

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
  final healthCheck = parsed.flag('health-check');
  final inputPath = parsed.option('input-json', required: !healthCheck);
  final outPath = parsed.option('out');
  final platformName =
      parsed.option('platform') ?? RuntimePlatformCurrent.current().name;
  final warmup = int.parse(parsed.option('warmup') ?? '1');
  final iters = int.parse(parsed.option('iters') ?? '5');
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
    if (parsed.option('litert-section-index') != null)
      'litertSectionIndex': int.parse(parsed.option('litert-section-index')!),
    if (parsed.flag('require-provider')) 'requireProvider': true,
    if (parsed.flag('require-delegate')) 'requireDelegate': true,
  };

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
        'artifact': artifact,
        'passed': true,
        'health': {'loaded': true},
        'metrics': {'peak_memory_bytes': _peakMemory(memoryAfterLoad)},
        'device_profile': {
          'runtime': 'dart_inference',
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
  final inputs = ModelInputs(readRuntimeInputsJson(inputPathValue));
  final memoryBefore = NativeRuntimeMemory.snapshot();
  var peakMemory = _peakMemory(memoryBefore);
  ModelOutputs? last;
  final watch = Stopwatch();
  try {
    for (var i = 0; i < warmup; i++) {
      session.run(inputs);
      peakMemory = _maxPeak(peakMemory, NativeRuntimeMemory.snapshot());
    }
    for (var i = 0; i < iters; i++) {
      watch.start();
      last = session.run(inputs);
      watch.stop();
      peakMemory = _maxPeak(peakMemory, NativeRuntimeMemory.snapshot());
    }
  } finally {
    session.close();
  }

  final perIterMs = iters > 0 ? watch.elapsedMicroseconds / 1000.0 / iters : 0;
  final memoryAfter = NativeRuntimeMemory.snapshot();
  peakMemory = _maxPeak(peakMemory, memoryAfter);
  final report = {
    'model_id': modelId,
    'platform': platformName,
    'engine': engine.name,
    'artifact': artifact,
    'correctness': _correctness(last?.values ?? const {}),
    'metrics': {'end_to_end_ms': perIterMs, 'peak_memory_bytes': peakMemory},
    'device_profile': {
      'runtime': 'dart_inference',
      'runtime_diagnostics': last?.diagnostics ?? const <String, Object?>{},
      'memory_before': memoryBefore,
      'memory_after': memoryAfter,
      'raw_peak_memory_field': _rawPeakField(memoryAfter),
    },
  };

  _writeReport(report, outPath);
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
    --input-json <inputs.json> \\
    [--root <artifact-root>] [--warmup 1] [--iters 5] [--out report.json]
    [--num-threads N] [--provider ORT_EP] [--delegate xnnpack]
    [--coreml-mode decode|prefill] [--litert-section-index N]
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
