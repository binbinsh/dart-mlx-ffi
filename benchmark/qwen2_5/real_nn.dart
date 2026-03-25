import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  final manifestPath = readStringArg(
    args,
    '--manifest',
    fallback: 'benchmark/out/real_nn/real_manifest.json',
  );
  final outputPath = readStringArg(
    args,
    '--output',
    fallback: 'benchmark/out/real_nn/real_dart.json',
  );
  final warmup = readIntArg(args, '--warmup', fallback: 3);
  final iters = readIntArg(args, '--iters', fallback: 20);
  final engine = readStringArg(args, '--engine', fallback: 'direct');
  final modelName = tryReadStringArg(args, '--model-name');

  final manifest = jsonDecode(File(manifestPath).readAsStringSync())
      as Map<String, Object?>;
  final allModelSpecs =
      (manifest['models'] as List<Object?>).cast<Map<String, Object?>>();
  final modelSpecs = modelName == null
      ? allModelSpecs
      : allModelSpecs.where((spec) => spec['name'] == modelName).toList();

  final report = <String, Object?>{
    'runtime': engine == 'compiled'
        ? 'dart_hf_mlx_compiled'
        : 'dart_hf_mlx_direct',
    'mlx_version': MlxVersion.current(),
    'device': MlxDevice.defaultDevice().toString(),
    'metal': MlxMetal.isAvailable(),
    'warmup': warmup,
    'iters': iters,
    'prompt': manifest['prompt'],
    'seq_len': manifest['seq_len'],
    'models': modelSpecs
        .map(
          (spec) => engine == 'compiled'
              ? benchCompiledModel(spec, warmup: warmup, iters: iters)
              : benchModel(spec, warmup: warmup, iters: iters),
        )
        .toList(),
  };

  final file = File(outputPath);
  file.parent.createSync(recursive: true);
  file.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
  stdout.writeln(file.path);
}

Map<String, Object?> benchModel(
  Map<String, Object?> spec, {
  required int warmup,
  required int iters,
}) {
  final snapshotPath = spec['snapshot_path'] as String;
  final tokens =
      (spec['tokens'] as List<Object?>).cast<num>().map((v) => v.toInt()).toList();
  final runner = QwenRunner.load(snapshotPath);

  try {
    for (var i = 0; i < warmup; i++) {
      final output = runner.run(tokens);
      output.close();
    }

    MlxMemory.resetPeak();
    final beforePeak = MlxMemory.peakBytes();
    final stopwatch = Stopwatch()..start();
    MlxArray? last;
    try {
      for (var i = 0; i < iters; i++) {
        last?.close();
        last = runner.run(tokens);
      }
      stopwatch.stop();
      final reportOut = last!.astype(MlxDType.MLX_FLOAT32);
      final flat = List<double>.from(reportOut.toList().cast<double>());
      reportOut.close();
      final afterPeak = MlxMemory.peakBytes();
      final totalMs = stopwatch.elapsedMicroseconds / 1000.0;
      return modelReport(spec, tokens, last.shape, flat, totalMs, iters, afterPeak - beforePeak);
    } finally {
      last?.close();
    }
  } finally {
    runner.close();
  }
}

Map<String, Object?> benchCompiledModel(
  Map<String, Object?> spec, {
  required int warmup,
  required int iters,
}) {
  final snapshotPath = spec['snapshot_path'] as String;
  final tokens =
      (spec['tokens'] as List<Object?>).cast<num>().map((v) => v.toInt()).toList();
  final runner = QwenRunner.load(snapshotPath);
  final input = MlxArray.fromInt32List(tokens, shape: [1, tokens.length]);
  final fn = MlxFunction.fromCallback((args) => [runner.buildGraph(args[0])]);
  final compiled = fn.compile();

  try {
    MlxCompile.enable();
    MlxCompile.setMode(MlxCompileMode.MLX_COMPILE_MODE_ENABLED);

    List<MlxArray> run() {
      final outputs = compiled([input]);
      MlxRuntime.evalAll(outputs);
      return outputs;
    }

    for (var i = 0; i < warmup; i++) {
      final outputs = run();
      for (final output in outputs) {
        output.close();
      }
    }

    MlxMemory.resetPeak();
    final beforePeak = MlxMemory.peakBytes();
    final stopwatch = Stopwatch()..start();
    List<MlxArray>? last;
    try {
      for (var i = 0; i < iters; i++) {
        if (last != null) {
          for (final output in last) {
            output.close();
          }
        }
        last = run();
      }
      stopwatch.stop();
      final reportOut = last!.first.astype(MlxDType.MLX_FLOAT32);
      final flat = List<double>.from(reportOut.toList().cast<double>());
      reportOut.close();
      final afterPeak = MlxMemory.peakBytes();
      final totalMs = stopwatch.elapsedMicroseconds / 1000.0;
      return modelReport(
        spec,
        tokens,
        last.first.shape,
        flat,
        totalMs,
        iters,
        afterPeak - beforePeak,
      );
    } finally {
      if (last != null) {
        for (final output in last) {
          output.close();
        }
      }
      MlxCompile.disable();
    }
  } finally {
    compiled.close();
    fn.close();
    input.close();
    runner.close();
  }
}

Map<String, Object?> modelReport(
  Map<String, Object?> spec,
  List<int> tokens,
  List<int> shape,
  List<double> flat,
  double totalMs,
  int iters,
  int peakBytesDelta,
) {
  return <String, Object?>{
    'name': spec['name'],
    'model_id': spec['model_id'],
    'snapshot_path': spec['snapshot_path'],
    'token_count': tokens.length,
    'input_shape': [1, tokens.length],
    'output_shape': shape,
    'output_preview': preview(flat),
    'output_values': flat,
    'total_ms': totalMs,
    'per_iter_ms': totalMs / iters,
    'peak_bytes_delta': peakBytesDelta,
  };
}

List<double> preview(List<double> values, {int limit = 8}) {
  final end = math.min(limit, values.length);
  return values.sublist(0, end);
}

int readIntArg(List<String> args, String name, {required int fallback}) {
  final value = readStringArg(args, name, fallback: '$fallback');
  return int.tryParse(value) ?? fallback;
}

String readStringArg(List<String> args, String name, {String? fallback}) {
  final value = tryReadStringArg(args, name);
  if (value != null) {
    return value;
  }
  if (fallback == null) {
    throw ArgumentError('Missing required argument: $name');
  }
  return fallback;
}

String? tryReadStringArg(List<String> args, String name) {
  final prefix = '$name=';
  for (var index = 0; index < args.length; index++) {
    final arg = args[index];
    if (arg.startsWith(prefix)) {
      return arg.substring(prefix.length);
    }
    if (arg == name && index + 1 < args.length) {
      return args[index + 1];
    }
  }
  return null;
}
