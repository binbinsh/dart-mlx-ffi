import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main(List<String> args) {
  if (args.length != 2 && args.length != 3) {
    stderr.writeln(
      'usage: dart run benchmark/generic_import_run.dart <export_path> <input_safetensors_path> [input_names_json]',
    );
    exitCode = 64;
    return;
  }

  final exportPath = args[0];
  final inputPath = args[1];
  final inputNames = args.length == 3
      ? (jsonDecode(args[2]) as List<Object?>).cast<String>()
      : const <String>['input_ids'];
  final valuesPath = Platform.environment['GENERIC_VALUES_PATH'];
  final fullValues = Platform.environment['GENERIC_FULL_VALUES'] == '1';
  final warmup = int.tryParse(Platform.environment['GENERIC_WARMUP'] ?? '') ?? 0;
  final iters = int.tryParse(Platform.environment['GENERIC_ITERS'] ?? '') ?? 1;

  final inputs = mx.io.loadSafetensors(inputPath).tensors;
  final orderedInputs = <MlxArray>[];
  for (final name in inputNames) {
    final input = inputs[name];
    if (input == null) {
      stderr.writeln('Missing $name in $inputPath');
      exitCode = 66;
      return;
    }
    orderedInputs.add(input);
  }

  final imported = MlxExport.importFunction(exportPath);
  try {
    for (var index = 0; index < warmup; index++) {
      final out = _runOnce(imported, orderedInputs);
      out.close();
    }

    final stopwatch = Stopwatch()..start();
    MlxArray? last;
    for (var index = 0; index < iters; index++) {
      last?.close();
      last = _runOnce(imported, orderedInputs);
    }
    stopwatch.stop();
    if (last == null) {
      throw StateError('No iterations executed.');
    }

    final flat = last.reshape([last.size]).astype(MlxDType.MLX_FLOAT32);
    final preview = flat
        .slice(start: [0], stop: [flat.shape[0] < 16 ? flat.shape[0] : 16])
        .astype(MlxDType.MLX_FLOAT32);
    final payload = <String, Object?>{
      'shape': last.shape,
      'per_iter_ms': stopwatch.elapsedMicroseconds / 1000.0 / iters,
      'output_preview': List<double>.from(preview.toList().cast<double>()),
    };
    if (valuesPath != null && valuesPath.isNotEmpty) {
      mx.io.saveSafetensors(valuesPath, <String, MlxArray>{'output': last});
      payload['values_path'] = valuesPath;
    }
    if (fullValues) {
      payload['values'] = List<double>.from(flat.toList().cast<double>());
    }
    preview.close();
    flat.close();
    stdout.writeln(jsonEncode(payload));
    last.close();
    exit(0);
  } finally {
    imported.close();
    for (final input in orderedInputs) {
      input.close();
    }
  }
}

MlxArray _runOnce(MlxImportedFunction imported, List<MlxArray> inputs) {
  final outputs = imported.call(inputs);
  if (outputs.length != 1) {
    throw StateError('Expected 1 output, got ${outputs.length}.');
  }
  final out = outputs[0];
  MlxRuntime.evalAll([out]);
  final stream = MlxStream.defaultGpu();
  try {
    stream.synchronize();
  } finally {
    stream.close();
  }
  return out;
}
