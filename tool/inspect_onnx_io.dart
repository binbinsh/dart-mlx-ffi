/// Dump ONNX model I/O metadata. Useful when wiring a new model.
///
/// ```sh
/// dart run tool/inspect_onnx_io.dart --model <path-to-onnx>
/// ```
library;

import 'dart:io';

import 'package:dart_inference/runtime.dart';

void main(List<String> argv) {
  String? modelPath;
  for (var i = 0; i < argv.length; i++) {
    if (argv[i] == '--model' && i + 1 < argv.length) {
      modelPath = argv[i + 1];
    }
  }
  if (modelPath == null) {
    stderr.writeln(
      'usage: dart run tool/inspect_onnx_io.dart --model <path>',
    );
    exit(2);
  }

  final session = DartOnnxSession.load(
    DartOnnxConfig(
      modelPath: modelPath,
      id: 'inspect',
      family: 'inspect',
      provider: 'cpu',
      requireProvider: false,
      numThreads: 2,
    ),
  );
  try {
    final diag = session.diagnostics;
    stdout.writeln('provider          : ${session.selectedProvider}');
    stdout.writeln('available providers: ${diag['available_providers']}');
    stdout.writeln('--- inputs ---');
    final inputs = (diag['input_metadata'] as List).cast<Map>();
    for (final m in inputs) {
      stdout.writeln(
        '  ${m['name']}  dtype=${m['dtype']}  shape=${m['shape']}  '
        'symbolic=${m['symbolic_shape']}',
      );
    }
    stdout.writeln('--- outputs ---');
    final outputs = (diag['output_metadata'] as List).cast<Map>();
    for (final m in outputs) {
      stdout.writeln(
        '  ${m['name']}  dtype=${m['dtype']}  shape=${m['shape']}  '
        'symbolic=${m['symbolic_shape']}',
      );
    }
  } finally {
    session.close();
  }
}
