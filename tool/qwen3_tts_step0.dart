import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/models.dart';

Object? _sanitize(Object? value) {
  if (value is double) {
    if (value.isNaN) return 'NaN';
    if (value == double.infinity) return 'Infinity';
    if (value == double.negativeInfinity) return '-Infinity';
    return value;
  }
  if (value is List) {
    return [for (final item in value) _sanitize(item)];
  }
  if (value is Map) {
    return {for (final entry in value.entries) entry.key: _sanitize(entry.value)};
  }
  return value;
}

Future<void> main(List<String> args) async {
  if (args.length < 3) {
    stderr.writeln(
      'usage: dart run tool/qwen3_tts_step0.dart <bundle_path> <prepared_ref.json> <text> [output.json]',
    );
    exitCode = 64;
    return;
  }
  final engine = Qwen3TtsEngine.load(args[0]);
  try {
    engine.setPreparedReference(Qwen3TtsPreparedReference.load(args[1]));
    final report = _sanitize(engine.debugStep0(args[2]));
    final encoded = const JsonEncoder.withIndent('  ').convert(report);
    if (args.length >= 4) {
      await File(args[3]).writeAsString('$encoded\n');
    } else {
      stdout.writeln(encoded);
    }
  } finally {
    engine.close();
  }
}
