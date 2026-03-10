import 'dart:convert';
import 'dart:io';

import 'qwen_run.dart';

void main(List<String> args) {
  final manifest =
      jsonDecode(File(args[0]).readAsStringSync()) as Map<String, Object?>;
  final modelName = args[1];
  final spec = (manifest['models'] as List<Object?>)
      .cast<Map<String, Object?>>()
      .firstWhere((model) => model['name'] == modelName);
  final runner = QwenRunner.load(spec['snapshot_path'] as String);
  final tokens =
      (spec['tokens'] as List<Object?>).cast<num>().map((v) => v.toInt()).toList();
  try {
    print(jsonEncode(runner.debugSlices(tokens)));
  } finally {
    runner.close();
  }
}
