import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  final warmup = readSyntheticBenchArg(args, '--warmup', fallback: 20);
  final iters = readSyntheticBenchArg(args, '--iters', fallback: 100);
  final asJson = args.contains('--json');
  final report = runSyntheticModelBenchmarks(warmup: warmup, iters: iters);
  if (asJson) {
    stdout.writeln(jsonEncode(report));
    return;
  }
  printSyntheticBenchmarkReport(report);
}
