library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import '../shared/tensor_map.dart';

part 'config.dart';
part 'cache.dart';
part 'gdelta.dart';
part 'linear.dart';
part 'decode.dart';
part 'bench.dart';
part 'rope.dart';
part 'runner.dart';
part 'layers.dart';
part 'session.dart';
part 'topk.dart';

typedef Qwen35TimedGeneration = ({
  List<int> tokenIds,
  List<int> generatedTokenIds,
  double promptMs,
  double firstTokenMs,
  double decodeMs,
  double totalMs,
  bool stoppedByStopToken,
});
