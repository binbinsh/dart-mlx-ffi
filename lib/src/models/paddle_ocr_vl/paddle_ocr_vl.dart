library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/mlx.dart';
import '../shared/tensor_map.dart';
import '../shared/tuning.dart';

part 'config.dart';
part 'debug.dart';
part 'linear.dart';
part 'norm.dart';
part 'vision.dart';
part 'vision_mlp.dart';
part 'gelu_lut.dart';
part 'runtime.dart';
part 'rope.dart';
part 'cache.dart';
part 'kv_quant.dart';
part 'turboquant.dart';
part 'turboquant_data.dart';
part 'trace.dart';
part 'layers.dart';
part 'decode.dart';
part 'sample.dart';
part 'prefill.dart';
part 'runner_debug.dart';
part 'embed.dart';
part 'runner_load.dart';
part 'runner.dart';
// `hybrid_runner.dart` is a *standalone* library (not a part-of). The
// CoreML+MLX hybrid OCR pipeline lives in `hybrid_runner.dart`,
// `coreml_loader.dart`, `coreml_image.dart`, `coreml_mrope.dart`,
// `coreml_pipeline_manifest.dart`.
