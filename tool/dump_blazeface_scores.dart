/// Dump BlazeFace raw score distribution for a portrait. Helps debug
/// detector decode mismatches.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';
import 'package:image/image.dart' as img;

void main(List<String> argv) {
  final onnxPath = argv[0];
  final portraitPath = argv[1];

  final pngBytes = File(portraitPath).readAsBytesSync();
  final decoded = img.decodePng(pngBytes)!;
  stdout.writeln('portrait: ${decoded.width}x${decoded.height}');

  // Letterbox to 128x128 NHWC float32 [-1,1]. Center-fit.
  final srcW = decoded.width;
  final srcH = decoded.height;
  final scale = 128.0 / math.max(srcW, srcH);
  final fitW = (srcW * scale).round();
  final fitH = (srcH * scale).round();
  final padX = ((128 - fitW) / 2).floor();
  final padY = ((128 - fitH) / 2).floor();
  final input = Float32List(128 * 128 * 3);
  for (var y = 0; y < 128; y++) {
    for (var x = 0; x < 128; x++) {
      final outIdx = (y * 128 + x) * 3;
      final fx = x - padX;
      final fy = y - padY;
      if (fx < 0 || fy < 0 || fx >= fitW || fy >= fitH) {
        input[outIdx + 0] = 0;
        input[outIdx + 1] = 0;
        input[outIdx + 2] = 0;
        continue;
      }
      final sx = (fx / scale).floor().clamp(0, srcW - 1);
      final sy = (fy / scale).floor().clamp(0, srcH - 1);
      final p = decoded.getPixel(sx, sy);
      input[outIdx + 0] = (p.r.toDouble() / 127.5) - 1.0;
      input[outIdx + 1] = (p.g.toDouble() / 127.5) - 1.0;
      input[outIdx + 2] = (p.b.toDouble() / 127.5) - 1.0;
    }
  }

  final session = DartOnnxSession.load(
    DartOnnxConfig(
      modelPath: onnxPath,
      id: 'blaze_dump',
      family: 'face_detection',
      provider: 'cpu',
      requireProvider: false,
    ),
  );
  final diag = session.diagnostics;
  final inputName = (diag['input_metadata'] as List).first['name'] as String;
  final result = session.run({
    inputName: RuntimeTensor.float32([1, 128, 128, 3], input),
  });
  for (final entry in result.outputs.entries) {
    final t = entry.value as RuntimeTensor;
    stdout.writeln('output ${entry.key} shape=${t.shape}');
  }
  // Find classificators (last dim 1)
  String className = '';
  String regName = '';
  for (final entry in result.outputs.entries) {
    final t = entry.value as RuntimeTensor;
    if (t.shape.last == 1) className = entry.key;
    if (t.shape.last == 16) regName = entry.key;
  }
  final scores = (result.outputs[className] as RuntimeTensor).asFloat32List();
  final regs = (result.outputs[regName] as RuntimeTensor).asFloat32List();
  // sigmoid all, sort desc, dump top 10
  final activated = <double>[];
  for (var i = 0; i < scores.length; i++) {
    final s = 1.0 / (1.0 + math.exp(-scores[i].clamp(-100.0, 100.0)));
    activated.add(s);
  }
  final indices = List<int>.generate(scores.length, (i) => i);
  indices.sort((a, b) => activated[b].compareTo(activated[a]));
  stdout.writeln('top 10 raw logits / sigmoid scores:');
  for (var k = 0; k < 10; k++) {
    final i = indices[k];
    stdout.writeln(
      '  anchor[$i] logit=${scores[i].toStringAsFixed(3)} '
      'sigmoid=${activated[i].toStringAsFixed(4)} '
      'reg[0..3]=[${regs[i * 16].toStringAsFixed(2)}, '
      '${regs[i * 16 + 1].toStringAsFixed(2)}, '
      '${regs[i * 16 + 2].toStringAsFixed(2)}, '
      '${regs[i * 16 + 3].toStringAsFixed(2)}]',
    );
  }
  stdout.writeln(
    'score distribution: min=${activated.reduce(math.min).toStringAsFixed(4)} '
    'max=${activated.reduce(math.max).toStringAsFixed(4)} '
    'count>0.5=${activated.where((s) => s > 0.5).length} '
    'count>0.3=${activated.where((s) => s > 0.3).length} '
    'count>0.1=${activated.where((s) => s > 0.1).length}',
  );
  result.close();
  session.close();
}
