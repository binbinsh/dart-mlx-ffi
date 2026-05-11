/// Smoke test for LivePortrait face crop (Phase 1).
///
/// Loads a source portrait PNG, runs YuNet, computes the 5-point
/// similarity transform, and writes the aligned 512x512 crop to disk.
///
/// ```sh
/// dart run tool/test_face_crop.dart \
///   --snapshot ~/Projects/Personal/cmdspace-app/.cache/live_portrait \
///   --portrait ~/Projects/Personal/cmdspace-app/assets/buddies/girlfriend/looks/look_02/portrait.png \
///   --out /tmp/look_02_crop.png
/// ```
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:image/image.dart' as img;

void main(List<String> argv) {
  String? snapshotDir;
  String? portraitPath;
  String? outPath;
  for (var i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--snapshot':
        snapshotDir = argv[++i];
      case '--portrait':
        portraitPath = argv[++i];
      case '--out':
        outPath = argv[++i];
    }
  }
  if (snapshotDir == null || portraitPath == null || outPath == null) {
    stderr.writeln(
      'usage: dart run tool/test_face_crop.dart '
      '--snapshot <dir> --portrait <png> --out <png>',
    );
    exit(2);
  }

  final snap = LivePortraitSnapshot.open(snapshotDir);
  final blazePath = snap.pathFor(LivePortraitModule.faceDetector);
  stdout.writeln('using YuNet: $blazePath');

  final pngBytes = File(portraitPath).readAsBytesSync();
  final decoded = img.decodePng(pngBytes);
  if (decoded == null) {
    stderr.writeln('failed to decode PNG: $portraitPath');
    exit(1);
  }
  // Convert to packed RGB uint8.
  final rgb = _imageToRgb(decoded);
  stdout.writeln(
    'portrait: ${decoded.width}x${decoded.height} '
    '(${pngBytes.length} bytes png, ${rgb.length} bytes rgb)',
  );

  final faceCrop = FaceCropService.yunet(onnxPath: blazePath);
  try {
    final stopwatch = Stopwatch()..start();
    final result = faceCrop.cropPortrait(
      sourceRgb: rgb,
      sourceWidth: decoded.width,
      sourceHeight: decoded.height,
    );
    stopwatch.stop();
    stdout.writeln(
      'crop ok: ${result.cropWidth}x${result.cropHeight} '
      'detection_score=${result.detectionScore.toStringAsFixed(3)} '
      'in ${stopwatch.elapsedMilliseconds} ms',
    );
    final lm = result.landmarks;
    stdout.writeln(
      '  landmarks (source px):'
      '\n    leftEye    = (${lm.leftEye.x.toStringAsFixed(1)}, ${lm.leftEye.y.toStringAsFixed(1)})'
      '\n    rightEye   = (${lm.rightEye.x.toStringAsFixed(1)}, ${lm.rightEye.y.toStringAsFixed(1)})'
      '\n    noseTip    = (${lm.noseTip.x.toStringAsFixed(1)}, ${lm.noseTip.y.toStringAsFixed(1)})'
      '\n    leftMouth  = (${lm.leftMouth.x.toStringAsFixed(1)}, ${lm.leftMouth.y.toStringAsFixed(1)})'
      '\n    rightMouth = (${lm.rightMouth.x.toStringAsFixed(1)}, ${lm.rightMouth.y.toStringAsFixed(1)})',
    );

    // Encode crop back to PNG.
    final cropImg = img.Image(width: 512, height: 512);
    for (var y = 0; y < 512; y++) {
      for (var x = 0; x < 512; x++) {
        final idx = (y * 512 + x) * 3;
        cropImg.setPixelRgb(
          x,
          y,
          result.cropRgb[idx],
          result.cropRgb[idx + 1],
          result.cropRgb[idx + 2],
        );
      }
    }
    File(outPath).writeAsBytesSync(img.encodePng(cropImg));
    stdout.writeln('wrote crop -> $outPath');
  } finally {
    faceCrop.close();
  }
}

Uint8List _imageToRgb(img.Image src) {
  final out = Uint8List(src.width * src.height * 3);
  var i = 0;
  for (final pixel in src) {
    out[i++] = pixel.r.toInt();
    out[i++] = pixel.g.toInt();
    out[i++] = pixel.b.toInt();
  }
  return out;
}
