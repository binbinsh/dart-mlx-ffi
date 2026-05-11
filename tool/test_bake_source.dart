/// Smoke test for LivePortrait source bake (Phase 2).
///
/// End-to-end: portrait PNG → YuNet face crop → appearance + motion
/// extractors → [SourceState]. Prints shape / range sanity for every
/// produced tensor.
///
/// ```sh
/// dart run tool/test_bake_source.dart \
///   --snapshot ~/Projects/Personal/cmdspace-app/.cache/live_portrait \
///   --portrait ~/Projects/Personal/cmdspace-app/assets/buddies/girlfriend/looks/look_02/portrait.png
/// ```
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:image/image.dart' as img;

Future<void> main(List<String> argv) async {
  String? snapshotDir;
  String? portraitPath;
  for (var i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--snapshot':
        snapshotDir = argv[++i];
      case '--portrait':
        portraitPath = argv[++i];
    }
  }
  if (snapshotDir == null || portraitPath == null) {
    stderr.writeln(
      'usage: dart run tool/test_bake_source.dart '
      '--snapshot <dir> --portrait <png>',
    );
    exit(2);
  }

  final snap = LivePortraitSnapshot.open(snapshotDir);
  final detectorPath = snap.pathFor(LivePortraitModule.faceDetector);
  stdout.writeln('snapshot: $snapshotDir');
  stdout.writeln('  face detector : $detectorPath');
  stdout.writeln('  appearance    : ${snap.pathFor(LivePortraitModule.appearance)}');
  stdout.writeln('  motion        : ${snap.pathFor(LivePortraitModule.motion)}');

  final pngBytes = File(portraitPath).readAsBytesSync();
  final decoded = img.decodePng(pngBytes);
  if (decoded == null) {
    stderr.writeln('failed to decode PNG: $portraitPath');
    exit(1);
  }
  final rgb = _imageToRgb(decoded);
  stdout.writeln(
    'portrait: ${decoded.width}x${decoded.height} rgb_bytes=${rgb.length}',
  );

  final engine = LivePortraitEngine.load(
    snapshotDir: snapshotDir,
    faceDetectorOnnxPath: detectorPath,
  );

  try {
    final sw = Stopwatch()..start();
    final source = await engine.bakePortrait(
      portraitRgb: rgb,
      width: decoded.width,
      height: decoded.height,
    );
    sw.stop();
    stdout.writeln('bake ok in ${sw.elapsedMilliseconds} ms');
    _dumpStats('appearanceVolume', source.appearanceVolume,
        expectedLen: 1 * 32 * 16 * 64 * 64);
    _dumpStats('canonicalKeypoints', source.canonicalKeypoints,
        expectedLen: 63);
    _dumpStats('rotation', source.rotation, expectedLen: 9);
    _dumpStats('translation', source.translation, expectedLen: 3);
    _dumpStats('expression', source.expression, expectedLen: 63);
    stdout.writeln('  scale = ${source.scale.toStringAsFixed(4)}');
    stdout.writeln(
      '  euler (deg) pitch=${source.pitchDeg.toStringAsFixed(2)} '
      'yaw=${source.yawDeg.toStringAsFixed(2)} '
      'roll=${source.rollDeg.toStringAsFixed(2)}',
    );

    // Quick sanity: rotation should be ~ orthonormal (det ≈ ±1).
    final det = _det3(source.rotation);
    stdout.writeln('  rotation determinant = ${det.toStringAsFixed(4)} '
        '(expect ≈ ±1.0)');
  } finally {
    engine.dispose();
  }
}

void _dumpStats(String name, Float32List data, {required int expectedLen}) {
  if (data.length != expectedLen) {
    stdout.writeln(
      '  $name: LENGTH MISMATCH ${data.length} != $expectedLen',
    );
    return;
  }
  var minV = data[0];
  var maxV = data[0];
  var sum = 0.0;
  var nan = 0;
  for (final v in data) {
    if (v.isNaN) {
      nan++;
      continue;
    }
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
    sum += v;
  }
  final mean = sum / data.length;
  stdout.writeln(
    '  $name: len=${data.length} '
    'min=${minV.toStringAsFixed(4)} '
    'max=${maxV.toStringAsFixed(4)} '
    'mean=${mean.toStringAsFixed(4)}'
    '${nan > 0 ? "  NaN=$nan" : ""}',
  );
}

double _det3(Float32List m) {
  // Row-major 3x3.
  return m[0] * (m[4] * m[8] - m[5] * m[7]) -
      m[1] * (m[3] * m[8] - m[5] * m[6]) +
      m[2] * (m[3] * m[7] - m[4] * m[6]);
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
