/// Phase 4b smoke: end-to-end identity render.
///
/// portrait PNG → bake source state → render with drive == source →
/// dump 512×512 PNG. With identity driving and stitching enabled, the
/// output should be visually close to the source face crop (warp +
/// decode round-trip is lossy but should not be unrecognizable).
///
/// ```sh
/// dart run tool/test_render_identity.dart \
///   --snapshot ~/Projects/Personal/cmdspace-app/.cache/live_portrait \
///   --portrait ~/Projects/Personal/cmdspace-app/assets/buddies/girlfriend/looks/look_02/portrait.png \
///   --out /tmp/lp_identity.png
/// ```
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:image/image.dart' as img;

Future<void> main(List<String> argv) async {
  String? snapshotDir;
  String? portraitPath;
  var outPath = '/tmp/lp_identity.png';
  var noStitch = false;
  for (var i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--snapshot':
        snapshotDir = argv[++i];
      case '--portrait':
        portraitPath = argv[++i];
      case '--out':
        outPath = argv[++i];
      case '--no-stitch':
        noStitch = true;
    }
  }
  if (snapshotDir == null || portraitPath == null) {
    stderr.writeln(
      'usage: dart run tool/test_render_identity.dart '
      '--snapshot <dir> --portrait <png> [--out <png>] [--no-stitch]',
    );
    exit(2);
  }

  final snap = LivePortraitSnapshot.open(snapshotDir);
  stdout.writeln('snapshot: $snapshotDir');
  stdout.writeln('  warp    : ${snap.pathFor(LivePortraitModule.warp)}');
  stdout.writeln('  decoder : ${snap.pathFor(LivePortraitModule.decoder)}');
  stdout.writeln('  stitch  : ${snap.pathFor(LivePortraitModule.stitch)}');

  final detectorPath = snap.pathFor(LivePortraitModule.faceDetector);

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
  final renderer = PortraitRenderer.mlx(
    config: snap.config,
    snapshot: snap,
    stitchEnabled: !noStitch,
  );

  try {
    var sw = Stopwatch()..start();
    final source = await engine.bakePortrait(
      portraitRgb: rgb,
      width: decoded.width,
      height: decoded.height,
    );
    sw.stop();
    stdout.writeln('bake source: ${sw.elapsedMilliseconds} ms');

    final drive = Driving.identity(source);
    stdout.writeln(
      'identity drive: pitch=${drive.pitchDeg.toStringAsFixed(2)} '
      'yaw=${drive.yawDeg.toStringAsFixed(2)} '
      'roll=${drive.rollDeg.toStringAsFixed(2)} '
      'scale=${drive.scale.toStringAsFixed(3)}',
    );

    sw = Stopwatch()..start();
    final frame = renderer.render(source: source, drive: drive);
    sw.stop();
    stdout.writeln(
      'render frame: ${sw.elapsedMilliseconds} ms '
      '(${frame.width}x${frame.height}, ${frame.rgb.length} bytes) '
      'stitch=${!noStitch}',
    );

    _dumpRgbStats('frame', frame.rgb);

    final outImg = img.Image(width: frame.width, height: frame.height);
    var i = 0;
    for (var y = 0; y < frame.height; y++) {
      for (var x = 0; x < frame.width; x++) {
        outImg.setPixelRgb(
          x,
          y,
          frame.rgb[i],
          frame.rgb[i + 1],
          frame.rgb[i + 2],
        );
        i += 3;
      }
    }
    File(outPath).writeAsBytesSync(img.encodePng(outImg));
    stdout.writeln('wrote: $outPath');
  } finally {
    renderer.close();
    engine.dispose();
  }
}

void _dumpRgbStats(String name, Uint8List rgb) {
  if (rgb.isEmpty) {
    stdout.writeln('  $name: empty');
    return;
  }
  var rMin = 255, gMin = 255, bMin = 255;
  var rMax = 0, gMax = 0, bMax = 0;
  var rSum = 0, gSum = 0, bSum = 0;
  final pixels = rgb.length ~/ 3;
  for (var i = 0; i < pixels; i++) {
    final r = rgb[i * 3];
    final g = rgb[i * 3 + 1];
    final b = rgb[i * 3 + 2];
    if (r < rMin) rMin = r;
    if (g < gMin) gMin = g;
    if (b < bMin) bMin = b;
    if (r > rMax) rMax = r;
    if (g > gMax) gMax = g;
    if (b > bMax) bMax = b;
    rSum += r;
    gSum += g;
    bSum += b;
  }
  stdout.writeln(
    '  $name pixels=$pixels '
    'r=$rMin..$rMax μ=${(rSum / pixels).toStringAsFixed(1)} '
    'g=$gMin..$gMax μ=${(gSum / pixels).toStringAsFixed(1)} '
    'b=$bMin..$bMax μ=${(bSum / pixels).toStringAsFixed(1)}',
  );
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
