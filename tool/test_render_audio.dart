/// Phase 5 smoke: end-to-end portrait + audio → animated PNGs.
///
/// portrait PNG → bake source state → push synthetic 16 kHz audio →
/// HuBERT → LMDM → list of MotionFrames → render each → dump first
/// 3 frames as PNGs.
///
/// ```sh
/// dart run tool/test_render_audio.dart \
///   --snapshot ~/Projects/Personal/cmdspace-app/.cache/live_portrait \
///   --portrait ~/Projects/Personal/cmdspace-app/assets/buddies/girlfriend/looks/look_02/portrait.png \
///   --out-dir /tmp/lp_audio
/// ```
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:image/image.dart' as img;

Future<void> main(List<String> argv) async {
  String? snapshotDir;
  String? portraitPath;
  var outDir = '/tmp/lp_audio';
  var maxFrames = 3;
  var samplingTimesteps = 10;
  var audioSeconds = 3.2;
  for (var i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--snapshot':
        snapshotDir = argv[++i];
      case '--portrait':
        portraitPath = argv[++i];
      case '--out-dir':
        outDir = argv[++i];
      case '--max-frames':
        maxFrames = int.parse(argv[++i]);
      case '--steps':
        samplingTimesteps = int.parse(argv[++i]);
      case '--seconds':
        audioSeconds = double.parse(argv[++i]);
    }
  }
  if (snapshotDir == null || portraitPath == null) {
    stderr.writeln(
      'usage: dart run tool/test_render_audio.dart '
      '--snapshot <dir> --portrait <png> [--out-dir <dir>] '
      '[--max-frames N] [--steps N] [--seconds N]',
    );
    exit(2);
  }

  Directory(outDir).createSync(recursive: true);

  final snap = LivePortraitSnapshot.open(snapshotDir);
  final detectorPath = snap.pathFor(LivePortraitModule.faceDetector);

  final pngBytes = File(portraitPath).readAsBytesSync();
  final decoded = img.decodePng(pngBytes);
  if (decoded == null) {
    stderr.writeln('failed to decode portrait PNG');
    exit(1);
  }
  final rgb = _imageToRgb(decoded);
  stdout.writeln('portrait: ${decoded.width}x${decoded.height}');

  final engine = LivePortraitEngine.load(
    snapshotDir: snapshotDir,
    faceDetectorOnnxPath: detectorPath,
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
    engine.setActiveSource(source);

    final samples = (audioSeconds * 16000).toInt();
    final wav = _syntheticAudio(samples);
    stdout.writeln(
      'synthesizing ${audioSeconds.toStringAsFixed(2)}s audio = $samples samples',
    );

    // Drive engine via the streaming animate() API to mirror real usage.
    final frames = <RenderedFrame>[];
    final stream = engine.animate();
    final sub = stream.listen((f) {
      frames.add(f);
    });

    sw = Stopwatch()..start();
    // Push audio. Phase 3.5: pushAudio is synchronous and emits all
    // motion frames + renders inside the call.
    engine.pushAudio(wav, maxRenderFrames: maxFrames);
    // Allow microtasks (StreamController.add) to flush.
    await Future<void>.delayed(Duration.zero);
    sw.stop();
    stdout.writeln(
      'audio→render: ${sw.elapsedMilliseconds} ms '
      '(${frames.length} frames; ~'
      '${(sw.elapsedMilliseconds / frames.length).toStringAsFixed(1)} ms/frame)',
    );

    final dump = math.min(maxFrames, frames.length);
    for (var i = 0; i < dump; i++) {
      final outPath = '$outDir/frame_${i.toString().padLeft(3, '0')}.png';
      final f = frames[i];
      final image = img.Image(width: f.width, height: f.height);
      var p = 0;
      for (var y = 0; y < f.height; y++) {
        for (var x = 0; x < f.width; x++) {
          image.setPixelRgb(x, y, f.rgb[p], f.rgb[p + 1], f.rgb[p + 2]);
          p += 3;
        }
      }
      File(outPath).writeAsBytesSync(img.encodePng(image));
      stdout.writeln('  wrote $outPath');
    }

    await sub.cancel();
    stdout.writeln('steps=$samplingTimesteps');
  } finally {
    engine.dispose();
  }
}

Float32List _syntheticAudio(int samples) {
  // 220 Hz sine, low amplitude — gives HuBERT something to chew on
  // without producing weird out-of-distribution motion.
  final out = Float32List(samples);
  const freq = 220.0;
  for (var i = 0; i < samples; i++) {
    out[i] = 0.05 * math.sin(2 * math.pi * freq * i / 16000.0);
  }
  return out;
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
