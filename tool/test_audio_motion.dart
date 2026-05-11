/// Smoke test for HuBERT + LMDM (Phase 3, offline mode).
///
/// Bakes look_02 → packs the source motion latent (kp_cond) → encodes
/// a generated 3.2s sine waveform with HuBERT → builds the 1103-dim
/// audio cond → samples LMDM → prints sanity stats per output frame
/// (scale delta + recovered Euler angles).
///
/// ```sh
/// dart run tool/test_audio_motion.dart \
///   --snapshot ~/Projects/Personal/cmdspace-app/.cache/live_portrait \
///   --portrait ~/Projects/Personal/cmdspace-app/assets/buddies/girlfriend/looks/look_02/portrait.png
/// ```
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/src/models/live_portrait/extractors/motion.dart'
    show bin66ToDegree;
import 'package:image/image.dart' as img;

Future<void> main(List<String> argv) async {
  String? snapshotDir;
  String? portraitPath;
  int? samplingSteps;
  for (var i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--snapshot':
        snapshotDir = argv[++i];
      case '--portrait':
        portraitPath = argv[++i];
      case '--steps':
        samplingSteps = int.parse(argv[++i]);
    }
  }
  if (snapshotDir == null || portraitPath == null) {
    stderr.writeln(
      'usage: dart run tool/test_audio_motion.dart '
      '--snapshot <dir> --portrait <png> [--steps N]',
    );
    exit(2);
  }
  final steps = samplingSteps ?? 10;

  final snap = LivePortraitSnapshot.open(snapshotDir);
  final detectorPath = snap.pathFor(LivePortraitModule.faceDetector);
  final hubertPath = snap.pathFor(LivePortraitModule.hubert);
  final lmdmPath = snap.pathFor(LivePortraitModule.lmdm);
  stdout.writeln('snapshot: $snapshotDir');
  stdout.writeln('  hubert : $hubertPath');
  stdout.writeln('  lmdm   : $lmdmPath');

  // 1) bake source via the engine (gives us 265-dim motion latent
  //    seed + 63-dim source canonical kp).
  final pngBytes = File(portraitPath).readAsBytesSync();
  final decoded = img.decodePng(pngBytes);
  if (decoded == null) {
    stderr.writeln('failed to decode PNG');
    exit(1);
  }
  final rgb = _imageToRgb(decoded);
  final engine = LivePortraitEngine.load(
    snapshotDir: snapshotDir,
    faceDetectorOnnxPath: detectorPath,
  );

  Float32List? motionLatents;
  try {
    final source = await engine.bakePortrait(
      portraitRgb: rgb,
      width: decoded.width,
      height: decoded.height,
    );

    final kpCond = packSourceMotionLatent(
      scale: source.scale,
      pitchBins: source.pitchBins,
      yawBins: source.yawBins,
      rollBins: source.rollBins,
      translation: source.translation,
      expression: source.expression,
    );
    stdout.writeln('source motion latent: len=${kpCond.length}');

    // 2) HuBERT — encode 3.2s of generated audio.
    final hubert = HubertEncoder.load(onnxPath: hubertPath);
    try {
      const seconds = 80 / kMotionFps; // 3.2s
      final samples = (seconds * kHubertSampleRate).round();
      final wav = _syntheticAudio(samples);
      final hubertSw = Stopwatch()..start();
      final feats = hubert.encode(wav);
      hubertSw.stop();
      stdout.writeln(
        'hubert ok: frames=${feats.frameCount} '
        'feat_len=${feats.features.length} '
        'in ${hubertSw.elapsedMilliseconds} ms',
      );

      // 3) Audio cond.
      final audioCond = buildAudioCondTensor(
        hubert: feats.features,
        sourceCanonical: source.canonicalKeypoints,
      );
      stdout.writeln('audio cond: len=${audioCond.length} (expect 80*1103=88240)');

      // 4) LMDM sample.
      final lmdm = LmdmSampler.load(onnxPath: lmdmPath);
      try {
        final lmdmSw = Stopwatch()..start();
        motionLatents = lmdm.sample(
          kpCond: kpCond,
          audioCond: audioCond,
          samplingTimesteps: steps,
          rng: math.Random(42),
        );
        lmdmSw.stop();
        final mots = motionLatents;
        stdout.writeln(
          'lmdm ok: out_len=${mots.length} '
          '(expect 80*265=21200) '
          'steps=$steps in ${lmdmSw.elapsedMilliseconds} ms '
          '(${(lmdmSw.elapsedMilliseconds / steps).toStringAsFixed(1)} ms/step)',
        );

        // Per-frame sanity: recover Euler angles for frames 0, 20, 40, 60, 79.
        for (final f in const [0, 20, 40, 60, 79]) {
          final frame = sliceMotionFrame(mots, f);
          final unp = unpackMotionLatent(frame);
          final pitch = bin66ToDegree(unp.pitchBins);
          final yaw = bin66ToDegree(unp.yawBins);
          final roll = bin66ToDegree(unp.rollBins);
          stdout.writeln(
            '  frame[$f]: scale=${unp.scale.toStringAsFixed(3)} '
            'pitch=${pitch.toStringAsFixed(2)}° '
            'yaw=${yaw.toStringAsFixed(2)}° '
            'roll=${roll.toStringAsFixed(2)}° '
            't=(${unp.translation[0].toStringAsFixed(3)}, '
            '${unp.translation[1].toStringAsFixed(3)}, '
            '${unp.translation[2].toStringAsFixed(3)}) '
            'exp_l2=${_l2(unp.expression).toStringAsFixed(3)}',
          );
        }
      } finally {
        lmdm.close();
      }
    } finally {
      hubert.close();
    }
  } finally {
    engine.dispose();
  }
}

double _l2(Float32List v) {
  var s = 0.0;
  for (final x in v) {
    s += x * x;
  }
  return math.sqrt(s);
}

Float32List _syntheticAudio(int samples) {
  // Mix of sines to create a plausible speech-like envelope. Not real
  // speech — LMDM will likely produce small motion, but enough to
  // verify the pipeline doesn't NaN.
  final out = Float32List(samples);
  for (var i = 0; i < samples; i++) {
    final t = i / kHubertSampleRate;
    final env = 0.5 * (1 + math.sin(2 * math.pi * 2.0 * t));
    final s1 = math.sin(2 * math.pi * 200 * t);
    final s2 = math.sin(2 * math.pi * 600 * t);
    out[i] = (env * (0.3 * s1 + 0.2 * s2)).clamp(-1.0, 1.0);
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
