/// Smoke test for [LivePortraitSnapshot.open]. Points at the local
/// converted snapshot under `<cmdspace-app>/.cache/live_portrait/` and
/// prints the per-module sizes pulled from the manifest. Useful as a
/// quick post-conversion sanity check.
///
/// Run from the dart-inference repo root:
///
/// ```sh
/// dart run tool/inspect_live_portrait_snapshot.dart \
///   --snapshot ~/Projects/Personal/cmdspace-app/.cache/live_portrait
/// ```
library;

import 'dart:io';

import 'package:dart_inference/models.dart';

void main(List<String> argv) {
  String? snapshotDir;
  for (var i = 0; i < argv.length; i++) {
    if (argv[i] == '--snapshot' && i + 1 < argv.length) {
      snapshotDir = argv[i + 1];
    }
  }
  if (snapshotDir == null) {
    stderr.writeln(
      'usage: dart run tool/inspect_live_portrait_snapshot.dart '
      '--snapshot <path>',
    );
    exit(2);
  }

  final snap = LivePortraitSnapshot.open(snapshotDir);
  stdout.writeln('source       : ${snap.config.source}');
  stdout.writeln(
    'frame        : ${snap.config.render.frameWidth}x'
    '${snap.config.render.frameHeight} @${snap.config.render.fpsTarget}fps',
  );
  stdout.writeln('keypoints    : ${snap.config.motion.keypointCount}');
  stdout.writeln(
    'audio        : ${snap.config.audio.sampleRate}Hz '
    'hop=${snap.config.audio.hopFrames} '
    'dim=${snap.config.audio.featureDim}',
  );
  stdout.writeln(
    'sampler      : ${snap.config.sampler.kind} '
    'steps=${snap.config.sampler.steps} '
    'window=${snap.config.sampler.windowFrames}',
  );
  stdout.writeln('--- modules ---');
  var totalBytes = 0;
  for (final m in LivePortraitModule.values) {
    final meta = snap.metaFor(m);
    final mb = meta == null
        ? '?'
        : (meta.bytes / (1024 * 1024)).toStringAsFixed(2);
    if (meta != null) totalBytes += meta.bytes;
    stdout.writeln(
      '  ${m.name.padRight(13)} ${mb.padLeft(9)} MB  '
      '${snap.pathFor(m)}',
    );
  }
  stdout.writeln(
    '--- total      ${(totalBytes / (1024 * 1024)).toStringAsFixed(2)} MB',
  );
}
