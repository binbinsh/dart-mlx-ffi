import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) async {
  final home = Platform.environment['HOME'] ?? '';
  final bundlePath = args.isNotEmpty
      ? args.first
      : '$home/.cmdspace/models/silero-vad/default';

  stderr.writeln('Loading Silero VAD bundle from $bundlePath');
  final bundle = await loadSileroVadBundle(bundlePath);
  try {
    final runtime = SileroVadRuntime(bundle);
    final state = runtime.createState();
    try {
      final samples = Float32List(runtime.frameSamples);
      final result = runtime.processFrame(samples: samples, state: state);
      state.close();
      result.state.close();
      stdout.writeln(
        jsonEncode({
          'bundle_path': bundlePath,
          'frame_samples': runtime.frameSamples,
          'sample_rate': runtime.sampleRate,
          'probability': result.probability,
        }),
      );
      exit(0);
    } catch (error, stackTrace) {
      state.close();
      stderr.writeln(error);
      stderr.writeln(stackTrace);
      exit(1);
    }
  } finally {
    bundle.close();
  }
}
