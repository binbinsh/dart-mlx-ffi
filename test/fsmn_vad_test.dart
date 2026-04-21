import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/models.dart';
import 'package:test/test.dart';

Float32List _deterministicFeatures(int frames, int dims) {
  final out = Float32List(frames * dims);
  for (var t = 0; t < frames; t += 1) {
    for (var d = 0; d < dims; d += 1) {
      out[(t * dims) + d] =
          (0.1 * math.sin(((t + 1) * 0.17) + (d * 0.013))) +
          (0.05 * math.cos(((t + 1) * 0.07) - (d * 0.019)));
    }
  }
  return out;
}

void main() {
  test('FSMN-VAD bundle loads when present', () async {
    final home =
        Platform.environment['HOME'] ??
        Platform.environment['USERPROFILE'] ??
        Directory.current.path;
    final bundleDir = Directory('$home/.cmdspace/models/fsmn-vad/default');
    if (!bundleDir.existsSync()) {
      return;
    }

    final bundle = await loadFsmnVadBundle(bundleDir.path);
    addTearDown(bundle.close);
    expect(bundle.manifest.inputDim, 400);
    expect(bundle.manifest.fsmnLayers, 4);
    expect(bundle.cmvn.offsets.length, 400);
    expect(bundle.cmvn.scales.length, 400);
  });

  test('FSMN-VAD runtime returns bounded speech probabilities', () async {
    final home =
        Platform.environment['HOME'] ??
        Platform.environment['USERPROFILE'] ??
        Directory.current.path;
    final bundleDir = Directory('$home/.cmdspace/models/fsmn-vad/default');
    if (!bundleDir.existsSync()) {
      return;
    }

    final bundle = await loadFsmnVadBundle(bundleDir.path);
    addTearDown(bundle.close);
    final runtime = FsmnVadRuntime(bundle);
    final state = runtime.createState();
    addTearDown(state.close);

    final result = runtime.processFeatures(
      features: _deterministicFeatures(30, runtime.inputDim),
      frames: 30,
      state: state,
    );
    addTearDown(result.state.close);

    expect(result.speechProbabilities.length, 30);
    for (final probability in result.speechProbabilities) {
      expect(probability, inInclusiveRange(0.0, 1.0));
    }
  });
}
