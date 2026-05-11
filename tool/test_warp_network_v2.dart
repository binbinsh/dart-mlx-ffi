/// Phase 4a smoke: load `warp_network_v2.onnx` (GridSample3D rewritten
/// to standard GridSample @ opset 20) and run a single forward pass
/// with synthetic inputs to verify ORT 1.25.1 accepts it.
///
/// ```sh
/// dart run tool/test_warp_network_v2.dart \
///   --warp ~/Projects/Personal/cmdspace-app/.cache/live_portrait/onnx/warp_network_v2.onnx
/// ```
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

const String _kWarpFamily = 'live_portrait_warp_v2';

Future<void> main(List<String> argv) async {
  String? warpPath;
  for (var i = 0; i < argv.length; i++) {
    if (argv[i] == '--warp') warpPath = argv[++i];
  }
  warpPath ??=
      '/Users/binbinsh/Projects/Personal/cmdspace-app/.cache/live_portrait/onnx/warp_network_v2.onnx';

  if (!File(warpPath).existsSync()) {
    stderr.writeln('warp_network_v2.onnx not found at $warpPath');
    exit(1);
  }

  stdout.writeln('loading: $warpPath');
  final session = DartOnnxSession.load(
    DartOnnxConfig(
      modelPath: warpPath,
      id: _kWarpFamily,
      family: _kWarpFamily,
      provider: 'cpu',
      requireProvider: false,
      numThreads: 4,
      runDiagnostics: true,
    ),
  );
  try {
    final diag = session.diagnostics;
    final inputs = (diag['input_metadata'] as List).cast<Map>();
    final outputs = (diag['output_metadata'] as List).cast<Map>();
    stdout.writeln('  provider: ${session.selectedProvider}');
    stdout.writeln(
      '  inputs : ${inputs.map((m) => "${m['name']}:${m['shape']}").toList()}',
    );
    stdout.writeln(
      '  outputs: ${outputs.map((m) => "${m['name']}:${m['shape']}").toList()}',
    );

    // Synthetic inputs matching expected shapes.
    final feature3d = _randomFloat32(1 * 32 * 16 * 64 * 64, seed: 7);
    final rng = math.Random(11);
    final kpSource = Float32List(1 * 21 * 3);
    final kpDriving = Float32List(1 * 21 * 3);
    for (var i = 0; i < 21 * 3; i++) {
      kpSource[i] = (rng.nextDouble() - 0.5) * 0.8;
      kpDriving[i] = kpSource[i] + (rng.nextDouble() - 0.5) * 0.05;
    }

    final inputMap = <String, Object?>{
      'feature_3d': RuntimeTensor.float32(
        const [1, 32, 16, 64, 64],
        feature3d,
      ),
      'kp_source': RuntimeTensor.float32(const [1, 21, 3], kpSource),
      'kp_driving': RuntimeTensor.float32(const [1, 21, 3], kpDriving),
    };

    stdout.writeln('running forward pass...');
    final sw = Stopwatch()..start();
    final result = session.run(inputMap);
    sw.stop();
    try {
      stdout.writeln('  forward ok in ${sw.elapsedMilliseconds} ms');
      for (final meta in outputs) {
        final name = meta['name'] as String;
        final t = result.outputs[name] as RuntimeTensor;
        final data = Float32List.fromList(t.asFloat32List());
        var minV = double.infinity;
        var maxV = double.negativeInfinity;
        var sumAbs = 0.0;
        var nan = 0;
        for (final x in data) {
          if (x.isNaN) {
            nan++;
            continue;
          }
          if (x < minV) minV = x;
          if (x > maxV) maxV = x;
          sumAbs += x.abs();
        }
        stdout.writeln(
          '  out[$name] shape=${meta['shape']} '
          'len=${data.length} '
          'min=${minV.toStringAsFixed(4)} '
          'max=${maxV.toStringAsFixed(4)} '
          'meanAbs=${(sumAbs / data.length).toStringAsFixed(4)} '
          'nan=$nan',
        );
      }
    } finally {
      result.close();
    }
  } finally {
    session.close();
  }
}

Float32List _randomFloat32(int n, {required int seed}) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (var i = 0; i < n; i++) {
    out[i] = (rng.nextDouble() - 0.5) * 2;
  }
  return out;
}
