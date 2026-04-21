import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/models.dart';

String _arg(List<String> args, String name, {String? fallback}) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return arg.substring(prefix.length);
    }
  }
  if (fallback != null) {
    return fallback;
  }
  throw ArgumentError('Missing $name');
}

int _intArg(List<String> args, String name, {required int fallback}) {
  final raw = _arg(args, name, fallback: '$fallback');
  return int.tryParse(raw) ?? fallback;
}

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

Future<void> main(List<String> args) async {
  final bundlePath = _arg(
    args,
    '--bundle',
    fallback:
        '${Platform.environment['HOME']}/.cmdspace/models/fsmn-vad/default',
  );
  final warmup = _intArg(args, '--warmup', fallback: 3);
  final iters = _intArg(args, '--iters', fallback: 10);
  final frames = _intArg(args, '--frames', fallback: 30);

  final bundle = await loadFsmnVadBundle(bundlePath);
  final runtime = FsmnVadRuntime(bundle);
  final features = _deterministicFeatures(frames, runtime.inputDim);

  for (var i = 0; i < warmup; i += 1) {
    final state = runtime.createState();
    final result = runtime.processFeatures(
      features: features,
      frames: frames,
      state: state,
    );
    result.state.close();
  }

  final watch = Stopwatch()..start();
  FsmnVadFrameResult? timedResult;
  for (var i = 0; i < iters; i += 1) {
    final state = runtime.createState();
    timedResult = runtime.processFeatures(
      features: features,
      frames: frames,
      state: state,
    );
    if (i < iters - 1) {
      timedResult.state.close();
    }
  }
  watch.stop();

  final preview = timedResult!.speechProbabilities
      .take(16)
      .toList(growable: false);
  stdout.writeln(
    jsonEncode(<String, Object?>{
      'per_iter_ms': watch.elapsedMicroseconds / 1000.0 / iters,
      'speech_preview': preview,
      'frames': frames,
    }),
  );

  timedResult.state.close();
  bundle.close();
  exit(0);
}
