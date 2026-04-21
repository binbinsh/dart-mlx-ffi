import 'dart:convert';
import 'dart:io';
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

Future<void> main(List<String> args) async {
  final pcmPath = _arg(args, '--pcm');
  final bundlePath = _arg(
    args,
    '--bundle',
    fallback:
        '${Platform.environment['HOME']}/.cmdspace/models/fsmn-vad/default',
  );
  final warmup = _intArg(args, '--warmup', fallback: 3);
  final iters = _intArg(args, '--iters', fallback: 10);
  final maxSamples = _intArg(args, '--max-samples', fallback: 160000);

  final bytes = await File(pcmPath).readAsBytes();
  final fullAudio = Float32List.sublistView(bytes);
  final useSamples = fullAudio.length > maxSamples
      ? maxSamples
      : fullAudio.length;
  final audio = Float32List.fromList(fullAudio.sublist(0, useSamples));

  final bundle = await loadFsmnVadBundle(bundlePath);
  final runtime = FsmnVadRuntime(bundle);

  for (var i = 0; i < warmup; i += 1) {
    final frontend = FsmnVadFrontend(
      manifest: runtime.manifest,
      cmvn: runtime.cmvn,
    );
    final state = runtime.createState();
    final features = frontend.process(audio, isFinal: true);
    final result = runtime.processFeatures(
      features: features.values,
      frames: features.frames,
      state: state,
    );
    result.state.close();
    frontend.close();
  }

  final watch = Stopwatch()..start();
  FsmnVadFrameResult? timedResult;
  FsmnVadFrontendOutput? timedFeatures;
  for (var i = 0; i < iters; i += 1) {
    final frontend = FsmnVadFrontend(
      manifest: runtime.manifest,
      cmvn: runtime.cmvn,
    );
    final state = runtime.createState();
    final features = frontend.process(audio, isFinal: true);
    final result = runtime.processFeatures(
      features: features.values,
      frames: features.frames,
      state: state,
    );
    if (i < iters - 1) {
      result.state.close();
    } else {
      timedResult = result;
      timedFeatures = features;
    }
    frontend.close();
  }
  watch.stop();

  final preview = timedResult!.speechProbabilities
      .take(16)
      .toList(growable: false);
  stdout.writeln(
    jsonEncode(<String, Object?>{
      'per_iter_ms': watch.elapsedMicroseconds / 1000.0 / iters,
      'speech_preview': preview,
      'frames': timedFeatures!.frames,
      'samples': useSamples,
    }),
  );

  timedResult.state.close();
  bundle.close();
  exit(0);
}
