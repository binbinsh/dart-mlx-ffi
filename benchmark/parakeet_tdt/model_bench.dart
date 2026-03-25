import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
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
  final melPath = _arg(args, '--mel');
  final metaPath = _arg(args, '--meta');
  final bundlePath = _arg(
    args,
    '--bundle',
    fallback:
        '${Platform.environment['HOME']}/.cmdspace/models/parakeet-tdt/default',
  );
  final warmup = _intArg(args, '--warmup', fallback: 3);
  final iters = _intArg(args, '--iters', fallback: 10);

  final meta =
      jsonDecode(await File(metaPath).readAsString()) as Map<String, dynamic>;
  final shape = (meta['shape'] as List).cast<int>();
  final bytes = await File(melPath).readAsBytes();
  final melData = Float32List.sublistView(bytes);
  final mel = MlxArray.fromFloat32List(melData, shape: shape).astype(
    MlxDType.MLX_FLOAT32,
  );
  final bundle = await loadParakeetTdtBundle(bundlePath);
  final runtime = ParakeetTdtRuntime(bundle, maxEncoderLayers: 24);
  final encoder = ParakeetTdtEncoder(bundle, maxLayers: 24);

  for (var i = 0; i < warmup; i += 1) {
    runtime.transcribeMel(mel, lengths: <int>[shape[1]]);
  }

  final watch = Stopwatch()..start();
  var text = '';
  for (var i = 0; i < iters; i += 1) {
    text = runtime.transcribeMel(mel, lengths: <int>[shape[1]]);
  }
  watch.stop();

  final encoded = encoder(mel, <int>[shape[1]]);
  final zero = runtime.zeroPredictState();
  final predictor = runtime.predictorStep(tokenId: null, state: zero);
  final frame0 = encoded.features.slice(
    start: <int>[0, 0, 0],
    stop: <int>[1, 1, bundle.manifest.encoderHidden],
  );
  final joint = runtime.jointStep(
    encoderFrame: frame0,
    predictorOutput: predictor.output,
  );
  final tokenPreview = joint.tokenLogits
      .astype(MlxDType.MLX_FLOAT32)
      .reshape(<int>[joint.tokenLogits.size])
      .toFloat32List()
      .take(16)
      .toList(growable: false);
  final durationValues = joint.durationLogits
      .astype(MlxDType.MLX_FLOAT32)
      .reshape(<int>[joint.durationLogits.size])
      .toFloat32List()
      .toList(growable: false);

  stdout.writeln(
    jsonEncode(<String, Object?>{
      'text': text,
      'per_iter_ms': watch.elapsedMicroseconds / 1000.0 / iters,
      'token_preview': tokenPreview,
      'duration_logits': durationValues,
    }),
  );

  joint.tokenLogits.close();
  joint.durationLogits.close();
  frame0.close();
  predictor.output.close();
  predictor.state.close();
  zero.close();
  encoded.features.close();
  mel.close();
  exit(0);
}
