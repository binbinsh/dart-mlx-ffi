import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dart_mlx_ffi/models.dart';

Future<void> main(List<String> args) async {
  if (args.length < 3) {
    stderr.writeln(
      'usage: dart run tool/qwen3_tts_smoke.dart <bundle_path> <prepared_ref.json> <text> [repetition_penalty]',
    );
    exitCode = 64;
    return;
  }

  final repetitionPenalty = args.length > 3 ? double.parse(args[3]) : 1.5;
  final engine = Qwen3TtsEngine.load(args[0]);
  engine.setPreparedReference(Qwen3TtsPreparedReference.load(args[1]));
  final stopwatch = Stopwatch()..start();
  final pcm = <double>[];
  await for (
    final chunk in engine.synthesiseStream(
      args[2],
      temperature: 0.0,
      repetitionPenalty: repetitionPenalty,
    )
  ) {
    pcm.addAll(chunk.pcm);
  }
  stopwatch.stop();
  final typed = Float32List.fromList(pcm);
  engine.close();

  final bytes = typed.buffer.asUint8List(
    typed.offsetInBytes,
    typed.lengthInBytes,
  );

  stdout.writeln(
    jsonEncode(<String, Object?>{
      'sample_rate': engine.sampleRate,
      'samples': typed.length,
      'elapsed_ms': stopwatch.elapsedMilliseconds,
      'sha256': sha256.convert(bytes).toString(),
      'head': typed.take(16).map((value) => value.toStringAsFixed(8)).toList(),
    }),
  );
}
