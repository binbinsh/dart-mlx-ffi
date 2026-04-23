import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:dart_mlx_ffi/models.dart';

Future<void> main(List<String> args) async {
  if (args.length < 2) {
    stderr.writeln(
      'usage: dart run tool/qwen3_tts_decode.dart <bundle_path> <frames.json> [pcm_out.bin]',
    );
    exitCode = 64;
    return;
  }
  final decoded = jsonDecode(File(args[1]).readAsStringSync()) as Map<String, Object?>;
  final rawFrames = decoded['frames'];
  if (rawFrames is! List) {
    throw StateError('frames.json is missing a frames list');
  }
  final frames = <List<int>>[
    for (final frame in rawFrames)
      [for (final value in (frame as List)) (value as num).toInt()],
  ];
  final engine = Qwen3TtsEngine.load(args[0]);
  final pcm = engine.debugDecodeCodeFrames(frames);
  engine.close();
  final bytes = pcm.buffer.asUint8List(pcm.offsetInBytes, pcm.lengthInBytes);
  if (args.length > 2) {
    File(args[2]).writeAsBytesSync(bytes);
  }
  stdout.writeln(
    jsonEncode(<String, Object?>{
      'samples': pcm.length,
      'sha256': sha256.convert(bytes).toString(),
      'head': pcm.take(16).map((value) => value.toStringAsFixed(8)).toList(),
    }),
  );
}
