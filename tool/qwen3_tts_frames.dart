import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/models.dart';

Future<void> main(List<String> args) async {
  if (args.length < 4) {
    stderr.writeln(
      'usage: dart run tool/qwen3_tts_frames.dart <bundle> <prepared_ref.json> <text> <output.json> [repetition_penalty]',
    );
    exitCode = 64;
    return;
  }
  final repetitionPenalty = args.length > 4 ? double.parse(args[4]) : 1.5;
  final engine = Qwen3TtsEngine.load(args[0]);
  engine.setPreparedReference(Qwen3TtsPreparedReference.load(args[1]));
  final frames = engine.debugGenerateFrames(
    args[2],
    temperature: 0.0,
    repetitionPenalty: repetitionPenalty,
  );
  engine.close();
  File(args[3]).writeAsStringSync(
    const JsonEncoder.withIndent('  ').convert(<String, Object?>{
      'frames': frames,
      'frame_count': frames.length,
      'groups': frames.isEmpty ? 0 : frames.first.length,
    }),
  );
  stdout.writeln(
    jsonEncode(<String, Object?>{
      'frame_count': frames.length,
      'groups': frames.isEmpty ? 0 : frames.first.length,
      'first3': frames.take(3).toList(),
    }),
  );
}
