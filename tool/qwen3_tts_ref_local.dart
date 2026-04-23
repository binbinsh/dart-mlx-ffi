import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/src/models/qwen3_tts/qwen3_tts.dart';

Future<void> main(List<String> args) async {
  if (args.length < 4) {
    stderr.writeln(
      'usage: dart run tool/qwen3_tts_ref_local.dart <bundle_path> <wav_path> <ref_text> <output.json>',
    );
    exitCode = 64;
    return;
  }

  final bundle = Qwen3TtsBundle.load(args[0]);
  try {
    final samples = _readWavMonoFloat32(args[1]);
    final speaker = Qwen3TtsSpeakerEncoder(bundle);
    try {
      final tokenizer = Qwen3TtsTokenizerEncoder(bundle);
      final payload = <String, Object?>{
        'model_path': bundle.manifest.rootPath,
        'audio_path': args[1],
        'ref_text': args[2],
        'speaker_embedding': speaker.embed(samples),
        'ref_codes': [for (final group in tokenizer.encode(samples)) group],
      };
      final encoded = '${const JsonEncoder.withIndent('  ').convert(payload)}\n';
      await File(args[3]).writeAsString(encoded);
      stdout.writeln(encoded.trimRight());
    } finally {
      speaker.close();
    }
  } finally {
    bundle.close();
  }
}

Float32List _readWavMonoFloat32(String path) {
  final bytes = File(path).readAsBytesSync();
  if (bytes.length < 44) {
    throw StateError('WAV file too small: $path');
  }
  final bd = ByteData.sublistView(bytes);
  int channels = 1;
  int bitsPerSample = 16;
  int dataOffset = 44;
  int dataSize = bytes.length - 44;
  var offset = 12;
  while (offset + 8 <= bytes.length) {
    final id = String.fromCharCodes(bytes.sublist(offset, offset + 4));
    final size = bd.getUint32(offset + 4, Endian.little);
    final chunkStart = offset + 8;
    if (id == 'fmt ' && size >= 16) {
      channels = bd.getUint16(chunkStart + 2, Endian.little);
      bitsPerSample = bd.getUint16(chunkStart + 14, Endian.little);
    } else if (id == 'data') {
      dataOffset = chunkStart;
      dataSize = size;
      break;
    }
    offset = chunkStart + size + (size.isOdd ? 1 : 0);
  }
  if (channels != 1) {
    throw StateError('Expected mono WAV, got $channels channels: $path');
  }
  if (bitsPerSample != 16 && bitsPerSample != 32) {
    throw StateError('Unsupported WAV bit depth $bitsPerSample in $path');
  }
  final bytesPerSample = bitsPerSample ~/ 8;
  final frameCount = dataSize ~/ bytesPerSample;
  final out = Float32List(frameCount);
  for (var i = 0; i < frameCount; i++) {
    final base = dataOffset + (i * bytesPerSample);
    if (bitsPerSample == 16) {
      out[i] = (bd.getInt16(base, Endian.little) / 32768.0).clamp(-1.0, 1.0);
    } else {
      out[i] = bd.getFloat32(base, Endian.little).clamp(-1.0, 1.0);
    }
  }
  return out;
}
