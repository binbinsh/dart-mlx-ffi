import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/src/models/qwen3_tts/qwen3_tts.dart';

Future<void> main(List<String> args) async {
  if (args.length < 2) {
    stderr.writeln(
      'usage: dart run tool/qwen3_tts_speaker.dart <bundle_path> <wav_path>',
    );
    exitCode = 64;
    return;
  }
  final bundle = Qwen3TtsBundle.load(args[0]);
  try {
    final encoder = Qwen3TtsSpeakerEncoder(bundle);
    final samples = _readWavMonoFloat32(args[1]);
    final embedding = encoder.embed(samples);
    final l2 = math.sqrt(embedding.fold<double>(0.0, (sum, v) => sum + v * v));
    stdout.writeln(
      jsonEncode(<String, Object?>{
        'sample_rate': encoder.sampleRate,
        'embedding_dim': encoder.embeddingDim,
        'l2': l2,
        'head': embedding.take(16).toList(),
      }),
    );
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
  if (bitsPerSample != 16 && bitsPerSample != 32) {
    throw StateError('Unsupported WAV bit depth $bitsPerSample in $path');
  }
  final bytesPerSample = bitsPerSample ~/ 8;
  final frameSize = bytesPerSample * channels;
  final frameCount = dataSize ~/ frameSize;
  final out = Float32List(frameCount);
  for (var i = 0; i < frameCount; i++) {
    final base = dataOffset + (i * frameSize);
    if (bitsPerSample == 16) {
      final sample = bd.getInt16(base, Endian.little) / 32768.0;
      out[i] = sample.clamp(-1.0, 1.0);
    } else {
      final sample = bd.getFloat32(base, Endian.little);
      out[i] = sample.clamp(-1.0, 1.0);
    }
  }
  return out;
}
