/// Quick speech transcription test using Qwen3-ASR.
///
/// Usage (from dart-mlx-ffi directory):
///   dart run benchmark/qwen3_asr/speech_test.dart /tmp/speech_test_16k.wav
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln(
      'Usage: dart run benchmark/qwen3_asr/speech_test.dart <wav>',
    );
    exitCode = 1;
    return;
  }

  final wavPath = args[0];
  final wavData = File(wavPath).readAsBytesSync();

  // Parse WAV to get float32 PCM.
  final audio = _parseWav(wavData);
  stderr.writeln('Audio: ${audio.length} samples (${audio.length / 16000}s)');

  final home = Platform.environment['HOME'] ?? '';
  final modelPath =
      Platform.environment['CMDSPACE_QWEN3_ASR_MODEL']?.trim() ??
      '$home/.cmdspace/models/qwen3-asr/default';

  stderr.writeln('Loading model from $modelPath');
  final sw = Stopwatch()..start();
  final runner = Qwen3AsrRunner.load(modelPath);
  sw.stop();
  stderr.writeln('Model loaded in ${sw.elapsedMilliseconds}ms');

  // Transcribe.
  sw
    ..reset()
    ..start();
  final text = runner.transcribe(audio, maxNewTokens: 448);
  sw.stop();
  stderr.writeln('Transcription (${sw.elapsedMilliseconds}ms):');
  stdout.writeln(text);

  // Also get token IDs.
  final ids = runner.transcribeToIds(audio, maxNewTokens: 448);
  stderr.writeln('Token IDs (${ids.length}): $ids');

  runner.close();
}

Float32List _parseWav(Uint8List data) {
  if (data.length < 44) throw FormatException('WAV too short');
  final bd = ByteData.sublistView(data);

  // Find 'data' chunk.
  var pos = 12;
  while (pos + 8 <= data.length) {
    final chunkId = String.fromCharCodes(data.sublist(pos, pos + 4));
    final chunkSize = bd.getUint32(pos + 4, Endian.little);
    if (chunkId == 'data') {
      final audioFormat = bd.getUint16(20, Endian.little);
      final bitsPerSample = bd.getUint16(34, Endian.little);
      final raw = data.sublist(pos + 8, pos + 8 + chunkSize);
      if (audioFormat == 1 && bitsPerSample == 16) {
        // PCM 16-bit
        final samples = Float32List(raw.length ~/ 2);
        final rbd = ByteData.sublistView(raw);
        for (var i = 0; i < samples.length; i++) {
          samples[i] = rbd.getInt16(i * 2, Endian.little) / 32768.0;
        }
        return samples;
      } else if (audioFormat == 3 && bitsPerSample == 32) {
        // IEEE float32
        return Float32List.sublistView(ByteData.sublistView(raw));
      }
      throw FormatException(
        'Unsupported: fmt=$audioFormat bits=$bitsPerSample',
      );
    }
    pos += 8 + chunkSize + (chunkSize % 2);
  }
  throw FormatException('No data chunk found');
}
