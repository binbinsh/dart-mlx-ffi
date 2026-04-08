/// Qwen3-ASR streaming (chunk-based) verification test.
///
/// Splits a WAV file into 2-second chunks and decodes incrementally using
/// [Qwen3AsrRunner.decodeChunk] with persistent [AsrStreamState].
///
/// Verifies:
///   1. KV cache position advances correctly across chunks
///   2. Each chunk produces non-empty output
///   3. Concatenated output is coherent text
///   4. Compares streaming vs one-shot output quality
///
/// Usage (from dart-mlx-ffi directory):
///   dart run benchmark/qwen3_asr/stream_test.dart /tmp/speech_test_16k.wav
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

const int _sampleRate = 16000;
const double _chunkDurationS = 2.0;

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln(
      'Usage: dart run benchmark/qwen3_asr/stream_test.dart <wav>',
    );
    exitCode = 1;
    return;
  }

  final audio = _parseWav(File(args[0]).readAsBytesSync());
  final durationS = audio.length / _sampleRate;
  stderr.writeln(
    'Audio: ${audio.length} samples (${durationS.toStringAsFixed(2)}s)',
  );

  final home = Platform.environment['HOME'] ?? '';
  final modelPath =
      Platform.environment['CMDSPACE_QWEN3_ASR_MODEL']?.trim() ??
      '$home/.cmdspace/models/qwen3-asr/default';

  stderr.writeln('Loading model from $modelPath');
  final sw = Stopwatch()..start();
  final runner = Qwen3AsrRunner.load(modelPath);
  sw.stop();
  stderr.writeln('Model loaded in ${sw.elapsedMilliseconds}ms');

  // --- One-shot reference ---
  stderr.writeln('\n=== One-shot transcription (reference) ===');
  sw
    ..reset()
    ..start();
  final oneshotText = runner.transcribe(audio, maxNewTokens: 448);
  sw.stop();
  stderr.writeln('  Time: ${sw.elapsedMilliseconds}ms');
  stderr.writeln('  Text: "$oneshotText"');

  // --- Streaming (chunk-based) ---
  stderr.writeln('\n=== Streaming transcription (2s chunks) ===');
  final chunkSamples = (_chunkDurationS * _sampleRate).toInt();
  final nChunks = (audio.length / chunkSamples).ceil();
  stderr.writeln('  Chunk size: $chunkSamples samples (${_chunkDurationS}s)');
  stderr.writeln('  Number of chunks: $nChunks');

  final state = runner.createStreamState();
  final allTokenIds = <int>[];
  final allTexts = <String>[];
  var prevPosition = 0;

  sw
    ..reset()
    ..start();
  for (var c = 0; c < nChunks; c++) {
    final start = c * chunkSamples;
    final end = (start + chunkSamples) > audio.length
        ? audio.length
        : start + chunkSamples;
    final chunk = Float32List.sublistView(audio, start, end);

    final chunkSw = Stopwatch()..start();
    final result = runner.decodeChunk(chunk, state: state, maxNewTokens: 200);
    chunkSw.stop();

    final newPosition = state.position;
    stderr.writeln(
      '  Chunk ${c + 1}/$nChunks: '
      'pos $prevPosition->$newPosition (+${newPosition - prevPosition}), '
      '${result.tokenIds.length} tokens, '
      '${chunkSw.elapsedMilliseconds}ms',
    );
    stderr.writeln('    Text: "${result.text}"');
    stderr.writeln('    IDs:  ${result.tokenIds}');

    // Verify position advanced.
    if (newPosition <= prevPosition) {
      stderr.writeln('  FAIL: position did not advance!');
      exitCode = 1;
    }

    allTokenIds.addAll(result.tokenIds);
    allTexts.add(result.text);
    prevPosition = newPosition;
  }
  sw.stop();
  state.close();

  final streamTotalText = allTexts.join();
  stderr.writeln('\n=== Streaming results ===');
  stderr.writeln('  Total time: ${sw.elapsedMilliseconds}ms');
  stderr.writeln('  Total tokens: ${allTokenIds.length}');
  stderr.writeln('  Combined text: "$streamTotalText"');

  // --- Comparison ---
  stderr.writeln('\n=== Comparison ===');
  stderr.writeln('  One-shot text:  "$oneshotText"');
  stderr.writeln('  Streaming text: "$streamTotalText"');

  // Streaming won't produce identical text to one-shot (since each chunk
  // is decoded independently with its own context), but we verify:
  // 1. Streaming produced non-empty output
  // 2. KV position tracked correctly
  // 3. Each chunk generated at least some tokens
  final allChunksProducedTokens = allTexts.every((t) => t.isNotEmpty);
  stderr.writeln('  All chunks produced tokens: $allChunksProducedTokens');
  stderr.writeln('  Final KV position: $prevPosition');
  stderr.writeln('  Streaming produced output: ${streamTotalText.isNotEmpty}');

  final pass =
      allChunksProducedTokens && streamTotalText.isNotEmpty && prevPosition > 0;
  stderr.writeln('\n${pass ? "PASS" : "FAIL"}: Streaming verification');
  exitCode = pass ? 0 : 1;

  runner.close();
}

Float32List _parseWav(Uint8List data) {
  if (data.length < 44) throw FormatException('WAV too short');
  final bd = ByteData.sublistView(data);
  var pos = 12;
  while (pos + 8 <= data.length) {
    final chunkId = String.fromCharCodes(data.sublist(pos, pos + 4));
    final chunkSize = bd.getUint32(pos + 4, Endian.little);
    if (chunkId == 'data') {
      final audioFormat = bd.getUint16(20, Endian.little);
      final bitsPerSample = bd.getUint16(34, Endian.little);
      final raw = data.sublist(pos + 8, pos + 8 + chunkSize);
      if (audioFormat == 1 && bitsPerSample == 16) {
        final samples = Float32List(raw.length ~/ 2);
        final rbd = ByteData.sublistView(raw);
        for (var i = 0; i < samples.length; i++) {
          samples[i] = rbd.getInt16(i * 2, Endian.little) / 32768.0;
        }
        return samples;
      } else if (audioFormat == 3 && bitsPerSample == 32) {
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
