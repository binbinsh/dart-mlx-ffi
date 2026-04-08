/// Qwen3-ASR speed benchmark (Dart / dart-mlx-ffi).
///
/// Measures inference speed excluding model load time, with multiple iterations
/// for reliable timing data. Outputs JSON to stdout, diagnostics to stderr.
///
/// Usage (from dart-mlx-ffi directory):
///   dart run benchmark/qwen3_asr/speed_bench.dart /tmp/speech_test_16k.wav [iters]
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln(
      'Usage: dart run benchmark/qwen3_asr/speed_bench.dart <wav> [iters]',
    );
    exitCode = 1;
    return;
  }

  final wavPath = args[0];
  final iters = args.length > 1 ? int.tryParse(args[1]) ?? 3 : 3;

  final audio = _parseWav(File(wavPath).readAsBytesSync());
  final durationS = audio.length / 16000;
  stderr.writeln(
    'Audio: ${audio.length} samples (${durationS.toStringAsFixed(2)}s)',
  );

  final home = Platform.environment['HOME'] ?? '';
  final modelPath =
      Platform.environment['CMDSPACE_QWEN3_ASR_MODEL']?.trim() ??
      '$home/.cmdspace/models/qwen3-asr/default';

  stderr.writeln('Loading model from $modelPath');
  final loadSw = Stopwatch()..start();
  final runner = Qwen3AsrRunner.load(modelPath);
  loadSw.stop();
  final loadMs = loadSw.elapsedMilliseconds;
  stderr.writeln('Model loaded in ${loadMs}ms');

  // Warmup run
  stderr.writeln('Warmup run...');
  runner.transcribe(audio, maxNewTokens: 448);

  // Get token count from a single run (to avoid double inference).
  final refIds = runner.transcribeToIds(audio, maxNewTokens: 448);
  final nTokens = refIds.length;

  // Timed runs — use transcribe() only (single inference per iteration).
  final results = <Map<String, dynamic>>[];
  for (var i = 0; i < iters; i++) {
    stderr.writeln('Iteration ${i + 1}/$iters...');
    final sw = Stopwatch()..start();
    final text = runner.transcribe(audio, maxNewTokens: 448);
    sw.stop();
    final totalMs = sw.elapsedMilliseconds;

    final tokPerSec = nTokens / (totalMs / 1000);
    results.add({
      'total_inference_ms': totalMs,
      'n_tokens': nTokens,
      'tokens_per_sec': tokPerSec,
      'text_preview': text.length > 80 ? text.substring(0, 80) : text,
    });
    stderr.writeln(
      '  total=${totalMs}ms  tokens=$nTokens  tok/s=${tokPerSec.toStringAsFixed(1)}',
    );
  }

  // Summary
  final avgTotalMs =
      results.fold<int>(0, (s, r) => s + (r['total_inference_ms'] as int)) /
      iters;
  final avgTokPerSec =
      results.fold<double>(0, (s, r) => s + (r['tokens_per_sec'] as double)) /
      iters;
  final rtf = (avgTotalMs / 1000) / durationS;

  stderr.writeln('\n=== Dart Benchmark Summary ($iters iters) ===');
  stderr.writeln('  Audio duration:    ${durationS.toStringAsFixed(2)}s');
  stderr.writeln('  Model load:        ${loadMs}ms');
  stderr.writeln('  Total inference:   ${avgTotalMs.toStringAsFixed(1)}ms');
  stderr.writeln('  RTF:               ${rtf.toStringAsFixed(4)}');
  stderr.writeln('  Tokens:            $nTokens');
  stderr.writeln('  Tokens/sec:        ${avgTokPerSec.toStringAsFixed(1)}');

  final summary = {
    'total_inference_ms': avgTotalMs,
    'n_tokens': nTokens,
    'tokens_per_sec': avgTokPerSec,
    'audio_duration_s': durationS,
    'rtf': rtf,
    'iters': iters,
    'model_load_ms': loadMs,
    'runtime': 'dart',
    'text_preview': results[0]['text_preview'],
  };
  stdout.writeln(jsonEncode(summary));

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
