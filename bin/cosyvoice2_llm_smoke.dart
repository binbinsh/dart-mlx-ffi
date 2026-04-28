// CosyVoice2 end-to-end LLM smoke test (pure Dart).
//
// Loads the split-LLM ONNX bundle (`llm_prefill`, `llm_decode`,
// `llm_decoder_head`) plus the embedding NPZ + Qwen2 tokenizer, and
// drives a short autoregressive decode loop.  The goal is to prove the
// Dart -> dinf C ABI -> Zig -> ONNX Runtime path is wired correctly for
// every component the production serving path will need.
//
// Usage:
//   dart run bin/cosyvoice2_llm_smoke.dart \
//       --model-dir <path to CosyVoice2-0.5B> \
//       [--text "你好，世界"] [--steps 5] [--provider cuda] [--device-id 0]
//
// Sampling is greedy argmax (no temperature/top-k); this is deliberate
// so the smoke output is fully deterministic and useful for CI sanity
// checks, not for reference-quality TTS.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/models.dart';

void main(List<String> args) async {
  final opts = _parseArgs(args);
  final paths = CosyVoice2Paths(modelDir: opts.modelDir);

  stdout.writeln('==> loading partial ONNX bundle from ${paths.modelDir}');
  final bundle = CosyVoice2PartialOnnxBundle.load(
    paths: paths,
    provider: opts.provider,
    deviceId: opts.deviceId,
    requireProvider: false,
    numThreads: opts.numThreads,
    componentNames: const {'llm_prefill', 'llm_decode', 'llm_decoder_head'},
  );

  // Surface load diagnostics for every component we asked for, even on
  // failure — these messages are the most useful CI signal.
  for (final s in bundle.statuses) {
    if (s.file.name == 'llm_prefill' ||
        s.file.name == 'llm_decode' ||
        s.file.name == 'llm_decoder_head') {
      stdout.writeln(
        '   ${s.file.name}: exists=${s.exists}, loaded=${s.loaded}, '
        'provider=${s.selectedProvider}, '
        'loadMs=${(s.loadElapsedMicroseconds ?? 0) / 1000.0}'
        '${s.error != null ? ", error=${s.error}" : ""}',
      );
    }
  }

  stdout.writeln('==> constructing CosyVoice2LlmDriver');
  final driver =
      await CosyVoice2LlmDriver.load(bundle: bundle, paths: paths);
  try {
    stdout.writeln('==> tokenizing text: ${jsonEncode(opts.text)}');
    final ids = driver.tokenizer.encode(opts.text);
    stdout.writeln('   text token ids (len ${ids.length}): $ids');

    final embeds = driver.embedTextTokens(ids);
    final prefillStart = DateTime.now().microsecondsSinceEpoch;
    final state = driver.prefill(inputsEmbeds: embeds, seqLen: ids.length);
    final prefillEnd = DateTime.now().microsecondsSinceEpoch;
    stdout.writeln(
      '==> prefill: ${(prefillEnd - prefillStart) / 1000.0} ms '
      '(seq=${ids.length}, totalSeq=${state.totalSeq})',
    );

    final logits0 = driver.headLogits(state.lastHidden);
    final t0 = _argmax(logits0);
    stdout.writeln('   step 0 sampled speech-token id: $t0');

    final stepLatencies = <double>[];
    final sampled = <int>[t0];
    var current = t0;
    for (var step = 1; step <= opts.steps; step += 1) {
      final embed = driver.embedSpeechToken(current);
      final s0 = DateTime.now().microsecondsSinceEpoch;
      driver.decodeStep(state: state, nextEmbed: embed);
      final s1 = DateTime.now().microsecondsSinceEpoch;
      stepLatencies.add((s1 - s0) / 1000.0);
      final logits = driver.headLogits(state.lastHidden);
      current = _argmax(logits);
      sampled.add(current);
    }
    stdout.writeln(
      '==> decode loop: ${opts.steps} steps, latencies (ms)='
      '${stepLatencies.map((v) => v.toStringAsFixed(2)).toList()}',
    );
    stdout.writeln('   sampled speech tokens: $sampled');

    stdout.writeln('==> SMOKE OK');
  } catch (error, stack) {
    stderr.writeln('SMOKE FAILED: $error');
    stderr.writeln(stack);
    exitCode = 1;
  } finally {
    driver.close();
  }
}

int _argmax(Float32List values) {
  // The head emits `[1, seq, 6564]`; we always feed it a single position
  // so values.length == 6564 in practice, but defensively scan the
  // entire buffer.
  var bestIdx = 0;
  var best = values[0];
  for (var i = 1; i < values.length; i += 1) {
    if (values[i] > best) {
      best = values[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

class _Opts {
  _Opts({
    required this.modelDir,
    required this.text,
    required this.steps,
    required this.provider,
    required this.deviceId,
    required this.numThreads,
  });
  final String modelDir;
  final String text;
  final int steps;
  final String provider;
  final int deviceId;
  final int numThreads;
}

_Opts _parseArgs(List<String> argv) {
  String? modelDir;
  var text = '你好，世界';
  var steps = 5;
  var provider = 'cuda';
  var deviceId = 0;
  var numThreads = 4;
  for (var i = 0; i < argv.length; i += 1) {
    final a = argv[i];
    String next() {
      if (i + 1 >= argv.length) {
        throw ArgumentError('Missing value for $a');
      }
      i += 1;
      return argv[i];
    }

    switch (a) {
      case '--model-dir':
        modelDir = next();
      case '--text':
        text = next();
      case '--steps':
        steps = int.parse(next());
      case '--provider':
        provider = next();
      case '--device-id':
        deviceId = int.parse(next());
      case '--num-threads':
        numThreads = int.parse(next());
      case '-h':
      case '--help':
        stdout.writeln(
            'Usage: cosyvoice2_llm_smoke --model-dir <path> [--text ...] [--steps N] [--provider cuda|cpu] [--device-id N] [--num-threads N]');
        exit(0);
      default:
        throw ArgumentError('Unknown flag: $a');
    }
  }
  if (modelDir == null) {
    throw ArgumentError('--model-dir is required');
  }
  return _Opts(
    modelDir: modelDir,
    text: text,
    steps: steps,
    provider: provider,
    deviceId: deviceId,
    numThreads: numThreads,
  );
}
