// CosyVoice2 voice-prompt smoke: load campplus + speech_tokenizer_v2,
// feed a synthetic 16 kHz waveform, and print the resulting
// SpeakerPrompt sizes / first values. This proves the pure-Dart
// extractors + ONNX call wiring matches the upstream ZeroShot path.

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/models.dart';

void main(List<String> args) async {
  final modelDir = _getOpt(args, '--model-dir') ??
      '/home/binbinsh/Projects/unifrontend/src/ttsbackends/providers/cosyvoice2/models/CosyVoice2-0.5B';
  final provider = _getOpt(args, '--provider') ?? 'cpu';
  final paths = CosyVoice2Paths(modelDir: modelDir);
  stdout.writeln('==> loading speech_tokenizer_v2 + campplus');
  final bundle = CosyVoice2PartialOnnxBundle.load(
    paths: paths,
    provider: provider,
    deviceId: 0,
    requireProvider: false,
    numThreads: 4,
    componentNames: const {'speech_tokenizer_v2', 'campplus'},
  );
  for (final s in bundle.statuses) {
    if (s.file.name == 'speech_tokenizer_v2' || s.file.name == 'campplus') {
      stdout.writeln(
        '   ${s.file.name}: exists=${s.exists}, loaded=${s.loaded}, '
        'provider=${s.selectedProvider}'
        '${s.error != null ? ", error=${s.error}" : ""}',
      );
    }
  }

  // 3-second 440 Hz tone @ 16k as the prompt.
  final sr = 16000;
  final n = sr * 3;
  final audio = Float32List(n);
  for (var i = 0; i < n; i += 1) {
    audio[i] = 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sr);
  }

  final extractor = SpeakerPromptExtractor(bundle: bundle);
  try {
    final t0 = DateTime.now().microsecondsSinceEpoch;
    final p = extractor.extract(audio, sr);
    final t1 = DateTime.now().microsecondsSinceEpoch;
    stdout.writeln('==> SpeakerPrompt extracted in '
        '${(t1 - t0) / 1000.0} ms');
    stdout.writeln('   speakerEmbedding.shape  = [${p.speakerEmbedding.length}]');
    stdout.writeln('   first 4 emb values      = '
        '${p.speakerEmbedding.sublist(0, 4)}');
    stdout.writeln('   promptSpeechTokens.len  = ${p.promptSpeechTokens.length}');
    stdout.writeln('   first 8 token ids       = '
        '${p.promptSpeechTokens.sublist(0, p.promptSpeechTokens.length < 8 ? p.promptSpeechTokens.length : 8)}');
    stdout.writeln('   promptSpeechFeat.shape  = '
        '[${p.promptSpeechFeatFrames}, 80]');
    stdout.writeln('==> SMOKE OK');
  } catch (error, stack) {
    stderr.writeln('SMOKE FAILED: $error');
    stderr.writeln(stack);
    exitCode = 1;
  }
}

String? _getOpt(List<String> args, String name) {
  for (var i = 0; i < args.length - 1; i += 1) {
    if (args[i] == name) return args[i + 1];
  }
  return null;
}
