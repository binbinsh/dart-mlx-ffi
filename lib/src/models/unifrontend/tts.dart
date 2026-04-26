import 'dart:typed_data';

import '../kokoro/kokoro.dart';
import 'structured_frontend.dart';

final class UniFrontendKokoroTtsResult {
  const UniFrontendKokoroTtsResult({
    required this.text,
    required this.frontendText,
    required this.ttsText,
    required this.phonemes,
    required this.audioWav,
    required this.frontendElapsedMicroseconds,
    required this.ttsElapsedMicroseconds,
    required this.frontendProvider,
    required this.kokoroProvider,
    required this.requestedVoice,
    required this.resolvedVoice,
    required this.phonemeTokenCount,
    required this.phonemeChunkCount,
    required this.warnings,
  });

  final String text;
  final String frontendText;
  final String ttsText;
  final String phonemes;
  final Uint8List audioWav;
  final int frontendElapsedMicroseconds;
  final int ttsElapsedMicroseconds;
  final String frontendProvider;
  final String kokoroProvider;
  final String requestedVoice;
  final String resolvedVoice;
  final int phonemeTokenCount;
  final int phonemeChunkCount;
  final List<String> warnings;
}

final class UniFrontendKokoroTtsRuntime {
  const UniFrontendKokoroTtsRuntime({
    required this.frontend,
    required this.kokoro,
    required this.phonemizer,
  });

  final DartStructuredFrontendRuntime frontend;
  final KokoroDartRuntime kokoro;
  final KokoroPhonemizer phonemizer;

  List<String> get voiceNames => kokoro.voiceNames;

  String get frontendProvider => frontend.selectedProvider;

  String get kokoroProvider => kokoro.selectedProvider;

  String get phonemizerBackend => phonemizer.backendName;

  bool get phonemizerProcessFallbackAllowed => phonemizer.allowProcessFallback;

  Future<UniFrontendKokoroTtsResult> synthesize({
    required String text,
    String phonemes = '',
    String voice = 'zf_xiaoni',
    double speed = 1.0,
  }) async {
    final normalizedText = text.trim();
    var normalizedPhonemes = phonemes.trim();
    StructuredFrontendResult? frontendResult;
    var ttsText = normalizedText;
    var frontendElapsed = 0;
    var frontendText = normalizedText.isEmpty
        ? composeSsml(normalizedPhonemes, FrontendIr())
        : composeSsml(normalizedText, FrontendIr());

    if (normalizedPhonemes.isEmpty) {
      if (normalizedText.isEmpty) {
        throw const FormatException('text or phonemes is required');
      }
      frontendResult = frontend.process(normalizedText);
      frontendElapsed = frontendResult.elapsedMicroseconds;
      frontendText = frontendResult.ssml;
      ttsText = frontendResult.ttsText.trim().isEmpty
          ? normalizedText
          : frontendResult.ttsText;
      final lang = looksChinese(ttsText) ? 'cmn' : 'en-us';
      normalizedPhonemes = await phonemizer.phonemizeSsml(
        frontendText,
        language: lang,
      );
    }
    normalizedPhonemes = kokoro.filterPhonemes(normalizedPhonemes);
    if (normalizedPhonemes.isEmpty) {
      throw const FormatException('phonemes produced no Kokoro token ids');
    }
    final resolvedVoice = kokoro.resolveVoice(voice);
    final phonemeTokenCount = kokoro.phonemeTokenCount(normalizedPhonemes);
    final phonemeChunkCount = kokoro.phonemeChunkCount(normalizedPhonemes);
    final warnings = <String>[
      if (resolvedVoice != voice)
        'voice "$voice" is unavailable; using "$resolvedVoice"',
    ];

    final ttsTimer = Stopwatch()..start();
    final audio = kokoro.synthesizePhonemes(
      phonemes: normalizedPhonemes,
      voice: voice,
      speed: speed,
    );
    ttsTimer.stop();

    return UniFrontendKokoroTtsResult(
      text: normalizedText,
      frontendText: frontendText,
      ttsText: ttsText,
      phonemes: normalizedPhonemes,
      audioWav: audio,
      frontendElapsedMicroseconds: frontendElapsed,
      ttsElapsedMicroseconds: ttsTimer.elapsedMicroseconds,
      frontendProvider: frontendResult?.provider ?? frontend.selectedProvider,
      kokoroProvider: kokoro.selectedProvider,
      requestedVoice: voice,
      resolvedVoice: resolvedVoice,
      phonemeTokenCount: phonemeTokenCount,
      phonemeChunkCount: phonemeChunkCount,
      warnings: warnings,
    );
  }

  void close() {
    frontend.close();
    kokoro.close();
    phonemizer.dispose();
  }
}

bool looksChinese(String text) => RegExp(r'[\u4e00-\u9fff]').hasMatch(text);
