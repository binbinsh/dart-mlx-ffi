import 'bpe.dart';
import 'config.dart';

const String qwen3AsrTextTag = '<asr_text>';

/// First-turn prompt: system + user audio placeholders + assistant prefix.
List<int> buildQwen3AsrPromptTokens(
  Qwen3AsrConfig config,
  Qwen3AsrBpeTokenizer tokenizer,
  int nAudioTokens, {
  required String locale,
}) {
  final tokens = <int>[];
  tokens.add(config.imStartTokenId);
  tokens.addAll(tokenizer.encode('system\n'));
  tokens.add(config.imEndTokenId);
  tokens.add(config.newlineTokenId);
  tokens.add(config.imStartTokenId);
  tokens.addAll(tokenizer.encode('user\n'));
  tokens.add(config.audioStartTokenId);
  tokens.addAll(List<int>.filled(nAudioTokens, config.audioPadTokenId));
  tokens.add(config.audioEndTokenId);
  tokens.add(config.imEndTokenId);
  tokens.add(config.newlineTokenId);
  tokens.add(config.imStartTokenId);
  tokens.addAll(tokenizer.encode('assistant\n'));
  final langTokens = qwen3AsrLanguageForcingTokens(tokenizer, locale);
  if (langTokens != null) tokens.addAll(langTokens);
  return tokens;
}

/// Follow-up prompt for chunked/streaming ASR.
List<int> buildQwen3AsrFollowupTokens(
  Qwen3AsrConfig config,
  Qwen3AsrBpeTokenizer tokenizer,
  int nAudioTokens, {
  required String locale,
}) {
  final tokens = <int>[];
  tokens.add(config.imEndTokenId);
  tokens.add(config.newlineTokenId);
  tokens.add(config.imStartTokenId);
  tokens.addAll(tokenizer.encode('user\n'));
  tokens.add(config.audioStartTokenId);
  tokens.addAll(List<int>.filled(nAudioTokens, config.audioPadTokenId));
  tokens.add(config.audioEndTokenId);
  tokens.add(config.imEndTokenId);
  tokens.add(config.newlineTokenId);
  tokens.add(config.imStartTokenId);
  tokens.addAll(tokenizer.encode('assistant\n'));
  final langTokens = qwen3AsrLanguageForcingTokens(tokenizer, locale);
  if (langTokens != null) tokens.addAll(langTokens);
  return tokens;
}

/// Build language forcing suffix tokens, or null if auto-detect.
List<int>? qwen3AsrLanguageForcingTokens(
  Qwen3AsrBpeTokenizer tokenizer,
  String locale,
) {
  final normalized = locale.trim().toLowerCase();
  final langName = switch (normalized) {
    '' || 'auto' => null,
    'zh' || 'zh-cn' || 'zh-hans' => 'Chinese',
    'zh-tw' || 'zh-hant' => 'Chinese',
    'en' || 'en-us' || 'en-gb' => 'English',
    'ja' || 'ja-jp' => 'Japanese',
    'ko' || 'ko-kr' => 'Korean',
    'de' || 'de-de' => 'German',
    'fr' || 'fr-fr' => 'French',
    'es' || 'es-es' || 'es-419' => 'Spanish',
    'ru' || 'ru-ru' => 'Russian',
    'ar' || 'ar-eg' => 'Arabic',
    'hi' || 'hi-in' => 'Hindi',
    'it' || 'it-it' => 'Italian',
    'pt' || 'pt-br' || 'pt-pt' => 'Portuguese',
    'tr' || 'tr-tr' => 'Turkish',
    'nl' || 'nl-nl' => 'Dutch',
    _ => locale.trim(),
  };
  if (langName == null) return null;
  return tokenizer.encode('language $langName$qwen3AsrTextTag');
}

/// Index of the first audio placeholder in the prompt.
int qwen3AsrAudioOffset(List<int> promptIds, Qwen3AsrConfig config) {
  final offset = promptIds.indexOf(config.audioPadTokenId);
  if (offset < 0) {
    throw StateError('Qwen3-ASR prompt contains no audio_pad placeholders.');
  }
  return offset;
}

/// Detect repetitive decode patterns to halt hallucination loops.
bool detectQwen3AsrRepetition(List<int> tokens) {
  if (tokens.length < 20) return false;
  final last = tokens.last;
  var run = 0;
  for (var i = tokens.length - 1; i >= 0 && tokens[i] == last; i--) {
    run++;
  }
  if (run > 20) return true;

  for (var n = 2; n <= 10 && n * 3 <= tokens.length; n++) {
    final pattern = tokens.sublist(tokens.length - n);
    var repeats = 0;
    for (var i = tokens.length - n; i >= n; i -= n) {
      var match = true;
      for (var j = 0; j < n; j++) {
        if (tokens[i - n + j] != pattern[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        repeats++;
      } else {
        break;
      }
    }
    if (repeats >= (30 ~/ n) + 1) return true;
  }
  return false;
}
