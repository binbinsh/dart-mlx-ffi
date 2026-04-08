import 'dart:convert';
import 'dart:io';

const Map<String, int> _fallbackAddedTokens = <String, int>{
  '<|endoftext|>': 151643,
  '<|im_start|>': 151644,
  '<|im_end|>': 151645,
  '<|audio_start|>': 151669,
  '<|audio_end|>': 151670,
  '<|audio_pad|>': 151676,
  '<asr_text>': 151704,
};

/// Byte-level BPE tokenizer compatible with Qwen2/GPT2-style
/// vocab.json + merges.txt format.
///
/// Implements the same algorithm as HuggingFace's
/// `PreTrainedTokenizerFast` for Qwen2 models.
final class Qwen3AsrBpeTokenizer {
  Qwen3AsrBpeTokenizer._({
    required this.encoder,
    required this.decoder,
    required this.bpeRanks,
    required this.byteEncoder,
    required this.byteDecoder,
    required this.eosTokenId,
    required this.padTokenId,
    required this.specialTokens,
  });

  /// Load from a bundle directory containing vocab.json and merges.txt.
  factory Qwen3AsrBpeTokenizer.load(String dirPath) {
    final vocabFile = File('$dirPath/vocab.json');
    final mergesFile = File('$dirPath/merges.txt');
    if (!vocabFile.existsSync()) {
      throw StateError('vocab.json not found in $dirPath');
    }
    if (!mergesFile.existsSync()) {
      throw StateError('merges.txt not found in $dirPath');
    }

    // Parse vocab.json: {"token": id, ...}
    final vocabJson =
        jsonDecode(vocabFile.readAsStringSync()) as Map<String, Object?>;
    final encoder = <String, int>{};
    for (final entry in vocabJson.entries) {
      encoder[entry.key] = (entry.value as num).toInt();
    }
    _mergeAddedTokens(dirPath, encoder);
    for (final entry in _fallbackAddedTokens.entries) {
      encoder.putIfAbsent(entry.key, () => entry.value);
    }

    // Build reverse mapping.
    final decoder = <int, String>{};
    for (final entry in encoder.entries) {
      decoder[entry.value] = entry.key;
    }

    // Parse merges.txt: skip header line starting with #version,
    // then each line is "token1 token2".
    final mergesContent = mergesFile.readAsStringSync();
    final lines = mergesContent.split('\n');
    final bpeRanks = <String, int>{};
    var rank = 0;
    for (final line in lines) {
      if (line.isEmpty || line.startsWith('#')) continue;
      bpeRanks[line] = rank;
      rank += 1;
    }

    // Build byte <-> unicode mappings.
    final byteEncoder = _bytesToUnicode();
    final byteDecoder = <String, int>{};
    for (final entry in byteEncoder.entries) {
      byteDecoder[entry.value] = entry.key;
    }

    // Qwen2 special tokens.
    const eosTokenId = 151645; // <|endoftext|>
    const padTokenId = 151643; // <|endoftext|>
    final specialTokens =
        encoder.keys.where(_looksLikeSpecialToken).toList(growable: false)
          ..sort((a, b) => b.length.compareTo(a.length));

    return Qwen3AsrBpeTokenizer._(
      encoder: encoder,
      decoder: decoder,
      bpeRanks: bpeRanks,
      byteEncoder: byteEncoder,
      byteDecoder: byteDecoder,
      eosTokenId: eosTokenId,
      padTokenId: padTokenId,
      specialTokens: specialTokens,
    );
  }

  final Map<String, int> encoder;
  final Map<int, String> decoder;
  final Map<String, int> bpeRanks;
  final Map<int, String> byteEncoder;
  final Map<String, int> byteDecoder;
  final int eosTokenId;
  final int padTokenId;
  final List<String> specialTokens;

  // BPE merge cache for previously seen tokens.
  final Map<String, List<String>> _bpeCache = {};

  /// Encode text to a list of token IDs.
  List<int> encode(String text) {
    if (text.isEmpty) return const [];
    final tokens = <int>[];
    final plain = StringBuffer();
    var index = 0;

    void flushPlain() {
      if (plain.isEmpty) {
        return;
      }
      final bytes = utf8.encode(plain.toString());
      final unicodeChars = bytes.map((b) => byteEncoder[b]!).join();
      final bpeTokens = _bpe(unicodeChars);
      for (final token in bpeTokens) {
        final id = encoder[token];
        if (id != null) {
          tokens.add(id);
        }
      }
      plain.clear();
    }

    while (index < text.length) {
      final special = _matchSpecialToken(text, index);
      if (special != null) {
        flushPlain();
        final id = encoder[special];
        if (id != null) {
          tokens.add(id);
        }
        index += special.length;
        continue;
      }
      plain.write(text[index]);
      index += 1;
    }

    flushPlain();
    return tokens;
  }

  /// Decode a list of token IDs back to text.
  String decode(List<int> ids, {bool skipSpecial = true}) {
    final buf = StringBuffer();
    for (final id in ids) {
      final token = decoder[id];
      if (token == null) continue;
      if (skipSpecial && _isSpecialToken(token)) continue;
      buf.write(token);
    }
    // Convert byte-level unicode chars back to actual bytes.
    final chars = buf.toString();
    final bytes = <int>[];
    for (var i = 0; i < chars.length; i++) {
      final char = chars[i];
      final byteVal = byteDecoder[char];
      if (byteVal != null) {
        bytes.add(byteVal);
      }
    }
    return utf8.decode(bytes, allowMalformed: true);
  }

  bool _isSpecialToken(String token) {
    return _looksLikeSpecialToken(token);
  }

  String? _matchSpecialToken(String text, int index) {
    for (final token in specialTokens) {
      if (text.startsWith(token, index)) {
        return token;
      }
    }
    return null;
  }

  /// Apply BPE merges to a unicode-encoded word.
  List<String> _bpe(String token) {
    final cached = _bpeCache[token];
    if (cached != null) return cached;

    var word = [for (var i = 0; i < token.length; i++) token[i]];
    if (word.length <= 1) {
      _bpeCache[token] = word;
      return word;
    }

    while (true) {
      // Find the pair with the lowest merge rank.
      int? bestRank;
      int bestIndex = -1;
      for (var i = 0; i < word.length - 1; i++) {
        final pair = '${word[i]} ${word[i + 1]}';
        final rank = bpeRanks[pair];
        if (rank != null && (bestRank == null || rank < bestRank)) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestRank == null) break;

      // Merge the best pair throughout the word.
      final merged = word[bestIndex] + word[bestIndex + 1];
      final newWord = <String>[];
      var i = 0;
      while (i < word.length) {
        if (i == bestIndex) {
          newWord.add(merged);
          i += 2;
        } else {
          newWord.add(word[i]);
          i += 1;
        }
      }
      word = newWord;
      if (word.length == 1) break;
    }

    _bpeCache[token] = word;
    return word;
  }

  /// GPT2-style byte-to-unicode mapping.
  ///
  /// Maps byte values 0-255 to unicode characters, avoiding control
  /// characters and whitespace that would cause issues in BPE.
  static Map<int, String> _bytesToUnicode() {
    final bs = <int>[];
    // Printable ASCII ranges that map to themselves.
    for (var i = 0x21; i <= 0x7E; i++) bs.add(i); // ! through ~
    for (var i = 0xA1; i <= 0xAC; i++) bs.add(i); // ¡ through ¬
    for (var i = 0xAE; i <= 0xFF; i++) bs.add(i); // ® through ÿ

    final cs = List<int>.from(bs);
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n += 1;
      }
    }

    final result = <int, String>{};
    for (var i = 0; i < bs.length; i++) {
      result[bs[i]] = String.fromCharCode(cs[i]);
    }
    return result;
  }

  static bool _looksLikeSpecialToken(String token) {
    return token.length >= 3 &&
        token.startsWith('<') &&
        token.endsWith('>') &&
        !token.contains(' ');
  }

  static void _mergeAddedTokens(String dirPath, Map<String, int> encoder) {
    final file = File('$dirPath/tokenizer_config.json');
    if (!file.existsSync()) {
      return;
    }
    try {
      final root = jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      final added =
          root['added_tokens_decoder'] as Map<String, Object?>? ?? const {};
      for (final entry in added.entries) {
        final id = int.tryParse(entry.key);
        final payload = entry.value;
        if (id == null || payload is! Map<String, Object?>) {
          continue;
        }
        final token = payload['content']?.toString();
        if (token == null || token.isEmpty) {
          continue;
        }
        encoder.putIfAbsent(token, () => id);
      }
    } catch (_) {
      // Fall back to the built-in required token map.
    }
  }
}
