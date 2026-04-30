// Qwen2 byte-level BPE tokenizer used by CosyVoice2 and NeuTTS Air.

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;

import '../../runtime/native_ffi.dart' as dz;

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;

final class Qwen2SpecialToken {
  const Qwen2SpecialToken(this.text, this.id);
  final String text;
  final int id;
}

const List<Qwen2SpecialToken> kCosyVoice2DefaultSpecials = [
  Qwen2SpecialToken('<|endoftext|>', 151643),
  Qwen2SpecialToken('<|im_start|>', 151644),
  Qwen2SpecialToken('<|im_end|>', 151645),
  Qwen2SpecialToken('<|endofprompt|>', 151646),
  Qwen2SpecialToken('[breath]', 151647),
  Qwen2SpecialToken('<strong>', 151648),
  Qwen2SpecialToken('</strong>', 151649),
  Qwen2SpecialToken('[noise]', 151650),
  Qwen2SpecialToken('[laughter]', 151651),
  Qwen2SpecialToken('[cough]', 151652),
  Qwen2SpecialToken('[clucking]', 151653),
  Qwen2SpecialToken('[accent]', 151654),
  Qwen2SpecialToken('[quick_breath]', 151655),
  Qwen2SpecialToken('<laughter>', 151656),
  Qwen2SpecialToken('</laughter>', 151657),
  Qwen2SpecialToken('[hissing]', 151658),
  Qwen2SpecialToken('[sigh]', 151659),
  Qwen2SpecialToken('[vocalized-noise]', 151660),
  Qwen2SpecialToken('[lipsmack]', 151661),
  Qwen2SpecialToken('[mn]', 151662),
];

final class Qwen2BpeTokenizer {
  factory Qwen2BpeTokenizer({
    required Map<String, int> vocab,
    required List<String> merges,
    List<Qwen2SpecialToken> specials = kCosyVoice2DefaultSpecials,
    int? declaredVocabSize,
  }) {
    return Qwen2BpeTokenizer._(
      vocab: Map<String, int>.unmodifiable(vocab),
      merges: _mergeRanks(merges),
      specials: _sortedSpecials(specials),
      vocabSize:
          declaredVocabSize ?? _declaredVocabSize(vocab.values, specials),
    );
  }

  Qwen2BpeTokenizer._({
    required Map<String, int> vocab,
    required Map<String, int> merges,
    required List<Qwen2SpecialToken> specials,
    required this.vocabSize,
  }) : _vocab = vocab,
       _merges = merges,
       _specials = specials;

  factory Qwen2BpeTokenizer.fromSidecarBytes(List<int> sidecarBytes) {
    final loaded = _loadSidecar(utf8.decode(sidecarBytes));
    return Qwen2BpeTokenizer._(
      vocab: loaded.vocab,
      merges: loaded.merges,
      specials: loaded.specials,
      vocabSize: loaded.vocabSize,
    );
  }

  final int vocabSize;
  final Map<String, int> _vocab;
  final Map<String, int> _merges;
  final List<Qwen2SpecialToken> _specials;
  bool _closed = false;

  ffi.Pointer<ffi.Void> get nativeHandle {
    if (_closed) {
      throw StateError('Qwen2 tokenizer is closed.');
    }
    return ffi.nullptr;
  }

  static Future<Qwen2BpeTokenizer> load(
    String tokenizerDir, {
    List<Qwen2SpecialToken> specials = kCosyVoice2DefaultSpecials,
    bool preferSidecar = true,
  }) async {
    final sidecar = File('$tokenizerDir/tokenizer.qwen2bpe');
    if (preferSidecar &&
        await sidecar.exists() &&
        _sameSpecials(specials, kCosyVoice2DefaultSpecials)) {
      return Qwen2BpeTokenizer.loadFromSidecar(sidecar.path);
    }
    final vocabRaw = await File('$tokenizerDir/vocab.json').readAsString();
    final mergesRaw = await File('$tokenizerDir/merges.txt').readAsString();
    final vocab = (jsonDecode(vocabRaw) as Map).map(
      (key, value) => MapEntry('$key', (value as num).toInt()),
    );
    final merges = <String>[];
    for (final line in const LineSplitter().convert(mergesRaw)) {
      if (line.isEmpty || line.startsWith('#')) continue;
      merges.add(line);
    }
    return Qwen2BpeTokenizer(vocab: vocab, merges: merges, specials: specials);
  }

  static Future<Qwen2BpeTokenizer> loadFromSidecar(String sidecarPath) async {
    return Qwen2BpeTokenizer.fromSidecarBytes(
      await File(sidecarPath).readAsBytes(),
    );
  }

  static Future<Qwen2BpeTokenizer> loadFromTokenizerJson(
    String tokenizerJsonPath, {
    List<Qwen2SpecialToken>? specials,
  }) async {
    final raw = await File(tokenizerJsonPath).readAsString();
    final decoded = jsonDecode(raw);
    if (decoded is! Map) {
      throw FormatException('tokenizer.json must be a JSON object.');
    }
    final model = decoded['model'];
    if (model is! Map || model['type'] != 'BPE') {
      throw FormatException('tokenizer.json must contain a BPE model.');
    }
    final vocabRaw = model['vocab'];
    if (vocabRaw is! Map) {
      throw FormatException('tokenizer.json model.vocab must be an object.');
    }
    final vocab = vocabRaw.map(
      (key, value) => MapEntry('$key', (value as num).toInt()),
    );
    final merges = _parseTokenizerJsonMerges(model['merges']);
    final allAdded = _parseTokenizerJsonAddedTokens(decoded['added_tokens']);
    final selectedSpecials = specials ?? allAdded;
    return Qwen2BpeTokenizer(
      vocab: vocab,
      merges: merges,
      specials: selectedSpecials,
      declaredVocabSize: _declaredVocabSize(vocab.values, allAdded),
    );
  }

  List<int> encode(String text, {int? maxLength}) {
    if (_closed) {
      throw StateError('Qwen2 tokenizer is closed.');
    }
    final ids = <int>[];
    _encodeSpecialAware(text, ids);
    if (maxLength != null && ids.length > maxLength) {
      throw StateError(
        'qwen2 bpe output buffer too small (need ${ids.length}, had $maxLength)',
      );
    }
    return List<int>.unmodifiable(ids);
  }

  NativeTensorBuffer encodeInt32Buffer(
    String text, {
    int? maxLength,
    dz.NativeFfi? ffiRuntime,
  }) => encodeInt32BufferNativeFfi(
    text,
    maxLength: maxLength,
    ffiRuntime: ffiRuntime,
  );

  NativeTensorBuffer encodeInt32BufferNativeFfi(
    String text, {
    int? maxLength,
    dz.NativeFfi? ffiRuntime,
  }) {
    final ids = encode(text, maxLength: maxLength);
    final out = NativeTensorBuffer.int32([ids.length], ffiRuntime: ffiRuntime);
    out.asInt32List().setAll(0, ids);
    return out;
  }

  void close() {
    _closed = true;
  }

  void _encodeSpecialAware(String text, List<int> out) {
    var cursor = 0;
    while (cursor < text.length) {
      final match = _findSpecial(text, cursor);
      if (match == null) {
        _encodeNormal(text.substring(cursor), out);
        return;
      }
      if (match.start > cursor) {
        _encodeNormal(text.substring(cursor, match.start), out);
      }
      out.add(match.id);
      cursor = match.end;
    }
  }

  _SpecialMatch? _findSpecial(String text, int from) {
    _SpecialMatch? best;
    for (final special in _specials) {
      final index = text.indexOf(special.text, from);
      if (index < 0) continue;
      final match = _SpecialMatch(
        index,
        index + special.text.length,
        special.id,
      );
      if (best == null ||
          match.start < best.start ||
          (match.start == best.start &&
              match.end - match.start > best.end - best.start)) {
        best = match;
      }
    }
    return best;
  }

  void _encodeNormal(String text, List<int> out) {
    if (text.isEmpty) return;
    final iterator = _PreTokenIterator(text);
    while (true) {
      final segment = iterator.next();
      if (segment == null) break;
      _encodePreToken(segment, out);
    }
  }

  void _encodePreToken(String segment, List<int> out) {
    final encoded = StringBuffer();
    for (final byte in utf8.encode(segment)) {
      encoded.writeCharCode(_byteEncoder[byte]);
    }
    final encodedText = encoded.toString();
    var pieces = [
      for (final rune in encodedText.runes) String.fromCharCode(rune),
    ];
    while (pieces.length > 1) {
      var bestRank = 0x7fffffffffffffff;
      var bestIndex = -1;
      for (var i = 0; i < pieces.length - 1; i += 1) {
        final rank = _merges['${pieces[i]} ${pieces[i + 1]}'];
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      pieces = [
        ...pieces.take(bestIndex),
        pieces[bestIndex] + pieces[bestIndex + 1],
        ...pieces.skip(bestIndex + 2),
      ];
    }
    for (final piece in pieces) {
      final id = _vocab[piece];
      if (id == null) {
        throw StateError('qwen2 bpe encountered a token outside the vocab');
      }
      out.add(id);
    }
  }
}

final class _SpecialMatch {
  const _SpecialMatch(this.start, this.end, this.id);
  final int start;
  final int end;
  final int id;
}

final class _LoadedSidecar {
  const _LoadedSidecar({
    required this.vocab,
    required this.merges,
    required this.specials,
    required this.vocabSize,
  });
  final Map<String, int> vocab;
  final Map<String, int> merges;
  final List<Qwen2SpecialToken> specials;
  final int vocabSize;
}

_LoadedSidecar _loadSidecar(String sidecar) {
  if (!sidecar.startsWith('qwen2bpe\t1\n')) {
    throw StateError('qwen2 bpe sidecar is invalid');
  }
  final vocab = <String, int>{};
  final merges = <String, int>{};
  final specials = <Qwen2SpecialToken>[];
  var declaredVocabSize = 0;
  var maxId = -1;
  var mergeRank = 0;
  for (final raw in const LineSplitter().convert(sidecar)) {
    final line = raw.endsWith('\r') ? raw.substring(0, raw.length - 1) : raw;
    if (line.isEmpty || line == 'qwen2bpe\t1') continue;
    final fields = line.split('\t');
    switch (fields.first) {
      case 'meta':
        if (fields.length >= 3 && fields[1] == 'declared_vocab_size') {
          declaredVocabSize = int.parse(fields[2]);
        }
      case 'v':
        if (fields.length < 3) throw StateError('qwen2 bpe sidecar is invalid');
        final id = int.parse(fields[1]);
        vocab[utf8.decode(_decodeHex(fields[2]))] = id;
        maxId = math.max(maxId, id);
      case 'm':
        if (fields.length < 3) throw StateError('qwen2 bpe sidecar is invalid');
        final left = utf8.decode(_decodeHex(fields[1]));
        final right = utf8.decode(_decodeHex(fields[2]));
        merges['$left $right'] = mergeRank;
        mergeRank += 1;
      case 's':
        if (fields.length < 3) throw StateError('qwen2 bpe sidecar is invalid');
        final id = int.parse(fields[1]);
        specials.add(Qwen2SpecialToken(utf8.decode(_decodeHex(fields[2])), id));
        maxId = math.max(maxId, id);
      default:
        throw StateError('qwen2 bpe sidecar is invalid');
    }
  }
  return _LoadedSidecar(
    vocab: Map<String, int>.unmodifiable(vocab),
    merges: Map<String, int>.unmodifiable(merges),
    specials: _sortedSpecials(specials),
    vocabSize: math.max(declaredVocabSize, maxId + 1),
  );
}

Map<String, int> _mergeRanks(List<String> merges) {
  return {for (var i = 0; i < merges.length; i += 1) merges[i]: i};
}

List<Qwen2SpecialToken> _sortedSpecials(List<Qwen2SpecialToken> specials) {
  final out = specials.toList();
  out.sort((a, b) {
    final byLength = b.text.length.compareTo(a.text.length);
    return byLength != 0 ? byLength : a.id.compareTo(b.id);
  });
  return List<Qwen2SpecialToken>.unmodifiable(out);
}

int _declaredVocabSize(
  Iterable<int> vocabIds,
  Iterable<Qwen2SpecialToken> specials,
) {
  var maxId = -1;
  for (final id in vocabIds) {
    if (id > maxId) maxId = id;
  }
  for (final special in specials) {
    if (special.id > maxId) maxId = special.id;
  }
  return maxId + 1;
}

bool _sameSpecials(List<Qwen2SpecialToken> a, List<Qwen2SpecialToken> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i += 1) {
    if (a[i].id != b[i].id || a[i].text != b[i].text) return false;
  }
  return true;
}

List<String> _parseTokenizerJsonMerges(Object? raw) {
  if (raw is! List) {
    throw FormatException('tokenizer.json model.merges must be a list.');
  }
  return [
    for (final entry in raw)
      if (entry is String)
        entry
      else if (entry is List && entry.length == 2)
        '${entry[0]} ${entry[1]}'
      else
        throw FormatException('Invalid tokenizer.json BPE merge entry: $entry'),
  ];
}

List<Qwen2SpecialToken> _parseTokenizerJsonAddedTokens(Object? raw) {
  if (raw == null) return const [];
  if (raw is! List) {
    throw FormatException('tokenizer.json added_tokens must be a list.');
  }
  final tokens = <Qwen2SpecialToken>[];
  for (final entry in raw) {
    if (entry is! Map) {
      throw FormatException('Invalid tokenizer.json added token: $entry');
    }
    final id = entry['id'];
    final content = entry['content'];
    if (id is! num || content is! String) {
      throw FormatException('Invalid tokenizer.json added token: $entry');
    }
    tokens.add(Qwen2SpecialToken(content, id.toInt()));
  }
  return tokens;
}

List<int> _decodeHex(String value) {
  if (value.length.isOdd) {
    throw StateError('qwen2 bpe sidecar is invalid');
  }
  return [
    for (var i = 0; i < value.length; i += 2)
      int.parse(value.substring(i, i + 2), radix: 16),
  ];
}

final List<int> _byteEncoder = List<int>.generate(256, (byte) {
  final visible =
      (byte >= 0x21 && byte <= 0x7e) ||
      (byte >= 0xa1 && byte <= 0xac) ||
      (byte >= 0xae && byte <= 0xff);
  if (visible) return byte;
  var visibleCount = 0;
  for (var i = 0; i < byte; i += 1) {
    final previousVisible =
        (i >= 0x21 && i <= 0x7e) ||
        (i >= 0xa1 && i <= 0xac) ||
        (i >= 0xae && i <= 0xff);
    if (!previousVisible) visibleCount += 1;
  }
  return 0x100 + visibleCount;
});

final class _RuneAt {
  const _RuneAt(this.value, this.start, this.end);
  final int value;
  final int start;
  final int end;
}

final class _PreTokenIterator {
  _PreTokenIterator(this.text) : _runes = _scanRunes(text);

  final String text;
  final List<_RuneAt> _runes;
  var _cursorRune = 0;

  String? next() {
    if (_cursorRune >= _runes.length) return null;
    final startRune = _cursorRune;
    final endRune = _matchOne();
    final start = _runes[startRune].start;
    final end = _runes[endRune - 1].end;
    _cursorRune = endRune;
    return text.substring(start, end);
  }

  int _matchOne() {
    final contraction = _matchContraction();
    if (contraction != null) return contraction;
    final letters = _matchOptSymThenLetters();
    if (letters != null) return letters;
    if (_isNumber(_runes[_cursorRune].value)) return _cursorRune + 1;
    final symbols = _matchOptSpaceSymbols();
    if (symbols != null) return symbols;
    final newlines = _matchSpacesThenNewlines();
    if (newlines != null) return newlines;
    final trailing = _matchTrailingWhitespace();
    if (trailing != null) return trailing;
    final whitespace = _matchWhitespaceRun();
    if (whitespace != null) return whitespace;
    return _cursorRune + 1;
  }

  int? _matchContraction() {
    final start = _cursorRune;
    if (_runes[start].value != 0x27) return null;
    final rest = text.substring(_runes[start].end).toLowerCase();
    for (final suffix in const ['re', 've', 'll', 's', 't', 'm', 'd']) {
      if (rest.startsWith(suffix)) {
        var end = start + 1;
        for (var i = 0; i < suffix.length && end < _runes.length; i += 1) {
          end += 1;
        }
        return end;
      }
    }
    return null;
  }

  int? _matchOptSymThenLetters() {
    var pos = _cursorRune;
    if (pos < _runes.length) {
      final cp = _runes[pos].value;
      if (cp != 0x0d && cp != 0x0a && !_isLetterOrNumber(cp)) {
        final after = pos + 1;
        if (after < _runes.length && _isLetter(_runes[after].value)) {
          pos = after;
        }
      }
    }
    var consumed = false;
    while (pos < _runes.length && _isLetter(_runes[pos].value)) {
      consumed = true;
      pos += 1;
    }
    return consumed ? pos : null;
  }

  int? _matchOptSpaceSymbols() {
    var pos = _cursorRune;
    if (pos < _runes.length && _runes[pos].value == 0x20) pos += 1;
    var count = 0;
    while (pos < _runes.length) {
      final cp = _runes[pos].value;
      if (_isWhitespace(cp) || _isLetterOrNumber(cp)) break;
      count += 1;
      pos += 1;
    }
    if (count == 0) return null;
    while (pos < _runes.length &&
        (_runes[pos].value == 0x0d || _runes[pos].value == 0x0a)) {
      pos += 1;
    }
    return pos;
  }

  int? _matchSpacesThenNewlines() {
    var pos = _cursorRune;
    var sawNewline = false;
    while (pos < _runes.length && _isWhitespace(_runes[pos].value)) {
      final cp = _runes[pos].value;
      if (cp == 0x0d || cp == 0x0a) {
        sawNewline = true;
      } else if (sawNewline) {
        break;
      }
      pos += 1;
    }
    return sawNewline ? pos : null;
  }

  int? _matchTrailingWhitespace() {
    var end = _cursorRune;
    while (end < _runes.length && _isWhitespace(_runes[end].value)) {
      end += 1;
    }
    if (end == _cursorRune) return null;
    if (end == _runes.length) return end;
    return end - 1 == _cursorRune ? null : end - 1;
  }

  int? _matchWhitespaceRun() {
    var pos = _cursorRune;
    while (pos < _runes.length && _isWhitespace(_runes[pos].value)) {
      pos += 1;
    }
    return pos == _cursorRune ? null : pos;
  }
}

List<_RuneAt> _scanRunes(String text) {
  final out = <_RuneAt>[];
  var offset = 0;
  for (final rune in text.runes) {
    final length = rune > 0xffff ? 2 : 1;
    out.add(_RuneAt(rune, offset, offset + length));
    offset += length;
  }
  return out;
}

bool _isAsciiAlpha(int cp) =>
    (cp >= 0x41 && cp <= 0x5a) || (cp >= 0x61 && cp <= 0x7a);

bool _isNumber(int cp) => cp >= 0x30 && cp <= 0x39;

bool _isLetter(int cp) {
  if (_isAsciiAlpha(cp)) return true;
  if (_isNumber(cp) || _isWhitespace(cp)) return false;
  if (cp < 0x80) return false;
  return true;
}

bool _isLetterOrNumber(int cp) => _isLetter(cp) || _isNumber(cp);

bool _isWhitespace(int cp) {
  return cp == 0x09 ||
      cp == 0x0a ||
      cp == 0x0b ||
      cp == 0x0c ||
      cp == 0x0d ||
      cp == 0x20 ||
      cp == 0x85 ||
      cp == 0xa0 ||
      cp == 0x1680 ||
      (cp >= 0x2000 && cp <= 0x200a) ||
      cp == 0x2028 ||
      cp == 0x2029 ||
      cp == 0x202f ||
      cp == 0x205f ||
      cp == 0x3000;
}
