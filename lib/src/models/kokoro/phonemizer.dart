import 'dart:async';

import 'espeak.dart';

typedef KokoroPhonemeBackend =
    FutureOr<String> Function(String text, {required String language});

const _ssmlPlain = 1;
const _ssmlExplicit = 2;
const _ssmlPinyin = 3;

final class KokoroPhonemizer {
  KokoroPhonemizer({
    this.espeakLibraryPath,
    this.espeakDataPath,
    int cacheSize = 512,
  }) : _backend = null,
       _cache = _LruCache<String, String>(cacheSize);

  KokoroPhonemizer.withBackend(
    KokoroPhonemeBackend backend, {
    int cacheSize = 512,
  }) : espeakLibraryPath = null,
       espeakDataPath = null,
       _backend = backend,
       _cache = _LruCache<String, String>(cacheSize);

  final String? espeakLibraryPath;
  final String? espeakDataPath;

  final KokoroPhonemeBackend? _backend;
  final _LruCache<String, String> _cache;
  EspeakG2p? _ffi;

  String get backendName => _backend != null
      ? 'injected'
      : _ffi != null
      ? 'espeak_native'
      : 'lazy';

  Future<String> phonemize(String text, {String language = 'en-us'}) async {
    final normalized = text.trim();
    if (normalized.isEmpty) {
      return '';
    }
    final requestedLanguage = _canonicalLanguage(language);
    final input = _normalizeTextForLanguage(normalized, requestedLanguage);
    final route = _kokoroRoute(input, requestedLanguage);
    final espeakLanguage = route.language;
    final cacheKey = '$espeakLanguage\u0000$input';
    final cached = _cache[cacheKey];
    if (cached != null) {
      return cached;
    }

    final raw = route.mixed
        ? await _phonemizeMixedUncached(input, espeakLanguage)
        : _postProcess(
            await _phonemizeUncached(input, espeakLanguage),
            espeakLanguage,
          );
    final phonemes = _filterKokoroPhonemes(raw);
    _cache[cacheKey] = phonemes;
    return phonemes;
  }

  Future<String> phonemizeSsml(String ssml, {String language = 'en-us'}) async {
    final normalized = ssml.trim();
    if (normalized.isEmpty) {
      return '';
    }
    final chunks = _kokoroSsml(normalized);
    final out = StringBuffer();
    for (final chunk in chunks) {
      switch (chunk.kind) {
        case _ssmlPlain:
          if (chunk.text.trim().isNotEmpty) {
            out.write(await phonemize(chunk.text, language: language));
          }
        case _ssmlPinyin:
          out.write(await _phonemizePinyinSequence(chunk.text));
        case _ssmlExplicit:
          out.write(_normalizeExplicitPhonemes(chunk.text));
      }
      if (chunk.spaceAfter) {
        out.write(' ');
      }
    }
    return _filterKokoroPhonemes(out.toString());
  }

  void dispose() {
    _ffi?.dispose();
    _ffi = null;
  }

  FutureOr<String> _phonemizeUncached(String text, String language) async {
    final backend = _backend;
    if (backend != null) {
      return backend(text, language: language);
    }

    final ffi = _ffi ??= EspeakG2p.auto(
      libraryPath: espeakLibraryPath,
      dataPath: espeakDataPath,
      voice: language,
      phonemeMode: EspeakG2p.cliIpaMode,
      separator: null,
    );
    return ffi.textToPhonemes(text, voice: language);
  }

  Future<String> _phonemizePinyinSequence(String pinyin) async {
    final normalized = _normalizePinyin(pinyin);
    if (normalized.isEmpty) {
      return '';
    }
    final cacheKey = 'pinyin\u0000$normalized';
    final cached = _cache[cacheKey];
    if (cached != null) {
      return cached;
    }
    final raw = await _phonemizeUncached(normalized, 'cmn');
    final phonemes = _filterKokoroPhonemes(_postProcess(raw, 'cmn'));
    _cache[cacheKey] = phonemes;
    return phonemes;
  }

  Future<String> _phonemizeMixedUncached(
    String text,
    String defaultLanguage,
  ) async {
    final runs = _kokoroRuns(text, defaultLanguage);
    final out = <String>[];
    for (final run in runs) {
      if (run.language.isEmpty) {
        out.add(run.text);
      } else {
        out.add(
          _postProcess(
            await _phonemizeUncached(run.text, run.language),
            run.language,
          ),
        );
      }
    }
    return out.where((value) => value.isNotEmpty).join(' ');
  }

  String _canonicalLanguage(String value) {
    final language = value.trim().toLowerCase();
    return switch (language) {
      'a' || 'en' || 'en-us' || 'en_us' || 'english' => 'en-us',
      'b' || 'en-gb' || 'en_gb' || 'british' => 'en',
      'z' || 'zh' || 'zh-cn' || 'zh_cn' || 'cmn' || 'mandarin' => 'cmn',
      _ => language.isEmpty ? 'en-us' : language,
    };
  }

  String _normalizeTextForLanguage(String text, String language) {
    var out = _collapseSpaces(text);
    if (language == 'en-us' || language == 'en') {
      out = out.replaceAllMapped(
        RegExp(r'\bDr\.\s*', caseSensitive: false),
        (_) => 'Doctor ',
      );
      out = out.replaceAllMapped(RegExp(r'\$(\d+)\.(\d{2})'), (match) {
        final dollars = int.parse(match.group(1)!);
        final cents = int.parse(match.group(2)!);
        final dollarWord = dollars == 1 ? 'dollar' : 'dollars';
        final centWord = cents == 1 ? 'cent' : 'cents';
        return '$dollars $dollarWord and $cents $centWord';
      });
      out = out.replaceAllMapped(
        RegExp(r'\b(\d{1,2}):0(\d)\b'),
        (match) => '${match.group(1)} oh ${match.group(2)}',
      );
      out = out.replaceAllMapped(
        RegExp(r'\b20(\d{2})\b'),
        (match) => '20 ${match.group(1)}',
      );
    }
    return _collapseSpaces(out);
  }

  String _postProcess(String value, String language) {
    var out = value.replaceAll('\u200d', '');
    out = out.replaceAll(RegExp(r'\([a-z]{2,3}\)\s*'), ' ');
    if (language == 'cmn') {
      return _collapseSpaces(out.replaceAll(RegExp(r'[0-9]'), ''));
    }
    out = out
        .replaceAll('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ')
        .replaceAll('nˈaɪnti', 'nˈaɪndi')
        .replaceAll('ahˈʌndɹɪd', 'a hˈʌndɹɪd')
        .replaceAll('d z!', 'dz!');
    return _collapseSpaces(
      out
          .split(RegExp(r'\s+'))
          .map((token) {
            return switch (token) {
              'r' => 'ɹ',
              'x' => 'k',
              'ɬ' => 'l',
              'ʲ' => 'j',
              _ => token,
            };
          })
          .join(' '),
    );
  }

  String _filterKokoroPhonemes(String value) {
    return _collapseSpaces(
      value
          .replaceAll('\u200d', '')
          .replaceAll(RegExp(r'\([a-z]{2,3}\)\s*'), ' '),
    );
  }

  String _normalizeExplicitPhonemes(String value) {
    return _collapseSpaces(value.replaceAll("'", 'ˈ'));
  }

  List<_KokoroRun> _kokoroRuns(String text, String defaultLanguage) {
    final runs = <_KokoroRun>[];
    final current = StringBuffer();
    String? currentLanguage;

    void flush() {
      if (current.isEmpty) {
        return;
      }
      runs.add(
        _KokoroRun(current.toString(), currentLanguage ?? defaultLanguage),
      );
      current.clear();
    }

    for (final rune in text.runes) {
      final char = String.fromCharCode(rune);
      if (char.trim().isEmpty) {
        current.write(char);
        continue;
      }
      final language = _isCjkRune(rune) ? 'cmn' : 'en-us';
      if (currentLanguage != null && language != currentLanguage) {
        flush();
      }
      currentLanguage = language;
      current.write(char);
    }
    flush();
    return runs;
  }

  List<_SsmlChunk> _kokoroSsml(String text) {
    var body = text
        .replaceAll(RegExp(r'</?speak\b[^>]*>', caseSensitive: false), '')
        .replaceAllMapped(
          RegExp(
            r'<sub\b[^>]*\balias="([^"]*)"[^>]*>.*?</sub>',
            caseSensitive: false,
            dotAll: true,
          ),
          (match) => _xmlUnescape(match.group(1)!),
        );
    final chunks = <_SsmlChunk>[];

    void addPlain(String value) {
      final plain = _xmlUnescape(
        value.replaceAll(RegExp(r'<[^>]+>', dotAll: true), ''),
      );
      if (plain.trim().isNotEmpty) {
        chunks.add(_SsmlChunk(_ssmlPlain, plain, false));
      }
    }

    final phoneme = RegExp(
      r'<phoneme\b[^>]*\bph="([^"]*)"[^>]*>.*?</phoneme>',
      caseSensitive: false,
      dotAll: true,
    );
    var cursor = 0;
    for (final match in phoneme.allMatches(body)) {
      addPlain(body.substring(cursor, match.start));
      final value = _xmlUnescape(match.group(1)!);
      chunks.add(
        _SsmlChunk(
          _looksLikePinyin(value) ? _ssmlPinyin : _ssmlExplicit,
          value,
          false,
        ),
      );
      cursor = match.end;
    }
    addPlain(body.substring(cursor));

    return [
      for (var i = 0; i < chunks.length; i += 1)
        _SsmlChunk(chunks[i].kind, chunks[i].text, i + 1 < chunks.length),
    ];
  }

  _KokoroRoute _kokoroRoute(String text, String requestedLanguage) {
    final hasCjk = text.runes.any(_isCjkRune);
    final hasLatin = text.runes.any(_isLatinRune);
    if (hasCjk && hasLatin) {
      return _KokoroRoute(requestedLanguage, true);
    }
    if (hasCjk) {
      return const _KokoroRoute('cmn', false);
    }
    if (hasLatin) {
      return const _KokoroRoute('en-us', false);
    }
    return _KokoroRoute(requestedLanguage, false);
  }

  String _normalizePinyin(String value) {
    return _collapseSpaces(value.toLowerCase());
  }

  bool _looksLikePinyin(String value) {
    return RegExp(
      r'^[a-züv:]+[1-5](\s+[a-züv:]+[1-5])*$',
      caseSensitive: false,
    ).hasMatch(value.trim());
  }

  String _xmlUnescape(String value) {
    return value
        .replaceAll('&quot;', '"')
        .replaceAll('&apos;', "'")
        .replaceAll('&lt;', '<')
        .replaceAll('&gt;', '>')
        .replaceAll('&amp;', '&');
  }

  String _collapseSpaces(String value) {
    return value.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  bool _isLatinRune(int rune) {
    return (rune >= 0x41 && rune <= 0x5a) || (rune >= 0x61 && rune <= 0x7a);
  }

  bool _isCjkRune(int rune) {
    return (rune >= 0x3400 && rune <= 0x4dbf) ||
        (rune >= 0x4e00 && rune <= 0x9fff) ||
        (rune >= 0xf900 && rune <= 0xfaff) ||
        (rune >= 0x20000 && rune <= 0x2a6df) ||
        (rune >= 0x2a700 && rune <= 0x2b73f) ||
        (rune >= 0x2b740 && rune <= 0x2b81f) ||
        (rune >= 0x2b820 && rune <= 0x2ceaf);
  }
}

final class _KokoroRun {
  const _KokoroRun(this.text, this.language);

  final String text;
  final String language;
}

final class _KokoroRoute {
  const _KokoroRoute(this.language, this.mixed);

  final String language;
  final bool mixed;
}

final class _SsmlChunk {
  const _SsmlChunk(this.kind, this.text, this.spaceAfter);

  final int kind;
  final String text;
  final bool spaceAfter;
}

final class _LruCache<K, V> {
  _LruCache(this.capacity);

  final int capacity;
  final _items = <K, V>{};

  V? operator [](K key) {
    final value = _items.remove(key);
    if (value != null) {
      _items[key] = value;
    }
    return value;
  }

  void operator []=(K key, V value) {
    if (capacity <= 0) {
      return;
    }
    _items.remove(key);
    _items[key] = value;
    while (_items.length > capacity) {
      _items.remove(_items.keys.first);
    }
  }
}
