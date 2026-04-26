import 'dart:async';
import 'dart:io';

import 'espeak.dart';

typedef KokoroPhonemeBackend =
    FutureOr<String> Function(String text, {required String language});

final class KokoroPhonemizer {
  KokoroPhonemizer({
    this.espeakBinary = 'espeak-ng',
    this.espeakLibraryPath,
    this.espeakDataPath,
    this.allowProcessFallback = false,
    int cacheSize = 512,
  }) : _backend = null,
       _cache = _LruCache<String, String>(cacheSize);

  KokoroPhonemizer.withBackend(
    KokoroPhonemeBackend backend, {
    int cacheSize = 512,
  }) : espeakBinary = 'espeak-ng',
       espeakLibraryPath = null,
       espeakDataPath = null,
       allowProcessFallback = false,
       _backend = backend,
       _cache = _LruCache<String, String>(cacheSize);

  final String espeakBinary;
  final String? espeakLibraryPath;
  final String? espeakDataPath;
  final bool allowProcessFallback;

  final KokoroPhonemeBackend? _backend;
  final _LruCache<String, String> _cache;
  EspeakG2p? _ffi;

  String get backendName => _backend != null
      ? 'injected'
      : _ffi != null
      ? 'espeak_ffi'
      : 'lazy';

  Future<String> phonemize(String text, {String language = 'en-us'}) async {
    final normalized = text.trim();
    if (normalized.isEmpty) {
      return '';
    }
    final requestedLanguage = _canonicalLanguage(language);
    final input = _normalizeTextForLanguage(normalized, requestedLanguage);
    final hasCjk = _containsCjk(input);
    final hasLatin = _containsLatin(input);
    final espeakLanguage = _languageForText(
      requestedLanguage,
      hasCjk: hasCjk,
      hasLatin: hasLatin,
    );
    final cacheKey = '$espeakLanguage\u0000$input';
    final cached = _cache[cacheKey];
    if (cached != null) {
      return cached;
    }

    final raw = hasCjk && hasLatin
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
    final tagPattern = RegExp(
      r'<phoneme\b[^>]*\bph="([^"]*)"[^>]*>(.*?)</phoneme>',
      caseSensitive: false,
      dotAll: true,
    );
    if (!tagPattern.hasMatch(normalized)) {
      return phonemize(
        _plainTextFromSsmlFragment(normalized),
        language: language,
      );
    }

    final out = StringBuffer();
    var cursor = 0;
    for (final match in tagPattern.allMatches(normalized)) {
      final before = normalized.substring(cursor, match.start);
      final plainBefore = _plainTextFromSsmlFragment(before);
      if (plainBefore.trim().isNotEmpty) {
        out.write(await phonemize(plainBefore, language: language));
        out.write(' ');
      }
      final ph = _decodeXml(match.group(1) ?? '').trim();
      if (ph.isNotEmpty) {
        out.write(
          _looksPinyinSequence(ph)
              ? await _phonemizePinyinSequence(ph)
              : _normalizeExplicitPhonemes(ph),
        );
        out.write(' ');
      }
      cursor = match.end;
    }
    final after = normalized.substring(cursor);
    final plainAfter = _plainTextFromSsmlFragment(after);
    if (plainAfter.trim().isNotEmpty) {
      out.write(await phonemize(plainAfter, language: language));
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

    try {
      final ffi = _ffi ??= EspeakG2p.auto(
        libraryPath: espeakLibraryPath,
        dataPath: _resolveEspeakDataPath(espeakDataPath),
        voice: language,
        phonemeMode: EspeakG2p.cliIpaMode,
        separator: null,
      );
      return ffi.textToPhonemes(text, voice: language);
    } catch (error) {
      if (!allowProcessFallback) {
        rethrow;
      }
      return _phonemizeWithProcess(text, language);
    }
  }

  Future<String> _phonemizePinyinSequence(String pinyin) async {
    final normalized = pinyin
        .replaceAll('Ü', 'V')
        .replaceAll('ü', 'v')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
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
    final out = StringBuffer();
    final run = StringBuffer();
    _Script? runScript;
    var lastFlushHadTrailingSpace = false;

    Future<void> flush() async {
      if (run.isEmpty || runScript == null) {
        lastFlushHadTrailingSpace = false;
        return;
      }
      final runText = run.toString();
      final hasLeadingSpace = RegExp(r'^\s').hasMatch(runText);
      final hasTrailingSpace = RegExp(r'\s$').hasMatch(runText);
      final language = switch (runScript!) {
        _Script.cjk => 'cmn',
        _Script.latin => defaultLanguage == 'en' ? 'en' : 'en-us',
      };
      final raw = await _phonemizeUncached(runText, language);
      if (hasLeadingSpace && out.isNotEmpty) {
        out.write(' ');
      }
      out.write(_postProcess(raw, language));
      if (hasTrailingSpace) {
        out.write(' ');
      }
      lastFlushHadTrailingSpace = hasTrailingSpace;
      run.clear();
      runScript = null;
    }

    for (final rune in text.runes) {
      final script = _scriptOf(rune);
      if (script == null) {
        if (runScript == null) {
          out.write(String.fromCharCode(rune));
        } else {
          run.write(String.fromCharCode(rune));
        }
        continue;
      }
      if (runScript != script) {
        final hadRun = runScript != null;
        await flush();
        if (hadRun && !lastFlushHadTrailingSpace && out.isNotEmpty) {
          out.write(' ');
        }
        runScript = script;
      }
      run.write(String.fromCharCode(rune));
    }
    await flush();
    return out.toString();
  }

  Future<String> _phonemizeWithProcess(String text, String language) async {
    final result = await Process.run(espeakBinary, [
      '-q',
      '--ipa=3',
      '-v',
      language,
      text,
    ]);
    if (result.exitCode != 0) {
      throw ProcessException(
        espeakBinary,
        ['-q', '--ipa=3', '-v', language, text],
        '${result.stderr}',
        result.exitCode,
      );
    }
    return '${result.stdout}';
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

  String _languageForText(
    String requested, {
    required bool hasCjk,
    required bool hasLatin,
  }) {
    if (hasLatin && !hasCjk && requested == 'cmn') {
      return 'en-us';
    }
    if (hasCjk && !hasLatin && (requested == 'en-us' || requested == 'en')) {
      return 'cmn';
    }
    return requested;
  }

  String _normalizeTextForLanguage(String text, String language) {
    if (language != 'en-us' && language != 'en') {
      return text;
    }
    var out = text
        .replaceAll(RegExp('[‘’]'), "'")
        .replaceAll('«', '“')
        .replaceAll('»', '”')
        .replaceAll(RegExp('[“”]'), '"')
        .replaceAll('(', '«')
        .replaceAll(')', '»')
        .replaceAll('、', ', ')
        .replaceAll('。', '. ')
        .replaceAll('！', '! ')
        .replaceAll('，', ', ')
        .replaceAll('：', ': ')
        .replaceAll('；', '; ')
        .replaceAll('？', '? ')
        .replaceAll(RegExp(r'[^\S \n]'), ' ')
        .replaceAll(RegExp(r' {2,}'), ' ')
        .replaceAll(RegExp(r'\bD[Rr]\.(?= [A-Z])'), 'Doctor')
        .replaceAll(RegExp(r'\b(?:Mr\.|MR\.(?= [A-Z]))'), 'Mister')
        .replaceAll(RegExp(r'\b(?:Ms\.|MS\.(?= [A-Z]))'), 'Miss')
        .replaceAll(RegExp(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))'), 'Mrs')
        .replaceAllMapped(
          RegExp(r'\betc\.(?! [A-Z])', caseSensitive: false),
          (_) => 'etc',
        )
        .replaceAllMapped(
          RegExp(r'\b(y)eah?\b', caseSensitive: false),
          (match) => "${match.group(1)}e'a",
        );

    out = out.replaceAllMapped(
      RegExp(r'\d*\.\d+|\b\d{4}s?\b|\b(?:[1-9]|1[0-2]):[0-5]\d\b'),
      (match) => _splitNumber(match.group(0)!),
    );
    out = _stripNumericCommas(out);
    out = out.replaceAllMapped(
      RegExp(
        r'[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b',
        caseSensitive: false,
      ),
      (match) => _flipMoney(match.group(0)!),
    );
    out = out.replaceAllMapped(
      RegExp(r'\d*\.\d+'),
      (match) => _pointNumber(match.group(0)!),
    );
    out = out
        .replaceAllMapped(
          RegExp(r'(\d)-(?=\d)'),
          (match) => '${match.group(1)} to ',
        )
        .replaceAllMapped(RegExp(r'(\d)S\b'), (match) => '${match.group(1)} S')
        .replaceAllMapped(
          RegExp(r"([BCDFGHJ-NP-TV-Z])'?s\b"),
          (match) => "${match.group(1)}'S",
        )
        .replaceAllMapped(RegExp(r"(X')S\b"), (match) => '${match.group(1)}s')
        .replaceAllMapped(
          RegExp(r'(?:[A-Za-z]\.){2,} [a-z]'),
          (match) => match.group(0)!.replaceAll('.', '-'),
        )
        .replaceAllMapped(
          RegExp(r'([A-Z])\.(?=[A-Z])', caseSensitive: false),
          (match) => '${match.group(1)}-',
        );
    return out.trim();
  }

  String _splitNumber(String match) {
    if (match.contains('.')) {
      return match;
    }
    if (match.contains(':')) {
      final parts = match.split(':');
      final h = int.parse(parts[0]);
      final m = int.parse(parts[1]);
      if (m == 0) {
        return "$h o'clock";
      }
      if (m < 10) {
        return '$h oh $m';
      }
      return '$h $m';
    }
    final year = int.parse(match.substring(0, 4));
    if (year < 1100 || year % 1000 < 10) {
      return match;
    }
    final left = match.substring(0, 2);
    final right = int.parse(match.substring(2, 4));
    final suffix = match.endsWith('s') ? 's' : '';
    if (year % 1000 >= 100 && year % 1000 <= 999) {
      if (right == 0) {
        return '$left hundred$suffix';
      }
      if (right < 10) {
        return '$left oh $right$suffix';
      }
    }
    return '$left $right$suffix';
  }

  String _stripNumericCommas(String text) {
    var out = text;
    while (RegExp(r'\d,\d').hasMatch(out)) {
      out = out.replaceAllMapped(
        RegExp(r'(\d),(\d)'),
        (match) => '${match.group(1)}${match.group(2)}',
      );
    }
    return out;
  }

  String _flipMoney(String match) {
    final bill = match.startsWith(r'$') ? 'dollar' : 'pound';
    final value = match.substring(1);
    if (double.tryParse(value.split(' ').first) == null) {
      return '$value ${bill}s';
    }
    if (!value.contains('.')) {
      final suffix = value == '1' ? '' : 's';
      return '$value $bill$suffix';
    }
    final parts = value.split('.');
    final whole = parts[0];
    final cents = int.parse(parts[1].padRight(2, '0').substring(0, 2));
    final coin = match.startsWith(r'$')
        ? cents == 1
              ? 'cent'
              : 'cents'
        : cents == 1
        ? 'penny'
        : 'pence';
    return '$whole $bill${whole == '1' ? '' : 's'} and $cents $coin';
  }

  String _pointNumber(String match) {
    final parts = match.split('.');
    return '${parts[0]} point ${parts[1].split('').join(' ')}';
  }

  String _postProcess(String value, String language) {
    if (language != 'en-us' && language != 'en') {
      return value;
    }
    var out = value
        .replaceAll('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ')
        .replaceAll('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
        .replaceAll('ʲ', 'j')
        .replaceAll('r', 'ɹ')
        .replaceAll('x', 'k')
        .replaceAll('ɬ', 'l')
        .replaceAllMapped(
          RegExp(r'([a-zɹː])(?=hˈʌndɹɪd)'),
          (match) => '${match.group(1)} ',
        )
        .replaceAllMapped(RegExp(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)'), (_) => 'z');
    if (language == 'en-us') {
      out = out.replaceAllMapped(RegExp(r'nˈaɪnti(?!ː)'), (_) => 'nˈaɪndi');
    }
    return out;
  }

  String _filterKokoroPhonemes(String value) {
    return value
        .replaceAll(RegExp(r'\([a-z]{2,3}\)', caseSensitive: false), ' ')
        .replaceAll('\u200d', '')
        .replaceAll(RegExp(r'[0-9]'), '')
        .replaceAll('\n', ' ')
        .replaceAll('\r', ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  bool _looksPinyinSequence(String value) {
    final raw = value.trim();
    if (raw.isEmpty || _containsCjk(raw)) {
      return false;
    }
    final tokens = raw
        .split(RegExp(r'\s+'))
        .map(
          (part) => part
              .split('|')
              .first
              .replaceAll(RegExp(r'^[.,!?:;"“”«»()\[\]{}]+'), '')
              .replaceAll(RegExp(r'[.,!?:;"“”«»()\[\]{}]+$'), ''),
        )
        .where((part) => part.isNotEmpty)
        .toList(growable: false);
    if (tokens.isEmpty) {
      return false;
    }
    return tokens.every(
      (part) => RegExp(r'^[A-Za-züÜvV]+[1-5]$').hasMatch(part),
    );
  }

  String _normalizeExplicitPhonemes(String value) {
    return value.replaceAll("'", 'ˈ').replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  String _plainTextFromSsmlFragment(String fragment) {
    final withSubAliases = fragment.replaceAllMapped(
      RegExp(
        r'<sub\b[^>]*alias="([^"]*)"[^>]*>.*?</sub>',
        caseSensitive: false,
        dotAll: true,
      ),
      (match) => match.group(1) ?? '',
    );
    return _decodeXml(withSubAliases.replaceAll(RegExp(r'<[^>]+>'), ''));
  }

  String _decodeXml(String value) => value
      .replaceAll('&quot;', '"')
      .replaceAll('&lt;', '<')
      .replaceAll('&gt;', '>')
      .replaceAll('&apos;', "'")
      .replaceAll('&amp;', '&');

  String? _resolveEspeakDataPath(String? explicit) {
    if (explicit != null && explicit.isNotEmpty) {
      return explicit;
    }
    for (final key in const ['ESPEAK_DATA_PATH', 'ESPEAKNG_DATA_PATH']) {
      final value = Platform.environment[key];
      if (value != null && value.isNotEmpty) {
        return value;
      }
    }
    for (final path in const [
      '/usr/lib/x86_64-linux-gnu/espeak-ng-data',
      '/usr/lib/aarch64-linux-gnu/espeak-ng-data',
      '/usr/share/espeak-ng-data',
      '/usr/local/share/espeak-ng-data',
    ]) {
      if (Directory(path).existsSync()) {
        return path;
      }
    }
    return null;
  }

  bool _containsCjk(String text) {
    for (final rune in text.runes) {
      if (_isCjk(rune)) {
        return true;
      }
    }
    return false;
  }

  bool _containsLatin(String text) {
    for (final rune in text.runes) {
      if (_isLatin(rune)) {
        return true;
      }
    }
    return false;
  }

  _Script? _scriptOf(int rune) {
    if (_isCjk(rune)) {
      return _Script.cjk;
    }
    if (_isLatin(rune)) {
      return _Script.latin;
    }
    return null;
  }

  bool _isCjk(int rune) =>
      (rune >= 0x3400 && rune <= 0x4dbf) ||
      (rune >= 0x4e00 && rune <= 0x9fff) ||
      (rune >= 0xf900 && rune <= 0xfaff);

  bool _isLatin(int rune) =>
      (rune >= 0x41 && rune <= 0x5a) || (rune >= 0x61 && rune <= 0x7a);
}

enum _Script { cjk, latin }

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
