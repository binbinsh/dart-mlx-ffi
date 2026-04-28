import 'dart:async';
import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../../runtime/native_bindings.dart' as native;
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
      ? 'espeak_zig'
      : 'lazy';

  Future<String> phonemize(String text, {String language = 'en-us'}) async {
    final normalized = text.trim();
    if (normalized.isEmpty) {
      return '';
    }
    if (_backend == null) {
      return _phonemizeNative(normalized, language);
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
    if (_backend == null) {
      return _phonemizeSsmlNative(normalized, language);
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

  String _phonemizeNative(String text, String language) {
    final cacheKey = 'native\u0000${language.trim().toLowerCase()}\u0000$text';
    final cached = _cache[cacheKey];
    if (cached != null) {
      return cached;
    }
    final phonemes = _nativeG2p().kokoroText(text, language: language);
    _cache[cacheKey] = phonemes;
    return phonemes;
  }

  String _phonemizeSsmlNative(String ssml, String language) {
    return _nativeG2p().kokoroSsml(ssml, language: language);
  }

  EspeakG2p _nativeG2p() {
    return _ffi ??= EspeakG2p.auto(
      libraryPath: espeakLibraryPath,
      dataPath: espeakDataPath,
      voice: 'en-us',
      phonemeMode: EspeakG2p.cliIpaMode,
      separator: null,
    );
  }

  Future<String> _phonemizePinyinSequence(String pinyin) async {
    final normalized = _kokoroCall(pinyin, native.kokPinyinNorm);
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
    final out = StringBuffer();
    for (final run in runs) {
      if (run.language.isEmpty) {
        out.write(run.text);
      } else {
        out.write(
          _postProcess(
            await _phonemizeUncached(run.text, run.language),
            run.language,
          ),
        );
      }
    }
    return out.toString();
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
    return _kokoroTextCall(text, language, native.kokNorm);
  }

  String _postProcess(String value, String language) {
    return _kokoroTextCall(value, language, native.kokPost);
  }

  String _filterKokoroPhonemes(String value) {
    return _kokoroCall(value, native.kokClean);
  }

  String _kokoroCall(
    String value,
    ffi.Pointer<ffi.Char> Function(
      ffi.Pointer<ffi.Char>,
      ffi.Pointer<ffi.Pointer<ffi.Char>>,
    )
    call,
  ) {
    final text = value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    ffi.Pointer<ffi.Char> out = ffi.nullptr;
    try {
      out = call(text, error);
      if (out == ffi.nullptr) {
        throw StateError(_takeNativeError(error));
      }
      return out.cast<Utf8>().toDartString();
    } finally {
      if (out != ffi.nullptr) {
        native.freeStr(out);
      }
      calloc
        ..free(text)
        ..free(error);
    }
  }

  String _kokoroTextCall(
    String value,
    String language,
    ffi.Pointer<ffi.Char> Function(
      ffi.Pointer<ffi.Char>,
      ffi.Pointer<ffi.Char>,
      ffi.Pointer<ffi.Pointer<ffi.Char>>,
    )
    call,
  ) {
    final text = value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final lang = language.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    ffi.Pointer<ffi.Char> out = ffi.nullptr;
    try {
      out = call(text, lang, error);
      if (out == ffi.nullptr) {
        throw StateError(_takeNativeError(error));
      }
      return out.cast<Utf8>().toDartString();
    } finally {
      if (out != ffi.nullptr) {
        native.freeStr(out);
      }
      calloc
        ..free(text)
        ..free(lang)
        ..free(error);
    }
  }

  String _normalizeExplicitPhonemes(String value) {
    return _kokoroCall(value, native.kokExplicit);
  }

  List<_KokoroRun> _kokoroRuns(String text, String defaultLanguage) {
    final textPtr = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final languagePtr = defaultLanguage
        .toNativeUtf8(allocator: calloc)
        .cast<ffi.Char>();
    final out = calloc<ffi.Pointer<native.KokoroRunAbi>>();
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    ffi.Pointer<native.KokoroRunAbi> items = ffi.nullptr;
    try {
      final status = native.kokRuns(textPtr, languagePtr, out, count, error);
      if (status != 0) {
        throw StateError(_takeNativeError(error));
      }
      items = out.value;
      return [
        for (var i = 0; i < count.value; i += 1)
          _KokoroRun(
            items[i].text.cast<Utf8>().toDartString(),
            items[i].language.cast<Utf8>().toDartString(),
          ),
      ];
    } finally {
      if (items != ffi.nullptr) {
        native.kokFreeRuns(items, count.value);
      }
      calloc
        ..free(textPtr)
        ..free(languagePtr)
        ..free(out)
        ..free(count)
        ..free(error);
    }
  }

  List<_SsmlChunk> _kokoroSsml(String text) {
    final textPtr = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final out = calloc<ffi.Pointer<native.KokoroSsmlAbi>>();
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    ffi.Pointer<native.KokoroSsmlAbi> items = ffi.nullptr;
    try {
      final status = native.kokSsml(textPtr, out, count, error);
      if (status != 0) {
        throw StateError(_takeNativeError(error));
      }
      items = out.value;
      return [
        for (var i = 0; i < count.value; i += 1)
          _SsmlChunk(
            items[i].kind,
            items[i].text.cast<Utf8>().toDartString(),
            items[i].spaceAfter != 0,
          ),
      ];
    } finally {
      if (items != ffi.nullptr) {
        native.kokFreeSsml(items, count.value);
      }
      calloc
        ..free(textPtr)
        ..free(out)
        ..free(count)
        ..free(error);
    }
  }

  _KokoroRoute _kokoroRoute(String text, String requestedLanguage) {
    final textPtr = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final requestedPtr = requestedLanguage
        .toNativeUtf8(allocator: calloc)
        .cast<ffi.Char>();
    final mixed = calloc<ffi.Int32>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    ffi.Pointer<ffi.Char> out = ffi.nullptr;
    try {
      out = native.kokLanguage(textPtr, requestedPtr, mixed, error);
      if (out == ffi.nullptr) {
        throw StateError(_takeNativeError(error));
      }
      return _KokoroRoute(out.cast<Utf8>().toDartString(), mixed.value != 0);
    } finally {
      if (out != ffi.nullptr) {
        native.freeStr(out);
      }
      calloc
        ..free(textPtr)
        ..free(requestedPtr)
        ..free(mixed)
        ..free(error);
    }
  }
}

String _takeNativeError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
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
