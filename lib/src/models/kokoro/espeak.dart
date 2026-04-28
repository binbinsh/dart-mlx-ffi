library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../../runtime/native_bindings.dart' as native;

final class EspeakG2p {
  factory EspeakG2p({
    required String libraryPath,
    required String dataPath,
    String voice = 'en-us',
    int phonemeMode = cliIpaMode,
    String? separator,
  }) => EspeakG2p.auto(
    libraryPath: libraryPath,
    dataPath: dataPath,
    voice: voice,
    phonemeMode: phonemeMode,
    separator: separator,
  );

  factory EspeakG2p.auto({
    String? libraryPath,
    String? dataPath,
    String voice = 'en-us',
    int phonemeMode = cliIpaMode,
    String? separator,
  }) {
    final mode = _resolvePhonemeMode(phonemeMode, separator);
    final handle = _openNative(libraryPath, dataPath, voice, mode);
    return EspeakG2p._(handle, voice: voice);
  }

  EspeakG2p._(this._handle, {required String voice}) : _voice = voice;

  static const int cliIpaMode = 0x02;

  final ffi.Pointer<ffi.Void> _handle;
  bool _disposed = false;
  String? _voice;

  String textToPhonemes(String text, {String? voice}) {
    _ensureOpen();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    final textPtr = text.toNativeUtf8();
    final nextVoice = voice != null && voice.isNotEmpty && voice != _voice
        ? voice
        : null;
    final voicePtr = _nativeStringOrNull(nextVoice);
    try {
      final out = native.espText(
        _handle,
        textPtr.cast<ffi.Char>(),
        voicePtr.cast<ffi.Char>(),
        error,
      );
      if (out == ffi.nullptr) {
        throw StateError(_takeNativeError(error));
      }
      try {
        if (nextVoice != null) {
          _voice = nextVoice;
        }
        return out.cast<Utf8>().toDartString();
      } finally {
        native.freeStr(out);
      }
    } finally {
      malloc.free(textPtr);
      _freeNativeString(voicePtr);
      calloc.free(error);
    }
  }

  String kokoroText(String text, {String language = 'en-us'}) {
    return _kokoroCall(text, language, native.espKokText);
  }

  String kokoroSsml(String ssml, {String language = 'en-us'}) {
    return _kokoroCall(ssml, language, native.espKokSsml);
  }

  void dispose() {
    if (_disposed) {
      return;
    }
    native.espFree(_handle);
    _disposed = true;
  }

  void _ensureOpen() {
    if (_disposed) {
      throw StateError('EspeakG2p is closed.');
    }
  }

  static int _resolvePhonemeMode(int phonemeMode, String? separator) {
    if (separator == null || separator.isEmpty) {
      return phonemeMode;
    }
    final rune = separator.runes.single;
    return phonemeMode | (rune << 8);
  }

  static ffi.Pointer<ffi.Void> _openNative(
    String? libraryPath,
    String? dataPath,
    String voice,
    int phonemeMode,
  ) {
    final error = calloc<ffi.Pointer<ffi.Char>>();
    final libraryPtr = _nativeStringOrNull(libraryPath);
    final dataPtr = _nativeStringOrNull(dataPath);
    final voicePtr = voice.toNativeUtf8();
    try {
      final handle = native.espNew(
        libraryPtr.cast<ffi.Char>(),
        dataPtr.cast<ffi.Char>(),
        voicePtr.cast<ffi.Char>(),
        phonemeMode,
        error,
      );
      if (handle == ffi.nullptr) {
        throw StateError(_takeNativeError(error));
      }
      return handle;
    } finally {
      _freeNativeString(libraryPtr);
      _freeNativeString(dataPtr);
      malloc.free(voicePtr);
      calloc.free(error);
    }
  }

  static ffi.Pointer<Utf8> _nativeStringOrNull(String? value) {
    if (value == null || value.isEmpty) {
      return ffi.nullptr.cast<Utf8>();
    }
    return value.toNativeUtf8();
  }

  static void _freeNativeString(ffi.Pointer<Utf8> value) {
    if (value != ffi.nullptr) {
      malloc.free(value);
    }
  }

  String _kokoroCall(
    String text,
    String language,
    ffi.Pointer<ffi.Char> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Char>,
      ffi.Pointer<ffi.Char>,
      ffi.Pointer<ffi.Pointer<ffi.Char>>,
    )
    call,
  ) {
    _ensureOpen();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    final textPtr = text.toNativeUtf8();
    final languagePtr = language.toNativeUtf8();
    try {
      final out = call(
        _handle,
        textPtr.cast<ffi.Char>(),
        languagePtr.cast<ffi.Char>(),
        error,
      );
      if (out == ffi.nullptr) {
        throw StateError(_takeNativeError(error));
      }
      try {
        return out.cast<Utf8>().toDartString();
      } finally {
        native.freeStr(out);
      }
    } finally {
      malloc.free(textPtr);
      malloc.free(languagePtr);
      calloc.free(error);
    }
  }
}

String _takeNativeError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native eSpeak call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
  }
}
