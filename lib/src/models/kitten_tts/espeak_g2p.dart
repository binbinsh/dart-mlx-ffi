library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

/// Dart FFI wrapper for eSpeak-NG grapheme-to-phoneme conversion.
///
/// Loads `libespeak-ng.dylib`, initialises with a data directory, and exposes
/// [textToPhonemes] which converts English text to IPA phoneme strings.
///
/// Usage:
/// ```dart
/// final espeak = EspeakG2p(
///   libraryPath: '/path/to/libespeak-ng.dylib',
///   dataPath: '/path/to/espeak-ng-data',
/// );
/// final ipa = espeak.textToPhonemes('Hello world');
/// espeak.dispose();
/// ```
final class EspeakG2p {
  /// Creates an eSpeak-NG G2P instance.
  ///
  /// [libraryPath] — path to `libespeak-ng.dylib`.
  /// [dataPath] — path to the `espeak-ng-data` directory.
  /// [voice] — eSpeak voice identifier (default `en-us`).
  factory EspeakG2p({
    required String libraryPath,
    required String dataPath,
    String voice = 'en-us',
  }) {
    final lib = ffi.DynamicLibrary.open(libraryPath);
    final instance = EspeakG2p._(lib);
    instance._init(dataPath, voice);
    return instance;
  }

  EspeakG2p._(this._lib) {
    _initialize = _lib
        .lookupFunction<
          ffi.Int32 Function(
            ffi.Int32,
            ffi.Int32,
            ffi.Pointer<Utf8>,
            ffi.Int32,
          ),
          int Function(int, int, ffi.Pointer<Utf8>, int)
        >('espeak_Initialize');

    _setVoiceByName = _lib
        .lookupFunction<
          ffi.Int32 Function(ffi.Pointer<Utf8>),
          int Function(ffi.Pointer<Utf8>)
        >('espeak_SetVoiceByName');

    _textToPhonemes = _lib
        .lookupFunction<
          ffi.Pointer<Utf8> Function(
            ffi.Pointer<ffi.Pointer<Utf8>>,
            ffi.Int32,
            ffi.Int32,
          ),
          ffi.Pointer<Utf8> Function(ffi.Pointer<ffi.Pointer<Utf8>>, int, int)
        >('espeak_TextToPhonemes');

    _terminate = _lib.lookupFunction<ffi.Int32 Function(), int Function()>(
      'espeak_Terminate',
    );
  }

  final ffi.DynamicLibrary _lib;
  bool _disposed = false;

  late final int Function(int, int, ffi.Pointer<Utf8>, int) _initialize;
  late final int Function(ffi.Pointer<Utf8>) _setVoiceByName;
  late final ffi.Pointer<Utf8> Function(
    ffi.Pointer<ffi.Pointer<Utf8>>,
    int,
    int,
  )
  _textToPhonemes;
  late final int Function() _terminate;

  /// Audio output mode: synchronous (no audio, phoneme processing only).
  static const _audioOutputSync = 0x02;

  /// phonememode flag: IPA output (bits 0–3 = 0x02).
  static const _ipaMode = 0x02;

  /// Separator character placed between phonemes within a word.
  /// `_` (underscore) → 0x5F shifted left 8 bits, OR'd with IPA flag.
  static const _phonemeMode = (0x5F << 8) | _ipaMode;

  /// textmode flag: input is UTF-8.
  static const _textModeUtf8 = 1;

  void _init(String dataPath, String voice) {
    final dataPathPtr = dataPath.toNativeUtf8();
    try {
      final sampleRate = _initialize(_audioOutputSync, 0, dataPathPtr, 0);
      if (sampleRate <= 0) {
        throw StateError(
          'espeak_Initialize failed (returned $sampleRate). '
          'Check that dataPath is correct: $dataPath',
        );
      }
    } finally {
      malloc.free(dataPathPtr);
    }

    final voicePtr = voice.toNativeUtf8();
    try {
      final result = _setVoiceByName(voicePtr);
      if (result != 0) {
        throw StateError(
          'espeak_SetVoiceByName("$voice") failed (returned $result).',
        );
      }
    } finally {
      malloc.free(voicePtr);
    }
  }

  /// Converts [text] to IPA phoneme string.
  ///
  /// Within each word, individual phonemes are separated by `_`.
  /// Words are separated by spaces.
  ///
  /// Example: `textToPhonemes('Hello')` → `h_ɛ_l_oʊ` (approximate).
  String textToPhonemes(String text) {
    _ensureOpen();
    final inputPtr = text.toNativeUtf8();
    // Double-pointer: espeak advances the inner pointer as it consumes text.
    final ptrPtr = malloc<ffi.Pointer<Utf8>>();
    ptrPtr.value = inputPtr;

    final clauses = <String>[];
    try {
      while (ptrPtr.value != ffi.nullptr) {
        // Check if the remaining text is empty (null-terminated empty string).
        if (ptrPtr.value.cast<ffi.Uint8>().value == 0) break;

        final result = _textToPhonemes(ptrPtr, _textModeUtf8, _phonemeMode);
        if (result != ffi.nullptr) {
          final clause = result.toDartString();
          if (clause.isNotEmpty) {
            clauses.add(clause);
          }
        }
      }
    } finally {
      malloc.free(ptrPtr);
      malloc.free(inputPtr);
    }
    return clauses.join(' ');
  }

  /// Releases eSpeak-NG resources.
  void dispose() {
    if (!_disposed) {
      _terminate();
      _disposed = true;
    }
  }

  void _ensureOpen() {
    if (_disposed) {
      throw StateError('EspeakG2p has been disposed.');
    }
  }
}
