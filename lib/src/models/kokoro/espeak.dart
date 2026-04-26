library;

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';

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
    final lib = _openLibrary(libraryPath);
    final instance = EspeakG2p._(
      lib,
      phonemeMode: _resolvePhonemeMode(phonemeMode, separator),
    );
    instance._init(dataPath, voice);
    return instance;
  }

  EspeakG2p._(this._lib, {required int phonemeMode})
    : _phonemeMode = phonemeMode {
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

  static const int cliIpaMode = 0x02;

  static const _audioOutputSync = 0x02;
  static const _textModeUtf8 = 1;

  final ffi.DynamicLibrary _lib;
  final int _phonemeMode;
  bool _disposed = false;
  String? _voice;

  late final int Function(int, int, ffi.Pointer<Utf8>, int) _initialize;
  late final int Function(ffi.Pointer<Utf8>) _setVoiceByName;
  late final ffi.Pointer<Utf8> Function(
    ffi.Pointer<ffi.Pointer<Utf8>>,
    int,
    int,
  )
  _textToPhonemes;
  late final int Function() _terminate;

  String textToPhonemes(String text, {String? voice}) {
    _ensureOpen();
    if (voice != null && voice.isNotEmpty && voice != _voice) {
      _setVoice(voice);
    }

    final inputPtr = text.toNativeUtf8();
    final ptrPtr = malloc<ffi.Pointer<Utf8>>();
    ptrPtr.value = inputPtr;
    final clauses = <String>[];
    try {
      while (ptrPtr.value != ffi.nullptr) {
        if (ptrPtr.value.cast<ffi.Uint8>().value == 0) {
          break;
        }
        final result = _textToPhonemes(
          ptrPtr,
          _textModeUtf8,
          _phonemeMode,
        );
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

  void dispose() {
    if (_disposed) {
      return;
    }
    _terminate();
    _disposed = true;
  }

  void _init(String? dataPath, String voice) {
    final pathPtr = dataPath == null ? ffi.nullptr : dataPath.toNativeUtf8();
    try {
      final sampleRate = _initialize(_audioOutputSync, 0, pathPtr, 0);
      if (sampleRate <= 0) {
        throw StateError('espeak_Initialize failed with $sampleRate.');
      }
    } finally {
      if (pathPtr != ffi.nullptr) {
        malloc.free(pathPtr);
      }
    }
    _setVoice(voice);
  }

  void _setVoice(String voice) {
    final voicePtr = voice.toNativeUtf8();
    try {
      final result = _setVoiceByName(voicePtr);
      if (result != 0) {
        throw StateError('espeak_SetVoiceByName("$voice") failed: $result.');
      }
      _voice = voice;
    } finally {
      malloc.free(voicePtr);
    }
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

  static ffi.DynamicLibrary _openLibrary(String? explicitPath) {
    if (explicitPath != null && explicitPath.isNotEmpty) {
      return ffi.DynamicLibrary.open(explicitPath);
    }
    final candidates = Platform.isMacOS
        ? const [
            'libespeak-ng.dylib',
            '/opt/homebrew/lib/libespeak-ng.dylib',
            '/usr/local/lib/libespeak-ng.dylib',
          ]
        : Platform.isWindows
        ? const ['libespeak-ng.dll', 'espeak-ng.dll']
        : const ['libespeak-ng.so.1', 'libespeak-ng.so'];
    Object? firstError;
    for (final candidate in candidates) {
      try {
        return ffi.DynamicLibrary.open(candidate);
      } catch (error) {
        firstError ??= error;
      }
    }
    throw StateError('Unable to open eSpeak-NG dynamic library: $firstError');
  }
}
