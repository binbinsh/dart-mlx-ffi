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
    instance._init(_resolveDataPath(dataPath), voice);
    return instance;
  }

  EspeakG2p._(this._lib, {required this.phonemeMode}) {
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
  static const _audioOutputSync = 2;
  static const _textModeUtf8 = 1;

  final ffi.DynamicLibrary _lib;
  final int phonemeMode;
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
    if (text.isEmpty) {
      return '';
    }
    if (voice != null && voice.isNotEmpty && voice != _voice) {
      _setVoice(voice);
    }

    final inputPtr = text.toNativeUtf8();
    final cursor = malloc<ffi.Pointer<Utf8>>();
    cursor.value = inputPtr;
    final clauses = <String>[];
    try {
      while (cursor.value != ffi.nullptr) {
        if (cursor.value.cast<ffi.Uint8>().value == 0) {
          break;
        }
        final result = _textToPhonemes(
          cursor,
          _textModeUtf8,
          phonemeMode,
        );
        if (result != ffi.nullptr) {
          final clause = result.toDartString();
          if (clause.isNotEmpty) {
            clauses.add(clause);
          }
        }
      }
    } finally {
      malloc.free(cursor);
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
    final dataPtr = dataPath == null
        ? ffi.nullptr.cast<Utf8>()
        : dataPath.toNativeUtf8();
    try {
      final sampleRate = _initialize(_audioOutputSync, 0, dataPtr, 0);
      if (sampleRate <= 0) {
        throw StateError('eSpeak-NG initialization failed.');
      }
    } finally {
      if (dataPtr != ffi.nullptr) {
        malloc.free(dataPtr);
      }
    }
    _setVoice(voice);
  }

  void _setVoice(String voice) {
    final voicePtr = voice.toNativeUtf8();
    try {
      final status = _setVoiceByName(voicePtr);
      if (status != 0) {
        throw StateError('eSpeak-NG failed to select voice "$voice".');
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
    final runes = separator.runes.toList(growable: false);
    if (runes.length != 1) {
      throw ArgumentError.value(separator, 'separator', 'must be one rune');
    }
    return phonemeMode | (runes.single << 8);
  }

  static ffi.DynamicLibrary _openLibrary(String? explicit) {
    final candidates = <String>[
      if (explicit != null && explicit.isNotEmpty) explicit,
      ..._libraryCandidates(),
    ];
    Object? lastError;
    for (final candidate in candidates) {
      try {
        return ffi.DynamicLibrary.open(candidate);
      } catch (error) {
        lastError = error;
      }
    }
    throw StateError('eSpeak-NG dynamic library unavailable: $lastError');
  }

  static List<String> _libraryCandidates() {
    if (Platform.isMacOS) {
      return const [
        'libespeak-ng.dylib',
        '/opt/homebrew/lib/libespeak-ng.dylib',
        '/usr/local/lib/libespeak-ng.dylib',
      ];
    }
    if (Platform.isWindows) {
      return const ['libespeak-ng.dll', 'espeak-ng.dll'];
    }
    return const ['libespeak-ng.so.1', 'libespeak-ng.so'];
  }

  static String? _resolveDataPath(String? explicit) {
    if (explicit != null && explicit.isNotEmpty) {
      return explicit;
    }
    for (final name in const ['ESPEAK_DATA_PATH', 'ESPEAKNG_DATA_PATH']) {
      final value = Platform.environment[name];
      if (value != null && value.isNotEmpty) {
        return value;
      }
    }
    for (final candidate in _dataCandidates()) {
      if (Directory(candidate).existsSync()) {
        return candidate;
      }
    }
    return null;
  }

  static List<String> _dataCandidates() {
    if (Platform.isMacOS) {
      return const [
        '/opt/homebrew/share/espeak-ng-data',
        '/usr/local/share/espeak-ng-data',
        '/usr/share/espeak-ng-data',
      ];
    }
    if (Platform.isWindows) {
      return const ['espeak-ng-data'];
    }
    return const [
      '/usr/lib/x86_64-linux-gnu/espeak-ng-data',
      '/usr/lib/aarch64-linux-gnu/espeak-ng-data',
      '/usr/share/espeak-ng-data',
      '/usr/local/share/espeak-ng-data',
    ];
  }
}
