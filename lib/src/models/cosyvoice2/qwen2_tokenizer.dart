// Qwen2 byte-level BPE tokenizer used by CosyVoice2's text frontend.
//
// Wraps the Zig runtime symbols `dinf_qwen2_bpe_*`.  Vocabulary and
// merges are loaded from the standard HuggingFace tokenizer files
// (`vocab.json`, `merges.txt`).  Special tokens — including the four
// Qwen2 controls and the 16 CosyVoice extension tokens — are passed in
// explicitly because the cosyvoice tokenizer registers them at runtime.

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';

import '../../runtime/native_bindings.dart' as native;

/// One special-token entry registered with the Zig tokenizer.
final class Qwen2SpecialToken {
  const Qwen2SpecialToken(this.text, this.id);
  final String text;
  final int id;
}

/// Default special tokens for the cosyvoice2 Qwen2 vocabulary.  The four
/// base ids match `Qwen2Tokenizer.special_tokens_map`; the remaining 16
/// are the cosyvoice instructional tags, registered in the order used by
/// `cosyvoice.tokenizer.tokenizer.QwenTokenizer.__init__`.
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

final _qwen2Finalizer = Finalizer<ffi.Pointer<ffi.Void>>((handle) {
  if (handle != ffi.nullptr) {
    native.qwen2BpeFree(handle);
  }
});

final class Qwen2BpeTokenizer {
  factory Qwen2BpeTokenizer({
    required Map<String, int> vocab,
    required List<String> merges,
    List<Qwen2SpecialToken> specials = kCosyVoice2DefaultSpecials,
  }) {
    final handle = _create(
      vocab: vocab,
      merges: merges,
      specials: specials,
    );
    final tk = Qwen2BpeTokenizer._(
      vocabSize: vocab.length + specials.length,
      handle: handle,
    );
    _qwen2Finalizer.attach(tk, handle, detach: tk);
    return tk;
  }

  Qwen2BpeTokenizer._({
    required this.vocabSize,
    required ffi.Pointer<ffi.Void> handle,
  }) : _handle = handle;

  /// Total token id space (base vocab + specials).
  final int vocabSize;
  final ffi.Pointer<ffi.Void> _handle;
  bool _closed = false;

  /// Loads a tokenizer from a directory containing `vocab.json` and
  /// `merges.txt` (the format produced by `Qwen2Tokenizer.save_pretrained`).
  /// `specials` defaults to the cosyvoice2 layout.
  static Future<Qwen2BpeTokenizer> load(
    String tokenizerDir, {
    List<Qwen2SpecialToken> specials = kCosyVoice2DefaultSpecials,
  }) async {
    final vocabRaw = await File('$tokenizerDir/vocab.json').readAsString();
    final mergesRaw = await File('$tokenizerDir/merges.txt').readAsString();
    final vocab = (jsonDecode(vocabRaw) as Map).map(
      (k, v) => MapEntry(k.toString(), (v as num).toInt()),
    );
    final merges = <String>[];
    for (final line in const LineSplitter().convert(mergesRaw)) {
      if (line.isEmpty) continue;
      if (line.startsWith('#')) continue;
      merges.add(line);
    }
    return Qwen2BpeTokenizer(
      vocab: vocab,
      merges: merges,
      specials: specials,
    );
  }

  /// Encodes `text` into token ids.  Throws [StateError] if the runtime
  /// reports a failure (which can also happen when the requested max
  /// length is too small — callers can catch and retry with the count
  /// reported by the runtime).
  List<int> encode(String text, {int? maxLength}) {
    if (_closed) {
      throw StateError('Qwen2 tokenizer is closed.');
    }
    // Rough upper bound: byte-level BPE produces at most one token per
    // input byte.  We bias up by a small constant so trivially short
    // strings still get a usable buffer.
    final cap = maxLength ?? (text.length * 4 + 16);
    final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final ids = cap > 0 ? calloc<ffi.Int64>(cap) : ffi.nullptr;
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.qwen2BpeEncode(
        _handle,
        input,
        cap,
        ids.cast<ffi.Int64>(),
        count,
        error,
      );
      if (status != 0) {
        // count.value reports the actual produced length, even on
        // "buffer too small" failures.  Surface both the message and the
        // hint so callers can decide whether to retry.
        final required = count.value;
        final msg = _takeError(error);
        throw StateError(
          required > cap
              ? '$msg (need $required, had $cap)'
              : msg,
        );
      }
      final n = count.value;
      return List<int>.generate(n, (i) => ids[i], growable: false);
    } finally {
      calloc.free(input);
      if (ids != ffi.nullptr) calloc.free(ids);
      calloc
        ..free(count)
        ..free(error);
    }
  }

  void close() {
    if (_closed) return;
    _closed = true;
    _qwen2Finalizer.detach(this);
    native.qwen2BpeFree(_handle);
  }
}

ffi.Pointer<ffi.Void> _create({
  required Map<String, int> vocab,
  required List<String> merges,
  required List<Qwen2SpecialToken> specials,
}) {
  final vocabEntries = vocab.entries.toList(growable: false);
  final vocabKeys = _CStringArray([
    for (final e in vocabEntries) e.key,
  ]);
  final mergeKeys = _CStringArray(merges);
  final specialTexts = _CStringArray([
    for (final s in specials) s.text,
  ]);
  final vocabIds = vocabEntries.isEmpty
      ? ffi.nullptr
      : calloc<ffi.Int64>(vocabEntries.length);
  final specialIds = specials.isEmpty
      ? ffi.nullptr
      : calloc<ffi.Int64>(specials.length);
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    for (var i = 0; i < vocabEntries.length; i += 1) {
      vocabIds[i] = vocabEntries[i].value;
    }
    for (var i = 0; i < specials.length; i += 1) {
      specialIds[i] = specials[i].id;
    }
    final handle = native.qwen2BpeNew(
      vocabKeys.pointer,
      vocabIds,
      vocabEntries.length,
      mergeKeys.pointer,
      merges.length,
      specialTexts.pointer,
      specialIds,
      specials.length,
      error,
    );
    if (handle == ffi.nullptr) {
      throw StateError(_takeError(error));
    }
    return handle;
  } finally {
    vocabKeys.close();
    mergeKeys.close();
    specialTexts.close();
    if (vocabIds != ffi.nullptr) calloc.free(vocabIds);
    if (specialIds != ffi.nullptr) calloc.free(specialIds);
    calloc.free(error);
  }
}

String _takeError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final ptr = error.value;
  if (ptr == ffi.nullptr) {
    return 'qwen2 tokenizer reported an unknown error';
  }
  final message = ptr.cast<Utf8>().toDartString();
  native.freeStr(ptr);
  return message;
}

final class _CStringArray {
  _CStringArray(List<String> values)
      : length = values.length,
        pointer = values.isEmpty
            ? ffi.nullptr
            : calloc<ffi.Pointer<ffi.Char>>(values.length) {
    try {
      for (var i = 0; i < values.length; i += 1) {
        pointer[i] = values[i].toNativeUtf8(allocator: calloc).cast<ffi.Char>();
      }
    } catch (_) {
      close();
      rethrow;
    }
  }

  final int length;
  final ffi.Pointer<ffi.Pointer<ffi.Char>> pointer;

  void close() {
    if (pointer == ffi.nullptr) return;
    for (var i = 0; i < length; i += 1) {
      final v = pointer[i];
      if (v != ffi.nullptr) calloc.free(v);
    }
    calloc.free(pointer);
  }
}
