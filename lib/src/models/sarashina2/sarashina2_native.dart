import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import '../../runtime/native_ffi.dart' as dz;

import '../../runtime/native_float32_source.dart';
import '../../runtime/native_int32_source.dart';
import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/runtime.dart' show RuntimeTensorDataType;

const _semanticVocabSize = 6561;
const _semanticPrefix = '<|semantic_';
const _semanticSuffix = '|>';
const _speechStartToken = '<|speech_start|>';

final class SarashinaTokenizerHandle {
  SarashinaTokenizerHandle._(this._tokenizer);

  final _SarashinaTokenizer _tokenizer;
  bool _closed = false;

  factory SarashinaTokenizerHandle.fromFile(String path) {
    return SarashinaTokenizerHandle.fromBytes(File(path).readAsBytesSync());
  }

  factory SarashinaTokenizerHandle.fromBytes(List<int> sidecarBytes) {
    return SarashinaTokenizerHandle._(
      _SarashinaTokenizer.parse(utf8.decode(sidecarBytes)),
    );
  }

  Int32List encode(String text) {
    _checkOpen();
    return Int32List.fromList(_tokenizer.encode(text));
  }

  NativeTensorBuffer encodeBuffer(String text, {dz.NativeFfi? ffiRuntime}) {
    _checkOpen();
    final ids = _tokenizer.encode(text);
    final out = _allocateInt32Buffer(ids.length, ffiRuntime);
    out.asInt32List().setAll(0, ids);
    return out;
  }

  NativeTensorBuffer encodePromptTokenIdsBuffer({
    required String text,
    required Object promptTokens,
    required int speechStartTokenId,
    required int semanticBaseId,
    required int semanticVocabSize,
    dz.NativeFfi? ffiRuntime,
  }) {
    _checkOpen();
    if (speechStartTokenId < 0 ||
        semanticBaseId < 0 ||
        semanticVocabSize <= 0) {
      throw StateError('Sarashina2 tokenizer received invalid input.');
    }
    final tokens = _int32Values(promptTokens);
    final ids = _tokenizer.encode(text)..add(speechStartTokenId);
    for (final token in tokens) {
      if (token < 0 || token >= semanticVocabSize) {
        throw StateError(
          'Sarashina2 semantic token id is outside the speech tokenizer vocabulary.',
        );
      }
      ids.add(semanticBaseId + token);
    }
    final out = _allocateInt32Buffer(ids.length, ffiRuntime);
    out.asInt32List().setAll(0, ids);
    return out;
  }

  void close() {
    _closed = true;
  }

  void _checkOpen() {
    if (_closed) {
      throw StateError('Sarashina2 tokenizer is closed.');
    }
  }
}

Int32List parseSarashinaSemanticTokens(String text) {
  return Int32List.fromList(_parseSemanticTokenText(text));
}

NativeTensorBuffer parseSarashinaSemanticTokensBuffer(String text) {
  final tokens = parseSarashinaSemanticTokens(text);
  final out = NativeTensorBuffer.int32([tokens.length]);
  out.asInt32List().setAll(0, tokens);
  return out;
}

String formatSarashinaSemanticTokens(List<int> tokens) {
  validateSarashinaSemanticTokensNative(tokens);
  return tokens.map((token) => '$_semanticPrefix$token$_semanticSuffix').join();
}

void validateSarashinaSemanticTokensNative(Object tokens) {
  for (final token in _int32Values(tokens)) {
    if (token < 0 || token >= _semanticVocabSize) {
      throw StateError(
        'Sarashina2 semantic token id is outside the speech tokenizer vocabulary.',
      );
    }
  }
}

String buildSarashinaPromptNative({
  required String text,
  String promptText = '',
  Object promptTokens = const <int>[],
}) {
  final tokens = _int32Values(promptTokens);
  if ((promptText.isEmpty) != tokens.isEmpty) {
    throw StateError('Sarashina2 native helper received invalid input.');
  }
  return '$promptText$text$_speechStartToken'
      '${formatSarashinaSemanticTokens(tokens)}';
}

int sampleSarashinaSemanticTokenizerId({
  required Object logits,
  required List<int> generatedSemanticTokens,
  required int semanticBaseId,
  required int semanticVocabSize,
  required int eosId,
  required double temperature,
  required double topP,
  required double frequencyPenalty,
  required double randomDraw,
}) {
  final counts = Int32List(semanticVocabSize);
  for (final semanticId in generatedSemanticTokens) {
    if (semanticId < 0 || semanticId >= semanticVocabSize) {
      throw StateError(
        'Sarashina2 semantic token id is outside the speech tokenizer vocabulary.',
      );
    }
    if (counts[semanticId] < 0x7fffffff) {
      counts[semanticId] += 1;
    }
  }
  return sampleSarashinaSemanticTokenizerIdFromCounts(
    logits: logits,
    semanticCounts: NativeTensorBuffer.int32([0])..close(),
    semanticBaseId: semanticBaseId,
    semanticVocabSize: semanticVocabSize,
    eosId: eosId,
    temperature: temperature,
    topP: topP,
    frequencyPenalty: frequencyPenalty,
    randomDraw: randomDraw,
    countsOverride: counts,
  );
}

int sampleSarashinaSemanticTokenizerIdFromCounts({
  required Object logits,
  required NativeTensorBuffer semanticCounts,
  required int semanticBaseId,
  required int semanticVocabSize,
  required int eosId,
  required double temperature,
  required double topP,
  required double frequencyPenalty,
  required double randomDraw,
  Int32List? countsOverride,
}) {
  if (semanticVocabSize <= 0 ||
      semanticBaseId < 0 ||
      eosId < 0 ||
      !temperature.isFinite ||
      !topP.isFinite ||
      !frequencyPenalty.isFinite ||
      !randomDraw.isFinite ||
      temperature < 0 ||
      topP <= 0 ||
      randomDraw < 0 ||
      randomDraw > 1) {
    throw StateError('Sarashina2 native helper received invalid input.');
  }
  final values = _float32Values(logits);
  final counts = countsOverride ?? semanticCounts.asInt32List();
  if (counts.length < semanticVocabSize ||
      eosId >= values.length ||
      semanticBaseId >= values.length ||
      semanticBaseId + semanticVocabSize > values.length) {
    throw StateError('Sarashina2 native helper received invalid input.');
  }
  final candidates = <_Candidate>[_Candidate(eosId, values[eosId].toDouble())];
  for (var semanticId = 0; semanticId < semanticVocabSize; semanticId += 1) {
    final seen = counts[semanticId];
    if (seen < 0) {
      throw StateError('Sarashina2 native helper received invalid input.');
    }
    var logit = values[semanticBaseId + semanticId].toDouble();
    if (!logit.isFinite) {
      throw StateError('Sarashina2 native helper received invalid input.');
    }
    if (frequencyPenalty != 0) {
      logit -= frequencyPenalty * seen;
    }
    candidates.add(_Candidate(semanticBaseId + semanticId, logit));
  }
  return _sampleTopP(candidates, temperature, topP, randomDraw);
}

void appendSarashinaSemanticIdNative({
  required NativeTensorBuffer generated,
  required int generatedLength,
  required NativeTensorBuffer semanticCounts,
  required int semanticId,
}) {
  _checkInt32Buffer(generated, 'generated');
  _checkInt32Buffer(semanticCounts, 'semanticCounts');
  final generatedValues = generated.asInt32List();
  final counts = semanticCounts.asInt32List();
  if (generatedLength < 0 || generatedLength >= generatedValues.length) {
    throw StateError('Sarashina2 native helper output buffer is too small.');
  }
  if (semanticId < 0 || semanticId >= counts.length) {
    throw StateError(
      'Sarashina2 semantic token id is outside the speech tokenizer vocabulary.',
    );
  }
  generatedValues[generatedLength] = semanticId;
  if (counts[semanticId] < 0x7fffffff) {
    counts[semanticId] += 1;
  }
}

final class _Piece {
  _Piece(this.id, this.score, this.bytes);
  final int id;
  final double score;
  final Uint8List bytes;
}

final class _SarashinaTokenizer {
  _SarashinaTokenizer({
    required this.pieces,
    required this.added,
    required this.byteFallbackIds,
    required this.replacement,
    required this.unkId,
    required this.addPrefix,
    required this.byteFallback,
  }) {
    pieceIndex = _buildIndex(pieces);
    addedIndex = _buildIndex(added);
  }

  final List<_Piece> pieces;
  final List<_Piece> added;
  final List<int> byteFallbackIds;
  final Uint8List replacement;
  final int unkId;
  final bool addPrefix;
  final bool byteFallback;
  late final List<List<_Piece>> pieceIndex;
  late final List<List<_Piece>> addedIndex;

  static _SarashinaTokenizer parse(String sidecar) {
    if (!sidecar.startsWith('sara2tok\t1\n')) {
      throw StateError('Sarashina2 tokenizer sidecar is invalid.');
    }
    var unkId = 0;
    var byteFallback = false;
    var replacement = Uint8List.fromList(utf8.encode('▁'));
    var addPrefix = false;
    final pieces = <_Piece>[];
    final added = <_Piece>[];
    for (final raw in const LineSplitter().convert(sidecar)) {
      final line = raw.endsWith('\r') ? raw.substring(0, raw.length - 1) : raw;
      if (line.isEmpty || line == 'sara2tok\t1') continue;
      final fields = line.split('\t');
      if (fields.first == 'meta') {
        if (fields.length < 3) {
          throw StateError('Sarashina2 tokenizer sidecar is invalid.');
        }
        switch (fields[1]) {
          case 'unk_id':
            unkId = int.parse(fields[2]);
          case 'byte_fallback':
            byteFallback = fields[2] == '1' || fields[2] == 'true';
          case 'replacement_hex':
            replacement = Uint8List.fromList(_decodeHex(fields[2]));
          case 'prepend_scheme':
            addPrefix = fields[2] != 'never';
        }
      } else if (fields.first == 'tok') {
        if (fields.length < 4) {
          throw StateError('Sarashina2 tokenizer sidecar is invalid.');
        }
        pieces.add(
          _Piece(
            int.parse(fields[1]),
            double.parse(fields[2]),
            Uint8List.fromList(_decodeHex(fields[3])),
          ),
        );
      } else if (fields.first == 'add') {
        if (fields.length < 4) {
          throw StateError('Sarashina2 tokenizer sidecar is invalid.');
        }
        added.add(
          _Piece(
            int.parse(fields[1]),
            0,
            Uint8List.fromList(_decodeHex(fields[3])),
          ),
        );
      } else {
        throw StateError('Sarashina2 tokenizer sidecar is invalid.');
      }
    }
    pieces.sort(_pieceCompare);
    added.sort(_pieceCompare);
    final fallbackIds = List<int>.filled(256, 0);
    if (byteFallback) {
      final found = List<bool>.filled(256, false);
      for (final piece in pieces) {
        final value = _byteFallbackValue(piece.bytes);
        if (value != null) {
          fallbackIds[value] = piece.id;
          found[value] = true;
        }
      }
      if (found.any((value) => !value)) {
        throw StateError('Sarashina2 tokenizer sidecar is invalid.');
      }
    }
    return _SarashinaTokenizer(
      pieces: pieces,
      added: added,
      byteFallbackIds: fallbackIds,
      replacement: replacement,
      unkId: unkId,
      addPrefix: addPrefix,
      byteFallback: byteFallback,
    );
  }

  List<int> encode(String text) {
    final bytes = Uint8List.fromList(utf8.encode(text));
    final out = <int>[];
    var segmentStart = 0;
    var index = 0;
    while (index < bytes.length) {
      final matched = _matchAdded(bytes, index);
      if (matched != null) {
        _encodeNormal(Uint8List.sublistView(bytes, segmentStart, index), out);
        out.add(matched.id);
        index += matched.bytes.length;
        segmentStart = index;
      } else {
        index += 1;
      }
    }
    _encodeNormal(Uint8List.sublistView(bytes, segmentStart), out);
    return out;
  }

  _Piece? _matchAdded(Uint8List bytes, int offset) {
    if (offset >= bytes.length) return null;
    for (final piece in addedIndex[bytes[offset]]) {
      if (_startsWith(bytes, offset, piece.bytes)) return piece;
    }
    return null;
  }

  void _encodeNormal(Uint8List bytes, List<int> out) {
    if (bytes.isEmpty) return;
    final normalized = _metaspace(bytes);
    _encodeUnigram(normalized, out);
  }

  Uint8List _metaspace(Uint8List bytes) {
    final builder = BytesBuilder(copy: false);
    if (addPrefix && (bytes.isEmpty || bytes.first != 0x20)) {
      builder.add(replacement);
    }
    for (final byte in bytes) {
      if (byte == 0x20) {
        builder.add(replacement);
      } else {
        builder.addByte(byte);
      }
    }
    return builder.takeBytes();
  }

  void _encodeUnigram(Uint8List text, List<int> out) {
    final n = text.length;
    if (n == 0) return;
    final best = List<double>.filled(n + 1, double.negativeInfinity);
    final prev = List<int>.filled(n + 1, 0);
    final ids = List<int>.filled(n + 1, unkId);
    best[0] = 0;
    for (var pos = 0; pos < n; pos += 1) {
      if (best[pos] == double.negativeInfinity) continue;
      var matched = false;
      for (final piece in pieceIndex[text[pos]]) {
        final end = pos + piece.bytes.length;
        if (end <= n && _startsWith(text, pos, piece.bytes)) {
          matched = true;
          final score = best[pos] + piece.score;
          if (score > best[end]) {
            best[end] = score;
            prev[end] = pos;
            ids[end] = piece.id;
          }
        }
      }
      if (!matched) {
        final end = byteFallback
            ? pos + 1
            : math.min(n, pos + _utf8ByteLen(text[pos]));
        final id = byteFallback ? byteFallbackIds[text[pos]] : unkId;
        final score = best[pos] - 100;
        if (score > best[end]) {
          best[end] = score;
          prev[end] = pos;
          ids[end] = id;
        }
      }
    }
    if (best[n] == double.negativeInfinity) {
      throw StateError('Sarashina2 tokenizer received invalid input.');
    }
    final reverse = <int>[];
    var cursor = n;
    while (cursor > 0) {
      reverse.add(ids[cursor]);
      final next = prev[cursor];
      if (next >= cursor) {
        throw StateError('Sarashina2 tokenizer received invalid input.');
      }
      cursor = next;
    }
    out.addAll(reverse.reversed);
  }
}

List<List<_Piece>> _buildIndex(List<_Piece> pieces) {
  final out = List<List<_Piece>>.generate(256, (_) => <_Piece>[]);
  for (final piece in pieces) {
    if (piece.bytes.isNotEmpty) {
      out[piece.bytes.first].add(piece);
    }
  }
  return out;
}

int _pieceCompare(_Piece a, _Piece b) {
  final af = a.bytes.isEmpty ? 0 : a.bytes.first;
  final bf = b.bytes.isEmpty ? 0 : b.bytes.first;
  if (af != bf) return af.compareTo(bf);
  if (a.bytes.length != b.bytes.length) {
    return b.bytes.length.compareTo(a.bytes.length);
  }
  return a.id.compareTo(b.id);
}

bool _startsWith(Uint8List bytes, int offset, Uint8List prefix) {
  if (offset + prefix.length > bytes.length) return false;
  for (var i = 0; i < prefix.length; i += 1) {
    if (bytes[offset + i] != prefix[i]) return false;
  }
  return true;
}

List<int> _decodeHex(String value) {
  if (value.length.isOdd) {
    throw StateError('Sarashina2 tokenizer sidecar is invalid.');
  }
  return [
    for (var i = 0; i < value.length; i += 2)
      int.parse(value.substring(i, i + 2), radix: 16),
  ];
}

int? _byteFallbackValue(Uint8List bytes) {
  if (bytes.length != 6 ||
      bytes[0] != 0x3c ||
      bytes[1] != 0x30 ||
      bytes[2] != 0x78 ||
      bytes[5] != 0x3e) {
    return null;
  }
  final text = String.fromCharCodes(bytes.sublist(3, 5));
  return int.tryParse(text, radix: 16);
}

int _utf8ByteLen(int first) {
  if (first < 0x80) return 1;
  if ((first & 0xe0) == 0xc0) return 2;
  if ((first & 0xf0) == 0xe0) return 3;
  if ((first & 0xf8) == 0xf0) return 4;
  return 1;
}

List<int> _parseSemanticTokenText(String text) {
  final tokens = <int>[];
  var cursor = 0;
  while (true) {
    final start = text.indexOf(_semanticPrefix, cursor);
    if (start < 0) break;
    final numberStart = start + _semanticPrefix.length;
    final end = text.indexOf(_semanticSuffix, numberStart);
    if (end < 0) break;
    final id = int.tryParse(text.substring(numberStart, end));
    if (id != null) {
      if (id < 0 || id >= _semanticVocabSize) {
        throw StateError(
          'Sarashina2 semantic token id is outside the speech tokenizer vocabulary.',
        );
      }
      tokens.add(id);
    }
    cursor = end + _semanticSuffix.length;
  }
  return tokens;
}

List<int> _int32Values(Object source) {
  return withNativeInt32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return const <int>[];
    return List<int>.from(pointer.asTypedList(length), growable: false);
  });
}

Float32List _float32Values(Object source) {
  return withNativeFloat32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return Float32List(0);
    return Float32List.fromList(pointer.asTypedList(length));
  });
}

NativeTensorBuffer _allocateInt32Buffer(int count, dz.NativeFfi? ffiRuntime) {
  return NativeTensorBuffer.int32([count], ffiRuntime: ffiRuntime);
}

void _checkInt32Buffer(NativeTensorBuffer buffer, String name) {
  if (buffer.dtype != RuntimeTensorDataType.int32) {
    throw StateError('Expected int32 $name, got ${buffer.dtype.name}.');
  }
}

final class _Candidate {
  _Candidate(this.id, this.logit);
  final int id;
  final double logit;
  double prob = 0;
}

bool _candidateBetter(_Candidate a, _Candidate b) {
  if (a.logit > b.logit) return true;
  if (a.logit < b.logit) return false;
  return a.id < b.id;
}

int _sampleTopP(
  List<_Candidate> candidates,
  double temperature,
  double topP,
  double randomDraw,
) {
  if (candidates.isEmpty) {
    throw StateError('Sarashina2 native helper received invalid input.');
  }
  if (temperature == 0) {
    var best = candidates.first;
    for (final candidate in candidates.skip(1)) {
      if (_candidateBetter(candidate, best)) best = candidate;
    }
    return best.id;
  }
  var maxLogit = candidates.first.logit;
  for (final candidate in candidates.skip(1)) {
    if (candidate.logit > maxLogit) maxLogit = candidate.logit;
  }
  var sum = 0.0;
  for (final candidate in candidates) {
    if (!candidate.logit.isFinite) {
      throw StateError('Sarashina2 native helper received invalid input.');
    }
    candidate.prob = math.exp((candidate.logit - maxLogit) / temperature);
    sum += candidate.prob;
  }
  if (!(sum > 0)) {
    throw StateError('Sarashina2 native helper received invalid input.');
  }
  for (final candidate in candidates) {
    candidate.prob /= sum;
  }
  candidates.sort((a, b) {
    final byProb = b.prob.compareTo(a.prob);
    return byProb != 0 ? byProb : a.id.compareTo(b.id);
  });
  final nucleusP = topP > 1 ? 1.0 : topP;
  var nucleusCount = 0;
  var cumulative = 0.0;
  while (nucleusCount < candidates.length && cumulative < nucleusP) {
    cumulative += candidates[nucleusCount].prob;
    nucleusCount += 1;
  }
  final slice = candidates.take(math.max(1, nucleusCount)).toList();
  final target =
      randomDraw.clamp(0.0, 0.9999999999999999) *
      slice.fold<double>(0, (sum, candidate) => sum + candidate.prob);
  cumulative = 0.0;
  for (final candidate in slice) {
    cumulative += candidate.prob;
    if (target < cumulative) return candidate.id;
  }
  return slice.last.id;
}
