part of 'structured_frontend.dart';

final class EncodedStructuredInput {
  EncodedStructuredInput({
    required this.text,
    required this.inputIds,
    required this.attentionMask,
    required this.charIds,
    required this.charAttentionMask,
    required this.homographTargetMasks,
    required this.homographCandidateMasks,
    required this.polyphoneTargetCharMasks,
    required this.polyphoneCandidateMasks,
    required this.homographTargets,
    required this.polyphoneTargets,
    required this.numChars,
    StructuredNativeInputs? nativeInputs,
  }) : _nativeInputs = nativeInputs;

  final String text;
  final Int64List inputIds;
  final Int64List attentionMask;
  final Int64List charIds;
  final Int64List charAttentionMask;
  final Uint8List homographTargetMasks;
  final Uint8List homographCandidateMasks;
  final Uint8List polyphoneTargetCharMasks;
  final Uint8List polyphoneCandidateMasks;
  final List<PronunciationItem> homographTargets;
  final List<PronunciationItem> polyphoneTargets;
  final int numChars;
  final StructuredNativeInputs? _nativeInputs;

  void close() {
    _nativeInputs?.close();
  }

  Map<String, Object?> toModelInputs({
    required int batchSize,
    required int tokenLength,
    required int charLength,
    required int homographTargets,
    required int polyphoneTargets,
    required int numHomographClasses,
    required int numPolyphoneClasses,
  }) {
    final nativeInputs = _nativeInputs;
    if (nativeInputs != null) {
      return nativeInputs.toModelInputs();
    }
    return {
      'input_ids': _int64Tensor(inputIds, [batchSize, tokenLength]),
      'attention_mask': _int64Tensor(attentionMask, [batchSize, tokenLength]),
      'char_ids': _int64Tensor(charIds, [batchSize, charLength]),
      'char_attention_mask': _int64Tensor(charAttentionMask, [
        batchSize,
        charLength,
      ]),
      'homograph_target_masks': _boolTensor(homographTargetMasks, [
        batchSize,
        homographTargets,
        tokenLength,
      ]),
      'homograph_candidate_masks': _boolTensor(homographCandidateMasks, [
        batchSize,
        homographTargets,
        numHomographClasses,
      ]),
      'polyphone_target_char_masks': _boolTensor(polyphoneTargetCharMasks, [
        batchSize,
        polyphoneTargets,
        charLength,
      ]),
      'polyphone_candidate_masks': _boolTensor(polyphoneCandidateMasks, [
        batchSize,
        polyphoneTargets,
        numPolyphoneClasses,
      ]),
    };
  }
}

final class EncodedStructuredBatch {
  EncodedStructuredBatch({
    required this.texts,
    required this.inputIds,
    required this.attentionMask,
    required this.charIds,
    required this.charAttentionMask,
    required this.homographTargetMasks,
    required this.homographCandidateMasks,
    required this.polyphoneTargetCharMasks,
    required this.polyphoneCandidateMasks,
    required this.homographTargets,
    required this.polyphoneTargets,
    required this.numChars,
    required this.activeRows,
    StructuredNativeInputs? nativeInputs,
  }) : _nativeInputs = nativeInputs;

  final List<String> texts;
  final Int64List inputIds;
  final Int64List attentionMask;
  final Int64List charIds;
  final Int64List charAttentionMask;
  final Uint8List homographTargetMasks;
  final Uint8List homographCandidateMasks;
  final Uint8List polyphoneTargetCharMasks;
  final Uint8List polyphoneCandidateMasks;
  final List<List<PronunciationItem>> homographTargets;
  final List<List<PronunciationItem>> polyphoneTargets;
  final List<int> numChars;
  final int activeRows;
  final StructuredNativeInputs? _nativeInputs;

  void close() {
    _nativeInputs?.close();
  }

  Map<String, Object?> toModelInputs({
    required int batchSize,
    required int tokenLength,
    required int charLength,
    required int homographTargets,
    required int polyphoneTargets,
    required int numHomographClasses,
    required int numPolyphoneClasses,
  }) {
    final nativeInputs = _nativeInputs;
    if (nativeInputs != null) {
      return nativeInputs.toModelInputs();
    }
    return {
      'input_ids': _int64Tensor(inputIds, [batchSize, tokenLength]),
      'attention_mask': _int64Tensor(attentionMask, [batchSize, tokenLength]),
      'char_ids': _int64Tensor(charIds, [batchSize, charLength]),
      'char_attention_mask': _int64Tensor(charAttentionMask, [
        batchSize,
        charLength,
      ]),
      'homograph_target_masks': _boolTensor(homographTargetMasks, [
        batchSize,
        homographTargets,
        tokenLength,
      ]),
      'homograph_candidate_masks': _boolTensor(homographCandidateMasks, [
        batchSize,
        homographTargets,
        numHomographClasses,
      ]),
      'polyphone_target_char_masks': _boolTensor(polyphoneTargetCharMasks, [
        batchSize,
        polyphoneTargets,
        charLength,
      ]),
      'polyphone_candidate_masks': _boolTensor(polyphoneCandidateMasks, [
        batchSize,
        polyphoneTargets,
        numPolyphoneClasses,
      ]),
    };
  }
}

RuntimeTensor _int64Tensor(Int64List values, List<int> shape) => RuntimeTensor(
  dtype: RuntimeTensorDataType.int64,
  shape: shape,
  bytes: Uint8List.view(values.buffer),
);

RuntimeTensor _boolTensor(Uint8List values, List<int> shape) => RuntimeTensor(
  dtype: RuntimeTensorDataType.boolean,
  shape: shape,
  bytes: values,
);

final class StructuredNativeInputs {
  StructuredNativeInputs(StructuredFrontendConfig config)
    : inputIdsBuffer = NativeTensorBuffer.int64([
        config.batchSize,
        config.tokenLength,
      ]),
      attentionBuffer = NativeTensorBuffer.int64([
        config.batchSize,
        config.tokenLength,
      ]),
      charIdsBuffer = NativeTensorBuffer.int64([
        config.batchSize,
        config.charLength,
      ]),
      charMaskBuffer = NativeTensorBuffer.int64([
        config.batchSize,
        config.charLength,
      ]),
      homographTargetBuffer = NativeTensorBuffer.boolean([
        config.batchSize,
        config.homographTargets,
        config.tokenLength,
      ]),
      homographCandidateBuffer = NativeTensorBuffer.boolean([
        config.batchSize,
        config.homographTargets,
        config.numHomographClasses,
      ]),
      polyphoneTargetBuffer = NativeTensorBuffer.boolean([
        config.batchSize,
        config.polyphoneTargets,
        config.charLength,
      ]),
      polyphoneCandidateBuffer = NativeTensorBuffer.boolean([
        config.batchSize,
        config.polyphoneTargets,
        config.numPolyphoneClasses,
      ]);

  final NativeTensorBuffer inputIdsBuffer;
  final NativeTensorBuffer attentionBuffer;
  final NativeTensorBuffer charIdsBuffer;
  final NativeTensorBuffer charMaskBuffer;
  final NativeTensorBuffer homographTargetBuffer;
  final NativeTensorBuffer homographCandidateBuffer;
  final NativeTensorBuffer polyphoneTargetBuffer;
  final NativeTensorBuffer polyphoneCandidateBuffer;

  bool _closed = false;

  Int64List get inputIds => inputIdsBuffer.asInt64List();
  Int64List get attention => attentionBuffer.asInt64List();
  Int64List get charIds => charIdsBuffer.asInt64List();
  Int64List get charMask => charMaskBuffer.asInt64List();
  Uint8List get homographTargetMasks => homographTargetBuffer.asUint8List();
  Uint8List get homographCandidateMasks =>
      homographCandidateBuffer.asUint8List();
  Uint8List get polyphoneTargetCharMasks => polyphoneTargetBuffer.asUint8List();
  Uint8List get polyphoneCandidateMasks =>
      polyphoneCandidateBuffer.asUint8List();

  void reset({required int tokenPadId, required int charPadId}) {
    _resetStructuredInputs(this, tokenPadId: tokenPadId, charPadId: charPadId);
  }

  Map<String, Object?> toModelInputs() => {
    'input_ids': inputIdsBuffer.tensor,
    'attention_mask': attentionBuffer.tensor,
    'char_ids': charIdsBuffer.tensor,
    'char_attention_mask': charMaskBuffer.tensor,
    'homograph_target_masks': homographTargetBuffer.tensor,
    'homograph_candidate_masks': homographCandidateBuffer.tensor,
    'polyphone_target_char_masks': polyphoneTargetBuffer.tensor,
    'polyphone_candidate_masks': polyphoneCandidateBuffer.tensor,
  };

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    inputIdsBuffer.close();
    attentionBuffer.close();
    charIdsBuffer.close();
    charMaskBuffer.close();
    homographTargetBuffer.close();
    homographCandidateBuffer.close();
    polyphoneTargetBuffer.close();
    polyphoneCandidateBuffer.close();
  }
}

void _resetStructuredInputs(
  StructuredNativeInputs inputs, {
  required int tokenPadId,
  required int charPadId,
}) {
  inputs.inputIds.fillRange(0, inputs.inputIds.length, tokenPadId);
  inputs.attention.fillRange(0, inputs.attention.length, 0);
  inputs.charIds.fillRange(0, inputs.charIds.length, charPadId);
  inputs.charMask.fillRange(0, inputs.charMask.length, 0);
  inputs.homographTargetMasks.fillRange(
    0,
    inputs.homographTargetMasks.length,
    0,
  );
  inputs.homographCandidateMasks.fillRange(
    0,
    inputs.homographCandidateMasks.length,
    0,
  );
  inputs.polyphoneTargetCharMasks.fillRange(
    0,
    inputs.polyphoneTargetCharMasks.length,
    0,
  );
  inputs.polyphoneCandidateMasks.fillRange(
    0,
    inputs.polyphoneCandidateMasks.length,
    0,
  );
}

final class _TokenOffsets {
  _TokenOffsets(List<(int, int)> offsets)
    : count = offsets.length,
      startValues = Int32List.fromList([
        for (final (start, _) in offsets) start,
      ]),
      endValues = Int32List.fromList([for (final (_, end) in offsets) end]);

  final int count;
  final Int32List startValues;
  final Int32List endValues;

  void close() {}
}

final class _NativeTokenizedText {
  _NativeTokenizedText({required this.count, required this.offsets});

  final int count;
  final _TokenOffsets offsets;

  void close() {
    offsets.close();
  }
}

void _fillMatchRows({
  required NativeTensorBuffer targetBuffer,
  required int targetOffset,
  required int targetWidth,
  required NativeTensorBuffer candidateBuffer,
  required int candidateOffset,
  required int candidateWidth,
  required _NativeTargetMatches matches,
  required int count,
  _TokenOffsets? tokens,
}) {
  if (count == 0) {
    return;
  }
  final target = targetBuffer.asUint8List();
  final candidates = candidateBuffer.asUint8List();
  for (var row = 0; row < count; row += 1) {
    final match = matches.items[row];
    final targetRow = targetOffset + row * targetWidth;
    if (tokens == null) {
      final start = match.start.clamp(0, targetWidth).toInt();
      final end = match.end.clamp(start, targetWidth).toInt();
      for (var i = start; i < end; i += 1) {
        final index = targetRow + i;
        if (index < target.length) target[index] = 1;
      }
    } else {
      for (var i = 0; i < math.min(tokens.count, targetWidth); i += 1) {
        final tokenStart = tokens.startValues[i];
        final tokenEnd = tokens.endValues[i];
        if (tokenEnd > match.start && tokenStart < match.end) {
          final index = targetRow + i;
          if (index < target.length) target[index] = 1;
        }
      }
    }
    final candidateRow = candidateOffset + row * candidateWidth;
    for (final id in match.ids) {
      if (id >= 0 && id < candidateWidth) {
        final index = candidateRow + id;
        if (index < candidates.length) candidates[index] = 1;
      }
    }
  }
}

final class StructuredFrontendConfig {
  StructuredFrontendConfig({
    required this.batchSize,
    required this.tokenLength,
    required this.charLength,
    required this.homographTargets,
    required this.polyphoneTargets,
    required this.numHomographClasses,
    required this.numPolyphoneClasses,
    required this.emphasisThreshold,
  });

  final int batchSize;
  final int tokenLength;
  final int charLength;
  final int homographTargets;
  final int polyphoneTargets;
  final int numHomographClasses;
  final int numPolyphoneClasses;
  final double emphasisThreshold;

  static Future<StructuredFrontendConfig> load({
    required String exportConfigPath,
    required String structuredConfigPath,
  }) async {
    final export =
        jsonDecode(await File(exportConfigPath).readAsString()) as Map;
    final structured =
        jsonDecode(await File(structuredConfigPath).readAsString()) as Map;
    return StructuredFrontendConfig(
      batchSize: _intConfig(export, 'export_batch_size', 1),
      tokenLength: _intConfig(export, 'export_token_length', 512),
      charLength: _intConfig(export, 'export_char_length', 1024),
      homographTargets: _intConfig(export, 'export_homograph_targets', 16),
      polyphoneTargets: _intConfig(export, 'export_polyphone_targets', 16),
      numHomographClasses: _intConfig(export, 'num_homograph_classes', 1),
      numPolyphoneClasses: _intConfig(export, 'num_polyphone_classes', 1),
      emphasisThreshold: _doubleConfig(
        structured,
        'emphasis_decoding_threshold',
        0.75,
      ),
    );
  }
}

int _intConfig(Map value, String key, int fallback) {
  final raw = value[key];
  return raw is num && raw >= 0 ? raw.toInt() : fallback;
}

double _doubleConfig(Map value, String key, double fallback) {
  final raw = value[key];
  return raw is num ? raw.toDouble() : fallback;
}

final class CharVocab {
  factory CharVocab(
    Map<String, int> charToId, {
    required int padId,
    required int unkId,
  }) {
    return CharVocab._(
      Map<String, int>.unmodifiable(charToId),
      padId: padId,
      unkId: unkId,
    );
  }

  CharVocab._(this.charToId, {required this.padId, required this.unkId});

  final Map<String, int> charToId;
  final int padId;
  final int unkId;

  static Future<CharVocab> load(String path) async {
    final payload = jsonDecode(await File(path).readAsString()) as Map;
    final idToChar = List<String>.from(payload['id_to_char'] as List);
    final charToId = <String, int>{};
    for (var i = 0; i < idToChar.length; i++) {
      charToId[idToChar[i]] = i;
    }
    return CharVocab(
      charToId,
      padId: charToId['<pad>'] ?? 0,
      unkId: charToId['<unk>'] ?? 1,
    );
  }

  int idFor(String char) => charToId[char] ?? unkId;

  int fillNative(
    String text, {
    required NativeTensorBuffer values,
    required NativeTensorBuffer mask,
    required int offset,
    required int width,
  }) {
    final valueRow = values.asInt64List();
    final maskRow = mask.asInt64List();
    if (offset < 0 || width < 0 || offset + width > valueRow.length) {
      throw RangeError.range(offset, 0, valueRow.length, 'offset');
    }
    valueRow.fillRange(offset, offset + width, padId);
    maskRow.fillRange(offset, offset + width, 0);
    var count = 0;
    for (final rune in text.runes) {
      if (count >= width) break;
      valueRow[offset + count] = idFor(String.fromCharCode(rune));
      maskRow[offset + count] = 1;
      count += 1;
    }
    return count;
  }

  void close() {}
}

const _sentencePieceMarker = '▁';

final class MmBertBpeTokenizer {
  factory MmBertBpeTokenizer({
    required Map<String, int> vocab,
    required Map<String, int> mergeRanks,
    required int bosId,
    required int eosId,
    required int padId,
    required int unkId,
  }) {
    return MmBertBpeTokenizer._(
      vocab: vocab,
      mergeRanks: mergeRanks,
      bosId: bosId,
      eosId: eosId,
      padId: padId,
      unkId: unkId,
    );
  }

  MmBertBpeTokenizer._({
    required this.vocab,
    required this.mergeRanks,
    required this.bosId,
    required this.eosId,
    required this.padId,
    required this.unkId,
  });

  final Map<String, int> vocab;
  final Map<String, int> mergeRanks;
  final int bosId;
  final int eosId;
  final int padId;
  final int unkId;
  bool _closed = false;

  static Future<MmBertBpeTokenizer> load(String tokenizerJsonPath) async {
    final payload =
        jsonDecode(await File(tokenizerJsonPath).readAsString()) as Map;
    final model = payload['model'] as Map;
    final vocabRaw = model['vocab'] as Map;
    final vocab = vocabRaw.map(
      (k, v) => MapEntry(k.toString(), (v as num).toInt()),
    );
    final merges = List<String>.from(model['merges'] as List? ?? const []);
    final mergeRanks = <String, int>{};
    for (var i = 0; i < merges.length; i++) {
      mergeRanks[merges[i]] = i;
    }
    return MmBertBpeTokenizer(
      vocab: vocab,
      mergeRanks: mergeRanks,
      bosId: vocab['<bos>'] ?? 2,
      eosId: vocab['<eos>'] ?? 1,
      padId: vocab['<pad>'] ?? 0,
      unkId: vocab['<unk>'] ?? 3,
    );
  }

  TokenizedText encode(String text, {required int maxLength}) {
    if (_closed) {
      throw StateError('BPE tokenizer is closed.');
    }
    final encoded = _encodeDart(text, maxLength: maxLength);
    return TokenizedText(ids: encoded.ids, offsets: encoded.offsets);
  }

  _DartTokenizedText _encodeDart(String text, {required int maxLength}) {
    if (maxLength <= 0) {
      throw RangeError.value(maxLength, 'maxLength');
    }
    final ids = List<int>.filled(maxLength, padId);
    final offsets = List<(int, int)>.filled(maxLength, (0, 0));
    if (maxLength == 1) {
      ids[0] = eosId;
      return _DartTokenizedText(ids, offsets, count: maxLength);
    }

    final pieces = _initialPieces(text);
    _applyMerges(pieces);

    ids[0] = bosId;
    var out = 1;
    for (final piece in pieces) {
      if (out >= maxLength - 1) {
        break;
      }
      ids[out] = _tokenId(piece.text);
      offsets[out] = (piece.start, piece.end);
      out += 1;
    }
    ids[out] = eosId;
    return _DartTokenizedText(ids, offsets, count: maxLength);
  }

  List<_BpePiece> _initialPieces(String text) {
    final pieces = <_BpePiece>[_BpePiece(_sentencePieceMarker, 0, 0)];
    var cursor = 0;
    for (final rune in text.runes) {
      final char = String.fromCharCode(rune);
      final next = cursor + char.length;
      if (rune == 0x20 || char == _sentencePieceMarker) {
        pieces.add(_BpePiece(_sentencePieceMarker, cursor, cursor));
      } else {
        pieces.add(_BpePiece(char, cursor, next));
      }
      cursor = next;
    }
    return pieces;
  }

  void _applyMerges(List<_BpePiece> pieces) {
    while (pieces.length > 1) {
      var bestRank = 0x7fffffffffffffff;
      var bestIndex = -1;
      for (var i = 0; i + 1 < pieces.length; i += 1) {
        final rank = mergeRanks['${pieces[i].text} ${pieces[i + 1].text}'];
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) {
        break;
      }
      final left = pieces[bestIndex];
      final right = pieces[bestIndex + 1];
      pieces[bestIndex] = _BpePiece(
        left.text + right.text,
        left.start,
        math.max(left.end, right.end),
      );
      pieces.removeAt(bestIndex + 1);
    }
  }

  int _tokenId(String piece) {
    final id = vocab[piece];
    if (id != null) {
      return id;
    }
    if (piece.length == 1) {
      final byte = piece.codeUnitAt(0);
      if (byte <= 0xff) {
        final hex = byte.toRadixString(16).toUpperCase().padLeft(2, '0');
        return vocab['<0x$hex>'] ?? unkId;
      }
    }
    return unkId;
  }

  _NativeTokenizedText _fillNative(
    String text, {
    required NativeTensorBuffer values,
    required NativeTensorBuffer mask,
    required int offset,
    required int width,
  }) {
    if (_closed) {
      throw StateError('BPE tokenizer is closed.');
    }
    final valueRow = values.asInt64List();
    final maskRow = mask.asInt64List();
    if (offset < 0 || width < 0 || offset + width > valueRow.length) {
      throw RangeError.range(offset, 0, valueRow.length, 'offset');
    }
    final encoded = _encodeDart(text, maxLength: width);
    valueRow.fillRange(offset, offset + width, padId);
    maskRow.fillRange(offset, offset + width, 0);
    for (var i = 0; i < encoded.count; i += 1) {
      valueRow[offset + i] = encoded.ids[i];
      maskRow[offset + i] = encoded.ids[i] == padId ? 0 : 1;
    }
    return _NativeTokenizedText(
      count: encoded.count,
      offsets: _TokenOffsets(encoded.offsets),
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
  }
}

final class _DartTokenizedText {
  const _DartTokenizedText(this.ids, this.offsets, {required this.count});
  final List<int> ids;
  final List<(int, int)> offsets;
  final int count;
}

final class _BpePiece {
  const _BpePiece(this.text, this.start, this.end);

  final String text;
  final int start;
  final int end;
}

final class TokenizedText {
  TokenizedText({required this.ids, required this.offsets});
  final List<int> ids;
  final List<(int, int)> offsets;
}

final class StructuredInputBuilder {
  StructuredInputBuilder({
    required this.tokenizer,
    required this.charVocab,
    required this.config,
    required this.targetResolver,
  });

  final MmBertBpeTokenizer tokenizer;
  final CharVocab charVocab;
  final StructuredFrontendConfig config;
  final PronunciationTargetResolver targetResolver;

  EncodedStructuredInput encode(String text) {
    final batch = encodeBatch([text]);
    return EncodedStructuredInput(
      text: text,
      inputIds: batch.inputIds,
      attentionMask: batch.attentionMask,
      charIds: batch.charIds,
      charAttentionMask: batch.charAttentionMask,
      homographTargetMasks: batch.homographTargetMasks,
      homographCandidateMasks: batch.homographCandidateMasks,
      polyphoneTargetCharMasks: batch.polyphoneTargetCharMasks,
      polyphoneCandidateMasks: batch.polyphoneCandidateMasks,
      homographTargets: batch.homographTargets.first,
      polyphoneTargets: batch.polyphoneTargets.first,
      numChars: batch.numChars.first,
      nativeInputs: batch._nativeInputs,
    );
  }

  EncodedStructuredBatch encodeBatch(List<String> texts) {
    if (texts.isEmpty) {
      throw ArgumentError.value(texts, 'texts', 'must not be empty');
    }
    if (texts.length > config.batchSize) {
      throw ArgumentError.value(
        texts.length,
        'texts.length',
        'must be <= fixed export batch size ${config.batchSize}',
      );
    }
    final nativeInputs = StructuredNativeInputs(config);
    var keepInputs = false;
    try {
      nativeInputs.reset(
        tokenPadId: tokenizer.padId,
        charPadId: charVocab.padId,
      );
      final homographTargetsByRow = <List<PronunciationItem>>[];
      final polyphoneTargetsByRow = <List<PronunciationItem>>[];
      final numCharsByRow = <int>[];
      for (var row = 0; row < texts.length; row++) {
        final text = texts[row];
        final tokenized = tokenizer._fillNative(
          text,
          values: nativeInputs.inputIdsBuffer,
          mask: nativeInputs.attentionBuffer,
          offset: row * config.tokenLength,
          width: config.tokenLength,
        );
        try {
          final charRowOffset = row * config.charLength;
          final numChars = charVocab.fillNative(
            text,
            values: nativeInputs.charIdsBuffer,
            mask: nativeInputs.charMaskBuffer,
            offset: charRowOffset,
            width: config.charLength,
          );
          numCharsByRow.add(numChars);

          final homographMatches = targetResolver._targetMatches(
            text,
            homographs: true,
          );
          try {
            final homographCount = math.min(
              homographMatches.count,
              config.homographTargets,
            );
            final homographTargets = targetResolver._homographItems(
              text,
              homographMatches,
              homographCount,
              includeCandidateIds: false,
            );
            homographTargetsByRow.add(homographTargets);
            _fillMatchRows(
              targetBuffer: nativeInputs.homographTargetBuffer,
              targetOffset: row * config.homographTargets * config.tokenLength,
              targetWidth: config.tokenLength,
              candidateBuffer: nativeInputs.homographCandidateBuffer,
              candidateOffset:
                  row * config.homographTargets * config.numHomographClasses,
              candidateWidth: config.numHomographClasses,
              matches: homographMatches,
              count: homographCount,
              tokens: tokenized.offsets,
            );
          } finally {
            homographMatches.close();
          }

          final polyphoneMatches = targetResolver._targetMatches(
            text,
            homographs: false,
          );
          try {
            final polyphoneCount = math.min(
              polyphoneMatches.count,
              config.polyphoneTargets,
            );
            final polyphoneTargets = targetResolver._polyphoneItems(
              polyphoneMatches,
              polyphoneCount,
              includeCandidateIds: false,
            );
            polyphoneTargetsByRow.add(polyphoneTargets);
            _fillMatchRows(
              targetBuffer: nativeInputs.polyphoneTargetBuffer,
              targetOffset: row * config.polyphoneTargets * config.charLength,
              targetWidth: config.charLength,
              candidateBuffer: nativeInputs.polyphoneCandidateBuffer,
              candidateOffset:
                  row * config.polyphoneTargets * config.numPolyphoneClasses,
              candidateWidth: config.numPolyphoneClasses,
              matches: polyphoneMatches,
              count: polyphoneCount,
            );
          } finally {
            polyphoneMatches.close();
          }
        } finally {
          tokenized.close();
        }
      }
      for (var row = texts.length; row < config.batchSize; row++) {
        homographTargetsByRow.add(const []);
        polyphoneTargetsByRow.add(const []);
        numCharsByRow.add(0);
      }
      keepInputs = true;
      return EncodedStructuredBatch(
        texts: List<String>.from(texts),
        inputIds: nativeInputs.inputIds,
        attentionMask: nativeInputs.attention,
        charIds: nativeInputs.charIds,
        charAttentionMask: nativeInputs.charMask,
        homographTargetMasks: nativeInputs.homographTargetMasks,
        homographCandidateMasks: nativeInputs.homographCandidateMasks,
        polyphoneTargetCharMasks: nativeInputs.polyphoneTargetCharMasks,
        polyphoneCandidateMasks: nativeInputs.polyphoneCandidateMasks,
        homographTargets: homographTargetsByRow,
        polyphoneTargets: polyphoneTargetsByRow,
        numChars: numCharsByRow,
        activeRows: texts.length,
        nativeInputs: nativeInputs,
      );
    } finally {
      if (!keepInputs) {
        nativeInputs.close();
      }
    }
  }

  void close() {
    tokenizer.close();
    charVocab.close();
  }
}
