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
  });

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

  Map<String, Object?> toModelInputs({
    required int batchSize,
    required int tokenLength,
    required int charLength,
    required int homographTargets,
    required int polyphoneTargets,
    required int numHomographClasses,
    required int numPolyphoneClasses,
  }) {
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
  });

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

  Map<String, Object?> toModelInputs({
    required int batchSize,
    required int tokenLength,
    required int charLength,
    required int homographTargets,
    required int polyphoneTargets,
    required int numHomographClasses,
    required int numPolyphoneClasses,
  }) {
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
    final exportConfig =
        jsonDecode(await File(exportConfigPath).readAsString()) as Map;
    final structuredConfig =
        jsonDecode(await File(structuredConfigPath).readAsString()) as Map;
    return StructuredFrontendConfig(
      batchSize: (exportConfig['export_batch_size'] as num?)?.toInt() ?? 1,
      tokenLength:
          (exportConfig['export_token_length'] as num?)?.toInt() ?? 512,
      charLength: (exportConfig['export_char_length'] as num?)?.toInt() ?? 1024,
      homographTargets:
          (exportConfig['export_homograph_targets'] as num?)?.toInt() ?? 16,
      polyphoneTargets:
          (exportConfig['export_polyphone_targets'] as num?)?.toInt() ?? 16,
      numHomographClasses:
          (exportConfig['num_homograph_classes'] as num?)?.toInt() ?? 1,
      numPolyphoneClasses:
          (exportConfig['num_polyphone_classes'] as num?)?.toInt() ?? 1,
      emphasisThreshold:
          (structuredConfig['emphasis_decoding_threshold'] as num?)
              ?.toDouble() ??
          0.75,
    );
  }
}

final class CharVocab {
  CharVocab(this.charToId, {required this.padId, required this.unkId});

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
}

final class MmBertBpeTokenizer {
  MmBertBpeTokenizer({
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
    final ids = <int>[bosId];
    final offsets = <(int, int)>[(0, 0)];
    final chars = text.runes.map(String.fromCharCode).toList(growable: false);
    final normalized = '▁${text.replaceAll(' ', '▁')}';
    final normChars = normalized.runes
        .map(String.fromCharCode)
        .toList(growable: false);
    var originalIndex = 0;
    final pieces = <_PieceWithOffset>[];
    for (final ch in normChars) {
      final start = ch == '▁' ? originalIndex : originalIndex;
      final end = ch == '▁'
          ? originalIndex
          : math.min(originalIndex + 1, chars.length);
      pieces.add(_PieceWithOffset(ch, start, end));
      if (ch != '▁') {
        originalIndex += 1;
      } else if (originalIndex < chars.length && chars[originalIndex] == ' ') {
        originalIndex += 1;
      }
    }
    final bpePieces = _bpe(pieces);
    for (final piece in bpePieces) {
      if (ids.length >= maxLength - 1) {
        break;
      }
      final id = vocab[piece.text] ?? _byteFallbackId(piece.text) ?? unkId;
      ids.add(id);
      offsets.add((piece.start, piece.end));
    }
    ids.add(eosId);
    offsets.add((0, 0));
    while (ids.length < maxLength) {
      ids.add(padId);
      offsets.add((0, 0));
    }
    if (ids.length > maxLength) {
      ids.length = maxLength;
      offsets.length = maxLength;
      ids[maxLength - 1] = eosId;
      offsets[maxLength - 1] = (0, 0);
    }
    return TokenizedText(ids: ids, offsets: offsets);
  }

  List<_PieceWithOffset> _bpe(List<_PieceWithOffset> pieces) {
    final out = pieces.toList();
    while (out.length > 1) {
      var bestRank = 1 << 62;
      var bestIndex = -1;
      for (var i = 0; i < out.length - 1; i++) {
        final rank = mergeRanks['${out[i].text} ${out[i + 1].text}'];
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) {
        break;
      }
      final a = out[bestIndex];
      final b = out[bestIndex + 1];
      out
        ..removeAt(bestIndex)
        ..removeAt(bestIndex)
        ..insert(
          bestIndex,
          _PieceWithOffset(a.text + b.text, a.start, math.max(a.end, b.end)),
        );
    }
    return out;
  }

  int? _byteFallbackId(String piece) {
    if (piece.runes.length != 1) {
      return null;
    }
    final bytes = utf8.encode(piece);
    if (bytes.length != 1) {
      return null;
    }
    return vocab['<0x${bytes.first.toRadixString(16).toUpperCase().padLeft(2, '0')}>'];
  }
}

final class _PieceWithOffset {
  _PieceWithOffset(this.text, this.start, this.end);
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
    final inputIds = Int64List(config.batchSize * config.tokenLength);
    final attention = Int64List(config.batchSize * config.tokenLength);
    for (var i = 0; i < inputIds.length; i++) {
      inputIds[i] = tokenizer.padId;
    }
    final charIds = Int64List(config.batchSize * config.charLength);
    final charMask = Int64List(config.batchSize * config.charLength);
    for (var i = 0; i < charIds.length; i++) {
      charIds[i] = charVocab.padId;
    }
    final homographTargetMasks = Uint8List(
      config.batchSize * config.homographTargets * config.tokenLength,
    );
    final homographCandidateMasks =
        Uint8List(
          config.batchSize *
              config.homographTargets *
              config.numHomographClasses,
        )..fillRange(
          0,
          config.batchSize *
              config.homographTargets *
              config.numHomographClasses,
          1,
        );
    final polyphoneTargetCharMasks = Uint8List(
      config.batchSize * config.polyphoneTargets * config.charLength,
    );
    final polyphoneCandidateMasks =
        Uint8List(
          config.batchSize *
              config.polyphoneTargets *
              config.numPolyphoneClasses,
        )..fillRange(
          0,
          config.batchSize *
              config.polyphoneTargets *
              config.numPolyphoneClasses,
          1,
        );
    final homographTargetsByRow = <List<PronunciationItem>>[];
    final polyphoneTargetsByRow = <List<PronunciationItem>>[];
    final numCharsByRow = <int>[];
    for (var row = 0; row < texts.length; row++) {
      final text = texts[row];
      final tokenized = tokenizer.encode(text, maxLength: config.tokenLength);
      final tokenRowOffset = row * config.tokenLength;
      for (var i = 0; i < tokenized.ids.length; i++) {
        inputIds[tokenRowOffset + i] = tokenized.ids[i];
        attention[tokenRowOffset + i] = tokenized.ids[i] == tokenizer.padId
            ? 0
            : 1;
      }

      final chars = text.runes.map(String.fromCharCode).toList(growable: false);
      final numChars = math.min(chars.length, config.charLength);
      numCharsByRow.add(numChars);
      final charRowOffset = row * config.charLength;
      for (var i = 0; i < numChars; i++) {
        charIds[charRowOffset + i] = charVocab.idFor(chars[i]);
        charMask[charRowOffset + i] = 1;
      }

      final homographTargets = targetResolver
          .proposeHomographs(text)
          .take(config.homographTargets)
          .toList(growable: false);
      final polyphoneTargets = targetResolver
          .proposePolyphones(text)
          .take(config.polyphoneTargets)
          .toList(growable: false);
      homographTargetsByRow.add(homographTargets);
      polyphoneTargetsByRow.add(polyphoneTargets);

      for (
        var targetIdx = 0;
        targetIdx < homographTargets.length;
        targetIdx++
      ) {
        final item = homographTargets[targetIdx];
        final targetOffset =
            (row * config.homographTargets + targetIdx) * config.tokenLength;
        for (final pos in tokenPositionsForSpan(
          tokenized.offsets,
          start: item.start,
          end: item.end,
        )) {
          if (pos >= 0 && pos < config.tokenLength) {
            homographTargetMasks[targetOffset + pos] = 1;
          }
        }
        final ids = targetResolver.homographCandidateIds(item);
        if (ids.isNotEmpty) {
          final candidateOffset =
              (row * config.homographTargets + targetIdx) *
              config.numHomographClasses;
          homographCandidateMasks.fillRange(
            candidateOffset,
            candidateOffset + config.numHomographClasses,
            0,
          );
          for (final id in ids) {
            if (id >= 0 && id < config.numHomographClasses) {
              homographCandidateMasks[candidateOffset + id] = 1;
            }
          }
        }
      }
      for (
        var targetIdx = 0;
        targetIdx < polyphoneTargets.length;
        targetIdx++
      ) {
        final item = polyphoneTargets[targetIdx];
        final start = math.max(0, item.start);
        final end = math.min(config.charLength, item.end);
        final targetOffset =
            (row * config.polyphoneTargets + targetIdx) * config.charLength;
        for (var c = start; c < end; c++) {
          polyphoneTargetCharMasks[targetOffset + c] = 1;
        }
        final ids = targetResolver.polyphoneCandidateIds(item);
        if (ids.isNotEmpty) {
          final candidateOffset =
              (row * config.polyphoneTargets + targetIdx) *
              config.numPolyphoneClasses;
          polyphoneCandidateMasks.fillRange(
            candidateOffset,
            candidateOffset + config.numPolyphoneClasses,
            0,
          );
          for (final id in ids) {
            if (id >= 0 && id < config.numPolyphoneClasses) {
              polyphoneCandidateMasks[candidateOffset + id] = 1;
            }
          }
        }
      }
    }
    for (var row = texts.length; row < config.batchSize; row++) {
      homographTargetsByRow.add(const []);
      polyphoneTargetsByRow.add(const []);
      numCharsByRow.add(0);
    }
    return EncodedStructuredBatch(
      texts: List<String>.from(texts),
      inputIds: inputIds,
      attentionMask: attention,
      charIds: charIds,
      charAttentionMask: charMask,
      homographTargetMasks: homographTargetMasks,
      homographCandidateMasks: homographCandidateMasks,
      polyphoneTargetCharMasks: polyphoneTargetCharMasks,
      polyphoneCandidateMasks: polyphoneCandidateMasks,
      homographTargets: homographTargetsByRow,
      polyphoneTargets: polyphoneTargetsByRow,
      numChars: numCharsByRow,
      activeRows: texts.length,
    );
  }
}
