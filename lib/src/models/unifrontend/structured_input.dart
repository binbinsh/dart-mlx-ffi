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
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.structReset(
      inputs.inputIdsBuffer.nativeData.cast<ffi.Int64>(),
      inputs.inputIdsBuffer.byteLength ~/ 8,
      tokenPadId,
      inputs.attentionBuffer.nativeData.cast<ffi.Int64>(),
      inputs.attentionBuffer.byteLength ~/ 8,
      inputs.charIdsBuffer.nativeData.cast<ffi.Int64>(),
      inputs.charIdsBuffer.byteLength ~/ 8,
      charPadId,
      inputs.charMaskBuffer.nativeData.cast<ffi.Int64>(),
      inputs.charMaskBuffer.byteLength ~/ 8,
      inputs.homographTargetBuffer.nativeData.cast<ffi.Uint8>(),
      inputs.homographTargetBuffer.byteLength,
      inputs.homographCandidateBuffer.nativeData.cast<ffi.Uint8>(),
      inputs.homographCandidateBuffer.byteLength,
      inputs.polyphoneTargetBuffer.nativeData.cast<ffi.Uint8>(),
      inputs.polyphoneTargetBuffer.byteLength,
      inputs.polyphoneCandidateBuffer.nativeData.cast<ffi.Uint8>(),
      inputs.polyphoneCandidateBuffer.byteLength,
      error,
    );
    if (status != 0) {
      throw StateError(_takeFillError(error));
    }
  } finally {
    calloc.free(error);
  }
}

final class _TokenOffsets {
  _TokenOffsets(List<(int, int)> offsets)
    : count = offsets.length,
      starts = offsets.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Int32>(offsets.length),
      ends = offsets.isEmpty ? ffi.nullptr : calloc<ffi.Int32>(offsets.length) {
    for (var i = 0; i < offsets.length; i++) {
      final (start, end) = offsets[i];
      starts[i] = start;
      ends[i] = end;
    }
  }

  _TokenOffsets.allocate(this.count)
    : starts = count == 0 ? ffi.nullptr : calloc<ffi.Int32>(count),
      ends = count == 0 ? ffi.nullptr : calloc<ffi.Int32>(count);

  final int count;
  final ffi.Pointer<ffi.Int32> starts;
  final ffi.Pointer<ffi.Int32> ends;

  void close() {
    if (starts != ffi.nullptr) {
      calloc.free(starts);
    }
    if (ends != ffi.nullptr) {
      calloc.free(ends);
    }
  }
}

final class _I64Scratch {
  _I64Scratch(this.length)
    : pointer = length == 0 ? ffi.nullptr : calloc<ffi.Int64>(length);

  final int length;
  final ffi.Pointer<ffi.Int64> pointer;

  void copy(List<int> values) {
    if (values.length > length) {
      throw RangeError.range(values.length, 0, length, 'values.length');
    }
    for (var i = 0; i < values.length; i++) {
      pointer[i] = values[i];
    }
  }

  void set(int index, int value) {
    if (index < 0 || index >= length) {
      throw RangeError.range(index, 0, length - 1, 'index');
    }
    pointer[index] = value;
  }

  void close() {
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
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
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.structMatches(
      targetBuffer.nativeData.cast<ffi.Uint8>(),
      targetBuffer.byteLength,
      targetOffset,
      targetWidth,
      candidateBuffer.nativeData.cast<ffi.Uint8>(),
      candidateBuffer.byteLength,
      candidateOffset,
      candidateWidth,
      matches.items,
      count,
      tokens?.starts ?? ffi.nullptr,
      tokens?.ends ?? ffi.nullptr,
      tokens?.count ?? 0,
      error,
    );
    if (status != 0) {
      throw StateError(_takeFillError(error));
    }
  } finally {
    calloc.free(error);
  }
}

String _takeFillError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native fill call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
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
    final exportPath = exportConfigPath
        .toNativeUtf8(allocator: calloc)
        .cast<ffi.Char>();
    final structuredPath = structuredConfigPath
        .toNativeUtf8(allocator: calloc)
        .cast<ffi.Char>();
    final out = calloc<native.StructConfigAbi>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.structConfig(exportPath, structuredPath, out, error);
      if (status != 0) {
        throw StateError(_takeFillError(error));
      }
      final value = out.ref;
      return StructuredFrontendConfig(
        batchSize: value.batchSize,
        tokenLength: value.tokenLength,
        charLength: value.charLength,
        homographTargets: value.homographTargets,
        polyphoneTargets: value.polyphoneTargets,
        numHomographClasses: value.homographClasses,
        numPolyphoneClasses: value.polyphoneClasses,
        emphasisThreshold: value.emphasisThreshold,
      );
    } finally {
      calloc
        ..free(exportPath)
        ..free(structuredPath)
        ..free(out)
        ..free(error);
    }
  }
}

final class CharVocab {
  factory CharVocab(
    Map<String, int> charToId, {
    required int padId,
    required int unkId,
  }) {
    final nativeChars = <(int, int)>[];
    for (final entry in charToId.entries) {
      final runes = entry.key.runes.toList(growable: false);
      if (runes.length == 1) {
        nativeChars.add((runes.single, entry.value));
      }
    }
    return CharVocab._(
      Map<String, int>.unmodifiable(charToId),
      padId: padId,
      unkId: unkId,
      nativeChars: nativeChars,
    );
  }

  CharVocab._(
    this.charToId, {
    required this.padId,
    required this.unkId,
    required List<(int, int)> nativeChars,
  }) : _codes = NativeTensorBuffer.int32([nativeChars.length]),
       _ids = NativeTensorBuffer.int64([nativeChars.length]) {
    final codes = _codes.asInt32List();
    final ids = _ids.asInt64List();
    for (var index = 0; index < nativeChars.length; index += 1) {
      codes[index] = nativeChars[index].$1;
      ids[index] = nativeChars[index].$2;
    }
  }

  final Map<String, int> charToId;
  final int padId;
  final int unkId;
  final NativeTensorBuffer _codes;
  final NativeTensorBuffer _ids;

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
    final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.fillCharsI64(
        values.nativeData.cast<ffi.Int64>(),
        mask.nativeData.cast<ffi.Int64>(),
        values.byteLength ~/ 8,
        offset,
        width,
        input,
        _codes.nativeData.cast<ffi.Int32>(),
        _ids.nativeData.cast<ffi.Int64>(),
        _codes.byteLength ~/ 4,
        padId,
        unkId,
        count,
        error,
      );
      if (status != 0) {
        throw StateError(_takeFillError(error));
      }
      return count.value;
    } finally {
      calloc
        ..free(input)
        ..free(count)
        ..free(error);
    }
  }

  void close() {
    _codes.close();
    _ids.close();
  }
}

final _bpeFinalizer = Finalizer<ffi.Pointer<ffi.Void>>((handle) {
  if (handle != ffi.nullptr) {
    native.bpeFree(handle);
  }
});

final class MmBertBpeTokenizer {
  factory MmBertBpeTokenizer({
    required Map<String, int> vocab,
    required Map<String, int> mergeRanks,
    required int bosId,
    required int eosId,
    required int padId,
    required int unkId,
  }) {
    final handle = _createBpe(
      vocab: vocab,
      mergeRanks: mergeRanks,
      bosId: bosId,
      eosId: eosId,
      padId: padId,
      unkId: unkId,
    );
    final tokenizer = MmBertBpeTokenizer._(
      vocab: vocab,
      mergeRanks: mergeRanks,
      bosId: bosId,
      eosId: eosId,
      padId: padId,
      unkId: unkId,
      handle: handle,
    );
    _bpeFinalizer.attach(tokenizer, handle, detach: tokenizer);
    return tokenizer;
  }

  MmBertBpeTokenizer._({
    required this.vocab,
    required this.mergeRanks,
    required this.bosId,
    required this.eosId,
    required this.padId,
    required this.unkId,
    required ffi.Pointer<ffi.Void> handle,
  }) : _handle = handle;

  final Map<String, int> vocab;
  final Map<String, int> mergeRanks;
  final int bosId;
  final int eosId;
  final int padId;
  final int unkId;
  final ffi.Pointer<ffi.Void> _handle;
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
    final scratch = _I64Scratch(maxLength);
    try {
      final encoded = _encodeNative(text, maxLength: maxLength, ids: scratch);
      try {
        return TokenizedText(
          ids: List<int>.generate(
            encoded.count,
            (index) => scratch.pointer[index],
            growable: false,
          ),
          offsets: List<(int, int)>.generate(
            encoded.count,
            (index) =>
                (encoded.offsets.starts[index], encoded.offsets.ends[index]),
            growable: false,
          ),
        );
      } finally {
        encoded.close();
      }
    } finally {
      scratch.close();
    }
  }

  _NativeTokenizedText _encodeNative(
    String text, {
    required int maxLength,
    required _I64Scratch ids,
  }) {
    if (_closed) {
      throw StateError('BPE tokenizer is closed.');
    }
    if (maxLength <= 0 || ids.length < maxLength) {
      throw RangeError.value(maxLength, 'maxLength');
    }
    final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final offsets = _TokenOffsets.allocate(maxLength);
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.bpeEncode(
        _handle,
        input,
        maxLength,
        ids.pointer,
        offsets.starts,
        offsets.ends,
        count,
        error,
      );
      if (status != 0) {
        throw StateError(_takeFillError(error));
      }
      return _NativeTokenizedText(count: count.value, offsets: offsets);
    } catch (_) {
      offsets.close();
      rethrow;
    } finally {
      calloc
        ..free(input)
        ..free(count)
        ..free(error);
    }
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
    if (values.byteLength != mask.byteLength) {
      throw StateError('Native token value and mask buffers differ in size.');
    }
    final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final offsets = _TokenOffsets.allocate(width);
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = native.bpeFill(
        _handle,
        input,
        values.nativeData.cast<ffi.Int64>(),
        mask.nativeData.cast<ffi.Int64>(),
        values.byteLength ~/ 8,
        offset,
        width,
        offsets.starts,
        offsets.ends,
        count,
        error,
      );
      if (status != 0) {
        throw StateError(_takeFillError(error));
      }
      return _NativeTokenizedText(count: count.value, offsets: offsets);
    } catch (_) {
      offsets.close();
      rethrow;
    } finally {
      calloc
        ..free(input)
        ..free(count)
        ..free(error);
    }
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _bpeFinalizer.detach(this);
    native.bpeFree(_handle);
  }
}

final class TokenizedText {
  TokenizedText({required this.ids, required this.offsets});
  final List<int> ids;
  final List<(int, int)> offsets;
}

ffi.Pointer<ffi.Void> _createBpe({
  required Map<String, int> vocab,
  required Map<String, int> mergeRanks,
  required int bosId,
  required int eosId,
  required int padId,
  required int unkId,
}) {
  final vocabEntries = vocab.entries.toList(growable: false);
  final mergeEntries = mergeRanks.entries.toList(growable: false)
    ..sort((a, b) => a.value.compareTo(b.value));
  final vocabKeys = _CStringArray([
    for (final entry in vocabEntries) entry.key,
  ]);
  final mergeKeys = _CStringArray([
    for (final entry in mergeEntries) entry.key,
  ]);
  final vocabIds = vocabEntries.isEmpty
      ? ffi.nullptr
      : calloc<ffi.Int64>(vocabEntries.length);
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    for (var i = 0; i < vocabEntries.length; i += 1) {
      vocabIds[i] = vocabEntries[i].value;
    }
    final handle = native.bpeNew(
      vocabKeys.pointer,
      vocabIds,
      vocabEntries.length,
      mergeKeys.pointer,
      mergeEntries.length,
      bosId,
      eosId,
      padId,
      unkId,
      error,
    );
    if (handle == ffi.nullptr) {
      throw StateError(_takeFillError(error));
    }
    return handle;
  } finally {
    vocabKeys.close();
    mergeKeys.close();
    if (vocabIds != ffi.nullptr) {
      calloc.free(vocabIds);
    }
    calloc.free(error);
  }
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
    if (pointer == ffi.nullptr) {
      return;
    }
    for (var i = 0; i < length; i += 1) {
      final value = pointer[i];
      if (value != ffi.nullptr) {
        calloc.free(value);
      }
    }
    calloc.free(pointer);
  }
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
