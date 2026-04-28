import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:dart_inference/runtime.dart';
import '../../runtime/native_bindings.dart' as native;

final class KokoroDartRuntime {
  KokoroDartRuntime._({
    required this.session,
    required _KokoroVocab nativeVocab,
    required this.voices,
    required this.selectedProvider,
    required _KokoroInputScratch scratch,
  }) : _nativeVocab = nativeVocab,
       _scratch = scratch;

  final DartOnnxSession session;
  final _KokoroVocab _nativeVocab;
  final Map<String, NpyArray> voices;
  final String selectedProvider;
  final _KokoroInputScratch _scratch;

  List<String> get voiceNames => voices.keys.toList(growable: false)..sort();

  static Future<KokoroDartRuntime> load({
    required String modelPath,
    required String voicesPath,
    required String configPath,
    required String provider,
    required int deviceId,
    required bool requireProvider,
    required int numThreads,
    Map<String, Object?> backendOptions = const {},
  }) async {
    final nativeVocab = _loadKokoroVocab(configPath);
    final voices = await loadNpz(voicesPath);
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: modelPath,
        id: 'kokoro_onnx_dart',
        family: 'kokoro',
        provider: provider,
        deviceId: deviceId,
        requireProvider: requireProvider,
        numThreads: numThreads,
        backendOptions: backendOptions,
      ),
    );
    return KokoroDartRuntime._(
      session: session,
      nativeVocab: nativeVocab,
      voices: voices,
      selectedProvider: session.selectedProvider,
      scratch: _KokoroInputScratch(
        maxInputLength: kokoroMaxPhonemeTokens + 2,
        styleLength: voices.values.first.rowLength,
      ),
    );
  }

  Uint8List synthesizePhonemes({
    required String phonemes,
    required String voice,
    required double speed,
  }) {
    final resolvedVoice = resolveVoice(voice);
    final voiceArray = voices[resolvedVoice];
    if (voiceArray == null) {
      throw FormatException('unknown Kokoro voice: $voice');
    }
    final plan = _planKokoro(phonemes, _nativeVocab);
    final audioChunks = <_KokoroAudioChunk>[];
    try {
      if (plan.tokenCount == 0 || plan.chunkCount == 0) {
        throw FormatException('phonemes produced no Kokoro token ids');
      }
      var tokenOffset = 0;
      for (var i = 0; i < plan.chunkCount; i += 1) {
        final tokenCount = plan.lengths[i];
        audioChunks.add(
          _synthesizeNativeTokenIds(
            tokenIds: plan.tokens + tokenOffset,
            tokenCount: tokenCount,
            voiceArray: voiceArray,
            speed: speed,
          ),
        );
        tokenOffset += tokenCount;
      }
      return _encodeWavPcm16TensorChunks(audioChunks, 24000);
    } finally {
      for (final chunk in audioChunks) {
        chunk.close();
      }
      plan.close();
    }
  }

  _KokoroAudioChunk _synthesizeNativeTokenIds({
    required ffi.Pointer<ffi.Int64> tokenIds,
    required int tokenCount,
    required NpyArray voiceArray,
    required double speed,
  }) {
    if (tokenCount == 0) {
      return _KokoroAudioChunk.empty();
    }
    if (tokenCount > kokoroMaxPhonemeTokens) {
      throw StateError(
        'Kokoro phoneme chunk has $tokenCount tokens, '
        'max is $kokoroMaxPhonemeTokens',
      );
    }
    final outputs = session.run(
      _scratch.inputs(
        tokenIds: tokenIds,
        tokenCount: tokenCount,
        voiceArray: voiceArray,
        speed: speed,
      ),
    );
    try {
      final audioTensor = outputs.outputs.values
          .whereType<RuntimeTensor>()
          .first;
      return _KokoroAudioChunk(audioTensor, outputs);
    } catch (_) {
      outputs.close();
      rethrow;
    }
  }

  String filterPhonemes(String phonemes) =>
      _filterKokoro(phonemes, _nativeVocab.codes, _nativeVocab.count);

  List<String> chunkPhonemes(
    String phonemes, {
    int maxTokens = kokoroMaxPhonemeTokens,
  }) {
    final plan = _planKokoro(
      phonemes,
      _nativeVocab,
      maxTokens: maxTokens,
      includeText: true,
    );
    try {
      return plan.chunks();
    } finally {
      plan.close();
    }
  }

  int phonemeTokenCount(String phonemes) {
    final plan = _planKokoro(phonemes, _nativeVocab);
    try {
      return plan.tokenCount;
    } finally {
      plan.close();
    }
  }

  int phonemeChunkCount(String phonemes) {
    final plan = _planKokoro(phonemes, _nativeVocab);
    try {
      return plan.chunkCount;
    } finally {
      plan.close();
    }
  }

  List<int> phonemeTokenIds(String phonemes) {
    final plan = _planKokoro(phonemes, _nativeVocab);
    try {
      return List<int>.generate(
        plan.tokenCount,
        (index) => plan.tokens[index],
        growable: false,
      );
    } finally {
      plan.close();
    }
  }

  String resolveVoice(String requested) {
    return resolveKokoroVoice(voices, requested);
  }

  void close() {
    _scratch.close();
    _nativeVocab.close();
    for (final voice in voices.values) {
      voice.close();
    }
    session.close();
  }
}

final class _KokoroInputScratch {
  _KokoroInputScratch({required this.maxInputLength, required int styleLength})
    : inputIds = NativeTensorBuffer.int64([1, maxInputLength]),
      style = NativeTensorBuffer.float32([1, styleLength]),
      speed = NativeTensorBuffer.float32(const [1]);

  final int maxInputLength;
  final NativeTensorBuffer inputIds;
  final NativeTensorBuffer style;
  final NativeTensorBuffer speed;
  bool _closed = false;

  Map<String, Object?> inputs({
    required ffi.Pointer<ffi.Int64> tokenIds,
    required int tokenCount,
    required NpyArray voiceArray,
    required double speed,
  }) {
    final inputLength = tokenCount + 2;
    if (inputLength > maxInputLength) {
      throw StateError(
        'Kokoro input length $inputLength exceeds native input buffer.',
      );
    }

    final styleLength = voiceArray.rowLength;
    _fillKokoroInputs(
      inputIds: inputIds,
      tokenIds: tokenIds,
      tokenCount: tokenCount,
      style: style,
      voiceArray: voiceArray,
      voiceRowLength: styleLength,
      speedBuffer: this.speed,
      speed: speed,
    );

    return {
      'input_ids': inputIds.tensorView(
        shape: [1, inputLength],
        byteLength: inputLength * 8,
      ),
      'style': style.tensorView(
        shape: [1, styleLength],
        byteLength: styleLength * 4,
      ),
      'speed': this.speed.tensor,
    };
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    inputIds.close();
    style.close();
    speed.close();
  }
}

void _fillKokoroInputs({
  required NativeTensorBuffer inputIds,
  required ffi.Pointer<ffi.Int64> tokenIds,
  required int tokenCount,
  required NativeTensorBuffer style,
  required NpyArray voiceArray,
  required int voiceRowLength,
  required NativeTensorBuffer speedBuffer,
  required double speed,
}) {
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.kokInputs(
      inputIds.nativeData.cast<ffi.Int64>(),
      inputIds.byteLength ~/ 8,
      tokenIds,
      tokenCount,
      style.nativeData.cast<ffi.Float>(),
      style.byteLength ~/ 4,
      voiceArray._buffer.nativeData.cast<ffi.Float>(),
      voiceArray._buffer.byteLength ~/ 4,
      voiceArray.shape.first,
      voiceRowLength,
      speedBuffer.nativeData.cast<ffi.Float>(),
      speedBuffer.byteLength ~/ 4,
      speed,
      error,
    );
    if (status != 0) {
      throw StateError(_takeNativeError(error));
    }
  } finally {
    calloc.free(error);
  }
}

final class _KokoroAudioChunk {
  _KokoroAudioChunk(this.tensor, this._outputs);

  _KokoroAudioChunk.empty() : tensor = null, _outputs = null;

  final RuntimeTensor? tensor;
  final DartOnnxResult? _outputs;

  void close() {
    _outputs?.close();
  }
}

String resolveKokoroVoice(Map<String, NpyArray> voices, String requested) {
  if (voices.containsKey(requested)) {
    return requested;
  }
  if (voices.containsKey('zf_xiaoni')) {
    return 'zf_xiaoni';
  }
  if (voices.containsKey('af_sky')) {
    return 'af_sky';
  }
  if (voices.isEmpty) {
    throw const FormatException('Kokoro voices are empty');
  }
  return voices.keys.first;
}

const kokoroMaxPhonemeTokens = 510;

String filterPhonemesForVocab(String phonemes, Set<String> vocabChars) {
  final codes = _KokoroCodeSet(vocabChars);
  try {
    return _filterKokoro(phonemes, codes.codes, codes.count);
  } finally {
    codes.close();
  }
}

List<String> chunkPhonemesForKokoro(
  String phonemes,
  Map<String, int> vocab, {
  int maxTokens = kokoroMaxPhonemeTokens,
}) {
  final nativeVocab = _KokoroVocab(vocab);
  try {
    final plan = _planKokoro(
      phonemes,
      nativeVocab,
      maxTokens: maxTokens,
      includeText: true,
    );
    try {
      return plan.chunks();
    } finally {
      plan.close();
    }
  } finally {
    nativeVocab.close();
  }
}

final class _KokoroCodeSet {
  _KokoroCodeSet(Iterable<String> chars)
    : codes = NativeTensorBuffer.int32([chars.length]) {
    final values = codes.asInt32List();
    var index = 0;
    for (final char in chars) {
      final runes = char.runes.toList(growable: false);
      if (runes.length != 1) {
        codes.close();
        throw FormatException('Kokoro vocab key must be one codepoint: $char');
      }
      values[index] = runes.single;
      index += 1;
    }
  }

  final NativeTensorBuffer codes;

  int get count => codes.byteLength ~/ 4;

  void close() {
    codes.close();
  }
}

final class _KokoroVocab {
  _KokoroVocab(Map<String, int> vocab)
    : codes = NativeTensorBuffer.int32([vocab.length]),
      ids = NativeTensorBuffer.int64([vocab.length]) {
    final codeValues = codes.asInt32List();
    final values = ids.asInt64List();
    var index = 0;
    for (final entry in vocab.entries) {
      final runes = entry.key.runes.toList(growable: false);
      if (runes.length != 1) {
        codes.close();
        ids.close();
        throw FormatException(
          'Kokoro vocab key must be one codepoint: ${entry.key}',
        );
      }
      codeValues[index] = runes.single;
      values[index] = entry.value;
      index += 1;
    }
  }

  _KokoroVocab._native({required this.codes, required this.ids});

  final NativeTensorBuffer codes;
  final NativeTensorBuffer ids;

  int get count => codes.byteLength ~/ 4;

  void close() {
    codes.close();
    ids.close();
  }
}

_KokoroVocab _loadKokoroVocab(String configPath) {
  final path = configPath.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final out = calloc<native.KokoroVocabAbi>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.kokVocab(path, out, error);
    if (status != 0) {
      throw StateError(_takeNativeError(error));
    }
    final value = out.ref;
    if (value.count < 0) {
      throw const FormatException('invalid native Kokoro vocab metadata');
    }
    if (value.count > 0 &&
        (value.codes == ffi.nullptr || value.ids == ffi.nullptr)) {
      throw const FormatException('native Kokoro vocab pointers are null');
    }
    final codes = NativeTensorBuffer.adopt(
      dtype: RuntimeTensorDataType.int32,
      shape: [value.count],
      byteLength: value.count * 4,
      pointer: value.codes.cast<ffi.Void>(),
    );
    final ids = NativeTensorBuffer.adopt(
      dtype: RuntimeTensorDataType.int64,
      shape: [value.count],
      byteLength: value.count * 8,
      pointer: value.ids.cast<ffi.Void>(),
    );
    value
      ..codes = ffi.nullptr
      ..ids = ffi.nullptr;
    return _KokoroVocab._native(codes: codes, ids: ids);
  } catch (_) {
    native.kokFreeVocab(out);
    rethrow;
  } finally {
    calloc
      ..free(path)
      ..free(out)
      ..free(error);
  }
}

final class _KokoroPlan {
  _KokoroPlan(this._pointer);

  final ffi.Pointer<native.KokoroPlanAbi> _pointer;
  bool _closed = false;

  native.KokoroPlanAbi get _value => _pointer.ref;

  ffi.Pointer<ffi.Int64> get tokens => _value.tokens;

  ffi.Pointer<ffi.IntPtr> get lengths => _value.lengths;

  int get tokenCount => _value.tokenCount;

  int get chunkCount => _value.chunkCount;

  List<String> chunks() {
    if (chunkCount == 0) {
      return const [];
    }
    final value = _value;
    final out = <String>[];
    for (var i = 0; i < value.chunkCount; i += 1) {
      final start = value.starts[i];
      final length = value.byteLengths[i];
      out.add((value.text + start).cast<Utf8>().toDartString(length: length));
    }
    return List<String>.unmodifiable(out);
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    native.kokFreePlan(_pointer);
    calloc.free(_pointer);
  }
}

String _filterKokoro(String phonemes, NativeTensorBuffer codes, int codeCount) {
  final text = phonemes.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  ffi.Pointer<ffi.Char> out = ffi.nullptr;
  try {
    out = native.kokFilter(
      text,
      codes.nativeData.cast<ffi.Int32>(),
      codeCount,
      error,
    );
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

_KokoroPlan _planKokoro(
  String phonemes,
  _KokoroVocab vocab, {
  int maxTokens = kokoroMaxPhonemeTokens,
  bool includeText = false,
}) {
  if (maxTokens <= 0) {
    throw ArgumentError.value(maxTokens, 'maxTokens', 'must be positive');
  }
  final text = phonemes.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final out = calloc<native.KokoroPlanAbi>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.kokPlan(
      text,
      vocab.codes.nativeData.cast<ffi.Int32>(),
      vocab.ids.nativeData.cast<ffi.Int64>(),
      vocab.count,
      maxTokens,
      includeText ? 1 : 0,
      out,
      error,
    );
    if (status != 0) {
      throw StateError(_takeNativeError(error));
    }
    return _KokoroPlan(out);
  } catch (_) {
    native.kokFreePlan(out);
    calloc.free(out);
    rethrow;
  } finally {
    calloc
      ..free(text)
      ..free(error);
  }
}

Float32List concatFloat32(List<Float32List> chunks) {
  if (chunks.isEmpty) {
    return Float32List(0);
  }
  if (chunks.length == 1) {
    return chunks.single;
  }
  final sampleCount = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  ffi.Pointer<ffi.Void> out = ffi.nullptr;
  try {
    return _withNativeF32Chunks(chunks, (chunkPtrs, chunkLengths) {
      out = native.audioConcatF32(
        chunkPtrs,
        chunkLengths,
        chunks.length,
        sampleCount,
        error,
      );
      if (out == ffi.nullptr) {
        if (sampleCount.value == 0 && error.value == ffi.nullptr) {
          return Float32List(0);
        }
        throw StateError(_takeNativeError(error));
      }
      return Float32List.fromList(
        out.cast<ffi.Float>().asTypedList(sampleCount.value),
      );
    });
  } finally {
    if (out != ffi.nullptr) {
      native.freeBuf(out);
    }
    calloc
      ..free(sampleCount)
      ..free(error);
  }
}

T _withNativeF32Chunks<T>(
  List<Float32List> chunks,
  T Function(
    ffi.Pointer<ffi.Pointer<ffi.Float>> chunkPtrs,
    ffi.Pointer<ffi.IntPtr> chunkLengths,
  )
  call,
) {
  final chunkPtrs = chunks.isEmpty
      ? ffi.nullptr
      : calloc<ffi.Pointer<ffi.Float>>(chunks.length);
  final chunkLengths = chunks.isEmpty
      ? ffi.nullptr
      : calloc<ffi.IntPtr>(chunks.length);
  final samplePtrs = <ffi.Pointer<ffi.Float>>[];
  try {
    for (var i = 0; i < chunks.length; i += 1) {
      final chunk = chunks[i];
      chunkLengths[i] = chunk.length;
      if (chunk.isEmpty) {
        chunkPtrs[i] = ffi.nullptr;
        continue;
      }
      final samples = calloc<ffi.Float>(chunk.length);
      samples.asTypedList(chunk.length).setAll(0, chunk);
      samplePtrs.add(samples);
      chunkPtrs[i] = samples;
    }
    return call(chunkPtrs, chunkLengths);
  } finally {
    for (final samples in samplePtrs) {
      calloc.free(samples);
    }
    if (chunkPtrs != ffi.nullptr) {
      calloc.free(chunkPtrs);
    }
    if (chunkLengths != ffi.nullptr) {
      calloc.free(chunkLengths);
    }
  }
}

Future<Map<String, NpyArray>> loadNpz(String path) async {
  final pathPtr = path.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final itemsOut = calloc<ffi.Pointer<native.NpyAbi>>();
  final countOut = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  final out = <String, NpyArray>{};
  ffi.Pointer<native.NpyAbi> items = ffi.nullptr;
  try {
    final status = native.kokNpz(pathPtr, itemsOut, countOut, error);
    if (status != 0) {
      throw StateError(_takeNativeError(error));
    }
    items = itemsOut.value;
    for (var i = 0; i < countOut.value; i += 1) {
      final itemPtr = items + i;
      final item = itemPtr.ref;
      if (item.name == ffi.nullptr) {
        throw const FormatException('voices npz entry is missing a name');
      }
      final name = item.name.cast<Utf8>().toDartString();
      final array = _npyFromAbi(item);
      item.data = ffi.nullptr;
      out.remove(name)?.close();
      out[name] = array;
    }
    if (out.isEmpty) {
      throw FormatException('voices npz contains no npy arrays: $path');
    }
    return out;
  } catch (_) {
    for (final value in out.values) {
      value.close();
    }
    rethrow;
  } finally {
    if (items != ffi.nullptr) {
      native.kokFreeNpz(items, countOut.value);
    }
    calloc
      ..free(pathPtr)
      ..free(itemsOut)
      ..free(countOut)
      ..free(error);
  }
}

final class NpyArray {
  NpyArray({required List<int> shape, required Float32List data})
    : shape = List<int>.unmodifiable(shape),
      _buffer = NativeTensorBuffer.float32(shape) {
    final expected = _shapeSize(shape);
    if (data.length != expected) {
      _buffer.close();
      throw FormatException(
        'npy shape expects $expected float32 values, got ${data.length}',
      );
    }
    _buffer.copyFrom(data);
  }

  NpyArray._native({
    required List<int> shape,
    required NativeTensorBuffer buffer,
  }) : shape = List<int>.unmodifiable(shape),
       _buffer = buffer;

  final List<int> shape;
  final NativeTensorBuffer _buffer;
  bool _closed = false;

  Float32List get data => _buffer.asFloat32List();

  int get rowLength {
    if (shape.isEmpty) {
      throw StateError('voice array has no dimensions');
    }
    return shape.skip(1).fold<int>(1, (a, b) => a * b);
  }

  Float32List row(int index) {
    final rows = shape.first;
    final safeIndex = index.clamp(0, rows - 1).toInt();
    final rowSize = rowLength;
    final offset = safeIndex * rowSize;
    return Float32List.sublistView(data, offset, offset + rowSize);
  }

  void copyRowTo(NativeTensorBuffer target, int index) {
    final rowSize = rowLength;
    _copyKokoroRow(
      target: target,
      voiceArray: this,
      voiceRowLength: rowSize,
      index: index < 0 ? 0 : index,
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _buffer.close();
  }
}

NpyArray _npyFromAbi(native.NpyAbi value) {
  if (value.rank < 0 || value.byteLength < 0) {
    throw const FormatException('invalid native npy metadata');
  }
  if (value.byteLength > 0 && value.data == ffi.nullptr) {
    throw const FormatException('native npy data pointer is null');
  }
  final shape = value.rank == 0
      ? const <int>[]
      : List<int>.generate(value.rank, (index) => value.shape[index]);
  return NpyArray._native(
    shape: shape,
    buffer: NativeTensorBuffer.adopt(
      dtype: RuntimeTensorDataType.float32,
      shape: shape,
      byteLength: value.byteLength,
      pointer: value.data,
    ),
  );
}

int _shapeSize(List<int> shape) {
  var size = 1;
  for (final dim in shape) {
    if (dim < 0) {
      throw FormatException('npy shape contains a negative dimension');
    }
    size *= dim;
  }
  return size;
}

NpyArray parseNpy(Uint8List bytes) {
  final input = bytes.isEmpty ? ffi.nullptr : calloc<ffi.Uint8>(bytes.length);
  final out = calloc<native.NpyAbi>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    if (input != ffi.nullptr) {
      input.asTypedList(bytes.length).setAll(0, bytes);
    }
    final status = native.kokNpy(input, bytes.length, out, error);
    if (status != 0) {
      throw StateError(_takeNativeError(error));
    }
    final array = _npyFromAbi(out.ref);
    out.ref.data = ffi.nullptr;
    return array;
  } finally {
    native.kokFreeNpy(out);
    if (input != ffi.nullptr) {
      calloc.free(input);
    }
    calloc
      ..free(out)
      ..free(error);
  }
}

Uint8List encodeWavPcm16(Float32List audio, int sampleRate) {
  final samples = audio.isEmpty ? ffi.nullptr : calloc<ffi.Float>(audio.length);
  final byteLength = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  ffi.Pointer<ffi.Void> wav = ffi.nullptr;
  try {
    if (samples != ffi.nullptr) {
      samples.asTypedList(audio.length).setAll(0, audio);
    }
    wav = native.audioWavPcm16(
      samples,
      audio.length,
      sampleRate,
      byteLength,
      error,
    );
    if (wav == ffi.nullptr) {
      throw StateError(_takeNativeError(error));
    }
    return Uint8List.fromList(
      wav.cast<ffi.Uint8>().asTypedList(byteLength.value),
    );
  } finally {
    if (wav != ffi.nullptr) {
      native.freeBuf(wav);
    }
    if (samples != ffi.nullptr) {
      calloc.free(samples);
    }
    calloc
      ..free(byteLength)
      ..free(error);
  }
}

Uint8List encodeWavPcm16Chunks(List<Float32List> chunks, int sampleRate) {
  final byteLength = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  ffi.Pointer<ffi.Void> wav = ffi.nullptr;
  try {
    return _withNativeF32Chunks(chunks, (chunkPtrs, chunkLengths) {
      wav = native.audioWavPcm16Chunks(
        chunkPtrs,
        chunkLengths,
        chunks.length,
        sampleRate,
        byteLength,
        error,
      );
      if (wav == ffi.nullptr) {
        throw StateError(_takeNativeError(error));
      }
      return Uint8List.fromList(
        wav.cast<ffi.Uint8>().asTypedList(byteLength.value),
      );
    });
  } finally {
    if (wav != ffi.nullptr) {
      native.freeBuf(wav);
    }
    calloc
      ..free(byteLength)
      ..free(error);
  }
}

Uint8List _encodeWavPcm16TensorChunks(
  List<_KokoroAudioChunk> chunks,
  int sampleRate,
) {
  final chunkPtrs = chunks.isEmpty
      ? ffi.nullptr
      : calloc<ffi.Pointer<ffi.Float>>(chunks.length);
  final chunkLengths = chunks.isEmpty
      ? ffi.nullptr
      : calloc<ffi.IntPtr>(chunks.length);
  final byteLength = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  ffi.Pointer<ffi.Void> wav = ffi.nullptr;
  try {
    for (var i = 0; i < chunks.length; i += 1) {
      final tensor = chunks[i].tensor;
      if (tensor == null) {
        chunkLengths[i] = 0;
        chunkPtrs[i] = ffi.nullptr;
        continue;
      }
      if (tensor.dtype != RuntimeTensorDataType.float32) {
        throw StateError('Kokoro audio output dtype is ${tensor.dtype.name}.');
      }
      final sampleCount = tensor.bytes.lengthInBytes ~/ 4;
      final data = tensor.nativeData ?? ffi.nullptr;
      if (sampleCount > 0 && data == ffi.nullptr) {
        throw StateError('Kokoro audio output is not native-backed.');
      }
      chunkLengths[i] = sampleCount;
      chunkPtrs[i] = sampleCount == 0 ? ffi.nullptr : data.cast<ffi.Float>();
    }
    wav = native.audioWavPcm16Chunks(
      chunkPtrs,
      chunkLengths,
      chunks.length,
      sampleRate,
      byteLength,
      error,
    );
    if (wav == ffi.nullptr) {
      throw StateError(_takeNativeError(error));
    }
    return Uint8List.fromList(
      wav.cast<ffi.Uint8>().asTypedList(byteLength.value),
    );
  } finally {
    if (wav != ffi.nullptr) {
      native.freeBuf(wav);
    }
    if (chunkPtrs != ffi.nullptr) {
      calloc.free(chunkPtrs);
    }
    if (chunkLengths != ffi.nullptr) {
      calloc.free(chunkLengths);
    }
    calloc
      ..free(byteLength)
      ..free(error);
  }
}

void _copyKokoroRow({
  required NativeTensorBuffer target,
  required NpyArray voiceArray,
  required int voiceRowLength,
  required int index,
}) {
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.kokRow(
      target.nativeData.cast<ffi.Float>(),
      target.byteLength ~/ 4,
      voiceArray._buffer.nativeData.cast<ffi.Float>(),
      voiceArray._buffer.byteLength ~/ 4,
      voiceArray.shape.first,
      voiceRowLength,
      index,
      error,
    );
    if (status != 0) {
      throw StateError(_takeNativeError(error));
    }
  } finally {
    calloc.free(error);
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
