import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

import '../cosyvoice2/cosyvoice2_audio.dart' as audio_io;

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
      filterPhonemesForVocab(phonemes, _nativeVocab.asMap().keys.toSet());

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
  final input = inputIds.asInt64List();
  if (input.length < tokenCount + 2) {
    throw StateError('input_ids buffer is too small for $tokenCount tokens.');
  }
  input[0] = 0;
  for (var i = 0; i < tokenCount; i += 1) {
    input[i + 1] = tokenIds[i];
  }
  input[tokenCount + 1] = 0;

  if (voiceRowLength > style.byteLength ~/ 4) {
    throw StateError('style buffer is too small for Kokoro voice row.');
  }
  voiceArray.copyRowTo(style, tokenCount);

  final speedValues = speedBuffer.asFloat32List();
  if (speedValues.isEmpty) {
    throw StateError('speed buffer is empty.');
  }
  speedValues[0] = speed;
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
  final out = StringBuffer();
  var lastWasSpace = true;
  for (final rune in phonemes.runes) {
    final char = String.fromCharCode(rune);
    if (!vocabChars.contains(char)) {
      continue;
    }
    if (char == ' ') {
      if (!lastWasSpace) {
        out.write(char);
        lastWasSpace = true;
      }
    } else {
      out.write(char);
      lastWasSpace = false;
    }
  }
  return out.toString().trim();
}

List<String> chunkPhonemesForKokoro(
  String phonemes,
  Map<String, int> vocab, {
  int maxTokens = kokoroMaxPhonemeTokens,
}) {
  if (maxTokens <= 0) {
    throw ArgumentError.value(maxTokens, 'maxTokens', 'must be positive');
  }
  final chunks = <String>[];
  final current = <String>[];

  void flush() {
    while (current.isNotEmpty && current.first.trim().isEmpty) {
      current.removeAt(0);
    }
    while (current.isNotEmpty && current.last.trim().isEmpty) {
      current.removeLast();
    }
    if (current.isNotEmpty) {
      chunks.add(current.join());
      current.clear();
    }
  }

  void appendChar(String char) {
    if (current.length == maxTokens) {
      flush();
    }
    current.add(char);
  }

  var lastWasSpace = true;
  for (final rune in phonemes.runes) {
    final char = String.fromCharCode(rune);
    if (!vocab.containsKey(char)) {
      continue;
    }
    final isSpace = char.trim().isEmpty;
    if (isSpace) {
      if (!lastWasSpace && current.isNotEmpty) {
        appendChar(' ');
        flush();
      }
      lastWasSpace = true;
      continue;
    }
    appendChar(char);
    lastWasSpace = false;
    if (_isKokoroBreak(char)) {
      flush();
    }
  }
  flush();
  return List<String>.unmodifiable(chunks);
}

bool _isKokoroBreak(String char) {
  return char == '.' ||
      char == ',' ||
      char == '!' ||
      char == '?' ||
      char == ':' ||
      char == ';' ||
      char == '—' ||
      char == '…';
}

final class _KokoroPlan {
  _KokoroPlan({
    required List<int> tokenIds,
    required List<int> chunkLengths,
    required List<String> textChunks,
  }) : _tokens = NativeTensorBuffer.int64([tokenIds.length]),
       _lengths = NativeTensorBuffer.nativeFfi(
         dtype: RuntimeTensorDataType.int64,
         shape: [chunkLengths.length],
       ),
       _chunks = List<String>.unmodifiable(textChunks) {
    _tokens.asInt64List().setAll(0, tokenIds);
    _lengths.asInt64List().setAll(0, chunkLengths);
  }

  final NativeTensorBuffer _tokens;
  final NativeTensorBuffer _lengths;
  final List<String> _chunks;
  bool _closed = false;

  ffi.Pointer<ffi.Int64> get tokens => _tokens.nativeData.cast<ffi.Int64>();

  ffi.Pointer<ffi.IntPtr> get lengths => _lengths.nativeData.cast<ffi.IntPtr>();

  int get tokenCount => _tokens.byteLength ~/ 8;

  int get chunkCount => _chunks.length;

  List<String> chunks() => _chunks;

  void close() {
    if (_closed) return;
    _closed = true;
    _lengths.close();
    _tokens.close();
  }
}

_KokoroVocab _loadKokoroVocab(String configPath) {
  final decoded = jsonDecode(File(configPath).readAsStringSync());
  if (decoded is! Map) {
    throw const FormatException('Kokoro config must be a JSON object');
  }
  final vocabRaw = decoded['vocab'] ?? decoded['tokenizer']?['vocab'];
  if (vocabRaw is! Map) {
    throw const FormatException('Kokoro config is missing vocab');
  }
  final vocab = <String, int>{};
  for (final entry in vocabRaw.entries) {
    if (entry.key is String && entry.value is num) {
      vocab[entry.key as String] = (entry.value as num).toInt();
    }
  }
  return _KokoroVocab(vocab);
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
  final vocabMap = vocab.asMap();
  final textChunks = includeText
      ? chunkPhonemesForKokoro(phonemes, vocabMap, maxTokens: maxTokens)
      : <String>[];
  final chunks = textChunks.isNotEmpty
      ? textChunks
      : chunkPhonemesForKokoro(phonemes, vocabMap, maxTokens: maxTokens);
  final tokenIds = <int>[];
  final lengths = <int>[];
  for (final chunk in chunks) {
    final start = tokenIds.length;
    for (final rune in chunk.runes) {
      final id = vocabMap[String.fromCharCode(rune)];
      if (id != null) tokenIds.add(id);
    }
    lengths.add(tokenIds.length - start);
  }
  return _KokoroPlan(
    tokenIds: tokenIds,
    chunkLengths: lengths,
    textChunks: includeText ? chunks : const [],
  );
}

extension on _KokoroVocab {
  Map<String, int> asMap() {
    final codeValues = codes.asInt32List();
    final idValues = ids.asInt64List();
    return {
      for (var i = 0; i < count; i += 1)
        String.fromCharCode(codeValues[i]): idValues[i],
    };
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

  final NativeTensorBuffer codes;
  final NativeTensorBuffer ids;

  int get count => codes.byteLength ~/ 4;

  void close() {
    codes.close();
    ids.close();
  }
}

Float32List concatFloat32(List<Float32List> chunks) {
  if (chunks.isEmpty) {
    return Float32List(0);
  }
  if (chunks.length == 1) {
    return chunks.single;
  }
  return audio_io.concatFloat32Sources(chunks);
}

Future<Map<String, NpyArray>> loadNpz(String path) async {
  final out = <String, NpyArray>{};
  try {
    for (final entry in _readStoredZip(File(path).readAsBytesSync()).entries) {
      if (!entry.key.endsWith('.npy')) continue;
      final name = entry.key
          .split('/')
          .last
          .replaceFirst(RegExp(r'\.npy$'), '');
      final array = parseNpy(entry.value);
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
  }
}

final class NpyArray {
  NpyArray({required List<int> shape, required Float32List data})
    : shape = List<int>.unmodifiable(shape),
      _buffer = _npyFloat32Buffer(shape, data);

  final List<int> shape;
  final NativeTensorBuffer _buffer;
  bool _closed = false;

  Float32List get data => _buffer.asFloat32List();

  ffi.Pointer<ffi.Float> get nativeFloatData =>
      _buffer.nativeData.cast<ffi.Float>();

  int get valueCount => _buffer.byteLength ~/ 4;

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

NativeTensorBuffer _npyFloat32Buffer(List<int> shape, Float32List data) {
  final expected = _shapeSize(shape);
  if (data.length != expected) {
    throw FormatException(
      'npy shape expects $expected float32 values, got ${data.length}',
    );
  }
  return nativeFloat32Buffer(data, shape: shape);
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
  if (bytes.length < 10 ||
      bytes[0] != 0x93 ||
      ascii.decode(bytes.sublist(1, 6)) != 'NUMPY') {
    throw const FormatException('invalid npy header');
  }
  final data = ByteData.sublistView(bytes);
  final major = bytes[6];
  final headerLength = switch (major) {
    1 => data.getUint16(8, Endian.little),
    2 || 3 => data.getUint32(8, Endian.little),
    _ => throw const FormatException('unsupported npy version'),
  };
  final headerStart = major == 1 ? 10 : 12;
  final header = ascii.decode(
    bytes.sublist(headerStart, headerStart + headerLength),
  );
  if (!header.contains("'descr': '<f4'") &&
      !header.contains('"descr": "<f4"')) {
    throw const FormatException('only little-endian float32 npy is supported');
  }
  final shapeMatch =
      RegExp(r"'shape': \(([^)]*)\)").firstMatch(header) ??
      RegExp(r'"shape": \[([^]]*)\]').firstMatch(header);
  if (shapeMatch == null) {
    throw const FormatException('npy shape is missing');
  }
  final shape = [
    for (final part in shapeMatch.group(1)!.split(','))
      if (part.trim().isNotEmpty) int.parse(part.trim()),
  ];
  final offset = headerStart + headerLength;
  final count = _shapeSize(shape);
  final values = Float32List(count);
  final payload = ByteData.sublistView(bytes, offset);
  for (var i = 0; i < count; i += 1) {
    values[i] = payload.getFloat32(i * 4, Endian.little);
  }
  return NpyArray(shape: shape, data: values);
}

Uint8List encodeWavPcm16(Float32List audio, int sampleRate) {
  return audio_io.encodeWavPcm16(audio, sampleRate: sampleRate);
}

Uint8List encodeWavPcm16Chunks(List<Float32List> chunks, int sampleRate) {
  return audio_io.encodeWavPcm16Sources(chunks, sampleRate: sampleRate);
}

Uint8List _encodeWavPcm16TensorChunks(
  List<_KokoroAudioChunk> chunks,
  int sampleRate,
) {
  return audio_io.encodeWavPcm16Sources([
    for (final chunk in chunks) chunk.tensor ?? Float32List(0),
  ], sampleRate: sampleRate);
}

void _copyKokoroRow({
  required NativeTensorBuffer target,
  required NpyArray voiceArray,
  required int voiceRowLength,
  required int index,
}) {
  if (target.dtype != RuntimeTensorDataType.float32) {
    throw StateError('Expected float32 target, got ${target.dtype.name}.');
  }
  final out = target.asFloat32List();
  if (out.length != voiceRowLength) {
    throw StateError(
      'target length is ${out.length}, expected $voiceRowLength.',
    );
  }
  final row = voiceArray.row(index);
  out.setAll(0, row);
}

Map<String, Uint8List> _readStoredZip(Uint8List bytes) {
  final out = <String, Uint8List>{};
  final data = ByteData.sublistView(bytes);
  var offset = 0;
  while (offset + 30 <= bytes.length) {
    if (data.getUint32(offset, Endian.little) != 0x04034b50) break;
    final method = data.getUint16(offset + 8, Endian.little);
    final compressedSize = data.getUint32(offset + 18, Endian.little);
    final uncompressedSize = data.getUint32(offset + 22, Endian.little);
    final nameLength = data.getUint16(offset + 26, Endian.little);
    final extraLength = data.getUint16(offset + 28, Endian.little);
    final nameStart = offset + 30;
    final bodyStart = nameStart + nameLength + extraLength;
    final bodyEnd = bodyStart + compressedSize;
    if (bodyEnd > bytes.length) {
      throw const FormatException('truncated zip entry');
    }
    if (method != 0 || compressedSize != uncompressedSize) {
      throw const FormatException('only stored npz entries are supported');
    }
    final name = utf8.decode(bytes.sublist(nameStart, nameStart + nameLength));
    out[name] = Uint8List.fromList(bytes.sublist(bodyStart, bodyEnd));
    offset = bodyEnd;
  }
  return out;
}
