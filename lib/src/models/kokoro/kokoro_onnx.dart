import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:dart_inference/runtime.dart';

final class KokoroDartRuntime {
  KokoroDartRuntime._({
    required this.session,
    required this.vocab,
    required this.voices,
    required this.selectedProvider,
  });

  final DartOnnxSession session;
  final Map<String, int> vocab;
  final Map<String, NpyArray> voices;
  final String selectedProvider;

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
    final config = jsonDecode(await File(configPath).readAsString());
    final vocabRaw = (config as Map)['vocab'] as Map;
    final vocab = vocabRaw.map((k, v) => MapEntry(k.toString(), v as int));
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
      vocab: vocab,
      voices: voices,
      selectedProvider: session.selectedProvider,
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
    final chunks = chunkPhonemes(phonemes);
    if (chunks.isEmpty) {
      throw FormatException('phonemes produced no Kokoro token ids');
    }
    final audioChunks = <Float32List>[];
    for (final chunk in chunks) {
      audioChunks.add(
        _synthesizeTokenIds(
          tokenIds: phonemeTokenIds(chunk),
          voiceArray: voiceArray,
          speed: speed,
        ),
      );
    }
    return encodeWavPcm16(concatFloat32(audioChunks), 24000);
  }

  Float32List _synthesizeTokenIds({
    required List<int> tokenIds,
    required NpyArray voiceArray,
    required double speed,
  }) {
    if (tokenIds.isEmpty) {
      return Float32List(0);
    }
    if (tokenIds.length > kokoroMaxPhonemeTokens) {
      throw StateError(
        'Kokoro phoneme chunk has ${tokenIds.length} tokens, '
        'max is $kokoroMaxPhonemeTokens',
      );
    }
    final tokenCount = tokenIds.length;
    final inputIds = Int64List(tokenCount + 2);
    inputIds[0] = 0;
    for (var i = 0; i < tokenCount; i++) {
      inputIds[i + 1] = tokenIds[i];
    }
    inputIds[tokenCount + 1] = 0;
    final style = voiceArray.row(tokenCount);
    final outputs = session.run({
      'input_ids': int64Tensor(inputIds, [1, inputIds.length]),
      'style': float32Tensor(style, [1, style.length]),
      'speed': float32Tensor(Float32List.fromList([speed]), const [1]),
    });
    final audioTensor = outputs.outputs.values.whereType<RuntimeTensor>().first;
    return Float32List.fromList(float32View(audioTensor));
  }

  String filterPhonemes(String phonemes) =>
      filterPhonemesForVocab(phonemes, vocab.keys.toSet());

  List<String> chunkPhonemes(
    String phonemes, {
    int maxTokens = kokoroMaxPhonemeTokens,
  }) => chunkPhonemesForKokoro(phonemes, vocab, maxTokens: maxTokens);

  int phonemeTokenCount(String phonemes) => phonemeTokenIds(phonemes).length;

  int phonemeChunkCount(String phonemes) => chunkPhonemes(phonemes).length;

  List<int> phonemeTokenIds(String phonemes) {
    final tokenIds = <int>[];
    for (final rune in phonemes.runes) {
      final token = vocab[String.fromCharCode(rune)];
      if (token != null) {
        tokenIds.add(token);
      }
    }
    return tokenIds;
  }

  String resolveVoice(String requested) {
    return resolveKokoroVoice(voices, requested);
  }

  void close() {
    session.close();
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
    final ch = String.fromCharCode(rune);
    if (!vocabChars.contains(ch)) {
      continue;
    }
    if (ch.trim().isEmpty) {
      if (!lastWasSpace && out.isNotEmpty) {
        out.write(' ');
      }
      lastWasSpace = true;
      continue;
    }
    out.write(ch);
    lastWasSpace = false;
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
  final filtered = filterPhonemesForVocab(phonemes, vocab.keys.toSet());
  if (filtered.isEmpty) {
    return const [];
  }
  final chunks = <String>[];
  final current = StringBuffer();
  var currentTokens = 0;

  void flush() {
    final chunk = current.toString().trim();
    if (chunk.isNotEmpty) {
      chunks.add(chunk);
    }
    current.clear();
    currentTokens = 0;
  }

  for (final segment in _phonemeSegments(filtered)) {
    final segmentTokens = _countVocabTokens(segment, vocab);
    if (segmentTokens == 0) {
      continue;
    }
    if (segmentTokens > maxTokens) {
      flush();
      final runes = segment.runes.toList(growable: false);
      for (final rune in runes) {
        final ch = String.fromCharCode(rune);
        if (!vocab.containsKey(ch)) {
          continue;
        }
        if (currentTokens == maxTokens) {
          flush();
        }
        current.write(ch);
        currentTokens += 1;
      }
      continue;
    }
    if (currentTokens > 0 && currentTokens + segmentTokens > maxTokens) {
      flush();
    }
    current.write(segment);
    currentTokens += segmentTokens;
  }
  flush();
  return chunks;
}

List<String> _phonemeSegments(String phonemes) {
  final segments = <String>[];
  final current = StringBuffer();
  for (final rune in phonemes.runes) {
    final ch = String.fromCharCode(rune);
    current.write(ch);
    if (ch.trim().isEmpty || _chunkBreakChars.contains(ch)) {
      segments.add(current.toString());
      current.clear();
    }
  }
  if (current.isNotEmpty) {
    segments.add(current.toString());
  }
  return segments;
}

const _chunkBreakChars = {'.', ',', '!', '?', ':', ';', '—', '…'};

int _countVocabTokens(String value, Map<String, int> vocab) {
  var count = 0;
  for (final rune in value.runes) {
    if (vocab.containsKey(String.fromCharCode(rune))) {
      count += 1;
    }
  }
  return count;
}

Float32List concatFloat32(List<Float32List> chunks) {
  if (chunks.isEmpty) {
    return Float32List(0);
  }
  if (chunks.length == 1) {
    return chunks.single;
  }
  final total = chunks.fold<int>(0, (sum, chunk) => sum + chunk.length);
  final out = Float32List(total);
  var offset = 0;
  for (final chunk in chunks) {
    out.setRange(offset, offset + chunk.length, chunk);
    offset += chunk.length;
  }
  return out;
}

Future<Map<String, NpyArray>> loadNpz(String path) async {
  final bytes = await File(path).readAsBytes();
  final archive = ZipDecoder().decodeBytes(bytes, verify: false);
  final out = <String, NpyArray>{};
  for (final file in archive.files) {
    if (!file.isFile || !file.name.endsWith('.npy')) {
      continue;
    }
    final name = file.name.replaceFirst(RegExp(r'\.npy$'), '');
    out[name] = parseNpy(Uint8List.fromList(file.content as List<int>));
  }
  if (out.isEmpty) {
    throw FormatException('voices npz contains no npy arrays: $path');
  }
  return out;
}

final class NpyArray {
  NpyArray({required this.shape, required this.data});

  final List<int> shape;
  final Float32List data;

  Float32List row(int index) {
    if (shape.isEmpty) {
      throw StateError('voice array has no dimensions');
    }
    final rows = shape.first;
    final safeIndex = index.clamp(0, rows - 1);
    final rowSize = shape.skip(1).fold<int>(1, (a, b) => a * b);
    final offset = safeIndex * rowSize;
    return Float32List.sublistView(data, offset, offset + rowSize);
  }
}

NpyArray parseNpy(Uint8List bytes) {
  if (bytes.length < 10 ||
      bytes[0] != 0x93 ||
      ascii.decode(bytes.sublist(1, 6)) != 'NUMPY') {
    throw FormatException('invalid npy header');
  }
  final major = bytes[6];
  final headerLen = major == 1
      ? ByteData.sublistView(bytes, 8, 10).getUint16(0, Endian.little)
      : ByteData.sublistView(bytes, 8, 12).getUint32(0, Endian.little);
  final headerStart = major == 1 ? 10 : 12;
  final header = ascii.decode(
    bytes.sublist(headerStart, headerStart + headerLen),
  );
  if (!header.contains("'descr': '<f4'") &&
      !header.contains('"descr": "<f4"')) {
    throw FormatException(
      'only little-endian float32 npy arrays are supported',
    );
  }
  if (header.contains("'fortran_order': True") ||
      header.contains('"fortran_order": true')) {
    throw FormatException('fortran-order npy arrays are not supported');
  }
  final shapeMatch =
      RegExp(r"'shape': \(([^)]*)\)").firstMatch(header) ??
      RegExp(r'"shape": \[([^\]]*)\]').firstMatch(header);
  if (shapeMatch == null) {
    throw FormatException('npy shape missing');
  }
  final shape = shapeMatch
      .group(1)!
      .split(',')
      .map((part) => part.trim())
      .where((part) => part.isNotEmpty)
      .map(int.parse)
      .toList(growable: false);
  final dataStart = headerStart + headerLen;
  final dataBytes = Uint8List.sublistView(bytes, dataStart);
  return NpyArray(
    shape: shape,
    data: Float32List.view(dataBytes.buffer, dataBytes.offsetInBytes),
  );
}

Uint8List encodeWavPcm16(Float32List audio, int sampleRate) {
  final pcm = Int16List(audio.length);
  for (var i = 0; i < audio.length; i++) {
    final clipped = audio[i].clamp(-1.0, 1.0);
    pcm[i] = (clipped * 32767.0).round();
  }
  final dataBytes = Uint8List.view(pcm.buffer);
  final out = BytesBuilder();
  void writeAscii(String value) => out.add(ascii.encode(value));
  void writeU16(int value) {
    final b = ByteData(2)..setUint16(0, value, Endian.little);
    out.add(Uint8List.view(b.buffer));
  }

  void writeU32(int value) {
    final b = ByteData(4)..setUint32(0, value, Endian.little);
    out.add(Uint8List.view(b.buffer));
  }

  writeAscii('RIFF');
  writeU32(36 + dataBytes.length);
  writeAscii('WAVEfmt ');
  writeU32(16);
  writeU16(1);
  writeU16(1);
  writeU32(sampleRate);
  writeU32(sampleRate * 2);
  writeU16(2);
  writeU16(16);
  writeAscii('data');
  writeU32(dataBytes.length);
  out.add(dataBytes);
  return out.toBytes();
}
