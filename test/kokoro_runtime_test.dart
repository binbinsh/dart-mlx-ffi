import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';

void main() {
  test('filters phonemes to the Kokoro vocab surface', () {
    final vocab = {' ', 'a', 'b', 'ˈ', 'ɹ'};

    expect(filterPhonemesForVocab(" 'a ɹ2_b ", vocab), 'a ɹb');
    expect(filterPhonemesForVocab('ˈa   b', vocab), 'ˈa b');
  });

  test('chunks phonemes without exceeding the Kokoro token budget', () {
    final vocab = {' ': 16, 'a': 43, 'b': 44, '.': 4};

    expect(chunkPhonemesForKokoro('aaa bbb aaa', vocab, maxTokens: 6), [
      'aaa',
      'bbb',
      'aaa',
    ]);
    expect(chunkPhonemesForKokoro('aaaaaaaaa', vocab, maxTokens: 4), [
      'aaaa',
      'aaaa',
      'a',
    ]);
  });

  test('resolves voice fallback deterministically', () {
    final voices = {
      'zf_xiaoni': NpyArray(
        shape: const [1, 1],
        data: Float32List.fromList([0]),
      ),
    };

    expect(resolveKokoroVoice(voices, 'zf_xiaoni'), 'zf_xiaoni');
    expect(resolveKokoroVoice(voices, 'missing'), 'zf_xiaoni');
  });

  test('copies Kokoro voice rows through native buffers', () {
    final voice = NpyArray(
      shape: const [2, 3],
      data: Float32List.fromList([1, 2, 3, 4, 5, 6]),
    );
    final out = NativeTensorBuffer.float32(const [3]);
    try {
      voice.copyRowTo(out, 1);
      expect(out.asFloat32List(), [4, 5, 6]);
    } finally {
      voice.close();
      out.close();
    }
  });

  test('parses npy voice arrays through Zig', () {
    final voice = parseNpy(_npyBytes(const [2, 3], const [1, 2, 3, 4, 5, 6]));
    try {
      expect(voice.shape, const [2, 3]);
      expect(voice.row(1), [4, 5, 6]);
    } finally {
      voice.close();
    }
  });

  test('loads npz voice archives through Zig', () async {
    final dir = await Directory.systemTemp.createTemp('dart_inference_kokoro_');
    final path = '${dir.path}/voices.npz';
    await File(path).writeAsBytes(
      _storedZip({
        'zf_xiaoni.npy': _npyBytes(const [1, 2], const [0.25, 0.5]),
        'af_sky.npy': _npyBytes(const [1, 2], const [0.75, 1.0]),
      }),
    );

    final voices = await loadNpz(path);
    try {
      expect(voices.keys, containsAll(['zf_xiaoni', 'af_sky']));
      expect(voices['zf_xiaoni']!.row(0), [0.25, 0.5]);
      expect(voices['af_sky']!.row(0), [0.75, 1.0]);
    } finally {
      for (final voice in voices.values) {
        voice.close();
      }
      await dir.delete(recursive: true);
    }
  });

  test('concatenates float32 audio chunks', () {
    final out = concatFloat32([
      Float32List.fromList([0.1, 0.2]),
      Float32List(0),
      Float32List.fromList([0.3]),
    ]);

    expect(out[0], closeTo(0.1, 1e-6));
    expect(out[1], closeTo(0.2, 1e-6));
    expect(out[2], closeTo(0.3, 1e-6));
  });

  test('concatenates empty float32 audio chunks', () {
    final out = concatFloat32([Float32List(0), Float32List(0)]);

    expect(out, isEmpty);
  });

  test('encodes float32 audio as Zig-owned PCM16 WAV bytes', () {
    final wav = encodeWavPcm16(Float32List.fromList([-1.0, 0.0, 1.0]), 24000);

    expect(String.fromCharCodes(wav.sublist(0, 4)), 'RIFF');
    expect(String.fromCharCodes(wav.sublist(8, 12)), 'WAVE');
    expect(String.fromCharCodes(wav.sublist(12, 16)), 'fmt ');
    expect(String.fromCharCodes(wav.sublist(36, 40)), 'data');
    expect(wav.length, 50);
    expect(wav.sublist(44), [0x01, 0x80, 0x00, 0x00, 0xff, 0x7f]);
  });

  test('encodes chunked float32 audio without Dart pre-concat', () {
    final wav = encodeWavPcm16Chunks([
      Float32List.fromList([-1.0, 0.0]),
      Float32List.fromList([1.0]),
    ], 24000);

    expect(wav.length, 50);
    expect(wav.sublist(44), [0x01, 0x80, 0x00, 0x00, 0xff, 0x7f]);
  });
}

Uint8List _npyBytes(List<int> shape, List<double> values) {
  final shapeText = shape.length == 1 ? '${shape.single},' : shape.join(', ');
  final headerText =
      "{'descr': '<f4', 'fortran_order': False, 'shape': ($shapeText), }";
  final header = ascii.encode(headerText);
  final pad = (16 - ((10 + header.length + 1) % 16)) % 16;
  final headerBytes = Uint8List.fromList([
    ...header,
    ...List<int>.filled(pad, 0x20),
    0x0a,
  ]);
  final data = ByteData(values.length * 4);
  for (var i = 0; i < values.length; i += 1) {
    data.setFloat32(i * 4, values[i], Endian.little);
  }
  final out = BytesBuilder(copy: false)
    ..add([0x93, ...ascii.encode('NUMPY'), 1, 0])
    ..add(_u16(headerBytes.length))
    ..add(headerBytes)
    ..add(data.buffer.asUint8List());
  return out.toBytes();
}

Uint8List _storedZip(Map<String, Uint8List> files) {
  final out = BytesBuilder(copy: false);
  final central = BytesBuilder(copy: false);
  for (final entry in files.entries) {
    final name = ascii.encode(entry.key);
    final data = entry.value;
    final localOffset = out.length;
    out
      ..add([0x50, 0x4b, 0x03, 0x04])
      ..add(_u16(20))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u32(0))
      ..add(_u32(data.length))
      ..add(_u32(data.length))
      ..add(_u16(name.length))
      ..add(_u16(0))
      ..add(name)
      ..add(data);
    central
      ..add([0x50, 0x4b, 0x01, 0x02])
      ..add(_u16(20))
      ..add(_u16(20))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u32(0))
      ..add(_u32(data.length))
      ..add(_u32(data.length))
      ..add(_u16(name.length))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u16(0))
      ..add(_u32(0))
      ..add(_u32(localOffset))
      ..add(name);
  }
  final centralOffset = out.length;
  final centralBytes = central.toBytes();
  out
    ..add(centralBytes)
    ..add([0x50, 0x4b, 0x05, 0x06])
    ..add(_u16(0))
    ..add(_u16(0))
    ..add(_u16(files.length))
    ..add(_u16(files.length))
    ..add(_u32(centralBytes.length))
    ..add(_u32(centralOffset))
    ..add(_u16(0));
  return out.toBytes();
}

List<int> _u16(int value) => [value & 0xff, (value >> 8) & 0xff];

List<int> _u32(int value) => [
  value & 0xff,
  (value >> 8) & 0xff,
  (value >> 16) & 0xff,
  (value >> 24) & 0xff,
];
