import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart' show NativeTensorBuffer;
import 'package:dart_inference/src/models/cosyvoice2/cosyvoice2_audio.dart'
    as cosy_audio;
import 'package:test/test.dart';

void main() {
  test('CosyVoice2 WAV helpers round-trip mono PCM16 audio', () {
    final samples = Float32List.fromList([-1.0, -0.25, 0.0, 0.25, 1.0]);
    final wav = cosy_audio.encodeWavPcm16(samples, sampleRate: 24000);
    final decoded = decodeWav(wav);

    expect(decoded.sampleRate, 24000);
    expect(decoded.samples.length, samples.length);
    expect(decoded.samples.first, closeTo(-1.0, 0.0001));
    expect(decoded.samples.last, closeTo(0.9999, 0.0001));
  });

  test('CosyVoice2 WAV helpers encode native-backed audio tensors', () {
    final buffer = NativeTensorBuffer.float32(const [3]);
    try {
      buffer.asFloat32List().setAll(0, [-1.0, 0.0, 1.0]);

      final wav = cosy_audio.encodeWavPcm16Tensor(
        buffer.tensor,
        sampleRate: 24000,
      );

      expect(String.fromCharCodes(wav.sublist(0, 4)), 'RIFF');
      expect(String.fromCharCodes(wav.sublist(8, 12)), 'WAVE');
      expect(wav.length, 50);
      expect(wav.sublist(44), [0x01, 0x80, 0x00, 0x00, 0xff, 0x7f]);
    } finally {
      buffer.close();
    }
  });

  test('CosyVoice2 WAV helpers encode a native-backed sample prefix', () {
    final buffer = NativeTensorBuffer.float32(const [5]);
    try {
      buffer.asFloat32List().setAll(0, [-1.0, 0.0, 1.0, 0.5, -0.5]);

      final wav = cosy_audio.encodeWavPcm16Source(
        buffer,
        sampleRate: 24000,
        sampleCount: 3,
      );
      final copied = cosy_audio.copyFloat32Prefix(buffer, 3);

      expect(wav.length, 50);
      expect(wav.sublist(44), [0x01, 0x80, 0x00, 0x00, 0xff, 0x7f]);
      expect(copied, [-1.0, 0.0, 1.0]);
    } finally {
      buffer.close();
    }
  });

  test('CosyVoice2 WAV helpers encode mixed native and heap chunks', () {
    final buffer = NativeTensorBuffer.float32(const [2]);
    try {
      buffer.asFloat32List().setAll(0, [-1.0, 0.0]);

      final wav = cosy_audio.encodeWavPcm16Sources([
        buffer,
        Float32List.fromList([1.0]),
      ], sampleRate: 24000);
      final concat = cosy_audio.concatFloat32Sources([
        Float32List.fromList([0.25]),
        buffer.tensor,
      ]);

      expect(wav.length, 50);
      expect(wav.sublist(44), [0x01, 0x80, 0x00, 0x00, 0xff, 0x7f]);
      expect(concat, [0.25, -1.0, 0.0]);
    } finally {
      buffer.close();
    }
  });

  test('CosyVoice2 audio data URL decoder accepts raw base64 payloads', () {
    final bytes = decodeAudioDataUrl('AQID');

    expect(bytes, [1, 2, 3]);
  });
}
