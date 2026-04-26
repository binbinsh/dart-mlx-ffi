import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

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

  test('concatenates float32 audio chunks', () {
    final out = concatFloat32([
      Float32List.fromList([0.1, 0.2]),
      Float32List.fromList([0.3]),
    ]);

    expect(out[0], closeTo(0.1, 1e-6));
    expect(out[1], closeTo(0.2, 1e-6));
    expect(out[2], closeTo(0.3, 1e-6));
  });
}
