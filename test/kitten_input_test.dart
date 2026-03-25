// ignore_for_file: unused_import

@TestOn('mac-os')

library;

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/models.dart';

void main() {
  test('tokenizes phoneme text with punctuation preserved', () {
    expect(
      basicEnglishTokenize('h ə l oʊ , w ɜː l d !'),
      <String>['h', 'ə', 'l', 'o', 'ʊ', ',', 'w', 'ɜ', 'ː', 'l', 'd', '!'],
    );
  });

  test('builds kitten input ids with surrounding pad tokens', () {
    final ids = buildInputIdsFromPhonemes('h ə l oʊ');
    expect(ids.first, 0);
    expect(ids.last, 0);
    expect(ids.length, greaterThan(2));
  });

  test('builds kitten input array from phonemes', () {
    final array = buildInputArrayFromPhonemes('h ə l oʊ');
    try {
      expect(array.shape.first, 1);
      expect(array.shape.last, greaterThan(2));
    } finally {
      array.close();
    }
  });
}
