import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

void main() {
  test('uses native eSpeak phonemization by default', () {
    final phonemizer = KokoroPhonemizer();

    expect(phonemizer.backendName, 'lazy');
  });

  test('normalizes input and caches backend results', () async {
    var calls = 0;
    final phonemizer = KokoroPhonemizer.withBackend((
      text, {
      required language,
    }) {
      calls += 1;
      return '$language|$text';
    });

    final first = await phonemizer.phonemize('Hello   Moon');
    final second = await phonemizer.phonemize('Hello Moon');

    expect(first, 'en-us|Hello Moon');
    expect(second, first);
    expect(calls, 1);
  });

  test('normalizes English text through native before phonemization', () async {
    final calls = <String>[];
    final phonemizer = KokoroPhonemizer.withBackend((
      text, {
      required language,
    }) {
      calls.add('$language:$text');
      return text;
    });

    await phonemizer.phonemize('Dr. Smith paid \$1.50 at 12:05 in 2026');

    expect(calls, [
      "en-us:Doctor Smith paid 1 dollar and 50 cents at 12 oh 5 in 20 26",
    ]);
  });

  test('post-processes English phoneme output through native', () async {
    final phonemizer = KokoroPhonemizer.withBackend((
      text, {
      required language,
    }) {
      return 'kəkˈoːɹoʊ nˈaɪnti ahˈʌndɹɪd z! r x ɬ ʲ';
    });

    final phonemes = await phonemizer.phonemize('kokoro');

    expect(phonemes, 'kˈoʊkəɹoʊ nˈaɪndi a hˈʌndɹɪdz! ɹ k l j');
  });

  test(
    'splits mixed Chinese and Latin text before backend phonemization',
    () async {
      final calls = <String>[];
      final phonemizer = KokoroPhonemizer.withBackend((
        text, {
        required language,
      }) {
        calls.add('$language:$text');
        return '<$language:$text>';
      });

      final phonemes = await phonemizer.phonemize(
        'Hello 你好 Moon',
        language: 'cmn',
      );

      expect(calls, ['en-us:Hello ', 'cmn:你好 ', 'en-us:Moon']);
      expect(phonemes, '<en-us:Hello > <cmn:你好 > <en-us:Moon>');
    },
  );

  test('routes single-script fragments to the matching eSpeak voice', () async {
    final calls = <String>[];
    final phonemizer = KokoroPhonemizer.withBackend((
      text, {
      required language,
    }) {
      calls.add('$language:$text');
      return '<$language:$text>';
    });

    expect(await phonemizer.phonemize('and', language: 'cmn'), '<en-us:and>');
    expect(await phonemizer.phonemize('你好', language: 'en-us'), '<cmn:你好>');
    expect(calls, ['en-us:and', 'cmn:你好']);
  });

  test(
    'keeps a phoneme boundary when scripts touch without whitespace',
    () async {
      final phonemizer = KokoroPhonemizer.withBackend((
        text, {
        required language,
      }) {
        return '<$language:$text>';
      });

      final phonemes = await phonemizer.phonemize('Pay \$一', language: 'cmn');

      expect(phonemes, '<en-us:Pay \$> <cmn:一>');
    },
  );

  test(
    'uses SSML phoneme tags instead of re-phonemizing tagged text',
    () async {
      final calls = <String>[];
      final phonemizer = KokoroPhonemizer.withBackend((
        text, {
        required language,
      }) {
        calls.add('$language:$text');
        return '<$language:$text>';
      });

      final phonemes = await phonemizer.phonemizeSsml(
        '<speak><phoneme ph="L_IY_D">Lead</phoneme> us '
        '<sub alias="one coin">\$1</sub>.</speak>',
      );

      expect(phonemes, 'L_IY_D <en-us:us one coin.>');
      expect(calls, ['en-us:us one coin.']);
    },
  );

  test('normalizes explicit IPA stress markers for Kokoro vocab', () async {
    final phonemizer = KokoroPhonemizer.withBackend((
      text, {
      required language,
    }) {
      return '<$language:$text>';
    });

    final phonemes = await phonemizer.phonemizeSsml(
      '<speak><phoneme ph="\'liːd">Lead</phoneme> '
      '<phoneme ph="ɹə\'kɔːɹd">record</phoneme></speak>',
    );

    expect(phonemes, 'ˈliːd ɹəˈkɔːɹd');
  });

  test('converts pinyin SSML phoneme tags with cmn backend', () async {
    final calls = <String>[];
    final phonemizer = KokoroPhonemizer.withBackend((
      text, {
      required language,
    }) {
      calls.add('$language:$text');
      return '<$language:$text>';
    });

    final phonemes = await phonemizer.phonemizeSsml(
      '<speak>银<phoneme ph="hang2">行</phoneme>'
      '<phoneme ph="xing2">行</phoneme></speak>',
      language: 'cmn',
    );

    expect(phonemes, '<cmn:银> <cmn:hang> <cmn:xing>');
    expect(calls, ['cmn:银', 'cmn:hang2', 'cmn:xing2']);
  });

  test('uses eSpeak-NG FFI when the native library is available', () async {
    final phonemizer = KokoroPhonemizer();
    try {
      final phonemes = await phonemizer.phonemize('Hello World');
      expect(phonemes, isNotEmpty);
      expect(phonemes, isNot(contains('\u200d')));
      expect(phonemes, contains('h'));
      expect(phonemizer.backendName, 'espeak_native');
    } catch (error) {
      markTestSkipped('eSpeak-NG FFI library unavailable: $error');
    } finally {
      phonemizer.dispose();
    }
  });

  test('mixed-language FFI output strips eSpeak voice markers', () async {
    final phonemizer = KokoroPhonemizer();
    try {
      final phonemes = await phonemizer.phonemize('Hello 你好', language: 'cmn');
      expect(phonemes, isNot(contains(RegExp(r'\([a-z]{2,3}\)'))));
      expect(phonemizer.backendName, 'espeak_native');
    } catch (error) {
      markTestSkipped('eSpeak-NG FFI library unavailable: $error');
    } finally {
      phonemizer.dispose();
    }
  });
}
