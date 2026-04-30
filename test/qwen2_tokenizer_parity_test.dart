// Dart-side parity test for the Qwen2 BPE tokenizer.
//
// Drives the full Dart -> C ABI -> native -> tokenizer pipeline against the
// HuggingFace ground-truth fixture committed under
// `test/fixtures/cosyvoice2/qwen2_tokenizer_cases.json`.
//
// The test is gated on the env var `QWEN2_TOKENIZER_DIR` so CI/local
// runs without the cosyvoice2 model snapshot are skipped (the cases JSON
// is committed; the vocab/merges are not).

import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

void main() {
  final tokenizerDir = Platform.environment['QWEN2_TOKENIZER_DIR'];
  final casesPath =
      Platform.environment['QWEN2_TOKENIZER_CASES'] ??
      'test/fixtures/cosyvoice2/qwen2_tokenizer_cases.json';

  group('Qwen2BpeTokenizer parity', () {
    late Qwen2BpeTokenizer tokenizer;
    late List<Map<String, dynamic>> cases;

    setUpAll(() async {
      if (tokenizerDir == null || tokenizerDir.isEmpty) {
        return; // tests below skip via runtime check
      }
      tokenizer = await Qwen2BpeTokenizer.load(tokenizerDir);
      final raw = await File(casesPath).readAsString();
      final payload = jsonDecode(raw) as Map<String, dynamic>;
      final list = payload['cases'] as List;
      cases = [for (final c in list) (c as Map).cast<String, dynamic>()];
    });

    tearDownAll(() {
      if (tokenizerDir != null && tokenizerDir.isNotEmpty) {
        tokenizer.close();
      }
    });

    test('matches HuggingFace ground truth on every fixture case', () {
      if (tokenizerDir == null || tokenizerDir.isEmpty) {
        markTestSkipped('QWEN2_TOKENIZER_DIR not set');
        return;
      }
      var passed = 0;
      final failures = <String>[];
      for (var i = 0; i < cases.length; i += 1) {
        final text = cases[i]['text'] as String;
        final expected = (cases[i]['ids'] as List)
            .map((e) => (e as num).toInt())
            .toList(growable: false);
        final actual = tokenizer.encode(text);
        if (actual.length == expected.length &&
            List.generate(
              actual.length,
              (k) => actual[k] == expected[k],
            ).every((v) => v)) {
          passed += 1;
        } else {
          failures.add(
            'case $i (${jsonEncode(text)}): expected $expected, got $actual',
          );
        }
      }
      expect(
        failures,
        isEmpty,
        reason:
            'parity: $passed/${cases.length} passed.\n'
            '${failures.join('\n')}',
      );
    });
  });
}
