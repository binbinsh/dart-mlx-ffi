// Parity test: Dart kaldi-fbank vs torchaudio.compliance.kaldi.fbank.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/src/models/cosyvoice2/cosyvoice2_kaldi_fbank.dart';
import 'package:test/test.dart';

void main() {
  group('CosyVoice2 kaldi fbank parity', () {
    final fixture = File('test/fixtures/cosyvoice2/kaldi_fbank_cases.json');
    if (!fixture.existsSync()) {
      throw StateError('Missing ${fixture.path}');
    }
    final cases = jsonDecode(fixture.readAsStringSync()) as List<dynamic>;
    for (final raw in cases) {
      final c = raw as Map<String, dynamic>;
      final name = c['name'] as String;
      test(name, () {
        final audio = Float32List.fromList(
            (c['audio'] as List).cast<num>().map((e) => e.toDouble()).toList());
        final shape = (c['expected_shape'] as List).cast<int>();
        final expected = (c['fbank'] as List).cast<num>();
        final got = computeKaldiFbank(audio, const KaldiFbankConfig());
        expect(got.nFrames, shape[0],
            reason: 'nFrames mismatch for $name');
        expect(got.numMelBins, shape[1]);
        // Kaldi fbank values are roughly in [-3, 12]; absolute tol
        // 1e-2 covers float32 vs float64 + matmul-DFT vs pocketfft.
        const tol = 1e-2;
        var fails = 0;
        var firstIdx = -1;
        var firstDiff = 0.0;
        for (var i = 0; i < expected.length; i += 1) {
          final e = expected[i].toDouble();
          final g = got.data[i];
          if (g.isNaN || g.isInfinite || (e - g).abs() > tol) {
            if (firstIdx == -1) {
              firstIdx = i;
              firstDiff = (e - g).abs();
            }
            fails += 1;
          }
        }
        expect(fails, 0,
            reason:
                '$fails/${expected.length} elements exceed tol=$tol; '
                'first idx=$firstIdx diff=$firstDiff '
                'expected=${firstIdx >= 0 ? expected[firstIdx] : ""} '
                'got=${firstIdx >= 0 ? got.data[firstIdx] : ""}');
      });
    }
  });
}
