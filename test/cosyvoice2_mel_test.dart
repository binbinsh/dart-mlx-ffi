// Parity test for Dart mel extractors vs upstream Python references.
//
// Fixture is generated once by Python (matcha mel_spectrogram +
// whisper.log_mel_spectrogram, see commit message) and committed at
// test/fixtures/cosyvoice2/mel_cases.json.
//
// Tolerance:
//   * matcha80: log-magnitude domain. Compare with absolute tol 5e-3
//     and relative tol 5e-3. Larger than typical because we replace
//     torch.stft (which uses pocketfft) with a matmul-DFT in float32
//     (DFT basis is float64, accum is float64, output cast to f32).
//   * whisper128: same. The whisper post-norm divides by 4 so the
//     output is in roughly [0, 1] and the same absolute tol works.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/src/models/cosyvoice2/cosyvoice2_mel.dart';
import 'package:dart_inference/src/models/cosyvoice2/cosyvoice2_prompt_native.dart';
import 'package:test/test.dart';

void main() {
  group('CosyVoice2 mel parity', () {
    final fixture = File('test/fixtures/cosyvoice2/mel_cases.json');
    if (!fixture.existsSync()) {
      throw StateError('Missing mel_cases.json fixture at ${fixture.path}');
    }
    final cases = jsonDecode(fixture.readAsStringSync()) as List<dynamic>;

    for (final raw in cases) {
      final c = raw as Map<String, dynamic>;
      final name = c['name'] as String;
      final kind = c['kind'] as String;
      test('$name ($kind)', () {
        final audio = Float32List.fromList(
          (c['audio'] as List).cast<num>().map((e) => e.toDouble()).toList(),
        );
        final expectedShape = (c['expected_shape'] as List).cast<int>();
        final expected = (c['expected'] as List).cast<num>();

        final cfg = kind == 'matcha80'
            ? MelConfig.matcha80
            : MelConfig.whisper128;
        final got = computeMelSpectrogram(audio, cfg);
        final nativePlan = CosyPromptNativePlan();
        addTearDown(nativePlan.close);
        final nativeGot = cosyPromptMelSpectrogramBuffer(
          audio,
          kind: kind == 'matcha80'
              ? CosyPromptMelKind.matcha80
              : CosyPromptMelKind.whisper128,
          plan: nativePlan,
        );
        addTearDown(nativeGot.close);

        expect(
          got.numMels,
          expectedShape[0],
          reason: 'numMels mismatch for $name',
        );
        expect(
          got.nFrames,
          expectedShape[1],
          reason: 'nFrames mismatch for $name',
        );
        expect(
          nativeGot.bins,
          expectedShape[0],
          reason: 'native numMels mismatch for $name',
        );
        expect(
          nativeGot.frames,
          expectedShape[1],
          reason: 'native nFrames mismatch for $name',
        );

        // Element-wise comparison.
        const absTol = 5e-3;
        const relTol = 5e-3;
        var maxAbs = 0.0;
        var argmaxAbs = -1;
        final nativeData = nativeGot.data.asFloat32List();
        for (var i = 0; i < expected.length; i += 1) {
          final e = expected[i].toDouble();
          final g = got.data[i];
          final ng = nativeData[i];
          final diff = (e - g).abs();
          final nativeDiff = (e - ng).abs();
          final tol = absTol + relTol * e.abs();
          if (g.isNaN ||
              g.isInfinite ||
              ng.isNaN ||
              ng.isInfinite ||
              diff > tol ||
              nativeDiff > tol) {
            if (diff > maxAbs ||
                nativeDiff > maxAbs ||
                g.isNaN ||
                g.isInfinite ||
                ng.isNaN ||
                ng.isInfinite) {
              maxAbs = diff;
              if (nativeDiff > maxAbs) maxAbs = nativeDiff;
              argmaxAbs = i;
            }
          }
        }
        expect(
          argmaxAbs,
          -1,
          reason:
              'first failing element idx=$argmaxAbs maxAbsDiff=$maxAbs '
              'expected=${argmaxAbs >= 0 ? expected[argmaxAbs] : ""} '
              'got=${argmaxAbs >= 0 ? got.data[argmaxAbs] : ""} '
              'nativeGot=${argmaxAbs >= 0 ? nativeData[argmaxAbs] : ""}',
        );
      });
    }
  });
}
