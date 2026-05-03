import 'dart:math' as math;
import 'dart:typed_data';

/// Shared Whisper-compatible mel frontend constants for Qwen3-ASR.
abstract final class Qwen3AsrMelConstants {
  static const int sampleRate = 16000;
  static const int nMels = 128;
  static const int nFft = 400;
  static const int hopLength = 160;
}

/// Hann window of length [n] using the periodic convention.
Float32List qwen3AsrHannWindow(int n) {
  final out = Float32List(n);
  if (n <= 1) {
    if (n == 1) out[0] = 1;
    return out;
  }
  for (var i = 0; i < n; i++) {
    out[i] = 0.5 - 0.5 * math.cos((2 * math.pi * i) / n);
  }
  return out;
}

/// Whisper-style mel filterbank using Slaney mel scale and normalization.
///
/// Returns a flat array of shape `[nMels, nFft ~/ 2 + 1]`.
Float32List qwen3AsrWhisperMelFilterbank({
  required int sampleRate,
  required int nFft,
  required int nMels,
}) {
  final bins = (nFft ~/ 2) + 1;
  final out = Float32List(nMels * bins);
  final melMin = _slaneyHzToMel(0.0);
  final melMax = _slaneyHzToMel(sampleRate / 2.0);
  final melPoints = List<double>.generate(
    nMels + 2,
    (i) => melMin + (melMax - melMin) * i / (nMels + 1),
  );
  final hzPoints = melPoints.map(_slaneyMelToHz).toList(growable: false);
  final fftFreqs = List<double>.generate(
    bins,
    (i) => sampleRate * i / nFft,
    growable: false,
  );

  for (var m = 0; m < nMels; m++) {
    final lower = hzPoints[m];
    final center = hzPoints[m + 1];
    final upper = hzPoints[m + 2];
    final enorm = 2.0 / (upper - lower);
    for (var bin = 0; bin < bins; bin++) {
      final freq = fftFreqs[bin];
      final left = center > lower ? (freq - lower) / (center - lower) : 0.0;
      final right = upper > center ? (upper - freq) / (upper - center) : 0.0;
      out[m * bins + bin] = math.max(0.0, math.min(left, right)) * enorm;
    }
  }
  return out;
}

/// Whisper-style log mel normalization:
/// `log10(max(mel, 1e-10))`, clamp to `max - 8`, then `(x + 4) / 4`.
void qwen3AsrNormalizeLogMelInPlace(Float32List melValues) {
  if (melValues.isEmpty) return;
  var maxLog = double.negativeInfinity;
  for (var i = 0; i < melValues.length; i++) {
    final value = math.log(math.max(melValues[i], 1e-10)) / math.ln10;
    melValues[i] = value;
    if (value > maxLog) maxLog = value;
  }
  final lower = maxLog - 8.0;
  for (var i = 0; i < melValues.length; i++) {
    final clamped = math.max(melValues[i], lower);
    melValues[i] = ((clamped + 4.0) / 4.0).toDouble();
  }
}

/// Reflect-pad mono audio by [padding] samples on both sides.
Float32List qwen3AsrReflectPad(Float32List input, int padding) {
  if (padding <= 0) return Float32List.fromList(input);
  final length = input.length;
  final out = Float32List(length + (padding * 2));
  if (length <= 1) {
    final value = length == 0 ? 0.0 : input[0];
    for (var i = 0; i < out.length; i++) {
      out[i] = value;
    }
    return out;
  }
  for (var i = 0; i < padding; i++) {
    out[i] = input[(padding - i).clamp(1, length - 1)];
  }
  out.setRange(padding, padding + length, input);
  for (var i = 0; i < padding; i++) {
    out[padding + length + i] = input[(length - 2 - i).clamp(0, length - 2)];
  }
  return out;
}

const double _ln6_4 = 1.8562979903656263;

double _slaneyHzToMel(double hz) {
  if (hz < 1000.0) return 3.0 * hz / 200.0;
  return 15.0 + 27.0 * math.log(hz / 1000.0) / _ln6_4;
}

double _slaneyMelToHz(double mel) {
  if (mel < 15.0) return 200.0 * mel / 3.0;
  return 1000.0 * math.exp((mel - 15.0) * _ln6_4 / 27.0);
}
