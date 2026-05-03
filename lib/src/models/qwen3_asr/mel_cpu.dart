import 'dart:math' as math;
import 'dart:typed_data';

import 'mel_math.dart';

/// CPU mel result for native runtimes that do not use MLX tensors.
final class Qwen3AsrCpuMelTensor {
  const Qwen3AsrCpuMelTensor({required this.data, required this.shape});

  /// Flat row-major data with shape `[1, frames, 128]`.
  final Float32List data;

  /// Tensor shape, always `[1, frames, 128]`.
  final List<int> shape;

  int get frameCount => shape[1];
}

/// Pure Dart Whisper-compatible mel frontend for Qwen3-ASR native backends.
///
/// This mirrors [Qwen3AsrMelFrontend] without depending on MLX so ONNX Runtime
/// and LiteRT sessions can be driven on Windows, Linux, and Android.
final class Qwen3AsrCpuMelFrontend {
  Qwen3AsrCpuMelFrontend()
    : _window = qwen3AsrHannWindow(Qwen3AsrMelConstants.nFft),
      _filterbank = qwen3AsrWhisperMelFilterbank(
        sampleRate: Qwen3AsrMelConstants.sampleRate,
        nFft: Qwen3AsrMelConstants.nFft,
        nMels: Qwen3AsrMelConstants.nMels,
      ),
      _cos = _rfftTable(math.cos),
      _sin = _rfftTable(math.sin);

  final Float32List _window;
  final Float32List _filterbank;
  final Float32List _cos;
  final Float32List _sin;

  Qwen3AsrCpuMelTensor compute(Float32List audio) {
    const nFft = Qwen3AsrMelConstants.nFft;
    const hop = Qwen3AsrMelConstants.hopLength;
    const nMels = Qwen3AsrMelConstants.nMels;
    final bins = (nFft ~/ 2) + 1;
    final padded = qwen3AsrReflectPad(audio, nFft ~/ 2);
    final frameCount = padded.length < nFft
        ? 1
        : 1 + ((padded.length - nFft) ~/ hop);
    final trimmedFrameCount = math.max(0, frameCount - 1);
    final mel = Float32List(nMels * trimmedFrameCount);
    final power = Float32List(bins);

    for (var frame = 0; frame < trimmedFrameCount; frame++) {
      final start = frame * hop;
      _powerSpectrum(padded, start, power);
      for (var m = 0; m < nMels; m++) {
        var value = 0.0;
        final filterOffset = m * bins;
        for (var bin = 0; bin < bins; bin++) {
          value += _filterbank[filterOffset + bin] * power[bin];
        }
        mel[m * trimmedFrameCount + frame] = value.toDouble();
      }
    }

    qwen3AsrNormalizeLogMelInPlace(mel);

    final out = Float32List(trimmedFrameCount * nMels);
    for (var frame = 0; frame < trimmedFrameCount; frame++) {
      for (var melBin = 0; melBin < nMels; melBin++) {
        out[(frame * nMels) + melBin] =
            mel[(melBin * trimmedFrameCount) + frame];
      }
    }
    return Qwen3AsrCpuMelTensor(
      data: out,
      shape: [1, trimmedFrameCount, nMels],
    );
  }

  void _powerSpectrum(Float32List padded, int start, Float32List out) {
    const nFft = Qwen3AsrMelConstants.nFft;
    final bins = out.length;
    for (var bin = 0; bin < bins; bin++) {
      var real = 0.0;
      var imag = 0.0;
      final tableOffset = bin * nFft;
      for (var i = 0; i < nFft; i++) {
        final sample = padded[start + i] * _window[i];
        real += sample * _cos[tableOffset + i];
        imag -= sample * _sin[tableOffset + i];
      }
      out[bin] = (real * real + imag * imag).toDouble();
    }
  }
}

Float32List _rfftTable(double Function(double) fn) {
  const nFft = Qwen3AsrMelConstants.nFft;
  final bins = (nFft ~/ 2) + 1;
  final out = Float32List(bins * nFft);
  for (var bin = 0; bin < bins; bin++) {
    for (var i = 0; i < nFft; i++) {
      out[(bin * nFft) + i] = fn(2 * math.pi * bin * i / nFft).toDouble();
    }
  }
  return out;
}
