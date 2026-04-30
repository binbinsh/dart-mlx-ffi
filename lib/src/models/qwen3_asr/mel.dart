import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'mel_math.dart';

/// Whisper-compatible mel spectrogram frontend for Qwen3-ASR.
///
/// Parameters: 128 mel bins, n_fft=400, hop_length=160, sample_rate=16000.
/// Uses log10 mel with max-clamping (Whisper convention).
final class Qwen3AsrMelFrontend {
  Qwen3AsrMelFrontend()
    : _window = qwen3AsrHannWindow(nFft),
      _filterbank = qwen3AsrWhisperMelFilterbank(
        sampleRate: sampleRate,
        nFft: nFft,
        nMels: nMels,
      );

  static const int sampleRate = Qwen3AsrMelConstants.sampleRate;
  static const int nMels = Qwen3AsrMelConstants.nMels;
  static const int nFft = Qwen3AsrMelConstants.nFft;
  static const int hopLength = Qwen3AsrMelConstants.hopLength;

  final Float32List _window;
  final Float32List _filterbank;

  MlxArray? _windowArray;
  MlxArray? _filterbankArray;

  /// Compute mel spectrogram on GPU (MLX).
  /// Returns shape [1, nFrames, 128].
  MlxArray compute(Float32List audio) {
    final input = MlxArray.fromFloat32List(audio, shape: [audio.length]);
    MlxArray? padded;
    MlxArray? frameStarts;
    MlxArray? offsets;
    MlxArray? indices;
    MlxArray? frames;
    MlxArray? windowed;
    MlxArray? spectrum;
    MlxArray? magnitude;
    MlxArray? power;
    MlxArray? mel;
    MlxArray? logMel;
    try {
      // Reflect-pad by nFft // 2 on each side.
      padded = _reflectPad(input, nFft ~/ 2);
      final paddedLen = padded.shape[0];
      final frameCount = paddedLen < nFft
          ? 1
          : 1 + ((paddedLen - nFft) ~/ hopLength);

      // Build frame indices and apply window.
      frameStarts = MlxArray.arange(
        0.0,
        (frameCount * hopLength).toDouble(),
        hopLength.toDouble(),
        dtype: MlxDType.MLX_INT32,
      );
      offsets = MlxArray.arange(
        0.0,
        nFft.toDouble(),
        1.0,
        dtype: MlxDType.MLX_INT32,
      );
      indices = mx.add(frameStarts.expandDims(1), offsets.expandDims(0));
      frames = padded.take(indices, axis: 0);
      windowed = mx.multiply(frames, _windowMlx().reshape([1, nFft]));

      // FFT and power spectrum.
      spectrum = mx.fft.rfft(windowed, axis: 1);
      magnitude = mx.abs(spectrum);
      power = mx.multiply(magnitude, magnitude);

      // Apply mel filterbank.
      mel = mx.matmul(_filterbankMlx(), power.T);

      // Trim the last STFT frame to match the reference implementation.
      // Reference: `log_spec = ... [:, :-1]` in moona3k/mlx-qwen3-asr.
      final trimmedFrameCount = frameCount - 1;
      MlxArray? melTrimmed;
      try {
        melTrimmed = mel.slice(start: [0, 0], stop: [nMels, trimmedFrameCount]);

        // Log mel spectrogram (Whisper convention).
        logMel = _logMelWhisper(melTrimmed);
      } finally {
        melTrimmed?.close();
      }

      // Transpose to [1, nFrames, nMels].
      final result = logMel.T.reshape([1, trimmedFrameCount, nMels]);
      return result;
    } finally {
      input.close();
      padded?.close();
      frameStarts?.close();
      offsets?.close();
      indices?.close();
      frames?.close();
      windowed?.close();
      spectrum?.close();
      magnitude?.close();
      power?.close();
      mel?.close();
      logMel?.close();
    }
  }

  void close() {
    _windowArray?.close();
    _windowArray = null;
    _filterbankArray?.close();
    _filterbankArray = null;
  }

  MlxArray _windowMlx() {
    final cached = _windowArray;
    if (cached != null) return cached;
    final created = MlxArray.fromFloat32List(_window, shape: [nFft]);
    _windowArray = created;
    return created;
  }

  MlxArray _filterbankMlx() {
    final cached = _filterbankArray;
    if (cached != null) return cached;
    final bins = (nFft ~/ 2) + 1;
    final created = MlxArray.fromFloat32List(_filterbank, shape: [nMels, bins]);
    _filterbankArray = created;
    return created;
  }

  /// Whisper-style log mel: log10(clamp(mel, min=1e-10)),
  /// then clamp to max - 8, then (x + 4) / 4.
  MlxArray _logMelWhisper(MlxArray mel) {
    final clamped = mx.maximum(mel, MlxArray.full([], 1e-10));
    final log10 = mx.multiply(
      mx.log(clamped),
      MlxArray.full([], 0.4342944819032518),
    );
    clamped.close();
    // Compute global max via flatten → argmax → index.
    final flat = log10.reshape([log10.size]);
    final maxIdx = flat.argmax();
    final maxVal = flat.take(maxIdx, axis: 0).reshape([]);
    flat.close();
    maxIdx.close();
    final lower = mx.subtract(maxVal, MlxArray.full([], 8.0));
    maxVal.close();
    final clampedLog = mx.maximum(log10, lower);
    log10.close();
    lower.close();
    final shifted = mx.add(clampedLog, MlxArray.full([], 4.0));
    clampedLog.close();
    final normalized = mx.divide(shifted, MlxArray.full([], 4.0));
    shifted.close();
    return normalized;
  }

  MlxArray _reflectPad(MlxArray input, int padding) {
    if (padding <= 0) return input.astype(MlxDType.MLX_FLOAT32);
    final length = input.shape[0];
    if (length <= 1) {
      final indices = MlxArray.fromInt32List(
        List<int>.filled(length + (padding * 2), 0),
        shape: [length + (padding * 2)],
      );
      final repeated = input.take(indices, axis: 0);
      indices.close();
      return repeated;
    }
    final leftVals = Int32List.fromList(
      List<int>.generate(padding, (i) => (padding - i).clamp(1, length - 1)),
    );
    final rightVals = Int32List.fromList(
      List<int>.generate(padding, (i) => (length - 2 - i).clamp(0, length - 2)),
    );
    final leftIdx = MlxArray.fromInt32List(leftVals, shape: [leftVals.length]);
    final rightIdx = MlxArray.fromInt32List(
      rightVals,
      shape: [rightVals.length],
    );
    final left = input.take(leftIdx, axis: 0);
    final right = input.take(rightIdx, axis: 0);
    leftIdx.close();
    rightIdx.close();
    final result = mx.concatenate([left, input, right], axis: 0);
    left.close();
    right.close();
    return result;
  }
}
