/// SpeechBrain-compatible ECAPA-TDNN feature frontend on MLX.
///
/// Pipeline (exact SpeechBrain `spkrec-ecapa-voxceleb` recipe):
///
/// 1. **STFT** — `center=True`, `pad_mode='constant'` (zero pad `n_fft//2` on
///    both sides), Hamming periodic window (`frontend.window` tensor baked into
///    the safetensors bundle), `n_fft = win_length = 400`, `hop_length = 160`,
///    `onesided = True`, `normalized_stft = False`.
/// 2. **Spectral power** — `|z|^2 = real^2 + imag^2`.
/// 3. **Filterbank** — multiply by baked triangular mel matrix
///    `frontend.mel_fb` of shape `(201, 80)` (htk mel, f_min=0, f_max=8000,
///    n_mels=80). Then `10 * log10(max(x, 1e-10))` and clamp by `top_db=80`
///    against the sequence-global max.
/// 4. **InputNorm (sentence)** — subtract per-feature mean across time. No
///    std normalization.
///
/// The output is a `(T, nMels)` MLX array in float32. Frame count `T` matches
/// SpeechBrain exactly: `T = len(waveform) // hop_length + 1` for the
/// centre-padded recipe with `win_length == n_fft`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';

/// Result of running the MLX frontend on a waveform.
class EcapaFbankResult {
  EcapaFbankResult({
    required this.raw,
    required this.norm,
    required this.frames,
    required this.nMels,
  });

  /// Pre mean-normalization log-mel array, `(frames, nMels)`. Matches
  /// `reference_fbank_raw.npy` exactly.
  final MlxArray raw;

  /// Post mean-normalization log-mel array, `(frames, nMels)`. Matches
  /// `reference_fbank.npy` exactly.
  final MlxArray norm;

  final int frames;
  final int nMels;

  void close() {
    raw.close();
    norm.close();
  }
}

/// Deterministic MLX implementation of the SpeechBrain ECAPA-TDNN feature
/// frontend. All weight tensors are owned by the caller-provided
/// [EcapaBundle]; this class keeps no persistent GPU state of its own.
final class EcapaFbankFrontend {
  EcapaFbankFrontend(this._bundle);

  final EcapaBundle _bundle;

  int get _nFft => _bundle.manifest.nFft;
  int get _winLength => _bundle.manifest.winLength;
  int get _hopLength => _bundle.manifest.hopLength;
  int get _nMels => _bundle.manifest.nMels;
  double get _logFloor => _bundle.manifest.logFloor;

  /// Number of frames produced for a waveform of [sampleCount] samples under
  /// the SpeechBrain centre-pad STFT (win_length == n_fft).
  int framesFor(int sampleCount) {
    if (sampleCount <= 0) return 0;
    final padAmount = _nFft ~/ 2;
    final padded = sampleCount + 2 * padAmount;
    if (padded < _nFft) return 0;
    return (padded - _nFft) ~/ _hopLength + 1;
  }

  /// Run the full SpeechBrain feature pipeline. The returned
  /// [EcapaFbankResult] owns two fresh MLX arrays; the caller must close
  /// them (via `result.close()`) when done.
  EcapaFbankResult encode(Float32List waveform) {
    if (waveform.isEmpty) {
      throw ArgumentError('EcapaFbankFrontend.encode requires non-empty audio');
    }
    if (_winLength != _nFft) {
      throw StateError(
        'ECAPA frontend assumes win_length == n_fft '
        '(got $_winLength vs $_nFft).',
      );
    }
    final window = _bundle.require('frontend.window');
    final melFb = _bundle.require('frontend.mel_fb');

    final hop = _hopLength;
    final nFft = _nFft;
    final winLen = _winLength;
    final nMels = _nMels;

    MlxArray? signal;
    MlxArray? zero;
    MlxArray? padded;
    MlxArray? frameStarts;
    MlxArray? offsets;
    MlxArray? indices;
    MlxArray? frameMatrix;
    MlxArray? windowed;
    MlxArray? spectrum;
    MlxArray? magnitude;
    MlxArray? power;
    MlxArray? mel;
    MlxArray? logFloorScalar;
    MlxArray? melClipped;
    MlxArray? natLog;
    MlxArray? log10Scale;
    MlxArray? logMelRaw;
    MlxArray? floorScalar;
    MlxArray? rawClamped;
    MlxArray? meanVec;
    MlxArray? rawClampedRetained;
    MlxArray? normRetained;

    try {
      // 1. Load waveform as (N,) float32 MLX array.
      signal = MlxArray.fromFloat32List(waveform, shape: [waveform.length]);

      // 2. Center-pad by n_fft//2 on each side with zeros (pad_mode='constant').
      final padAmount = nFft ~/ 2;
      zero = MlxArray.full([], 0.0);
      padded = signal.pad(
        axes: [0],
        lowPads: [padAmount],
        highPads: [padAmount],
        padValue: zero,
        mode: 'constant',
      );
      final paddedLen = waveform.length + 2 * padAmount;
      final actualFrames = (paddedLen - nFft) ~/ hop + 1;
      if (actualFrames <= 0) {
        throw StateError('ECAPA frontend produced 0 frames.');
      }

      // 3. Gather frames of length `nFft` (== winLen).
      frameStarts = MlxArray.arange(
        0.0,
        (actualFrames * hop).toDouble(),
        hop.toDouble(),
        dtype: MlxDType.MLX_INT32,
      );
      offsets = MlxArray.arange(
        0.0,
        nFft.toDouble(),
        1.0,
        dtype: MlxDType.MLX_INT32,
      );
      indices = mx.add(frameStarts.expandDims(1), offsets.expandDims(0));
      frameMatrix = padded.take(indices, axis: 0); // (frames, nFft)

      // 4. Apply Hamming window.
      windowed = mx.multiply(frameMatrix, window.reshape([1, winLen]));

      // 5. rfft along last axis. Output shape (frames, nFft/2+1) complex.
      spectrum = mx.fft.rfft(windowed, n: nFft, axis: 1);

      // 6. Power spectrum |z|^2.
      magnitude = mx.abs(spectrum);
      power = mx.multiply(magnitude, magnitude);

      // 7. Mel filterbank.
      mel = mx.matmul(power, melFb);

      // 8. 10 * log10(max(x, amin)).
      logFloorScalar = MlxArray.full([], _logFloor);
      melClipped = mx.maximum(mel, logFloorScalar);
      natLog = mx.log(melClipped);
      log10Scale = MlxArray.full([], 10.0 / math.ln10);
      logMelRaw = mx.multiply(natLog, log10Scale);

      // 9. top_db clamp: x = max(x, x.max() - 80). MLX ops have no max
      // reduction; evaluate once and compute the global max on CPU, then
      // broadcast a scalar floor back into MLX.
      MlxRuntime.evalAll([logMelRaw]);
      final flat = logMelRaw.toFloat32List();
      double globalMax = flat[0];
      for (var i = 1; i < flat.length; i++) {
        final v = flat[i];
        if (v > globalMax) globalMax = v;
      }
      const topDb = 80.0;
      floorScalar = MlxArray.full([], globalMax - topDb);
      rawClamped = mx.maximum(logMelRaw, floorScalar);

      // 10. InputNorm sentence: subtract per-feature mean over time axis.
      meanVec = rawClamped.mean(axis: 0, keepDims: true);
      final norm = mx.subtract(rawClamped, meanVec);

      // Realize outputs and transfer ownership to the caller.
      MlxRuntime.evalAll([rawClamped, norm]);
      rawClampedRetained = rawClamped;
      normRetained = norm;
      rawClamped = null; // prevent close in finally
      return EcapaFbankResult(
        raw: rawClampedRetained,
        norm: normRetained,
        frames: actualFrames,
        nMels: nMels,
      );
    } finally {
      signal?.close();
      zero?.close();
      padded?.close();
      frameStarts?.close();
      offsets?.close();
      indices?.close();
      frameMatrix?.close();
      windowed?.close();
      spectrum?.close();
      magnitude?.close();
      power?.close();
      mel?.close();
      logFloorScalar?.close();
      melClipped?.close();
      natLog?.close();
      log10Scale?.close();
      logMelRaw?.close();
      floorScalar?.close();
      rawClamped?.close();
      meanVec?.close();
      // rawClampedRetained / normRetained deliberately NOT closed here.
    }
  }
}
