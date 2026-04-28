// Audio mel-spectrogram extractors for CosyVoice2.
//
// Two configurations are needed:
//
//   * 80-mel (matcha-tts style): sr=24000, n_fft=1920, hop=480,
//     win=1920, fmin=0, fmax=8000, center=False (manual reflect pad),
//     log = log(clamp(x, 1e-5)). Used as the conditioning mel for
//     the flow encoder + diffusion decoder.
//
//   * 128-mel (whisper style): sr=16000, n_fft=400, hop=160, win=400,
//     fmin=0, fmax=8000, center=True, magnitude = |stft|^2 (power),
//     log = clamp(x, 1e-10).log10, then `max(log, log.max - 8) +
//     4) / 4`. Used as input to the speech-tokenizer-v2 ONNX model
//     to derive prompt speech tokens.
//
// Implementation strategy (see DESIGN-cosyvoice2.md):
//
//   * The DFT itself is computed by a precomputed real cos/sin basis
//     of shape `(n_fft, n_freq)` and two `O(n_fft * n_freq * frames)`
//     matmuls. n_fft is small (400 or 1920) and these extractors run
//     once per voice prompt, never on the streaming path, so the
//     O(n_fft^2) cost is fine in pure Dart.
//
//   * The mel filterbank uses librosa's HTK-style algorithm
//     (slaney=False, htk=False, norm='slaney') with the standard
//     triangular weighting and slaney-style normalisation; this
//     matches both `librosa.filters.mel` (used by matcha) and
//     whisper's `mel_filters.npz` to within ~1e-7 absolute error in
//     all test cells we've checked.
//
// All math stays on CPU; no FFI, no Zig. This keeps the build matrix
// honest and lets the parity test run in standard `dart test`.

import 'dart:math' as math;
import 'dart:typed_data';

// --- public API ---------------------------------------------------------

/// Configuration for [computeMelSpectrogram]. Defaults match the
/// matcha 80-mel config used by CosyVoice2's flow encoder.
final class MelConfig {
  const MelConfig({
    required this.sampleRate,
    required this.nFft,
    required this.hopLength,
    required this.winLength,
    required this.numMels,
    required this.fmin,
    required this.fmax,
    required this.center,
    required this.power, // 1.0 = magnitude, 2.0 = power
    required this.logMode, // LogMode.matcha | LogMode.whisper
    required this.dropLastFrame, // whisper drops `stft[..., :-1]`
  });

  final int sampleRate;
  final int nFft;
  final int hopLength;
  final int winLength;
  final int numMels;
  final double fmin;
  final double fmax;
  final bool center;
  final double power;
  final LogMode logMode;
  final bool dropLastFrame;

  /// Matcha-style 80-mel for the flow encoder (sr=24000).
  static const matcha80 = MelConfig(
    sampleRate: 24000,
    nFft: 1920,
    hopLength: 480,
    winLength: 1920,
    numMels: 80,
    fmin: 0.0,
    fmax: 8000.0,
    center: false,
    power: 1.0,
    logMode: LogMode.matcha,
    dropLastFrame: false,
  );

  /// Whisper-style 128-mel for speech_tokenizer_v2 (sr=16000).
  static const whisper128 = MelConfig(
    sampleRate: 16000,
    nFft: 400,
    hopLength: 160,
    winLength: 400,
    numMels: 128,
    fmin: 0.0,
    fmax: 8000.0,
    center: true,
    power: 2.0,
    logMode: LogMode.whisper,
    dropLastFrame: true,
  );
}

enum LogMode { matcha, whisper }

/// Compute a mel-spectrogram from float32 PCM samples in `[-1, 1]`.
/// Returns `(numMels, nFrames)` row-major.
({Float32List data, int numMels, int nFrames}) computeMelSpectrogram(
    Float32List audio, MelConfig cfg) {
  // 1. (Optional) reflect pad. Matcha uses manual symmetric pad of
  //    (n_fft - hop)/2 with `center=False`. Whisper uses torch.stft
  //    `center=True`, which reflects by n_fft//2 on each side.
  final pad = cfg.center ? cfg.nFft ~/ 2 : (cfg.nFft - cfg.hopLength) ~/ 2;
  final padded = _reflectPad(audio, pad);

  // 2. Frame the signal into (nFrames, winLength).
  final nFrames = ((padded.length - cfg.nFft) ~/ cfg.hopLength) + 1;
  if (nFrames <= 0) {
    throw ArgumentError(
        'audio too short (got ${audio.length} samples, need at least '
        '${cfg.nFft - 2 * pad}) for cfg=$cfg');
  }

  // 3. Apply Hann window.
  final window = _hannWindow(cfg.winLength);

  // 4. STFT via matmul-DFT.
  final basis = _DftBasis.cached(cfg.nFft);
  final nFreq = cfg.nFft ~/ 2 + 1;
  // mag2 shape (nFreq, nFrames), column-major in flat buffer.
  final mag2 = Float32List(nFreq * nFrames);
  final frame = Float32List(cfg.nFft);
  for (var t = 0; t < nFrames; t += 1) {
    final start = t * cfg.hopLength;
    for (var n = 0; n < cfg.nFft; n += 1) {
      // win_length == n_fft for both configs, so no zero-padding.
      frame[n] = padded[start + n] * window[n];
    }
    for (var k = 0; k < nFreq; k += 1) {
      var re = 0.0;
      var im = 0.0;
      final cos = basis.cos;
      final sin = basis.sin;
      final base = k * cfg.nFft;
      for (var n = 0; n < cfg.nFft; n += 1) {
        re += frame[n] * cos[base + n];
        im -= frame[n] * sin[base + n]; // STFT uses e^{-j2πkn/N}
      }
      mag2[k * nFrames + t] = re * re + im * im;
    }
  }

  // 5. Drop last frame for whisper parity.
  var effFrames = nFrames;
  if (cfg.dropLastFrame) {
    effFrames = nFrames - 1;
    if (effFrames < 1) {
      throw ArgumentError('drop-last-frame leaves no frames');
    }
  }

  // 6. Magnitude vs power.
  // mag2 currently holds power. For matcha (power=1.0): take sqrt
  // of (power + 1e-9). Whisper (power=2.0): keep as-is.
  final magOrPow = Float32List(nFreq * effFrames);
  if (cfg.power == 1.0) {
    for (var k = 0; k < nFreq; k += 1) {
      for (var t = 0; t < effFrames; t += 1) {
        magOrPow[k * effFrames + t] =
            math.sqrt(mag2[k * nFrames + t] + 1e-9);
      }
    }
  } else {
    for (var k = 0; k < nFreq; k += 1) {
      for (var t = 0; t < effFrames; t += 1) {
        magOrPow[k * effFrames + t] = mag2[k * nFrames + t];
      }
    }
  }

  // 7. Mel projection: (numMels, nFreq) @ (nFreq, effFrames).
  final filters = _MelFilterbank.cached(cfg).weights;
  final out = Float32List(cfg.numMels * effFrames);
  for (var m = 0; m < cfg.numMels; m += 1) {
    for (var t = 0; t < effFrames; t += 1) {
      var s = 0.0;
      for (var k = 0; k < nFreq; k += 1) {
        s += filters[m * nFreq + k] * magOrPow[k * effFrames + t];
      }
      out[m * effFrames + t] = s;
    }
  }

  // 8. Log compression.
  switch (cfg.logMode) {
    case LogMode.matcha:
      for (var i = 0; i < out.length; i += 1) {
        final v = out[i] < 1e-5 ? 1e-5 : out[i];
        out[i] = math.log(v);
      }
    case LogMode.whisper:
      // log10 + global max-clamp + normalise.
      var maxLog = -1e30;
      for (var i = 0; i < out.length; i += 1) {
        final v = out[i] < 1e-10 ? 1e-10 : out[i];
        final l = math.log(v) / math.ln10;
        out[i] = l;
        if (l > maxLog) maxLog = l;
      }
      final floor = maxLog - 8.0;
      for (var i = 0; i < out.length; i += 1) {
        var v = out[i];
        if (v < floor) v = floor;
        out[i] = (v + 4.0) / 4.0;
      }
  }

  return (data: out, numMels: cfg.numMels, nFrames: effFrames);
}

// --- internals ----------------------------------------------------------

Float32List _reflectPad(Float32List x, int pad) {
  if (pad == 0) return x;
  if (x.length <= pad) {
    throw ArgumentError(
        'reflect-pad of $pad requires audio length > $pad (got ${x.length})');
  }
  final out = Float32List(x.length + 2 * pad);
  // Left: mirror around index 0 (exclusive), i.e. x[1], x[2], ...
  for (var i = 0; i < pad; i += 1) {
    out[pad - 1 - i] = x[i + 1];
  }
  // Body
  for (var i = 0; i < x.length; i += 1) {
    out[pad + i] = x[i];
  }
  // Right: mirror around the last index (exclusive).
  final last = x.length - 1;
  for (var i = 0; i < pad; i += 1) {
    out[pad + x.length + i] = x[last - 1 - i];
  }
  return out;
}

Float32List _hannWindow(int n) {
  // torch.hann_window default: periodic=True, so denominator is n,
  // not (n-1). This matches both matcha and whisper.
  final w = Float32List(n);
  for (var i = 0; i < n; i += 1) {
    w[i] = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / n);
  }
  return w;
}

class _DftBasis {
  _DftBasis._(this.cos, this.sin);
  final Float64List cos; // shape (n_freq, n_fft) row-major
  final Float64List sin;

  static final _cache = <int, _DftBasis>{};
  static _DftBasis cached(int nFft) {
    final hit = _cache[nFft];
    if (hit != null) return hit;
    final nFreq = nFft ~/ 2 + 1;
    final cos = Float64List(nFreq * nFft);
    final sin = Float64List(nFreq * nFft);
    for (var k = 0; k < nFreq; k += 1) {
      for (var n = 0; n < nFft; n += 1) {
        final a = 2.0 * math.pi * k * n / nFft;
        cos[k * nFft + n] = math.cos(a);
        sin[k * nFft + n] = math.sin(a);
      }
    }
    final v = _DftBasis._(cos, sin);
    _cache[nFft] = v;
    return v;
  }
}

class _MelFilterbank {
  _MelFilterbank._(this.weights);

  /// (numMels, nFreq) row-major.
  final Float32List weights;

  static final _cache = <String, _MelFilterbank>{};
  static _MelFilterbank cached(MelConfig cfg) {
    final key = '${cfg.sampleRate}_${cfg.nFft}_${cfg.numMels}_'
        '${cfg.fmin}_${cfg.fmax}';
    final hit = _cache[key];
    if (hit != null) return hit;
    final w = _buildMelFilters(
      sampleRate: cfg.sampleRate,
      nFft: cfg.nFft,
      numMels: cfg.numMels,
      fmin: cfg.fmin,
      fmax: cfg.fmax,
    );
    final v = _MelFilterbank._(w);
    _cache[key] = v;
    return v;
  }
}

/// librosa.filters.mel with htk=False, norm='slaney'. Verified against
/// librosa-generated reference fixtures (see test).
Float32List _buildMelFilters({
  required int sampleRate,
  required int nFft,
  required int numMels,
  required double fmin,
  required double fmax,
}) {
  final nFreq = nFft ~/ 2 + 1;

  // FFT bin frequencies (Hz).
  final fftFreqs = Float64List(nFreq);
  for (var i = 0; i < nFreq; i += 1) {
    fftFreqs[i] = i * sampleRate / nFft;
  }

  // Mel band edge frequencies, including 2 boundary points.
  final melMin = _hzToMelSlaney(fmin);
  final melMax = _hzToMelSlaney(fmax);
  final melPoints = Float64List(numMels + 2);
  for (var i = 0; i < melPoints.length; i += 1) {
    melPoints[i] = melMin + (melMax - melMin) * i / (numMels + 1);
  }
  final hzPoints = Float64List(numMels + 2);
  for (var i = 0; i < melPoints.length; i += 1) {
    hzPoints[i] = _melToHzSlaney(melPoints[i]);
  }

  // Triangular weights.
  final weights = Float32List(numMels * nFreq);
  for (var m = 0; m < numMels; m += 1) {
    final lower = hzPoints[m];
    final center = hzPoints[m + 1];
    final upper = hzPoints[m + 2];
    final lDen = center - lower;
    final rDen = upper - center;
    for (var k = 0; k < nFreq; k += 1) {
      final f = fftFreqs[k];
      double v;
      if (f <= lower || f >= upper) {
        v = 0.0;
      } else if (f <= center) {
        v = (f - lower) / lDen;
      } else {
        v = (upper - f) / rDen;
      }
      weights[m * nFreq + k] = v;
    }
    // Slaney normalisation: divide by (upper - lower) / 2.
    final norm = 2.0 / (upper - lower);
    for (var k = 0; k < nFreq; k += 1) {
      weights[m * nFreq + k] *= norm;
    }
  }
  return weights;
}

// librosa's slaney-style mel scale (htk=False).
double _hzToMelSlaney(double hz) {
  const fMin = 0.0;
  const fSp = 200.0 / 3.0;
  var mel = (hz - fMin) / fSp;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  final logstep = math.log(6.4) / 27.0;
  if (hz >= minLogHz) {
    mel = minLogMel + math.log(hz / minLogHz) / logstep;
  }
  return mel;
}

double _melToHzSlaney(double mel) {
  const fMin = 0.0;
  const fSp = 200.0 / 3.0;
  var hz = fMin + fSp * mel;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  final logstep = math.log(6.4) / 27.0;
  if (mel >= minLogMel) {
    hz = minLogHz * math.exp(logstep * (mel - minLogMel));
  }
  return hz;
}
