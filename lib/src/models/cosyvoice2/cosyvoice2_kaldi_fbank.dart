// Kaldi-compatible filterbank features (subset) for CosyVoice2's
// campplus speaker encoder.
//
// Reproduces `torchaudio.compliance.kaldi.fbank(num_mel_bins=80,
// dither=0, sample_frequency=16000)` for the upstream call site in
// CosyVoice2's frontend. Other defaults are taken as-is from kaldi:
//
//   * frame_length = 25.0 ms (= 400 samples @ 16 kHz)
//   * frame_shift  = 10.0 ms (= 160 samples @ 16 kHz)
//   * preemphasis_coefficient = 0.97
//   * remove_dc_offset = true
//   * window_type = 'povey'
//   * round_to_power_of_two = true (=> n_fft = 512 for 400 samples)
//   * low_freq = 20.0, high_freq = sr/2
//   * use_power = true, use_log_fbank = true
//   * snip_edges = true, htk_compat = false
//   * energy_floor = float32 eps (1.1920929e-7) — matches the
//     upstream CosyVoice2 kaldi.fbank call which uses the float32
//     epsilon as a pre-log power floor so log(0) becomes finite.
//
// Like the matcha/whisper extractors this lives entirely in pure
// Dart (no FFI) — kaldi fbank only runs once per voice prompt to
// drive the campplus speaker encoder, so the matmul-DFT cost is fine.

import 'dart:math' as math;
import 'dart:typed_data';

class KaldiFbankConfig {
  const KaldiFbankConfig({
    this.sampleRate = 16000,
    this.numMelBins = 80,
    this.frameLengthMs = 25.0,
    this.frameShiftMs = 10.0,
    this.preemphasis = 0.97,
    this.removeDcOffset = true,
    this.lowFreq = 20.0,
    this.highFreq = 0.0, // 0 means sr/2
    this.energyFloor = 1.1920929e-7,
  });
  final int sampleRate;
  final int numMelBins;
  final double frameLengthMs;
  final double frameShiftMs;
  final double preemphasis;
  final bool removeDcOffset;
  final double lowFreq;
  final double highFreq;
  final double energyFloor;
}

/// Output shape (nFrames, numMelBins) row-major.
({Float32List data, int nFrames, int numMelBins}) computeKaldiFbank(
  Float32List audio,
  KaldiFbankConfig cfg,
) {
  final frameLen = (cfg.sampleRate * cfg.frameLengthMs / 1000.0).round();
  final frameShift = (cfg.sampleRate * cfg.frameShiftMs / 1000.0).round();
  final nFft = _nextPow2(frameLen);
  final nFreq = nFft ~/ 2 + 1;
  final highFreq = cfg.highFreq <= 0.0 ? cfg.sampleRate / 2.0 : cfg.highFreq;

  // snip_edges = true: number of frames is floor((N - frameLen) / hop) + 1
  if (audio.length < frameLen) {
    return (data: Float32List(0), nFrames: 0, numMelBins: cfg.numMelBins);
  }
  final nFrames = ((audio.length - frameLen) ~/ frameShift) + 1;

  final window = _poveyWindow(frameLen);
  final basis = _DftBasisCache.cached(nFft);
  final filters = _MelFilterbankCache.cached(
    sampleRate: cfg.sampleRate,
    nFft: nFft,
    numMels: cfg.numMelBins,
    lowFreq: cfg.lowFreq,
    highFreq: highFreq,
  );

  final out = Float32List(nFrames * cfg.numMelBins);
  final buf = Float64List(
    nFft,
  ); // zero-padded frame, f64 for accumulation parity

  for (var t = 0; t < nFrames; t += 1) {
    final start = t * frameShift;

    // 1. Copy raw frame.
    for (var i = 0; i < frameLen; i += 1) {
      buf[i] = audio[start + i];
    }
    for (var i = frameLen; i < nFft; i += 1) {
      buf[i] = 0.0;
    }

    // 2. Remove DC offset.
    if (cfg.removeDcOffset) {
      var mean = 0.0;
      for (var i = 0; i < frameLen; i += 1) {
        mean += buf[i];
      }
      mean /= frameLen;
      for (var i = 0; i < frameLen; i += 1) {
        buf[i] -= mean;
      }
    }

    // 3. Pre-emphasis: y[n] = x[n] - a*x[n-1], with x[-1] := x[0]
    //    (kaldi's convention: replicate first sample).
    if (cfg.preemphasis != 0.0) {
      final a = cfg.preemphasis;
      var prev = buf[0];
      for (var i = 0; i < frameLen; i += 1) {
        final cur = buf[i];
        buf[i] = cur - a * prev;
        prev = cur;
      }
    }

    // 4. Apply povey window.
    for (var i = 0; i < frameLen; i += 1) {
      buf[i] *= window[i];
    }

    // 5. DFT (matmul). Compute power = re^2 + im^2 into a separate
    //    buffer (we cannot reuse `buf[k]` because the inner DFT loop
    //    keeps reading `buf[0..nFft)` for every k).
    final cosB = basis.cos;
    final sinB = basis.sin;
    final power = Float64List(nFreq);
    for (var k = 0; k < nFreq; k += 1) {
      var re = 0.0, im = 0.0;
      final base = k * nFft;
      for (var n = 0; n < nFft; n += 1) {
        re += buf[n] * cosB[base + n];
        im -= buf[n] * sinB[base + n];
      }
      power[k] = re * re + im * im;
    }
    // 6. Mel projection.
    final wts = filters.weights;
    for (var m = 0; m < cfg.numMelBins; m += 1) {
      var s = 0.0;
      for (var k = 0; k < nFreq; k += 1) {
        s += wts[m * nFreq + k] * power[k];
      }
      // 7. Log + energy floor (matches torchaudio: max(s, mel_floor)
      //    BEFORE log, not after).
      final clamped = s > cfg.energyFloor ? s : cfg.energyFloor;
      out[t * cfg.numMelBins + m] = math.log(clamped);
    }
  }
  return (data: out, nFrames: nFrames, numMelBins: cfg.numMelBins);
}

/// Convenience: subtract per-utterance mean (CMN) along time, in-place.
void cepstralMeanNormalize(Float32List feat, int nFrames, int numMelBins) {
  if (nFrames == 0) return;
  final mean = Float64List(numMelBins);
  for (var t = 0; t < nFrames; t += 1) {
    for (var b = 0; b < numMelBins; b += 1) {
      mean[b] += feat[t * numMelBins + b];
    }
  }
  for (var b = 0; b < numMelBins; b += 1) {
    mean[b] /= nFrames;
  }
  for (var t = 0; t < nFrames; t += 1) {
    for (var b = 0; b < numMelBins; b += 1) {
      feat[t * numMelBins + b] -= mean[b];
    }
  }
}

// --- internals ---------------------------------------------------------

int _nextPow2(int n) {
  var p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

Float64List _poveyWindow(int n) {
  // Povey window: w[i] = (0.5 - 0.5 cos(2pi i/(n-1)))^0.85
  final w = Float64List(n);
  for (var i = 0; i < n; i += 1) {
    final base = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1));
    w[i] = math.pow(base, 0.85).toDouble();
  }
  return w;
}

class _DftBasisCache {
  _DftBasisCache._(this.cos, this.sin);
  final Float64List cos;
  final Float64List sin;
  static final _cache = <int, _DftBasisCache>{};
  static _DftBasisCache cached(int nFft) {
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
    final v = _DftBasisCache._(cos, sin);
    _cache[nFft] = v;
    return v;
  }
}

class _MelFilterbankCache {
  _MelFilterbankCache._(this.weights);
  final Float32List weights;
  static final _cache = <String, _MelFilterbankCache>{};
  static _MelFilterbankCache cached({
    required int sampleRate,
    required int nFft,
    required int numMels,
    required double lowFreq,
    required double highFreq,
  }) {
    final key = '${sampleRate}_${nFft}_${numMels}_${lowFreq}_$highFreq';
    final hit = _cache[key];
    if (hit != null) return hit;
    final w = _buildKaldiMel(
      sampleRate: sampleRate,
      nFft: nFft,
      numMels: numMels,
      lowFreq: lowFreq,
      highFreq: highFreq,
    );
    final v = _MelFilterbankCache._(w);
    _cache[key] = v;
    return v;
  }
}

/// Kaldi-style mel filterbank: htk mel scale, triangular weights,
/// no slaney normalisation.
Float32List _buildKaldiMel({
  required int sampleRate,
  required int nFft,
  required int numMels,
  required double lowFreq,
  required double highFreq,
}) {
  final nFreq = nFft ~/ 2 + 1;
  final fftFreqs = Float64List(nFreq);
  for (var k = 0; k < nFreq; k += 1) {
    fftFreqs[k] = k * sampleRate / nFft;
  }
  final melLow = _hzToMelHtk(lowFreq);
  final melHigh = _hzToMelHtk(highFreq);
  final melPoints = Float64List(numMels + 2);
  for (var i = 0; i < melPoints.length; i += 1) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (numMels + 1);
  }
  final w = Float32List(numMels * nFreq);
  for (var m = 0; m < numMels; m += 1) {
    final left = melPoints[m];
    final center = melPoints[m + 1];
    final right = melPoints[m + 2];
    for (var k = 0; k < nFreq; k += 1) {
      final mel = _hzToMelHtk(fftFreqs[k]);
      double v;
      if (mel <= left || mel >= right) {
        v = 0.0;
      } else if (mel <= center) {
        v = (mel - left) / (center - left);
      } else {
        v = (right - mel) / (right - center);
      }
      w[m * nFreq + k] = v;
    }
  }
  return w;
}

double _hzToMelHtk(double hz) => 1127.0 * math.log(1.0 + hz / 700.0);
