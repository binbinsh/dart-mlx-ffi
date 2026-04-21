/// SincNet frontend for pyannote/segmentation-3.0 (MLX).
///
/// Architecture reference:
///   pyannote.audio.models.blocks.sincnet.SincNet
///   asteroid_filterbanks.param_sinc_fb.ParamSincFB  (for conv1d.0 filter math)
///
/// Pipeline (matches PyTorch reference exactly):
///
///   waveform (B, 1, T)
///     → InstanceNorm1d (wav_norm1d, affine=True)
///     → Conv1d (sinc filters, 80, k=251, stride=10, no pad)
///     → abs()                                              (Ravanelli issue #4)
///     → MaxPool1d(k=3, s=3)
///     → InstanceNorm1d (80, affine=True)
///     → leaky_relu(0.01)
///     → Conv1d(60, k=5, stride=1, no pad)
///     → MaxPool1d(k=3, s=3)
///     → InstanceNorm1d (60, affine=True)
///     → leaky_relu(0.01)
///     → Conv1d(60, k=5, stride=1, no pad)
///     → MaxPool1d(k=3, s=3)
///     → InstanceNorm1d (60, affine=True)
///     → leaky_relu(0.01)
///   ─ output (B, 60, 589) for 10 s @ 16 kHz.
///
/// The 80 sinc filters are recomputed from the 4 learnable params
/// (`low_hz_`, `band_hz_`, `n_`, `window_`) at construction time and cached
/// as a `(80, 1, 251)` tensor — they are deterministic for a fixed bundle.
library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';

/// Pyannote 3.0 uses the defaults from `ParamSincFB`: `min_low_hz=50,
/// min_band_hz=50`.
const double _minLowHz = 50.0;
const double _minBandHz = 50.0;

/// Leaky ReLU slope used by pyannote (PyTorch default, not kitten_tts 0.2).
const double _leakySlope = 0.01;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Leaky ReLU with `negative_slope = 0.01` (PyTorch default).
MlxArray _leakyRelu(MlxArray x) {
  final zero = MlxArray.full(const <int>[], 0.0);
  final slope = MlxArray.full(const <int>[], _leakySlope);
  try {
    final mask = MlxMore.greater(x, zero);
    final scaled = mx.multiply(x, slope);
    try {
      return mx.where(mask, x, scaled);
    } finally {
      mask.close();
      scaled.close();
    }
  } finally {
    slope.close();
    zero.close();
  }
}

/// Flip an MlxArray along `axis` using `take` with reversed indices.
///
/// Safe alternative to slice-with-negative-stride (which crashes on MLX 0.31).
MlxArray _flip(MlxArray input, int axis) {
  final n = input.shape[axis];
  final idx = MlxArray.fromInt32List(
    List<int>.generate(n, (i) => n - 1 - i),
    shape: <int>[n],
  );
  try {
    return input.take(idx, axis: axis);
  } finally {
    idx.close();
  }
}

/// PyTorch MaxPool1d(kernel=3, stride=3, padding=0) over the time axis of a
/// `(B, C, T)` tensor.
///
/// Implementation: trim T to a multiple of 3, reshape to `(B, C, T/3, 3)`,
/// then reduce via pair-wise `maximum`.
MlxArray _maxPool3s3(MlxArray input) {
  final b = input.shape[0];
  final c = input.shape[1];
  final t = input.shape[2];
  final tTrim = t - (t % 3);
  MlxArray trimmed;
  if (tTrim == t) {
    trimmed = input;
  } else {
    trimmed = input.slice(
      start: <int>[0, 0, 0],
      stop: <int>[b, c, tTrim],
    );
  }
  final reshaped = trimmed.reshape(<int>[b, c, tTrim ~/ 3, 3]);
  if (!identical(trimmed, input)) {
    trimmed.close();
  }
  // Split last axis into 3 singletons.
  final parts = MlxTensor.splitSections(reshaped, <int>[1, 2], axis: 3);
  reshaped.close();
  try {
    final p0 = parts[0].reshape(<int>[b, c, tTrim ~/ 3]);
    final p1 = parts[1].reshape(<int>[b, c, tTrim ~/ 3]);
    final p2 = parts[2].reshape(<int>[b, c, tTrim ~/ 3]);
    try {
      final m01 = mx.maximum(p0, p1);
      try {
        return mx.maximum(m01, p2);
      } finally {
        m01.close();
      }
    } finally {
      p0.close();
      p1.close();
      p2.close();
    }
  } finally {
    for (final part in parts) {
      part.close();
    }
  }
}

/// PyTorch InstanceNorm1d with `affine=True`, `eps=1e-5`, over a `(B, C, T)`
/// tensor. Normalises each `(b, c, :)` slice independently.
///
/// y = gamma * (x - mean_t) / sqrt(var_t + eps) + beta
MlxArray _instanceNorm1d(
  MlxArray input, {
  required MlxArray weight,
  required MlxArray bias,
  double eps = 1e-5,
}) {
  final c = input.shape[1];
  final mean = input.mean(axis: 2, keepDims: true); // (B, C, 1)
  final centred = mx.subtract(input, mean);
  mean.close();
  final sq = mx.multiply(centred, centred);
  final variance = sq.mean(axis: 2, keepDims: true);
  sq.close();
  final epsArr = MlxArray.full(const <int>[], eps);
  final varPlusEps = mx.add(variance, epsArr);
  variance.close();
  epsArr.close();
  final invStd = MlxMore.rsqrt(varPlusEps);
  varPlusEps.close();
  final normed = mx.multiply(centred, invStd);
  centred.close();
  invStd.close();
  // weight/bias are 1D (C,) → broadcast as (1, C, 1).
  final w3 = weight.reshape(<int>[1, c, 1]);
  final b3 = bias.reshape(<int>[1, c, 1]);
  try {
    final scaled = mx.multiply(normed, w3);
    normed.close();
    final shifted = mx.add(scaled, b3);
    scaled.close();
    return shifted;
  } finally {
    w3.close();
    b3.close();
  }
}

/// Plain Conv1d on `(B, C_in, T)` with `stride`, no padding.
///
/// `weight` is stored in PyTorch layout `(C_out, C_in, kW)`; converted to
/// MLX layout `(C_out, kW, C_in)` here.  `bias` may be `null` (sinc conv).
MlxArray _plainConv1d(
  MlxArray input, {
  required MlxArray weight,
  MlxArray? bias,
  required int stride,
}) {
  final nct = input.transposeAxes(<int>[0, 2, 1]); // (B, T, C_in)
  final w = weight.transposeAxes(<int>[0, 2, 1]); // (C_out, kW, C_in)
  try {
    final conv = mx.conv1d(nct, w, stride: stride, padding: 0);
    nct.close();
    if (bias == null) {
      return conv.transposeAxes(<int>[0, 2, 1]);
    }
    final cOut = bias.shape[0];
    final b3 = bias.reshape(<int>[1, 1, cOut]);
    try {
      final added = mx.add(conv, b3);
      conv.close();
      return added.transposeAxes(<int>[0, 2, 1]); // (B, C_out, T_out)
    } finally {
      b3.close();
    }
  } finally {
    w.close();
  }
}

// ---------------------------------------------------------------------------
// Sinc filterbank construction (from stored parameters)
// ---------------------------------------------------------------------------

/// Build the 80 sinc band-pass filters in MLX layout `(80, 1, 251)` from the
/// four learnable params stored in the bundle.
///
/// Mirrors `asteroid_filterbanks.ParamSincFB.filters()` exactly:
///   * 40 cosine (even) filters followed by 40 sine (odd) filters.
MlxArray _buildSincFilters({
  required MlxArray lowHz, // (40, 1)
  required MlxArray bandHz, // (40, 1)
  required MlxArray n, // (1, 125)
  required MlxArray window, // (125,)
  required int kernelSize, // 251
  required double sampleRate,
}) {
  // low = min_low_hz + |low_hz_|
  final absLow = mx.abs(lowHz);
  final minLow = MlxArray.full(const <int>[], _minLowHz);
  final minBand = MlxArray.full(const <int>[], _minBandHz);
  final low = mx.add(absLow, minLow);
  absLow.close();

  // high = clamp(low + min_band_hz + |band_hz_|, min_low_hz, sr/2)
  // Use `maximum`/`minimum` with float32 scalar arrays to avoid the float64
  // promotion that `mx.clip` applies when given double bounds.
  final absBand = mx.abs(bandHz);
  final lowPlusBand = mx.add(low, minBand);
  final rawHigh = mx.add(lowPlusBand, absBand);
  lowPlusBand.close();
  absBand.close();
  final nyquistArr = MlxArray.full(const <int>[], sampleRate / 2.0);
  final lowerBound = MlxArray.full(const <int>[], _minLowHz);
  final highLower = mx.maximum(rawHigh, lowerBound);
  final high = mx.minimum(highLower, nyquistArr);
  highLower.close();
  nyquistArr.close();
  lowerBound.close();
  rawHigh.close();
  minLow.close();
  minBand.close();

  // band = (high - low).squeeze(-1)  → (40,)
  final bandColumn = mx.subtract(high, low); // (40, 1)
  final band = bandColumn.reshape(<int>[bandColumn.shape[0]]);
  bandColumn.close();

  // ft_low = low @ n  → (40, 125)
  final ftLow = mx.matmul(low, n);
  final ftHigh = mx.matmul(high, n);
  low.close();
  high.close();

  final halfKernel = kernelSize ~/ 2; // 125

  // bp_center_cos = 2 * band  → reshape to (40, 1)
  final two = MlxArray.full(const <int>[], 2.0);
  final twoBand = mx.multiply(band, two);
  final bpCenterCos = twoBand.reshape(<int>[band.shape[0], 1]);
  twoBand.close();

  // bp_left_cos = (sin(ft_high) - sin(ft_low)) / (n/2) * window
  final sinHigh = mx.sin(ftHigh);
  final sinLow = mx.sin(ftLow);
  final sinDiff = mx.subtract(sinHigh, sinLow);
  sinHigh.close();
  sinLow.close();
  final halfN = mx.divide(n, two); // (1, 125)
  final divided = mx.divide(sinDiff, halfN); // broadcasts (40,125)/(1,125)
  sinDiff.close();
  // window is (125,) → broadcast as (1,125) for multiply against (40,125)
  final window2 = window.reshape(<int>[1, halfKernel]);
  final bpLeftCos = mx.multiply(divided, window2);
  divided.close();

  // bp_left_sin = (cos(ft_low) - cos(ft_high)) / (n/2) * window
  final cosLow = mx.cos(ftLow);
  final cosHigh = mx.cos(ftHigh);
  final cosDiff = mx.subtract(cosLow, cosHigh);
  cosLow.close();
  cosHigh.close();
  ftLow.close();
  ftHigh.close();
  final cosDivided = mx.divide(cosDiff, halfN);
  cosDiff.close();
  halfN.close();
  final bpLeftSin = mx.multiply(cosDivided, window2);
  cosDivided.close();
  window2.close();
  two.close();

  // bp_center_sin = zeros(40, 1)
  final bpCenterSin = MlxArray.zeros(<int>[band.shape[0], 1]);

  // bp_right_cos = flip(bp_left_cos, dim=1)
  final bpRightCos = _flip(bpLeftCos, 1);
  // bp_right_sin = -flip(bp_left_sin, dim=1)
  final bpRightSinRaw = _flip(bpLeftSin, 1);
  final minusOne = MlxArray.full(const <int>[], -1.0);
  final bpRightSin = mx.multiply(bpRightSinRaw, minusOne);
  bpRightSinRaw.close();
  minusOne.close();

  // band_pass = cat([bp_left, bp_center, bp_right], dim=1)  → (40, 251)
  final cosBand = mx.concatenate(
    <MlxArray>[bpLeftCos, bpCenterCos, bpRightCos],
    axis: 1,
  );
  final sinBand = mx.concatenate(
    <MlxArray>[bpLeftSin, bpCenterSin, bpRightSin],
    axis: 1,
  );
  bpLeftCos.close();
  bpCenterCos.close();
  bpRightCos.close();
  bpLeftSin.close();
  bpCenterSin.close();
  bpRightSin.close();

  // band_pass = band_pass / (2 * band[:, None])
  final bandReshaped = band.reshape(<int>[band.shape[0], 1]); // (40, 1)
  final twoS = MlxArray.full(const <int>[], 2.0);
  final twoBandCol = mx.multiply(bandReshaped, twoS);
  bandReshaped.close();
  twoS.close();
  band.close();

  final cosFilters = mx.divide(cosBand, twoBandCol); // (40, 251)
  final sinFilters = mx.divide(sinBand, twoBandCol);
  cosBand.close();
  sinBand.close();
  twoBandCol.close();

  // view as (40, 1, 251) each
  final cos3 = cosFilters.reshape(<int>[cosFilters.shape[0], 1, kernelSize]);
  final sin3 = sinFilters.reshape(<int>[sinFilters.shape[0], 1, kernelSize]);
  cosFilters.close();
  sinFilters.close();

  // filters = cat([cos, sin], dim=0)  → (80, 1, 251)
  final filters = mx.concatenate(<MlxArray>[cos3, sin3], axis: 0);
  cos3.close();
  sin3.close();
  return filters;
}

// ---------------------------------------------------------------------------
// Public SincNet module
// ---------------------------------------------------------------------------

/// MLX SincNet frontend for pyannote/segmentation-3.0.
///
/// Construct via [PyannoteSincNet.fromBundle]; call [encode] on a 10 s mono
/// waveform `(1, T)` to obtain the `(B, 60, 589)` embedding.
///
/// The module caches the 80 sinc filters as a precomputed `(80, 1, 251)`
/// tensor at construction time; inference then runs three fused stages of
/// Conv1d / MaxPool1d / InstanceNorm1d / leaky_relu.
final class PyannoteSincNet {
  PyannoteSincNet._({
    required this.manifest,
    required this.wavNormWeight,
    required this.wavNormBias,
    required MlxArray sincFilters,
    required this.conv1Weight,
    required this.conv1Bias,
    required this.conv2Weight,
    required this.conv2Bias,
    required this.norm0Weight,
    required this.norm0Bias,
    required this.norm1Weight,
    required this.norm1Bias,
    required this.norm2Weight,
    required this.norm2Bias,
  }) : _sincFilters = sincFilters;

  final PyannoteSegManifest manifest;

  /// `wav_norm1d` affine params, shape `(1,)`.
  final MlxArray wavNormWeight;
  final MlxArray wavNormBias;

  /// 80 sinc band-pass filters in MLX conv1d layout `(80, 1, 251)`.
  final MlxArray _sincFilters;

  /// `sincnet.conv1d.1.*` (PyTorch (60, 80, 5) / (60,))
  final MlxArray conv1Weight;
  final MlxArray conv1Bias;

  /// `sincnet.conv1d.2.*` (PyTorch (60, 60, 5) / (60,))
  final MlxArray conv2Weight;
  final MlxArray conv2Bias;

  /// `sincnet.norm1d.0.*` (80,)
  final MlxArray norm0Weight;
  final MlxArray norm0Bias;
  final MlxArray norm1Weight;
  final MlxArray norm1Bias;
  final MlxArray norm2Weight;
  final MlxArray norm2Bias;

  /// Build the frontend from a loaded [PyannoteSegBundle].
  factory PyannoteSincNet.fromBundle(PyannoteSegBundle bundle) {
    MlxArray must(String key) {
      final v = bundle.tensors[key];
      if (v == null) {
        throw StateError('Missing pyannote-seg tensor: $key');
      }
      return v;
    }

    final lowHz = must('sincnet.conv1d.0.filterbank.low_hz_');
    final bandHz = must('sincnet.conv1d.0.filterbank.band_hz_');
    final n = must('sincnet.conv1d.0.filterbank.n_');
    final window = must('sincnet.conv1d.0.filterbank.window_');

    final sincFilters = _buildSincFilters(
      lowHz: lowHz,
      bandHz: bandHz,
      n: n,
      window: window,
      kernelSize: bundle.manifest.sincnet.kernelSize,
      sampleRate: bundle.manifest.sincnet.sampleRate.toDouble(),
    );

    // Sanity check: ensure filter shape is (80, 1, 251).
    final expectedOut = bundle.manifest.sincnet.nFilters[0];
    if (sincFilters.shape[0] != expectedOut ||
        sincFilters.shape[2] != bundle.manifest.sincnet.kernelSize) {
      throw StateError(
        'SincNet filter shape mismatch: ${sincFilters.shape} '
        '(expected [$expectedOut, 1, ${bundle.manifest.sincnet.kernelSize}])',
      );
    }

    return PyannoteSincNet._(
      manifest: bundle.manifest,
      wavNormWeight: must('sincnet.wav_norm1d.weight'),
      wavNormBias: must('sincnet.wav_norm1d.bias'),
      sincFilters: sincFilters,
      conv1Weight: must('sincnet.conv1d.1.weight'),
      conv1Bias: must('sincnet.conv1d.1.bias'),
      conv2Weight: must('sincnet.conv1d.2.weight'),
      conv2Bias: must('sincnet.conv1d.2.bias'),
      norm0Weight: must('sincnet.norm1d.0.weight'),
      norm0Bias: must('sincnet.norm1d.0.bias'),
      norm1Weight: must('sincnet.norm1d.1.weight'),
      norm1Bias: must('sincnet.norm1d.1.bias'),
      norm2Weight: must('sincnet.norm1d.2.weight'),
      norm2Bias: must('sincnet.norm1d.2.bias'),
    );
  }

  /// Free the cached sinc filterbank. Other tensors are owned by the bundle.
  void close() {
    _sincFilters.close();
  }

  /// Forward a mono waveform through the SincNet frontend.
  ///
  /// Accepted input ranks:
  ///   * `(samples,)` → treated as batch=1, channel=1.
  ///   * `(B, samples)` → treated as channel=1.
  ///   * `(B, 1, samples)` → already channel-first.
  ///
  /// Returns `(B, 60, T_out)` where `T_out` matches the PyTorch reference
  /// (589 for 10 s input @ 16 kHz).
  MlxArray encode(MlxArray waveform) {
    MlxArray x;
    if (waveform.ndim == 1) {
      x = waveform.reshape(<int>[1, 1, waveform.shape[0]]);
    } else if (waveform.ndim == 2) {
      x = waveform.reshape(<int>[waveform.shape[0], 1, waveform.shape[1]]);
    } else if (waveform.ndim == 3) {
      x = waveform;
    } else {
      throw StateError(
        'Unexpected waveform rank: ${waveform.ndim} (want 1/2/3)',
      );
    }

    final firstStage = _forwardSincStage(x);
    if (!identical(x, waveform)) {
      x.close();
    }

    final stage1 = _forwardPlainStage(
      firstStage,
      weight: conv1Weight,
      bias: conv1Bias,
      normWeight: norm1Weight,
      normBias: norm1Bias,
    );
    firstStage.close();

    final stage2 = _forwardPlainStage(
      stage1,
      weight: conv2Weight,
      bias: conv2Bias,
      normWeight: norm2Weight,
      normBias: norm2Bias,
    );
    stage1.close();

    return stage2;
  }

  /// First SincNet stage:
  ///   wav_norm1d → sincConv → abs → maxpool(3,3) → norm0 → leaky_relu.
  MlxArray _forwardSincStage(MlxArray waveform) {
    final normed = _instanceNorm1d(
      waveform,
      weight: wavNormWeight,
      bias: wavNormBias,
    );
    final sincConv = _plainConv1d(
      normed,
      weight: _sincFilters,
      bias: null,
      stride: manifest.sincnet.stride,
    );
    normed.close();
    final absed = mx.abs(sincConv);
    sincConv.close();
    final pooled = _maxPool3s3(absed);
    absed.close();
    final norm0 = _instanceNorm1d(
      pooled,
      weight: norm0Weight,
      bias: norm0Bias,
    );
    pooled.close();
    final activated = _leakyRelu(norm0);
    norm0.close();
    return activated;
  }

  /// Standard Conv1d + pool + norm + leaky_relu stage.
  MlxArray _forwardPlainStage(
    MlxArray input, {
    required MlxArray weight,
    required MlxArray bias,
    required MlxArray normWeight,
    required MlxArray normBias,
  }) {
    final conv = _plainConv1d(
      input,
      weight: weight,
      bias: bias,
      stride: 1,
    );
    final pooled = _maxPool3s3(conv);
    conv.close();
    final normed = _instanceNorm1d(
      pooled,
      weight: normWeight,
      bias: normBias,
    );
    pooled.close();
    final activated = _leakyRelu(normed);
    normed.close();
    return activated;
  }
}

/// Expose the sinc filterbank builder for parity tests.
MlxArray debugBuildSincFilters({
  required MlxArray lowHz,
  required MlxArray bandHz,
  required MlxArray n,
  required MlxArray window,
  required int kernelSize,
  required double sampleRate,
}) => _buildSincFilters(
      lowHz: lowHz,
      bandHz: bandHz,
      n: n,
      window: window,
      kernelSize: kernelSize,
      sampleRate: sampleRate,
    );
