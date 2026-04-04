library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'core.dart';
import 'vocoder_core.dart';

/// Hann window of length [size] as a 1-D MlxArray (float32).
MlxArray _hannWindow(int size) {
  final values = List<double>.generate(size, (i) {
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * i / size));
  });
  return MlxArray.fromFloat32List(values, shape: [size]);
}

/// Forward STFT: time-domain → (magnitude, phase).
///
/// Takes a real-valued signal of shape `[batch, samples]` and returns
/// `(spec, phase)` where:
///   - `spec = |STFT|` of shape `[batch, nfft/2+1, frames]`
///   - `phase = angle(STFT)` of shape `[batch, nfft/2+1, frames]`
///
/// Uses a manual DFT approach since the Dart MLX FFI does not expose
/// complex-number operations (`mx.real`, `mx.imag`, `mx.arctan2`).
/// For small nfft (typically 16 in KittenTTS), the matmul DFT is efficient.
({MlxArray spec, MlxArray phase}) stftTransform(
  MlxArray signal, {
  required int nfft,
  required int hopSize,
}) {
  final freqBins = (nfft ~/ 2) + 1;
  final window = _hannWindow(nfft);
  final padAmount = nfft ~/ 2;

  // Centre-pad the signal: pad by nfft//2 on each side.
  // signal shape: [batch, samples]
  final zeroPad = scalar(0.0);
  final padded = mx.pad(
    signal,
    axes: [1],
    lowPads: [padAmount],
    highPads: [padAmount],
    padValue: zeroPad,
  ); // [batch, paddedLen]
  zeroPad.close();
  final paddedLen = padded.shape[1];

  // Number of frames.
  final numFrames = (paddedLen - nfft) ~/ hopSize + 1;

  // Build frame indices: [numFrames * nfft]
  final frameStarts = List<int>.generate(numFrames, (f) => f * hopSize);
  final indexValues = <int>[];
  for (final start in frameStarts) {
    for (var k = 0; k < nfft; k++) {
      indexValues.add(start + k);
    }
  }
  final indices = MlxArray.fromInt32List(
    indexValues,
    shape: [numFrames * nfft],
  );

  // Gather frames from the padded signal.
  // padded: [batch, paddedLen] → gather along axis 1 → [batch, numFrames*nfft]
  final gathered = padded.take(indices, axis: 1);
  padded.close();
  indices.close();

  // Reshape to [batch, numFrames, nfft]
  final batchSize = signal.shape[0];
  final frames = gathered.reshape([batchSize, numFrames, nfft]);
  gathered.close();

  // Apply Hann window: frames * window[1, 1, nfft]
  final windowBroadcast = window.reshape([1, 1, nfft]);
  final windowed = frames * windowBroadcast; // [batch, numFrames, nfft]
  frames.close();
  window.close();
  windowBroadcast.close();

  // Manual DFT using basis matrices.
  // For freq bin k and time n:
  //   realDFT[k] = sum_n x[n] * cos(2π k n / N)
  //   imagDFT[k] = sum_n x[n] * (-sin(2π k n / N))
  // We only need k in [0, freqBins) for the real-input DFT.
  final cosVals = List<double>.filled(nfft * freqBins, 0.0);
  final sinVals = List<double>.filled(nfft * freqBins, 0.0);
  for (var k = 0; k < freqBins; k++) {
    for (var n = 0; n < nfft; n++) {
      final angle = 2.0 * math.pi * k * n / nfft;
      cosVals[k * nfft + n] = math.cos(angle);
      sinVals[k * nfft + n] = -math.sin(angle); // note: -sin for DFT
    }
  }

  // cosBasis: [freqBins, nfft], sinBasis: [freqBins, nfft]
  final cosBasis = MlxArray.fromFloat32List(cosVals, shape: [freqBins, nfft]);
  final sinBasis = MlxArray.fromFloat32List(sinVals, shape: [freqBins, nfft]);

  // Transpose bases for matmul: [nfft, freqBins]
  final cosT = cosBasis.transpose();
  final sinT = sinBasis.transpose();
  cosBasis.close();
  sinBasis.close();

  // windowed: [batch, numFrames, nfft]
  // realPart = windowed @ cosT → [batch, numFrames, freqBins]
  // imagPart = windowed @ sinT → [batch, numFrames, freqBins]
  final realPart = mx.matmul(windowed, cosT);
  final imagPart = mx.matmul(windowed, sinT);
  windowed.close();
  cosT.close();
  sinT.close();

  // magnitude = sqrt(real^2 + imag^2)
  final realSq = realPart * realPart;
  final imagSq = imagPart * imagPart;
  final magSq = realSq + imagSq;
  realSq.close();
  imagSq.close();
  final mag = magSq.sqrt(); // [batch, numFrames, freqBins]
  magSq.close();

  // phase = atan2(imag, real)
  // Since the Dart MLX FFI has no atan2 op, we evaluate the DFT results
  // on CPU and compute atan2 in Dart. For small nfft (e.g. 16), the
  // data size is batch * numFrames * freqBins which is very manageable.

  // Evaluate to pull data to CPU for atan2 computation.
  mx.evalAll([realPart, imagPart]);

  final realData = realPart.toFloat32List();
  final imagData = imagPart.toFloat32List();
  realPart.close();
  imagPart.close();

  final totalElements = realData.length;
  final phaseData = Float32List(totalElements);
  for (var i = 0; i < totalElements; i++) {
    phaseData[i] = math.atan2(imagData[i], realData[i]);
  }

  final phaseArr = MlxArray.fromFloat32List(
    phaseData,
    shape: [batchSize, numFrames, freqBins],
  );

  // Transpose from [batch, frames, freqBins] → [batch, freqBins, frames]
  final specOut = mx.transposeAxes(mag, [0, 2, 1]);
  final phaseOut = mx.transposeAxes(phaseArr, [0, 2, 1]);
  mag.close();
  phaseArr.close();

  return (spec: specOut, phase: phaseOut);
}

/// Inverse STFT from a [KittenGeneratorProjection].
///
/// The projection contains:
///   - `spec`: magnitude spectrum, shape `[batch, nfft/2+1, frames]`
///   - `phase`: `sin(angle)` of phase, shape `[batch, nfft/2+1, frames]`
///
/// Reconstructs time-domain audio via manual inverse DFT and overlap-add.
/// Returns audio of shape `[batch, numSamples]`.
MlxArray istftFromProjection(
  KittenGeneratorProjection projection, {
  required int nfft,
  required int hopSize,
}) {
  final spec = projection.spec;
  final sinPhase = projection.phase;

  final batch = spec.shape[0];
  final numFrames = spec.shape[2];

  // Reconstruct cos(angle) from sin(angle).
  // cos = sqrt(max(0, 1 - sin^2))
  final sinSq = sinPhase * sinPhase;
  final one = scalar(1.0);
  final diff = one - sinSq;
  one.close();
  sinSq.close();
  final zero = scalar(0.0);
  final clamped = mx.maximum(diff, zero);
  diff.close();
  zero.close();
  final cosPhase = clamped.sqrt();
  clamped.close();

  // Complex spectrum: real = spec * cos, imag = spec * sin
  final realPart = spec * cosPhase;
  final imagPart = spec * sinPhase;
  cosPhase.close();

  // Transpose to [batch, frames, freqBins].
  final realT = mx.transposeAxes(realPart, [0, 2, 1]);
  final imagT = mx.transposeAxes(imagPart, [0, 2, 1]);
  realPart.close();
  imagPart.close();

  return _istftCore(
    realT,
    imagT,
    nfft: nfft,
    hopSize: hopSize,
    batch: batch,
    numFrames: numFrames,
  );
}

/// Core iSTFT: from real and imaginary spectral frames to time-domain audio.
///
/// `realT`, `imagT`: shape `[batch, frames, freqBins]`
/// Returns audio `[batch, numSamples]` with centre-trimming applied.
MlxArray _istftCore(
  MlxArray realT,
  MlxArray imagT, {
  required int nfft,
  required int hopSize,
  required int batch,
  required int numFrames,
}) {
  final freqBins = (nfft ~/ 2) + 1;

  // Build inverse DFT basis matrices.
  // x[n] = (1/N) * sum_k multiplier[k] *
  //        (real[k]*cos(2πkn/N) - imag[k]*sin(2πkn/N))
  // multiplier = 1 for DC and Nyquist, 2 for other bins.
  final cosValues = List<double>.filled(nfft * freqBins, 0.0);
  final sinValues = List<double>.filled(nfft * freqBins, 0.0);
  for (var n = 0; n < nfft; n++) {
    for (var k = 0; k < freqBins; k++) {
      final angle = 2.0 * math.pi * k * n / nfft;
      var multiplier = 2.0;
      if (k == 0 || k == nfft ~/ 2) {
        multiplier = 1.0;
      }
      cosValues[n * freqBins + k] = multiplier * math.cos(angle) / nfft;
      sinValues[n * freqBins + k] = multiplier * math.sin(angle) / nfft;
    }
  }

  final cosBasis = MlxArray.fromFloat32List(cosValues, shape: [nfft, freqBins]);
  final sinBasis = MlxArray.fromFloat32List(sinValues, shape: [nfft, freqBins]);

  // timeFrames = realT @ cosBasis^T - imagT @ sinBasis^T
  // [batch, frames, freqBins] @ [freqBins, nfft] → [batch, frames, nfft]
  final cosT = cosBasis.transpose();
  final sinT = sinBasis.transpose();
  cosBasis.close();
  sinBasis.close();

  final realContrib = mx.matmul(realT, cosT);
  final imagContrib = mx.matmul(imagT, sinT);
  realT.close();
  imagT.close();
  cosT.close();
  sinT.close();

  final timeFrames = realContrib - imagContrib; // [batch, frames, nfft]
  realContrib.close();
  imagContrib.close();

  // Apply Hann window.
  final window = _hannWindow(nfft);
  final windowReshaped = window.reshape([1, 1, nfft]);
  final windowed = timeFrames * windowReshaped; // [batch, frames, nfft]
  timeFrames.close();
  windowReshaped.close();

  // Overlap-add using the fold approach.
  // nfft / hopSize = overlapFactor (e.g. 16/4 = 4).
  final overlapFactor = nfft ~/ hopSize;
  assert(nfft == overlapFactor * hopSize, 'nfft must be divisible by hopSize');

  // Reshape windowed: [batch, frames, overlapFactor, hopSize]
  final reshaped = windowed.reshape([batch, numFrames, overlapFactor, hopSize]);
  windowed.close();

  // For each overlap group g, the contribution is delayed by g frames.
  // Pad with g zero-frames before and (overlapFactor-1-g) after, then sum.
  final totalFrames = numFrames + overlapFactor - 1;
  final zeroFrame = scalar(0.0);

  MlxArray? accumulated;
  for (var g = 0; g < overlapFactor; g++) {
    // Extract group g: [batch, numFrames, hopSize]
    final group = reshaped
        .slice(start: [0, 0, g, 0], stop: [batch, numFrames, g + 1, hopSize])
        .reshape([batch, numFrames, hopSize]);

    // Pad: g frames before, (overlapFactor - 1 - g) frames after
    final padBefore = g;
    final padAfter = overlapFactor - 1 - g;
    final padded = mx.pad(
      group,
      axes: [1],
      lowPads: [padBefore],
      highPads: [padAfter],
      padValue: zeroFrame,
    ); // [batch, totalFrames, hopSize]
    group.close();

    if (accumulated == null) {
      accumulated = padded;
    } else {
      final next = accumulated + padded;
      accumulated.close();
      padded.close();
      accumulated = next;
    }
  }
  reshaped.close();
  zeroFrame.close();

  if (accumulated == null) {
    throw StateError('No frames to overlap-add.');
  }

  // Flatten: [batch, totalFrames * hopSize]
  final rawOutput = accumulated.reshape([batch, totalFrames * hopSize]);
  accumulated.close();

  // COLA (Constant Overlap-Add) normalization.
  // Compute the window-squared sum envelope and divide.
  final windowSumValues = List<double>.filled(totalFrames * hopSize, 0.0);
  final hannValues = List<double>.generate(nfft, (i) {
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * i / nfft));
  });
  for (var f = 0; f < numFrames; f++) {
    for (var n = 0; n < nfft; n++) {
      final pos = f * hopSize + n;
      if (pos < windowSumValues.length) {
        final w = hannValues[n];
        windowSumValues[pos] += w * w; // Hann^2 for COLA
      }
    }
  }
  // Clamp small values to avoid division by zero.
  for (var i = 0; i < windowSumValues.length; i++) {
    if (windowSumValues[i] < 1e-8) {
      windowSumValues[i] = 1.0;
    }
  }
  final windowSum = MlxArray.fromFloat32List(
    windowSumValues,
    shape: [1, totalFrames * hopSize],
  );

  final normalized = rawOutput / windowSum;
  rawOutput.close();
  windowSum.close();
  window.close();

  // Centre-trim: remove nfft//2 from start and end.
  final padTrim = nfft ~/ 2;
  final trimmedLength = totalFrames * hopSize - 2 * padTrim;
  if (trimmedLength <= 0) {
    return normalized;
  }
  final trimmed = normalized.slice(
    start: [0, padTrim],
    stop: [batch, padTrim + trimmedLength],
  );
  normalized.close();
  return trimmed;
}
