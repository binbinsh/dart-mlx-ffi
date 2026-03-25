import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';

final class ParakeetTdtMelFrontend {
  ParakeetTdtMelFrontend(this.manifest)
    : _window = _stftWindow(
        winLength: manifest.winLength,
        nFft: manifest.nFft,
      ),
      _filterbank = _melFilterbank(
        sampleRate: manifest.sampleRate,
        nFft: manifest.nFft,
        nMels: manifest.melFeatures,
      );

  final ParakeetTdtManifest manifest;
  final Float32List _window;
  final Float32List _filterbank;

  int get winLength => manifest.winLength;

  MlxArray compute(Float32List audio) {
    final samples = _preemphasize(audio);
    final padded = _reflectPad(samples, manifest.nFft ~/ 2);
    final frames = _frameAudio(padded);
    final frameCount = frames.length ~/ manifest.nFft;
    final frameArray = MlxArray.fromFloat32List(
      frames,
      shape: <int>[frameCount, manifest.nFft],
    );
    final spectrum = mx.fft.rfft(frameArray, axis: 1);
    frameArray.close();
    final magnitude = mx.abs(spectrum);
    spectrum.close();
    final power = mx.multiply(magnitude, magnitude);
    magnitude.close();

    final bins = (manifest.nFft ~/ 2) + 1;
    final filterbank = MlxArray.fromFloat32List(
      _filterbank,
      shape: <int>[manifest.melFeatures, bins],
    );
    final mel = mx.matmul(filterbank, power.T);
    filterbank.close();
    power.close();
    final logMel = mx.log(mx.add(mel, MlxArray.full([], 1e-5)));
    mel.close();
    final normalized = _normalize(logMel);
    logMel.close();
    final transposed = normalized.T.reshape(
      <int>[1, frameCount, manifest.melFeatures],
    );
    normalized.close();
    return transposed;
  }

  MlxArray _normalize(MlxArray input) {
    if (manifest.normalize == 'per_feature') {
      final mean = input.mean(axis: 1, keepDims: true);
      final variance = input.variance(axis: 1, keepDims: true);
      final std = mx.sqrt(mx.add(variance, MlxArray.full([], 1e-5)));
      variance.close();
      final centered = mx.subtract(input, mean);
      mean.close();
      final normalized = mx.divide(centered, std);
      centered.close();
      std.close();
      return normalized;
    }
    final mean = input.mean();
    final variance = input.variance();
    final std = mx.sqrt(mx.add(variance, MlxArray.full([], 1e-5)));
    variance.close();
    final centered = mx.subtract(input, mean);
    mean.close();
    final normalized = mx.divide(centered, std);
    centered.close();
    std.close();
    return normalized;
  }

  Float32List _preemphasize(Float32List input) {
    final targetLength = manifest.padTo > input.length
        ? manifest.padTo
        : input.length;
    final out = Float32List(targetLength);
    if (input.isEmpty) {
      for (var i = 0; i < out.length; i += 1) {
        out[i] = manifest.padValue;
      }
      return out;
    }
    out[0] = input[0];
    for (var i = 1; i < input.length; i += 1) {
      out[i] = input[i] - manifest.preemph * input[i - 1];
    }
    for (var i = input.length; i < out.length; i += 1) {
      out[i] = manifest.padValue;
    }
    return out;
  }

  Float32List _frameAudio(Float32List input) {
    final frameCount = input.length < manifest.nFft
        ? 1
        : 1 + ((input.length - manifest.nFft) ~/ manifest.hopLength);
    final frames = Float32List(frameCount * manifest.nFft);
    for (var frame = 0; frame < frameCount; frame += 1) {
      final start = frame * manifest.hopLength;
      for (var i = 0; i < manifest.nFft; i += 1) {
        final sourceIndex = start + i;
        final sample = sourceIndex < input.length ? input[sourceIndex] : 0.0;
        frames[frame * manifest.nFft + i] = sample * _window[i];
      }
    }
    return frames;
  }
}

final class ParakeetTdtMlxMelFrontend {
  ParakeetTdtMlxMelFrontend(this.manifest);

  final ParakeetTdtManifest manifest;
  MlxArray? _windowArray;
  MlxArray? _filterbankArray;

  MlxArray compute(Float32List audio) {
    final input = MlxArray.fromFloat32List(audio, shape: <int>[audio.length]);
    MlxArray? working;
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
    MlxArray? epsilon;
    MlxArray? logMel;
    MlxArray? normalized;
    try {
      working = _preemphasize(input);
      if (manifest.padTo > working.shape[0]) {
        final padValue = MlxArray.full(<int>[], manifest.padValue);
        try {
          final paddedTo = working.pad(
            axes: const <int>[0],
            lowPads: const <int>[0],
            highPads: <int>[manifest.padTo - working.shape[0]],
            padValue: padValue,
            mode: 'constant',
          );
          working.close();
          working = paddedTo;
        } finally {
          padValue.close();
        }
      }
      padded = working.shape[0] == 0
          ? MlxArray.zeros(<int>[manifest.nFft], dtype: MlxDType.MLX_FLOAT32)
          : _reflectPadMlx(working, manifest.nFft ~/ 2);
      final paddedLength = padded.shape[0];
      final frameCount = paddedLength < manifest.nFft
          ? 1
          : 1 + ((paddedLength - manifest.nFft) ~/ manifest.hopLength);
      frameStarts = MlxArray.arange(
        0.0,
        (frameCount * manifest.hopLength).toDouble(),
        manifest.hopLength.toDouble(),
        dtype: MlxDType.MLX_INT32,
      );
      offsets = MlxArray.arange(
        0.0,
        manifest.nFft.toDouble(),
        1.0,
        dtype: MlxDType.MLX_INT32,
      );
      indices = mx.add(frameStarts.expandDims(1), offsets.expandDims(0));
      frames = padded.take(indices, axis: 0);
      windowed = mx.multiply(frames, _window().reshape(<int>[1, manifest.nFft]));
      spectrum = mx.fft.rfft(windowed, axis: 1);
      magnitude = mx.abs(spectrum);
      power = mx.multiply(magnitude, magnitude);
      mel = mx.matmul(_filterbank(), power.T);
      epsilon = MlxArray.full(<int>[], 1e-5);
      logMel = mx.log(mx.add(mel, epsilon));
      normalized = _normalize(logMel);
      final transposed = normalized.T.reshape(
        <int>[1, frameCount, manifest.melFeatures],
      );
      return transposed;
    } finally {
      input.close();
      working?.close();
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
      epsilon?.close();
      logMel?.close();
      normalized?.close();
    }
  }

  void close() {
    _windowArray?.close();
    _windowArray = null;
    _filterbankArray?.close();
    _filterbankArray = null;
  }

  MlxArray _window() {
    final cached = _windowArray;
    if (cached != null) {
      return cached;
    }
    final created = MlxArray.fromFloat32List(
      _stftWindow(winLength: manifest.winLength, nFft: manifest.nFft),
      shape: <int>[manifest.nFft],
    );
    _windowArray = created;
    return created;
  }

  MlxArray _filterbank() {
    final cached = _filterbankArray;
    if (cached != null) {
      return cached;
    }
    final bins = (manifest.nFft ~/ 2) + 1;
    final created = MlxArray.fromFloat32List(
      _melFilterbank(
        sampleRate: manifest.sampleRate,
        nFft: manifest.nFft,
        nMels: manifest.melFeatures,
      ),
      shape: <int>[manifest.melFeatures, bins],
    );
    _filterbankArray = created;
    return created;
  }

  MlxArray _preemphasize(MlxArray input) {
    final inputLength = input.shape[0];
    if (inputLength == 0) {
      return MlxArray.zeros(<int>[0], dtype: MlxDType.MLX_FLOAT32);
    }
    if (inputLength == 1) {
      return input.astype(MlxDType.MLX_FLOAT32);
    }
    final first = input.slice(start: const <int>[0], stop: const <int>[1]);
    final current = input.slice(start: const <int>[1], stop: <int>[inputLength]);
    final previous = input.slice(start: const <int>[0], stop: <int>[inputLength - 1]);
    final scaledPrevious = mx.multiply(previous, MlxArray.full(<int>[], manifest.preemph));
    final rest = mx.subtract(current, scaledPrevious);
    final result = mx.concatenate(<MlxArray>[first, rest], axis: 0);
    first.close();
    current.close();
    previous.close();
    scaledPrevious.close();
    rest.close();
    return result;
  }

  MlxArray _normalize(MlxArray input) {
    if (manifest.normalize == 'per_feature') {
      final mean = input.mean(axis: 1, keepDims: true);
      final variance = input.variance(axis: 1, keepDims: true);
      final std = mx.sqrt(mx.add(variance, MlxArray.full(<int>[], 1e-5)));
      variance.close();
      final centered = mx.subtract(input, mean);
      mean.close();
      final normalized = mx.divide(centered, std);
      centered.close();
      std.close();
      return normalized;
    }
    final mean = input.mean();
    final variance = input.variance();
    final std = mx.sqrt(mx.add(variance, MlxArray.full(<int>[], 1e-5)));
    variance.close();
    final centered = mx.subtract(input, mean);
    mean.close();
    final normalized = mx.divide(centered, std);
    centered.close();
    std.close();
    return normalized;
  }

  MlxArray _reflectPadMlx(MlxArray input, int padding) {
    if (padding <= 0) {
      return input.astype(MlxDType.MLX_FLOAT32);
    }
    final length = input.shape[0];
    if (length <= 1) {
      final indices = MlxArray.fromInt32List(
        List<int>.filled(length + (padding * 2), 0),
        shape: <int>[length + (padding * 2)],
      );
      final repeated = input.take(indices, axis: 0);
      indices.close();
      return repeated;
    }
    final leftIndexValues = Int32List.fromList(
      List<int>.generate(
        padding,
        (int index) => (padding - index).clamp(1, length - 1),
      ),
    );
    final rightIndexValues = Int32List.fromList(
      List<int>.generate(
        padding,
        (int index) => (length - 2 - index).clamp(0, length - 2),
      ),
    );
    final leftIndices = MlxArray.fromInt32List(
      leftIndexValues,
      shape: <int>[leftIndexValues.length],
    );
    final rightIndices = MlxArray.fromInt32List(
      rightIndexValues,
      shape: <int>[rightIndexValues.length],
    );
    final left = input.take(leftIndices, axis: 0);
    final right = input.take(rightIndices, axis: 0);
    leftIndices.close();
    rightIndices.close();
    final result = mx.concatenate(<MlxArray>[left, input, right], axis: 0);
    left.close();
    right.close();
    return result;
  }
}

Float32List _stftWindow({required int winLength, required int nFft}) {
  final out = Float32List(nFft);
  if (winLength <= 1) {
    if (winLength == 1) {
      out[0] = 1;
    }
    return out;
  }
  for (var i = 0; i < winLength; i += 1) {
    out[i] = 0.5 - 0.5 * math.cos((2 * math.pi * i) / winLength);
  }
  return out;
}

Float32List _reflectPad(Float32List input, int padding) {
  if (padding <= 0 || input.isEmpty) {
    return input;
  }
  if (input.length == 1) {
    final out = Float32List(input.length + (padding * 2));
    for (var i = 0; i < out.length; i += 1) {
      out[i] = input[0];
    }
    return out;
  }
  final out = Float32List(input.length + (padding * 2));
  for (var i = 0; i < padding; i += 1) {
    out[i] = input[(padding - i).clamp(1, input.length - 1)];
  }
  out.setRange(padding, padding + input.length, input);
  for (var i = 0; i < padding; i += 1) {
    out[padding + input.length + i] =
        input[(input.length - 2 - i).clamp(0, input.length - 2)];
  }
  return out;
}

Float32List _melFilterbank({
  required int sampleRate,
  required int nFft,
  required int nMels,
}) {
  final bins = (nFft ~/ 2) + 1;
  final out = Float32List(nMels * bins);
  final melMin = _hzToSlaneyMel(0);
  final melMax = _hzToSlaneyMel(sampleRate / 2);
  final melPoints = List<double>.generate(
    nMels + 2,
    (i) => melMin + (melMax - melMin) * i / (nMels + 1),
  );
  final hzPoints = melPoints.map(_slaneyMelToHz).toList(growable: false);
  final fftFreqs = List<double>.generate(
    bins,
    (index) => sampleRate * index / nFft,
    growable: false,
  );

  for (var m = 0; m < nMels; m += 1) {
    final lower = hzPoints[m];
    final center = hzPoints[m + 1];
    final upper = hzPoints[m + 2];
    final enorm = 2.0 / (upper - lower);
    for (var bin = 0; bin < bins; bin += 1) {
      final freq = fftFreqs[bin];
      final left = center > lower ? (freq - lower) / (center - lower) : 0.0;
      final right = upper > center ? (upper - freq) / (upper - center) : 0.0;
      out[m * bins + bin] = math.max(0.0, math.min(left, right)) * enorm;
    }
  }
  return out;
}

double _hzToSlaneyMel(double hz) {
  const fSp = 200.0 / 3.0;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  const logstep = 0.06875177742094912;
  if (hz < minLogHz) {
    return hz / fSp;
  }
  return minLogMel + math.log(hz / minLogHz) / logstep;
}

double _slaneyMelToHz(double mel) {
  const fSp = 200.0 / 3.0;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  const logstep = 0.06875177742094912;
  if (mel < minLogMel) {
    return mel * fSp;
  }
  return minLogHz * math.exp(logstep * (mel - minLogMel));
}
