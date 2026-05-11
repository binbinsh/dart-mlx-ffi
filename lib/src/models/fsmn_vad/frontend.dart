import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/dart_mlx_ffi.dart';

import 'bundle.dart';

class FsmnVadFrontendOutput {
  const FsmnVadFrontendOutput({required this.values, required this.frames});

  final Float32List values;
  final int frames;

  bool get isEmpty => frames == 0 || values.isEmpty;

  static final FsmnVadFrontendOutput empty = FsmnVadFrontendOutput(
    values: Float32List(0),
    frames: 0,
  );
}

final class FsmnVadFrontend {
  FsmnVadFrontend({
    required FsmnVadManifest manifest,
    required FsmnVadCmvn cmvn,
  }) : _manifest = manifest,
       _cmvn = cmvn,
       _window = _hammingWindow(manifest.frameSampleLength),
       _filterbank = _kaldiMelFilterbank(
         sampleRate: manifest.sampleRate,
         fftSize: manifest.fftSize,
         nMels: manifest.numMels,
       );

  final FsmnVadManifest _manifest;
  final FsmnVadCmvn _cmvn;
  final Float32List _window;
  final Float32List _filterbank;

  MlxArray? _windowArray;
  MlxArray? _filterbankArray;

  Float32List _inputCache = Float32List(0);
  final List<Float32List> _lfrSpliceCache = <Float32List>[];

  FsmnVadFrontendOutput process(Float32List samples, {bool isFinal = false}) {
    final merged = _mergeSamples(_inputCache, samples);
    final frameCount = _computeFrameCount(merged.length);
    final freshFrames = <Float32List>[];

    if (frameCount > 0) {
      freshFrames.addAll(_computeFbankFrames(merged, frameCount));
      final nextCacheOffset =
          merged.length -
          (merged.length - frameCount * _manifest.frameShiftSampleLength);
      _inputCache = Float32List.fromList(merged.sublist(nextCacheOffset));
    } else {
      _inputCache = Float32List.fromList(merged);
    }

    if (freshFrames.isNotEmpty && _lfrSpliceCache.isEmpty) {
      final leftPad = (_manifest.lfrM - 1) ~/ 2;
      for (var i = 0; i < leftPad; i += 1) {
        _lfrSpliceCache.add(Float32List.fromList(freshFrames.first));
      }
    }

    if (freshFrames.isNotEmpty) {
      _lfrSpliceCache.addAll(freshFrames);
    }

    if (_lfrSpliceCache.length < _manifest.lfrM) {
      if (!isFinal) {
        return FsmnVadFrontendOutput.empty;
      }
      if (_lfrSpliceCache.isEmpty) {
        return FsmnVadFrontendOutput.empty;
      }
    }

    final lfr = _applyLfr(isFinal: isFinal);
    if (lfr.isEmpty) {
      return FsmnVadFrontendOutput.empty;
    }
    _applyCmvn(lfr);
    return FsmnVadFrontendOutput(
      values: _flattenFrames(lfr),
      frames: lfr.length,
    );
  }

  void reset() {
    _inputCache = Float32List(0);
    _lfrSpliceCache.clear();
  }

  void close() {
    _windowArray?.close();
    _windowArray = null;
    _filterbankArray?.close();
    _filterbankArray = null;
  }

  int _computeFrameCount(int sampleLength) {
    final frameLength = _manifest.frameSampleLength;
    final shift = _manifest.frameShiftSampleLength;
    final frames = ((sampleLength - frameLength) ~/ shift) + 1;
    if (sampleLength < frameLength || frames < 1) {
      return 0;
    }
    return frames;
  }

  List<Float32List> _computeFbankFrames(Float32List audio, int frameCount) {
    final input = MlxArray.fromFloat32List(audio, shape: [audio.length]);
    MlxArray? frameStarts;
    MlxArray? offsets;
    MlxArray? indices;
    MlxArray? frames;
    MlxArray? centered;
    MlxArray? prefix;
    MlxArray? first;
    MlxArray? previous;
    MlxArray? preemphasized;
    MlxArray? windowed;
    MlxArray? padding;
    MlxArray? padded;
    MlxArray? spectrum;
    MlxArray? power;
    MlxArray? mel;
    MlxArray? logMel;
    try {
      frameStarts = MlxArray.arange(
        0.0,
        (frameCount * _manifest.frameShiftSampleLength).toDouble(),
        _manifest.frameShiftSampleLength.toDouble(),
        dtype: MlxDType.MLX_INT32,
      );
      offsets = MlxArray.arange(
        0.0,
        _manifest.frameSampleLength.toDouble(),
        1.0,
        dtype: MlxDType.MLX_INT32,
      );
      indices = mx.add(frameStarts.expandDims(1), offsets.expandDims(0));
      frames = input.take(indices, axis: 0);

      centered = mx.subtract(frames, frames.mean(axis: 1, keepDims: true));
      prefix = centered.slice(
        start: [0, 0],
        stop: [frameCount, _manifest.frameSampleLength - 1],
      );
      first = centered.slice(start: [0, 0], stop: [frameCount, 1]);
      previous = mx.concatenate([first, prefix], axis: 1);
      preemphasized = mx.subtract(
        centered,
        mx.multiply(previous, MlxArray.full([], 0.97)),
      );

      windowed = mx.multiply(
        preemphasized,
        _windowMlx().reshape([1, _manifest.frameSampleLength]),
      );
      final padLength = _manifest.fftSize - _manifest.frameSampleLength;
      if (padLength > 0) {
        padding = mx.zeros([frameCount, padLength]);
        padded = mx.concatenate([windowed, padding], axis: 1);
      } else {
        padded = windowed;
      }

      spectrum = mx.fft.rfft(padded, axis: 1);
      final magnitude = mx.abs(spectrum);
      power = mx.multiply(magnitude, magnitude);
      magnitude.close();
      mel = mx.matmul(power, _filterbankMlx());
      logMel = mx.log(mx.maximum(mel, MlxArray.full([], 1e-10)));
      MlxRuntime.evalAll([logMel]);
      final flat = logMel.toFloat32List();
      final out = <Float32List>[];
      final melBins = _manifest.numMels;
      for (var frame = 0; frame < frameCount; frame += 1) {
        final start = frame * melBins;
        out.add(Float32List.fromList(flat.sublist(start, start + melBins)));
      }
      return out;
    } finally {
      input.close();
      frameStarts?.close();
      offsets?.close();
      indices?.close();
      frames?.close();
      centered?.close();
      prefix?.close();
      first?.close();
      previous?.close();
      preemphasized?.close();
      if (!identical(padded, windowed)) {
        windowed?.close();
      }
      padding?.close();
      padded?.close();
      spectrum?.close();
      power?.close();
      mel?.close();
      logMel?.close();
    }
  }

  List<Float32List> _applyLfr({required bool isFinal}) {
    final frames = List<Float32List>.from(_lfrSpliceCache, growable: true);
    if (frames.isEmpty) {
      return const <Float32List>[];
    }
    final leftContext = (_manifest.lfrM - 1) ~/ 2;
    final total = frames.length;
    final tLfr = ((total - leftContext) / _manifest.lfrN).ceil();
    final lastIdx = ((total - _manifest.lfrM) ~/ _manifest.lfrN) + 1;
    var spliceRows = tLfr;
    var spliceIdx = tLfr;

    final numPadding = _manifest.lfrM - (total - lastIdx * _manifest.lfrN);
    if (isFinal) {
      if (numPadding > 0) {
        final extra =
            ((2 * _manifest.lfrM) -
                (2 * total) +
                ((tLfr - 1 + lastIdx) * _manifest.lfrN)) ~/
            2;
        final copies = extra * (tLfr - lastIdx);
        for (var i = 0; i < copies; i += 1) {
          frames.add(Float32List.fromList(frames.last));
        }
      }
    } else if (numPadding > 0) {
      spliceRows = lastIdx;
      spliceIdx = lastIdx;
    }

    if (spliceRows <= 0) {
      return const <Float32List>[];
    }

    spliceIdx = math.min(total - 1, spliceIdx * _manifest.lfrN);
    final out = <Float32List>[];
    for (var row = 0; row < spliceRows; row += 1) {
      final start = row * _manifest.lfrN;
      final merged = Float32List(_manifest.inputDim);
      for (var i = 0; i < _manifest.lfrM; i += 1) {
        merged.setRange(
          i * _manifest.numMels,
          (i + 1) * _manifest.numMels,
          frames[start + i],
        );
      }
      out.add(merged);
    }

    _lfrSpliceCache
      ..clear()
      ..addAll(frames.sublist(spliceIdx));
    return out;
  }

  void _applyCmvn(List<Float32List> frames) {
    final offsets = _cmvn.offsets;
    final scales = _cmvn.scales;
    for (final frame in frames) {
      for (var index = 0; index < frame.length; index += 1) {
        frame[index] = (frame[index] + offsets[index]) * scales[index];
      }
    }
  }

  Float32List _flattenFrames(List<Float32List> frames) {
    final out = Float32List(frames.length * _manifest.inputDim);
    for (var i = 0; i < frames.length; i += 1) {
      out.setRange(
        i * _manifest.inputDim,
        (i + 1) * _manifest.inputDim,
        frames[i],
      );
    }
    return out;
  }

  MlxArray _windowMlx() {
    final cached = _windowArray;
    if (cached != null) {
      return cached;
    }
    final created = MlxArray.fromFloat32List(
      _window,
      shape: [_manifest.frameSampleLength],
    );
    _windowArray = created;
    return created;
  }

  MlxArray _filterbankMlx() {
    final cached = _filterbankArray;
    if (cached != null) {
      return cached;
    }
    final bins = (_manifest.fftSize ~/ 2) + 1;
    final created = MlxArray.fromFloat32List(
      _filterbank,
      shape: [bins, _manifest.numMels],
    );
    _filterbankArray = created;
    return created;
  }
}

Float32List _hammingWindow(int length) {
  final out = Float32List(length);
  if (length <= 1) {
    if (length == 1) {
      out[0] = 1.0;
    }
    return out;
  }
  for (var i = 0; i < length; i += 1) {
    out[i] = 0.54 - (0.46 * math.cos((2 * math.pi * i) / (length - 1)));
  }
  return out;
}

Float32List _kaldiMelFilterbank({
  required int sampleRate,
  required int fftSize,
  required int nMels,
}) {
  final bins = (fftSize ~/ 2) + 1;
  final out = Float32List(bins * nMels);
  final melMin = _kaldiHzToMel(0.0);
  final melMax = _kaldiHzToMel(sampleRate / 2.0);
  final melPoints = List<double>.generate(
    nMels + 2,
    (i) => melMin + (melMax - melMin) * i / (nMels + 1),
    growable: false,
  );
  final hzPoints = melPoints.map(_kaldiMelToHz).toList(growable: false);
  final fftFreqs = List<double>.generate(
    bins,
    (i) => sampleRate * i / fftSize,
    growable: false,
  );
  for (var mel = 0; mel < nMels; mel += 1) {
    final lower = hzPoints[mel];
    final center = hzPoints[mel + 1];
    final upper = hzPoints[mel + 2];
    for (var bin = 0; bin < bins; bin += 1) {
      final freq = fftFreqs[bin];
      final left = center > lower ? (freq - lower) / (center - lower) : 0.0;
      final right = upper > center ? (upper - freq) / (upper - center) : 0.0;
      out[(bin * nMels) + mel] = math
          .max(0.0, math.min(left, right))
          .toDouble();
    }
  }
  return out;
}

double _kaldiHzToMel(double hz) => 1127.0 * math.log(1.0 + (hz / 700.0));

double _kaldiMelToHz(double mel) => 700.0 * (math.exp(mel / 1127.0) - 1.0);

Float32List _mergeSamples(Float32List left, Float32List right) {
  if (left.isEmpty) {
    return right;
  }
  if (right.isEmpty) {
    return left;
  }
  final out = Float32List(left.length + right.length);
  out.setRange(0, left.length, left);
  out.setRange(left.length, out.length, right);
  return out;
}
