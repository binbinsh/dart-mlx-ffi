part of 'qwen3_tts.dart';

final class Qwen3TtsSpeakerEncoder {
  Qwen3TtsSpeakerEncoder(this.bundle) : _cfg = bundle.manifest.speaker;

  final Qwen3TtsBundle bundle;
  final Qwen3TtsSpeakerConfig _cfg;
  _Qwen3TtsSpeakerFrontend? _frontend;

  int get sampleRate => _cfg.sampleRate;
  int get embeddingDim => _cfg.encDim;

  Float32List embed(Float32List samples) {
    final frontend = _frontend ??= _Qwen3TtsSpeakerFrontend(_cfg);
    MlxArray? mel;
    MlxArray? x;
    MlxArray? block0;
    MlxArray? block1;
    MlxArray? block2;
    MlxArray? block3;
    MlxArray? concat;
    MlxArray? mfa;
    MlxArray? asp;
    MlxArray? fc;
    MlxArray? embedding;
    try {
      mel = frontend.encode(samples);
      x = mel;
      block0 = _tdnnNoBn(
        x,
        weight: bundle.require('speaker_encoder.blocks.0.conv.weight'),
        bias: bundle.require('speaker_encoder.blocks.0.conv.bias'),
        kernelSize: _cfg.encKernelSizes[0],
        dilation: _cfg.encDilations[0],
      );
      block1 = _seRes2NetBlockNoBn(block0, index: 1, kernelSize: _cfg.encKernelSizes[1], dilation: _cfg.encDilations[1]);
      block2 = _seRes2NetBlockNoBn(block1, index: 2, kernelSize: _cfg.encKernelSizes[2], dilation: _cfg.encDilations[2]);
      block3 = _seRes2NetBlockNoBn(block2, index: 3, kernelSize: _cfg.encKernelSizes[3], dilation: _cfg.encDilations[3]);
      concat = mx.concatenate([block1, block2, block3], axis: 2);
      mfa = _tdnnNoBn(
        concat,
        weight: bundle.require('speaker_encoder.mfa.conv.weight'),
        bias: bundle.require('speaker_encoder.mfa.conv.bias'),
        kernelSize: _cfg.encKernelSizes.last,
        dilation: _cfg.encDilations.last,
      );
      asp = _attentiveStatisticsPoolNoBn(
        mfa,
        tdnnW: bundle.require('speaker_encoder.asp.tdnn.conv.weight'),
        tdnnB: bundle.require('speaker_encoder.asp.tdnn.conv.bias'),
        convW: bundle.require('speaker_encoder.asp.conv.weight'),
        convB: bundle.require('speaker_encoder.asp.conv.bias'),
        eps: 1e-12,
      );
      fc = spk_nn.speechBrainConv1d(
        asp,
        weight: bundle.require('speaker_encoder.fc.weight'),
        bias: bundle.require('speaker_encoder.fc.bias'),
        kernelSize: 1,
        dilation: 1,
        padSame: false,
      );
      embedding = fc.reshape([_cfg.encDim]);
      MlxRuntime.evalAll([embedding]);
      return embedding.toFloat32List();
    } finally {
      mel?.close();
      x = null;
      block0?.close();
      block1?.close();
      block2?.close();
      block3?.close();
      concat?.close();
      mfa?.close();
      asp?.close();
      fc?.close();
      embedding?.close();
    }
  }

  void close() {
    _frontend?.close();
    _frontend = null;
  }

  MlxArray _tdnnNoBn(
    MlxArray input, {
    required MlxArray weight,
    required MlxArray bias,
    required int kernelSize,
    required int dilation,
  }) {
    final conv = spk_nn.speechBrainConv1d(
      input,
      weight: weight,
      bias: bias,
      kernelSize: kernelSize,
      dilation: dilation,
      padSame: true,
    );
    try {
      return mx.maximum(conv, MlxArray.full([], 0.0).astype(conv.dtype));
    } finally {
      conv.close();
    }
  }

  MlxArray _res2NetNoBn(
    MlxArray input, {
    required List<MlxArray> weights,
    required List<MlxArray> biases,
    required int scale,
    required int kernelSize,
    required int dilation,
  }) {
    final c = input.shape[2];
    if (c % scale != 0) {
      throw StateError('Qwen3-TTS speaker Res2Net expected channels divisible by scale=$scale.');
    }
    final chunk = c ~/ scale;
    final parts = <MlxArray>[];
    MlxArray? running;
    try {
      for (var i = 0; i < scale; i++) {
        final xi = input.slice(
          start: [0, 0, i * chunk],
          stop: [input.shape[0], input.shape[1], (i + 1) * chunk],
        );
        late final MlxArray yi;
        if (i == 0) {
          yi = xi;
        } else if (i == 1) {
          yi = _tdnnNoBn(
            xi,
            weight: weights[i - 1],
            bias: biases[i - 1],
            kernelSize: kernelSize,
            dilation: dilation,
          );
          xi.close();
        } else {
          final summed = mx.add(xi, running!);
          xi.close();
          yi = _tdnnNoBn(
            summed,
            weight: weights[i - 1],
            bias: biases[i - 1],
            kernelSize: kernelSize,
            dilation: dilation,
          );
          summed.close();
        }
        if (i >= 1) {
          running = yi;
        }
        parts.add(yi);
      }
      return mx.concatenate(parts, axis: 2);
    } finally {
      for (final part in parts) {
        part.close();
      }
    }
  }

  MlxArray _seRes2NetBlockNoBn(
    MlxArray input, {
    required int index,
    required int kernelSize,
    required int dilation,
  }) {
    MlxArray? tdnn1;
    MlxArray? res;
    MlxArray? tdnn2;
    MlxArray? se;
    try {
      final prefix = 'speaker_encoder.blocks.$index';
      tdnn1 = _tdnnNoBn(
        input,
        weight: bundle.require('$prefix.tdnn1.conv.weight'),
        bias: bundle.require('$prefix.tdnn1.conv.bias'),
        kernelSize: 1,
        dilation: 1,
      );
      final resWeights = <MlxArray>[
        for (var i = 0; i < _cfg.encRes2netScale - 1; i++)
          bundle.require('$prefix.res2net_block.blocks.$i.conv.weight'),
      ];
      final resBiases = <MlxArray>[
        for (var i = 0; i < _cfg.encRes2netScale - 1; i++)
          bundle.require('$prefix.res2net_block.blocks.$i.conv.bias'),
      ];
      res = _res2NetNoBn(
        tdnn1,
        weights: resWeights,
        biases: resBiases,
        scale: _cfg.encRes2netScale,
        kernelSize: kernelSize,
        dilation: dilation,
      );
      tdnn2 = _tdnnNoBn(
        res,
        weight: bundle.require('$prefix.tdnn2.conv.weight'),
        bias: bundle.require('$prefix.tdnn2.conv.bias'),
        kernelSize: 1,
        dilation: 1,
      );
      se = spk_nn.seBlock(
        tdnn2,
        conv1W: bundle.require('$prefix.se_block.conv1.weight'),
        conv1B: bundle.require('$prefix.se_block.conv1.bias'),
        conv2W: bundle.require('$prefix.se_block.conv2.weight'),
        conv2B: bundle.require('$prefix.se_block.conv2.bias'),
      );
      return mx.add(se, input);
    } finally {
      tdnn1?.close();
      res?.close();
      tdnn2?.close();
      se?.close();
    }
  }

  MlxArray _attentiveStatisticsPoolNoBn(
    MlxArray input, {
    required MlxArray tdnnW,
    required MlxArray tdnnB,
    required MlxArray convW,
    required MlxArray convB,
    required double eps,
  }) {
    final b = input.shape[0];
    final t = input.shape[1];
    final c = input.shape[2];
    MlxArray? mean1;
    MlxArray? centered;
    MlxArray? sq;
    MlxArray? varRaw;
    MlxArray? varClamped;
    MlxArray? std1;
    MlxArray? meanBcast;
    MlxArray? stdBcast;
    MlxArray? concat;
    MlxArray? tdnnOut;
    MlxArray? tanh;
    MlxArray? conv;
    MlxArray? attn;
    MlxArray? weighted;
    MlxArray? weightedMean;
    MlxArray? diff;
    MlxArray? diffSq;
    MlxArray? weightedDiffSq;
    MlxArray? varAttn;
    MlxArray? varAttnClamped;
    MlxArray? stdAttn;
    MlxArray? pooled;
    try {
      mean1 = input.mean(axis: 1, keepDims: true);
      centered = mx.subtract(input, mean1);
      sq = mx.multiply(centered, centered);
      varRaw = sq.mean(axis: 1, keepDims: true);
      varClamped = mx.maximum(varRaw, MlxArray.full([], eps).astype(varRaw.dtype));
      std1 = mx.sqrt(varClamped);
      meanBcast = mx.broadcastTo(mean1, [b, t, c]);
      stdBcast = mx.broadcastTo(std1, [b, t, c]);
      concat = mx.concatenate([input, meanBcast, stdBcast], axis: 2);
      tdnnOut = _tdnnNoBn(
        concat,
        weight: tdnnW,
        bias: tdnnB,
        kernelSize: 1,
        dilation: 1,
      );
      tanh = tdnnOut.tanh();
      conv = spk_nn.speechBrainConv1d(
        tanh,
        weight: convW,
        bias: convB,
        kernelSize: 1,
        dilation: 1,
        padSame: false,
      );
      attn = mx.softmax(conv, axis: 1);
      weighted = mx.multiply(input, attn);
      weightedMean = weighted.sum(axis: 1, keepDims: true);
      diff = mx.subtract(input, weightedMean);
      diffSq = mx.multiply(diff, diff);
      weightedDiffSq = mx.multiply(diffSq, attn);
      varAttn = weightedDiffSq.sum(axis: 1, keepDims: true);
      varAttnClamped = mx.maximum(varAttn, MlxArray.full([], eps).astype(varAttn.dtype));
      stdAttn = mx.sqrt(varAttnClamped);
      pooled = mx.concatenate([weightedMean, stdAttn], axis: 2);
      return pooled;
    } finally {
      mean1?.close();
      centered?.close();
      sq?.close();
      varRaw?.close();
      varClamped?.close();
      std1?.close();
      meanBcast?.close();
      stdBcast?.close();
      concat?.close();
      tdnnOut?.close();
      tanh?.close();
      conv?.close();
      attn?.close();
      weighted?.close();
      weightedMean?.close();
      diff?.close();
      diffSq?.close();
      weightedDiffSq?.close();
      varAttn?.close();
      varAttnClamped?.close();
      stdAttn?.close();
    }
  }
}

final class _Qwen3TtsSpeakerFrontend {
  _Qwen3TtsSpeakerFrontend(this._cfg)
    : _window = _hannWindow(1024),
      _filterbank = _slaneyMelFilterbank(
        sampleRate: _cfg.sampleRate,
        nFft: 1024,
        nMels: _cfg.melDim,
        fMin: 0.0,
        fMax: _cfg.sampleRate / 2.0,
      );

  final Qwen3TtsSpeakerConfig _cfg;
  final Float32List _window;
  final Float32List _filterbank;
  MlxArray? _windowArr;
  MlxArray? _filterbankArr;

  MlxArray encode(Float32List audio) {
    final input = MlxArray.fromFloat32List(audio, shape: [audio.length]);
    MlxArray? padded;
    MlxArray? frameStarts;
    MlxArray? offsets;
    MlxArray? indices;
    MlxArray? frames;
    MlxArray? windowed;
    MlxArray? spectrum;
    MlxArray? magnitude;
    MlxArray? magnitudeSq;
    MlxArray? stabilized;
    MlxArray? mel;
    MlxArray? clipped;
    MlxArray? logMel;
    try {
      const nFft = 1024;
      const hop = 256;
      const pad = (nFft - hop) ~/ 2;
      padded = _reflectPad(input, pad);
      final frameCount = padded.shape[0] < nFft ? 1 : 1 + ((padded.shape[0] - nFft) ~/ hop);
      frameStarts = MlxArray.arange(0.0, (frameCount * hop).toDouble(), hop.toDouble(), dtype: MlxDType.MLX_INT32);
      offsets = MlxArray.arange(0.0, nFft.toDouble(), 1.0, dtype: MlxDType.MLX_INT32);
      indices = mx.add(frameStarts.expandDims(1), offsets.expandDims(0));
      frames = padded.take(indices, axis: 0);
      windowed = mx.multiply(frames, _windowMlx().reshape([1, nFft]));
      spectrum = mx.fft.rfft(windowed, axis: 1);
      magnitude = mx.abs(spectrum);
      magnitudeSq = mx.multiply(magnitude, magnitude);
      stabilized = mx.add(magnitudeSq, MlxArray.full([], 1e-9).astype(magnitudeSq.dtype));
      final specMag = mx.sqrt(stabilized);
      mel = mx.matmul(specMag, _filterbankMlx().T);
      specMag.close();
      clipped = mx.maximum(mel, MlxArray.full([], 1e-5).astype(mel.dtype));
      logMel = mx.log(clipped);
      return logMel.reshape([1, frameCount, _cfg.melDim]);
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
      magnitudeSq?.close();
      stabilized?.close();
      mel?.close();
      clipped?.close();
    }
  }

  void close() {
    _windowArr?.close();
    _windowArr = null;
    _filterbankArr?.close();
    _filterbankArr = null;
  }

  MlxArray _windowMlx() {
    final cached = _windowArr;
    if (cached != null) return cached;
    final created = MlxArray.fromFloat32List(_window, shape: [1024]);
    _windowArr = created;
    return created;
  }

  MlxArray _filterbankMlx() {
    final cached = _filterbankArr;
    if (cached != null) return cached;
    final bins = 1024 ~/ 2 + 1;
    final created = MlxArray.fromFloat32List(_filterbank, shape: [_cfg.melDim, bins]);
    _filterbankArr = created;
    return created;
  }
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
  final rightIdx = MlxArray.fromInt32List(rightVals, shape: [rightVals.length]);
  final left = input.take(leftIdx, axis: 0);
  final right = input.take(rightIdx, axis: 0);
  leftIdx.close();
  rightIdx.close();
  final result = mx.concatenate([left, input, right], axis: 0);
  left.close();
  right.close();
  return result;
}

Float32List _hannWindow(int n) {
  final out = Float32List(n);
  for (var i = 0; i < n; i++) {
    out[i] = 0.5 - 0.5 * math.cos((2 * math.pi * i) / n);
  }
  return out;
}

Float32List _slaneyMelFilterbank({
  required int sampleRate,
  required int nFft,
  required int nMels,
  required double fMin,
  required double fMax,
}) {
  final bins = (nFft ~/ 2) + 1;
  final out = Float32List(nMels * bins);
  final melMin = _slaneyHzToMel(fMin);
  final melMax = _slaneyHzToMel(fMax);
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

double _slaneyHzToMel(double hz) {
  if (hz < 1000.0) return 3.0 * hz / 200.0;
  return 15.0 + 27.0 * math.log(hz / 1000.0) / _ln6_4;
}

double _slaneyMelToHz(double mel) {
  if (mel < 15.0) return 200.0 * mel / 3.0;
  return 1000.0 * math.exp((mel - 15.0) * _ln6_4 / 27.0);
}

const double _ln6_4 = 1.8562979903656263;
