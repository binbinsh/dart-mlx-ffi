import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'config.dart';

/// Minimum windows to trigger per-window segmented execution.
const int _windowedSegmentMinWindows = 20;

/// Audio encoder for Qwen3-ASR.
///
/// Architecture matching moona3k/mlx-qwen3-asr reference:
///   1. Split mel into chunks of n_window*2 frames
///   2. Per-chunk Conv2d stem (3 layers, stride 2 each = 8x downsample)
///   3. Channel-major reshape: (B,F',T',C) → (B,T',C*F')
///   4. Per-chunk sinusoidal position encoding (each chunk from position 0)
///   5. Windowed transformer attention (block-diagonal by chunk boundaries)
///   6. Output projection: LayerNorm → Linear → GELU → Linear
final class Qwen3AsrAudioEncoder {
  Qwen3AsrAudioEncoder._(
    this.config,
    this._conv1W,
    this._conv1B,
    this._conv2W,
    this._conv2B,
    this._conv3W,
    this._conv3B,
    this._convOutW,
    this._convOutB,
    this._layers,
    this._outNormW,
    this._outNormB,
    this._outProj1W,
    this._outProj1B,
    this._outProj2W,
    this._outProj2B,
  );

  factory Qwen3AsrAudioEncoder.load(
    Map<String, MlxArray> tensors,
    Qwen3AsrConfig config,
  ) {
    const p = 'audio_tower.';
    final layers = List<_AudioEncoderLayer>.generate(
      config.audioEncoderLayers,
      (i) => _AudioEncoderLayer(
        selfAttnLnW: tensors['${p}layers.$i.self_attn_layer_norm.weight']!,
        selfAttnLnB: tensors['${p}layers.$i.self_attn_layer_norm.bias']!,
        qProjW: tensors['${p}layers.$i.self_attn.q_proj.weight']!,
        qProjB: tensors['${p}layers.$i.self_attn.q_proj.bias']!,
        kProjW: tensors['${p}layers.$i.self_attn.k_proj.weight']!,
        kProjB: tensors['${p}layers.$i.self_attn.k_proj.bias']!,
        vProjW: tensors['${p}layers.$i.self_attn.v_proj.weight']!,
        vProjB: tensors['${p}layers.$i.self_attn.v_proj.bias']!,
        outProjW: tensors['${p}layers.$i.self_attn.out_proj.weight']!,
        outProjB: tensors['${p}layers.$i.self_attn.out_proj.bias']!,
        ffnLnW: tensors['${p}layers.$i.final_layer_norm.weight']!,
        ffnLnB: tensors['${p}layers.$i.final_layer_norm.bias']!,
        fc1W: tensors['${p}layers.$i.fc1.weight']!,
        fc1B: tensors['${p}layers.$i.fc1.bias']!,
        fc2W: tensors['${p}layers.$i.fc2.weight']!,
        fc2B: tensors['${p}layers.$i.fc2.bias']!,
      ),
    );

    return Qwen3AsrAudioEncoder._(
      config,
      tensors['${p}conv2d1.weight']!,
      tensors['${p}conv2d1.bias']!,
      tensors['${p}conv2d2.weight']!,
      tensors['${p}conv2d2.bias']!,
      tensors['${p}conv2d3.weight']!,
      tensors['${p}conv2d3.bias']!,
      tensors['${p}conv_out.weight']!,
      null,
      layers,
      tensors['${p}ln_post.weight']!,
      tensors['${p}ln_post.bias']!,
      tensors['${p}proj1.weight']!,
      tensors['${p}proj1.bias']!,
      tensors['${p}proj2.weight']!,
      tensors['${p}proj2.bias']!,
    );
  }

  final Qwen3AsrConfig config;

  // Conv frontend weights.
  final MlxArray _conv1W, _conv1B;
  final MlxArray _conv2W, _conv2B;
  final MlxArray _conv3W, _conv3B;
  final MlxArray _convOutW;
  final MlxArray? _convOutB;

  // Transformer layers.
  final List<_AudioEncoderLayer> _layers;

  // Output projection.
  final MlxArray _outNormW, _outNormB;
  final MlxArray _outProj1W, _outProj1B;
  final MlxArray _outProj2W, _outProj2B;

  MlxArray? _posEmb;

  /// Encode mel spectrogram to audio features.
  ///
  /// Input: [1, nFrames, melBins] mel spectrogram (already trimmed).
  /// Output: [1, nTokens, outputDim] audio features.
  MlxArray encode(MlxArray mel) {
    final totalFrames = mel.shape[1];
    final chunkSize = config.audioChunkSize;
    final nWindowInfer = config.audioNWindowInfer;

    // --- Per-chunk Conv2d processing ---
    final chunkTokenLens = <int>[];
    final chunkConvOutputs = <MlxArray>[];
    final nFullChunks = totalFrames ~/ chunkSize;

    // Process full-size chunks as a batch.
    if (nFullChunks > 0) {
      _processFullChunks(
        mel,
        nFullChunks,
        chunkSize,
        chunkTokenLens,
        chunkConvOutputs,
      );
    }
    // Process tail chunk (if any).
    if (nFullChunks * chunkSize < totalFrames) {
      _processTailChunk(
        mel,
        nFullChunks * chunkSize,
        totalFrames,
        chunkTokenLens,
        chunkConvOutputs,
      );
    }

    // Concatenate and project to d_model.
    var x = mx.concatenate(chunkConvOutputs, axis: 0);
    for (final c in chunkConvOutputs) {
      c.close();
    }
    final projected = _linearWithOptBias(x, _convOutW, _convOutB);
    x.close();
    x = projected;

    // --- Per-chunk sinusoidal PE (each chunk from position 0) ---
    final maxChunkTokens = chunkTokenLens.reduce(math.max);
    final pe = _getPositionEncoding(maxChunkTokens, x.dtype);
    final peParts = <MlxArray>[];
    for (final ct in chunkTokenLens) {
      peParts.add(
        pe.slice(start: [0, 0], stop: [ct, config.audioEncoderDModel]),
      );
    }
    final peFull = mx.concatenate(peParts, axis: 0);
    for (final p in peParts) {
      p.close();
    }
    final xPos = mx.add(x, peFull);
    peFull.close();
    x.close();
    x = xPos;

    // --- Windowed attention ---
    final totalTokens = x.shape[0];
    final tokensPerFullChunk = chunkTokenLens.isNotEmpty
        ? chunkTokenLens.first
        : totalTokens;
    final windowTokens = tokensPerFullChunk * (nWindowInfer ~/ chunkSize);

    // Build cu_seqlens.
    final cuSeqlens = <int>[0];
    var pos = 0;
    while (pos < totalTokens) {
      final windowEnd = pos + windowTokens < totalTokens
          ? pos + windowTokens
          : totalTokens;
      cuSeqlens.add(windowEnd);
      pos = windowEnd;
    }

    // Add batch dim: [totalTokens, dModel] → [1, totalTokens, dModel]
    var h = x.reshape([1, totalTokens, config.audioEncoderDModel]);
    x.close();

    final numWindows = cuSeqlens.length - 1;
    if (numWindows >= _windowedSegmentMinWindows) {
      h = _windowedLayers(h, cuSeqlens);
    } else {
      final mask = _createWindowedMask(totalTokens, cuSeqlens, h.dtype);
      for (var i = 0; i < _layers.length; i++) {
        final next = _transformerLayer(_layers[i], h, totalTokens, mask: mask);
        h.close();
        h = next;
      }
      mask?.close();
    }

    // Remove batch dim.
    final hFlat = h.reshape([totalTokens, config.audioEncoderDModel]);
    h.close();

    // --- Post-processing ---
    final normed = mx.fast.layerNorm(hFlat, weight: _outNormW, bias: _outNormB);
    hFlat.close();
    final proj1 = _linearWithBias(normed, _outProj1W, _outProj1B);
    normed.close();
    final activated = _gelu(proj1);
    proj1.close();
    final proj2 = _linearWithBias(activated, _outProj2W, _outProj2B);
    activated.close();

    // Reshape to [1, nTokens, outputDim].
    return proj2.reshape([1, totalTokens, config.audioOutputDim]);
  }

  /// Process full-size chunks through conv stem as a batch.
  void _processFullChunks(
    MlxArray mel,
    int nFullChunks,
    int chunkSize,
    List<int> chunkTokenLens,
    List<MlxArray> outputs,
  ) {
    final melBins = mel.shape[2];
    // Extract full chunks: [1, nFull*chunkSize, melBins]
    final fullMel = mel.slice(
      start: [0, 0, 0],
      stop: [1, nFullChunks * chunkSize, melBins],
    );
    // Reshape to [nFull, melBins, chunkSize] then NHWC [nFull, melBins, chunkSize, 1]
    final reshaped = fullMel
        .reshape([nFullChunks, chunkSize, melBins])
        .transposeAxes([0, 2, 1]);
    fullMel.close();
    final nhwc = reshaped.reshape([nFullChunks, melBins, chunkSize, 1]);
    reshaped.close();

    final convOut = _convStem(nhwc);
    nhwc.close();

    // Channel-major reshape: (nFull, F', T', C) → (nFull, T', C, F')
    final fD = convOut.shape[1]; // freq downsampled
    final tD = convOut.shape[2]; // time downsampled
    final cD = convOut.shape[3]; // channels
    final transposed = convOut.transposeAxes([0, 2, 3, 1]);
    convOut.close();
    final flat = transposed.reshape([nFullChunks * tD, cD * fD]);
    transposed.close();

    outputs.add(flat);
    for (var i = 0; i < nFullChunks; i++) {
      chunkTokenLens.add(tD);
    }
  }

  /// Process the tail chunk (fewer than chunkSize frames).
  void _processTailChunk(
    MlxArray mel,
    int startFrame,
    int totalFrames,
    List<int> chunkTokenLens,
    List<MlxArray> outputs,
  ) {
    final melBins = mel.shape[2];
    final tailLen = totalFrames - startFrame;
    // Extract tail: [1, tailLen, melBins]
    final tailMel = mel.slice(
      start: [0, startFrame, 0],
      stop: [1, totalFrames, melBins],
    );
    // NHWC: [1, melBins, tailLen, 1]
    final tailT = tailMel.transposeAxes([0, 2, 1]).reshape([
      1,
      melBins,
      tailLen,
      1,
    ]);
    tailMel.close();

    final convOut = _convStem(tailT);
    tailT.close();

    // Channel-major reshape.
    final fD = convOut.shape[1];
    final tD = convOut.shape[2];
    final cD = convOut.shape[3];
    final transposed = convOut.transposeAxes([0, 2, 3, 1]);
    convOut.close();
    final flat = transposed.reshape([tD, cD * fD]);
    transposed.close();

    outputs.add(flat);
    chunkTokenLens.add(tD);
  }

  /// Conv2d stem: 3 layers with stride 2 + GELU.
  /// Input: NHWC [B, H=melBins, W=time, C=1].
  /// Output: NHWC [B, F', T', dhs].
  MlxArray _convStem(MlxArray x) {
    var h = mx.conv2d(x, _conv1W, stride: [2, 2], padding: [1, 1]);
    h = _addBias4d(h, _conv1B);
    h = _geluInPlace(h);
    h = mx.conv2d(h, _conv2W, stride: [2, 2], padding: [1, 1]);
    h = _addBias4d(h, _conv2B);
    h = _geluInPlace(h);
    h = mx.conv2d(h, _conv3W, stride: [2, 2], padding: [1, 1]);
    h = _addBias4d(h, _conv3B);
    h = _geluInPlace(h);
    return h;
  }

  /// Apply transformer layers with per-window segmented execution.
  MlxArray _windowedLayers(MlxArray x, List<int> cuSeqlens) {
    if (cuSeqlens.length <= 2) {
      for (var i = 0; i < _layers.length; i++) {
        final next = _transformerLayer(_layers[i], x, x.shape[1]);
        x.close();
        x = next;
      }
      return x;
    }
    for (var i = 0; i < _layers.length; i++) {
      final parts = <MlxArray>[];
      for (var w = 0; w < cuSeqlens.length - 1; w++) {
        final s = cuSeqlens[w];
        final e = cuSeqlens[w + 1];
        final wLen = e - s;
        final segment = x.slice(
          start: [0, s, 0],
          stop: [1, e, config.audioEncoderDModel],
        );
        final out = _transformerLayer(_layers[i], segment, wLen);
        segment.close();
        parts.add(out);
      }
      final joined = mx.concatenate(parts, axis: 1);
      for (final p in parts) {
        p.close();
      }
      x.close();
      x = joined;
    }
    return x;
  }

  MlxArray _transformerLayer(
    _AudioEncoderLayer layer,
    MlxArray input,
    int seqLen, {
    MlxArray? mask,
  }) {
    // Pre-norm self-attention.
    final norm1 = mx.fast.layerNorm(
      input,
      weight: layer.selfAttnLnW,
      bias: layer.selfAttnLnB,
    );
    final attn = _selfAttention(layer, norm1, seqLen, mask: mask);
    norm1.close();
    final residual1 = mx.add(input, attn);
    attn.close();

    // Pre-norm FFN.
    final norm2 = mx.fast.layerNorm(
      residual1,
      weight: layer.ffnLnW,
      bias: layer.ffnLnB,
    );
    final fc1 = _linearWithBias(norm2, layer.fc1W, layer.fc1B);
    norm2.close();
    final activated = _gelu(fc1);
    fc1.close();
    final fc2 = _linearWithBias(activated, layer.fc2W, layer.fc2B);
    activated.close();
    final residual2 = mx.add(residual1, fc2);
    residual1.close();
    fc2.close();
    return residual2;
  }

  MlxArray _selfAttention(
    _AudioEncoderLayer layer,
    MlxArray input,
    int seqLen, {
    MlxArray? mask,
  }) {
    final dModel = config.audioEncoderDModel;
    final nHeads = config.audioEncoderHeads;
    final headDim = config.audioHeadDim;

    final q = _linearWithBias(input, layer.qProjW, layer.qProjB);
    final k = _linearWithBias(input, layer.kProjW, layer.kProjB);
    final v = _linearWithBias(input, layer.vProjW, layer.vProjB);

    final q4 = q.reshape([1, seqLen, nHeads, headDim]).transposeAxes([
      0,
      2,
      1,
      3,
    ]);
    final k4 = k.reshape([1, seqLen, nHeads, headDim]).transposeAxes([
      0,
      2,
      1,
      3,
    ]);
    final v4 = v.reshape([1, seqLen, nHeads, headDim]).transposeAxes([
      0,
      2,
      1,
      3,
    ]);
    q.close();
    k.close();
    v.close();

    final scale = 1.0 / math.sqrt(headDim);
    final MlxArray attn;
    if (mask != null) {
      attn = mx.fast.scaledDotProductAttention(
        q4,
        k4,
        v4,
        scale: scale,
        mask: mask,
      );
    } else {
      attn = mx.fast.scaledDotProductAttention(q4, k4, v4, scale: scale);
    }
    q4.close();
    k4.close();
    v4.close();

    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      1,
      seqLen,
      dModel,
    ]);
    attn.close();
    final out = _linearWithBias(merged, layer.outProjW, layer.outProjB);
    merged.close();
    return out;
  }

  /// Sinusoidal position encoding matching official Qwen3-ASR formula.
  ///
  /// Uses log_timescale_increment = log(10000) / (half_dim - 1),
  /// matching the HuggingFace reference SinusoidsPositionEmbedding.
  MlxArray _getPositionEncoding(int maxLen, MlxDType dtype) {
    final cached = _posEmb;
    if (cached != null && cached.shape[0] >= maxLen) return cached;
    _posEmb?.close();
    final dModel = config.audioEncoderDModel;
    final halfDim = dModel ~/ 2;
    final logInc = math.log(10000.0) / (halfDim - 1);
    final pe = Float32List(maxLen * dModel);
    for (var pos = 0; pos < maxLen; pos++) {
      for (var i = 0; i < halfDim; i++) {
        final scaledTime = pos * math.exp(-logInc * i);
        pe[pos * dModel + i] = math.sin(scaledTime).toDouble();
        pe[pos * dModel + halfDim + i] = math.cos(scaledTime).toDouble();
      }
    }
    final arr = MlxArray.fromFloat32List(
      pe,
      shape: [maxLen, dModel],
    ).astype(dtype);
    _posEmb = arr;
    return arr;
  }

  void close() {
    _posEmb?.close();
    _posEmb = null;
  }
}

/// Create windowed block-diagonal attention mask.
///
/// Tokens within the same window can attend to each other; tokens in
/// different windows cannot. Returns null if only one window.
MlxArray? _createWindowedMask(int seqLen, List<int> cuSeqlens, MlxDType dtype) {
  if (cuSeqlens.length <= 2) return null;

  // Build window assignment for each position.
  final windowIds = Int32List(seqLen);
  for (var w = 0; w < cuSeqlens.length - 1; w++) {
    for (var p = cuSeqlens[w]; p < cuSeqlens[w + 1]; p++) {
      windowIds[p] = w;
    }
  }

  // Build mask: 0 if same window, -1e9 if different.
  final maskData = Float32List(seqLen * seqLen);
  for (var i = 0; i < seqLen; i++) {
    for (var j = 0; j < seqLen; j++) {
      if (windowIds[i] != windowIds[j]) {
        maskData[i * seqLen + j] = -1e9;
      }
    }
  }
  return MlxArray.fromFloat32List(
    maskData,
    shape: [1, 1, seqLen, seqLen],
  ).astype(dtype);
}

/// Weights for one audio encoder transformer layer.
final class _AudioEncoderLayer {
  const _AudioEncoderLayer({
    required this.selfAttnLnW,
    required this.selfAttnLnB,
    required this.qProjW,
    required this.qProjB,
    required this.kProjW,
    required this.kProjB,
    required this.vProjW,
    required this.vProjB,
    required this.outProjW,
    required this.outProjB,
    required this.ffnLnW,
    required this.ffnLnB,
    required this.fc1W,
    required this.fc1B,
    required this.fc2W,
    required this.fc2B,
  });

  final MlxArray selfAttnLnW, selfAttnLnB;
  final MlxArray qProjW, qProjB;
  final MlxArray kProjW, kProjB;
  final MlxArray vProjW, vProjB;
  final MlxArray outProjW, outProjB;
  final MlxArray ffnLnW, ffnLnB;
  final MlxArray fc1W, fc1B;
  final MlxArray fc2W, fc2B;
}

// ── Helper functions ──

MlxArray _linearWithBias(MlxArray input, MlxArray weight, MlxArray bias) {
  final y = mx.matmul(input, weight.transpose());
  final biasReshaped = bias.reshape([1, bias.shape[0]]);
  try {
    final result = mx.add(y, biasReshaped);
    y.close();
    return result;
  } finally {
    biasReshaped.close();
  }
}

MlxArray _linearWithOptBias(MlxArray input, MlxArray weight, MlxArray? bias) {
  if (bias == null) return mx.matmul(input, weight.transpose());
  return _linearWithBias(input, weight, bias);
}

MlxArray _addBias4d(MlxArray input, MlxArray bias) {
  final biasReshaped = bias.reshape([1, 1, 1, bias.shape[0]]);
  try {
    final result = mx.add(input, biasReshaped);
    input.close();
    return result;
  } finally {
    biasReshaped.close();
  }
}

MlxArray _gelu(MlxArray x) {
  final invSqrt2 = MlxArray.fromFloat32List(
    [1.0 / math.sqrt(2.0)],
    shape: [1],
  ).astype(x.dtype);
  final half = MlxArray.fromFloat32List([0.5], shape: [1]).astype(x.dtype);
  final one = MlxArray.fromFloat32List([1.0], shape: [1]).astype(x.dtype);
  try {
    final scaled = x * invSqrt2;
    final erfVal = scaled.erf();
    scaled.close();
    final sum = mx.add(one, erfVal);
    erfVal.close();
    final left = x * half;
    final result = left * sum;
    left.close();
    sum.close();
    return result;
  } finally {
    invSqrt2.close();
    half.close();
    one.close();
  }
}

MlxArray _geluInPlace(MlxArray x) {
  final result = _gelu(x);
  x.close();
  return result;
}
