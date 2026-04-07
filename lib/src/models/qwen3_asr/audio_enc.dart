import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'config.dart';

/// Audio encoder for Qwen3-ASR.
///
/// Architecture: Conv2d frontend (3 layers, stride 2 each = 8x downsample)
/// → sinusoidal position encoding → 18 Transformer layers
/// → output projection (Linear → GELU → Linear).
///
/// Tensor prefix: "model.audio_tower."
final class Qwen3AsrAudioEncoder {
  Qwen3AsrAudioEncoder._(
    this.config,
    this._conv1W,
    this._conv1B,
    this._conv2W,
    this._conv2B,
    this._conv3W,
    this._conv3B,
    this._linearInW,
    this._linearInB,
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
    const p = 'model.audio_tower.';
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
      tensors['${p}conv1.weight']!,
      tensors['${p}conv1.bias']!,
      tensors['${p}conv2.weight']!,
      tensors['${p}conv2.bias']!,
      tensors['${p}conv3.weight']!,
      tensors['${p}conv3.bias']!,
      tensors['${p}linear_in.weight']!,
      tensors['${p}linear_in.bias']!,
      layers,
      tensors['${p}layer_norm.weight']!,
      tensors['${p}layer_norm.bias']!,
      tensors['${p}output_projection.0.weight']!,
      tensors['${p}output_projection.0.bias']!,
      tensors['${p}output_projection.2.weight']!,
      tensors['${p}output_projection.2.bias']!,
    );
  }

  final Qwen3AsrConfig config;

  // Conv frontend weights.
  final MlxArray _conv1W, _conv1B;
  final MlxArray _conv2W, _conv2B;
  final MlxArray _conv3W, _conv3B;
  final MlxArray _linearInW, _linearInB;

  // Transformer layers.
  final List<_AudioEncoderLayer> _layers;

  // Output projection.
  final MlxArray _outNormW, _outNormB;
  final MlxArray _outProj1W, _outProj1B;
  final MlxArray _outProj2W, _outProj2B;

  MlxArray? _posEmb;

  /// Encode mel spectrogram to audio features.
  /// Input: [1, nFrames, 128] mel spectrogram.
  /// Output: [1, nEncoderFrames, outputDim] audio features.
  MlxArray encode(MlxArray mel) {
    // Conv frontend: mel [1, T, 128] → [1, T', dModel].
    var x = _convFrontend(mel);

    // Add sinusoidal position encoding.
    final seqLen = x.shape[1];
    final posEmb = _getPositionEncoding(seqLen, x.dtype);
    final posSlice = posEmb
        .slice(start: [0, 0], stop: [seqLen, config.audioEncoderDModel])
        .reshape([1, seqLen, config.audioEncoderDModel]);
    final xPos = mx.add(x, posSlice);
    posSlice.close();
    x.close();
    x = xPos;

    // Transformer layers.
    for (var i = 0; i < _layers.length; i++) {
      final next = _transformerLayer(_layers[i], x, seqLen);
      x.close();
      x = next;
    }

    // Output projection: LayerNorm → Linear → GELU → Linear.
    final normed = mx.fast.layerNorm(x, weight: _outNormW, bias: _outNormB);
    x.close();
    final proj1 = _linearWithBias(normed, _outProj1W, _outProj1B);
    normed.close();
    final activated = _gelu(proj1);
    proj1.close();
    final proj2 = _linearWithBias(activated, _outProj2W, _outProj2B);
    activated.close();
    return proj2;
  }

  /// Conv2d frontend: 3 conv layers with stride 2 + GELU, then linear.
  /// Input: [1, T, 128] → reshape to [1, 128, T, 1] for Conv2d.
  MlxArray _convFrontend(MlxArray mel) {
    // MLX Conv2d expects NHWC: [batch, height, width, channels].
    // Treat mel as [1, melBins, T, 1] after transpose.
    final melT = mel.transposeAxes([0, 2, 1]).reshape([
      1,
      mel.shape[2],
      mel.shape[1],
      1,
    ]);
    try {
      // Conv1: [1, 128, T, 1] → [1, 64, T/2, outChannels].
      var x = mx.conv2d(melT, _conv1W, stride: [2, 2], padding: [1, 1]);
      x = _addBias4d(x, _conv1B);
      x = _geluInPlace(x);

      // Conv2: stride 2 again.
      x = mx.conv2d(x, _conv2W, stride: [2, 2], padding: [1, 1]);
      x = _addBias4d(x, _conv2B);
      x = _geluInPlace(x);

      // Conv3: stride 2 again → total 8x downsample.
      x = mx.conv2d(x, _conv3W, stride: [2, 2], padding: [1, 1]);
      x = _addBias4d(x, _conv3B);
      x = _geluInPlace(x);

      // Flatten spatial dims: [1, H', W', C] → [1, H'*W', C]
      // But the actual architecture flattens to [1, T', flatten_dim]
      // where T' = W' (time) and flatten_dim = H' * C.
      final h = x.shape[1]; // melBins / 8
      final w = x.shape[2]; // T / 8
      final c = x.shape[3]; // output channels
      final flatDim = h * c;
      // Reshape: [1, H', W', C] → [1, W', H'*C] via transpose+reshape.
      final xPerm = x.transposeAxes([0, 2, 1, 3]);
      x.close();
      final flat = xPerm.reshape([1, w, flatDim]);
      xPerm.close();

      // Linear projection: flatten_dim → d_model.
      final projected = _linearWithBias(flat, _linearInW, _linearInB);
      flat.close();
      return projected;
    } finally {
      melT.close();
    }
  }

  MlxArray _transformerLayer(
    _AudioEncoderLayer layer,
    MlxArray input,
    int seqLen,
  ) {
    // Pre-norm self-attention.
    final norm1 = mx.fast.layerNorm(
      input,
      weight: layer.selfAttnLnW,
      bias: layer.selfAttnLnB,
    );
    final attn = _selfAttention(layer, norm1, seqLen);
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
    int seqLen,
  ) {
    final dModel = config.audioEncoderDModel;
    final nHeads = config.audioEncoderHeads;
    final headDim = config.audioHeadDim;

    // Q, K, V projections (all with bias).
    final q = _linearWithBias(input, layer.qProjW, layer.qProjB);
    final k = _linearWithBias(input, layer.kProjW, layer.kProjB);
    final v = _linearWithBias(input, layer.vProjW, layer.vProjB);

    // Reshape to [1, seqLen, nHeads, headDim] → [1, nHeads, seqLen, headDim].
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

    // Scaled dot-product attention (no causal mask for encoder).
    final attn = mx.fast.scaledDotProductAttention(
      q4,
      k4,
      v4,
      scale: 1.0 / math.sqrt(headDim),
    );
    q4.close();
    k4.close();
    v4.close();

    // Merge heads and project.
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

  /// Get or compute sinusoidal position encoding.
  MlxArray _getPositionEncoding(int maxLen, MlxDType dtype) {
    final cached = _posEmb;
    if (cached != null && cached.shape[0] >= maxLen) return cached;
    _posEmb?.close();
    // Build sinusoidal PE: [maxLen, dModel].
    final dModel = config.audioEncoderDModel;
    final pe = Float64List(maxLen * dModel);
    for (var pos = 0; pos < maxLen; pos++) {
      for (var i = 0; i < dModel; i += 2) {
        final angle = pos / math.pow(10000.0, i / dModel);
        pe[pos * dModel + i] = math.sin(angle);
        if (i + 1 < dModel) {
          pe[pos * dModel + i + 1] = math.cos(angle);
        }
      }
    }
    final peF32 = Float32List(pe.length);
    for (var i = 0; i < pe.length; i++) peF32[i] = pe[i];
    final arr = MlxArray.fromFloat32List(
      peF32,
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
