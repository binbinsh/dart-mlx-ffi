import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';
import 'nn.dart';

// ---------------------------------------------------------------------------
// VAD State
// ---------------------------------------------------------------------------

/// LSTM state carried across frames.
class SileroVadState {
  SileroVadState({required this.h, required this.c});

  final MlxArray h;
  final MlxArray c;

  void close() {
    h.close();
    c.close();
  }
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

/// Silero VAD v6 runtime — evaluates one audio frame at a time.
///
/// Architecture (16 kHz, from official silero_vad_16k.safetensors):
///   Reflect-pad 64 on right → unsqueeze → STFT conv → magnitude
///   → conv1(129→128, s=1) + ReLU
///   → conv2(128→64, s=2)  + ReLU
///   → conv3(64→64, s=2)   + ReLU
///   → conv4(64→128, s=1)  + ReLU
///   → squeeze(-1)
///   → LSTMCell(128, 128)
///   → ReLU → Conv1d(128→1, k=1) → Sigmoid → mean
///   → speech probability [0.0, 1.0]
final class SileroVadRuntime {
  SileroVadRuntime(SileroVadBundle bundle)
    : _manifest = bundle.manifest,
      _stftBasis = _require(bundle.tensors, 'stft_conv.weight'),
      _conv1 = VadConv1d.load(bundle.tensors, 'conv1', padding: 1),
      _conv2 = VadConv1d.load(bundle.tensors, 'conv2', stride: 2, padding: 1),
      _conv3 = VadConv1d.load(bundle.tensors, 'conv3', stride: 2, padding: 1),
      _conv4 = VadConv1d.load(bundle.tensors, 'conv4', padding: 1),
      _lstm = VadLstmCell.load(bundle.tensors, 'lstm_cell', hiddenSize: 128),
      _finalConvW = _require(bundle.tensors, 'final_conv.weight'),
      _finalConvB = _require(bundle.tensors, 'final_conv.bias');

  final SileroVadManifest _manifest;
  final MlxArray _stftBasis; // (258, 1, 256)
  final VadConv1d _conv1; // (129→128, k=3, s=1, p=1)
  final VadConv1d _conv2; // (128→64, k=3, s=2, p=1)
  final VadConv1d _conv3; // (64→64, k=3, s=2, p=1)
  final VadConv1d _conv4; // (64→128, k=3, s=1, p=1)
  final VadLstmCell _lstm;
  final MlxArray _finalConvW; // (1, 128, 1)
  final MlxArray _finalConvB; // (1,)

  /// Number of audio samples per frame (window + context).
  int get frameSamples => _manifest.windowSamples + _manifest.contextSamples;

  /// Sample rate.
  int get sampleRate => _manifest.sampleRate;

  /// Hidden size for LSTM state.
  int get hiddenSize => _manifest.hiddenSize;

  /// Create a fresh zero state for a new audio stream.
  SileroVadState createState({int batch = 1}) {
    return SileroVadState(
      h: MlxArray.zeros([batch, hiddenSize]),
      c: MlxArray.zeros([batch, hiddenSize]),
    );
  }

  /// Reset to re-use. Caller must close the returned old state if desired.
  SileroVadState resetState({int batch = 1}) => createState(batch: batch);

  /// Run one frame through the model.
  ///
  /// [samples] must contain exactly [frameSamples] float32 samples.
  /// Returns the speech probability and the updated LSTM state.
  /// The caller must close the old [state] after adopting the new one.
  ({double probability, SileroVadState state}) processFrame({
    required Float32List samples,
    required SileroVadState state,
  }) {
    assert(
      samples.length == frameSamples,
      'Expected $frameSamples samples, got ${samples.length}',
    );

    // Build input: (1, frameSamples).
    final inputArr = MlxArray.fromFloat32List(
      samples,
      shape: [1, frameSamples],
    );

    try {
      // --- STFT ---
      final magnitude = _stft(inputArr);

      // --- Encoder ---
      var x = magnitude;
      x = _encBlock(_conv1, x);
      x = _encBlock(_conv2, x); // stride=2: T 4→2
      x = _encBlock(_conv3, x); // stride=2: T 2→1
      x = _encBlock(_conv4, x); // T stays 1
      // x: (1, 128, 1)

      // --- squeeze(-1) → (1, 128) for LSTMCell ---
      final squeezed = x.reshape([x.shape[0], x.shape[1]]);
      x.close();

      // --- LSTMCell (single step) ---
      final lstmResult = _lstm.call(xt: squeezed, h: state.h, c: state.c);

      squeezed.close();

      // h: (batch, hidden) → ReLU → Conv1d(128, 1, 1) → Sigmoid
      // Expand h to (1, 128, 1) for final_conv.
      final hExpand = lstmResult.h.reshape([1, hiddenSize, 1]);
      final hRelu = vadRelu(hExpand);
      hExpand.close();

      // Manual conv1d(k=1): output = wFlat @ xFlat + bias.
      final wFlat = _finalConvW.reshape([1, hiddenSize]);
      final xFlat = hRelu.reshape([hiddenSize, 1]);
      hRelu.close();
      final convOut = mx.matmul(wFlat, xFlat); // (1, 1)
      wFlat.close();
      xFlat.close();

      // Add bias and sigmoid.
      final biased = mx.add(convOut, _finalConvB);
      convOut.close();
      final sigOut = biased.sigmoid();
      biased.close();

      // Mean over output → scalar probability.
      final probArr = sigOut.mean();
      sigOut.close();

      probArr.eval();
      final prob = probArr.toFloat32List()[0];
      probArr.close();

      final newState = SileroVadState(h: lstmResult.h, c: lstmResult.c);
      return (probability: prob.clamp(0.0, 1.0), state: newState);
    } finally {
      inputArr.close();
    }
  }

  // -------------------------------------------------------------------------
  // STFT: reflect-pad 64 right → conv with basis → magnitude
  // -------------------------------------------------------------------------

  MlxArray _stft(MlxArray input) {
    // input: (1, frameSamples=576)
    // v6: reflect pad only 64 on the RIGHT (matching tinygrad reference).
    final padded = input.pad(
      axes: [1],
      lowPads: [0],
      highPads: [_manifest.contextSamples], // 64
      mode: 'reflect',
    );

    // Unsqueeze to (1, 1, padded_len) for 1D conv.
    final x3d = padded.reshape([1, 1, padded.shape[1]]);
    padded.close();

    // STFT conv: basis is (258, 1, 256), stride = hop_length=128.
    // MLX conv1d: input (N, T, C_in), weight (C_out, kW, C_in).
    final xNtc = x3d.transposeAxes([0, 2, 1]);
    x3d.close();

    // Basis: (258, 1, 256) PyTorch → (258, 256, 1) MLX.
    final basisMlx = _stftBasis.transposeAxes([0, 2, 1]);
    final convResult = mx.conv1d(xNtc, basisMlx, stride: _manifest.hopLength);
    xNtc.close();
    basisMlx.close();

    // convResult: (1, T_frames, 258) → back to (1, 258, T_frames).
    final spec = convResult.transposeAxes([0, 2, 1]);
    convResult.close();

    // Split: real = [0:129], imag = [129:258] along dim 1.
    final nFreq = _manifest.nFft ~/ 2 + 1; // 129
    final real = spec.slice(start: [0, 0, 0], stop: [1, nFreq, spec.shape[2]]);
    final imag = spec.slice(
      start: [0, nFreq, 0],
      stop: [1, nFreq * 2, spec.shape[2]],
    );
    spec.close();

    // Magnitude: sqrt(real² + imag²).
    final r2 = real.square();
    real.close();
    final i2 = imag.square();
    imag.close();
    final sum = mx.add(r2, i2);
    r2.close();
    i2.close();
    final mag = mx.sqrt(sum);
    sum.close();

    return mag; // (1, 129, T_frames=4)
  }

  // -------------------------------------------------------------------------
  // Encoder block
  // -------------------------------------------------------------------------

  MlxArray _encBlock(VadConv1d conv, MlxArray input) {
    final out = conv.call(input);
    if (!identical(out, input)) input.close();
    final activated = vadRelu(out);
    out.close();
    return activated;
  }

  static MlxArray _require(Map<String, MlxArray> t, String key) {
    final v = t[key];
    if (v == null) throw StateError('Missing Silero VAD tensor: $key');
    return v;
  }
}
