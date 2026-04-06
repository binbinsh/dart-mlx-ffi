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

/// Silero VAD v5 runtime — evaluates one audio frame at a time.
///
/// Architecture (16 kHz branch):
///   STFT (conv with pre-computed basis) → magnitude
///   → 4× (Conv1d + ReLU)
///   → LSTM(128,128) → ReLU → Conv1d(128→1,k=1) → Sigmoid → mean
///   → speech probability [0.0, 1.0]
final class SileroVadRuntime {
  SileroVadRuntime(SileroVadBundle bundle)
    : _manifest = bundle.manifest,
      _stftBasis = _require(bundle.tensors, 'stft.forward_basis_buffer'),
      _enc0 = VadConv1d.load(
        bundle.tensors,
        'encoder.0.reparam_conv',
        padding: 1,
      ),
      _enc1 = VadConv1d.load(
        bundle.tensors,
        'encoder.1.reparam_conv',
        padding: 1,
      ),
      _enc2 = VadConv1d.load(
        bundle.tensors,
        'encoder.2.reparam_conv',
        padding: 1,
      ),
      _enc3 = VadConv1d.load(
        bundle.tensors,
        'encoder.3.reparam_conv',
        padding: 1,
      ),
      _lstm = VadLstmCell.load(bundle.tensors, 'decoder.rnn', hiddenSize: 128),
      _decConvW = _require(bundle.tensors, 'decoder.decoder.2.weight'),
      _decConvB = _require(bundle.tensors, 'decoder.decoder.2.bias');

  final SileroVadManifest _manifest;
  final MlxArray _stftBasis; // (258, 1, 256)
  final VadConv1d _enc0;
  final VadConv1d _enc1;
  final VadConv1d _enc2;
  final VadConv1d _enc3;
  final VadLstmCell _lstm;
  final MlxArray _decConvW; // (1, 128, 1)
  final MlxArray _decConvB; // (1,)

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
      x = _encBlock(_enc0, x);
      x = _encBlock(_enc1, x);
      x = _encBlock(_enc2, x);
      x = _encBlock(_enc3, x);
      // x: (1, 128, T')

      // --- Decoder ---
      // Squeeze time dim if T' == 1.
      final encShape = x.shape;
      MlxArray decInput;
      if (encShape.length == 3 && encShape[2] == 1) {
        // (1, 128, 1) → (1, 128)
        decInput = x.reshape([encShape[0], encShape[1]]);
        x.close();
      } else {
        decInput = x;
      }

      // LSTM expects (batch, time, features). Add time dim if needed.
      MlxArray lstmIn;
      if (decInput.shape.length == 2) {
        lstmIn = decInput.reshape([decInput.shape[0], 1, decInput.shape[1]]);
        decInput.close();
      } else {
        lstmIn = decInput;
      }

      final lstmResult = _lstm.call(input: lstmIn, h0: state.h, c0: state.c);
      lstmIn.close();

      // h: (batch, hidden) → ReLU → Conv1d(128, 1, 1) → Sigmoid
      final hRelu = vadRelu(lstmResult.h);

      // Conv1d expects (N, C, T) — expand h to (1, 128, 1).
      final hExpand = hRelu.reshape([hRelu.shape[0], hRelu.shape[1], 1]);
      hRelu.close();

      // Manual conv1d(1): matmul-style since kernel=1.
      // decConvW: (1, 128, 1) — squeeze to (128, 1) → hExpand (1,128,1)
      // Simpler: decConvW is (C_out=1, C_in=128, k=1).
      // For k=1 conv: output[b, co, t] = sum_ci(w[co, ci, 0] * x[b, ci, t])
      // = (w reshaped as (1, 128)) @ (x reshaped as (128, T)) → (1, T)
      final wFlat = _decConvW.reshape([1, hiddenSize]);
      final xFlat = hExpand.reshape([hiddenSize, 1]);
      hExpand.close();
      final convOut = mx.matmul(wFlat, xFlat); // (1, 1)
      wFlat.close();
      xFlat.close();

      // Add bias and sigmoid.
      final biased = mx.add(convOut, _decConvB);
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
  // STFT: reflected-pad → conv with basis → split real/imag → magnitude
  // -------------------------------------------------------------------------

  MlxArray _stft(MlxArray input) {
    // input: (1, frameSamples)
    final nFft = _manifest.nFft; // 256
    final halfFft = nFft ~/ 2; // 128

    // Reflect-pad: pad left and right by n_fft // 2.
    final padded = input.pad(
      axes: [1],
      lowPads: [halfFft],
      highPads: [halfFft],
      mode: 'reflect',
    );

    // Unsqueeze to (1, 1, padded_len) for 1D conv.
    final x3d = padded.reshape([1, 1, padded.shape[1]]);
    padded.close();

    // STFT conv: basis is (258, 1, 256), stride = hop_length.
    // MLX conv1d: input (N, T, C_in), weight (C_out, kW, C_in).
    // Input: (1, 1, L) → transpose to (1, L, 1).
    final xNtc = x3d.transposeAxes([0, 2, 1]);
    x3d.close();

    // Basis: (258, 1, 256). PyTorch conv weight is (C_out, C_in, kW).
    // → MLX needs (C_out, kW, C_in). Transpose dims 1↔2.
    final basisMlx = _stftBasis.transposeAxes([0, 2, 1]);
    final convResult = mx.conv1d(xNtc, basisMlx, stride: _manifest.hopLength);
    xNtc.close();
    basisMlx.close();

    // convResult: (1, T_frames, 258) → back to (1, 258, T_frames).
    final spec = convResult.transposeAxes([0, 2, 1]);
    convResult.close();

    // Split: real = [0:129], imag = [129:258] along dim 1.
    final nFreq = halfFft + 1; // 129
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

    return mag; // (1, 129, T_frames)
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
