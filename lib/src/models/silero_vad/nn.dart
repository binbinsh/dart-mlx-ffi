import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

/// Retrieve a required tensor from the weight map.
MlxArray _require(Map<String, MlxArray> t, String key) {
  final v = t[key];
  if (v == null) throw StateError('Missing Silero VAD tensor: $key');
  return v;
}

// ---------------------------------------------------------------------------
// Conv1d — PyTorch-compatible wrapper around MLX conv1d
// ---------------------------------------------------------------------------

final class VadConv1d {
  VadConv1d({
    required this.weight,
    required this.bias,
    required this.stride,
    required this.padding,
  });

  final MlxArray weight;
  final MlxArray? bias;
  final int stride;
  final int padding;

  factory VadConv1d.load(
    Map<String, MlxArray> t,
    String prefix, {
    int stride = 1,
    int padding = 0,
  }) {
    return VadConv1d(
      weight: _require(t, '$prefix.weight'),
      bias: t['$prefix.bias'],
      stride: stride,
      padding: padding,
    );
  }

  /// Input: (batch, channels_in, time).
  /// Output: (batch, channels_out, time_out).
  MlxArray call(MlxArray input) {
    // MLX conv1d expects (N, T, C); PyTorch uses (N, C, T).
    var x = input.transposeAxes([0, 2, 1]);
    // PyTorch weight: (C_out, C_in, kW) → MLX: (C_out, kW, C_in).
    final w = weight.transposeAxes([0, 2, 1]);
    try {
      x = mx.conv1d(x, w, stride: stride, padding: padding);
      if (bias != null) {
        final b = bias!.reshape([1, 1, bias!.shape[0]]);
        final added = mx.add(x, b);
        b.close();
        x.close();
        x = added;
      }
      // Back to (N, C, T).
      final out = x.transposeAxes([0, 2, 1]);
      x.close();
      return out;
    } finally {
      w.close();
    }
  }
}

// ---------------------------------------------------------------------------
// LSTM Cell (matches Silero VAD state shape)
// ---------------------------------------------------------------------------

/// A single-layer LSTM cell for Silero VAD.
///
/// Silero VAD carries hidden state as `(2, batch, hidden)` where index 0
/// is `h` and index 1 is `c`. Each call processes the full time sequence
/// and returns the final `(h, c)`.
final class VadLstmCell {
  VadLstmCell({
    required this.inputWeight,
    required this.hiddenWeight,
    required this.inputBias,
    required this.hiddenBias,
    required this.hiddenSize,
  }) : _iwT = inputWeight.T,
       _hwT = hiddenWeight.T;

  final MlxArray inputWeight;
  final MlxArray hiddenWeight;
  final MlxArray inputBias;
  final MlxArray hiddenBias;
  final int hiddenSize;
  final MlxArray _iwT;
  final MlxArray _hwT;

  factory VadLstmCell.load(
    Map<String, MlxArray> t,
    String prefix, {
    required int hiddenSize,
  }) {
    return VadLstmCell(
      inputWeight: _require(t, '$prefix.weight_ih'),
      hiddenWeight: _require(t, '$prefix.weight_hh'),
      inputBias: _require(t, '$prefix.bias_ih'),
      hiddenBias: _require(t, '$prefix.bias_hh'),
      hiddenSize: hiddenSize,
    );
  }

  /// Run LSTM over a time-sequence input `(batch, time, features)`.
  /// Returns the final hidden and cell states, each `(batch, hidden)`.
  ({MlxArray h, MlxArray c}) call({
    required MlxArray input,
    required MlxArray h0,
    required MlxArray c0,
  }) {
    final batch = input.shape[0];
    final timeSteps = input.shape[1];
    var h = h0;
    var c = c0;
    for (var t = 0; t < timeSteps; t++) {
      // Extract timestep: (batch, features).
      final xt = input
          .slice(start: [0, t, 0], stop: [batch, t + 1, input.shape[2]])
          .reshape([batch, input.shape[2]]);
      // gates = xt @ W_ih^T + bias_ih + h @ W_hh^T + bias_hh
      final gates1 = mx.addmm(inputBias, xt, _iwT);
      xt.close();
      final gates = mx.addmm(mx.add(gates1, hiddenBias), h, _hwT);
      gates1.close();

      final iGate = gates
          .slice(start: [0, 0], stop: [batch, hiddenSize])
          .sigmoid();
      final fGate = gates
          .slice(start: [0, hiddenSize], stop: [batch, hiddenSize * 2])
          .sigmoid();
      final gGate = gates
          .slice(start: [0, hiddenSize * 2], stop: [batch, hiddenSize * 3])
          .tanh();
      final oGate = gates
          .slice(start: [0, hiddenSize * 3], stop: [batch, hiddenSize * 4])
          .sigmoid();
      gates.close();

      final newC = mx.add(mx.multiply(fGate, c), mx.multiply(iGate, gGate));
      final newH = mx.multiply(oGate, newC.tanh());
      iGate.close();
      fGate.close();
      gGate.close();
      oGate.close();
      if (!identical(h, h0)) h.close();
      if (!identical(c, c0)) c.close();
      h = newH;
      c = newC;
    }
    return (h: h, c: c);
  }
}

// ---------------------------------------------------------------------------
// ReLU helper
// ---------------------------------------------------------------------------

MlxArray vadRelu(MlxArray input) {
  final zero = input.zerosLike();
  try {
    return mx.maximum(input, zero);
  } finally {
    zero.close();
  }
}
