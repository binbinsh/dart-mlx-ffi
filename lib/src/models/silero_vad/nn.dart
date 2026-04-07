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
// LSTM Cell — single-step for Silero VAD v6
// ---------------------------------------------------------------------------

/// A single-step LSTM cell for Silero VAD v6.
///
/// Silero VAD v6 collapses encoder output to a single time step via strided
/// convolutions, so only one LSTM cell step is needed per audio frame.
/// State is `(h, c)` each with shape `(batch, hidden)`.
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

  /// Run a single LSTM step. `xt` is `(batch, features)`.
  /// Returns the new `(h, c)`, each `(batch, hidden)`.
  ({MlxArray h, MlxArray c}) call({
    required MlxArray xt,
    required MlxArray h,
    required MlxArray c,
  }) {
    // gates = xt @ W_ih^T + bias_ih + h @ W_hh^T + bias_hh
    final gates1 = mx.addmm(inputBias, xt, _iwT);
    final gates = mx.addmm(mx.add(gates1, hiddenBias), h, _hwT);
    gates1.close();

    final batch = xt.shape[0];
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

    return (h: newH, c: newC);
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
