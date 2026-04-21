/// Pyannote/segmentation-3.0 post-SincNet network: 4-layer bidirectional
/// LSTM, two Linear+leaky_relu hidden layers, and a final Linear classifier.
///
/// Weight key convention (PyTorch, as stored in
/// `pyannote/segmentation-3.0/pytorch_model.bin`):
///
/// ```
///   lstm.weight_ih_l{0..3}[_reverse]    (4*hidden, input_size) float32
///   lstm.weight_hh_l{0..3}[_reverse]    (4*hidden, hidden)     float32
///   lstm.bias_ih_l{0..3}[_reverse]      (4*hidden,)            float32
///   lstm.bias_hh_l{0..3}[_reverse]      (4*hidden,)            float32
///   linear.{0,1}.weight                 (out, in)              float32
///   linear.{0,1}.bias                   (out,)                 float32
///   classifier.weight                   (num_classes, hidden)  float32
///   classifier.bias                     (num_classes,)         float32
/// ```
///
/// Gate order in PyTorch LSTM: `[i, f, g, o]`.  The reused
/// `kitten_tts/lstm.dart` [LSTM] runner applies the same convention, so we can
/// feed the PyTorch tensors directly without reordering.
library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import '../kitten_tts/lstm.dart' show LSTM, LstmResult;
import 'bundle.dart';

/// A 4-layer bidirectional LSTM stack over the PyanNet convention.
final class PyannoteBiLstmStack {
  PyannoteBiLstmStack._(this.layers);

  /// One `LSTM` per layer (each internally runs forward + backward).
  final List<LSTM> layers;

  factory PyannoteBiLstmStack.fromBundle(PyannoteSegBundle bundle) {
    final p = bundle.manifest.lstm;
    if (!p.bidirectional) {
      throw StateError('pyannote-seg expects a bidirectional LSTM');
    }
    final layers = <LSTM>[];
    for (var l = 0; l < p.numLayers; l++) {
      final layerInput = l == 0 ? p.inputSize : p.hiddenSize * 2;
      layers.add(
        LSTM(
          inputSize: layerInput,
          hiddenSize: p.hiddenSize,
          wxForward: bundle.require('lstm.weight_ih_l$l'),
          whForward: bundle.require('lstm.weight_hh_l$l'),
          biasIhForward: bundle.require('lstm.bias_ih_l$l'),
          biasHhForward: bundle.require('lstm.bias_hh_l$l'),
          wxBackward: bundle.require('lstm.weight_ih_l${l}_reverse'),
          whBackward: bundle.require('lstm.weight_hh_l${l}_reverse'),
          biasIhBackward: bundle.require('lstm.bias_ih_l${l}_reverse'),
          biasHhBackward: bundle.require('lstm.bias_hh_l${l}_reverse'),
        ),
      );
    }
    return PyannoteBiLstmStack._(layers);
  }

  /// Forward a `(B, T, input_size)` tensor through the full 4-layer stack.
  /// Returns `(B, T, 2 * hidden)`.
  MlxArray call(MlxArray input) {
    var current = input;
    for (var i = 0; i < layers.length; i++) {
      final LstmResult res = layers[i](current);
      if (!identical(current, input)) {
        current.close();
      }
      // Discard final hidden/cell states (we only need the sequence output).
      res.forward.hidden.close();
      res.forward.cell.close();
      res.backward.hidden.close();
      res.backward.cell.close();
      current = res.output;
    }
    if (identical(current, input)) {
      // Degenerate case (no layers) — return a copy-like view to keep
      // ownership clear.
      return input.reshape(List<int>.from(input.shape));
    }
    return current;
  }
}

/// A single `nn.Linear(in, out)` with PyTorch key layout.
///
/// Weight stored as `(out, in)`; applied as `matmul(x, W^T) + b`.
final class PyannoteLinear {
  PyannoteLinear({required this.weight, required this.bias})
      : _wT = weight.T;

  final MlxArray weight;
  final MlxArray bias;
  final MlxArray _wT;

  factory PyannoteLinear.load(PyannoteSegBundle bundle, String prefix) {
    return PyannoteLinear(
      weight: bundle.require('$prefix.weight'),
      bias: bundle.require('$prefix.bias'),
    );
  }

  /// Apply to an input whose last axis matches `weight.shape[1]`.
  ///
  /// Supports 2-D `(B, in)` and 3-D `(B, T, in)` inputs by using a broadcast
  /// matmul + elementwise bias add.
  MlxArray call(MlxArray input) {
    final projected = mx.matmul(input, _wT);
    try {
      return mx.add(projected, bias);
    } finally {
      projected.close();
    }
  }

  void close() {
    _wT.close();
  }
}
