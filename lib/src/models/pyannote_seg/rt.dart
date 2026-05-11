/// Pyannote/segmentation-3.0 full inference runtime in MLX.
///
/// Composition:
///   1. [PyannoteSincNet] frontend       — (B, 1, 160000) → (B, 60, 589)
///   2. Transpose to `(B, T=589, 60)`.
///   3. [PyannoteBiLstmStack] 4-layer bi  — (B, T, 60) → (B, T, 256)
///   4. `linear.0` + leaky_relu          — (B, T, 256) → (B, T, 128)
///   5. `linear.1` + leaky_relu          — (B, T, 128) → (B, T, 128)
///   6. `classifier`                     — (B, T, 128) → (B, T, 7)
///   7. log-softmax over classes → powerset probabilities via exp().
///
/// The public [forward] returns raw logits plus the powerset probabilities
/// (softmax, NOT log-softmax — matches `torch.exp(log_softmax(x))` in the
/// PyanNet reference).
///
/// Staged-parity hooks ([forwardStaged]) capture each intermediate activation
/// for the existing reference fixtures under `test/data/pyannote_seg/`.
library;

import 'dart:typed_data';

import 'package:dart_inference/dart_mlx_ffi.dart';

import 'bundle.dart';
import 'nn.dart';
import 'sincnet.dart';

/// Leaky-ReLU slope used between the 2 post-LSTM linear layers.
const double _linearLeakySlope = 0.01;

MlxArray _leakyRelu(MlxArray x) {
  final zero = MlxArray.full(const <int>[], 0.0);
  final slope = MlxArray.full(const <int>[], _linearLeakySlope);
  try {
    final mask = MlxMore.greater(x, zero);
    final scaled = mx.multiply(x, slope);
    try {
      return mx.where(mask, x, scaled);
    } finally {
      mask.close();
      scaled.close();
    }
  } finally {
    slope.close();
    zero.close();
  }
}

/// A captured intermediate used for parity testing.
class PyannoteForwardTrace {
  const PyannoteForwardTrace({
    required this.sincnet,
    required this.lstm,
    required this.linear0,
    required this.linear1,
    required this.logits,
    required this.logProbs,
    required this.powerset,
  });

  /// `(B, 60, T_out)` frontend output.
  final MlxArray sincnet;

  /// `(B, T_out, 2*hidden)` bidirectional LSTM output.
  final MlxArray lstm;

  /// `(B, T_out, 128)` after `linear.0` + leaky_relu.
  final MlxArray linear0;

  /// `(B, T_out, 128)` after `linear.1` + leaky_relu.
  final MlxArray linear1;

  /// `(B, T_out, num_classes)` raw logits.
  final MlxArray logits;

  /// `(B, T_out, num_classes)` log-softmax (PyanNet's activation).
  final MlxArray logProbs;

  /// `(B, T_out, num_classes)` softmax probabilities.
  final MlxArray powerset;

  void close() {
    sincnet.close();
    lstm.close();
    linear0.close();
    linear1.close();
    logits.close();
    logProbs.close();
    powerset.close();
  }
}

/// The single frame-level prediction for one input window.
class PyannoteSegResult {
  const PyannoteSegResult({
    required this.numFrames,
    required this.numClasses,
    required this.powersetFlat,
  });

  /// Number of output time steps (`589` for the 10 s @ 16 kHz window).
  final int numFrames;

  /// Number of powerset classes (`7` for `max_classes=2, num_speakers=3`).
  final int numClasses;

  /// Softmax output flattened to `[numFrames * numClasses]` in row-major
  /// order (so `powersetFlat[t * numClasses + c]`).
  final Float32List powersetFlat;

  /// Accessor for the probability of powerset class `c` at frame `t`.
  double at(int frame, int cls) => powersetFlat[frame * numClasses + cls];
}

/// Inference runtime for pyannote/segmentation-3.0.
final class PyannoteSegRuntime {
  PyannoteSegRuntime._({
    required this.manifest,
    required this.sincnet,
    required this.lstm,
    required this.linear0,
    required this.linear1,
    required this.classifier,
  });

  final PyannoteSegManifest manifest;
  final PyannoteSincNet sincnet;
  final PyannoteBiLstmStack lstm;
  final PyannoteLinear linear0;
  final PyannoteLinear linear1;
  final PyannoteLinear classifier;

  factory PyannoteSegRuntime.fromBundle(PyannoteSegBundle bundle) {
    return PyannoteSegRuntime._(
      manifest: bundle.manifest,
      sincnet: PyannoteSincNet.fromBundle(bundle),
      lstm: PyannoteBiLstmStack.fromBundle(bundle),
      linear0: PyannoteLinear.load(bundle, 'linear.0'),
      linear1: PyannoteLinear.load(bundle, 'linear.1'),
      classifier: PyannoteLinear.load(bundle, 'classifier'),
    );
  }

  /// Release cached tensors (sinc filterbank + linear transposes).  Tensors
  /// shared with the underlying bundle remain owned by the bundle.
  void close() {
    sincnet.close();
    linear0.close();
    linear1.close();
    classifier.close();
  }

  /// Full forward pass over a mono 10 s waveform.
  ///
  /// Accepts raw `Float32List` PCM (length == `manifest.windowSamples`) or an
  /// already-allocated `MlxArray`.  Returns a [PyannoteSegResult] whose
  /// `powersetFlat` is usable directly from Dart without further evaluation.
  PyannoteSegResult predict(dynamic waveform) {
    final inputOwned = _asWaveformArray(waveform);
    final ownsInput = !identical(inputOwned.input, waveform);

    final trace = _forward(inputOwned.input, captureTrace: false);
    if (ownsInput) {
      inputOwned.input.close();
    }

    // Pull powerset (B, T, C) to Dart — drop batch.
    final batchless = trace.powerset.ndim == 3
        ? trace.powerset.reshape(<int>[
            trace.powerset.shape[1],
            trace.powerset.shape[2],
          ])
        : trace.powerset;
    try {
      MlxRuntime.evalAll([batchless]);
      final flat = batchless.toFloat32List();
      return PyannoteSegResult(
        numFrames: batchless.shape[0],
        numClasses: batchless.shape[1],
        powersetFlat: Float32List.fromList(flat),
      );
    } finally {
      if (!identical(batchless, trace.powerset)) {
        batchless.close();
      }
      trace.close();
    }
  }

  /// Staged forward — useful for parity tests.  Caller owns every tensor in
  /// the returned trace.
  PyannoteForwardTrace forwardStaged(dynamic waveform) {
    final inputOwned = _asWaveformArray(waveform);
    final ownsInput = !identical(inputOwned.input, waveform);
    final trace = _forward(inputOwned.input, captureTrace: true);
    if (ownsInput) {
      inputOwned.input.close();
    }
    return trace;
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  PyannoteForwardTrace _forward(
    MlxArray waveform, {
    required bool captureTrace,
  }) {
    final sincnetOut = sincnet.encode(waveform); // (B, 60, T_out)
    // Transpose to (B, T, 60) for LSTM input.
    final lstmIn = sincnetOut.transposeAxes(<int>[0, 2, 1]);

    final lstmOut = lstm(lstmIn); // (B, T, 256)
    lstmIn.close();

    final linear0Raw = linear0(lstmOut); // (B, T, 128)
    final linear0Act = _leakyRelu(linear0Raw);
    linear0Raw.close();

    final linear1Raw = linear1(linear0Act); // (B, T, 128)
    final linear1Act = _leakyRelu(linear1Raw);
    linear1Raw.close();

    final logits = classifier(linear1Act); // (B, T, 7)
    final logProbs = _logSoftmaxLastAxis(logits);
    final powerset = MlxOps.exp(logProbs);

    if (!captureTrace) {
      // Consolidate ownership: close intermediates we don't return.
      sincnetOut.close();
      lstmOut.close();
      linear0Act.close();
      linear1Act.close();
      logits.close();
      logProbs.close();
      // Return a trace where only `powerset` holds live memory.
      return PyannoteForwardTrace(
        sincnet: MlxArray.zeros(<int>[0]),
        lstm: MlxArray.zeros(<int>[0]),
        linear0: MlxArray.zeros(<int>[0]),
        linear1: MlxArray.zeros(<int>[0]),
        logits: MlxArray.zeros(<int>[0]),
        logProbs: MlxArray.zeros(<int>[0]),
        powerset: powerset,
      );
    }

    return PyannoteForwardTrace(
      sincnet: sincnetOut,
      lstm: lstmOut,
      linear0: linear0Act,
      linear1: linear1Act,
      logits: logits,
      logProbs: logProbs,
      powerset: powerset,
    );
  }

  /// Log-softmax over the last axis: `x - logsumexp(x, axis=-1, keepdims=True)`.
  MlxArray _logSoftmaxLastAxis(MlxArray x) {
    final axis = x.ndim - 1;
    // Numerically stable: subtract max, exp, log(sum), add max back is
    // equivalent to logsumexp.
    final softmax = MlxOps.softmax(x, axis: axis);
    try {
      return MlxOps.log(softmax);
    } finally {
      softmax.close();
    }
  }

  ({MlxArray input}) _asWaveformArray(dynamic waveform) {
    if (waveform is MlxArray) {
      return (input: waveform);
    }
    if (waveform is Float32List) {
      if (waveform.length != manifest.windowSamples) {
        throw StateError(
          'pyannote-seg expects exactly ${manifest.windowSamples} samples '
          '(got ${waveform.length}).',
        );
      }
      final arr = MlxArray.fromFloat32List(
        waveform,
        shape: <int>[1, 1, waveform.length],
      );
      return (input: arr);
    }
    if (waveform is List<double>) {
      final f32 = Float32List.fromList(waveform);
      return _asWaveformArray(f32);
    }
    throw StateError(
      'Unsupported waveform type ${waveform.runtimeType}; '
      'expected MlxArray, Float32List, or List<double>.',
    );
  }
}
