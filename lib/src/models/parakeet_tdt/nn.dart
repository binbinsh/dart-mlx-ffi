import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

MlxArray requireParakeetTensor(Map<String, MlxArray> tensors, String key) {
  final tensor = tensors[key];
  if (tensor == null) {
    throw StateError('Missing Parakeet tensor: $key');
  }
  return tensor;
}

final class ParakeetDenseLinear {
  ParakeetDenseLinear(this.weight, this.bias) : _weightT = weight.T;

  final MlxArray weight;
  final MlxArray? bias;
  final MlxArray _weightT;

  factory ParakeetDenseLinear.load(
    Map<String, MlxArray> tensors,
    String prefix,
  ) {
    return ParakeetDenseLinear(
      requireParakeetTensor(tensors, '$prefix.weight'),
      tensors['$prefix.bias'],
    );
  }

  MlxArray call(MlxArray input) {
    final y = mx.matmul(input, _weightT);
    if (bias == null) {
      return y;
    }
    final reshapedBias = _broadcastVector(bias!, y.shape.length);
    try {
      return mx.add(y, reshapedBias);
    } finally {
      reshapedBias.close();
      y.close();
    }
  }
}

final class ParakeetEmbedding {
  const ParakeetEmbedding(this.weight);

  final MlxArray weight;

  factory ParakeetEmbedding.load(Map<String, MlxArray> tensors, String prefix) {
    return ParakeetEmbedding(requireParakeetTensor(tensors, '$prefix.weight'));
  }

  MlxArray call(MlxArray tokenIds) => mx.take(weight, tokenIds, axis: 0);
}

final class ParakeetLayerNorm {
  const ParakeetLayerNorm(this.weight, this.bias, {this.eps = 1e-5});

  final MlxArray weight;
  final MlxArray bias;
  final double eps;

  factory ParakeetLayerNorm.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    double eps = 1e-5,
  }) {
    return ParakeetLayerNorm(
      requireParakeetTensor(tensors, '$prefix.weight'),
      requireParakeetTensor(tensors, '$prefix.bias'),
      eps: eps,
    );
  }

  MlxArray call(MlxArray input) =>
      mx.fast.layerNorm(input, weight: weight, bias: bias, eps: eps);
}

final class ParakeetBatchNorm {
  const ParakeetBatchNorm({
    required this.weight,
    required this.bias,
    required this.runningMean,
    required this.runningVar,
    this.eps = 1e-5,
  });

  final MlxArray weight;
  final MlxArray bias;
  final MlxArray runningMean;
  final MlxArray runningVar;
  final double eps;

  factory ParakeetBatchNorm.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    double eps = 1e-5,
  }) {
    return ParakeetBatchNorm(
      weight: requireParakeetTensor(tensors, '$prefix.weight'),
      bias: requireParakeetTensor(tensors, '$prefix.bias'),
      runningMean: requireParakeetTensor(tensors, '$prefix.running_mean'),
      runningVar: requireParakeetTensor(tensors, '$prefix.running_var'),
      eps: eps,
    );
  }

  MlxArray call(MlxArray input) {
    final ndim = input.shape.length;
    final mean = _broadcastVector(runningMean, ndim);
    final variance = _broadcastVector(runningVar, ndim);
    final scale = _broadcastVector(weight, ndim);
    final shift = _broadcastVector(bias, ndim);
    final epsArray = MlxArray.full([], eps);
    try {
      final centered = mx.subtract(input, mean);
      final denom = mx.sqrt(mx.add(variance, epsArray));
      final normalized = mx.divide(centered, denom);
      final scaled = mx.multiply(normalized, scale);
      try {
        return mx.add(scaled, shift);
      } finally {
        scaled.close();
        normalized.close();
        denom.close();
        centered.close();
      }
    } finally {
      epsArray.close();
      mean.close();
      variance.close();
      scale.close();
      shift.close();
    }
  }
}

final class ParakeetLstmCell {
  ParakeetLstmCell({
    required this.wx,
    required this.wh,
    required this.bias,
    required this.hiddenSize,
  }) : _wxT = wx.T,
       _whT = wh.T;

  final MlxArray wx;
  final MlxArray wh;
  final MlxArray bias;
  final int hiddenSize;
  final MlxArray _wxT;
  final MlxArray _whT;

  factory ParakeetLstmCell.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int hiddenSize,
  }) {
    return ParakeetLstmCell(
      wx: requireParakeetTensor(tensors, '$prefix.Wx'),
      wh: requireParakeetTensor(tensors, '$prefix.Wh'),
      bias: requireParakeetTensor(tensors, '$prefix.bias'),
      hiddenSize: hiddenSize,
    );
  }

  ({MlxArray hidden, MlxArray cell}) call({
    required MlxArray input,
    required MlxArray hidden,
    required MlxArray cell,
  }) {
    final projected = mx.addmm(bias, input, _wxT);
    final recurrent = mx.addmm(projected, hidden, _whT);
    projected.close();

    final i = recurrent.slice(start: <int>[0, 0], stop: <int>[1, hiddenSize]);
    final f = recurrent.slice(
      start: <int>[0, hiddenSize],
      stop: <int>[1, hiddenSize * 2],
    );
    final g = recurrent.slice(
      start: <int>[0, hiddenSize * 2],
      stop: <int>[1, hiddenSize * 3],
    );
    final o = recurrent.slice(
      start: <int>[0, hiddenSize * 3],
      stop: <int>[1, hiddenSize * 4],
    );
    recurrent.close();

    final inputGate = i.sigmoid();
    final forgetGate = f.sigmoid();
    final candidate = g.tanh();
    final outputGate = o.sigmoid();
    i.close();
    f.close();
    g.close();
    o.close();

    final nextCell = mx.add(
      mx.multiply(forgetGate, cell),
      mx.multiply(inputGate, candidate),
    );
    final nextHidden = mx.multiply(outputGate, nextCell.tanh());
    forgetGate.close();
    inputGate.close();
    candidate.close();
    outputGate.close();
    return (hidden: nextHidden, cell: nextCell);
  }
}

MlxArray parakeetRelu(MlxArray input) {
  final zero = input.zerosLike();
  try {
    return mx.maximum(input, zero);
  } finally {
    zero.close();
  }
}

MlxArray parakeetSilu(MlxArray input) {
  final sigmoid = input.sigmoid();
  try {
    return mx.multiply(input, sigmoid);
  } finally {
    sigmoid.close();
  }
}

MlxArray parakeetGlu(MlxArray input) {
  final channels = input.shape.last;
  if (channels.isOdd) {
    throw StateError('GLU input channel dimension must be even.');
  }
  final split = channels ~/ 2;
  final left = input.slice(
    start: List<int>.filled(input.shape.length, 0),
    stop: <int>[...input.shape.take(input.shape.length - 1), split],
  );
  final right = input.slice(
    start: <int>[...List<int>.filled(input.shape.length - 1, 0), split],
    stop: input.shape,
  );
  final gate = right.sigmoid();
  try {
    return mx.multiply(left, gate);
  } finally {
    left.close();
    right.close();
    gate.close();
  }
}

MlxArray _broadcastVector(MlxArray vector, int ndim) {
  final shape = List<int>.filled(math.max(1, ndim), 1);
  shape[shape.length - 1] = vector.shape.last;
  return vector.reshape(shape);
}
