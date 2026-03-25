part of 'qwen3_5.dart';

extension on Qwen3_5Runner {
  ({MlxArray q, MlxArray k}) _applyMrope(
    MlxArray q,
    MlxArray k,
    int seqLen,
  ) {
    if (Platform.environment['QWEN35_SIMPLE_ROPE'] == '1') {
      return (
        q: mx.fast.rope(
          q,
          dims: config.rotaryDims,
          traditional: false,
          base: config.ropeTheta,
        ),
        k: mx.fast.rope(
          k,
          dims: config.rotaryDims,
          traditional: false,
          base: config.ropeTheta,
        ),
      );
    }
    final rotaryDim = config.rotaryDims;
    final invFreq = MlxArray.fromFloat32List([
      for (var i = 0; i < rotaryDim; i += 2)
        math.exp(-(i / rotaryDim) * math.log(config.ropeTheta)),
    ], shape: [rotaryDim ~/ 2]);
    try {
      final positionIds = MlxArray.fromFloat32List([
        for (var i = 0; i < seqLen; i++) i.toDouble(),
      ], shape: [seqLen]).reshape([
        1,
        seqLen,
      ]);
      try {
        final freqs = _freqRow(invFreq, positionIds);
        try {
          final emb = mx.concatenate([freqs, freqs], axis: -1);
          try {
            final cos = emb.cos().astype(q.dtype);
            final sin = emb.sin().astype(q.dtype);
            try {
              return (
                q: _applyRotary(q, cos, sin),
                k: _applyRotary(k, cos, sin),
              );
            } finally {
              sin.close();
              cos.close();
            }
          } finally {
            emb.close();
          }
        } finally {
          freqs.close();
        }
      } finally {
        positionIds.close();
      }
    } finally {
      invFreq.close();
    }
  }

  MlxArray _freqRow(MlxArray invFreq, MlxArray positionIds) {
    final inv = invFreq.reshape([1, 1, invFreq.shape[0]]);
    final pos = positionIds.reshape([1, positionIds.shape[1], 1]);
    final freq = pos * inv;
    inv.close();
    pos.close();
    return freq;
  }

  MlxArray _applyRotary(MlxArray input, MlxArray cos, MlxArray sin) {
    final rotaryDim = cos.shape[2];
    final rot = input.slice(
      start: [0, 0, 0, 0],
      stop: [input.shape[0], input.shape[1], input.shape[2], rotaryDim],
    );
    final pass = input.slice(
      start: [0, 0, 0, rotaryDim],
      stop: [input.shape[0], input.shape[1], input.shape[2], input.shape[3]],
    );
    try {
      final rotated = _rotateHalf(rot);
      try {
        final cos4 = cos.reshape([1, 1, cos.shape[1], cos.shape[2]]);
        final sin4 = sin.reshape([1, 1, sin.shape[1], sin.shape[2]]);
        try {
          final embed = (rot * cos4) + (rotated * sin4);
          try {
            return mx.concatenate([embed, pass], axis: 3);
          } finally {
            embed.close();
          }
        } finally {
          sin4.close();
          cos4.close();
        }
      } finally {
        rotated.close();
      }
    } finally {
      pass.close();
      rot.close();
    }
  }

  MlxArray _rotateHalf(MlxArray input) {
    final half = input.shape[3] ~/ 2;
    final x1 = input.slice(
      start: [0, 0, 0, 0],
      stop: [input.shape[0], input.shape[1], input.shape[2], half],
    );
    final x2 = input.slice(
      start: [0, 0, 0, half],
      stop: [input.shape[0], input.shape[1], input.shape[2], input.shape[3]],
    );
    try {
      final negX2 = x2.negative();
      try {
        return mx.concatenate([negX2, x1], axis: 3);
      } finally {
        negX2.close();
      }
    } finally {
      x2.close();
      x1.close();
    }
  }
}
