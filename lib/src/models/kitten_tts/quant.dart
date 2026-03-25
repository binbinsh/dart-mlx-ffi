library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

MlxArray fakeQuantDynamicU8(MlxArray x) {
  final xF = x.astype(MlxDType.MLX_FLOAT32);
  final sorted = xF.flatten().sort();
  final zero = MlxArray.full([], 0.0);
  final xMin = mx.minimum(
    sorted.slice(start: [0], stop: [1]).reshape([]),
    zero,
  );
  final xMax = mx.maximum(
    sorted
        .slice(start: [sorted.shape[0] - 1], stop: [sorted.shape[0]])
        .reshape([]),
    zero,
  );
  final scale = (xMax - xMin) / MlxArray.full([], 255.0);
  final scaleZeroMask = scale.equal(MlxArray.full([], 0.0));
  final scaleSafe = mx.where(scaleZeroMask, MlxArray.full([], 1.0), scale);
  final zeroPoint = mx.clip((-xMin / scaleSafe).round(), min: 0.0, max: 255.0);
  final q = mx.clip((xF / scaleSafe + zeroPoint).round(), min: 0.0, max: 255.0);
  final deq = (q - zeroPoint) * scaleSafe;
  final zerosLike = deq.zerosLike();
  try {
    return mx.where(scaleZeroMask, zerosLike, deq);
  } finally {
    zerosLike.close();
    deq.close();
    q.close();
    zeroPoint.close();
    scaleSafe.close();
    scaleZeroMask.close();
    scale.close();
    xMax.close();
    xMin.close();
    zero.close();
    sorted.close();
    xF.close();
  }
}

MlxArray maybeFakeQuant(MlxArray x, bool enabled) =>
    enabled ? fakeQuantDynamicU8(x) : x;
