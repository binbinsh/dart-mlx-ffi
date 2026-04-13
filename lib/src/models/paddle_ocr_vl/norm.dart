part of 'paddle_ocr_vl.dart';

MlxArray _lmRmsNorm(
  MlxArray input, {
  required MlxArray weight,
  required double eps,
}) {
  final x = input.astype(MlxDType.MLX_FLOAT32);
  final sq = MlxMore.square(x);
  final meanSq = mx.mean(sq, axis: x.ndim - 1, keepDims: true);
  sq.close();
  final epsArr = MlxArray.full([], eps, dtype: MlxDType.MLX_FLOAT32);
  final denom = MlxMore.sqrt(meanSq + epsArr);
  meanSq.close();
  epsArr.close();
  final unit = x / denom;
  x.close();
  denom.close();
  final weightShape = List<int>.filled(input.ndim, 1)
    ..[input.ndim - 1] = input.shape.last;
  final w = weight.astype(MlxDType.MLX_FLOAT32).reshape(weightShape);
  final out = unit * w;
  unit.close();
  w.close();
  return out;
}

MlxArray _lmRmsNormCompat(
  MlxArray input, {
  required MlxArray weight,
  required double eps,
}) {
  final seqLen = input.shape.length >= 2 ? input.shape[1] : 1;
  if (seqLen == 1) {
    return mx.fast.rmsNorm(
      input,
      weight: weight,
      eps: eps,
    );
  }
  return _lmRmsNorm(
    input,
    weight: weight,
    eps: eps,
  );
}
