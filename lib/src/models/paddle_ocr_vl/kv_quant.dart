part of 'paddle_ocr_vl.dart';

final class _QuantizedKvSlice {
  _QuantizedKvSlice({required this.keys, required this.values});

  final MlxQuantizedMatrix keys;
  final MlxQuantizedMatrix values;

  void close() {
    keys.close();
    values.close();
  }
}

final class _QuantizedKvPack {
  _QuantizedKvPack(this.weights, this.scales, [this.biases]);

  final MlxArray weights;
  final MlxArray scales;
  final MlxArray? biases;

  factory _QuantizedKvPack.fromMatrix(MlxQuantizedMatrix matrix) =>
      _QuantizedKvPack(matrix.weights, matrix.scales, matrix.biases);

  MlxQuantizedMatrix toMatrix() => MlxQuantizedMatrix(weights, scales, biases);

  _QuantizedKvPack sliceRows(int stop) => _QuantizedKvPack(
    weights.slice(
      start: [0, 0, 0, 0],
      stop: [weights.shape[0], weights.shape[1], stop, weights.shape[3]],
    ),
    scales.slice(
      start: [0, 0, 0, 0],
      stop: [scales.shape[0], scales.shape[1], stop, scales.shape[3]],
    ),
    biases?.slice(
      start: [0, 0, 0, 0],
      stop: [biases!.shape[0], biases!.shape[1], stop, biases!.shape[3]],
    ),
  );

  void close() {
    biases?.close();
    scales.close();
    weights.close();
  }

  int approxBytes() =>
      _approxArrayBytes(weights) +
      _approxArrayBytes(scales) +
      (biases == null ? 0 : _approxArrayBytes(biases!));
}

final class _QuantizedKvCache implements _LayerCache {
  _QuantizedKvCache({
    required this.numKvHeads,
    required this.headDim,
    required this.groupSize,
    required this.bits,
  });

  factory _QuantizedKvCache.fromDense(
    _KvCache dense, {
    required int bits,
    required int groupSize,
  }) {
    final cache = _QuantizedKvCache(
      numKvHeads: dense.numKvHeads,
      headDim: dense.headDim,
      groupSize: groupSize,
      bits: bits,
    );
    cache._offset = dense.offset;
    final keys = dense._keys;
    final values = dense._values;
    if (keys != null && values != null && dense.offset > 0) {
      final validKeys = keys.slice(
        start: [0, 0, 0, 0],
        stop: [1, dense.numKvHeads, dense.offset, dense.headDim],
      );
      final validValues = values.slice(
        start: [0, 0, 0, 0],
        stop: [1, dense.numKvHeads, dense.offset, dense.headDim],
      );
      try {
        cache._keys = _QuantizedKvPack.fromMatrix(
          mx.quant.quantize(validKeys, groupSize: groupSize, bits: bits),
        );
        cache._values = _QuantizedKvPack.fromMatrix(
          mx.quant.quantize(validValues, groupSize: groupSize, bits: bits),
        );
      } finally {
        validKeys.close();
        validValues.close();
      }
    }
    return cache;
  }

  static const int _step = 256;

  final int numKvHeads;
  final int headDim;
  final int groupSize;
  final int bits;

  _QuantizedKvPack? _keys;
  _QuantizedKvPack? _values;
  int _offset = 0;

  @override
  int get offset => _offset;

  int get capacity => _keys?.weights.shape[2] ?? 0;

  MlxQuantizedMatrix get borrowedKeys {
    final keys = _keys;
    if (keys == null) {
      throw StateError('Quantized KV cache is empty.');
    }
    return MlxQuantizedMatrix(keys.weights, keys.scales, keys.biases);
  }

  MlxQuantizedMatrix get borrowedValues {
    final values = _values;
    if (values == null) {
      throw StateError('Quantized KV cache is empty.');
    }
    return MlxQuantizedMatrix(values.weights, values.scales, values.biases);
  }

  void update(MlxArray nextKeys, MlxArray nextValues) {
    final batch = nextKeys.shape[0];
    final newTokens = nextKeys.shape[2];
    final keyDType = nextKeys.dtype;
    final valueDType = nextValues.dtype;
    final valueHeadDim = nextValues.shape[3];
    final prev = _offset;

    if (_keys == null || (prev + newTokens) > _keys!.weights.shape[2]) {
      final newSteps = ((_step + newTokens - 1) ~/ _step) * _step;
      final expandedKeys = _expandOrInit(
        current: _keys,
        batch: batch,
        newSteps: newSteps,
        dtype: keyDType,
        dim: headDim,
      );
      final expandedValues = _expandOrInit(
        current: _values,
        batch: batch,
        newSteps: newSteps,
        dtype: valueDType,
        dim: valueHeadDim,
      );
      _keys?.close();
      _values?.close();
      _keys = expandedKeys;
      _values = expandedValues;
    }

    MlxRuntime.evalAll([nextKeys, nextValues]);

    final qKeys = _QuantizedKvPack.fromMatrix(
      mx.quant.quantize(nextKeys, groupSize: groupSize, bits: bits),
    );
    final qValues = _QuantizedKvPack.fromMatrix(
      mx.quant.quantize(nextValues, groupSize: groupSize, bits: bits),
    );
    nextKeys.close();
    nextValues.close();

    final updatedKeys = _sliceUpdatePack(
      _keys!,
      qKeys,
      start: prev,
      stop: prev + newTokens,
    );
    final updatedValues = _sliceUpdatePack(
      _values!,
      qValues,
      start: prev,
      stop: prev + newTokens,
    );
    _keys!.close();
    _values!.close();
    qKeys.close();
    qValues.close();
    _keys = updatedKeys;
    _values = updatedValues;
    _offset += newTokens;
  }

  _QuantizedKvSlice updateAndFetch(MlxArray nextKeys, MlxArray nextValues) {
    update(nextKeys, nextValues);
    final updatedKeys = _keys!;
    final updatedValues = _values!;
    return _QuantizedKvSlice(
      keys: updatedKeys.sliceRows(_offset).toMatrix(),
      values: updatedValues.sliceRows(_offset).toMatrix(),
    );
  }

  @override
  void close() {
    _keys?.close();
    _values?.close();
    _keys = null;
    _values = null;
    _offset = 0;
  }

  @override
  void evalState() {
    final keys = _keys;
    final values = _values;
    if (keys == null || values == null) return;
    final eval = <MlxArray>[
      keys.weights,
      keys.scales,
      values.weights,
      values.scales,
    ];
    if (keys.biases != null) eval.add(keys.biases!);
    if (values.biases != null) eval.add(values.biases!);
    MlxRuntime.evalAll(eval);
  }

  @override
  int approxBytes() {
    final keys = _keys;
    final values = _values;
    if (keys == null || values == null) return 0;
    return keys.approxBytes() + values.approxBytes();
  }

  _QuantizedKvPack _expandOrInit({
    required _QuantizedKvPack? current,
    required int batch,
    required int newSteps,
    required MlxDType dtype,
    required int dim,
  }) {
    final zeroPack = _zerosPack(
      batch: batch,
      steps: newSteps,
      dtype: dtype,
      dim: dim,
    );
    if (current == null) return zeroPack;
    return _concatPacks(current, zeroPack);
  }

  _QuantizedKvPack _zerosPack({
    required int batch,
    required int steps,
    required MlxDType dtype,
    required int dim,
  }) {
    final packedDim = dim ~/ (32 ~/ bits);
    final scaleDim = dim ~/ groupSize;
    return _QuantizedKvPack(
      MlxArray.zeros([
        batch,
        numKvHeads,
        steps,
        packedDim,
      ], dtype: MlxDType.MLX_UINT32),
      MlxArray.zeros([batch, numKvHeads, steps, scaleDim], dtype: dtype),
      MlxArray.zeros([batch, numKvHeads, steps, scaleDim], dtype: dtype),
    );
  }

  _QuantizedKvPack _concatPacks(_QuantizedKvPack left, _QuantizedKvPack right) {
    final out = _QuantizedKvPack(
      mx.concatenate([left.weights, right.weights], axis: 2),
      mx.concatenate([left.scales, right.scales], axis: 2),
      mx.concatenate([left.biases!, right.biases!], axis: 2),
    );
    right.close();
    return out;
  }

  _QuantizedKvPack _sliceUpdatePack(
    _QuantizedKvPack target,
    _QuantizedKvPack source, {
    required int start,
    required int stop,
  }) {
    return _QuantizedKvPack(
      target.weights.sliceUpdate(
        source.weights,
        start: [0, 0, start, 0],
        stop: [
          target.weights.shape[0],
          target.weights.shape[1],
          stop,
          target.weights.shape[3],
        ],
      ),
      target.scales.sliceUpdate(
        source.scales,
        start: [0, 0, start, 0],
        stop: [
          target.scales.shape[0],
          target.scales.shape[1],
          stop,
          target.scales.shape[3],
        ],
      ),
      target.biases == null || source.biases == null
          ? null
          : target.biases!.sliceUpdate(
              source.biases!,
              start: [0, 0, start, 0],
              stop: [
                target.biases!.shape[0],
                target.biases!.shape[1],
                stop,
                target.biases!.shape[3],
              ],
            ),
    );
  }
}

extension PaddleOcrVlKvQuant on PaddleOcrVlRunner {
  MlxArray _quantizedScaledDotProductAttention(
    MlxArray queries,
    MlxQuantizedMatrix qKeys,
    MlxQuantizedMatrix qValues, {
    required double scale,
    required Object? mask,
    int? validKeyLen,
    required int groupSize,
    required int bits,
  }) {
    final batch = queries.shape[0];
    final numQHeads = queries.shape[1];
    final queryLen = queries.shape[2];
    final dim = queries.shape[3];
    final numKvHeads = qKeys.weights.shape[1];
    final repeats = numQHeads ~/ numKvHeads;

    final scaleArray = MlxArray.fromFloat32List(
      [scale],
      shape: [1],
    ).astype(queries.dtype);
    final scaledQueries = queries * scaleArray;
    scaleArray.close();

    MlxArray qMat = scaledQueries;
    MlxQuantizedMatrix keysMat = qKeys;
    MlxQuantizedMatrix valuesMat = qValues;
    var closeKeyTemps = false;
    var closeValueTemps = false;
    if (repeats > 1) {
      qMat = scaledQueries.reshape([batch, numKvHeads, repeats, queryLen, dim]);
      keysMat = MlxQuantizedMatrix(
        qKeys.weights.expandDims(-3),
        qKeys.scales.expandDims(-3),
        qKeys.biases?.expandDims(-3),
      );
      valuesMat = MlxQuantizedMatrix(
        qValues.weights.expandDims(-3),
        qValues.scales.expandDims(-3),
        qValues.biases?.expandDims(-3),
      );
      closeKeyTemps = true;
      closeValueTemps = true;
    }

    try {
      var scores = mx.quant.matmul(
        qMat,
        keysMat,
        transpose: true,
        groupSize: groupSize,
        bits: bits,
      );
      final masked = _applyAttentionMask(scores, mask);
      if (masked != scores) {
        scores.close();
        scores = masked;
      }
      if (validKeyLen != null && validKeyLen >= 0) {
        final validMasked = _applyValidKeyLength(scores, validKeyLen);
        if (validMasked != scores) {
          scores.close();
          scores = validMasked;
        }
      }
      final probs = mx.softmax(scores, axis: -1, precise: true);
      scores.close();

      var out = mx.quant.matmul(
        probs,
        valuesMat,
        transpose: false,
        groupSize: groupSize,
        bits: bits,
      );
      probs.close();
      if (repeats > 1) {
        final reshaped = out.reshape([batch, numQHeads, queryLen, dim]);
        out.close();
        out = reshaped;
      }
      return out;
    } finally {
      if (scaledQueries != queries) {
        scaledQueries.close();
      }
      if (qMat != scaledQueries) {
        qMat.close();
      }
      if (closeKeyTemps) {
        keysMat.close();
      }
      if (closeValueTemps) {
        valuesMat.close();
      }
    }
  }

  MlxArray _applyAttentionMask(MlxArray scores, Object? mask) {
    if (mask == null) return scores;
    if (mask is MlxArray) {
      return mx.add(scores, mask);
    }
    if (mask is! String || mask != 'causal') {
      return scores;
    }

    final queryLen = scores.shape[scores.shape.length - 2];
    final keyLen = scores.shape[scores.shape.length - 1];
    final qIndices = MlxArray.arange(
      (keyLen - queryLen).toDouble(),
      keyLen.toDouble(),
      1.0,
    );
    final kIndices = MlxArray.arange(0.0, keyLen.toDouble(), 1.0);
    final maskBool = mx.greaterEqual(
      qIndices.reshape([queryLen, 1]),
      kIndices.reshape([1, keyLen]),
    );
    qIndices.close();
    kIndices.close();
    final minValue = scores.dtype == MlxDType.MLX_FLOAT16
        ? -65504.0
        : -3.3895313892515355e38;
    final minArray = MlxArray.full([], minValue, dtype: scores.dtype);
    final masked = mx.where(maskBool, scores, minArray);
    minArray.close();
    maskBool.close();
    return masked;
  }

  MlxArray _applyValidKeyLength(MlxArray scores, int validKeyLen) {
    final keyLen = scores.shape.last;
    if (validKeyLen <= 0 || validKeyLen >= keyLen) {
      return scores;
    }

    final minValue = scores.dtype == MlxDType.MLX_FLOAT16
        ? -65504.0
        : -3.3895313892515355e38;
    final values = Float32List(keyLen);
    for (var i = validKeyLen; i < keyLen; i++) {
      values[i] = minValue;
    }
    final mask1d = MlxArray.fromFloat32List(values, shape: [keyLen]);
    final cast = mask1d.astype(scores.dtype);
    mask1d.close();
    final shape = List<int>.filled(scores.shape.length, 1);
    shape[shape.length - 1] = keyLen;
    final reshaped = cast.reshape(shape);
    cast.close();
    final masked = mx.add(scores, reshaped);
    reshaped.close();
    return masked;
  }
}
