part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// KV cache for ERNIE-4.5 autoregressive decoding
// ---------------------------------------------------------------------------

sealed class _LayerCache {
  int get offset;
  int approxBytes();
  void close();
  void evalState();
}

/// Per-layer dense KV cache backed by pre-allocated buffers.
final class _KvCache implements _LayerCache {
  _KvCache({
    required this.numKvHeads,
    required this.headDim,
    required this.maxSeqLen,
  });

  final int numKvHeads;
  final int headDim;
  final int maxSeqLen;

  MlxArray? _keys; // [1, numKvHeads, maxSeqLen, headDim]
  MlxArray? _values;
  int _offset = 0;

  @override
  int get offset => _offset;

  (MlxArray, MlxArray) updateAndFetch(MlxArray nextKeys, MlxArray nextValues) {
    final newTokens = nextKeys.shape[2];
    if (_keys == null) {
      final dt = nextKeys.dtype;
      _keys = MlxArray.zeros([1, numKvHeads, maxSeqLen, headDim], dtype: dt);
      _values = MlxArray.zeros([1, numKvHeads, maxSeqLen, headDim], dtype: dt);
    }

    final updatedKeys = _keys!.sliceUpdate(
      nextKeys,
      start: [0, 0, _offset, 0],
      stop: [1, numKvHeads, _offset + newTokens, headDim],
    );
    final updatedValues = _values!.sliceUpdate(
      nextValues,
      start: [0, 0, _offset, 0],
      stop: [1, numKvHeads, _offset + newTokens, headDim],
    );

    _keys!.close();
    _values!.close();
    nextKeys.close();
    nextValues.close();
    _keys = updatedKeys;
    _values = updatedValues;
    _offset += newTokens;

    final validKeys = updatedKeys.slice(
      start: [0, 0, 0, 0],
      stop: [1, numKvHeads, _offset, headDim],
    );
    final validValues = updatedValues.slice(
      start: [0, 0, 0, 0],
      stop: [1, numKvHeads, _offset, headDim],
    );
    return (validKeys, validValues);
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
    MlxRuntime.evalAll([keys, values]);
  }

  @override
  int approxBytes() {
    final keys = _keys;
    final values = _values;
    if (keys == null || values == null) return 0;
    return _approxArrayBytes(keys) + _approxArrayBytes(values);
  }

  _QuantizedKvCache toQuantized({required int bits, required int groupSize}) =>
      _QuantizedKvCache.fromDense(this, bits: bits, groupSize: groupSize);
}

/// Full model decode cache — one cache entry per decoder layer.
final class _ModelCache {
  _ModelCache._(
    this.layers, {
    required this.kvBits,
    required this.kvGroupSize,
    required this.quantizedStart,
    required this.kvScheme,
    required this.turboBits,
    required this.turboStart,
  });

  factory _ModelCache.create({required PaddleOcrVlConfig config}) =>
      _ModelCache._(
        List.generate(config.numHiddenLayers, (index) {
          if (config.kvCacheQuantSchemeForCurrentPlatform == 'turboquant' &&
              !config.turboDensePrefillForCurrentPlatform &&
              config.turboQuantizedStartForCurrentPlatform <= 0 &&
              index != config.numHiddenLayers - 1) {
            return _TurboQuantKvCache(
              numKvHeads: config.numKeyValueHeads,
              headDim: config.headDim,
              maxSeqLen: config.maxKvCacheSeqLenForCurrentPlatform,
              bits: config.turboQuantBitsForCurrentPlatform ?? 4.0,
              capacityStep: config.turboCapacityStepForCurrentPlatform,
            );
          }
          if (config.kvCacheQuantSchemeForCurrentPlatform == 'uniform' &&
              config.uniformQuantizedPrefillForCurrentPlatform &&
              config.kvCacheQuantBitsForCurrentPlatform > 0 &&
              config.kvCacheQuantizedStartForCurrentPlatform <= 0) {
            return _QuantizedKvCache(
              numKvHeads: config.numKeyValueHeads,
              headDim: config.headDim,
              groupSize: config.kvCacheQuantGroupSizeForCurrentPlatform,
              bits: config.kvCacheQuantBitsForCurrentPlatform,
            );
          }
          return _KvCache(
            numKvHeads: config.numKeyValueHeads,
            headDim: config.headDim,
            maxSeqLen: config.maxKvCacheSeqLenForCurrentPlatform,
          );
        }),
        kvBits: config.kvCacheQuantBitsForCurrentPlatform,
        kvGroupSize: config.kvCacheQuantGroupSizeForCurrentPlatform,
        quantizedStart: config.kvCacheQuantizedStartForCurrentPlatform,
        kvScheme: config.kvCacheQuantSchemeForCurrentPlatform,
        turboBits: config.turboQuantBitsForCurrentPlatform,
        turboStart: config.turboQuantizedStartForCurrentPlatform,
      );

  final List<_LayerCache> layers;
  final int kvBits;
  final int kvGroupSize;
  final int quantizedStart;
  final String kvScheme;
  final double? turboBits;
  final int turboStart;

  int get offset => layers.isEmpty ? 0 : layers.first.offset;

  String describeLayerKinds() {
    var dense = 0;
    var uniform = 0;
    var turbo = 0;
    var turboPhysical = -1;
    var turboCapacity = -1;
    var turboCompactions = 0;
    var approxBytes = 0;
    for (final layer in layers) {
      approxBytes += layer.approxBytes();
      if (layer is _KvCache) {
        dense++;
      } else if (layer is _QuantizedKvCache) {
        uniform++;
      } else if (layer is _TurboQuantKvCache) {
        turbo++;
        turboPhysical = layer.physicalLength;
        turboCapacity = layer.capacity;
        turboCompactions = layer.compactCount;
      }
    }
    final extra = turbo > 0
        ? ' physical=$turboPhysical capacity=$turboCapacity compactions=$turboCompactions approx=${_formatApproxBytes(approxBytes)}'
        : '';
    return 'dense=$dense uniform=$uniform turbo=$turbo offset=$offset$extra';
  }

  void evalStates() {
    for (final layer in layers) {
      layer.evalState();
    }
  }

  int approxBytes() =>
      layers.fold<int>(0, (total, layer) => total + layer.approxBytes());

  void maybeCompact({required PaddleOcrVlConfig config}) {
    final budget = config.turboCompactBudgetForCurrentPlatform;
    if (budget <= 0 || kvScheme != 'turboquant') return;
    final interval = config.turboCompactIntervalForCurrentPlatform;
    if (interval > 1 && (offset % interval) != 0) return;
    final hysteresis = config.turboCompactHysteresisForCurrentPlatform;
    final turboLayers = <_TurboQuantKvCache>[
      for (final layer in layers)
        if (layer is _TurboQuantKvCache) layer,
    ];
    if (turboLayers.isEmpty) return;
    final physical = turboLayers.first.physicalLength;
    if (physical <= budget + hysteresis) return;
    for (final layer in turboLayers.skip(1)) {
      if (layer.physicalLength != physical) return;
    }

    final keepRecent = math.min(
      config.turboCompactKeepRecentForCurrentPlatform.clamp(0, budget - 1),
      physical - 1,
    );
    final keepPrefix = math.min(
      config.turboCompactKeepPrefixForCurrentPlatform.clamp(0, budget - 1),
      physical - 1,
    );
    if (keepPrefix >= budget) return;
    final scores = List<double>.filled(physical, 0);
    for (final layer in turboLayers) {
      final aggregate = layer.aggregateKeyNorms();
      try {
        final valuesF32 = aggregate.astype(MlxDType.MLX_FLOAT32);
        final values = valuesF32.toList();
        valuesF32.close();
        for (var i = 0; i < physical; i++) {
          scores[i] += (values[i] as num).toDouble();
        }
      } finally {
        aggregate.close();
      }
    }

    final keep = _selectCompactIndices(
      scores,
      budget: budget,
      keepPrefix: keepPrefix,
      keepRecent: keepRecent,
    );
    final keepArr = MlxArray.fromInt32List(keep, shape: [keep.length]);
    final headroom = math.max(hysteresis, interval > 0 ? interval * 2 : 64);
    try {
      for (final layer in turboLayers) {
        layer.compact(keepArr, headroom: headroom);
      }
    } finally {
      keepArr.close();
    }
    evalStates();
  }

  void maybeQuantize({required PaddleOcrVlConfig config}) {
    if (kvScheme == 'turboquant') {
      if (turboBits == null || turboStart < 0 || offset < turboStart) return;
      final lastIdx = layers.length > 2 ? layers.length - 1 : -1;
      final uniformLast =
          PaddleOcrVlDebugOverrides.turboUniformLastLayer == true;
      final turboLast =
          PaddleOcrVlDebugOverrides.turboQuantizeLastLayer ??
          Platform.isIOS;
      for (var i = 0; i < layers.length; i++) {
        if (i == lastIdx && !turboLast) continue;
        final layer = layers[i];
        if (layer is! _KvCache) continue;
        final turbo = _TurboQuantKvCache.fromDense(
          layer,
          bits: turboBits!,
          capacityStep: config.turboCapacityStepForCurrentPlatform,
        );
        layer.close();
        layers[i] = turbo;
      }
      if (uniformLast && !turboLast && lastIdx >= 0) {
        final layer = layers[lastIdx];
        if (layer is _KvCache) {
          final quantized = _QuantizedKvCache.fromDense(
            layer,
            bits: 4,
            groupSize: kvGroupSize,
          );
          layer.close();
          layers[lastIdx] = quantized;
        }
      }
      return;
    }
    if (kvBits <= 0 || quantizedStart < 0 || offset < quantizedStart) return;
    for (var i = 0; i < layers.length; i++) {
      final layer = layers[i];
      if (layer is! _KvCache) continue;
      final quantized = layer.toQuantized(bits: kvBits, groupSize: kvGroupSize);
      layer.close();
      layers[i] = quantized;
    }
  }

  void close() {
    for (final layer in layers) {
      layer.close();
    }
  }
}

String _formatApproxBytes(int bytes) {
  const kb = 1024;
  const mb = kb * 1024;
  if (bytes >= mb) {
    return '${(bytes / mb).toStringAsFixed(1)}MB';
  }
  if (bytes >= kb) {
    return '${(bytes / kb).toStringAsFixed(1)}KB';
  }
  return '${bytes}B';
}

int _approxArrayBytes(MlxArray array) =>
    array.shape.fold<int>(1, (acc, dim) => acc * dim) *
    _dtypeBytes(array.dtype);

int _dtypeBytes(MlxDType dtype) => switch (dtype) {
  MlxDType.MLX_BOOL => 1,
  MlxDType.MLX_UINT8 || MlxDType.MLX_INT8 => 1,
  MlxDType.MLX_UINT16 ||
  MlxDType.MLX_INT16 ||
  MlxDType.MLX_FLOAT16 ||
  MlxDType.MLX_BFLOAT16 => 2,
  MlxDType.MLX_UINT32 || MlxDType.MLX_INT32 || MlxDType.MLX_FLOAT32 => 4,
  MlxDType.MLX_UINT64 ||
  MlxDType.MLX_INT64 ||
  MlxDType.MLX_FLOAT64 ||
  MlxDType.MLX_COMPLEX64 => 8,
};

List<int> _selectCompactIndices(
  List<double> scores, {
  required int budget,
  required int keepPrefix,
  required int keepRecent,
}) {
  final seqLen = scores.length;
  if (seqLen <= budget) {
    return List<int>.generate(seqLen, (i) => i, growable: false);
  }
  final prefix = math.min(keepPrefix, budget);
  final recent = math.min(keepRecent, budget - prefix);
  final middleStart = prefix;
  final middleEnd = seqLen - recent;
  final keepFromHistory = budget - prefix - recent;
  final ranked = List<int>.generate(
    math.max(0, middleEnd - middleStart),
    (i) => i + middleStart,
  );
  ranked.sort((a, b) => scores[b].compareTo(scores[a]));
  final kept = <int>[
    for (var i = 0; i < prefix; i++) i,
    ...ranked.take(keepFromHistory),
  ]..sort();
  for (var i = seqLen - recent; i < seqLen; i++) {
    kept.add(i);
  }
  return kept;
}
