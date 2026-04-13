part of 'paddle_ocr_vl.dart';

String? _turboLastAttentionPath;

bool get _turboDisableFastValuePath {
  final debug = PaddleOcrVlDebugOverrides.turboDisableFastValue;
  if (debug != null) return debug;
  final override =
      Platform.environment['DART_MLX_PADDLE_TURBO_DISABLE_FAST_VALUE'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

bool get _turboDisableFastScorePath {
  final debug = PaddleOcrVlDebugOverrides.turboDisableFastScore;
  if (debug != null) return debug;
  final override =
      Platform.environment['DART_MLX_PADDLE_TURBO_DISABLE_FAST_SCORE'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

bool get _turboDisableFusedKvPath {
  final debug = PaddleOcrVlDebugOverrides.turboDisableFusedKv;
  if (debug != null) return debug;
  final override =
      Platform.environment['DART_MLX_PADDLE_TURBO_DISABLE_FUSED_KV'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

bool get _turboDisableFusedDecodePath {
  final debug = PaddleOcrVlDebugOverrides.turboDisableFusedDecode;
  if (debug != null) return debug;
  final override =
      Platform.environment['DART_MLX_PADDLE_TURBO_DISABLE_FUSED_DECODE'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

bool get _turboDisableSingleTokenQuantizePath {
  final debug = PaddleOcrVlDebugOverrides.turboDisableSingleQuant;
  if (debug != null) return debug;
  final override =
      Platform.environment['DART_MLX_PADDLE_TURBO_DISABLE_SINGLE_QUANT'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

bool get _turboCompactDebug {
  final override = Platform.environment['DART_MLX_PADDLE_TURBO_COMPACT_DEBUG'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

bool get _turboUpdateTrace {
  final debug = PaddleOcrVlDebugOverrides.turboUpdateTrace;
  if (debug != null) return debug;
  final override = Platform.environment['DART_MLX_PADDLE_TURBO_UPDATE_TRACE'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

bool get _turboDequantKvToQueryDTypePath {
  final debug = PaddleOcrVlDebugOverrides.turboDequantKvToQueryDType;
  if (debug != null) return debug;
  final override =
      Platform.environment['DART_MLX_PADDLE_TURBO_DEQUANT_KV_TO_QUERY_DTYPE'];
  if (override != null) {
    return override == '1' || override.toLowerCase() == 'true';
  }
  return Platform.isIOS;
}

bool get _turboDetachStateEachUpdate {
  final debug = PaddleOcrVlDebugOverrides.turboDetachStateEachUpdate;
  if (debug != null) return debug;
  final override =
      Platform.environment['DART_MLX_PADDLE_TURBO_DETACH_STATE_EACH_UPDATE'];
  if (override == null) return false;
  return override == '1' || override.toLowerCase() == 'true';
}

final class _TurboQuantMseState {
  _TurboQuantMseState(this.norms, this.indices);

  MlxArray norms; // [B, H, T]
  MlxArray indices; // [B, H, T, packedWidth]

  void close() {
    indices.close();
    norms.close();
  }
}

final class _TurboQuantMseCodec {
  factory _TurboQuantMseCodec({
    required int dim,
    required int bits,
    required int seed,
  }) {
    final codebook = _turboCodebook(dim, bits);
    final midpoints = _turboMidpoints(codebook);
    return _TurboQuantMseCodec._(
      dim: dim,
      bits: bits,
      signs: _turboRhtSignVector(dim, seed),
      rotation: _turboRotationMatrix(dim, seed),
      codebook: codebook,
      midpoints: midpoints,
    );
  }

  _TurboQuantMseCodec._({
    required this.dim,
    required this.bits,
    required this.signs,
    required this.rotation,
    required this.codebook,
    required this.midpoints,
  });

  final int dim;
  final int bits;
  final MlxArray signs;
  final MlxArray rotation;
  final MlxArray codebook;
  final MlxArray midpoints;

  MlxArray _rotateForward(MlxArray input) {
    final signsForInput = input.dtype == signs.dtype
        ? signs
        : signs.astype(input.dtype);
    final signed = input * signsForInput;
    if (!identical(signsForInput, signs)) {
      signsForInput.close();
    }
    final scale = 1.0 / math.sqrt(dim.toDouble());
    final rotated = MlxMore.hadamardTransform(signed, scale: scale);
    signed.close();
    return rotated;
  }

  MlxArray _rotateInverse(MlxArray input) {
    final scale = 1.0 / math.sqrt(dim.toDouble());
    final rotated = MlxMore.hadamardTransform(input, scale: scale);
    final signsForInput = input.dtype == signs.dtype
        ? signs
        : signs.astype(input.dtype);
    final out = rotated * signsForInput;
    if (!identical(signsForInput, signs)) {
      signsForInput.close();
    }
    rotated.close();
    return out;
  }

  _TurboQuantMseState quantize(MlxArray vectors) {
    if (vectors.shape[vectors.ndim - 2] == 1 &&
        !_turboDisableSingleTokenQuantizePath) {
      final fast = _trySingleTokenQuantize(vectors);
      if (fast != null) {
        return fast;
      }
    }
    final vectorsF32 = vectors.astype(MlxDType.MLX_FLOAT32);
    final squares = MlxMore.square(vectorsF32);
    final sqSum = mx.sum(squares, axis: vectors.ndim - 1);
    squares.close();
    final norms = MlxMore.sqrt(sqSum);
    sqSum.close();
    final eps = MlxArray.full([], 1e-6, dtype: MlxDType.MLX_FLOAT32);
    final safeNorms = mx.maximum(norms, eps);
    eps.close();
    final unit = vectorsF32 / safeNorms.expandDims(vectors.ndim - 1);
    vectorsF32.close();
    final rotated = _rotateForward(unit);
    unit.close();

    var indices = MlxArray.zeros(rotated.shape, dtype: MlxDType.MLX_UINT32);
    for (var i = 0; i < midpoints.shape[0]; i++) {
      final midpoint = midpoints.slice(start: [i], stop: [i + 1]).reshape([]);
      final mask = MlxMore.greater(rotated, midpoint);
      midpoint.close();
      final maskInt = mask.astype(MlxDType.MLX_UINT32);
      mask.close();
      final next = indices + maskInt;
      maskInt.close();
      indices.close();
      indices = next;
    }
    rotated.close();

    final packed = _turboPackLowbit(indices, bits);
    indices.close();
    final normsF16 = norms.astype(MlxDType.MLX_FLOAT16);
    norms.close();
    return _TurboQuantMseState(normsF16, packed);
  }

  _TurboQuantMseState? _trySingleTokenQuantize(MlxArray vectors) {
    if (!MlxMetal.isAvailable() || !_isPowerOfTwo(dim)) return null;
    final batchHeads = vectors.size ~/ dim;
    final flat = vectors
        .reshape([batchHeads, dim])
        .astype(MlxDType.MLX_FLOAT32);
    final norms = mx.linalg.norm(flat, axes: [1]);
    final eps = MlxArray.full([], 1e-6, dtype: MlxDType.MLX_FLOAT32);
    final safeNorms = mx.maximum(norms, eps);
    eps.close();
    final unit = flat / safeNorms.expandDims(1);
    flat.close();
    final rotated = _rotateForward(unit);
    unit.close();
    final rotatedFlat = rotated.reshape([batchHeads, dim]);
    rotated.close();

    final packedWidth = _turboPackedWidth(dim, bits);
    final kernel = _getTurboNoRotQuantizeKernel(bits);
    final config = mx.fast.metalConfig();
    config.addOutputArg([batchHeads, packedWidth], MlxDType.MLX_UINT32);
    config.setGrid(dim * batchHeads, 1, 1);
    config.setThreadGroup(dim, 1, 1);
    config.addTemplateInt('Dim', dim);
    config.addTemplateInt('Bits', bits);
    config.addTemplateInt('PackedWidth', packedWidth);
    final outputs = kernel.apply([rotatedFlat, midpoints], config);
    rotatedFlat.close();
    final orig = [...vectors.shape.sublist(0, vectors.ndim - 1)];
    final normsF16 = norms.astype(MlxDType.MLX_FLOAT16).reshape(orig);
    norms.close();
    final packed = outputs.first.reshape([...orig, packedWidth]);
    outputs.first.close();
    return _TurboQuantMseState(normsF16, packed);
  }

  MlxArray dequantize(_TurboQuantMseState state) {
    final unpacked = _turboUnpackLowbit(
      state.indices,
      bits,
      dim,
    ).astype(MlxDType.MLX_INT32);
    final rotated = codebook.take(unpacked, axis: 0);
    unpacked.close();
    final unit = _rotateInverse(rotated);
    rotated.close();
    final norms = state.norms
        .astype(MlxDType.MLX_FLOAT32)
        .expandDims(state.norms.ndim);
    final out = unit * norms;
    norms.close();
    unit.close();
    return out;
  }

  MlxArray prepareQueries(MlxArray queries) => _rotateForward(queries);

  MlxArray scorePrepared(MlxArray preparedQueries, _TurboQuantMseState state) {
    final qShape = preparedQueries.shape;
    if (qShape[qShape.length - 2] == 1 && !_turboDisableFastScorePath) {
      final fast = _turboMseScore(
        preparedQueries.reshape([
          qShape[0],
          qShape[1],
          qShape[2],
          qShape[qShape.length - 1],
        ]),
        state,
        bits,
        codebook,
      );
      if (fast != null) {
        return fast.expandDims(3);
      }
    }

    final unpacked = _turboUnpackLowbit(
      state.indices,
      bits,
      dim,
    ).astype(MlxDType.MLX_INT32);
    final rotated = codebook.take(unpacked, axis: 0);
    unpacked.close();
    final dots = mx.einsum('bhmld,bhtd->bhmlt', [preparedQueries, rotated]);
    rotated.close();
    final norms = state.norms
        .astype(MlxDType.MLX_FLOAT32)
        .expandDims(2)
        .expandDims(2);
    final out = dots * norms;
    norms.close();
    dots.close();
    return out;
  }

  void close() {
    midpoints.close();
    codebook.close();
    rotation.close();
    signs.close();
  }
}

bool _isPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

final class _TurboQuantKvCache implements _LayerCache {
  _TurboQuantKvCache({
    required this.numKvHeads,
    required this.headDim,
    required this.maxSeqLen,
    required this.bits,
    this.capacityStep = 256,
  }) : _keyCodec = _TurboQuantMseCodec(
         dim: headDim,
         bits: bits.floor(),
         seed: 0,
       ),
       _valueCodec = _TurboQuantMseCodec(
         dim: headDim,
         bits: bits.ceil(),
         seed: 1,
       ) {
    if (capacityStep <= 0) {
      throw ArgumentError.value(capacityStep, 'capacityStep', 'must be > 0');
    }
  }

  factory _TurboQuantKvCache.fromDense(
    _KvCache dense, {
    required double bits,
    int capacityStep = 256,
  }) {
    final cache = _TurboQuantKvCache(
      numKvHeads: dense.numKvHeads,
      headDim: dense.headDim,
      maxSeqLen: dense.maxSeqLen,
      bits: bits,
      capacityStep: capacityStep,
    );
    final keys = dense._keys;
    final values = dense._values;
    if (keys != null && values != null) {
      final used = dense.offset;
      if (used > 0) {
        final keySlice = keys.slice(
          start: [0, 0, 0, 0],
          stop: [keys.shape[0], keys.shape[1], used, keys.shape[3]],
        );
        final valueSlice = values.slice(
          start: [0, 0, 0, 0],
          stop: [values.shape[0], values.shape[1], used, values.shape[3]],
        );
        cache.update(keySlice, valueSlice);
      }
    }
    return cache;
  }

  static const int _evalInterval = 50;

  final int numKvHeads;
  final int headDim;
  final int maxSeqLen;
  final double bits;
  final int capacityStep;
  final _TurboQuantMseCodec _keyCodec;
  final _TurboQuantMseCodec _valueCodec;

  _TurboQuantMseState? _keys;
  _TurboQuantMseState? _values;
  _TurboQuantMseState? _cachedKeys;
  _TurboQuantMseState? _cachedValues;
  int _cachedOffset = -1;
  int _offset = 0;
  int _physicalOffset = 0;
  int _compactCount = 0;
  int _protectedPrefixLength = 0;

  @override
  int get offset => _offset;

  int get physicalLength => _physicalOffset;

  int get compactCount => _compactCount;

  int get capacity => _keys?.norms.shape[2] ?? 0;

  int get protectedPrefixLength => _protectedPrefixLength;

  set protectedPrefixLength(int value) {
    _protectedPrefixLength = math.max(0, value);
  }

  (_TurboQuantMseState, _TurboQuantMseState) get _state {
    final keys = _keys;
    final values = _values;
    if (keys == null || values == null) {
      throw StateError('TurboQuant cache is empty.');
    }
    if (_cachedOffset == _offset &&
        _cachedKeys != null &&
        _cachedValues != null) {
      return (_cachedKeys!, _cachedValues!);
    }
    _invalidateCachedState();
    final slicedKeys = _turboSliceState(keys, _physicalOffset);
    final slicedValues = _turboSliceState(values, _physicalOffset);
    _cachedKeys = slicedKeys;
    _cachedValues = slicedValues;
    _cachedOffset = _offset;
    return (slicedKeys, slicedValues);
  }

  void _invalidateCachedState() {
    _cachedKeys?.close();
    _cachedValues?.close();
    _cachedKeys = null;
    _cachedValues = null;
    _cachedOffset = -1;
  }

  void update(MlxArray keys, MlxArray values) {
    _invalidateCachedState();
    final newTokens = keys.shape[2];
    final trace = _turboUpdateTrace ? _TurboUpdateTrace.start(_offset) : null;
    final fused = _turboTryFusedKvQuantize(
      keys,
      values,
      _keyCodec,
      _valueCodec,
    );
    final newKeys = fused?.$1 ?? _keyCodec.quantize(keys);
    final newValues = fused?.$2 ?? _valueCodec.quantize(values);
    trace?.mark('quantize');
    keys.close();
    values.close();

    final newEnd = _physicalOffset + newKeys.norms.shape[2];
    if (_keys == null) {
      final initialCapacity =
          ((math.max(capacityStep, newEnd) + capacityStep - 1) ~/
              capacityStep) *
          capacityStep;
      _keys = _turboAllocateStateLike(newKeys, initialCapacity);
      _values = _turboAllocateStateLike(newValues, initialCapacity);
    } else {
      _keys = _turboReserveStateCapacity(
        _keys!,
        _physicalOffset,
        newEnd,
        capacityStep,
      );
      _values = _turboReserveStateCapacity(
        _values!,
        _physicalOffset,
        newEnd,
        capacityStep,
      );
    }
    trace?.mark('reserve');
    _turboWriteState(_keys!, newKeys, _physicalOffset);
    _turboWriteState(_values!, newValues, _physicalOffset);
    trace?.mark('write');
    final quantizedTokens = newKeys.norms.shape[2];
    newKeys.close();
    newValues.close();
    _physicalOffset = newEnd;
    _offset += newTokens;
    if (quantizedTokens > 1 || (_physicalOffset % _evalInterval) == 0) {
      evalState();
      trace?.mark('eval');
    }
    trace?.finish();
  }

  MlxArray aggregateKeyNorms() {
    final state = _state.$1;
    final byHead = state.norms.sum(axis: 1);
    final aggregate = byHead.sum(axis: 0);
    byHead.close();
    return aggregate;
  }

  void compact(MlxArray keptIndices, {int headroom = 0}) {
    final keys = _keys;
    final values = _values;
    if (keys == null || values == null) return;
    if (keptIndices.shape[0] >= _physicalOffset) return;
    final originalOffset = _offset;
    final originalPhysical = _physicalOffset;
    _invalidateCachedState();
    final keyState = _turboSliceState(keys, originalPhysical);
    final valueState = _turboSliceState(values, originalPhysical);
    final keptKeyState = _turboTakeState(keyState, keptIndices);
    final keptValueState = _turboTakeState(valueState, keptIndices);
    final compactedLength = keptIndices.shape[0];
    final compactedCapacity = compactedLength + math.max(0, headroom).toInt();
    final compactedKeys = _turboAllocateStateLike(
      keptKeyState,
      compactedCapacity,
    );
    final compactedValues = _turboAllocateStateLike(
      keptValueState,
      compactedCapacity,
    );
    final keyStateShape = keyState.norms.shape;
    final keptKeyStateShape = keptKeyState.norms.shape;
    final compactedKeysShape = compactedKeys.norms.shape;
    _turboWriteState(compactedKeys, keptKeyState, 0);
    _turboWriteState(compactedValues, keptValueState, 0);
    keyState.close();
    valueState.close();
    keptKeyState.close();
    keptValueState.close();
    if (_turboCompactDebug) {
      stdout.writeln(
        'turboCompact '
        'keep=${keptIndices.shape[0]} '
        'keyState=$keyStateShape '
        'keptKeyState=$keptKeyStateShape '
        'compactedKeys=$compactedKeysShape '
        'headroom=$headroom',
      );
    }
    keys.close();
    values.close();
    _keys = compactedKeys;
    _values = compactedValues;
    _physicalOffset = compactedLength;
    _offset = originalOffset;
    _compactCount++;
  }

  MlxArray quantizedAttention(
    MlxArray queries, {
    required double scale,
    required Object? mask,
  }) {
    final state = _state;
    final keyState = state.$1;
    final valueState = state.$2;
    if (queries.shape[2] == 1 && mask == null) {
      final batch = queries.shape[0];
      final repeats = queries.shape[1] ~/ numKvHeads;
      if (!_turboDisableFusedDecodePath) {
        final fused = _turboMseFusedDecodeAttention(
          queries,
          keyState,
          valueState,
          keyCodec: _keyCodec,
          valueCodec: _valueCodec,
          repeats: repeats,
          scale: scale,
        );
        if (fused != null) {
          _turboLastAttentionPath = 'fused';
          return fused;
        }
      }
      if (repeats > 1 && !_turboDisableFastValuePath) {
        final scaleArr = MlxArray.full([], scale, dtype: queries.dtype);
        final scaled = queries * scaleArr;
        scaleArr.close();
        final grouped = scaled.reshape([
          batch,
          numKvHeads,
          repeats,
          1,
          headDim,
        ]);
        scaled.close();
        final prepared = _keyCodec.prepareQueries(grouped);
        grouped.close();
        final scores = _keyCodec.scorePrepared(prepared, keyState);
        prepared.close();
        final direct = _turboMseWeightedSumFromScoresDirect(
          scores,
          valueState,
          _valueCodec,
        );
        scores.close();
        if (direct != null) {
          _turboLastAttentionPath = 'fast_value';
          final reshaped = direct.reshape([
            batch,
            numKvHeads * repeats,
            1,
            headDim,
          ]);
          return reshaped;
        }
      }
    }

    final k = _keyCodec.dequantize(keyState);
    final v = _valueCodec.dequantize(valueState);
    final targetDType = queries.dtype;
    final castKvToQueryDType = _turboDequantKvToQueryDTypePath;
    final kForAttn = castKvToQueryDType && k.dtype != targetDType
        ? k.astype(targetDType)
        : k;
    final vForAttn = castKvToQueryDType && v.dtype != targetDType
        ? v.astype(targetDType)
        : v;
    final out = mx.fast.scaledDotProductAttention(
      queries,
      kForAttn,
      vForAttn,
      scale: scale,
      maskMode: mask == null ? '' : 'causal',
    );
    _turboLastAttentionPath = 'dequant';
    if (!identical(kForAttn, k)) {
      kForAttn.close();
    }
    if (!identical(vForAttn, v)) {
      vForAttn.close();
    }
    k.close();
    v.close();
    return out;
  }

  @override
  void close() {
    _invalidateCachedState();
    _keys?.close();
    _values?.close();
    _keys = null;
    _values = null;
    _offset = 0;
    _physicalOffset = 0;
    _compactCount = 0;
    _keyCodec.close();
    _valueCodec.close();
  }

  @override
  void evalState() {
    final keys = _keys;
    final values = _values;
    if (keys == null || values == null) return;
    MlxRuntime.evalAll([
      keys.norms,
      keys.indices,
      values.norms,
      values.indices,
    ]);
  }

  @override
  int approxBytes() {
    final keys = _keys;
    final values = _values;
    if (keys == null || values == null) return 0;
    return _approxArrayBytes(keys.norms) +
        _approxArrayBytes(keys.indices) +
        _approxArrayBytes(values.norms) +
        _approxArrayBytes(values.indices);
  }
}

({MlxArray output, MlxArray denom, MlxArray maxScores})?
_turboMseWeightedSumStatsFromScores(
  MlxArray scores,
  _TurboQuantMseState state,
  _TurboQuantMseCodec codec,
) {
  final maxScores = _turboMaxLastAxis(scores);
  if (scores.ndim == 5 && scores.shape[3] == 1) {
    final maxScores2d = maxScores.reshape([
      maxScores.shape[0],
      maxScores.shape[1],
      maxScores.shape[2],
    ]);
    final fastOutput = _turboMseWeightedSumFromScores(
      scores,
      state,
      codec,
      maxScores2d,
    );
    maxScores2d.close();
    if (fastOutput != null) {
      final maxScoresExpanded = maxScores.expandDims(maxScores.ndim);
      final weights = mx.exp(scores - maxScoresExpanded);
      maxScoresExpanded.close();
      final denom = mx.sum(weights, axis: -1);
      weights.close();
      return (output: fastOutput, denom: denom, maxScores: maxScores);
    }
  }

  final maxScoresExpanded = maxScores.expandDims(maxScores.ndim);
  final weights = mx.exp(scores - maxScoresExpanded);
  maxScoresExpanded.close();
  final output = _turboMseWeightedSum(weights, state, codec);
  if (output == null) {
    weights.close();
    maxScores.close();
    return null;
  }
  final denom = mx.sum(weights, axis: -1);
  weights.close();
  return (output: output, denom: denom, maxScores: maxScores);
}

MlxArray? _turboMseFusedDecodeAttention(
  MlxArray queries,
  _TurboQuantMseState keyState,
  _TurboQuantMseState valueState, {
  required _TurboQuantMseCodec keyCodec,
  required _TurboQuantMseCodec valueCodec,
  required int repeats,
  required double scale,
}) {
  if (!MlxMetal.isAvailable()) return null;
  if (queries.shape[2] != 1) return null;
  final keyBits = keyCodec.bits;
  final valBits = valueCodec.bits;
  final dim = queries.shape[3];
  final totalTokens = keyState.norms.shape[2];
  if (totalTokens > 2048 || dim < 32 || dim % 32 != 0) return null;
  final batch = queries.shape[0];
  final kvHeads = keyState.norms.shape[1];
  final qHeads = queries.shape[1];
  final valueDim = valueCodec.dim;

  final scaleArr = MlxArray.full([], scale, dtype: queries.dtype);
  final scaled = queries * scaleArr;
  scaleArr.close();
  final grouped = scaled.reshape([batch, kvHeads, repeats, 1, dim]);
  scaled.close();
  final qRot = keyCodec.prepareQueries(grouped);
  grouped.close();
  final qRotFlat = qRot.reshape([batch * kvHeads * repeats, dim]);
  qRot.close();

  final kernel = _getTurboFusedMseDecodeKernel(keyBits, valBits, dim);
  final config = mx.fast.metalConfig();
  config.addOutputArg([batch * qHeads, dim], MlxDType.MLX_FLOAT32);
  config.setGrid(batch * qHeads * 1024, 1, 1);
  config.setThreadGroup(1024, 1, 1);
  config.addTemplateInt('Dim', dim);
  config.addTemplateInt('RepeatCount', repeats);
  config.addTemplateInt('KPackedWidth', keyState.indices.shape[3]);
  config.addTemplateInt('VPackedWidth', valueState.indices.shape[3]);
  final outputs = kernel.apply([
    qRotFlat,
    keyState.norms,
    keyState.indices,
    keyCodec.codebook,
    valueState.norms,
    valueState.indices,
    valueCodec.codebook,
  ], config);
  final out = outputs.first;
  final outRotated = out.reshape([batch, kvHeads, repeats, dim]);
  final rotatedBack = valueCodec._rotateInverse(outRotated);
  final reshaped = rotatedBack.reshape([batch, qHeads, 1, valueDim]);
  try {
    // Materialize the fused decode result before releasing upstream kernel
    // outputs; on iPhone the lazy graph can later fail inside residual2 eval.
    MlxRuntime.evalAll([reshaped]);
    return reshaped;
  } catch (_) {
    reshaped.close();
    rethrow;
  } finally {
    rotatedBack.close();
    outRotated.close();
    out.close();
    qRotFlat.close();
  }
}

MlxArray? _turboMseWeightedSumFromScoresDirect(
  MlxArray scores,
  _TurboQuantMseState state,
  _TurboQuantMseCodec codec,
) {
  final weights = mx.softmax(scores, axis: -1);
  final out = _turboMseWeightedSum(weights, state, codec);
  weights.close();
  return out;
}

MlxArray? _turboMseWeightedSumFromScores(
  MlxArray scores,
  _TurboQuantMseState state,
  _TurboQuantMseCodec codec,
  MlxArray maxScores,
) {
  if (!MlxMetal.isAvailable() || _turboDisableFastValuePath) return null;
  if (scores.ndim != 5 || scores.shape[3] != 1 || state.norms.shape[2] == 0) {
    return null;
  }
  final scores2d = scores.reshape([
    scores.shape[0],
    scores.shape[1],
    scores.shape[2],
    scores.shape[4],
  ]);
  final batch = scores2d.shape[0];
  final kvHeads = scores2d.shape[1];
  final repeats = scores2d.shape[2];
  if (repeats <= 1) {
    scores2d.close();
    return null;
  }
  final kernel = _getTurboWeightedRotSumFromScoresRepeatKernel(repeats);
  final config = mx.fast.metalConfig();
  config.addOutputArg([
    batch,
    kvHeads,
    repeats,
    codec.dim,
  ], MlxDType.MLX_FLOAT32);
  config.setGrid(32, codec.dim, batch * kvHeads);
  config.setThreadGroup(32, 1, 1);
  config.addTemplateInt('Dim', codec.dim);
  config.addTemplateInt('Bits', codec.bits);
  config.addTemplateInt('PackedWidth', state.indices.shape[3]);
  final outputs = kernel.apply([
    scores2d,
    state.norms,
    state.indices,
    codec.codebook,
    maxScores,
  ], config);
  scores2d.close();
  final weightedRot = outputs.first;
  final output = mx.matmul(weightedRot, codec.rotation);
  weightedRot.close();
  final expanded = output.expandDims(3);
  output.close();
  return expanded;
}

MlxArray _turboMaxLastAxis(MlxArray input) {
  final length = input.shape.last;
  final flat = input.reshape([input.size ~/ length, length]);
  final kernel = _turboMaxLastAxisKernel ??= mx.fast.metalKernel(
    'turbo_reduce_max_last_axis',
    ['input'],
    ['out'],
    r'''
auto lane = thread_position_in_grid.x;
auto row = thread_position_in_grid.y;
if (row >= input_shape[0]) return;
auto ptr = input + row * Length;
float acc = -INFINITY;
for (int i = lane; i < Length; i += 32) {
  acc = max(acc, static_cast<float>(ptr[i]));
}
acc = simd_max(acc);
if (thread_index_in_simdgroup == 0) out[row] = acc;
''',
  );
  final config = mx.fast.metalConfig();
  config.addOutputArg([flat.shape[0]], MlxDType.MLX_FLOAT32);
  config.setGrid(32, flat.shape[0], 1);
  config.setThreadGroup(32, 1, 1);
  config.addTemplateInt('Length', length);
  final outputs = kernel.apply([flat], config);
  flat.close();
  final out = outputs.first.reshape(input.shape.sublist(0, input.ndim - 1));
  outputs.first.close();
  return out;
}

_TurboQuantMseState _turboAllocateStateLike(
  _TurboQuantMseState state,
  int length,
) => _TurboQuantMseState(
  MlxArray.zeros([
    state.norms.shape[0],
    state.norms.shape[1],
    length,
  ], dtype: state.norms.dtype),
  MlxArray.zeros([
    state.indices.shape[0],
    state.indices.shape[1],
    length,
    state.indices.shape[3],
  ], dtype: state.indices.dtype),
);

_TurboQuantMseState _turboSliceState(_TurboQuantMseState state, int length) =>
    _TurboQuantMseState(
      state.norms.slice(
        start: [0, 0, 0],
        stop: [state.norms.shape[0], state.norms.shape[1], length],
      ),
      state.indices.slice(
        start: [0, 0, 0, 0],
        stop: [
          state.indices.shape[0],
          state.indices.shape[1],
          length,
          state.indices.shape[3],
        ],
      ),
    );

_TurboQuantMseState _turboSliceStateRange(
  _TurboQuantMseState state,
  int start,
  int stop,
) => _TurboQuantMseState(
  state.norms.slice(
    start: [0, 0, start],
    stop: [state.norms.shape[0], state.norms.shape[1], stop],
  ),
  state.indices.slice(
    start: [0, 0, start, 0],
    stop: [
      state.indices.shape[0],
      state.indices.shape[1],
      stop,
      state.indices.shape[3],
    ],
  ),
);

_TurboQuantMseState _turboTakeState(
  _TurboQuantMseState state,
  MlxArray indices,
) {
  final gatheredNorms = state.norms.take(indices, axis: 2);
  final gatheredIndices = state.indices.take(indices, axis: 2);
  final length = indices.shape[0];
  final compacted = _TurboQuantMseState(
    MlxArray.zeros([
      state.norms.shape[0],
      state.norms.shape[1],
      length,
    ], dtype: state.norms.dtype),
    MlxArray.zeros([
      state.indices.shape[0],
      state.indices.shape[1],
      length,
      state.indices.shape[3],
    ], dtype: state.indices.dtype),
  );
  final nextNorms = compacted.norms.sliceUpdate(
    gatheredNorms,
    start: [0, 0, 0],
    stop: [compacted.norms.shape[0], compacted.norms.shape[1], length],
  );
  final nextIndices = compacted.indices.sliceUpdate(
    gatheredIndices,
    start: [0, 0, 0, 0],
    stop: [
      compacted.indices.shape[0],
      compacted.indices.shape[1],
      length,
      compacted.indices.shape[3],
    ],
  );
  compacted.norms.close();
  compacted.indices.close();
  gatheredNorms.close();
  gatheredIndices.close();
  compacted.norms = nextNorms;
  compacted.indices = nextIndices;
  return compacted;
}

void _turboWriteState(
  _TurboQuantMseState dst,
  _TurboQuantMseState src,
  int start,
) {
  final end = start + src.norms.shape[2];
  final nextNorms = dst.norms.sliceUpdate(
    src.norms,
    start: [0, 0, start],
    stop: [dst.norms.shape[0], dst.norms.shape[1], end],
  );
  final nextIndices = dst.indices.sliceUpdate(
    src.indices,
    start: [0, 0, start, 0],
    stop: [
      dst.indices.shape[0],
      dst.indices.shape[1],
      end,
      dst.indices.shape[3],
    ],
  );
  dst.norms.close();
  dst.indices.close();
  dst.norms = nextNorms;
  dst.indices = nextIndices;
  if (_turboDetachStateEachUpdate) {
    final detachedNorms = _turboDetachArray(dst.norms);
    final detachedIndices = _turboDetachArray(dst.indices);
    dst.norms.close();
    dst.indices.close();
    dst.norms = detachedNorms;
    dst.indices = detachedIndices;
  }
}

MlxArray _turboDetachArray(MlxArray input) {
  final zeros = MlxArray.zeros(input.shape, dtype: input.dtype);
  try {
    return input + zeros;
  } finally {
    zeros.close();
  }
}

final class _TurboUpdateTrace {
  _TurboUpdateTrace._(
    this.offset,
    this._active,
    this._cache,
    this._peak,
    this._resourceCount,
    this._commitCount,
    this._watch,
  );

  factory _TurboUpdateTrace.start(int offset) => _TurboUpdateTrace._(
    offset,
    _safeActiveBytes(),
    _safeCacheBytes(),
    _safePeakBytesTrace(),
    _safeResourceCountTrace(),
    _safeCommitCountTrace(),
    Stopwatch()..start(),
  );

  final int offset;
  int _active;
  int _cache;
  int _peak;
  int _resourceCount;
  int _commitCount;
  final Stopwatch _watch;

  void mark(String step) {
    final sink = PaddleOcrVlDebugOverrides.traceSink;
    final active = _safeActiveBytes();
    final cache = _safeCacheBytes();
    final peak = _safePeakBytesTrace();
    final resources = _safeResourceCountTrace();
    final commits = _safeCommitCountTrace();
    final line =
        'turboUpdate '
        'offset=$offset '
        'step=$step '
        'ms=${_watch.elapsedMilliseconds} '
        'active=${_formatApproxBytes(active)} '
        'cache=${_formatApproxBytes(cache)} '
        'peak=${_formatApproxBytes(peak)} '
        'dActive=${_formatSignedBytes(active - _active)} '
        'dCache=${_formatSignedBytes(cache - _cache)} '
        'dPeak=${_formatSignedBytes(peak - _peak)} '
        'dRsrc=${resources - _resourceCount} '
        'dCommits=${commits - _commitCount}';
    if (sink != null) {
      sink(line);
    } else {
      stdout.writeln(line);
    }
    _active = active;
    _cache = cache;
    _peak = peak;
    _resourceCount = resources;
    _commitCount = commits;
  }

  void finish() => _watch.stop();
}

_TurboQuantMseState _turboReserveStateCapacity(
  _TurboQuantMseState state,
  int used,
  int needed,
  int step,
) {
  final capacity = state.norms.shape[2];
  if (capacity >= needed) return state;
  final newCapacity = ((needed + step - 1) ~/ step) * step;
  final grown = _turboAllocateStateLike(state, newCapacity);
  if (used > 0) {
    final existing = _turboSliceState(state, used);
    try {
      _turboWriteState(grown, existing, 0);
    } finally {
      existing.close();
    }
  }
  state.close();
  return grown;
}

MlxArray _turboRhtSignVector(int dim, int seed) {
  final baked = _turboBakedSigns(dim, seed);
  if (baked != null) {
    return MlxArray.fromFloat32List(baked, shape: [dim]);
  }
  final random = math.Random(seed + dim * 7919);
  final values = Float32List(dim);
  for (var i = 0; i < dim; i++) {
    values[i] = random.nextBool() ? 1.0 : -1.0;
  }
  return MlxArray.fromFloat32List(values, shape: [dim]);
}

MlxArray _turboRotationMatrix(int dim, int seed) {
  final baked = _turboBakedRotationData(dim, seed);
  if (baked != null) {
    return MlxArray.fromFloat32List(baked, shape: [dim, dim]);
  }
  final eye = Float32List(dim * dim);
  for (var i = 0; i < dim; i++) {
    eye[i * dim + i] = 1.0;
  }
  return MlxArray.fromFloat32List(eye, shape: [dim, dim]);
}

MlxArray _turboCodebook(int dim, int bits) {
  final baked = _turboBakedCodebook(dim, bits);
  if (baked != null) {
    return MlxArray.fromFloat32List(baked, shape: [baked.length]);
  }
  final levels = 1 << bits;
  if (levels <= 1) {
    return MlxArray.zeros([levels], dtype: MlxDType.MLX_FLOAT32);
  }
  final gridSize = 4096;
  final grid = Float64List(gridSize);
  final weights = Float64List(gridSize);
  for (var i = 0; i < gridSize; i++) {
    final x = -1.0 + (2.0 * i + 1.0) / gridSize;
    grid[i] = x;
    weights[i] = dim <= 1
        ? 1.0
        : math.exp(
            _turboLogGamma(dim / 2.0) -
                0.5 * math.log(math.pi) -
                _turboLogGamma((dim - 1) / 2.0) +
                ((dim - 3) / 2.0) * math.log(math.max(1e-30, 1.0 - (x * x))),
          );
  }
  var maxWeight = 0.0;
  for (final weight in weights) {
    if (weight > maxWeight) maxWeight = weight;
  }
  var sum = 0.0;
  for (var i = 0; i < weights.length; i++) {
    weights[i] = math.exp(math.log(weights[i]) - math.log(maxWeight));
    sum += weights[i];
  }
  final cdf = Float64List(gridSize);
  var running = 0.0;
  for (var i = 0; i < gridSize; i++) {
    running += weights[i] / sum;
    cdf[i] = running;
  }
  final centroids = Float32List(levels);
  for (var i = 0; i < levels; i++) {
    final q = (i + 0.5) / levels;
    var idx = 0;
    while (idx < cdf.length && cdf[idx] < q) {
      idx++;
    }
    centroids[i] = grid[math.min(idx, grid.length - 1)].toDouble();
  }
  return MlxArray.fromFloat32List(centroids, shape: [levels]);
}

Float32List? _turboBakedSigns(int dim, int seed) {
  if (dim != 128) return null;
  if (seed == 0) {
    return Float32List.fromList(
      [
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,
        1,
        -1,
        1,
        1,
        -1,
      ].map((e) => e.toDouble()).toList(),
    );
  }
  if (seed == 1) {
    return Float32List.fromList(
      [
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
      ].map((e) => e.toDouble()).toList(),
    );
  }
  return null;
}

Float32List? _turboBakedCodebook(int dim, int bits) {
  if (dim != 128) return null;
  if (bits == 3) {
    return Float32List.fromList([
      -0.188288122,
      -0.118012108,
      -0.066479988,
      -0.021594388,
      0.021503456,
      0.066420421,
      0.117953405,
      0.188235566,
    ]);
  }
  if (bits == 4) {
    return Float32List.fromList([
      -0.236391634,
      -0.179339990,
      -0.140236318,
      -0.108817965,
      -0.081575535,
      -0.056818351,
      -0.033570420,
      -0.011142485,
      0.011021041,
      0.033449013,
      0.056725606,
      0.081514940,
      0.108757496,
      0.140176103,
      0.179280475,
      0.236336887,
    ]);
  }
  return null;
}

double _turboLogGamma(double z) {
  const coeffs = [
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return math.log(math.pi) -
        math.log(math.sin(math.pi * z)) -
        _turboLogGamma(1.0 - z);
  }
  var x = 0.99999999999980993;
  final tZ = z - 1.0;
  for (var i = 0; i < coeffs.length; i++) {
    x += coeffs[i] / (tZ + i + 1.0);
  }
  final t = tZ + coeffs.length - 0.5;
  return 0.5 * math.log(2.0 * math.pi) +
      ((tZ + 0.5) * math.log(t)) -
      t +
      math.log(x);
}

MlxArray _turboMidpoints(MlxArray codebook) {
  if (codebook.shape[0] <= 1) {
    return MlxArray.zeros([0], dtype: MlxDType.MLX_FLOAT32);
  }
  final left = codebook.slice(start: [0], stop: [codebook.shape[0] - 1]);
  final right = codebook.slice(start: [1], stop: [codebook.shape[0]]);
  final sum = left + right;
  left.close();
  right.close();
  final half = MlxArray.full([], 0.5, dtype: MlxDType.MLX_FLOAT32);
  final out = sum * half;
  half.close();
  sum.close();
  return out;
}

int _turboPackedWidth(int length, int bits) =>
    bits <= 0 ? 0 : ((length * bits) + 31) ~/ 32;

MlxArray _turboPackLowbit(MlxArray values, int bits) {
  final length = values.shape.last;
  final packedWidth = _turboPackedWidth(length, bits);
  final flat = values
      .reshape([values.size ~/ length, length])
      .astype(MlxDType.MLX_UINT32);
  final kernel = _getTurboPackKernel();
  final config = mx.fast.metalConfig();
  config.addOutputArg([flat.shape[0], packedWidth], MlxDType.MLX_UINT32);
  config.setGrid(packedWidth, flat.shape[0], 1);
  config.setThreadGroup(math.min(32, math.max(1, packedWidth)), 1, 1);
  config.addTemplateInt('Bits', bits);
  config.addTemplateInt('Length', length);
  config.addTemplateInt('PackedWidth', packedWidth);
  final outputs = kernel.apply([flat], config);
  flat.close();
  final out = outputs.first.reshape([
    ...values.shape.sublist(0, values.ndim - 1),
    packedWidth,
  ]);
  outputs.first.close();
  return out;
}

MlxArray _turboUnpackLowbit(MlxArray packed, int bits, int length) {
  final flat = packed
      .reshape([packed.size ~/ packed.shape.last, packed.shape.last])
      .astype(MlxDType.MLX_UINT32);
  final kernel = _getTurboUnpackKernel();
  final config = mx.fast.metalConfig();
  config.addOutputArg([flat.shape[0], length], MlxDType.MLX_UINT32);
  config.setGrid(length, flat.shape[0], 1);
  config.setThreadGroup(32, 1, 1);
  config.addTemplateInt('Bits', bits);
  config.addTemplateInt('Length', length);
  config.addTemplateInt('PackedWidth', flat.shape.last);
  final outputs = kernel.apply([flat], config);
  flat.close();
  final out = outputs.first.reshape([
    ...packed.shape.sublist(0, packed.ndim - 1),
    length,
  ]);
  outputs.first.close();
  return out;
}

MlxMetalKernel _getTurboPackKernel() => _turboPackKernel ??= mx.fast
    .metalKernel('turbo_pack_lowbit', ['values'], ['out'], r'''
auto word = thread_position_in_grid.x;
auto row = thread_position_in_grid.y;
if (row >= values_shape[0] || word >= PackedWidth) return;
auto values_ptr = values + row * Length;
uint packed_word = 0u;
int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);
for (int idx = start; idx < end; ++idx) {
  int bit_offset = idx * Bits;
  int word_idx = bit_offset / 32;
  int offset = bit_offset % 32;
  uint value = values_ptr[idx] & ((1u << Bits) - 1u);
  if (word_idx == word) packed_word |= value << offset;
  if (word_idx + 1 == word) {
    int spill = offset + Bits - 32;
    if (spill > 0) packed_word |= value >> (Bits - spill);
  }
}
out[row * PackedWidth + word] = packed_word;
''');

MlxMetalKernel _getTurboUnpackKernel() => _turboUnpackKernel ??= mx.fast
    .metalKernel('turbo_unpack_lowbit', ['packed'], ['out'], r'''
auto idx = thread_position_in_grid.x;
auto row = thread_position_in_grid.y;
if (row >= packed_shape[0] || idx >= Length) return;
auto packed_ptr = packed + row * PackedWidth;
int bit_offset = idx * Bits;
int word_idx = bit_offset / 32;
int offset = bit_offset % 32;
uint value = packed_ptr[word_idx] >> offset;
int spill = offset + Bits - 32;
if (spill > 0) value |= packed_ptr[word_idx + 1] << (Bits - spill);
out[row * Length + idx] = value & ((1u << Bits) - 1u);
''');

MlxArray? _turboMseScore(
  MlxArray queries,
  _TurboQuantMseState state,
  int bits,
  MlxArray codebook,
) {
  if (!MlxMetal.isAvailable()) return null;
  final tokenCount = state.norms.shape[2];
  final packedWidth = state.indices.shape[3];
  final kernel = _turboMseScoreKernel ??= mx.fast.metalKernel(
    'turbo_mse_score',
    ['q_rot', 'norms', 'packed', 'codebook'],
    ['out'],
    r'''
auto lane = thread_position_in_grid.x;
auto repeat_idx = thread_position_in_grid.y;
auto n = thread_position_in_grid.z;
auto token_count = norms_shape[2];
auto kv_heads = norms_shape[1];
auto repeat_count = q_rot_shape[2];
if (repeat_idx >= repeat_count) return;
auto b = n / (kv_heads * token_count);
auto rem = n % (kv_heads * token_count);
auto h = rem / token_count;
auto t = rem % token_count;
auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
auto packed_ptr = packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;
float acc = 0.0f;
for (int d = lane; d < Dim; d += 32) {
  int bit_offset = d * Bits;
  int word_idx = bit_offset / 32;
  int offset = bit_offset % 32;
  uint value = packed_ptr[word_idx] >> offset;
  int spill = offset + Bits - 32;
  if (spill > 0) value |= packed_ptr[word_idx + 1] << (Bits - spill);
  value &= ((1u << Bits) - 1u);
  acc += static_cast<float>(q_ptr[d]) * codebook[value];
}
acc = simd_sum(acc);
if (thread_index_in_simdgroup == 0) {
  out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
      acc * static_cast<float>(norms[(b * kv_heads + h) * token_count + t]);
}
''',
  );
  final config = mx.fast.metalConfig();
  config.addOutputArg([
    queries.shape[0],
    queries.shape[1],
    queries.shape[2],
    tokenCount,
  ], MlxDType.MLX_FLOAT32);
  config.setGrid(
    32,
    queries.shape[2],
    queries.shape[0] * queries.shape[1] * tokenCount,
  );
  config.setThreadGroup(32, 1, 1);
  config.addTemplateInt('Bits', bits);
  config.addTemplateInt('Dim', queries.shape[3]);
  config.addTemplateInt('PackedWidth', packedWidth);
  final outputs = kernel.apply([
    queries,
    state.norms,
    state.indices,
    codebook,
  ], config);
  return outputs.first;
}

(_TurboQuantMseState, _TurboQuantMseState)? _turboTryFusedKvQuantize(
  MlxArray keys,
  MlxArray values,
  _TurboQuantMseCodec keyCodec,
  _TurboQuantMseCodec valueCodec,
) {
  if (!MlxMetal.isAvailable() || keys.shape[2] != 1) return null;
  if (_turboDisableFusedKvPath) return null;
  final keyBits = keyCodec.bits;
  final valBits = valueCodec.bits;
  final kernel = _getTurboFusedKvKernel(keyBits, valBits);
  final dim = keys.shape.last;
  final kFlat = keys.reshape([-1, dim]);
  final vFlat = values.reshape([-1, dim]);
  final bh = kFlat.shape[0];
  final kPackedWidth = _turboPackedWidth(dim, keyBits);
  final vPackedWidth = _turboPackedWidth(dim, valBits);
  final config = mx.fast.metalConfig();
  config.addOutputArg([bh], MlxDType.MLX_FLOAT16);
  config.addOutputArg([bh, kPackedWidth], MlxDType.MLX_UINT32);
  config.addOutputArg([bh], MlxDType.MLX_FLOAT16);
  config.addOutputArg([bh, vPackedWidth], MlxDType.MLX_UINT32);
  config.setGrid(dim * bh, 2, 1);
  config.setThreadGroup(dim, 1, 1);
  config.addTemplateInt('Dim', dim);
  config.addTemplateInt('KPackedWidth', kPackedWidth);
  config.addTemplateInt('VPackedWidth', vPackedWidth);
  final outputs = kernel.apply([
    kFlat,
    vFlat,
    keyCodec.rotation,
    valueCodec.rotation,
    keyCodec.midpoints,
    valueCodec.midpoints,
  ], config);
  kFlat.close();
  vFlat.close();
  final orig = [...keys.shape.sublist(0, keys.ndim - 1)];
  final keyState = _TurboQuantMseState(
    outputs[0].reshape(orig),
    outputs[1].reshape([...orig, kPackedWidth]),
  );
  final valueState = _TurboQuantMseState(
    outputs[2].reshape(orig),
    outputs[3].reshape([...orig, vPackedWidth]),
  );
  return (keyState, valueState);
}

MlxMetalKernel _getTurboFusedKvKernel(int keyBits, int valBits) =>
    (_turboFusedKvKernels['$keyBits:$valBits'] ??= mx.fast.metalKernel(
      'turbo_fused_kv_quantize_k${keyBits}_v$valBits',
      [
        'key_vectors',
        'val_vectors',
        'key_rotation',
        'val_rotation',
        'key_midpoints',
        'val_midpoints',
      ],
      ['out_key_norms', 'out_key_packed', 'out_val_norms', 'out_val_packed'],
      '''
auto d = thread_position_in_threadgroup.x;
auto bh = threadgroup_position_in_grid.x;
auto is_val = threadgroup_position_in_grid.y;
auto sg_id = simdgroup_index_in_threadgroup;
auto sg_lid = thread_index_in_simdgroup;
int bits = is_val ? $valBits : $keyBits;
int n_mid = is_val ? ${(1 << valBits) - 1} : ${(1 << keyBits) - 1};
uint idx_mask = is_val ? ${(1 << valBits) - 1}u : ${(1 << keyBits) - 1}u;
int pw = is_val ? VPackedWidth : KPackedWidth;
float v = 0.0f;
if (is_val)
  v = (d < Dim) ? static_cast<float>(val_vectors[bh * Dim + d]) : 0.0f;
else
  v = (d < Dim) ? static_cast<float>(key_vectors[bh * Dim + d]) : 0.0f;
float sq = v * v;
float sg_sum = simd_sum(sq);
threadgroup float sg_norms[8];
if (sg_lid == 0) sg_norms[sg_id] = sg_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total_sq = (sg_id == 0 && sg_lid < 8) ? sg_norms[sg_lid] : 0.0f;
total_sq = simd_sum(total_sq);
if (sg_id == 0 && sg_lid == 0) sg_norms[0] = total_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);
float norm = sqrt(sg_norms[0]);
float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
if (d == 0) {
  if (is_val) out_val_norms[bh] = half(norm);
  else out_key_norms[bh] = half(norm);
}
threadgroup float shared[Dim];
if (d < Dim) shared[d] = v * inv_norm;
threadgroup_barrier(mem_flags::mem_threadgroup);
float rotated = 0.0f;
if (d < Dim) {
  auto row = is_val ? (val_rotation + d * Dim) : (key_rotation + d * Dim);
  for (int j = 0; j < (int)Dim; j++) rotated += shared[j] * row[j];
}
threadgroup uint shared_idx[Dim];
uint idx = 0;
if (d < Dim) {
  if (is_val) {
    for (int m = 0; m < n_mid; m++) idx += (rotated > val_midpoints[m]) ? 1u : 0u;
  } else {
    for (int m = 0; m < n_mid; m++) idx += (rotated > key_midpoints[m]) ? 1u : 0u;
  }
  shared_idx[d] = idx;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup uint packed_shared[Dim];
if (d < pw) packed_shared[d] = 0u;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (d < Dim) {
  uint idx_val = shared_idx[d] & idx_mask;
  int bo = d * bits;
  int w = bo >> 5;
  int shift = bo & 31;
  packed_shared[w] |= idx_val << shift;
  if (shift + bits > 32) {
    packed_shared[w + 1] |= idx_val >> (32 - shift);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (d < pw) {
  if (is_val) out_val_packed[bh * pw + d] = packed_shared[d];
  else out_key_packed[bh * pw + d] = packed_shared[d];
}
''',
    ));

MlxMetalKernel _getTurboNoRotQuantizeKernel(int bits) =>
    (_turboNoRotQuantizeKernels[bits] ??= mx.fast.metalKernel(
      'turbo_norot_quantize_$bits',
      ['rotated', 'midpoints'],
      ['out'],
      '''
auto d = thread_position_in_threadgroup.x;
auto bh = threadgroup_position_in_grid.x;
float val = (d < Dim) ? static_cast<float>(rotated[bh * Dim + d]) : 0.0f;
threadgroup uint shared_idx[Dim];
uint idx = 0;
if (d < Dim) {
  for (int m = 0; m < ${(1 << bits) - 1}; m++) idx += (val > midpoints[m]) ? 1u : 0u;
  shared_idx[d] = idx;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup uint packed_shared[PackedWidth];
if (d < PackedWidth) packed_shared[d] = 0u;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (d < Dim) {
  uint idx_val = shared_idx[d] & ${(1 << bits) - 1}u;
  int bo = d * $bits;
  int w = bo >> 5;
  int shift = bo & 31;
  packed_shared[w] |= idx_val << shift;
  if (shift + $bits > 32) {
    packed_shared[w + 1] |= idx_val >> (32 - shift);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (d < PackedWidth) out[bh * PackedWidth + d] = packed_shared[d];
''',
    ));

MlxMetalKernel _getTurboFusedMseDecodeKernel(
  int keyBits,
  int valBits,
  int dim,
) => (_turboFusedMseDecodeKernels['$keyBits:$valBits:$dim'] ??= mx.fast
    .metalKernel(
      'turbo_fused_mse_decode_k${keyBits}_v${valBits}_d$dim',
      [
        'queries',
        'key_norms',
        'key_packed',
        'key_codebook',
        'val_norms',
        'val_packed',
        'val_codebook',
      ],
      ['out'],
      _turboFusedMseDecodeKernelSource(keyBits, valBits, dim),
    ));

String _turboFusedMseDecodeKernelSource(int keyBits, int valBits, int dim) {
  final elemsPerLane = dim ~/ 32;
  final keyBitOffVar = (elemsPerLane * keyBits) % 8 == 0 ? '' : 'k_bit_off';
  final valBitOffVar = (elemsPerLane * valBits) % 8 == 0 ? '' : 'v_bit_off';
  return '''
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int qk_per_thread = Dim / BD;
constexpr int v_per_thread = Dim / BD;
constexpr uint k_mask = ${(1 << keyBits) - 1}u;
constexpr uint v_mask = ${(1 << valBits) - 1}u;
constexpr int k_bits = $keyBits;
constexpr int v_bits = $valBits;
typedef float U;
auto bqh = threadgroup_position_in_grid.x;
auto simd_gid = simdgroup_index_in_threadgroup;
auto simd_lid = thread_index_in_simdgroup;
auto token_count = key_norms_shape[2];
auto kv_heads = key_norms_shape[1];
auto bh = bqh / RepeatCount;
auto k_nm = key_norms + bh * token_count;
auto k_pk = key_packed + bh * token_count * KPackedWidth;
auto v_nm = val_norms + bh * token_count;
auto v_pk = val_packed + bh * token_count * VPackedWidth;
threadgroup U max_scores[BN];
threadgroup U sum_exp_scores[BN];
threadgroup U shared[BN * BD];
thread U q[qk_per_thread];
auto qr = queries + bqh * Dim + simd_lid * qk_per_thread;
for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(qr[i]);
thread U o[v_per_thread] = {};
U max_score = -INFINITY;
U sum_exp_score = 0;
int k_bit_start = simd_lid * qk_per_thread * k_bits;
int v_bit_start = simd_lid * v_per_thread * v_bits;
int k_byte_base = k_bit_start >> 3;
int v_byte_base = v_bit_start >> 3;
${keyBitOffVar.isNotEmpty ? 'int k_bit_off = k_bit_start & 7;' : ''}
${valBitOffVar.isNotEmpty ? 'int v_bit_off = v_bit_start & 7;' : ''}
for (int t = simd_gid; t < (int)token_count; t += BN) {
  U kn = static_cast<U>(k_nm[t]);
  auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth) + k_byte_base;
  U score = ${_turboGenUnrolledScore(keyBits, elemsPerLane, keyBitOffVar)};
  score = simd_sum(score) * kn;
  auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth) + v_byte_base;
  U vn = static_cast<U>(v_nm[t]);
  U new_max = max(max_score, score);
  U factor = fast::exp(max_score - new_max);
  U exp_score = fast::exp(score - new_max);
  max_score = new_max;
  sum_exp_score = sum_exp_score * factor + exp_score;
  ${_turboGenUnrolledValue(valBits, elemsPerLane, valBitOffVar)}
}
if (simd_lid == 0) {
  max_scores[simd_gid] = max_score;
  sum_exp_scores[simd_gid] = sum_exp_score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
U sg_max = max_scores[simd_lid];
U new_max = simd_max(sg_max);
U factor = fast::exp(sg_max - new_max);
U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);
U my_factor = fast::exp(max_score - new_max);
for (int i = 0; i < v_per_thread; i++) {
  shared[simd_lid * BD + simd_gid] = o[i] * my_factor;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  o[i] = simd_sum(shared[simd_gid * BD + simd_lid]);
  o[i] = total_sum > 0 ? o[i] / total_sum : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (simd_lid == 0) {
  for (int i = 0; i < v_per_thread; i++) {
    out[bqh * Dim + simd_gid * v_per_thread + i] = static_cast<U>(o[i]);
  }
}
''';
}

String _turboGenUnrolledScore(int bits, int nElems, String bitOffVar) {
  final exprs = _turboGenUnrolledExtract(
    bits,
    nElems,
    'key_codebook',
    bitOffVar,
  );
  return List.generate(
    exprs.length,
    (i) => 'q[$i] * ${exprs[i]}',
  ).join('\n                + ');
}

String _turboGenUnrolledValue(int bits, int nElems, String bitOffVar) {
  final exprs = _turboGenUnrolledExtract(
    bits,
    nElems,
    'val_codebook',
    bitOffVar,
  ).map((e) => e.replaceAll('kb[', 'vb[')).toList(growable: false);
  return List.generate(
    exprs.length,
    (i) => 'o[$i] = o[$i] * factor + exp_score * ${exprs[i]} * vn;',
  ).join('\n  ');
}

List<String> _turboGenUnrolledExtract(
  int bits,
  int nElems,
  String codebookName,
  String bitOffVar,
) {
  final mask = (1 << bits) - 1;
  final exprs = <String>[];
  if (bitOffVar.isEmpty) {
    for (var i = 0; i < nElems; i++) {
      final bitOffset = i * bits;
      final byteIdx = bitOffset ~/ 8;
      final bitInByte = bitOffset % 8;
      if (bitInByte + bits <= 8) {
        if (bitInByte == 0) {
          exprs.add('$codebookName[kb[$byteIdx] & ${mask}u]');
        } else {
          exprs.add('$codebookName[(kb[$byteIdx] >> $bitInByte) & ${mask}u]');
        }
      } else {
        final lowBits = 8 - bitInByte;
        final highMask = (1 << (bits - lowBits)) - 1;
        exprs.add(
          '$codebookName[((kb[$byteIdx] >> $bitInByte) & ${((1 << lowBits) - 1)}u) | ((kb[${byteIdx + 1}] & ${highMask}u) << $lowBits)]',
        );
      }
    }
  } else {
    for (var i = 0; i < nElems; i++) {
      final bo = i == 0 ? bitOffVar : '($bitOffVar + ${i * bits})';
      final by = '($bo >> 3)';
      final bi = '($bo & 7)';
      exprs.add(
        '$codebookName[((($bi + $bits <= 8) ? (kb[$by] >> $bi) : ((kb[$by] >> $bi) | (kb[$by + 1] << (8 - $bi)))) & ${mask}u)]',
      );
    }
  }
  return exprs;
}

MlxArray? _turboMseWeightedSum(
  MlxArray weights,
  _TurboQuantMseState state,
  _TurboQuantMseCodec codec,
) {
  if (weights.ndim != 5 || weights.shape[3] != 1 || state.norms.shape[2] == 0) {
    return null;
  }
  final weights2d = weights.reshape([
    weights.shape[0],
    weights.shape[1],
    weights.shape[2],
    weights.shape[4],
  ]);
  final batch = weights2d.shape[0];
  final kvHeads = weights2d.shape[1];
  final repeats = weights2d.shape[2];
  if (repeats > 1) {
    if (MlxMetal.isAvailable() && !_turboDisableFastValuePath) {
      final kernel = _getTurboWeightedRotRepeatKernel(repeats);
      final config = mx.fast.metalConfig();
      config.addOutputArg([
        batch,
        kvHeads,
        repeats,
        codec.dim,
      ], MlxDType.MLX_FLOAT32);
      config.setGrid(32, codec.dim, batch * kvHeads);
      config.setThreadGroup(32, 1, 1);
      config.addTemplateInt('Dim', codec.dim);
      config.addTemplateInt('Bits', codec.bits);
      config.addTemplateInt('PackedWidth', state.indices.shape[3]);
      config.addTemplateInt('RepeatCount', repeats);
      final outputs = kernel.apply([
        weights2d,
        state.norms,
        state.indices,
        codec.codebook,
      ], config);
      weights2d.close();
      final weightedRot = outputs.first;
      final output = mx.matmul(weightedRot, codec.rotation);
      weightedRot.close();
      final expanded = output.expandDims(3);
      output.close();
      return expanded;
    }
  } else if (MlxMetal.isAvailable() && !_turboDisableFastValuePath) {
    final kernel = _getTurboWeightedRotKernel();
    final config = mx.fast.metalConfig();
    config.addOutputArg([
      batch,
      kvHeads,
      repeats,
      codec.dim,
    ], MlxDType.MLX_FLOAT32);
    config.setGrid(32, codec.dim, batch * kvHeads * repeats);
    config.setThreadGroup(32, 1, 1);
    config.addTemplateInt('Dim', codec.dim);
    config.addTemplateInt('Bits', codec.bits);
    config.addTemplateInt('PackedWidth', state.indices.shape[3]);
    final outputs = kernel.apply([
      weights2d,
      state.norms,
      state.indices,
      codec.codebook,
    ], config);
    weights2d.close();
    final weightedRot = outputs.first;
    final output = mx.matmul(weightedRot, codec.rotation);
    weightedRot.close();
    final expanded = output.expandDims(3);
    output.close();
    return expanded;
  }
  weights2d.close();
  final unpacked = _turboUnpackLowbit(
    state.indices,
    codec.bits,
    codec.dim,
  ).astype(MlxDType.MLX_INT32);
  final rotated = codec.codebook.take(unpacked, axis: 0);
  unpacked.close();
  final norms = state.norms.astype(MlxDType.MLX_FLOAT32);
  final weightedRot = mx.einsum('bhmlt,bht,bhtd->bhmld', [
    weights,
    norms,
    rotated,
  ]);
  norms.close();
  rotated.close();
  final output = codec._rotateInverse(weightedRot);
  weightedRot.close();
  return output;
}

MlxMetalKernel _getTurboWeightedRotRepeatKernel(int repeats) =>
    (_turboWeightedRotRepeatKernels[repeats] ??= mx.fast.metalKernel(
      'turbo_mse_weighted_rot_repeat_$repeats',
      ['weights', 'norms', 'packed', 'codebook'],
      ['out'],
      _turboWeightedRotRepeatKernelSource(repeats),
    ));

MlxMetalKernel _getTurboWeightedRotKernel() =>
    _turboWeightedRotKernel ??= mx.fast.metalKernel(
      'turbo_mse_weighted_rot',
      ['weights', 'norms', 'packed', 'codebook'],
      ['out'],
      r'''
auto lane = thread_position_in_grid.x;
auto dim_idx = thread_position_in_grid.y;
auto n = thread_position_in_grid.z;
if (dim_idx >= Dim) return;
auto token_count = norms_shape[2];
auto kv_heads = norms_shape[1];
auto repeat_count = weights_shape[2];
auto b = n / (kv_heads * repeat_count);
auto rem = n % (kv_heads * repeat_count);
auto h = rem / repeat_count;
auto repeat_idx = rem % repeat_count;
auto weights_ptr = weights + ((b * kv_heads + h) * repeat_count + repeat_idx) * token_count;
auto norms_ptr = norms + (b * kv_heads + h) * token_count;
auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;
float acc = 0.0f;
for (int t = lane; t < token_count; t += 32) {
  auto token_ptr = packed_ptr + t * PackedWidth;
  int bit_offset = dim_idx * Bits;
  int word_idx = bit_offset / 32;
  int offset = bit_offset % 32;
  uint value = token_ptr[word_idx] >> offset;
  int spill = offset + Bits - 32;
  if (spill > 0) value |= token_ptr[word_idx + 1] << (Bits - spill);
  value &= ((1u << Bits) - 1u);
  acc += static_cast<float>(weights_ptr[t]) *
      static_cast<float>(norms_ptr[t]) * codebook[value];
}
acc = simd_sum(acc);
if (thread_index_in_simdgroup == 0) {
  out[((b * kv_heads + h) * repeat_count + repeat_idx) * Dim + dim_idx] = acc;
}
''',
    );

MlxMetalKernel _getTurboWeightedRotSumFromScoresRepeatKernel(int repeats) =>
    (_turboWeightedRotSumFromScoresRepeatKernels[repeats] ??= mx.fast
        .metalKernel(
          'turbo_mse_scores_weighted_rot_sum_repeat_$repeats',
          ['scores', 'norms', 'packed', 'codebook', 'max_scores'],
          ['out'],
          _turboWeightedRotSumFromScoresRepeatKernelSource(repeats),
        ));

String _turboWeightedRotRepeatKernelSource(int repeats) {
  final lines = <String>[
    'auto lane = thread_position_in_grid.x;',
    'auto dim_idx = thread_position_in_grid.y;',
    'auto n = thread_position_in_grid.z;',
    '',
    'if (dim_idx >= Dim) return;',
    '',
    'auto token_count = norms_shape[2];',
    'auto kv_heads = norms_shape[1];',
    'auto repeat_count = weights_shape[2];',
    'auto b = n / kv_heads;',
    'auto h = n % kv_heads;',
    '',
    'auto weights_base = weights + ((b * kv_heads + h) * repeat_count) * token_count;',
    'auto norms_ptr = norms + (b * kv_heads + h) * token_count;',
    'auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;',
    '',
    'int bit_offset = dim_idx * Bits;',
    'int word_idx = bit_offset / 32;',
    'int offset = bit_offset % 32;',
    '',
  ];
  for (var r = 0; r < repeats; r++) {
    lines.add('float acc_$r = 0.0f;');
  }
  lines.addAll([
    '',
    'for (int t = lane; t < token_count; t += 32) {',
    '  auto token_ptr = packed_ptr + t * PackedWidth;',
    '  uint value = token_ptr[word_idx] >> offset;',
    '  int spill = offset + Bits - 32;',
    '  if (spill > 0) value |= token_ptr[word_idx + 1] << (Bits - spill);',
    '  value &= ((1u << Bits) - 1u);',
    '  float code = codebook[value];',
    '  float norm = static_cast<float>(norms_ptr[t]);',
  ]);
  for (var r = 0; r < repeats; r++) {
    lines.add(
      '  acc_$r += static_cast<float>(weights_base[$r * token_count + t]) * norm * code;',
    );
  }
  lines.add('}');
  lines.add('');
  for (var r = 0; r < repeats; r++) {
    lines.add('float acc_sum_$r = simd_sum(acc_$r);');
  }
  lines.addAll(['', 'if (thread_index_in_simdgroup == 0) {']);
  for (var r = 0; r < repeats; r++) {
    lines.add(
      '  out[((b * kv_heads + h) * repeat_count + $r) * Dim + dim_idx] = acc_sum_$r;',
    );
  }
  lines.add('}');
  return lines.join('\n');
}

String _turboWeightedRotSumFromScoresRepeatKernelSource(int repeats) {
  final lines = <String>[
    'auto lane = thread_position_in_grid.x;',
    'auto dim_idx = thread_position_in_grid.y;',
    'auto n = thread_position_in_grid.z;',
    '',
    'if (dim_idx >= Dim) return;',
    '',
    'auto token_count = norms_shape[2];',
    'auto kv_heads = norms_shape[1];',
    'auto repeat_count = scores_shape[2];',
    'auto b = n / kv_heads;',
    'auto h = n % kv_heads;',
    '',
    'auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;',
    'auto norms_ptr = norms + (b * kv_heads + h) * token_count;',
    'auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;',
    'auto max_base = max_scores + (b * kv_heads + h) * repeat_count;',
    '',
    'int bit_offset = dim_idx * Bits;',
    'int word_idx = bit_offset / 32;',
    'int offset = bit_offset % 32;',
    '',
  ];
  for (var r = 0; r < repeats; r++) {
    lines.add('float max_score_$r = static_cast<float>(max_base[$r]);');
    lines.add('float acc_$r = 0.0f;');
  }
  lines.addAll([
    '',
    'for (int t = lane; t < token_count; t += 32) {',
    '  auto token_ptr = packed_ptr + t * PackedWidth;',
    '  uint value = token_ptr[word_idx] >> offset;',
    '  int spill = offset + Bits - 32;',
    '  if (spill > 0) value |= token_ptr[word_idx + 1] << (Bits - spill);',
    '  value &= ((1u << Bits) - 1u);',
    '  float code = codebook[value];',
    '  float norm = static_cast<float>(norms_ptr[t]);',
  ]);
  for (var r = 0; r < repeats; r++) {
    lines.add(
      '  float weight_$r = exp(static_cast<float>(scores_base[$r * token_count + t]) - max_score_$r);',
    );
    lines.add('  acc_$r += weight_$r * norm * code;');
  }
  lines.add('}');
  lines.add('');
  for (var r = 0; r < repeats; r++) {
    lines.add('float acc_sum_$r = simd_sum(acc_$r);');
  }
  lines.addAll(['', 'if (thread_index_in_simdgroup == 0) {']);
  for (var r = 0; r < repeats; r++) {
    lines.add(
      '  out[((b * kv_heads + h) * repeat_count + $r) * Dim + dim_idx] = acc_sum_$r;',
    );
  }
  lines.add('}');
  return lines.join('\n');
}

MlxMetalKernel? _turboPackKernel;
MlxMetalKernel? _turboUnpackKernel;
MlxMetalKernel? _turboMseScoreKernel;
MlxMetalKernel? _turboWeightedRotKernel;
MlxMetalKernel? _turboMaxLastAxisKernel;
final Map<int, MlxMetalKernel> _turboWeightedRotRepeatKernels = {};
final Map<int, MlxMetalKernel> _turboWeightedRotSumFromScoresRepeatKernels = {};
final Map<String, MlxMetalKernel> _turboFusedKvKernels = {};
final Map<int, MlxMetalKernel> _turboNoRotQuantizeKernels = {};
final Map<String, MlxMetalKernel> _turboFusedMseDecodeKernels = {};
