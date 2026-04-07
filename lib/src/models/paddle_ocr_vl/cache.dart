part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// KV cache for ERNIE-4.5 autoregressive decoding
//
// Uses pre-allocated buffers with slice_update to avoid O(n) copies per step.
// ---------------------------------------------------------------------------

/// Per-layer KV cache backed by pre-allocated buffers.
///
/// On the first call to [updateAndFetch], the buffer is allocated to
/// `[1, numKvHeads, maxSeqLen, headDim]` and new KV entries are written at the
/// current offset via `sliceUpdate`.  Subsequent calls write into the existing
/// buffer without any concatenation.
///
/// The returned K/V from [updateAndFetch] are *slices* of the buffer up to
/// `offset` — the caller must NOT close them (they alias the buffer).
final class _KvCache {
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

  int get offset => _offset;

  /// Write new K/V into the pre-allocated buffer and return the valid slice
  /// `[0:offset]` along the sequence axis.
  ///
  /// [nextKeys] / [nextValues] shape: `[1, numKvHeads, newTokens, headDim]`.
  /// After this call the cache owns the buffer; the caller must still close
  /// [nextKeys] and [nextValues].
  (MlxArray, MlxArray) updateAndFetch(MlxArray nextKeys, MlxArray nextValues) {
    final newTokens = nextKeys.shape[2];

    // Allocate on first use, matching the dtype of the incoming tensors.
    if (_keys == null) {
      final dt = nextKeys.dtype;
      _keys = MlxArray.zeros([1, numKvHeads, maxSeqLen, headDim], dtype: dt);
      _values = MlxArray.zeros([1, numKvHeads, maxSeqLen, headDim], dtype: dt);
    }

    // Write new tokens at [0:1, 0:numKvHeads, offset:offset+newTokens, 0:headDim]
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

    // Replace old buffers with updated ones.
    _keys!.close();
    _values!.close();
    nextKeys.close();
    nextValues.close();
    _keys = updatedKeys;
    _values = updatedValues;
    _offset += newTokens;

    // Return valid slice [0:offset].
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

  void close() {
    _keys?.close();
    _values?.close();
    _keys = null;
    _values = null;
    _offset = 0;
  }
}

/// Full model decode cache — one [_KvCache] per decoder layer.
final class _ModelCache {
  _ModelCache(this.layers);

  factory _ModelCache.create({
    required int numLayers,
    required int numKvHeads,
    required int headDim,
    required int maxSeqLen,
  }) => _ModelCache(
    List.generate(
      numLayers,
      (_) => _KvCache(
        numKvHeads: numKvHeads,
        headDim: headDim,
        maxSeqLen: maxSeqLen,
      ),
    ),
  );

  final List<_KvCache> layers;

  /// Total number of cached tokens (from the first layer).
  int get offset => layers.isEmpty ? 0 : layers.first.offset;

  void close() {
    for (final layer in layers) {
      layer.close();
    }
  }
}
