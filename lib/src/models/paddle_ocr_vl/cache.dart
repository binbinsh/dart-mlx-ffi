part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// KV cache for ERNIE-4.5 autoregressive decoding
// ---------------------------------------------------------------------------

/// Per-layer KV cache for a single attention layer.
final class _KvCache {
  MlxArray? keys;
  MlxArray? values;
  int offset = 0;

  /// Append new key/value tensors and return the full accumulated K, V.
  ///
  /// After this call, [nextKeys] and [nextValues] are owned by the cache
  /// and must NOT be closed by the caller.
  (MlxArray, MlxArray) updateAndFetch(MlxArray nextKeys, MlxArray nextValues) {
    final currentKeys = keys;
    final currentValues = values;
    if (currentKeys == null || currentValues == null) {
      keys = nextKeys;
      values = nextValues;
      offset = nextKeys.shape[2];
      return (nextKeys, nextValues);
    }
    final mergedKeys = mx.concatenate([currentKeys, nextKeys], axis: 2);
    final mergedValues = mx.concatenate([currentValues, nextValues], axis: 2);
    currentKeys.close();
    currentValues.close();
    nextKeys.close();
    nextValues.close();
    keys = mergedKeys;
    values = mergedValues;
    offset = mergedKeys.shape[2];
    return (mergedKeys, mergedValues);
  }

  void close() {
    keys?.close();
    values?.close();
    keys = null;
    values = null;
    offset = 0;
  }
}

/// Full model decode cache — one [_KvCache] per decoder layer.
final class _ModelCache {
  _ModelCache(this.layers);

  factory _ModelCache.create(int numLayers) =>
      _ModelCache(List.generate(numLayers, (_) => _KvCache()));

  final List<_KvCache> layers;

  /// Total number of cached tokens (from the first layer).
  int get offset => layers.isEmpty ? 0 : layers.first.offset;

  void close() {
    for (final layer in layers) {
      layer.close();
    }
  }
}
