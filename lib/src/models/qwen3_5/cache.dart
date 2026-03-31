part of 'qwen3_5.dart';

sealed class _LayerDecodeCache {
  void close();
}

final class _KvDecodeCache extends _LayerDecodeCache {
  MlxArray? keys;
  MlxArray? values;
  int offset = 0;

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

  @override
  void close() {
    keys?.close();
    values?.close();
    keys = null;
    values = null;
    offset = 0;
  }
}

final class _LinearDecodeCache extends _LayerDecodeCache {
  MlxArray? _convState;
  MlxArray? _state;

  MlxArray? takeConvState() {
    final value = _convState;
    _convState = null;
    return value;
  }

  MlxArray? takeState() {
    final value = _state;
    _state = null;
    return value;
  }

  void replaceConvState(MlxArray value) {
    _convState?.close();
    _convState = value;
  }

  void replaceState(MlxArray value) {
    _state?.close();
    _state = value;
  }

  @override
  void close() {
    _convState?.close();
    _state?.close();
    _convState = null;
    _state = null;
  }
}

final class _ModelDecodeCache {
  _ModelDecodeCache(this.layers);

  final List<_LayerDecodeCache> layers;

  void close() {
    for (final layer in layers) {
      layer.close();
    }
  }
}
