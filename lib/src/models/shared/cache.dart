/// Unified KV-cache abstraction for all decoder models.
///
/// Provides a common interface over the three existing cache strategies:
/// - Concatenation-based (Qwen3.5 `_KvDecodeCache`, Qwen3-ASR `AsrKvCache`)
/// - Pre-allocated buffer (PaddleOCR-VL `_KvCache`)
/// - Quantized / TurboQuant (PaddleOCR-VL `_QuantizedKvCache`)
///
/// Models that need cache now implement [LayerCache] and compose it inside a
/// [DecodeCache].
library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

// ---------------------------------------------------------------------------
// Layer-level cache interface
// ---------------------------------------------------------------------------

/// A single layer's decode-time state.
///
/// Sealed so that pattern matching is exhaustive over the supported
/// cache strategies.
sealed class LayerCache {
  /// Number of tokens cached so far.
  int get offset;

  /// Materialise internal tensors (call `eval` on live arrays).
  void evalState();

  /// Release all GPU memory held by this cache.
  void close();
}

/// Standard key-value attention cache using concatenation.
///
/// On each step, new key/value tensors are concatenated onto the running
/// state.  Simple and general but allocation-heavy for long sequences.
final class ConcatKvCache extends LayerCache {
  ConcatKvCache();

  MlxArray? _keys;
  MlxArray? _values;
  int _offset = 0;

  @override
  int get offset => _offset;

  /// Current cached keys, or `null` if empty.
  MlxArray? get keys => _keys;

  /// Current cached values, or `null` if empty.
  MlxArray? get values => _values;

  /// Append new key/value tensors and return the merged pair.
  ///
  /// Both [newKeys] and [newValues] must have shape
  /// `[batch, heads, seqLen, headDim]`.
  ({MlxArray keys, MlxArray values}) updateAndFetch(
    MlxArray newKeys,
    MlxArray newValues,
  ) {
    if (_keys == null || _values == null) {
      _keys = newKeys;
      _values = newValues;
    } else {
      final mergedK = mx.concatenate([_keys!, newKeys], axis: 2);
      final mergedV = mx.concatenate([_values!, newValues], axis: 2);
      _keys!.close();
      _values!.close();
      newKeys.close();
      newValues.close();
      _keys = mergedK;
      _values = mergedV;
    }
    _offset = _keys!.shape[2];
    return (keys: _keys!, values: _values!);
  }

  /// Deep-clone this cache (both key and value tensors are copied).
  ConcatKvCache clone() {
    final copy = ConcatKvCache();
    if (_keys != null) {
      // Copy via add-zero trick: forces a fresh allocation.
      final zeroK = MlxArray.zeros(_keys!.shape, dtype: _keys!.dtype);
      copy._keys = mx.add(_keys!, zeroK);
      zeroK.close();
      MlxRuntime.evalAll([copy._keys!]);

      final zeroV = MlxArray.zeros(_values!.shape, dtype: _values!.dtype);
      copy._values = mx.add(_values!, zeroV);
      zeroV.close();
      MlxRuntime.evalAll([copy._values!]);
    }
    copy._offset = _offset;
    return copy;
  }

  /// Trim the last [count] tokens from the cache.
  void trim(int count) {
    if (count <= 0 || count > _offset) return;
    final newLen = _offset - count;
    if (_keys != null) {
      final k = _keys!;
      _keys = k.slice(
        start: [0, 0, 0, 0],
        stop: [k.shape[0], k.shape[1], newLen, k.shape[3]],
      );
      k.close();
    }
    if (_values != null) {
      final v = _values!;
      _values = v.slice(
        start: [0, 0, 0, 0],
        stop: [v.shape[0], v.shape[1], newLen, v.shape[3]],
      );
      v.close();
    }
    _offset = newLen;
  }

  @override
  void evalState() {
    final toEval = <MlxArray>[];
    if (_keys != null) toEval.add(_keys!);
    if (_values != null) toEval.add(_values!);
    if (toEval.isNotEmpty) MlxRuntime.evalAll(toEval);
  }

  @override
  void close() {
    _keys?.close();
    _values?.close();
    _keys = null;
    _values = null;
    _offset = 0;
  }
}

/// Pre-allocated buffer KV cache using slice-update.
///
/// Allocates a fixed `[batch, heads, maxSeqLen, headDim]` buffer up front
/// and uses [MlxModule.sliceUpdate] to write new tokens at the current
/// offset.  Avoids repeated allocation but requires a known `maxSeqLen`.
final class PreallocKvCache extends LayerCache {
  PreallocKvCache({
    required this.numKvHeads,
    required this.headDim,
    required this.maxSeqLen,
    this.dtype = MlxDType.MLX_FLOAT16,
  });

  final int numKvHeads;
  final int headDim;
  final int maxSeqLen;
  final MlxDType dtype;

  MlxArray? _keys;
  MlxArray? _values;
  int _offset = 0;

  @override
  int get offset => _offset;

  MlxArray? get keys => _keys;
  MlxArray? get values => _values;

  /// Write new tokens at the current offset and return the valid slice.
  ({MlxArray keys, MlxArray values}) updateAndFetch(
    MlxArray newKeys,
    MlxArray newValues,
  ) {
    final seqLen = newKeys.shape[2];
    if (_keys == null) {
      _keys = MlxArray.zeros([1, numKvHeads, maxSeqLen, headDim], dtype: dtype);
      _values = MlxArray.zeros([
        1,
        numKvHeads,
        maxSeqLen,
        headDim,
      ], dtype: dtype);
    }

    // Write at [0, 0, offset, 0].
    final updatedK = mx.sliceUpdate(
      _keys!,
      newKeys,
      start: [0, 0, _offset, 0],
      stop: [1, numKvHeads, _offset + seqLen, headDim],
    );
    final updatedV = mx.sliceUpdate(
      _values!,
      newValues,
      start: [0, 0, _offset, 0],
      stop: [1, numKvHeads, _offset + seqLen, headDim],
    );
    _keys!.close();
    _values!.close();
    newKeys.close();
    newValues.close();
    _keys = updatedK;
    _values = updatedV;
    _offset += seqLen;

    // Slice valid region.
    final validK = _keys!.slice(
      start: [0, 0, 0, 0],
      stop: [1, numKvHeads, _offset, headDim],
    );
    final validV = _values!.slice(
      start: [0, 0, 0, 0],
      stop: [1, numKvHeads, _offset, headDim],
    );
    return (keys: validK, values: validV);
  }

  @override
  void evalState() {
    final toEval = <MlxArray>[];
    if (_keys != null) toEval.add(_keys!);
    if (_values != null) toEval.add(_values!);
    if (toEval.isNotEmpty) MlxRuntime.evalAll(toEval);
  }

  @override
  void close() {
    _keys?.close();
    _values?.close();
    _keys = null;
    _values = null;
    _offset = 0;
  }
}

/// Mamba/linear-attention layer cache (state-machine based).
///
/// Stores a conv state and a recurrent state, with move semantics
/// (take + replace) to avoid accidental aliasing.
final class LinearStateCache extends LayerCache {
  LinearStateCache();

  MlxArray? _convState;
  MlxArray? _state;
  int _offset = 0;

  @override
  int get offset => _offset;

  /// Take ownership of the conv state (nulls the slot).
  MlxArray? takeConvState() {
    final s = _convState;
    _convState = null;
    return s;
  }

  /// Replace the conv state (closes the old one).
  void replaceConvState(MlxArray newState) {
    _convState?.close();
    _convState = newState;
  }

  /// Take ownership of the recurrent state (nulls the slot).
  MlxArray? takeState() {
    final s = _state;
    _state = null;
    return s;
  }

  /// Replace the recurrent state (closes the old one).
  void replaceState(MlxArray newState) {
    _state?.close();
    _state = newState;
    _offset++;
  }

  /// Deep-clone this cache.
  LinearStateCache clone() {
    final copy = LinearStateCache();
    copy._offset = _offset;
    if (_convState != null) {
      final z = MlxArray.zeros(_convState!.shape, dtype: _convState!.dtype);
      copy._convState = mx.add(_convState!, z);
      z.close();
      MlxRuntime.evalAll([copy._convState!]);
    }
    if (_state != null) {
      final z = MlxArray.zeros(_state!.shape, dtype: _state!.dtype);
      copy._state = mx.add(_state!, z);
      z.close();
      MlxRuntime.evalAll([copy._state!]);
    }
    return copy;
  }

  @override
  void evalState() {
    final toEval = <MlxArray>[];
    if (_convState != null) toEval.add(_convState!);
    if (_state != null) toEval.add(_state!);
    if (toEval.isNotEmpty) MlxRuntime.evalAll(toEval);
  }

  @override
  void close() {
    _convState?.close();
    _state?.close();
    _convState = null;
    _state = null;
    _offset = 0;
  }
}

// ---------------------------------------------------------------------------
// Model-level cache container
// ---------------------------------------------------------------------------

/// A complete decode cache spanning all layers of a model.
///
/// Models create a [DecodeCache] at generation start, pass it through each
/// decode step, and close it when done.
final class DecodeCache {
  DecodeCache(this.layers);

  /// One [LayerCache] per model layer.
  final List<LayerCache> layers;

  /// Create a cache with [numLayers] concatenation-based KV caches.
  factory DecodeCache.concat(int numLayers) =>
      DecodeCache(List.generate(numLayers, (_) => ConcatKvCache()));

  /// Create a cache with [numLayers] pre-allocated buffer KV caches.
  factory DecodeCache.prealloc({
    required int numLayers,
    required int numKvHeads,
    required int headDim,
    required int maxSeqLen,
    MlxDType dtype = MlxDType.MLX_FLOAT16,
  }) => DecodeCache(
    List.generate(
      numLayers,
      (_) => PreallocKvCache(
        numKvHeads: numKvHeads,
        headDim: headDim,
        maxSeqLen: maxSeqLen,
        dtype: dtype,
      ),
    ),
  );

  /// Current offset of the first layer (assumes all layers are in sync).
  int get offset => layers.isEmpty ? 0 : layers.first.offset;

  /// Materialise all layer states.
  void evalStates() {
    for (final layer in layers) {
      layer.evalState();
    }
  }

  /// Deep-clone the entire cache.
  DecodeCache clone() {
    final cloned = <LayerCache>[];
    for (final layer in layers) {
      switch (layer) {
        case ConcatKvCache():
          cloned.add(layer.clone());
        case PreallocKvCache():
          // Pre-alloc caches are not typically cloned; fall back to concat.
          throw UnsupportedError(
            'Cloning PreallocKvCache is not supported — '
            'use ConcatKvCache for sessions that need cloning.',
          );
        case LinearStateCache():
          cloned.add(layer.clone());
      }
    }
    return DecodeCache(cloned);
  }

  /// Release all GPU memory.
  void close() {
    for (final layer in layers) {
      layer.close();
    }
  }
}
