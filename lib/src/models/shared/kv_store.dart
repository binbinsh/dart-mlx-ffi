/// Safetensors KV cache persistence.
///
/// Save and restore [DecodeCache] state to/from safetensors files, enabling
/// session resumption across app restarts.
///
/// Inspired by osaurus `KVCacheStore.swift`.  Uses the existing IO
/// capabilities in `MlxIo` for the actual safetensors serialisation.
library;

import 'dart:io';

import 'package:dart_inference/mlx.dart';

import 'cache.dart';

// ---------------------------------------------------------------------------
// KvCacheStore
// ---------------------------------------------------------------------------

/// Persists and restores [DecodeCache] state via safetensors files.
///
/// Each layer's key/value tensors are saved as separate entries:
/// - `layer.{i}.keys`
/// - `layer.{i}.values`
///
/// Metadata in the safetensors header stores the cache configuration:
/// - `num_layers`, `offsets` (comma-separated per-layer offsets),
///   `cache_type` (concat / prealloc / linear).
///
/// Usage:
/// ```dart
/// final store = KvCacheStore(directory: '/tmp/kv_cache');
/// await store.save('session_1', cache);
/// final restored = await store.load('session_1');
/// ```
final class KvCacheStore {
  KvCacheStore({required this.directory}) {
    Directory(directory).createSync(recursive: true);
  }

  /// Root directory where cache files are stored.
  final String directory;

  /// Save a [DecodeCache] to disk under the given [sessionId].
  ///
  /// Overwrites any existing cache for this session.
  void save(String sessionId, DecodeCache cache) {
    final tensors = <String, MlxArray>{};
    final offsets = <String>[];

    for (var i = 0; i < cache.layers.length; i++) {
      final layer = cache.layers[i];
      offsets.add(layer.offset.toString());

      switch (layer) {
        case ConcatKvCache():
          if (layer.keys != null) {
            tensors['layer.$i.keys'] = layer.keys!;
          }
          if (layer.values != null) {
            tensors['layer.$i.values'] = layer.values!;
          }
        case PreallocKvCache():
          if (layer.keys != null) {
            // Only save the valid portion.
            final k = layer.keys!.slice(
              start: [0, 0, 0, 0],
              stop: [
                layer.keys!.shape[0],
                layer.keys!.shape[1],
                layer.offset,
                layer.keys!.shape[3],
              ],
            );
            tensors['layer.$i.keys'] = k;
          }
          if (layer.values != null) {
            final v = layer.values!.slice(
              start: [0, 0, 0, 0],
              stop: [
                layer.values!.shape[0],
                layer.values!.shape[1],
                layer.offset,
                layer.values!.shape[3],
              ],
            );
            tensors['layer.$i.values'] = v;
          }
        case LinearStateCache():
          // Linear state caches don't have standard KV tensors.
          // We could save convState / state here in the future.
          break;
      }
    }

    if (tensors.isEmpty) return;

    final metadata = <String, String>{
      'num_layers': cache.layers.length.toString(),
      'offsets': offsets.join(','),
      'cache_type': _cacheTypeName(cache.layers.first),
      'version': '1',
    };

    final path = _pathFor(sessionId);
    MlxIo.saveSafetensors(path, tensors, metadata: metadata);
  }

  /// Load a [DecodeCache] from disk for the given [sessionId].
  ///
  /// Returns `null` if no saved cache exists.
  DecodeCache? load(String sessionId) {
    final path = _pathFor(sessionId);
    if (!File(path).existsSync()) return null;

    final (:tensors, :metadata) = MlxIo.loadSafetensors(path);
    final numLayers = int.parse(metadata['num_layers'] ?? '0');
    if (numLayers == 0) return null;

    // offsets are stored for future use (e.g. PreallocKvCache restore)
    // but ConcatKvCache derives offset from the tensor shapes.
    final _ = metadata['offsets'];

    final layers = <LayerCache>[];
    for (var i = 0; i < numLayers; i++) {
      final k = tensors['layer.$i.keys'];
      final v = tensors['layer.$i.values'];
      final cache = ConcatKvCache();
      if (k != null && v != null) {
        cache.updateAndFetch(k, v);
      }
      layers.add(cache);
    }

    return DecodeCache(layers);
  }

  /// Delete the saved cache for [sessionId].
  void delete(String sessionId) {
    final file = File(_pathFor(sessionId));
    if (file.existsSync()) file.deleteSync();
  }

  /// List all saved session IDs.
  List<String> listSessions() {
    final dir = Directory(directory);
    if (!dir.existsSync()) return const [];
    return dir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.safetensors'))
        .map((f) {
          final name = f.uri.pathSegments.last;
          return name.replaceAll('.safetensors', '');
        })
        .toList();
  }

  /// Total disk usage in bytes for all saved caches.
  int diskUsageBytes() {
    final dir = Directory(directory);
    if (!dir.existsSync()) return 0;
    var total = 0;
    for (final f in dir.listSync().whereType<File>()) {
      if (f.path.endsWith('.safetensors')) {
        total += f.lengthSync();
      }
    }
    return total;
  }

  String _pathFor(String sessionId) => '$directory/$sessionId.safetensors';

  static String _cacheTypeName(LayerCache layer) {
    return switch (layer) {
      ConcatKvCache() => 'concat',
      PreallocKvCache() => 'prealloc',
      LinearStateCache() => 'linear',
    };
  }
}
