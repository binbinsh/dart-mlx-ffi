@TestOn('mac-os')
library;

import 'dart:io';

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main() {
  group('KvCacheStore', () {
    late Directory tmpDir;
    late KvCacheStore store;

    setUp(() {
      tmpDir = Directory.systemTemp.createTempSync('kv_store_test_');
      store = KvCacheStore(directory: tmpDir.path);
    });

    tearDown(() {
      tmpDir.deleteSync(recursive: true);
    });

    test('creates directory on construction', () {
      final nested = '${tmpDir.path}/sub/dir';
      KvCacheStore(directory: nested);
      expect(Directory(nested).existsSync(), isTrue);
    });

    test('save and load ConcatKvCache round-trip', () {
      final cache = DecodeCache.concat(2);
      final layer0 = cache.layers[0] as ConcatKvCache;
      final layer1 = cache.layers[1] as ConcatKvCache;

      // Add data to both layers.
      final k0 = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 1, 2, 2]);
      final v0 = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [1, 1, 2, 2]);
      MlxRuntime.evalAll([k0, v0]);
      layer0.updateAndFetch(k0, v0);

      final k1 = MlxArray.fromFloat32List(
        [10, 20, 30, 40],
        shape: [1, 1, 2, 2],
      );
      final v1 = MlxArray.fromFloat32List(
        [50, 60, 70, 80],
        shape: [1, 1, 2, 2],
      );
      MlxRuntime.evalAll([k1, v1]);
      layer1.updateAndFetch(k1, v1);

      // Save.
      store.save('test_session', cache);

      // Load.
      final restored = store.load('test_session');
      expect(restored, isNotNull);
      expect(restored!.layers.length, 2);

      final restoredLayer0 = restored.layers[0] as ConcatKvCache;
      expect(restoredLayer0.offset, 2);
      expect(restoredLayer0.keys!.shape, [1, 1, 2, 2]);

      // Verify data integrity.
      MlxRuntime.evalAll([restoredLayer0.keys!, restoredLayer0.values!]);
      final data = restoredLayer0.keys!.toList();
      expect(data, [1.0, 2.0, 3.0, 4.0]);

      cache.close();
      restored.close();
    });

    test('load returns null for non-existent session', () {
      expect(store.load('nonexistent'), isNull);
    });

    test('delete removes session file', () {
      final cache = DecodeCache.concat(1);
      final layer = cache.layers[0] as ConcatKvCache;
      final k = MlxArray.fromFloat32List([1, 2], shape: [1, 1, 1, 2]);
      final v = MlxArray.fromFloat32List([3, 4], shape: [1, 1, 1, 2]);
      MlxRuntime.evalAll([k, v]);
      layer.updateAndFetch(k, v);

      store.save('to_delete', cache);
      expect(store.load('to_delete'), isNotNull);

      store.delete('to_delete');
      expect(store.load('to_delete'), isNull);

      cache.close();
    });

    test('listSessions lists saved sessions', () {
      final cache = DecodeCache.concat(1);
      final layer = cache.layers[0] as ConcatKvCache;
      final k = MlxArray.fromFloat32List([1, 2], shape: [1, 1, 1, 2]);
      final v = MlxArray.fromFloat32List([3, 4], shape: [1, 1, 1, 2]);
      MlxRuntime.evalAll([k, v]);
      layer.updateAndFetch(k, v);

      store.save('session_a', cache);

      // Need new data for second save (first save consumed the tensors).
      final layer2 = DecodeCache.concat(1).layers[0] as ConcatKvCache;
      final k2 = MlxArray.fromFloat32List([5, 6], shape: [1, 1, 1, 2]);
      final v2 = MlxArray.fromFloat32List([7, 8], shape: [1, 1, 1, 2]);
      MlxRuntime.evalAll([k2, v2]);
      layer2.updateAndFetch(k2, v2);
      final cache2 = DecodeCache([layer2]);
      store.save('session_b', cache2);

      final sessions = store.listSessions();
      expect(sessions, containsAll(['session_a', 'session_b']));

      cache.close();
      cache2.close();
    });

    test('diskUsageBytes returns non-zero after save', () {
      final cache = DecodeCache.concat(1);
      final layer = cache.layers[0] as ConcatKvCache;
      final k = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 1, 2, 2]);
      final v = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [1, 1, 2, 2]);
      MlxRuntime.evalAll([k, v]);
      layer.updateAndFetch(k, v);

      store.save('size_test', cache);
      expect(store.diskUsageBytes(), greaterThan(0));

      cache.close();
    });

    test('save overwrites existing session', () {
      // First save.
      final cache1 = DecodeCache.concat(1);
      final l1 = cache1.layers[0] as ConcatKvCache;
      final k1 = MlxArray.fromFloat32List([1, 2], shape: [1, 1, 1, 2]);
      final v1 = MlxArray.fromFloat32List([3, 4], shape: [1, 1, 1, 2]);
      MlxRuntime.evalAll([k1, v1]);
      l1.updateAndFetch(k1, v1);
      store.save('overwrite', cache1);

      // Second save with different data.
      final cache2 = DecodeCache.concat(1);
      final l2 = cache2.layers[0] as ConcatKvCache;
      final k2 = MlxArray.fromFloat32List(
        [10, 20, 30, 40, 50, 60],
        shape: [1, 1, 3, 2],
      );
      final v2 = MlxArray.fromFloat32List(
        [70, 80, 90, 100, 110, 120],
        shape: [1, 1, 3, 2],
      );
      MlxRuntime.evalAll([k2, v2]);
      l2.updateAndFetch(k2, v2);
      store.save('overwrite', cache2);

      final restored = store.load('overwrite');
      expect(restored, isNotNull);
      final restoredLayer = restored!.layers[0] as ConcatKvCache;
      expect(restoredLayer.offset, 3); // New data has 3 tokens.

      cache1.close();
      cache2.close();
      restored.close();
    });
  });
}
