@TestOn('mac-os')
library;

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main() {
  // -----------------------------------------------------------------------
  // ConcatKvCache
  // -----------------------------------------------------------------------

  group('ConcatKvCache', () {
    test('starts empty', () {
      final cache = ConcatKvCache();
      expect(cache.offset, 0);
      expect(cache.keys, isNull);
      expect(cache.values, isNull);
    });

    test('updateAndFetch stores first KV pair', () {
      final cache = ConcatKvCache();
      final k = MlxArray.fromFloat32List(
        List.generate(8, (i) => i.toDouble()),
        shape: [1, 2, 2, 2],
      );
      final v = MlxArray.fromFloat32List(
        List.generate(8, (i) => (i + 10).toDouble()),
        shape: [1, 2, 2, 2],
      );
      MlxRuntime.evalAll([k, v]);

      final (:keys, :values) = cache.updateAndFetch(k, v);
      expect(cache.offset, 2);
      expect(keys.shape, [1, 2, 2, 2]);
      expect(values.shape, [1, 2, 2, 2]);

      cache.close();
    });

    test('updateAndFetch concatenates subsequent KV pairs', () {
      final cache = ConcatKvCache();

      // First pair: seqLen=2.
      final k1 = MlxArray.fromFloat32List(
        List.generate(8, (i) => i.toDouble()),
        shape: [1, 2, 2, 2],
      );
      final v1 = MlxArray.fromFloat32List(
        List.generate(8, (i) => i.toDouble()),
        shape: [1, 2, 2, 2],
      );
      MlxRuntime.evalAll([k1, v1]);
      cache.updateAndFetch(k1, v1);

      // Second pair: seqLen=1.
      final k2 = MlxArray.fromFloat32List(
        List.generate(4, (i) => i.toDouble()),
        shape: [1, 2, 1, 2],
      );
      final v2 = MlxArray.fromFloat32List(
        List.generate(4, (i) => i.toDouble()),
        shape: [1, 2, 1, 2],
      );
      MlxRuntime.evalAll([k2, v2]);
      final (:keys, :values) = cache.updateAndFetch(k2, v2);

      expect(cache.offset, 3);
      expect(keys.shape[2], 3); // 2 + 1
      expect(values.shape[2], 3);

      cache.close();
    });

    test('clone creates independent copy', () {
      final cache = ConcatKvCache();
      final k = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 1, 2, 2]);
      final v = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [1, 1, 2, 2]);
      MlxRuntime.evalAll([k, v]);
      cache.updateAndFetch(k, v);

      final copy = cache.clone();
      expect(copy.offset, cache.offset);
      expect(copy.keys!.shape, cache.keys!.shape);

      // Closing original should not affect clone.
      cache.close();
      expect(copy.keys, isNotNull);
      expect(copy.offset, 2);

      copy.close();
    });

    test('trim removes tokens from end', () {
      final cache = ConcatKvCache();
      final k = MlxArray.fromFloat32List(
        List.generate(12, (i) => i.toDouble()),
        shape: [1, 1, 6, 2],
      );
      final v = MlxArray.fromFloat32List(
        List.generate(12, (i) => i.toDouble()),
        shape: [1, 1, 6, 2],
      );
      MlxRuntime.evalAll([k, v]);
      cache.updateAndFetch(k, v);
      expect(cache.offset, 6);

      cache.trim(2);
      expect(cache.offset, 4);
      expect(cache.keys!.shape[2], 4);

      cache.close();
    });

    test('evalState does not crash on empty cache', () {
      final cache = ConcatKvCache();
      cache.evalState(); // Should not throw.
      cache.close();
    });
  });

  // -----------------------------------------------------------------------
  // PreallocKvCache
  // -----------------------------------------------------------------------

  group('PreallocKvCache', () {
    test('starts empty', () {
      final cache = PreallocKvCache(numKvHeads: 2, headDim: 4, maxSeqLen: 16);
      expect(cache.offset, 0);
      expect(cache.keys, isNull);
      cache.close();
    });

    test('updateAndFetch writes at offset', () {
      final cache = PreallocKvCache(numKvHeads: 2, headDim: 4, maxSeqLen: 16);

      final k = MlxArray.fromFloat32List(
        List.generate(8, (i) => i.toDouble()),
        shape: [1, 2, 1, 4],
      );
      final v = MlxArray.fromFloat32List(
        List.generate(8, (i) => (i + 10).toDouble()),
        shape: [1, 2, 1, 4],
      );
      MlxRuntime.evalAll([k, v]);

      final (:keys, :values) = cache.updateAndFetch(k, v);
      expect(cache.offset, 1);
      expect(keys.shape, [1, 2, 1, 4]);
      expect(values.shape, [1, 2, 1, 4]);

      keys.close();
      values.close();
      cache.close();
    });

    test('subsequent updates advance offset', () {
      final cache = PreallocKvCache(numKvHeads: 1, headDim: 2, maxSeqLen: 32);

      // Write 3 tokens.
      final k1 = MlxArray.fromFloat32List(
        List.generate(6, (i) => i.toDouble()),
        shape: [1, 1, 3, 2],
      );
      final v1 = MlxArray.fromFloat32List(
        List.generate(6, (i) => i.toDouble()),
        shape: [1, 1, 3, 2],
      );
      MlxRuntime.evalAll([k1, v1]);
      final r1 = cache.updateAndFetch(k1, v1);
      r1.keys.close();
      r1.values.close();
      expect(cache.offset, 3);

      // Write 2 more tokens.
      final k2 = MlxArray.fromFloat32List(
        List.generate(4, (i) => i.toDouble()),
        shape: [1, 1, 2, 2],
      );
      final v2 = MlxArray.fromFloat32List(
        List.generate(4, (i) => i.toDouble()),
        shape: [1, 1, 2, 2],
      );
      MlxRuntime.evalAll([k2, v2]);
      final r2 = cache.updateAndFetch(k2, v2);
      r2.keys.close();
      r2.values.close();
      expect(cache.offset, 5);

      cache.close();
    });
  });

  // -----------------------------------------------------------------------
  // LinearStateCache
  // -----------------------------------------------------------------------

  group('LinearStateCache', () {
    test('starts empty', () {
      final cache = LinearStateCache();
      expect(cache.offset, 0);
      expect(cache.takeConvState(), isNull);
      expect(cache.takeState(), isNull);
      cache.close();
    });

    test('replace and take conv state', () {
      final cache = LinearStateCache();
      final state = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
      MlxRuntime.evalAll([state]);

      cache.replaceConvState(state);
      final taken = cache.takeConvState();
      expect(taken, isNotNull);
      expect(taken!.shape, [2, 2]);
      expect(cache.takeConvState(), isNull); // Nulled after take.

      taken.close();
      cache.close();
    });

    test('replaceState increments offset', () {
      final cache = LinearStateCache();
      final s1 = MlxArray.fromFloat32List([1, 2], shape: [1, 2]);
      MlxRuntime.evalAll([s1]);
      cache.replaceState(s1);
      expect(cache.offset, 1);

      final s2 = MlxArray.fromFloat32List([3, 4], shape: [1, 2]);
      MlxRuntime.evalAll([s2]);
      cache.replaceState(s2);
      expect(cache.offset, 2);

      cache.close();
    });

    test('clone creates independent copy', () {
      final cache = LinearStateCache();
      final conv = MlxArray.fromFloat32List([1, 2], shape: [1, 2]);
      final state = MlxArray.fromFloat32List([3, 4], shape: [1, 2]);
      MlxRuntime.evalAll([conv, state]);
      cache.replaceConvState(conv);
      cache.replaceState(state);

      final copy = cache.clone();
      expect(copy.offset, cache.offset);

      cache.close();
      // Clone should still be valid.
      final copiedState = copy.takeState();
      expect(copiedState, isNotNull);
      copiedState!.close();
      copy.close();
    });
  });

  // -----------------------------------------------------------------------
  // DecodeCache
  // -----------------------------------------------------------------------

  group('DecodeCache', () {
    test('concat factory creates N ConcatKvCache layers', () {
      final cache = DecodeCache.concat(4);
      expect(cache.layers.length, 4);
      expect(cache.layers.every((l) => l is ConcatKvCache), isTrue);
      expect(cache.offset, 0);
      cache.close();
    });

    test('prealloc factory creates N PreallocKvCache layers', () {
      final cache = DecodeCache.prealloc(
        numLayers: 2,
        numKvHeads: 4,
        headDim: 8,
        maxSeqLen: 32,
      );
      expect(cache.layers.length, 2);
      expect(cache.layers.every((l) => l is PreallocKvCache), isTrue);
      cache.close();
    });

    test('clone works for ConcatKvCache layers', () {
      final cache = DecodeCache.concat(2);
      // Add some data to the first layer.
      final layer0 = cache.layers[0] as ConcatKvCache;
      final k = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [1, 1, 2, 2]);
      final v = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [1, 1, 2, 2]);
      MlxRuntime.evalAll([k, v]);
      layer0.updateAndFetch(k, v);

      final copy = cache.clone();
      expect(copy.layers.length, 2);
      expect((copy.layers[0] as ConcatKvCache).offset, 2);

      cache.close();
      copy.close();
    });

    test('clone throws for PreallocKvCache layers', () {
      final cache = DecodeCache.prealloc(
        numLayers: 1,
        numKvHeads: 1,
        headDim: 2,
        maxSeqLen: 8,
      );
      expect(() => cache.clone(), throwsUnsupportedError);
      cache.close();
    });

    test('evalStates does not crash on empty cache', () {
      final cache = DecodeCache.concat(3);
      cache.evalStates(); // Should not throw.
      cache.close();
    });
  });

  // -----------------------------------------------------------------------
  // Session types (data classes only — no real model needed)
  // -----------------------------------------------------------------------

  group('GenerationTiming', () {
    test('tokensPerSecond calculates correctly', () {
      const timing = GenerationTiming(
        promptMs: 100,
        firstTokenMs: 10,
        decodeMs: 2000,
        totalMs: 2100,
      );
      // 10 tokens, 2s decode -> (10-1)/2.0 = 4.5 tok/s
      expect(timing.tokensPerSecond(10), closeTo(4.5, 0.1));
    });

    test('tokensPerSecond returns 0 for single token', () {
      const timing = GenerationTiming(
        promptMs: 100,
        firstTokenMs: 10,
        decodeMs: 1000,
        totalMs: 1100,
      );
      expect(timing.tokensPerSecond(1), 0);
    });

    test('toString includes all fields', () {
      const timing = GenerationTiming(
        promptMs: 50.5,
        firstTokenMs: 10.3,
        decodeMs: 200.7,
        totalMs: 251.2,
      );
      final s = timing.toString();
      expect(s, contains('prompt=50.5ms'));
      expect(s, contains('first=10.3ms'));
    });
  });

  group('StopReason', () {
    test('has all expected values', () {
      expect(StopReason.values.length, 3);
      expect(
        StopReason.values.map((e) => e.name),
        containsAll(['maxTokens', 'stopToken', 'repetition']),
      );
    });
  });
}
