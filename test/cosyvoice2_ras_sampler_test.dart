// Unit tests for the RAS sampler. These exercise the Dart policy wrapper
// around the native softmax/top-p kernels without requiring model assets.

import 'dart:math';
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';
import 'package:dart_inference/src/models/cosyvoice2/cosyvoice2_ras_sampler.dart';
import 'package:test/test.dart';

void main() {
  group('RasSampler', () {
    test('with one dominant logit, picks that index every time', () {
      final logits = Float32List.fromList([
        for (var i = 0; i < 100; i += 1) (i == 42) ? 50.0 : 0.0,
      ]);
      final s = RasSampler(rng: Random(0));
      for (var i = 0; i < 32; i += 1) {
        expect(s.sample(logits, const []), 42);
      }
    });

    test('top-k cap limits candidate pool', () {
      // Uniform logits => every index has equal softmax mass. With
      // top_p=0.8 and top_k=25, we always restrict to the first 25
      // (after stable-descending sort, equal probs preserve original
      // order — i.e. ids 0..24).
      final n = 256;
      final logits = Float32List(n); // all zeros
      final s = RasSampler(
        rng: Random(7),
        config: const RasConfig(topP: 0.8, topK: 25),
      );
      final seen = <int>{};
      for (var i = 0; i < 1000; i += 1) {
        seen.add(s.sample(logits, const []));
      }
      for (final id in seen) {
        expect(
          id,
          lessThan(25),
          reason: 'sampled id $id outside top-k=25 window',
        );
      }
    });

    test('repetition triggers full-distribution fallback', () {
      // Logits favour id 5 strongly under nucleus; history is full of
      // 5s, so the RAS rule should fall back to multinomial over the
      // entire distribution. With a seeded RNG we just verify that at
      // least one non-5 id is produced over many calls (greedy nucleus
      // would keep returning 5).
      final n = 64;
      final logits = Float32List(n);
      logits[5] = 10.0;
      // Other logits at 0 still have non-trivial mass after softmax.
      final history = List<int>.filled(20, 5);
      final s = RasSampler(
        rng: Random(123),
        config: const RasConfig(topP: 0.8, topK: 25, winSize: 10, tauR: 0.1),
      );
      var nonFive = 0;
      for (var i = 0; i < 200; i += 1) {
        final id = s.sample(logits, history);
        if (id != 5) nonFive += 1;
      }
      expect(
        nonFive,
        greaterThan(0),
        reason: 'fallback path never produced a non-repeating token',
      );
    });

    test('no repetition => keeps nucleus winner', () {
      final n = 64;
      final logits = Float32List(n);
      logits[7] = 20.0; // dominates softmax, top-1 always wins
      final s = RasSampler(rng: Random(1));
      for (var i = 0; i < 50; i += 1) {
        expect(s.sample(logits, const [0, 1, 2, 3]), 7);
      }
    });

    test('masks EOS and special-token logits in native', () {
      final logits = Float32List.fromList([0, 0, 100, 200]);
      final s = RasSampler(rng: Random(1));

      expect(s.sample(logits, const [], eosToken: 2, ignoreEos: true), 1);
      expect(s.sample(logits, const [], eosToken: 2), 2);
    });

    test('samples native-backed logits without copying to Dart first', () {
      final logits = NativeTensorBuffer.float32([4]);
      try {
        logits.asFloat32List().setAll(0, [0, 0, 100, 200]);
        final s = RasSampler(rng: Random(1));

        expect(
          s.sample(logits.tensor, const [], eosToken: 2, ignoreEos: true),
          1,
        );
        expect(s.sample(logits.tensor, const [], eosToken: 2), 2);
      } finally {
        logits.close();
      }
    });

    test('decode buffer keeps history in native memory', () {
      final n = 64;
      final logits = Float32List(n);
      logits[5] = 10.0;
      final history = List<int>.filled(20, 5);
      const config = RasConfig(topP: 0.8, topK: 25, winSize: 10, tauR: 0.1);

      final expected = RasSampler(
        rng: Random(123),
        config: config,
      ).sample(logits, history);
      final buffer = RasDecodeBuffer(
        maxTokens: 32,
        rng: Random(123),
        config: config,
      );
      try {
        buffer.appendAll(history);
        final actual = buffer.sample(logits);
        expect(actual, expected);
        buffer.append(actual);
        expect(buffer.length, history.length + 1);
        expect(buffer.toList().sublist(0, history.length), history);
        expect(buffer.tokensTensor().asInt32List().length, history.length + 1);
        expect(
          buffer.tokensTensor().asInt32List().sublist(0, history.length),
          history,
        );
      } finally {
        buffer.close();
      }
    });

    test('decode buffer samples and appends accepted tokens natively', () {
      final buffer = RasDecodeBuffer(maxTokens: 2, rng: Random(0));
      try {
        final logits = Float32List(8);
        logits[3] = 20.0;
        final token = buffer.sampleAndAppendNonEos(logits, eosToken: 7);

        expect(token, 3);
        expect(buffer.length, 1);
        expect(buffer.toList(), [3]);

        final eosLogits = Float32List.fromList([0, 0, 50, 0]);
        final eos = buffer.sampleAndAppendNonEos(eosLogits, eosToken: 2);

        expect(eos, 2);
        expect(buffer.length, 1);
        expect(buffer.toList(), [3]);
      } finally {
        buffer.close();
      }
    });

    test('decode buffer rejects capacity overflow', () {
      final buffer = RasDecodeBuffer(maxTokens: 1, rng: Random(0));
      try {
        buffer.append(1);
        expect(() => buffer.append(2), throwsStateError);
      } finally {
        buffer.close();
      }
    });
  });
}
