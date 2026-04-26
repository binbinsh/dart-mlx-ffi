@TestOn('mac-os')
library;

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

void main() {
  group('StreamAccumulator', () {
    test('accumulates tokens with detokeniser', () {
      final acc = StreamAccumulator(
        detokenise: (ids) => ids.map((i) => String.fromCharCode(i + 65)).join(),
      );

      final d1 = acc.add(0); // 'A'
      expect(d1, isNotNull);
      expect(d1!.text, 'A');
      expect(d1.totalTokens, 1);

      final d2 = acc.add(1); // 'AB' -> delta 'B'
      expect(d2, isNotNull);
      expect(d2!.text, 'B');
      expect(d2.totalTokens, 2);
    });

    test('tokenIds tracks all added tokens', () {
      final acc = StreamAccumulator();
      acc.add(10);
      acc.add(20);
      acc.add(30);
      expect(acc.tokenIds, [10, 20, 30]);
      expect(acc.length, 3);
    });

    test('stops on stop token', () {
      final acc = StreamAccumulator(stopTokenIds: {99});
      acc.add(1);
      expect(acc.shouldStop, isFalse);
      acc.add(99);
      expect(acc.shouldStop, isTrue);
      expect(acc.stopTokenId, 99);
    });

    test('stops on maxTokens', () {
      final acc = StreamAccumulator(maxTokens: 3);
      acc.add(1);
      acc.add(2);
      expect(acc.shouldStop, isFalse);
      acc.add(3);
      expect(acc.shouldStop, isTrue);
    });

    test('returns null after stop', () {
      final acc = StreamAccumulator(stopTokenIds: {99});
      acc.add(99);
      expect(acc.add(100), isNull);
    });

    test('calls onToken callback', () {
      final received = <int>[];
      final acc = StreamAccumulator(onToken: (id, delta) => received.add(id));
      acc.add(5);
      acc.add(10);
      expect(received, [5, 10]);
    });

    test('finish produces StreamResult', () {
      final acc = StreamAccumulator(
        detokenise: (ids) => ids.join(','),
        stopTokenIds: {99},
      );
      acc.add(1);
      acc.add(2);
      acc.add(99);
      final result = acc.finish();
      expect(result.tokenIds, [1, 2, 99]);
      expect(result.text, '1,2,99');
      expect(result.stoppedByToken, isTrue);
      expect(result.stopTokenId, 99);
    });

    test('reset clears state', () {
      final acc = StreamAccumulator(stopTokenIds: {99});
      acc.add(1);
      acc.add(99);
      expect(acc.shouldStop, isTrue);

      acc.reset();
      expect(acc.length, 0);
      expect(acc.shouldStop, isFalse);
      expect(acc.tokenIds, isEmpty);
    });

    test('returns null delta when detokeniser produces no new text', () {
      // A detokeniser that always returns 'x' regardless of input.
      final acc = StreamAccumulator(detokenise: (_) => 'x');
      final d1 = acc.add(1);
      expect(d1, isNotNull);
      expect(d1!.text, 'x');

      // Second add: full text is still 'x', so delta is empty.
      final d2 = acc.add(2);
      expect(d2, isNull);
    });

    test('text property returns current decoded text', () {
      final acc = StreamAccumulator(
        detokenise: (ids) => ids.map((i) => '$i').join(' '),
      );
      acc.add(10);
      acc.add(20);
      expect(acc.text, '10 20');
    });
  });

  group('StreamAccumulatorStreamExt', () {
    test('asStream yields deltas', () async {
      final acc = StreamAccumulator(
        detokenise: (ids) => ids.map((i) => String.fromCharCode(i + 65)).join(),
        stopTokenIds: {3},
      );

      final deltas = await acc.asStream([0, 1, 2, 3, 4]).toList();
      // Should stop at token 3 (but 3 itself may or may not yield a delta).
      expect(deltas.length, greaterThanOrEqualTo(3));
      expect(deltas.first.text, 'A');
    });
  });

  group('StreamDelta', () {
    test('stores fields correctly', () {
      const delta = StreamDelta(tokenId: 5, text: 'hi', totalTokens: 10);
      expect(delta.tokenId, 5);
      expect(delta.text, 'hi');
      expect(delta.totalTokens, 10);
    });
  });

  group('StreamResult', () {
    test('tokensPerSecond calculates correctly', () {
      const result = StreamResult(
        tokenIds: [1, 2, 3, 4, 5],
        text: 'hello',
        elapsed: Duration(seconds: 2),
        stoppedByToken: false,
        stopTokenId: -1,
      );
      // (5-1) / 2 = 2.0 tok/s
      expect(result.tokensPerSecond, closeTo(2.0, 0.1));
    });

    test('tokensPerSecond returns 0 for single token', () {
      const result = StreamResult(
        tokenIds: [1],
        text: 'x',
        elapsed: Duration(seconds: 1),
        stoppedByToken: false,
        stopTokenId: -1,
      );
      expect(result.tokensPerSecond, 0);
    });
  });
}
