@TestOn('mac-os')
library;

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/models.dart';

void main() {
  group('MetalGate', () {
    test('runs a single task', () async {
      final gate = MetalGate();
      final result = await gate.run(() => 42);
      expect(result, 42);
    });

    test('serialises tasks by default (maxConcurrent=1)', () async {
      final gate = MetalGate();
      final order = <int>[];

      final f1 = gate.run(() async {
        await Future<void>.delayed(const Duration(milliseconds: 20));
        order.add(1);
        return 'a';
      });
      final f2 = gate.run(() async {
        order.add(2);
        return 'b';
      });

      await Future.wait([f1, f2]);
      // Task 1 should complete before task 2 starts.
      expect(order, [1, 2]);
    });

    test('allows concurrent tasks when maxConcurrent > 1', () async {
      final gate = MetalGate(maxConcurrent: 2);
      final order = <int>[];

      final f1 = gate.run(() async {
        await Future<void>.delayed(const Duration(milliseconds: 50));
        order.add(1);
      });
      final f2 = gate.run(() async {
        await Future<void>.delayed(const Duration(milliseconds: 10));
        order.add(2);
      });

      await Future.wait([f1, f2]);
      // Both start concurrently; task 2 finishes first.
      expect(order, [2, 1]);
    });

    test('propagates errors', () async {
      final gate = MetalGate();
      expect(gate.run(() => throw StateError('boom')), throwsStateError);
    });

    test('cancelPending cancels queued tasks', () async {
      final gate = MetalGate();
      // Fill the gate with a slow task.
      final f1 = gate.run(() async {
        await Future<void>.delayed(const Duration(milliseconds: 100));
        return 1;
      });
      // Queue a second task and immediately set up error expectation.
      final f2 = gate.run(() => 2);
      // Set up the error catcher before cancelling.
      final f2Expectation = expectLater(f2, throwsStateError);
      // Cancel pending — f2 should get an error.
      gate.cancelPending();

      expect(await f1, 1);
      await f2Expectation;
    });

    test('drain waits for all tasks', () async {
      final gate = MetalGate();
      var done = false;
      gate.run(() async {
        await Future<void>.delayed(const Duration(milliseconds: 30));
        done = true;
      });
      await gate.drain();
      expect(done, isTrue);
    });

    test('activeCount and pendingCount are tracked', () async {
      final gate = MetalGate();
      expect(gate.activeCount, 0);
      expect(gate.pendingCount, 0);

      gate.run(() async {
        await Future<void>.delayed(const Duration(milliseconds: 100));
      });
      // Allow the task to start.
      await Future<void>.delayed(Duration.zero);
      expect(gate.activeCount, 1);

      await gate.drain();
      expect(gate.activeCount, 0);
    });
  });

  group('MetalGateScope', () {
    test('runs tasks and closes', () async {
      final gate = MetalGate();
      final scope = MetalGateScope(gate);

      final result = await scope.run(() => 'hello');
      expect(result, 'hello');

      await scope.close();
    });

    test('throws after close', () async {
      final gate = MetalGate();
      final scope = MetalGateScope(gate);
      await scope.close();

      expect(() => scope.run(() => 1), throwsStateError);
    });
  });

  group('metalGate singleton', () {
    test('is accessible and functional', () async {
      final result = await metalGate.run(() => 'global');
      expect(result, 'global');
    });
  });
}
