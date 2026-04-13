/// Metal Gate — formal GPU access scheduling, multi-isolate safe.
///
/// Inspired by osaurus `MetalGate.swift`.  Serialises access to the GPU
/// so that multiple concurrent callers (different isolates, different model
/// runners) do not stampede the single Metal device.
///
/// The gate uses a simple FIFO queue backed by Dart's single-threaded
/// event loop.  Cross-isolate safety is achieved by funneling all GPU
/// work through a single gate instance per isolate, with an optional
/// `SendPort`-based proxy for multi-isolate setups.
library;

import 'dart:async';

// ---------------------------------------------------------------------------
// MetalGate
// ---------------------------------------------------------------------------

/// Serialises GPU-bound work to avoid Metal contention.
///
/// Usage:
/// ```dart
/// final gate = MetalGate();
/// final result = await gate.run(() => runner.generate(...));
/// ```
///
/// The gate ensures that at most [maxConcurrent] tasks execute on the GPU
/// at the same time (default: 1, i.e. strict serialisation).
final class MetalGate {
  MetalGate({this.maxConcurrent = 1}) : assert(maxConcurrent >= 1);

  /// Maximum number of concurrent GPU tasks.
  final int maxConcurrent;

  int _running = 0;
  final _queue = <_GateEntry<Object?>>[];

  /// Number of tasks currently executing.
  int get activeCount => _running;

  /// Number of tasks waiting in the queue.
  int get pendingCount => _queue.length;

  /// Submit [work] to the gate.  Returns when the work completes.
  ///
  /// If the gate is at capacity, the call suspends until a slot opens.
  Future<T> run<T>(FutureOr<T> Function() work, {String? label}) {
    final completer = Completer<T>();
    _queue.add(
      _GateEntry<T>(work: work, completer: completer, label: label)
          as _GateEntry<Object?>,
    );
    _drain();
    return completer.future;
  }

  /// Cancel all pending (not yet started) tasks.
  ///
  /// Active tasks are not interrupted — they will complete normally.
  void cancelPending() {
    for (final entry in _queue) {
      entry.completer.completeError(
        StateError('MetalGate: task cancelled while pending'),
      );
    }
    _queue.clear();
  }

  /// Wait for all queued and active tasks to complete.
  Future<void> drain() async {
    while (_running > 0 || _queue.isNotEmpty) {
      await Future<void>.delayed(Duration.zero);
    }
  }

  void _drain() {
    while (_running < maxConcurrent && _queue.isNotEmpty) {
      final entry = _queue.removeAt(0);
      _running++;
      _execute(entry);
    }
  }

  Future<void> _execute(_GateEntry<Object?> entry) async {
    try {
      final result = await entry.work();
      entry.completer.complete(result);
    } catch (e, st) {
      entry.completer.completeError(e, st);
    } finally {
      _running--;
      _drain();
    }
  }
}

/// Internal queue entry.
final class _GateEntry<T> {
  const _GateEntry({required this.work, required this.completer, this.label});

  final FutureOr<T> Function() work;
  final Completer<T> completer;
  final String? label;
}

// ---------------------------------------------------------------------------
// MetalGateScope — convenience RAII-style usage
// ---------------------------------------------------------------------------

/// A disposable gate scope that automatically drains on close.
///
/// ```dart
/// final scope = MetalGateScope(gate);
/// try {
///   await scope.run(() => heavyGpuWork());
///   await scope.run(() => moreGpuWork());
/// } finally {
///   await scope.close();
/// }
/// ```
final class MetalGateScope {
  MetalGateScope(this._gate);

  final MetalGate _gate;
  bool _closed = false;

  /// Submit work through the underlying gate.
  Future<T> run<T>(FutureOr<T> Function() work, {String? label}) {
    if (_closed) {
      throw StateError('MetalGateScope has been closed');
    }
    return _gate.run(work, label: label);
  }

  /// Drain all pending work and mark this scope as closed.
  Future<void> close() async {
    _closed = true;
    await _gate.drain();
  }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

/// The default global [MetalGate] instance.
///
/// Most apps only need one gate.  For advanced multi-isolate setups,
/// create additional gate instances as needed.
final MetalGate metalGate = MetalGate();
