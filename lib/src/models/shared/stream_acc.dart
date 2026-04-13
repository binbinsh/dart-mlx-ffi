/// Stream Accumulator — generic streaming token output assembly.
///
/// Inspired by osaurus `StreamAccumulator.swift`.  Collects token IDs
/// emitted one-at-a-time during autoregressive generation and provides:
///
/// - Running text assembly (via a pluggable detokeniser).
/// - Token-level callbacks (for progress UIs).
/// - Windowed throughput measurement.
/// - Stop-condition detection (EOS, repetition, max tokens).
library;

import 'dart:async';

// ---------------------------------------------------------------------------
// Detokeniser callback
// ---------------------------------------------------------------------------

/// Converts a list of token IDs to a string.
///
/// Implementations should handle BPE merge, sentencepiece, etc.
typedef Detokenise = String Function(List<int> tokenIds);

/// A no-op detokeniser that returns an empty string.
String _nullDetokenise(List<int> _) => '';

// ---------------------------------------------------------------------------
// StreamAccumulator
// ---------------------------------------------------------------------------

/// Accumulates tokens from a streaming generation and exposes them as a
/// [Stream] of incremental results.
///
/// ```dart
/// final acc = StreamAccumulator(detokenise: tokenizer.decode);
/// for (final tokenId in generation) {
///   final delta = acc.add(tokenId);
///   if (delta != null) print(delta.text);
///   if (acc.shouldStop) break;
/// }
/// final result = acc.finish();
/// ```
final class StreamAccumulator {
  StreamAccumulator({
    Detokenise? detokenise,
    this.stopTokenIds = const <int>{},
    this.maxTokens = 0,
    this.onToken,
  }) : _detokenise = detokenise ?? _nullDetokenise;

  final Detokenise _detokenise;

  /// Token IDs that signal generation should stop.
  final Set<int> stopTokenIds;

  /// Maximum tokens to accumulate (0 = unlimited).
  final int maxTokens;

  /// Optional per-token callback.
  final void Function(int tokenId, StreamDelta delta)? onToken;

  final List<int> _tokens = [];
  String _prevText = '';
  final Stopwatch _watch = Stopwatch();
  bool _stopped = false;
  int _stopTokenId = -1;

  /// All accumulated token IDs so far.
  List<int> get tokenIds => List<int>.unmodifiable(_tokens);

  /// Number of tokens accumulated.
  int get length => _tokens.length;

  /// Whether a stop condition has been reached.
  bool get shouldStop => _stopped;

  /// The stop token that triggered the stop, or -1 if not stopped by a token.
  int get stopTokenId => _stopTokenId;

  /// Current full decoded text.
  String get text => _detokenise(_tokens);

  /// Elapsed time since the first token was added.
  Duration get elapsed => _watch.elapsed;

  /// Tokens per second (excludes the first token for fairness).
  double get tokensPerSecond {
    if (_tokens.length <= 1 || !_watch.isRunning) return 0;
    final seconds = _watch.elapsedMicroseconds / 1e6;
    return (_tokens.length - 1) / seconds;
  }

  /// Add a token and return the incremental text delta, or `null` if
  /// nothing new was produced.
  ///
  /// After calling this, check [shouldStop] to see if generation should end.
  StreamDelta? add(int tokenId) {
    if (_stopped) return null;

    if (_tokens.isEmpty) _watch.start();
    _tokens.add(tokenId);

    // Check stop conditions.
    if (stopTokenIds.contains(tokenId)) {
      _stopped = true;
      _stopTokenId = tokenId;
    }
    if (maxTokens > 0 && _tokens.length >= maxTokens) {
      _stopped = true;
    }

    // Compute incremental text.
    final fullText = _detokenise(_tokens);
    final newText = fullText.substring(_prevText.length);
    _prevText = fullText;

    final delta = StreamDelta(
      tokenId: tokenId,
      text: newText,
      totalTokens: _tokens.length,
    );
    onToken?.call(tokenId, delta);
    return newText.isEmpty ? null : delta;
  }

  /// Finalise and return the complete result.
  StreamResult finish() {
    _watch.stop();
    return StreamResult(
      tokenIds: List<int>.unmodifiable(_tokens),
      text: _detokenise(_tokens),
      elapsed: _watch.elapsed,
      stoppedByToken: _stopTokenId >= 0,
      stopTokenId: _stopTokenId,
    );
  }

  /// Reset to initial state for reuse.
  void reset() {
    _tokens.clear();
    _prevText = '';
    _watch.reset();
    _stopped = false;
    _stopTokenId = -1;
  }
}

// ---------------------------------------------------------------------------
// Data classes
// ---------------------------------------------------------------------------

/// An incremental update from the accumulator.
final class StreamDelta {
  const StreamDelta({
    required this.tokenId,
    required this.text,
    required this.totalTokens,
  });

  /// The token ID that was just added.
  final int tokenId;

  /// New text produced by this token (may be empty for partial BPE merges).
  final String text;

  /// Total tokens accumulated so far.
  final int totalTokens;
}

/// Final result from the accumulator.
final class StreamResult {
  const StreamResult({
    required this.tokenIds,
    required this.text,
    required this.elapsed,
    required this.stoppedByToken,
    required this.stopTokenId,
  });

  /// All generated token IDs.
  final List<int> tokenIds;

  /// Full decoded text.
  final String text;

  /// Total generation time.
  final Duration elapsed;

  /// Whether generation stopped because of a stop token.
  final bool stoppedByToken;

  /// The stop token ID, or -1.
  final int stopTokenId;

  /// Tokens per second (excluding first token).
  double get tokensPerSecond {
    if (tokenIds.length <= 1) return 0;
    final seconds = elapsed.inMicroseconds / 1e6;
    return (tokenIds.length - 1) / seconds;
  }
}

// ---------------------------------------------------------------------------
// StreamAccumulator as a Dart Stream
// ---------------------------------------------------------------------------

/// Extension that wraps a synchronous token generator into a Dart [Stream]
/// of [StreamDelta] values.
extension StreamAccumulatorStreamExt on StreamAccumulator {
  /// Convert an iterable of token IDs into a stream of deltas.
  Stream<StreamDelta> asStream(Iterable<int> tokenIds) async* {
    for (final tokenId in tokenIds) {
      final delta = add(tokenId);
      if (delta != null) yield delta;
      if (shouldStop) break;
    }
  }
}
