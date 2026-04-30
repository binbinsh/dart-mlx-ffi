/// Public [PromptSession] interface for multi-turn cached generation.
///
/// A session caches provider state after the initial prompt so follow-up
/// generations can start from the cached prefix and avoid redundant prefill.
library;

// ---------------------------------------------------------------------------
// Generation result
// ---------------------------------------------------------------------------

/// Reason a generation stopped.
enum StopReason {
  /// Reached `maxNewTokens`.
  maxTokens,

  /// Hit a stop-token (e.g. EOS).
  stopToken,

  /// Detected repetition and bailed out.
  repetition,
}

/// Timing breakdown for a single generation.
final class GenerationTiming {
  const GenerationTiming({
    required this.promptMs,
    required this.firstTokenMs,
    required this.decodeMs,
    required this.totalMs,
  });

  /// Time spent computing the prompt (prefill).
  final double promptMs;

  /// Latency to the first generated token.
  final double firstTokenMs;

  /// Total time in the decode loop (including first token).
  final double decodeMs;

  /// Wall-clock total.
  final double totalMs;

  /// Tokens per second (decode only, excluding the first token).
  double tokensPerSecond(int generatedTokens) {
    if (generatedTokens <= 1 || decodeMs <= 0) return 0;
    return (generatedTokens - 1) / (decodeMs / 1000.0);
  }

  @override
  String toString() =>
      'GenerationTiming(prompt=${promptMs.toStringAsFixed(1)}ms, '
      'first=${firstTokenMs.toStringAsFixed(1)}ms, '
      'decode=${decodeMs.toStringAsFixed(1)}ms, '
      'total=${totalMs.toStringAsFixed(1)}ms)';
}

/// Result of a generation call.
final class GenerationResult {
  const GenerationResult({
    required this.tokenIds,
    required this.generatedTokenIds,
    required this.stopReason,
    required this.timing,
  });

  /// Full token sequence (prompt + generated).
  final List<int> tokenIds;

  /// Only the generated part.
  final List<int> generatedTokenIds;

  /// Why generation stopped.
  final StopReason stopReason;

  /// Timing breakdown.
  final GenerationTiming timing;
}

// ---------------------------------------------------------------------------
// PromptSession
// ---------------------------------------------------------------------------

/// A prompt session caches KV state after an initial prefill so that
/// multiple generations can share the same prompt prefix.
///
/// This is the public interface — concrete implementations are provided by
/// each model runner.
///
/// Usage:
/// ```dart
/// final session = runner.createPromptSession(promptTokenIds);
/// final result1 = session.generateGreedy(maxNewTokens: 128);
/// final result2 = session.generateGreedy(maxNewTokens: 256);
/// session.close();
/// ```
abstract interface class PromptSession {
  /// The token IDs that were used to prime this session.
  List<int> get promptTokenIds;

  /// Run greedy autoregressive decoding from the cached prompt state.
  ///
  /// Each call clones the internal cache so the session remains reusable
  /// for additional generations.
  GenerationResult generateGreedy({
    required int maxNewTokens,
    Set<int> stopTokenIds,
  });

  /// Release all GPU resources held by this session.
  void close();
}

// ---------------------------------------------------------------------------
// StreamSession
// ---------------------------------------------------------------------------

/// A streaming session for chunk-based incremental generation (e.g. ASR).
///
/// Unlike [PromptSession], a stream session mutates its internal state
/// with each chunk — it is not reusable after close.
abstract interface class StreamSession {
  /// Number of tokens processed so far (across all chunks).
  int get position;

  /// Process the next chunk and return generated token IDs.
  ///
  /// The semantics of [chunk] depend on the model:
  /// - For ASR models: raw audio samples or mel features.
  /// - For text models: additional prompt token IDs.
  List<int> decodeChunk(Object chunk, {int maxNewTokens});

  /// Reset the session to a clean state (discards all cached KV).
  void reset();

  /// Release all GPU resources.
  void close();
}

// ---------------------------------------------------------------------------
// Helper: greedy decode loop
// ---------------------------------------------------------------------------

/// Reusable greedy decode loop used by [PromptSession] implementations.
///
/// [stepFn] receives the current token IDs list and returns the next logit
/// tensor from which the argmax is taken.
///
/// This function manages timing, stop-token detection, and result assembly.
typedef StepFunction<T extends Object> =
    Object Function(List<int> tokens, T cache);

/// Run a greedy decode loop.
///
/// [promptLogits] is the logits tensor from the prefill (last-token slice).
/// [cache] is the provider cache to mutate during generation.
/// [argmaxFn] extracts the next token ID from a logits tensor.
/// [stepFn] runs one decode step and returns the next logits tensor.
/// [closeFn] disposes a logits tensor.
/// [closeCacheFn] releases the provider cache.
///
/// Returns a [GenerationResult] with full timing.
GenerationResult greedyDecodeLoop<T extends Object>({
  required List<int> promptTokenIds,
  required Object promptLogits,
  required T cache,
  required int maxNewTokens,
  required Set<int> stopTokenIds,
  required int Function(Object logits) argmaxFn,
  required Object Function(int tokenId, T cache) stepFn,
  required void Function(Object logits) closeFn,
  required void Function(T cache) closeCacheFn,
  required double promptMs,
}) {
  if (maxNewTokens <= 0) {
    closeFn(promptLogits);
    closeCacheFn(cache);
    return GenerationResult(
      tokenIds: List<int>.unmodifiable(promptTokenIds),
      generatedTokenIds: const <int>[],
      stopReason: StopReason.maxTokens,
      timing: GenerationTiming(
        promptMs: promptMs,
        firstTokenMs: 0,
        decodeMs: 0,
        totalMs: promptMs,
      ),
    );
  }

  final tokens = List<int>.from(promptTokenIds);
  final decodeWatch = Stopwatch()..start();
  var logits = promptLogits;
  double firstTokenMs = 0;
  var stopReason = StopReason.maxTokens;

  try {
    for (var i = 0; i < maxNewTokens; i++) {
      final sampleWatch = Stopwatch()..start();
      final next = argmaxFn(logits);
      sampleWatch.stop();
      if (i == 0) {
        firstTokenMs = sampleWatch.elapsedMicroseconds / 1000.0;
      }
      tokens.add(next);
      if (stopTokenIds.contains(next)) {
        stopReason = StopReason.stopToken;
        break;
      }
      if (i + 1 >= maxNewTokens) break;
      closeFn(logits);
      logits = stepFn(next, cache);
    }
  } finally {
    decodeWatch.stop();
    closeFn(logits);
    closeCacheFn(cache);
  }

  final totalMs = promptMs + decodeWatch.elapsedMicroseconds / 1000.0;
  return GenerationResult(
    tokenIds: List<int>.unmodifiable(tokens),
    generatedTokenIds: List<int>.unmodifiable(
      tokens.sublist(promptTokenIds.length),
    ),
    stopReason: stopReason,
    timing: GenerationTiming(
      promptMs: promptMs,
      firstTokenMs: firstTokenMs,
      decodeMs: decodeWatch.elapsedMicroseconds / 1000.0,
      totalMs: totalMs,
    ),
  );
}
