part of 'qwen3_5.dart';

extension Qwen3_5RunnerBench on Qwen3_5Runner {
  Qwen35TimedGeneration timedGenerateGreedy(
    List<int> promptIds,
    int maxNewTokens, {
    List<int> stopTokenIds = const <int>[],
  }) {
    if (maxNewTokens < 0) {
      throw ArgumentError.value(
        maxNewTokens,
        'maxNewTokens',
        'Must be non-negative.',
      );
    }
    if (maxNewTokens == 0) {
      return (
        tokenIds: List<int>.from(promptIds),
        generatedTokenIds: const <int>[],
        promptMs: 0,
        firstTokenMs: 0,
        decodeMs: 0,
        totalMs: 0,
        stoppedByStopToken: false,
      );
    }
    final stopSet = stopTokenIds.toSet();
    final totalWatch = Stopwatch()..start();
    final cache = _makeDecodeCache(this);
    try {
      final tokens = List<int>.from(promptIds);
      final prompt = MlxArray.fromInt32List(
        promptIds,
        shape: [1, promptIds.length],
      );
      final promptWatch = Stopwatch()..start();
      var logits = _runWithCache(prompt, cache, fullLogits: true);
      promptWatch.stop();
      prompt.close();
      var firstTokenMs = 0.0;
      var stoppedByStopToken = false;
      final decodeWatch = Stopwatch()..start();
      try {
        for (var index = 0; index < maxNewTokens; index++) {
          final sampleWatch = Stopwatch()..start();
          final next = _nextTokenFromLogits(logits);
          sampleWatch.stop();
          if (index == 0) {
            firstTokenMs = sampleWatch.elapsedMicroseconds / 1000.0;
          }
          tokens.add(next);
          if (stopSet.contains(next)) {
            stoppedByStopToken = true;
            break;
          }
          if (index + 1 >= maxNewTokens) {
            break;
          }
          logits.close();
          final step = MlxArray.fromInt32List([next], shape: [1, 1]);
          logits = _runWithCache(step, cache, fullLogits: true);
          step.close();
        }
      } finally {
        decodeWatch.stop();
        logits.close();
      }
      totalWatch.stop();
      return (
        tokenIds: List<int>.unmodifiable(tokens),
        generatedTokenIds: List<int>.unmodifiable(
          tokens.sublist(promptIds.length),
        ),
        promptMs: promptWatch.elapsedMicroseconds / 1000.0,
        firstTokenMs: firstTokenMs,
        decodeMs: decodeWatch.elapsedMicroseconds / 1000.0,
        totalMs: totalWatch.elapsedMicroseconds / 1000.0,
        stoppedByStopToken: stoppedByStopToken,
      );
    } finally {
      cache.close();
    }
  }

  Map<String, Object?> debugTimedCachedGeneration(
    List<int> promptIds,
    int maxNewTokens, {
    int? eosTokenId,
  }) {
    final timed = timedGenerateGreedy(
      promptIds,
      maxNewTokens,
      stopTokenIds: eosTokenId == null ? const <int>[] : <int>[eosTokenId],
    );
    return {
      'generated_token_ids': timed.tokenIds,
      'prompt_ms': timed.promptMs,
      'first_token_ms': timed.firstTokenMs,
      'decode_ms': timed.decodeMs,
      'total_ms': timed.totalMs,
      'stopped_by_stop_token': timed.stoppedByStopToken,
    };
  }
}
