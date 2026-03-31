part of 'qwen3_5.dart';

extension Qwen3_5RunnerBench on Qwen3_5Runner {
  Map<String, Object?> debugTimedCachedGeneration(
    List<int> promptIds,
    int maxNewTokens, {
    int? eosTokenId,
  }) {
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
      final decodeWatch = Stopwatch();
      try {
        for (var index = 0; index < maxNewTokens; index++) {
          final next = _nextTokenFromLogits(logits);
          tokens.add(next);
          if (eosTokenId != null && next == eosTokenId) {
            break;
          }
          if (index + 1 >= maxNewTokens) {
            break;
          }
          logits.close();
          final step = MlxArray.fromInt32List([next], shape: [1, 1]);
          decodeWatch.start();
          logits = _runWithCache(step, cache, fullLogits: true);
          decodeWatch.stop();
          step.close();
        }
      } finally {
        logits.close();
      }
      return {
        'generated_token_ids': tokens,
        'prompt_ms': promptWatch.elapsedMicroseconds / 1000.0,
        'decode_ms': decodeWatch.elapsedMicroseconds / 1000.0,
      };
    } finally {
      cache.close();
    }
  }
}
