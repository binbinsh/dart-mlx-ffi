part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// Chunked decoder prefill
// ---------------------------------------------------------------------------

extension PaddleOcrVlPrefill on PaddleOcrVlRunner {
  MlxArray _prefillFromIdsWithCache(
    List<int> promptIds,
    MlxArray positionIds,
    _ModelCache cache, {
    void Function(String message)? onStage,
  }) {
    final chunkSize = config.prefillChunkSizeForCurrentPlatform;
    if (chunkSize <= 0 || promptIds.length <= chunkSize) {
      final promptArr = MlxArray.fromInt32List(
        promptIds,
        shape: [1, promptIds.length],
      );
      try {
        final logits = _forwardWithCache(
          promptArr,
          positionIds,
          cache,
          maybeQuantizeAfter: false,
        );
        cache.maybeQuantize(config: config);
        cache.evalStates();
        return logits;
      } finally {
        promptArr.close();
      }
    }

    onStage?.call(
      'prefill: chunked ids enabled chunkSize=$chunkSize '
      'total=${promptIds.length}',
    );
    final logits = _prefillChunked(
      totalTokens: promptIds.length,
      positionIds: positionIds,
      cache: cache,
      onStage: onStage,
      runChunk: (start, end, posChunk) {
        final idsChunk = MlxArray.fromInt32List(
          promptIds.sublist(start, end),
          shape: [1, end - start],
        );
        try {
          return _forwardWithCache(
            idsChunk,
            posChunk,
            cache,
            maybeQuantizeAfter: false,
          );
        } finally {
          idsChunk.close();
        }
      },
    );
    cache.maybeQuantize(config: config);
    cache.evalStates();
    return logits;
  }

  MlxArray _prefillFromEmbeddingWithCache(
    MlxArray embeddings,
    MlxArray positionIds,
    _ModelCache cache, {
    void Function(String message)? onStage,
  }) {
    final totalTokens = embeddings.shape[1];
    final chunkSize = config.prefillChunkSizeForCurrentPlatform;
    if (chunkSize <= 0 || totalTokens <= chunkSize) {
      final logits = _forwardFromEmbedding(
        embeddings,
        positionIds,
        cache,
        maybeQuantizeAfter: false,
      );
      cache.maybeQuantize(config: config);
      cache.evalStates();
      return logits;
    }

    onStage?.call(
      'prefill: chunked embeddings enabled chunkSize=$chunkSize '
      'total=$totalTokens',
    );
    final logits = _prefillChunked(
      totalTokens: totalTokens,
      positionIds: positionIds,
      cache: cache,
      onStage: onStage,
      runChunk: (start, end, posChunk) {
        final embChunk = embeddings.slice(
          start: [0, start, 0],
          stop: [1, end, config.hiddenSize],
        );
          try {
            return _forwardFromEmbedding(
              embChunk,
              posChunk,
              cache,
              maybeQuantizeAfter: false,
            );
          } finally {
            embChunk.close();
          }
      },
    );
    cache.maybeQuantize(config: config);
    cache.evalStates();
    return logits;
  }

  MlxArray _prefillChunked({
    required int totalTokens,
    required MlxArray positionIds,
    required _ModelCache cache,
    required MlxArray Function(int start, int end, MlxArray posChunk) runChunk,
    void Function(String message)? onStage,
  }) {
    final chunkSize = config.prefillChunkSizeForCurrentPlatform;
    var processedTokens = 0;
    while (totalTokens - processedTokens > 1) {
      final remaining = (totalTokens - processedTokens) - 1;
      final nextChunkSize = math.min(chunkSize, remaining);
      final chunkEnd = processedTokens + nextChunkSize;
      final posChunk = positionIds.slice(
        start: [0, 0, processedTokens],
        stop: [3, 1, chunkEnd],
      );
      final chunkLogits = runChunk(processedTokens, chunkEnd, posChunk);
      posChunk.close();
      chunkLogits.close();
      cache.evalStates();
      if (config.enableAggressiveCacheClearingForCurrentPlatform) {
        try {
          MlxMemory.clearCache();
        } catch (_) {}
      }
      processedTokens = chunkEnd;
      onStage?.call('prefill: processed $processedTokens/$totalTokens tokens');
    }

    final tailPos = positionIds.slice(
      start: [0, 0, processedTokens],
      stop: [3, 1, totalTokens],
    );
    try {
      return runChunk(processedTokens, totalTokens, tailPos);
    } finally {
      tailPos.close();
    }
  }
}
