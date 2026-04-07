part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// Greedy decoding with KV cache for PaddleOCR-VL
// ---------------------------------------------------------------------------

int _nextTokenFromLogits(MlxArray logits) {
  final argmax = logits.argmax(axis: -1);
  try {
    return argmax.toScalarInt();
  } finally {
    argmax.close();
  }
}

/// Run greedy autoregressive decoding.
///
/// [runner] must already have its weights loaded.
/// [promptIds] are the token IDs for the full prompt (text + image tokens).
/// [positionIds] shape `[3, 1, totalPromptLen]`.
/// [maxNewTokens] maximum tokens to generate.
/// [eosTokenId] stop when this token is produced.
///
/// Returns the full list of tokens (prompt + generated).
List<int> _generateGreedy(
  PaddleOcrVlRunner runner,
  List<int> promptIds,
  MlxArray positionIds,
  int maxNewTokens, {
  required int eosTokenId,
}) {
  final cache = _ModelCache.create(
    numLayers: runner.config.numHiddenLayers,
    numKvHeads: runner.config.numKeyValueHeads,
    headDim: runner.config.headDim,
    maxSeqLen: runner.config.maxKvCacheSeqLenForCurrentPlatform,
  );
  try {
    final tokens = List<int>.from(promptIds);

    // Prefill: run entire prompt through the model
    final promptArr = MlxArray.fromInt32List(
      promptIds,
      shape: [1, promptIds.length],
    );
    var logits = runner._forwardWithCache(promptArr, positionIds, cache);
    promptArr.close();

    try {
      for (var step = 0; step < maxNewTokens; step++) {
        final next = _nextTokenFromLogits(logits);
        tokens.add(next);
        if (next == eosTokenId) break;
        if (step + 1 >= maxNewTokens) break;

        logits.close();

        // Decode step: single token
        final stepArr = MlxArray.fromInt32List([next], shape: [1, 1]);
        final stepPos = runner._textPositionIds(1, offset: cache.offset);
        logits = runner._forwardWithCache(stepArr, stepPos, cache);
        stepArr.close();
        stepPos.close();
      }
    } finally {
      logits.close();
    }
    return tokens;
  } finally {
    cache.close();
  }
}
