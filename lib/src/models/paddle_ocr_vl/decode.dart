part of 'paddle_ocr_vl.dart';

// ---------------------------------------------------------------------------
// Greedy decoding with KV cache for PaddleOCR-VL
// ---------------------------------------------------------------------------

int _nextTokenFromLogits(MlxArray logits) {
  final flat = logits.reshape([logits.size]);
  final flatF32 = flat.dtype == MlxDType.MLX_FLOAT32
      ? flat
      : flat.astype(MlxDType.MLX_FLOAT32);
  try {
    return flatF32.argmaxFlatScalarInt();
  } on MlxException catch (e) {
    PaddleOcrVlDebugOverrides.traceSink?.call(
      'sample_token fallback flat_native_helper_failed='
      '${e.message} logitsShape=${logits.shape} logitsDType=${logits.dtype} '
      'flatDType=${flatF32.dtype}',
    );
    try {
      final flatArgmax = flatF32.argmax();
      try {
        return flatArgmax.toScalarInt();
      } on MlxException catch (flatError) {
        PaddleOcrVlDebugOverrides.traceSink?.call(
          'sample_token fallback flat_argmax_failed='
          '${flatError.message} flatShape=${flatF32.shape} '
          'argmaxDType=${flatArgmax.dtype}',
        );
      } finally {
        flatArgmax.close();
      }
    } on MlxException catch (flatSetupError) {
      PaddleOcrVlDebugOverrides.traceSink?.call(
        'sample_token fallback flat_setup_failed=${flatSetupError.message}',
      );
    }
    try {
      final argmax = logits.argmax(axis: -1);
      try {
        argmax.eval();
        return argmax.toScalarInt();
      } finally {
        argmax.close();
      }
    } on MlxException catch (helperError) {
      PaddleOcrVlDebugOverrides.traceSink?.call(
        'sample_token fallback axis_argmax_failed=${helperError.message}',
      );
      rethrow;
    }
  } finally {
    if (!identical(flatF32, flat)) {
      flatF32.close();
    }
    flat.close();
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
  final cache = _ModelCache.create(config: runner.config);
  try {
    final tokens = List<int>.from(promptIds);
    var logits = runner._prefillFromIdsWithCache(promptIds, positionIds, cache);

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
