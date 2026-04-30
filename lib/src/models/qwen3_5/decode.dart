part of 'qwen3_5.dart';

_ModelDecodeCache _makeDecodeCache(Qwen3_5Runner runner) {
  final layers = <_LayerDecodeCache>[
    for (final layer in runner._layers)
      if (layer.fullAttention != null)
        _KvDecodeCache()
      else
        _LinearDecodeCache(),
  ];
  return _ModelDecodeCache(layers);
}

int _nextTokenFromLogits(MlxArray logits) {
  final argmax = logits.argmax(axis: 1);
  try {
    return argmax.toScalarInt();
  } finally {
    argmax.close();
  }
}

List<int> _generateGreedyCached(
  Qwen3_5Runner runner,
  List<int> promptIds,
  int maxNewTokens, {
  int? eosTokenId,
}) {
  final cache = _makeDecodeCache(runner);
  try {
    final tokens = List<int>.from(promptIds);
    final prompt = MlxArray.fromInt32List(
      promptIds,
      shape: [1, promptIds.length],
    );
    var logits = runner._runWithCache(prompt, cache, fullLogits: true);
    prompt.close();
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
        logits = runner._runWithCache(step, cache, fullLogits: true);
        step.close();
      }
    } finally {
      logits.close();
    }
    return tokens;
  } finally {
    cache.close();
  }
}

void _warmCachedDecodeSteps(
  Qwen3_5Runner runner,
  List<int> promptIds,
  int decodeSteps,
) {
  if (decodeSteps <= 0) {
    return;
  }
  final cache = _makeDecodeCache(runner);
  try {
    final prompt = MlxArray.fromInt32List(
      promptIds,
      shape: [1, promptIds.length],
    );
    var logits = runner._runWithCache(prompt, cache, fullLogits: true);
    prompt.close();
    try {
      for (var index = 0; index < decodeSteps; index++) {
        final next = _nextTokenFromLogits(logits);
        logits.close();
        final step = MlxArray.fromInt32List([next], shape: [1, 1]);
        logits = runner._runWithCache(step, cache, fullLogits: true);
        step.close();
      }
    } finally {
      logits.close();
    }
  } finally {
    cache.close();
  }
}
