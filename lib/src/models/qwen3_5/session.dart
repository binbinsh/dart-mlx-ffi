part of 'qwen3_5.dart';

final class Qwen35PromptSession {
  Qwen35PromptSession._(
    this._runner,
    this.promptTokenIds,
    this._cache,
    this._promptLogits,
  );

  final Qwen3_5Runner _runner;
  final List<int> promptTokenIds;
  final _ModelDecodeCache _cache;
  final MlxArray _promptLogits;

  static Qwen35PromptSession prime(
    Qwen3_5Runner runner,
    List<int> promptTokenIds,
  ) {
    final cache = _makeDecodeCache(runner);
    final prompt = MlxArray.fromInt32List(
      promptTokenIds,
      shape: [1, promptTokenIds.length],
    );
    try {
      final logits = runner._runWithCache(prompt, cache, fullLogits: true);
      return Qwen35PromptSession._(
        runner,
        List<int>.unmodifiable(promptTokenIds),
        cache,
        logits,
      );
    } finally {
      prompt.close();
    }
  }

  Qwen35TimedGeneration timedGenerateGreedy(
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
        tokenIds: promptTokenIds,
        generatedTokenIds: const <int>[],
        promptMs: 0,
        firstTokenMs: 0,
        decodeMs: 0,
        totalMs: 0,
        stoppedByStopToken: false,
      );
    }
    final cache = _cloneDecodeCache(_cache);
    final stopSet = stopTokenIds.toSet();
    final tokens = List<int>.from(promptTokenIds);
    final totalWatch = Stopwatch()..start();
    final logits = _cloneArray(_promptLogits);
    var currentLogits = logits;
    var firstTokenMs = 0.0;
    var stoppedByStopToken = false;
    final decodeWatch = Stopwatch()..start();
    try {
      for (var index = 0; index < maxNewTokens; index++) {
        final sampleWatch = Stopwatch()..start();
        final next = _nextTokenFromLogits(currentLogits);
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
        currentLogits.close();
        final step = MlxArray.fromInt32List([next], shape: [1, 1]);
        currentLogits = _runner._runWithCache(step, cache, fullLogits: true);
        step.close();
      }
    } finally {
      decodeWatch.stop();
      currentLogits.close();
      cache.close();
    }
    totalWatch.stop();
    return (
      tokenIds: List<int>.unmodifiable(tokens),
      generatedTokenIds: List<int>.unmodifiable(
        tokens.sublist(promptTokenIds.length),
      ),
      promptMs: 0,
      firstTokenMs: firstTokenMs,
      decodeMs: decodeWatch.elapsedMicroseconds / 1000.0,
      totalMs: totalWatch.elapsedMicroseconds / 1000.0,
      stoppedByStopToken: stoppedByStopToken,
    );
  }

  void close() {
    _promptLogits.close();
    _cache.close();
  }
}

extension Qwen35PromptSessionApi on Qwen3_5Runner {
  Qwen35PromptSession createPromptSession(List<int> promptTokenIds) =>
      Qwen35PromptSession.prime(this, promptTokenIds);
}

_ModelDecodeCache _cloneDecodeCache(_ModelDecodeCache source) =>
    _ModelDecodeCache([
      for (final layer in source.layers)
        switch (layer) {
          final _KvDecodeCache kv => _cloneKvDecodeCache(kv),
          final _LinearDecodeCache linear => _cloneLinearDecodeCache(linear),
        },
    ]);

_KvDecodeCache _cloneKvDecodeCache(_KvDecodeCache source) {
  final clone = _KvDecodeCache();
  if (source.keys != null) {
    clone.keys = _cloneArray(source.keys!);
  }
  if (source.values != null) {
    clone.values = _cloneArray(source.values!);
  }
  clone.offset = source.offset;
  return clone;
}

_LinearDecodeCache _cloneLinearDecodeCache(_LinearDecodeCache source) {
  final clone = _LinearDecodeCache();
  final convState = source.takeConvState();
  if (convState != null) {
    source.replaceConvState(convState);
    clone.replaceConvState(_cloneArray(convState));
  }
  final state = source.takeState();
  if (state != null) {
    source.replaceState(state);
    clone.replaceState(_cloneArray(state));
  }
  return clone;
}

MlxArray _cloneArray(MlxArray input) {
  final zeros = MlxArray.zeros(input.shape, dtype: input.dtype);
  try {
    final copy = mx.add(input, zeros);
    MlxRuntime.evalAll([copy]);
    return copy;
  } finally {
    zeros.close();
  }
}
