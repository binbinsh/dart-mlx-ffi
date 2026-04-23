part of 'qwen3_tts.dart';

extension Qwen3TtsDebug on Qwen3TtsEngine {
  /// Returns step-0 parity diagnostics as a JSON-encodable map.
  /// Runs prefill + one talker forward pass only; does not iterate.
  Map<String, dynamic> debugStep0(String text) {
    final source = text.trim();
    final reference = _reference;
    if (source.isEmpty) {
      throw StateError('debugStep0 requires non-empty text.');
    }
    if (reference == null) {
      throw StateError('debugStep0 requires a prepared reference.');
    }
    final prefill = _prepareIclInputs(source, reference);
    final talkerCache = _talker.createTalkerCache();
    try {
      MlxRuntime.evalAll([prefill.inputEmbeds]);
      final embedStats = _statsOf('input_embeds', prefill.inputEmbeds);

      final out = _talker.forwardTalker(prefill.inputEmbeds, talkerCache);
      final logits = out.logits;
      final hidden = out.hidden;
      final seq = hidden.shape[1];
      final lastHidden = hidden
          .slice(start: [0, seq - 1, 0], stop: [1, seq, bundle.manifest.talker.hiddenSize])
          .reshape([1, bundle.manifest.talker.hiddenSize]);

      MlxArray suppressed = logits;
      if (_suppressTokenIds.isNotEmpty) {
        final fill = MlxArray.full([], double.negativeInfinity).astype(logits.dtype);
        final values = mx.broadcastTo(fill, [1, _suppressTokenIds.length]);
        final put = mx.putAlongAxis(
          logits,
          _suppressIdx.reshape([1, _suppressTokenIds.length]),
          values,
          axis: 1,
        );
        fill.close();
        values.close();
        suppressed = put;
      }
      final step0Token = suppressed.argmax(axis: 1, keepDims: true);
      MlxRuntime.evalAll([lastHidden, suppressed, step0Token]);

      final hiddenStats = _statsOf('hidden_last', lastHidden);
      final logitsStats = _statsOf('logits_last', suppressed);
      final logitsRow = suppressed.toFloat32List();
      final topk = _topKFromRow(logitsRow, 20);
      final tokenId = step0Token.toScalarInt();

      if (!identical(suppressed, logits)) {
        logits.close();
      }
      suppressed.close();
      lastHidden.close();
      hidden.close();
      step0Token.close();

      return <String, dynamic>{
        'input_embeds_stats': embedStats,
        'post_prefill_hidden_stats': hiddenStats,
        'step0_logits_stats': logitsStats,
        'step0_logits_topk': topk,
        'step0_token': tokenId,
        'eos_token_id': bundle.manifest.talker.codecEosTokenId,
        'vocab_size': bundle.manifest.talker.vocabSize,
      };
    } finally {
      prefill.ttsPadEmbed.close();
      prefill.refCodes.close();
      _talker.closeCache(talkerCache);
    }
  }

  Map<String, dynamic> _statsOf(String name, MlxArray arr) {
    final flat = arr.toFloat32List();
    final n = flat.length;
    var sq = 0.0;
    for (final v in flat) {
      sq += v * v;
    }
    const head = 8;
    const tail = 8;
    final headList = <double>[for (var i = 0; i < head && i < n; i++) flat[i]];
    final tailList = n > tail
        ? <double>[for (var i = n - tail; i < n; i++) flat[i]]
        : const <double>[];
    return <String, dynamic>{
      'name': name,
      'shape': arr.shape,
      'count': n,
      'l2': math.sqrt(sq),
      'head': headList,
      'tail': tailList,
    };
  }

  List<Map<String, dynamic>> _topKFromRow(Float32List row, int k) {
    final pairs = <MapEntry<int, double>>[
      for (var i = 0; i < row.length; i++) MapEntry(i, row[i]),
    ];
    pairs.sort((a, b) => b.value.compareTo(a.value));
    return [
      for (var i = 0; i < k && i < pairs.length; i++)
        {'id': pairs[i].key, 'value': pairs[i].value},
    ];
  }

  List<List<int>> debugGenerateFrames(
    String text, {
    double temperature = 0.0,
    double? repetitionPenalty,
  }) {
    final source = text.trim();
    final reference = _reference;
    if (source.isEmpty) {
      throw StateError('debugGenerateFrames requires non-empty text.');
    }
    if (reference == null) {
      throw StateError('debugGenerateFrames requires a prepared reference.');
    }
    final prefill = _prepareIclInputs(source, reference);
    final talkerCache = _talker.createTalkerCache();
    final codeCache = _talker.createCodePredictorCache();
    final generatedTokenIds = <int>[];
    final frames = <List<int>>[];
    final resolvedPenalty = repetitionPenalty ?? _repetitionPenalty;
    var inputEmbeds = prefill.inputEmbeds;
    final ttsPadEmbed = prefill.ttsPadEmbed;
    try {
      final maxTokens = math.min(4096, math.max(75, prefill.targetTokenCount * 6));
      for (var step = 0; step < maxTokens; step++) {
        final out = _talker.forwardTalker(inputEmbeds, talkerCache);
        inputEmbeds = out.hidden;
        final nextToken = _sampleToken(
          out.logits,
          generatedTokenIds,
          temperature: temperature,
          topK: _topK,
          repetitionPenalty: resolvedPenalty,
        );
        out.logits.close();
        MlxRuntime.evalAll([nextToken]);
        final nextId = nextToken.toScalarInt();
        if (nextId == bundle.manifest.talker.codecEosTokenId) {
          nextToken.close();
          break;
        }
        generatedTokenIds.add(nextId);
        _talker.resetCache(codeCache);
        final lastHidden = inputEmbeds.slice(
          start: [0, inputEmbeds.shape[1] - 1, 0],
          stop: [1, inputEmbeds.shape[1], bundle.manifest.talker.hiddenSize],
        );
        final codeTokens = <MlxArray>[nextToken];
        final frameInts = <int>[nextId];
        try {
          for (var codeIdx = 0;
              codeIdx < bundle.manifest.talker.numCodeGroups - 1;
              codeIdx++) {
            late final MlxArray codeInput;
            if (codeIdx == 0) {
              final code0 = _talker.embedCodecToken(codeTokens.last);
              codeInput = mx.concatenate([lastHidden, code0], axis: 1);
              code0.close();
            } else {
              codeInput = _talker.embedPredictorToken(codeIdx - 1, codeTokens.last);
            }
            final logits = _talker.forwardCodePredictor(codeInput, codeIdx, codeCache);
            codeInput.close();
            final nextCode = _sampleCodeToken(
              logits,
              temperature: temperature,
              topK: _topK,
            );
            logits.close();
            MlxRuntime.evalAll([nextCode]);
            frameInts.add(nextCode.toScalarInt());
            codeTokens.add(nextCode);
          }
          frames.add(frameInts);

          var codecEmbed = _talker.embedCodecToken(codeTokens.first);
          for (var index = 1; index < codeTokens.length; index++) {
            final extra = _talker.embedPredictorToken(index - 1, codeTokens[index]);
            final sum = mx.add(codecEmbed, extra);
            codecEmbed.close();
            extra.close();
            codecEmbed = sum;
          }
          final nextInput = mx.add(ttsPadEmbed, codecEmbed);
          codecEmbed.close();
          inputEmbeds.close();
          inputEmbeds = nextInput;
        } finally {
          lastHidden.close();
          for (final code in codeTokens) {
            code.close();
          }
        }
      }
    } finally {
      inputEmbeds.close();
      ttsPadEmbed.close();
      prefill.refCodes.close();
      _talker.closeCache(talkerCache);
      _talker.closeCache(codeCache);
    }
    return frames;
  }

  Float32List debugDecodeCodeFrames(List<List<int>> frames) {
    if (frames.isEmpty) {
      return Float32List(0);
    }
    final groups = frames.first.length;
    final pending = <MlxArray>[];
    try {
      for (final frame in frames) {
        if (frame.length != groups) {
          throw StateError('All Qwen3-TTS code frames must share the same group count.');
        }
        pending.add(MlxArray.fromInt32List(frame, shape: [1, groups]));
      }
      _decoder.resetStreamingState();
      return _decodePending(pending);
    } finally {
      for (final item in pending) {
        item.close();
      }
      pending.clear();
      _decoder.resetStreamingState();
    }
  }

}
