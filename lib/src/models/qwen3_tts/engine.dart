part of 'qwen3_tts.dart';

final class Qwen3TtsChunk {
  const Qwen3TtsChunk({required this.pcm});

  final Float32List pcm;
}

final class _Qwen3TtsIclPrefill {
  const _Qwen3TtsIclPrefill({
    required this.inputEmbeds,
    required this.ttsPadEmbed,
    required this.refCodes,
    required this.targetTokenCount,
  });

  final MlxArray inputEmbeds;
  final MlxArray ttsPadEmbed;
  final MlxArray refCodes;
  final int targetTokenCount;
}

final class Qwen3TtsEngine {
  Qwen3TtsEngine._({required this.bundle, required _Qwen3TtsTalker talker, required _Qwen3TtsSpeechDecoder decoder})
    : _talker = talker,
      _decoder = decoder,
      _tokenizer = Qwen3AsrBpeTokenizer.load(bundle.manifest.rootPath),
      _temperature = ((bundle.generationConfig['temperature'] as num?) ?? 0.9).toDouble(),
      _topK = ((bundle.generationConfig['top_k'] as num?) ?? 50).toInt(),
      _repetitionPenalty =
          ((bundle.generationConfig['repetition_penalty'] as num?) ?? 1.05).toDouble(),
      _streamingChunkSize = math.max(
        1,
        (((bundle.generationConfig['streaming_interval'] as num?) ?? 2.0).toDouble() * 12.5).round(),
      ),
      _suppressTokenIds = [
        for (var i = bundle.manifest.talker.vocabSize - 1024;
            i < bundle.manifest.talker.vocabSize;
            i++)
          if (i != bundle.manifest.talker.codecEosTokenId) i,
      ],
      _suppressIdx = MlxArray.fromInt32List(
        [
          for (var i = bundle.manifest.talker.vocabSize - 1024;
              i < bundle.manifest.talker.vocabSize;
              i++)
            if (i != bundle.manifest.talker.codecEosTokenId) i,
        ],
        shape: [bundle.manifest.talker.vocabSize > 1024 ? 1023 : 0],
      ),
      _eosPutIdx = MlxArray.fromInt32List(
        [bundle.manifest.talker.codecEosTokenId],
        shape: [1, 1],
      );

  factory Qwen3TtsEngine.load(String bundlePath) {
    final bundle = Qwen3TtsBundle.load(bundlePath);
    return Qwen3TtsEngine._(
      bundle: bundle,
      talker: _Qwen3TtsTalker(bundle),
      decoder: _Qwen3TtsSpeechDecoder(bundle),
    );
  }

  final Qwen3TtsBundle bundle;
  final _Qwen3TtsTalker _talker;
  final _Qwen3TtsSpeechDecoder _decoder;
  final Qwen3AsrBpeTokenizer _tokenizer;
  final double _temperature;
  final int _topK;
  final double _repetitionPenalty;
  final int _streamingChunkSize;
  final List<int> _suppressTokenIds;
  final MlxArray _suppressIdx;
  final MlxArray _eosPutIdx;
  Qwen3TtsPreparedReference? _reference;

  int get sampleRate => bundle.manifest.sampleRate;

  List<String> get builtinVoices => const <String>['Qwen3-TTS'];

  void setPreparedReference(Qwen3TtsPreparedReference reference) {
    _reference = reference;
  }

  Stream<Qwen3TtsChunk> synthesiseStream(
    String text, {
    double speed = 1.0,
    double? temperature,
    int? topK,
    double? repetitionPenalty,
  }) async* {
    final source = text.trim();
    final reference = _reference;
    if (source.isEmpty) {
      return;
    }
    if (reference == null) {
      throw StateError('Qwen3-TTS prepared reference is not configured.');
    }
    final _ = speed;
    final resolvedTemperature = temperature ?? _temperature;
    final resolvedTopK = topK ?? _topK;
    final resolvedPenalty = repetitionPenalty ?? _repetitionPenalty;
    final prefill = _prepareIclInputs(source, reference);
    final talkerCache = _talker.createTalkerCache();
    final codeCache = _talker.createCodePredictorCache();
    final generatedTokenIds = <int>[];
    final pendingCodes = <MlxArray>[];
    _decoder.resetStreamingState();

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
          temperature: resolvedTemperature,
          topK: resolvedTopK,
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
              temperature: resolvedTemperature,
              topK: resolvedTopK,
            );
            logits.close();
            codeTokens.add(nextCode);
          }
          final allCodes = mx.concatenate(codeTokens, axis: 1);
          pendingCodes.add(allCodes);

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

        if (pendingCodes.length >= _streamingChunkSize) {
          final chunk = _decodePending(pendingCodes);
          if (chunk.isNotEmpty) {
            yield Qwen3TtsChunk(pcm: chunk);
          }
        }
      }
      if (pendingCodes.isNotEmpty) {
        final chunk = _decodePending(pendingCodes);
        if (chunk.isNotEmpty) {
          yield Qwen3TtsChunk(pcm: chunk);
        }
      }
    } finally {
      inputEmbeds.close();
      ttsPadEmbed.close();
      prefill.refCodes.close();
      _talker.closeCache(talkerCache);
      _talker.closeCache(codeCache);
      for (final code in pendingCodes) {
        code.close();
      }
      _decoder.resetStreamingState();
    }
  }

  Float32List _decodePending(List<MlxArray> pendingCodes) {
    final stacked = mx.stack(pendingCodes, axis: 1); // [1, tokens, groups]
    for (final code in pendingCodes) {
      code.close();
    }
    pendingCodes.clear();
    final transposed = stacked.transposeAxes([0, 2, 1]);
    stacked.close();
    final audio = _decoder.streamingStep(transposed);
    transposed.close();
    try {
      final flat = audio.toList();
      return Float32List.fromList(
        flat.map((value) => (value as num).toDouble()).toList(growable: false),
      );
    } finally {
      audio.close();
    }
  }

  _Qwen3TtsIclPrefill _prepareIclInputs(String text, Qwen3TtsPreparedReference reference) {
    final cfg = bundle.manifest.talker;
    final refCodes = reference.createRefCodesArray();
    final speakerEmbed = reference.createSpeakerEmbeddingArray();
    final refChat = '<|im_start|>assistant\n${reference.refText}<|im_end|>\n';
    final refIds = _tokenizer.encode(refChat);
    final targetChat = '<|im_start|>assistant\n$text<|im_end|>\n<|im_start|>assistant\n';
    final targetIds = _tokenizer.encode(targetChat);
    if (refIds.length < 5 || targetIds.length < 8) {
      speakerEmbed.close();
      refCodes.close();
      throw StateError('Qwen3-TTS prompt templates encoded unexpectedly short token sequences.');
    }

    final ttsEmbeds = _talker.projectText(
      _talker.embedTextIds([
        bundle.manifest.ttsBosTokenId,
        bundle.manifest.ttsEosTokenId,
        bundle.manifest.ttsPadTokenId,
      ]),
    );
    final ttsBos = ttsEmbeds.slice(start: [0, 0, 0], stop: [1, 1, cfg.hiddenSize]);
    final ttsEos = ttsEmbeds.slice(start: [0, 1, 0], stop: [1, 2, cfg.hiddenSize]);
    final ttsPad = ttsEmbeds.slice(start: [0, 2, 0], stop: [1, 3, cfg.hiddenSize]);
    ttsEmbeds.close();

    final combinedTextIds = <int>[...refIds.sublist(3, refIds.length - 2), ...targetIds.sublist(3, targetIds.length - 5)];
    var textEmbed = _talker.projectText(_talker.embedTextIds(combinedTextIds));
    final withEos = mx.concatenate([textEmbed, ttsEos], axis: 1);
    textEmbed.close();
    textEmbed = withEos;

    final refTime = refCodes.shape[2];
    final firstCb = refCodes.slice(start: [0, 0, 0], stop: [1, 1, refTime]).reshape([1, refTime]);
    var refCodecEmbed = _talker.embedCodecToken(firstCb);
    firstCb.close();
    for (var index = 0; index < cfg.numCodeGroups - 1; index++) {
      final group = refCodes.slice(
        start: [0, index + 1, 0],
        stop: [1, index + 2, refTime],
      ).reshape([1, refTime]);
      final extra = _talker.embedPredictorToken(index, group);
      group.close();
      final sum = mx.add(refCodecEmbed, extra);
      refCodecEmbed.close();
      extra.close();
      refCodecEmbed = sum;
    }
    final codecBos = _talker.embedCodecIds([cfg.codecBosId]);
    final codecEmbedIcl = mx.concatenate([codecBos, refCodecEmbed], axis: 1);
    codecBos.close();
    refCodecEmbed.close();

    final codecPad = _talker.embedCodecIds([cfg.codecPadId]);
    final textWithCodecPad = mx.add(
      textEmbed,
      mx.broadcastTo(codecPad, [1, textEmbed.shape[1], cfg.hiddenSize]),
    );
    final codecWithTextPad = mx.add(
      codecEmbedIcl,
      mx.broadcastTo(ttsPad, [1, codecEmbedIcl.shape[1], cfg.hiddenSize]),
    );
    codecPad.close();
    final iclInput = mx.concatenate([textWithCodecPad, codecWithTextPad], axis: 1);
    textEmbed.close();
    textWithCodecPad.close();
    codecEmbedIcl.close();
    codecWithTextPad.close();

    final int? languageId = null;
    final codecPrefill = languageId == null
        ? <int>[cfg.codecNoThinkId, cfg.codecThinkBosId, cfg.codecThinkEosId]
        : <int>[cfg.codecThinkId, cfg.codecThinkBosId, languageId, cfg.codecThinkEosId];
    var codecPrefix = _talker.embedCodecIds(codecPrefill);
    final codecPrefixSuffix = _talker.embedCodecIds([cfg.codecPadId, cfg.codecBosId]);
    final prefixWithSpeaker = mx.concatenate([codecPrefix, speakerEmbed, codecPrefixSuffix], axis: 1);
    codecPrefix.close();
    speakerEmbed.close();
    codecPrefixSuffix.close();
    codecPrefix = prefixWithSpeaker;

    final roleEmbed = _talker.projectText(_talker.embedTextIds(targetIds.take(3).toList(growable: false)));
    final padCount = codecPrefix.shape[1] - 2;
    final padEmbeds = mx.broadcastTo(ttsPad, [1, padCount, cfg.hiddenSize]);
    final combinedPrefixBase = mx.concatenate([padEmbeds, ttsBos], axis: 1);
    padEmbeds.close();
    ttsBos.close();
    final prefixBody = codecPrefix.slice(
      start: [0, 0, 0],
      stop: [1, codecPrefix.shape[1] - 1, cfg.hiddenSize],
    );
    final combinedPrefix = mx.add(combinedPrefixBase, prefixBody);
    combinedPrefixBase.close();
    prefixBody.close();
    codecPrefix.close();

    final inputEmbeds = mx.concatenate([roleEmbed, combinedPrefix, iclInput], axis: 1);
    roleEmbed.close();
    combinedPrefix.close();
    iclInput.close();
    return _Qwen3TtsIclPrefill(
      inputEmbeds: inputEmbeds,
      ttsPadEmbed: ttsPad,
      refCodes: refCodes,
      targetTokenCount: _tokenizer.encode(text).length,
    );
  }

  MlxArray _sampleToken(
    MlxArray logits,
    List<int> generatedTokens, {
    required double temperature,
    required int topK,
    required double repetitionPenalty,
  }) {
    var current = logits;
    if (_suppressTokenIds.isNotEmpty) {
      final fill = MlxArray.full([], double.negativeInfinity).astype(current.dtype);
      final values = mx.broadcastTo(fill, [1, _suppressTokenIds.length]);
      final suppressed = mx.putAlongAxis(current, _suppressIdx.reshape([1, _suppressTokenIds.length]), values, axis: 1);
      fill.close();
      values.close();
      current.close();
      current = suppressed;
    }
    if (generatedTokens.isNotEmpty && repetitionPenalty != 1.0) {
      final unique = generatedTokens.toSet().toList(growable: false);
      final ids = MlxArray.fromInt32List(unique, shape: [unique.length]);
      final selected = current.take(ids, axis: 1);
      final zero = MlxArray.full([], 0.0).astype(current.dtype);
      final penalty = MlxArray.full([], repetitionPenalty).astype(current.dtype);
      final multiplied = selected * penalty;
      final divided = mx.divide(selected, penalty);
      final mask = mx.less(selected, zero);
      final penalized = mx.where(mask, multiplied, divided);
      final rowIdx = ids.reshape([1, unique.length]);
      final updated = mx.putAlongAxis(current, rowIdx, penalized, axis: 1);
      ids.close();
      selected.close();
      zero.close();
      penalty.close();
      multiplied.close();
      divided.close();
      mask.close();
      penalized.close();
      current.close();
      current = updated;
    }
    final eos = current.slice(
      start: [0, bundle.manifest.talker.codecEosTokenId],
      stop: [1, bundle.manifest.talker.codecEosTokenId + 1],
    );
    if (topK > 0 && topK < bundle.manifest.talker.vocabSize) {
      final topValues = current.topK(topK, axis: 1);
      final threshold = topValues.slice(start: [0, topK - 1], stop: [1, topK]);
      topValues.close();
      final mask = mx.greaterEqual(current, threshold);
      threshold.close();
      final fill = MlxArray.full([], double.negativeInfinity).astype(current.dtype);
      final kept = mx.where(mask, current, mx.broadcastTo(fill, current.shape));
      fill.close();
      mask.close();
      current.close();
      current = kept;
    }
    final restored = mx.putAlongAxis(current, _eosPutIdx, eos, axis: 1);
    eos.close();
    current.close();
    current = restored;
    if (temperature <= 0) {
      final sampled = current.argmax(axis: 1, keepDims: true);
      current.close();
      return sampled;
    }
    final temp = MlxArray.full([], temperature).astype(current.dtype);
    final scaled = mx.divide(current, temp);
    temp.close();
    current.close();
    final sampled = MlxRandom.categorical(scaled, axis: 1).reshape([1, 1]);
    scaled.close();
    return sampled;
  }

  MlxArray _sampleCodeToken(
    MlxArray logits, {
    required double temperature,
    required int topK,
  }) {
    var current = logits;
    if (topK > 0 && topK < bundle.manifest.talker.codePredictor.vocabSize) {
      final topValues = current.topK(topK, axis: 1);
      final threshold = topValues.slice(start: [0, topK - 1], stop: [1, topK]);
      topValues.close();
      final mask = mx.greaterEqual(current, threshold);
      threshold.close();
      final fill = MlxArray.full([], double.negativeInfinity).astype(current.dtype);
      final kept = mx.where(mask, current, mx.broadcastTo(fill, current.shape));
      fill.close();
      mask.close();
      current.close();
      current = kept;
    }
    if (temperature <= 0) {
      final sampled = current.argmax(axis: 1, keepDims: true);
      current.close();
      return sampled;
    }
    final temp = MlxArray.full([], temperature).astype(current.dtype);
    final scaled = mx.divide(current, temp);
    temp.close();
    current.close();
    final sampled = MlxRandom.categorical(scaled, axis: 1).reshape([1, 1]);
    scaled.close();
    return sampled;
  }

  void close() {
    _decoder.resetStreamingState();
    _suppressIdx.close();
    _eosPutIdx.close();
    bundle.close();
  }
}
