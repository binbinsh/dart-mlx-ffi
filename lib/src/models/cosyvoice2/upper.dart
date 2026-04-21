part of 'cosyvoice2.dart';

final class CosyVoice2UpperConfig {
  CosyVoice2UpperConfig({
    required this.hiddenSize,
    required this.numHiddenLayers,
    required this.intermediateSize,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.rmsNormEps,
    required this.vocabSize,
    required this.ropeTheta,
    required this.tieWordEmbeddings,
    required this.groupSize,
    required this.bits,
    required this.speechTokenSize,
  });

  factory CosyVoice2UpperConfig.fromJson(Map<String, Object?> json) {
    final quant = json['quantization'] as Map<String, Object?>? ?? const {};
    return CosyVoice2UpperConfig(
      hiddenSize: 896,
      numHiddenLayers: 24,
      intermediateSize: 4864,
      numAttentionHeads: 14,
      numKeyValueHeads: 2,
      rmsNormEps: 1e-6,
      vocabSize: 151936,
      ropeTheta: 1000000.0,
      tieWordEmbeddings: true,
      groupSize: (quant['group_size'] as num?)?.toInt() ?? 64,
      bits: (quant['bits'] as num?)?.toInt() ?? 4,
      speechTokenSize: (json['speech_token_size'] as num?)?.toInt() ?? 6561,
    );
  }

  final int hiddenSize;
  final int numHiddenLayers;
  final int intermediateSize;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final double rmsNormEps;
  final int vocabSize;
  final double ropeTheta;
  final bool tieWordEmbeddings;
  final int groupSize;
  final int bits;
  final int speechTokenSize;

  int get headDim => hiddenSize ~/ numAttentionHeads;
  int get speechVocabSize => speechTokenSize + 3;
}

final class _Cv2QuantLinear {
  _Cv2QuantLinear({
    required this.weight,
    required this.scales,
    required this.biases,
    this.bias,
  });

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray biases;
  final MlxArray? bias;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);
}

final class _Cv2Layer {
  _Cv2Layer({
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.gateProj,
    required this.upProj,
    required this.downProj,
    required this.inputNorm,
    required this.postNorm,
  });

  final _Cv2QuantLinear qProj;
  final _Cv2QuantLinear kProj;
  final _Cv2QuantLinear vProj;
  final _Cv2QuantLinear oProj;
  final _Cv2QuantLinear gateProj;
  final _Cv2QuantLinear upProj;
  final _Cv2QuantLinear downProj;
  final MlxArray inputNorm;
  final MlxArray postNorm;
}

final class _Cv2KvCache {
  MlxArray? keys;
  MlxArray? values;
  int offset = 0;

  (MlxArray, MlxArray) updateAndFetch(MlxArray nextKeys, MlxArray nextValues) {
    final currentKeys = keys;
    final currentValues = values;
    if (currentKeys == null || currentValues == null) {
      keys = nextKeys;
      values = nextValues;
      offset = nextKeys.shape[2];
      return (nextKeys, nextValues);
    }
    final mergedKeys = mx.concatenate([currentKeys, nextKeys], axis: 2);
    final mergedValues = mx.concatenate([currentValues, nextValues], axis: 2);
    currentKeys.close();
    currentValues.close();
    nextKeys.close();
    nextValues.close();
    keys = mergedKeys;
    values = mergedValues;
    offset = mergedKeys.shape[2];
    return (mergedKeys, mergedValues);
  }

  void close() {
    keys?.close();
    values?.close();
    keys = null;
    values = null;
    offset = 0;
  }
}

final class _Cv2ModelCache {
  _Cv2ModelCache(int layers)
    : layers = List<_Cv2KvCache>.generate(layers, (_) => _Cv2KvCache());

  final List<_Cv2KvCache> layers;

  void close() {
    for (final layer in layers) {
      layer.close();
    }
  }
}

final class CosyVoice2UpperRunner {
  CosyVoice2UpperRunner._({
    required this.snapshotPath,
    required this.config,
    required this.tensors,
    required List<_Cv2Layer> layers,
    required this.tokenizer,
    required this.embedTokens,
    required this.finalNorm,
    required this.llmEmbedding,
    required this.speechEmbedding,
    required this.llmDecoderWeight,
    required this.llmDecoderWeightT,
    required this.llmDecoderBias,
  }) : _layers = layers;

  factory CosyVoice2UpperRunner.load(
    String snapshotPath, {
    String? tokenizerPath,
  }) {
    final configJson =
        jsonDecode(File('$snapshotPath/config.json').readAsStringSync())
            as Map<String, Object?>;
    final config = CosyVoice2UpperConfig.fromJson(configJson);
    final tensors = loadTensorMap(snapshotPath);
    final layers = List<_Cv2Layer>.generate(config.numHiddenLayers, (index) {
      final prefix = 'qwen2.model.layers.$index';
      return _Cv2Layer(
        qProj: _linear(tensors, '$prefix.self_attn.q_proj'),
        kProj: _linear(tensors, '$prefix.self_attn.k_proj'),
        vProj: _linear(tensors, '$prefix.self_attn.v_proj'),
        oProj: _linear(tensors, '$prefix.self_attn.o_proj'),
        gateProj: _linear(tensors, '$prefix.mlp.gate_proj'),
        upProj: _linear(tensors, '$prefix.mlp.up_proj'),
        downProj: _linear(tensors, '$prefix.mlp.down_proj'),
        inputNorm: tensors['$prefix.input_layernorm.weight']!,
        postNorm: tensors['$prefix.post_attention_layernorm.weight']!,
      );
    });
    return CosyVoice2UpperRunner._(
      snapshotPath: snapshotPath,
      config: config,
      tensors: tensors,
      layers: layers,
      tokenizer: CosyVoice2BpeTokenizer.load(tokenizerPath ?? snapshotPath),
      embedTokens: tensors['qwen2.model.embed_tokens.weight']!,
      finalNorm: tensors['qwen2.model.norm.weight']!,
      llmEmbedding: tensors['llm.llm_embedding.weight']!,
      speechEmbedding: tensors['llm.speech_embedding.weight']!,
      llmDecoderWeight: tensors['llm.llm_decoder.weight']!,
      llmDecoderWeightT: tensors['llm.llm_decoder.weight']!.transpose(),
      llmDecoderBias: tensors['llm.llm_decoder.bias']!,
    );
  }

  final String snapshotPath;
  final CosyVoice2UpperConfig config;
  final Map<String, MlxArray> tensors;
  final List<_Cv2Layer> _layers;
  final CosyVoice2BpeTokenizer tokenizer;
  final MlxArray embedTokens;
  final MlxArray finalNorm;
  final MlxArray llmEmbedding;
  final MlxArray speechEmbedding;
  final MlxArray llmDecoderWeight;
  final MlxArray llmDecoderWeightT;
  final MlxArray llmDecoderBias;

  List<int> encodeText(String text) => tokenizer.encode(text);

  List<int> generateSpeechTokens({
    required String text,
    required String refText,
    required List<int> promptSpeechTokens,
    int sampling = 25,
    double maxTokenTextRatio = 20.0,
    double minTokenTextRatio = 2.0,
    int? seed,
    bool greedy = false,
  }) {
    final promptTextIds = tokenizer.encode(refText.trim());
    final textIds = tokenizer.encode(text.trim());
    if (textIds.isEmpty) {
      return const <int>[];
    }
    final random = math.Random(seed);
    final minLen = (textIds.length * minTokenTextRatio).toInt();
    final maxLen = (textIds.length * maxTokenTextRatio).toInt();

    final sequence = <MlxArray>[];
    final textEmbeds = _embedTextTokens([...promptTextIds, ...textIds]);
    final promptSpeechEmbeds = _embedSpeechTokens(promptSpeechTokens);
    sequence.add(_specialEmbed(0));
    sequence.add(textEmbeds);
    sequence.add(_specialEmbed(1));
    if (promptSpeechEmbeds != null) {
      sequence.add(promptSpeechEmbeds);
    }

    final decoded = <int>[];
    final cache = _Cv2ModelCache(_layers.length);
    try {
      final input = mx.concatenate(sequence, axis: 1);
      var logits = _runEmbeddings(input, cache: cache);
      input.close();
      for (var step = 0; step < maxLen; step++) {
        final next = _sampleToken(
          logits,
          decoded,
          random,
          topK: sampling,
          ignoreEos: step < minLen,
          greedy: greedy,
        );
        logits.close();
        if (next == config.speechTokenSize) {
          break;
        }
        decoded.add(next);
        final nextEmbed = _speechTokenEmbed(next);
        sequence.add(nextEmbed);
        if (step + 1 >= maxLen) {
          break;
        }
        logits = _runEmbeddings(nextEmbed, cache: cache);
      }
    } finally {
      cache.close();
      for (final item in sequence) {
        item.close();
      }
    }

    return decoded.where((token) => token <= config.speechTokenSize).toList();
  }

  MlxArray _runEmbeddings(MlxArray hidden, {_Cv2ModelCache? cache}) {
    final seqLen = hidden.shape[1];
    try {
      for (var index = 0; index < _layers.length; index++) {
        final layer = _layers[index];
        final norm1 = mx.fast.rmsNorm(
          hidden,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );
        final attn = _attention(
          layer,
          norm1,
          seqLen,
          cache: cache?.layers[index],
        );
        final h = mx.add(hidden, attn);
        attn.close();
        norm1.close();
        hidden.close();

        final norm2 = mx.fast.rmsNorm(
          h,
          weight: layer.postNorm,
          eps: config.rmsNormEps,
        );
        final mlpOut = _mlp(layer, norm2, seqLen);
        norm2.close();
        final next = mx.add(h, mlpOut);
        mlpOut.close();
        h.close();
        hidden = next;
      }

      final norm = mx.fast.rmsNorm(
        hidden,
        weight: finalNorm,
        eps: config.rmsNormEps,
      );
      hidden.close();
      final last = norm.slice(
        start: [0, seqLen - 1, 0],
        stop: [1, seqLen, config.hiddenSize],
      );
      norm.close();
      final logits = _dense(last.reshape([1, config.hiddenSize]));
      last.close();
      return logits;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray _attention(
    _Cv2Layer layer,
    MlxArray input,
    int seqLen, {
    _Cv2KvCache? cache,
  }) {
    final x2d = input.reshape([seqLen, config.hiddenSize]);
    final q = _quantLinear(
      x2d,
      layer.qProj,
      outDim: config.hiddenSize,
      addBias: true,
    );
    final k = _quantLinear(
      x2d,
      layer.kProj,
      outDim: config.numKeyValueHeads * config.headDim,
      addBias: true,
    );
    final v = _quantLinear(
      x2d,
      layer.vProj,
      outDim: config.numKeyValueHeads * config.headDim,
      addBias: true,
    );
    x2d.close();

    final q4 = q
        .reshape([1, seqLen, config.numAttentionHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    final k4 = k
        .reshape([1, seqLen, config.numKeyValueHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    final v4 = v
        .reshape([1, seqLen, config.numKeyValueHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    q.close();
    k.close();
    v.close();

    final qRope = mx.fast.rope(
      q4,
      dims: config.headDim,
      traditional: false,
      base: config.ropeTheta,
      offset: cache?.offset ?? 0,
    );
    final kRope = mx.fast.rope(
      k4,
      dims: config.headDim,
      traditional: false,
      base: config.ropeTheta,
      offset: cache?.offset ?? 0,
    );
    q4.close();
    k4.close();

    final repeatKv = config.numAttentionHeads ~/ config.numKeyValueHeads;
    MlxArray kBase = kRope;
    MlxArray vBase = v4;
    if (cache != null) {
      final fetched = cache.updateAndFetch(kRope, v4);
      kBase = fetched.$1;
      vBase = fetched.$2;
    }
    final kAttn = repeatKv > 1 ? kBase.repeat(repeatKv, axis: 1) : kBase;
    final vAttn = repeatKv > 1 ? vBase.repeat(repeatKv, axis: 1) : vBase;
    if (cache == null && !identical(kAttn, kBase)) {
      kBase.close();
    }
    if (cache == null && !identical(vAttn, vBase)) {
      vBase.close();
    }

    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kAttn,
      vAttn,
      scale: 1 / math.sqrt(config.headDim),
      maskMode: cache != null && seqLen == 1 ? '' : 'causal',
    );
    qRope.close();
    if (cache == null || !identical(kAttn, kBase)) {
      kAttn.close();
    }
    if (cache == null || !identical(vAttn, vBase)) {
      vAttn.close();
    }

    final merged = attn.transposeAxes([0, 2, 1, 3]).reshape([
      seqLen,
      config.hiddenSize,
    ]);
    attn.close();
    final projected = _quantLinear(
      merged,
      layer.oProj,
      outDim: config.hiddenSize,
      addBias: false,
    );
    merged.close();
    return projected.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray _mlp(_Cv2Layer layer, MlxArray input, int seqLen) {
    final x2d = input.reshape([seqLen, config.hiddenSize]);
    final gate = _quantLinear(
      x2d,
      layer.gateProj,
      outDim: config.intermediateSize,
      addBias: false,
    );
    final up = _quantLinear(
      x2d,
      layer.upProj,
      outDim: config.intermediateSize,
      addBias: false,
    );
    x2d.close();
    final sig = gate.sigmoid();
    final silu = gate * sig;
    sig.close();
    gate.close();
    final fused = silu * up;
    silu.close();
    up.close();
    final down = _quantLinear(
      fused,
      layer.downProj,
      outDim: config.hiddenSize,
      addBias: false,
    );
    fused.close();
    return down.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray _embedTextTokens(List<int> ids) {
    if (ids.isEmpty) {
      return MlxArray.fromFloat32List(
        const <double>[],
        shape: [1, 0, config.hiddenSize],
      );
    }
    final arr = MlxArray.fromInt32List(ids, shape: [1, ids.length]);
    try {
      return embedTokens.take(arr, axis: 0);
    } finally {
      arr.close();
    }
  }

  MlxArray? _embedSpeechTokens(List<int> ids) {
    if (ids.isEmpty) {
      return null;
    }
    final arr = MlxArray.fromInt32List(ids, shape: [1, ids.length]);
    try {
      return speechEmbedding.take(arr, axis: 0);
    } finally {
      arr.close();
    }
  }

  MlxArray _specialEmbed(int index) {
    final arr = MlxArray.fromInt32List([index], shape: [1, 1]);
    try {
      return llmEmbedding.take(arr, axis: 0);
    } finally {
      arr.close();
    }
  }

  MlxArray _speechTokenEmbed(int tokenId) {
    final arr = MlxArray.fromInt32List([tokenId], shape: [1, 1]);
    try {
      return speechEmbedding.take(arr, axis: 0);
    } finally {
      arr.close();
    }
  }

  MlxArray _quantLinear(
    MlxArray input,
    _Cv2QuantLinear linear, {
    required int outDim,
    required bool addBias,
  }) {
    final out = mx.quant.matmul(
      input,
      linear.matrix,
      transpose: true,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: 'affine',
    );
    if (!addBias || linear.bias == null) {
      return out;
    }
    final bias = linear.bias!.reshape([1, outDim]);
    try {
      return mx.add(out, bias);
    } finally {
      bias.close();
      out.close();
    }
  }

  MlxArray _dense(MlxArray last2d) {
    final logits = mx.matmul(last2d, llmDecoderWeightT);
    final bias = llmDecoderBias.reshape([1, config.speechVocabSize]);
    try {
      final out = mx.add(logits, bias);
      logits.close();
      return out.reshape([1, config.speechVocabSize]);
    } finally {
      bias.close();
    }
  }

  int _sampleToken(
    MlxArray logits,
    List<int> decoded,
    math.Random random, {
    required int topK,
    required bool ignoreEos,
    required bool greedy,
  }) {
    if (greedy) {
      var next = _argmaxTensor(logits);
      if (ignoreEos && next == config.speechTokenSize) {
        next = _argmaxMaskedTensor(logits, config.speechTokenSize);
      }
      return next;
    }
    final values = logits.astype(MlxDType.MLX_FLOAT32).toList().cast<double>();
    var next = _nucleusSample(values, random, topP: 0.8, topK: topK);
    if (decoded.isNotEmpty) {
      final recent = decoded.length > 10
          ? decoded.sublist(decoded.length - 10)
          : decoded;
      final repCount = recent.where((value) => value == next).length;
      if (repCount >= 1) {
        next = _sampleSoftmax(values, random);
      }
    }
    if (ignoreEos && next == config.speechTokenSize) {
      for (var trial = 0; trial < 100; trial++) {
        next = _nucleusSample(values, random, topP: 0.8, topK: topK);
        if (next != config.speechTokenSize) {
          break;
        }
      }
    }
    return next;
  }

  int _nucleusSample(
    List<double> logits,
    math.Random random, {
    required double topP,
    required int topK,
  }) {
    final probs = _softmax(logits);
    final pairs = <({int index, double value})>[
      for (var index = 0; index < probs.length; index++)
        (index: index, value: probs[index]),
    ]..sort((a, b) => b.value.compareTo(a.value));
    final limit = topK.clamp(1, pairs.length);
    var cumulative = 0.0;
    final selected = <({int index, double value})>[];
    for (var i = 0; i < limit; i++) {
      selected.add(pairs[i]);
      cumulative += pairs[i].value;
      if (cumulative >= topP) {
        break;
      }
    }
    return _sampleFromPairs(selected, random);
  }

  int _sampleSoftmax(List<double> logits, math.Random random) {
    final probs = _softmax(logits);
    final pairs = <({int index, double value})>[
      for (var index = 0; index < probs.length; index++)
        (index: index, value: probs[index]),
    ];
    return _sampleFromPairs(pairs, random);
  }

  int _sampleFromPairs(
    List<({int index, double value})> pairs,
    math.Random random,
  ) {
    final total = pairs.fold<double>(0.0, (sum, item) => sum + item.value);
    var target = random.nextDouble() * total;
    for (final pair in pairs) {
      target -= pair.value;
      if (target <= 0) {
        return pair.index;
      }
    }
    return pairs.last.index;
  }

  List<double> _softmax(List<double> logits) {
    final maxValue = logits.reduce(math.max);
    final exp = <double>[
      for (final value in logits) math.exp(value - maxValue),
    ];
    final sum = exp.fold<double>(0.0, (a, b) => a + b);
    return <double>[for (final value in exp) value / sum];
  }

  int _argmaxTensor(MlxArray logits) {
    final argmax = logits.argmax(axis: 1);
    try {
      return argmax.toScalarInt();
    } finally {
      argmax.close();
    }
  }

  int _argmaxMaskedTensor(MlxArray logits, int disallow) {
    final values = logits.toList().cast<double>();
    return _argmaxMasked(values, disallow);
  }

  int _argmaxMasked(List<double> logits, int disallow) {
    var bestIndex = 0;
    var bestValue = double.negativeInfinity;
    for (var index = 0; index < logits.length; index++) {
      if (index == disallow) {
        continue;
      }
      if (logits[index] > bestValue) {
        bestValue = logits[index];
        bestIndex = index;
      }
    }
    return bestIndex;
  }

  void close() {
    llmDecoderWeightT.close();
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }

  static _Cv2QuantLinear _linear(
    Map<String, MlxArray> tensors,
    String prefix,
  ) => _Cv2QuantLinear(
    weight: tensors['$prefix.weight']!,
    scales: tensors['$prefix.scales']!,
    biases: tensors['$prefix.biases']!,
    bias: tensors['$prefix.bias'],
  );
}
