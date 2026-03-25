library;

import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'args.dart';
import 'core.dart';
import 'quant.dart';

final class AlbertEmbeddings {
  AlbertEmbeddings({
    required this.wordEmbeddings,
    required this.positionEmbeddings,
    required this.tokenTypeEmbeddings,
    required this.layerNorm,
  });

  factory AlbertEmbeddings.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required AlbertModelArgs config,
    required KittenQuantConfig quant,
  }) {
    return AlbertEmbeddings(
      wordEmbeddings: Embedding.load(
        tensors,
        '$prefix.word_embeddings',
        quant: quant,
      ),
      positionEmbeddings: Embedding.load(
        tensors,
        '$prefix.position_embeddings',
        quant: quant,
      ),
      tokenTypeEmbeddings: Embedding.load(
        tensors,
        '$prefix.token_type_embeddings',
        quant: quant,
      ),
      layerNorm: LayerNorm.load(
        tensors,
        '$prefix.LayerNorm',
        eps: config.layerNormEps,
      ),
    );
  }

  final Embedding wordEmbeddings;
  final Embedding positionEmbeddings;
  final Embedding tokenTypeEmbeddings;
  final LayerNorm layerNorm;

  MlxArray call(
    MlxArray inputIds, {
    MlxArray? tokenTypeIds,
    MlxArray? positionIds,
  }) {
    final batch = inputIds.shape[0];
    final seqLen = inputIds.shape[1];
    final resolvedPositionIds =
        positionIds ??
        mx.arange(0, seqLen.toDouble(), 1.0, dtype: MlxDType.MLX_INT32).reshape(
          [1, seqLen],
        );
    final resolvedTokenTypeIds =
        tokenTypeIds ??
        MlxArray.zeros([batch, seqLen], dtype: MlxDType.MLX_INT32);
    final words = wordEmbeddings(inputIds);
    final positions = positionEmbeddings(resolvedPositionIds);
    final tokenTypes = tokenTypeEmbeddings(resolvedTokenTypeIds);
    final summed = (words + positions) + tokenTypes;
    try {
      return layerNorm(summed);
    } finally {
      summed.close();
      tokenTypes.close();
      positions.close();
      words.close();
      if (positionIds == null) {
        resolvedPositionIds.close();
      }
      if (tokenTypeIds == null) {
        resolvedTokenTypeIds.close();
      }
    }
  }
}

final class AlbertSelfAttention {
  AlbertSelfAttention({
    required this.config,
    required this.query,
    required this.key,
    required this.value,
    required this.dense,
    required this.layerNorm,
    this.activationQuant = false,
  });

  factory AlbertSelfAttention.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required AlbertModelArgs config,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    return AlbertSelfAttention(
      config: config,
      query: Linear.load(
        tensors,
        '$prefix.query',
        quant: quant,
        activationQuant: activationQuant,
      ),
      key: Linear.load(
        tensors,
        '$prefix.key',
        quant: quant,
        activationQuant: activationQuant,
      ),
      value: Linear.load(
        tensors,
        '$prefix.value',
        quant: quant,
        activationQuant: activationQuant,
      ),
      dense: Linear.load(
        tensors,
        '$prefix.dense',
        quant: quant,
        activationQuant: activationQuant,
      ),
      layerNorm: LayerNorm.load(
        tensors,
        '$prefix.LayerNorm',
        eps: config.layerNormEps,
      ),
      activationQuant: activationQuant,
    );
  }

  final AlbertModelArgs config;
  final Linear query;
  final Linear key;
  final Linear value;
  final Linear dense;
  final LayerNorm layerNorm;
  final bool activationQuant;

  MlxArray call(MlxArray hiddenStates, {MlxArray? attentionMask}) {
    final hiddenIn = maybeFakeQuant(hiddenStates, activationQuant);
    final mixedQuery = query(hiddenIn);
    final mixedKey = key(hiddenIn);
    final mixedValue = value(hiddenIn);
    if (!identical(hiddenIn, hiddenStates)) {
      hiddenIn.close();
    }
    final queryLayer = _transposeForScores(mixedQuery);
    final keyLayer = _transposeForScores(mixedKey);
    final valueLayer = _transposeForScores(mixedValue);
    mixedQuery.close();
    mixedKey.close();
    mixedValue.close();
    final keyT = mx.transposeAxes(keyLayer, [0, 1, 3, 2]);
    final scores = mx.matmul(queryLayer, keyT);
    final scale = scalar(math.sqrt(config.attentionHeadSize.toDouble()));
    var attentionScores = scores / scale;
    scores.close();
    scale.close();
    keyT.close();
    if (attentionMask != null) {
      final withMask = attentionScores + attentionMask;
      attentionScores.close();
      attentionScores = withMask;
    }
    final probs = mx.softmax(attentionScores, axis: -1, precise: true);
    attentionScores.close();
    final context = mx.matmul(probs, valueLayer);
    probs.close();
    valueLayer.close();
    final transposed = mx.transposeAxes(context, [0, 2, 1, 3]);
    context.close();
    final merged = transposed.reshape([
      hiddenStates.shape[0],
      hiddenStates.shape[1],
      config.allHeadSize,
    ]);
    transposed.close();
    final contextIn = maybeFakeQuant(merged, activationQuant);
    final projected = dense(contextIn);
    if (!identical(contextIn, merged)) {
      contextIn.close();
    }
    merged.close();
    final residual = projected + hiddenStates;
    projected.close();
    try {
      return layerNorm(residual);
    } finally {
      residual.close();
      queryLayer.close();
      keyLayer.close();
    }
  }

  MlxArray _transposeForScores(MlxArray x) {
    final reshaped = x.reshape([
      x.shape[0],
      x.shape[1],
      config.numAttentionHeads,
      config.attentionHeadSize,
    ]);
    try {
      return mx.transposeAxes(reshaped, [0, 2, 1, 3]);
    } finally {
      reshaped.close();
    }
  }
}

final class AlbertLayer {
  AlbertLayer({
    required this.attention,
    required this.fullLayerNorm,
    required this.ffn,
    required this.ffnOutput,
    this.activationQuant = false,
  });

  factory AlbertLayer.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required AlbertModelArgs config,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    return AlbertLayer(
      attention: AlbertSelfAttention.load(
        tensors,
        prefix: '$prefix.attention',
        config: config,
        quant: quant,
        activationQuant: activationQuant,
      ),
      fullLayerNorm: LayerNorm.load(
        tensors,
        '$prefix.full_layer_layer_norm',
        eps: config.layerNormEps,
      ),
      ffn: Linear.load(
        tensors,
        '$prefix.ffn',
        quant: quant,
        activationQuant: activationQuant,
      ),
      ffnOutput: Linear.load(
        tensors,
        '$prefix.ffn_output',
        quant: quant,
        activationQuant: activationQuant,
      ),
      activationQuant: activationQuant,
    );
  }

  final AlbertSelfAttention attention;
  final LayerNorm fullLayerNorm;
  final Linear ffn;
  final Linear ffnOutput;
  final bool activationQuant;

  MlxArray call(MlxArray hiddenStates, {MlxArray? attentionMask}) {
    final attentionOutput = attention(
      hiddenStates,
      attentionMask: attentionMask,
    );
    final ffInput = maybeFakeQuant(attentionOutput, activationQuant);
    final ffHidden = ffn(ffInput);
    if (!identical(ffInput, attentionOutput)) {
      ffInput.close();
    }
    final activated = tanhGelu(ffHidden);
    ffHidden.close();
    final ffOutput = ffnOutput(activated);
    activated.close();
    final residual = ffOutput + attentionOutput;
    ffOutput.close();
    attentionOutput.close();
    try {
      return fullLayerNorm(residual);
    } finally {
      residual.close();
    }
  }
}

final class AlbertLayerGroup {
  AlbertLayerGroup(this.layers);

  factory AlbertLayerGroup.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required AlbertModelArgs config,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    final layers = List<AlbertLayer>.generate(
      config.innerGroupNum,
      (index) => AlbertLayer.load(
        tensors,
        prefix: '$prefix.albert_layers.$index',
        config: config,
        quant: quant,
        activationQuant: activationQuant,
      ),
      growable: false,
    );
    return AlbertLayerGroup(layers);
  }

  final List<AlbertLayer> layers;

  MlxArray call(MlxArray hiddenStates, {MlxArray? attentionMask}) {
    var current = hiddenStates;
    for (final layer in layers) {
      final next = layer(current, attentionMask: attentionMask);
      if (!identical(current, hiddenStates)) {
        current.close();
      }
      current = next;
    }
    return current;
  }
}

final class AlbertEncoder {
  AlbertEncoder({
    required this.config,
    required this.embeddingHiddenMappingIn,
    required this.groups,
    this.activationQuant = false,
  });

  factory AlbertEncoder.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required AlbertModelArgs config,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    return AlbertEncoder(
      config: config,
      embeddingHiddenMappingIn: Linear.load(
        tensors,
        '$prefix.embedding_hidden_mapping_in',
        quant: quant,
        activationQuant: activationQuant,
      ),
      groups: List<AlbertLayerGroup>.generate(
        config.numHiddenGroups,
        (index) => AlbertLayerGroup.load(
          tensors,
          prefix: '$prefix.albert_layer_groups.$index',
          config: config,
          quant: quant,
          activationQuant: activationQuant,
        ),
        growable: false,
      ),
      activationQuant: activationQuant,
    );
  }

  final AlbertModelArgs config;
  final Linear embeddingHiddenMappingIn;
  final List<AlbertLayerGroup> groups;
  final bool activationQuant;

  MlxArray call(MlxArray hiddenStates, {MlxArray? attentionMask}) {
    final hiddenIn = maybeFakeQuant(hiddenStates, activationQuant);
    var current = embeddingHiddenMappingIn(hiddenIn);
    if (!identical(hiddenIn, hiddenStates)) {
      hiddenIn.close();
    }
    for (var index = 0; index < config.numHiddenLayers; index++) {
      final groupIndex =
          (index / (config.numHiddenLayers / config.numHiddenGroups)).floor();
      final next = groups[groupIndex](current, attentionMask: attentionMask);
      current.close();
      current = next;
    }
    return current;
  }
}

final class AlbertOutput {
  const AlbertOutput(this.sequenceOutput, this.pooledOutput);

  final MlxArray sequenceOutput;
  final MlxArray pooledOutput;
}

final class KittenAlbert {
  KittenAlbert({
    required this.config,
    required this.embeddings,
    required this.encoder,
    required this.pooler,
  });

  factory KittenAlbert.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required AlbertModelArgs config,
    required KittenQuantConfig quant,
    bool activationQuant = false,
  }) {
    return KittenAlbert(
      config: config,
      embeddings: AlbertEmbeddings.load(
        tensors,
        prefix: '$prefix.embeddings',
        config: config,
        quant: quant,
      ),
      encoder: AlbertEncoder.load(
        tensors,
        prefix: '$prefix.encoder',
        config: config,
        quant: quant,
        activationQuant: activationQuant,
      ),
      pooler: Linear.load(
        tensors,
        '$prefix.pooler',
        quant: quant,
        activationQuant: activationQuant,
      ),
    );
  }

  final AlbertModelArgs config;
  final AlbertEmbeddings embeddings;
  final AlbertEncoder encoder;
  final Linear pooler;

  AlbertOutput call(
    MlxArray inputIds, {
    MlxArray? tokenTypeIds,
    MlxArray? attentionMask,
  }) {
    final embeddingOutput = embeddings(inputIds, tokenTypeIds: tokenTypeIds);
    MlxArray? expandedMask;
    if (attentionMask != null) {
      final maskFloat = attentionMask.astype(MlxDType.MLX_FLOAT32);
      final one = scalar(1.0);
      final neg = scalar(-10000.0);
      final step1 = maskFloat.expandDims(1);
      final step2 = step1.expandDims(1);
      final inverted = one - step2;
      expandedMask = inverted * neg;
      maskFloat.close();
      one.close();
      neg.close();
      step1.close();
      step2.close();
      inverted.close();
    }
    final sequenceOutput = encoder(
      embeddingOutput,
      attentionMask: expandedMask,
    );
    embeddingOutput.close();
    expandedMask?.close();
    final firstTokenIndex = MlxArray.fromInt32List([0], shape: [1]);
    final firstToken = sequenceOutput.take(firstTokenIndex, axis: 1).reshape([
      sequenceOutput.shape[0],
      sequenceOutput.shape[2],
    ]);
    firstTokenIndex.close();
    final pooled = pooler(firstToken).tanh();
    firstToken.close();
    return AlbertOutput(sequenceOutput, pooled);
  }
}
