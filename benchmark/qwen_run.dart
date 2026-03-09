import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

final class QwenConfig {
  QwenConfig({
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
  });

  factory QwenConfig.fromJson(Map<String, Object?> json) {
    final quant = json['quantization'] as Map<String, Object?>;
    return QwenConfig(
      hiddenSize: (json['hidden_size'] as num).toInt(),
      numHiddenLayers: (json['num_hidden_layers'] as num).toInt(),
      intermediateSize: (json['intermediate_size'] as num).toInt(),
      numAttentionHeads: (json['num_attention_heads'] as num).toInt(),
      numKeyValueHeads: (json['num_key_value_heads'] as num).toInt(),
      rmsNormEps: (json['rms_norm_eps'] as num).toDouble(),
      vocabSize: (json['vocab_size'] as num).toInt(),
      ropeTheta: (json['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
      tieWordEmbeddings: json['tie_word_embeddings'] as bool? ?? true,
      groupSize: (quant['group_size'] as num).toInt(),
      bits: (quant['bits'] as num).toInt(),
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

  int get headDim => hiddenSize ~/ numAttentionHeads;
}

final class QuantLinear {
  QuantLinear({
    required this.weights,
    required this.scales,
    required this.qBiases,
    this.bias,
  });

  final MlxArray weights;
  final MlxArray scales;
  final MlxArray qBiases;
  final MlxArray? bias;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weights, scales, qBiases);
}

final class LayerWeights {
  LayerWeights({
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

  final QuantLinear qProj;
  final QuantLinear kProj;
  final QuantLinear vProj;
  final QuantLinear oProj;
  final QuantLinear gateProj;
  final QuantLinear upProj;
  final QuantLinear downProj;
  final MlxArray inputNorm;
  final MlxArray postNorm;
}

final class QwenRunner {
  QwenRunner._(
    this.config,
    this.tensors,
    this.layers,
    this.embed,
    this.finalNorm,
    this.logitIndices,
  );

  factory QwenRunner.load(String snapshotPath) {
    final configJson = jsonDecode(
      File('$snapshotPath/config.json').readAsStringSync(),
    ) as Map<String, Object?>;
    final config = QwenConfig.fromJson(configJson);
    final tensors = loadTensorMap(snapshotPath);
    final layers = List<LayerWeights>.generate(config.numHiddenLayers, (index) {
      final prefix = 'model.layers.$index';
      return LayerWeights(
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
    final embed = QuantLinear(
      weights: tensors['model.embed_tokens.weight']!,
      scales: tensors['model.embed_tokens.scales']!,
      qBiases: tensors['model.embed_tokens.biases']!,
    );
    final finalNorm = tensors['model.norm.weight']!;
    final logitIndices = MlxArray.fromInt32List(
      List<int>.generate(16, (index) => index),
      shape: [16],
    );
    return QwenRunner._(config, tensors, layers, embed, finalNorm, logitIndices);
  }

  final QwenConfig config;
  final Map<String, MlxArray> tensors;
  final List<LayerWeights> layers;
  final QuantLinear embed;
  final MlxArray finalNorm;
  final MlxArray logitIndices;

  MlxArray run(List<int> tokenIds) {
    final seqLen = tokenIds.length;
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, seqLen]);
    final hidden = embedRows(ids);
    ids.close();
    return runHidden(hidden, seqLen);
  }

  MlxArray buildGraph(MlxArray ids) {
    final shape = ids.shape;
    if (shape.length != 2 || shape[0] != 1) {
      throw ArgumentError('Expected token ids shape [1, seq], got $shape.');
    }
    var hidden = embedRowsGraph(ids);
    final seqLen = shape[1];

    for (final layer in layers) {
      final norm1 = mx.fast.rmsNorm(
        hidden,
        weight: layer.inputNorm,
        eps: config.rmsNormEps,
      );
      final attn = attentionGraph(layer, norm1, seqLen);
      final h = mx.add(hidden, attn);
      final norm2 = mx.fast.rmsNorm(
        h,
        weight: layer.postNorm,
        eps: config.rmsNormEps,
      );
      final mlp = mlpGraph(layer, norm2, seqLen);
      hidden = mx.add(h, mlp);
    }

    final norm = mx.fast.rmsNorm(
      hidden,
      weight: finalNorm,
      eps: config.rmsNormEps,
    );
    return lmHeadPrefixGraph(norm, seqLen);
  }

  Map<String, List<double>> debugSlices(List<int> tokenIds, {int width = 16}) {
    final seqLen = tokenIds.length;
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, seqLen]);
    var hidden = embedRows(ids);
    ids.close();

    final out = <String, List<double>>{};
    try {
      out['embed'] = extractSlice(hidden, width: width);
      for (var index = 0; index < layers.length; index++) {
        final layer = layers[index];
        final norm1 = mx.fast.rmsNorm(
          hidden,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );
        final attn = attention(layer, norm1, seqLen);
        final h = mx.add(hidden, attn);
        out['block_${index}_resid1'] = extractSlice(h, width: width);

        final norm2 = mx.fast.rmsNorm(
          h,
          weight: layer.postNorm,
          eps: config.rmsNormEps,
        );
        final mlpOut = mlp(layer, norm2, seqLen);
        final next = mx.add(h, mlpOut);
        out['block_${index}_out'] = extractSlice(next, width: width);

        norm1.close();
        attn.close();
        hidden.close();
        norm2.close();
        mlpOut.close();
        h.close();
        hidden = next;
      }

      final norm = mx.fast.rmsNorm(
        hidden,
        weight: finalNorm,
        eps: config.rmsNormEps,
      );
      out['final_norm'] = extractSlice(norm, width: width);
      final logits = lmHeadPrefix(norm, seqLen).astype(MlxDType.MLX_FLOAT32);
      out['logits16'] = List<double>.from(logits.toList().cast<double>());
      logits.close();
      norm.close();
      return out;
    } finally {
      hidden.close();
    }
  }

  MlxArray runHidden(MlxArray hidden, int seqLen) {
    try {
      for (final layer in layers) {
        final norm1 = mx.fast.rmsNorm(
          hidden,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );
        final attn = attention(layer, norm1, seqLen);
        final h = mx.add(hidden, attn);
        attn.close();
        norm1.close();
        hidden.close();

        final norm2 = mx.fast.rmsNorm(
          h,
          weight: layer.postNorm,
          eps: config.rmsNormEps,
        );
        final mlpOut = mlp(layer, norm2, seqLen);
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
      final output = lmHeadPrefix(norm, seqLen);
      norm.close();
      MlxRuntime.evalAll([output]);
      return output;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray embedRows(MlxArray ids) {
    final shape = ids.shape;
    final rowsW = embed.weights.take(ids, axis: 0);
    final rowsS = embed.scales.take(ids, axis: 0);
    final rowsB = embed.qBiases.take(ids, axis: 0);
    final gathered = MlxQuantizedMatrix(rowsW, rowsS, rowsB);
    try {
      final dequantized = mx.quant.dequantize(
        gathered,
        groupSize: config.groupSize,
        bits: config.bits,
        mode: 'affine',
        dtype: MlxDType.MLX_FLOAT16,
      );
      return dequantized.reshape([shape[0], shape[1], config.hiddenSize]);
    } finally {
      gathered.close();
    }
  }

  MlxArray embedRowsGraph(MlxArray ids) {
    final shape = ids.shape;
    final rowsW = embed.weights.take(ids, axis: 0);
    final rowsS = embed.scales.take(ids, axis: 0);
    final rowsB = embed.qBiases.take(ids, axis: 0);
    final gathered = MlxQuantizedMatrix(rowsW, rowsS, rowsB);
    final dequantized = mx.quant.dequantize(
      gathered,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: 'affine',
      dtype: MlxDType.MLX_FLOAT16,
    );
    return dequantized.reshape([shape[0], shape[1], config.hiddenSize]);
  }

  MlxArray attention(LayerWeights layer, MlxArray input, int seqLen) {
    final x2d = input.reshape([seqLen, config.hiddenSize]);
    final q = linear2d(
      x2d,
      layer.qProj,
      outDim: config.hiddenSize,
      addBias: true,
    );
    final k = linear2d(
      x2d,
      layer.kProj,
      outDim: config.numKeyValueHeads * config.headDim,
      addBias: true,
    );
    final v = linear2d(
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
    );
    final kRope = mx.fast.rope(
      k4,
      dims: config.headDim,
      traditional: false,
      base: config.ropeTheta,
    );
    q4.close();
    k4.close();

    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kRope,
      v4,
      scale: 1 / math.sqrt(config.headDim),
      maskMode: 'causal',
    );
    qRope.close();
    kRope.close();
    v4.close();

    final merged =
        attn.transposeAxes([0, 2, 1, 3]).reshape([seqLen, config.hiddenSize]);
    attn.close();
    final projected = linear2d(
      merged,
      layer.oProj,
      outDim: config.hiddenSize,
      addBias: false,
    );
    merged.close();
    return projected.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray attentionGraph(LayerWeights layer, MlxArray input, int seqLen) {
    final x2d = input.reshape([seqLen, config.hiddenSize]);
    final q = linear2dGraph(
      x2d,
      layer.qProj,
      outDim: config.hiddenSize,
      addBias: true,
    );
    final k = linear2dGraph(
      x2d,
      layer.kProj,
      outDim: config.numKeyValueHeads * config.headDim,
      addBias: true,
    );
    final v = linear2dGraph(
      x2d,
      layer.vProj,
      outDim: config.numKeyValueHeads * config.headDim,
      addBias: true,
    );

    final q4 = q
        .reshape([1, seqLen, config.numAttentionHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    final k4 = k
        .reshape([1, seqLen, config.numKeyValueHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);
    final v4 = v
        .reshape([1, seqLen, config.numKeyValueHeads, config.headDim])
        .transposeAxes([0, 2, 1, 3]);

    final qRope = mx.fast.rope(
      q4,
      dims: config.headDim,
      traditional: false,
      base: config.ropeTheta,
    );
    final kRope = mx.fast.rope(
      k4,
      dims: config.headDim,
      traditional: false,
      base: config.ropeTheta,
    );

    final attn = mx.fast.scaledDotProductAttention(
      qRope,
      kRope,
      v4,
      scale: 1 / math.sqrt(config.headDim),
      maskMode: 'causal',
    );
    final merged =
        attn.transposeAxes([0, 2, 1, 3]).reshape([seqLen, config.hiddenSize]);
    final projected = linear2dGraph(
      merged,
      layer.oProj,
      outDim: config.hiddenSize,
      addBias: false,
    );
    return projected.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray mlp(LayerWeights layer, MlxArray input, int seqLen) {
    final x2d = input.reshape([seqLen, config.hiddenSize]);
    final gate = linear2d(
      x2d,
      layer.gateProj,
      outDim: config.intermediateSize,
      addBias: false,
    );
    final up = linear2d(
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
    final down = linear2d(
      fused,
      layer.downProj,
      outDim: config.hiddenSize,
      addBias: false,
    );
    fused.close();
    return down.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray mlpGraph(LayerWeights layer, MlxArray input, int seqLen) {
    final x2d = input.reshape([seqLen, config.hiddenSize]);
    final gate = linear2dGraph(
      x2d,
      layer.gateProj,
      outDim: config.intermediateSize,
      addBias: false,
    );
    final up = linear2dGraph(
      x2d,
      layer.upProj,
      outDim: config.intermediateSize,
      addBias: false,
    );
    final sig = gate.sigmoid();
    final silu = gate * sig;
    final fused = silu * up;
    final down = linear2dGraph(
      fused,
      layer.downProj,
      outDim: config.hiddenSize,
      addBias: false,
    );
    return down.reshape([1, seqLen, config.hiddenSize]);
  }

  MlxArray linear2d(
    MlxArray x2d,
    QuantLinear linear, {
    required int outDim,
    required bool addBias,
  }) {
    final y = mx.quant.matmul(
      x2d,
      linear.matrix,
      transpose: true,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: 'affine',
    );
    if (!addBias || linear.bias == null) {
      return y;
    }
    final reshapedBias = linear.bias!.reshape([1, outDim]);
    try {
      return mx.add(y, reshapedBias);
    } finally {
      reshapedBias.close();
      y.close();
    }
  }

  MlxArray linear2dGraph(
    MlxArray x2d,
    QuantLinear linear, {
    required int outDim,
    required bool addBias,
  }) {
    final y = mx.quant.matmul(
      x2d,
      linear.matrix,
      transpose: true,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: 'affine',
    );
    if (!addBias || linear.bias == null) {
      return y;
    }
    final reshapedBias = linear.bias!.reshape([1, outDim]);
    return mx.add(y, reshapedBias);
  }

  MlxArray lmHeadPrefix(MlxArray hidden3d, int seqLen) {
    final logits = mx.quant.matmul(
      hidden3d,
      embed.matrix,
      transpose: true,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: 'affine',
    );
    final slice = logits
        .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, 16])
        .reshape([1, 16]);
    logits.close();
    return slice;
  }

  MlxArray lmHeadPrefixGraph(MlxArray hidden3d, int seqLen) {
    final logits = mx.quant.matmul(
      hidden3d,
      embed.matrix,
      transpose: true,
      groupSize: config.groupSize,
      bits: config.bits,
      mode: 'affine',
    );
    final slice = logits
        .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, 16])
        .reshape([1, 16]);
    return slice;
  }

  List<double> extractSlice(MlxArray hidden, {int width = 16}) {
    final seqLen = hidden.shape[1];
    final slice = hidden
        .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, width])
        .reshape([1, width])
        .astype(MlxDType.MLX_FLOAT32);
    try {
      return List<double>.from(slice.toList().cast<double>());
    } finally {
      slice.close();
    }
  }

  void close() {
    logitIndices.close();
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }

  static QuantLinear _linear(Map<String, MlxArray> tensors, String prefix) =>
      QuantLinear(
        weights: tensors['$prefix.weight']!,
        scales: tensors['$prefix.scales']!,
        qBiases: tensors['$prefix.biases']!,
        bias: tensors['$prefix.bias'],
      );
}

Map<String, MlxArray> loadTensorMap(String snapshotPath) {
  final dir = Directory(snapshotPath);
  final files = dir
      .listSync()
      .whereType<File>()
      .where((file) => file.path.endsWith('.safetensors'))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));
  if (files.isEmpty) {
    throw StateError('No safetensors files found under $snapshotPath.');
  }

  final merged = <String, MlxArray>{};
  for (final file in files) {
    final loaded = mx.io.loadSafetensors(file.path);
    for (final entry in loaded.tensors.entries) {
      if (merged.containsKey(entry.key)) {
        throw StateError('Duplicate tensor key ${entry.key} in ${file.path}.');
      }
      merged[entry.key] = entry.value;
    }
  }
  return merged;
}
