// ignore_for_file: camel_case_types

part of 'qwen3_5.dart';

final class Qwen3_5Runner {
  Qwen3_5Runner._(
    this.config,
    this.tensors,
    this.textPrefix,
    this._layers,
    this._embedWeights,
    this.finalNorm,
    this._lmHeadWeights,
    this._logitPrefixIndices,
    this._logitPrefixMatrix,
  );

  factory Qwen3_5Runner.load(String snapshotPath) {
    final config = Qwen3_5Config.fromSnapshot(snapshotPath);
    final tensors = loadTensorMap(snapshotPath);
    final textPrefix = _detectTextPrefix(tensors.keys);
    final embed = _LinearBase.load(tensors, '${textPrefix}embed_tokens');
    final finalNorm = tensors['${textPrefix}norm.weight']!;
    final lmHead = config.tieWordEmbeddings
        ? null
        : _firstLoadLinear(tensors, _lmHeadPrefixes(textPrefix));
    final layers = List<_LayerWeights>.generate(config.numHiddenLayers, (index) {
      final prefix = '${textPrefix}layers.$index.';
      return _LayerWeights(
        inputNorm: tensors['${prefix}input_layernorm.weight']!,
        postNorm: tensors['${prefix}post_attention_layernorm.weight']!,
        fullAttention: config.isLinearLayer(index)
            ? null
            : _FullAttentionWeights(
                qProj: _LinearBase.load(tensors, '${prefix}self_attn.q_proj'),
                kProj: _LinearBase.load(tensors, '${prefix}self_attn.k_proj'),
                vProj: _LinearBase.load(tensors, '${prefix}self_attn.v_proj'),
                oProj: _LinearBase.load(tensors, '${prefix}self_attn.o_proj'),
                qNormWeight: tensors['${prefix}self_attn.q_norm.weight'],
                kNormWeight: tensors['${prefix}self_attn.k_norm.weight'],
              ),
        linearAttention: !config.isLinearLayer(index)
            ? null
            : _LinearAttentionWeights(
                convWeight: tensors['${prefix}linear_attn.conv1d.weight']!,
                inProjQkv: _LinearBase.load(
                  tensors,
                  '${prefix}linear_attn.in_proj_qkv',
                ),
                inProjZ: _LinearBase.load(
                  tensors,
                  '${prefix}linear_attn.in_proj_z',
                ),
                inProjB: _LinearBase.load(
                  tensors,
                  '${prefix}linear_attn.in_proj_b',
                ),
                inProjA: _LinearBase.load(
                  tensors,
                  '${prefix}linear_attn.in_proj_a',
                ),
                dtBias: tensors['${prefix}linear_attn.dt_bias']!,
                aLog: tensors['${prefix}linear_attn.A_log']!,
                normWeight: tensors['${prefix}linear_attn.norm.weight']!,
                outProj: _LinearBase.load(
                  tensors,
                  '${prefix}linear_attn.out_proj',
                ),
              ),
        denseMlp: config.isMoe
            ? null
            : _DenseMlpWeights(
                gateProj: _LinearBase.load(tensors, '${prefix}mlp.gate_proj'),
                upProj: _LinearBase.load(tensors, '${prefix}mlp.up_proj'),
                downProj: _LinearBase.load(tensors, '${prefix}mlp.down_proj'),
              ),
        moe: !config.isMoe
            ? null
            : _MoeWeights(
                gate: _LinearBase.load(tensors, '${prefix}mlp.gate'),
                switchGateProj: _SwitchLinearBase.load(
                  tensors,
                  '${prefix}mlp.switch_mlp.gate_proj',
                ),
                switchUpProj: _SwitchLinearBase.load(
                  tensors,
                  '${prefix}mlp.switch_mlp.up_proj',
                ),
                switchDownProj: _SwitchLinearBase.load(
                  tensors,
                  '${prefix}mlp.switch_mlp.down_proj',
                ),
                sharedGateProj: _LinearBase.load(
                  tensors,
                  '${prefix}mlp.shared_expert.gate_proj',
                ),
                sharedUpProj: _LinearBase.load(
                  tensors,
                  '${prefix}mlp.shared_expert.up_proj',
                ),
                sharedDownProj: _LinearBase.load(
                  tensors,
                  '${prefix}mlp.shared_expert.down_proj',
                ),
                sharedExpertGate: _LinearBase.load(
                  tensors,
                  '${prefix}mlp.shared_expert_gate',
                ),
              ),
      );
    });
    return Qwen3_5Runner._(
      config,
      tensors,
      textPrefix,
      layers,
      embed,
      finalNorm,
      lmHead,
      MlxArray.fromInt32List(List<int>.generate(16, (index) => index), shape: [16]),
      _buildLogitPrefixMatrix(
        config.tieWordEmbeddings ? embed : lmHead!,
        config: config,
      ),
    );
  }

  final Qwen3_5Config config;
  final Map<String, MlxArray> tensors;
  final String textPrefix;
  final List<_LayerWeights> _layers;
  final _LinearBase _embedWeights;
  final MlxArray finalNorm;
  final _LinearBase? _lmHeadWeights;
  final MlxArray _logitPrefixIndices;
  final MlxArray _logitPrefixMatrix;

  MlxArray run(List<int> tokenIds) {
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, tokenIds.length]);
    try {
      final hidden = _embed(ids);
      return _runHidden(hidden, tokenIds.length);
    } finally {
      ids.close();
    }
  }

  MlxArray buildGraph(MlxArray ids) {
    final hidden = _embed(ids);
    return _runHidden(hidden, ids.shape[1]);
  }

  MlxArray _embed(MlxArray ids) {
    final rows = ids.reshape([ids.shape[1]]);
    try {
      if (_embedWeights case final _QuantLinear q) {
        final rowsW = q.weight.take(rows, axis: 0);
        final rowsS = q.scales.take(rows, axis: 0);
        final rowsB = q.biases?.take(rows, axis: 0);
        final gathered = MlxQuantizedMatrix(rowsW, rowsS, rowsB);
        try {
          final out = mx.quant.dequantize(
            gathered,
            groupSize: config.groupSize,
            bits: config.bits,
            mode: config.mode,
            dtype: config.computeDType,
          );
          return out.reshape([1, ids.shape[1], config.hiddenSize]);
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      if (_embedWeights case final _DenseLinear d) {
        final out = d.weight.take(rows, axis: 0);
        return out.reshape([1, ids.shape[1], config.hiddenSize]);
      }
      throw StateError('Unsupported embedding type.');
    } finally {
      rows.close();
    }
  }

  MlxArray _runHidden(MlxArray hidden, int seqLen) {
    final debug = Platform.environment['QWEN35_DEBUG'] == '1';
    final trace = Platform.environment['QWEN35_TRACE'] == '1';
    final traceStage = Platform.environment['QWEN35_TRACE_STAGE'] == '1';
    final traceLayer = int.tryParse(Platform.environment['QWEN35_TRACE_LAYER'] ?? '');
    final evalEvery = int.tryParse(Platform.environment['QWEN35_EVAL_EVERY'] ?? '') ?? 0;
    final dumpLayer = int.tryParse(Platform.environment['QWEN35_DUMP_LAYER'] ?? '');
    final dumpStage = Platform.environment['QWEN35_DUMP_STAGE'];
    final dumpPath = Platform.environment['QWEN35_DUMP_PATH'];
    if (trace) {
      stderr.writeln('qwen35_run: embed ${_previewLastToken(hidden)}');
    }
    try {
      for (var index = 0; index < _layers.length; index++) {
        if (debug) {
          stderr.writeln('qwen35_run: layer $index start');
        }
        final layer = _layers[index];
        final norm1 = mx.fast.rmsNorm(
          hidden,
          weight: layer.inputNorm,
          eps: config.rmsNormEps,
        );
        if (traceStage && index == (traceLayer ?? 0)) {
          stderr.writeln('qwen35_run: layer$index norm1 ${_previewLastToken(norm1)}');
        }
        final attn = layer.fullAttention != null
            ? _fullAttention(layer.fullAttention!, norm1, seqLen, layerIndex: index)
            : _linearAttention(layer.linearAttention!, norm1, seqLen, layerIndex: index);
        if (dumpLayer != null &&
            dumpPath != null &&
            dumpLayer == index &&
            dumpStage == 'attn') {
          _dumpAny(attn, dumpPath);
        }
        if (traceStage && index == (traceLayer ?? 0)) {
          stderr.writeln('qwen35_run: layer$index attn ${_previewLastToken(attn)}');
        }
        final h = mx.add(hidden, attn);
        attn.close();
        norm1.close();
        hidden.close();

        final norm2 = mx.fast.rmsNorm(
          h,
          weight: layer.postNorm,
          eps: config.rmsNormEps,
        );
        final mlp = layer.moe != null
            ? _moeMlp(layer.moe!, norm2, seqLen)
            : _denseMlp(layer.denseMlp!, norm2, seqLen, layerIndex: index);
        if (dumpLayer != null &&
            dumpPath != null &&
            dumpLayer == index &&
            dumpStage == 'mlp') {
          _dumpAny(mlp, dumpPath);
        }
        if (traceStage && index == (traceLayer ?? 0)) {
          stderr.writeln('qwen35_run: layer$index mlp ${_previewLastToken(mlp)}');
        }
        norm2.close();
        final next = mx.add(h, mlp);
        mlp.close();
        h.close();
        hidden = next;
        if (dumpLayer != null &&
            dumpPath != null &&
            dumpLayer == index &&
            (dumpStage == null || dumpStage == 'hidden')) {
          _dumpLastToken(hidden, dumpPath);
        }
        if (evalEvery > 0 && (index + 1) % evalEvery == 0) {
          MlxRuntime.evalAll([hidden]);
        }
        if (trace) {
          stderr.writeln('qwen35_run: layer $index ${_previewLastToken(hidden)}');
        }
        if (debug) {
          stderr.writeln('qwen35_run: layer $index done');
        }
      }

      final norm = mx.fast.rmsNorm(
        hidden,
        weight: finalNorm,
        eps: config.rmsNormEps,
      );
      hidden.close();
      final output = _lmHead(norm, seqLen);
      norm.close();
      MlxRuntime.evalAll([output]);
      return output;
    } catch (_) {
      hidden.close();
      rethrow;
    }
  }

  MlxArray _lmHead(MlxArray hidden, int seqLen) {
    final last = hidden
        .slice(start: [0, seqLen - 1, 0], stop: [1, seqLen, config.hiddenSize])
        .reshape([1, config.hiddenSize]);
    try {
      final linear = config.tieWordEmbeddings ? _embedWeights : _lmHeadWeights!;
      return _logitPrefix(last, linear);
    } finally {
      last.close();
    }
  }

  MlxArray _logitPrefix(MlxArray last2d, _LinearBase linear) {
    return mx.matmul(last2d, _logitPrefixMatrix.transpose()).reshape([1, 16]);
  }

  void close() {
    _logitPrefixMatrix.close();
    _logitPrefixIndices.close();
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }

  static _LinearBase? _maybeLoadLinear(
    Map<String, MlxArray> tensors,
    String prefix,
  ) {
    if (!tensors.containsKey('$prefix.weight')) {
      return null;
    }
    return _LinearBase.load(tensors, prefix);
  }

  static _LinearBase? _firstLoadLinear(
    Map<String, MlxArray> tensors,
    Iterable<String> prefixes,
  ) {
    for (final prefix in prefixes) {
      final linear = _maybeLoadLinear(tensors, prefix);
      if (linear != null) {
        return linear;
      }
    }
    return null;
  }

  static Iterable<String> _lmHeadPrefixes(String textPrefix) sync* {
    yield 'lm_head';
    yield '${textPrefix}lm_head';
    if (textPrefix.endsWith('model.')) {
      yield '${textPrefix.substring(0, textPrefix.length - 'model.'.length)}lm_head';
    }
  }

  static MlxArray _buildLogitPrefixMatrix(
    _LinearBase linear, {
    required Qwen3_5Config config,
  }) {
    final indices = MlxArray.fromInt32List(
      List<int>.generate(16, (index) => index),
      shape: [16],
    );
    try {
      if (linear is _QuantLinear) {
        final rowsW = linear.weight.take(indices, axis: 0);
        final rowsS = linear.scales.take(indices, axis: 0);
        final rowsB = linear.biases?.take(indices, axis: 0);
        try {
          return mx.quant.dequantize(
            MlxQuantizedMatrix(rowsW, rowsS, rowsB),
            groupSize: config.groupSize,
            bits: config.bits,
            mode: config.mode,
            dtype: config.computeDType,
          );
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      if (linear is _DenseLinear) {
        return linear.weight.take(indices, axis: 0);
      }
      throw StateError('Unsupported lm_head type.');
    } finally {
      indices.close();
    }
  }

  static String _detectTextPrefix(Iterable<String> keys) {
    final candidates = [
      for (final key in keys)
        if (key.endsWith('layers.0.input_layernorm.weight'))
          key.substring(
            0,
            key.length - 'layers.0.input_layernorm.weight'.length,
          ),
    ]..sort((a, b) => a.length.compareTo(b.length));
    if (candidates.isEmpty) {
      throw StateError('Unable to detect Qwen3.5 text tensor prefix.');
    }
    return candidates.first;
  }

  static List<double> _previewLastToken(MlxArray array, {int limit = 8}) {
    final last = array
        .slice(start: [0, array.shape[1] - 1, 0], stop: [1, array.shape[1], limit])
        .reshape([limit])
        .astype(MlxDType.MLX_FLOAT32);
    try {
      return List<double>.from(last.toList().cast<double>());
    } finally {
      last.close();
    }
  }

  static void _dumpLastToken(MlxArray array, String path) {
    final width = array.shape[2];
    final last = array
        .slice(start: [0, array.shape[1] - 1, 0], stop: [1, array.shape[1], width])
        .reshape([width])
        .astype(MlxDType.MLX_FLOAT32);
    try {
      File(path).writeAsStringSync(
        jsonEncode(List<double>.from(last.toList().cast<double>())),
      );
    } finally {
      last.close();
    }
  }

  static void _dumpAny(MlxArray array, String path) {
    final flat = array.reshape([array.size]).astype(MlxDType.MLX_FLOAT32);
    try {
      File(path).writeAsStringSync(
        jsonEncode(List<double>.from(flat.toList().cast<double>())),
      );
    } finally {
      flat.close();
    }
  }
}
