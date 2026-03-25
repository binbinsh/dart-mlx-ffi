part of 'qwen3_5.dart';

Set<int>? _parseAttnLayerFilterEnv() {
  final raw = Platform.environment['QWEN35_PRIVATE_ANE_ATTN_LAYER_FILTER'];
  if (raw == null || raw.trim().isEmpty) {
    return null;
  }
  return raw.split(',').map((value) => int.parse(value.trim())).toSet();
}

final class _Qwen35PrivateAneAttnPostLayer {
  _Qwen35PrivateAneAttnPostLayer({
    required this.layerIndex,
    required this.inputDim,
    required this.outputDim,
    required this.lane,
    required this.model,
    required this.session,
  });

  final int layerIndex;
  final int inputDim;
  final int outputDim;
  final int lane;
  final MlxAnePrivateModel model;
  final MlxAnePrivateSession session;

  MlxArray run(MlxArray ctx, MlxArray gate, int seqLen) {
    final ctxF32 = ctx.astype(MlxDType.MLX_FLOAT32);
    final gateF32 = gate.astype(MlxDType.MLX_FLOAT32);
    try {
      session.writeInputPackedArrayFloat32(
        0,
        ctxF32,
        seqLen: seqLen,
        dim: inputDim,
        lane: lane,
      );
      session.writeInputPackedArrayFloat32(
        1,
        gateF32,
        seqLen: seqLen,
        dim: inputDim,
        lane: lane,
      );
      session.evaluate();
      final packed = session.readOutputRawFloat32View(0);
      final output = Float32List(seqLen * outputDim);
      for (var token = 0; token < seqLen; token++) {
        final dstBase = token * outputDim;
        for (var channel = 0; channel < outputDim; channel++) {
          output[dstBase + channel] = packed[channel * lane + token];
        }
      }
      return MlxArray.fromFloat32List(output, shape: [1, seqLen, outputDim]);
    } finally {
      ctxF32.close();
      gateF32.close();
    }
  }

  void close() {
    session.close();
    model.close();
  }
}

final class _Qwen35PrivateAneAttnPostRuntime {
  _Qwen35PrivateAneAttnPostRuntime._(this.layers);

  static _Qwen35PrivateAneAttnPostRuntime? loadIfPresent(
    Map<String, Object?> metadata,
  ) {
    if (!_envEnabled('QWEN35_PRIVATE_ANE_ATTN_POST', defaultValue: false)) {
      return null;
    }
    final layerFilter = _parseAttnLayerFilterEnv();
    final specs = ((metadata['attn_post_layers'] as List?)?.cast<Map>() ?? const [])
        .where((spec) {
          if (layerFilter == null) {
            return true;
          }
          return layerFilter.contains((spec['layer'] as num).toInt());
        })
        .toList();
    if (specs.isEmpty) {
      return null;
    }
    final layers = <int, _Qwen35PrivateAneAttnPostLayer>{};
    try {
      for (final rawSpec in specs) {
        final spec = Map<String, Object?>.from(rawSpec);
        final layerIndex = (spec['layer'] as num).toInt();
        final model = _loadAttnPostModel(spec);
        model.compile();
        model.load();
        final session = model.createSession(
          inputByteSizes: List<int>.from(
            (spec['input_byte_sizes'] as List).cast<num>(),
          ),
          outputByteSizes: List<int>.from(
            (spec['output_byte_sizes'] as List).cast<num>(),
          ),
        );
        layers[layerIndex] = _Qwen35PrivateAneAttnPostLayer(
          layerIndex: layerIndex,
          inputDim: (spec['input_dim'] as num).toInt(),
          outputDim: (spec['output_dim'] as num).toInt(),
          lane: (spec['lane'] as num).toInt(),
          model: model,
          session: session,
        );
      }
    } catch (_) {
      for (final layer in layers.values) {
        layer.close();
      }
      rethrow;
    }
    _aneTrace('attn_post_layers=${layers.keys.toList()..sort()}');
    return _Qwen35PrivateAneAttnPostRuntime._(
      Map<int, _Qwen35PrivateAneAttnPostLayer>.unmodifiable(layers),
    );
  }

  final Map<int, _Qwen35PrivateAneAttnPostLayer> layers;

  bool hasLayer(int layerIndex) => layers.containsKey(layerIndex);

  MlxArray runLayer(int layerIndex, MlxArray ctx, MlxArray gate, int seqLen) {
    final layer = layers[layerIndex];
    if (layer == null) {
      throw StateError('Missing private ANE attention-post layer $layerIndex.');
    }
    return layer.run(ctx, gate, seqLen);
  }

  void close() {
    for (final layer in layers.values) {
      layer.close();
    }
  }
}

MlxAnePrivateModel _loadAttnPostModel(Map<String, Object?> spec) {
  final milText = File(spec['model_mil']! as String).readAsStringSync();
  final weights = <MlxAneWeightWithOffset>[
    for (final weight in List<Map<String, Object?>>.from(
      (spec['weights'] as List).cast<Map>(),
    ))
      (
        path: weight['path']! as String,
        data: Uint8List.fromList(
          File(weight['file']! as String).readAsBytesSync(),
        ),
        offset: (weight['offset']! as num).toInt(),
      ),
  ];
  return mx.anePrivate.modelFromMilWithOffsets(milText, weights: weights);
}
