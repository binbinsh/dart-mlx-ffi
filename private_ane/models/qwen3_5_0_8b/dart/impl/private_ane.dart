part of 'qwen3_5.dart';

bool _envEnabled(String key, {required bool defaultValue}) {
  final raw = Platform.environment[key];
  if (raw == null || raw.isEmpty) {
    return defaultValue;
  }
  switch (raw.toLowerCase()) {
    case '0':
    case 'false':
    case 'no':
    case 'off':
      return false;
    default:
      return true;
  }
}

void _aneTrace(String message) {
  if (Platform.environment['QWEN35_PRIVATE_ANE_TRACE'] != '1') {
    return;
  }
  stderr.writeln('qwen35_private_ane: $message');
}

const _qwen35OnlineFastLayers = <int>[2, 4, 12, 14, 18];

Set<int>? _parseLayerFilterEnv() {
  final preset = Platform.environment['QWEN35_PRIVATE_ANE_LAYER_PRESET'];
  if (preset == 'online-fast') {
    return _qwen35OnlineFastLayers.toSet();
  }
  final raw = Platform.environment['QWEN35_PRIVATE_ANE_LAYER_FILTER'];
  if (raw == null || raw.trim().isEmpty) {
    return null;
  }
  return raw.split(',').map((value) => int.parse(value.trim())).toSet();
}

final class _AneLayerStats {
  int calls = 0;
  double castMs = 0;
  double copyMs = 0;
  double packMs = 0;
  double aneMs = 0;
  double unpackMs = 0;
  double outputMs = 0;
}

final class _AneProfile {
  _AneProfile()
    : enabled = Platform.environment['QWEN35_PRIVATE_ANE_PROFILE'] == '1';

  final bool enabled;
  final Map<int, _AneLayerStats> _layers = <int, _AneLayerStats>{};

  void record(
    int layer, {
    required double castMs,
    required double copyMs,
    required double packMs,
    required double aneMs,
    required double unpackMs,
    required double outputMs,
  }) {
    if (!enabled) {
      return;
    }
    final stats = _layers.putIfAbsent(layer, _AneLayerStats.new);
    stats.calls++;
    stats.castMs += castMs;
    stats.copyMs += copyMs;
    stats.packMs += packMs;
    stats.aneMs += aneMs;
    stats.unpackMs += unpackMs;
    stats.outputMs += outputMs;
  }

  void dump() {
    if (!enabled || _layers.isEmpty) {
      return;
    }
    final keys = _layers.keys.toList()..sort();
    stderr.writeln(
      'qwen35_private_ane_profile: layer,calls,cast,copy,pack,ane,unpack,output,total_ms',
    );
    for (final layer in keys) {
      final s = _layers[layer]!;
      final total =
          s.castMs + s.copyMs + s.packMs + s.aneMs + s.unpackMs + s.outputMs;
      stderr.writeln(
        'qwen35_private_ane_profile: '
        '$layer,${s.calls},'
        '${s.castMs.toStringAsFixed(3)},'
        '${s.copyMs.toStringAsFixed(3)},'
        '${s.packMs.toStringAsFixed(3)},'
        '${s.aneMs.toStringAsFixed(3)},'
        '${s.unpackMs.toStringAsFixed(3)},'
        '${s.outputMs.toStringAsFixed(3)},'
        '${total.toStringAsFixed(3)}',
      );
    }
  }
}

final class _Qwen35PrivateAneLayer {
  _Qwen35PrivateAneLayer({
    required this.layerIndex,
    required this.dim,
    required this.lane,
    required this.model,
    required this.session,
    required this.useRealtime,
  });

  final int layerIndex;
  final int dim;
  final int lane;
  final MlxAnePrivateModel model;
  final MlxAnePrivateSession session;
  bool useRealtime;

  Float32List runPackedFromArray(MlxArray input, {required int seqLen}) {
    session.writeInputPackedArrayFloat32(
      0,
      input,
      seqLen: seqLen,
      dim: dim,
      lane: lane,
    );
    if (useRealtime) {
      try {
        session.evaluateRealtime();
      } on MlxException {
        useRealtime = false;
        try {
          session.teardownRealtime();
        } on MlxException {
          // Best effort fallback.
        }
        _aneTrace('layer=$layerIndex realtime->standard fallback');
        session.evaluate();
      }
    } else {
      session.evaluate();
    }
    return session.readOutputRawFloat32View(0);
  }

  Float32List runPacked(Float32List packedInput) {
    session.writeInputRawFloat32(0, packedInput);
    if (useRealtime) {
      try {
        session.evaluateRealtime();
      } on MlxException {
        useRealtime = false;
        try {
          session.teardownRealtime();
        } on MlxException {
          // Best effort fallback.
        }
        _aneTrace('layer=$layerIndex realtime->standard fallback');
        session.evaluate();
      }
    } else {
      session.evaluate();
    }
    return session.readOutputRawFloat32View(0);
  }

  void close() {
    session.close();
    model.close();
  }
}

final class _Qwen35PrivateAneMlpRuntime {
  _Qwen35PrivateAneMlpRuntime._(
    this.dim,
    this.hidden,
    this.lane,
    this.layers,
    this._profile,
  );

  factory _Qwen35PrivateAneMlpRuntime.load(
    String artifactDir, {
    required Qwen3_5Config config,
  }) {
    if (!mx.anePrivate.isEnabled()) {
      throw const MlxException('Private ANE runtime is unavailable.');
    }
    final probe = mx.anePrivate.probe();
    if (!probe.frameworkLoaded || !probe.supportsBasicEval) {
      throw const MlxException('Private ANE basic evaluation is unavailable.');
    }
    final metadataPath = File('$artifactDir/metadata.json');
    if (!metadataPath.existsSync()) {
      throw StateError('Missing private ANE metadata: ${metadataPath.path}');
    }
    final metadata = Map<String, Object?>.from(
      jsonDecode(metadataPath.readAsStringSync()) as Map,
    );
    if (metadata['runtime'] != 'qwen35_private_ffn') {
      throw StateError('Unsupported private ANE runtime metadata.');
    }
    final dim = (metadata['dim'] as num).toInt();
    final hidden = (metadata['hidden'] as num).toInt();
    final lane = (metadata['lane'] as num).toInt();
    if (dim != config.hiddenSize) {
      throw StateError(
        'Private ANE dim mismatch: metadata=$dim config=${config.hiddenSize}.',
      );
    }
    if (hidden != config.intermediateSize) {
      throw StateError(
        'Private ANE hidden mismatch: metadata=$hidden config=${config.intermediateSize}.',
      );
    }
    final preferRealtime = _envEnabled(
      'QWEN35_PRIVATE_ANE_REALTIME',
      defaultValue: false,
    );
    final layerFilter = _parseLayerFilterEnv();
    final layerSpecs =
        List<Map<String, Object?>>.from(
          (metadata['layers'] as List).cast<Map>(),
        ).where((spec) {
          if (layerFilter == null) {
            return true;
          }
          return layerFilter.contains((spec['layer'] as num).toInt());
        }).toList();
    if (layerFilter != null) {
      _aneTrace('active_layers=${layerSpecs.map((e) => e['layer']).join(",")}');
    }
    final profile = _AneProfile();
    final layers = <int, _Qwen35PrivateAneLayer>{};
    try {
      for (final spec in layerSpecs) {
        final layerIndex = (spec['layer'] as num).toInt();
        final milText = File(spec['model_mil'] as String).readAsStringSync();
        final weightSpecs = List<Map<String, Object?>>.from(
          (spec['weights'] as List).cast<Map>(),
        );
        final weights = <MlxAneWeightWithOffset>[
          for (final weight in weightSpecs)
            (
              path: weight['path']! as String,
              data: Uint8List.fromList(
                File(weight['file']! as String).readAsBytesSync(),
              ),
              offset: (weight['offset']! as num).toInt(),
            ),
        ];
        MlxAnePrivateModel? model;
        MlxAnePrivateSession? session;
        try {
          model = mx.anePrivate.modelFromMilWithOffsets(
            milText,
            weights: weights,
          );
          model.compile();
          model.load();
          session = model.createSession(
            inputByteSizes: [(spec['input_bytes']! as num).toInt()],
            outputByteSizes: [(spec['output_bytes']! as num).toInt()],
          );
          var useRealtime = false;
          if (preferRealtime) {
            try {
              session.prepareRealtime();
              useRealtime = session.isRealtimeLoaded;
            } on MlxException catch (error) {
              _aneTrace(
                'layer=$layerIndex realtime prepare failed, fallback standard: $error',
              );
            }
          }
          layers[layerIndex] = _Qwen35PrivateAneLayer(
            layerIndex: layerIndex,
            dim: dim,
            lane: lane,
            model: model,
            session: session,
            useRealtime: useRealtime,
          );
          model = null;
          session = null;
        } finally {
          session?.close();
          model?.close();
        }
      }
    } catch (_) {
      for (final layer in layers.values) {
        layer.close();
      }
      rethrow;
    }
    return _Qwen35PrivateAneMlpRuntime._(
      dim,
      hidden,
      lane,
      Map<int, _Qwen35PrivateAneLayer>.unmodifiable(layers),
      profile,
    );
  }

  final int dim;
  final int hidden;
  final int lane;
  final Map<int, _Qwen35PrivateAneLayer> layers;
  final _AneProfile _profile;

  bool hasLayer(int layerIndex) => layers.containsKey(layerIndex);

  MlxArray runLayer(int layerIndex, MlxArray input, int seqLen) {
    final layer = layers[layerIndex];
    if (layer == null) {
      throw StateError('Missing private ANE layer $layerIndex.');
    }
    final castWatch = Stopwatch()..start();
    final inputF32 = input.astype(MlxDType.MLX_FLOAT32);
    castWatch.stop();
    try {
      if (seqLen <= lane) {
        final output = Float32List(seqLen * dim);
        final aneWatch = Stopwatch()..start();
        final packedOut = layer.runPackedFromArray(inputF32, seqLen: seqLen);
        aneWatch.stop();
        final unpackWatch = Stopwatch()..start();
        _unpack(packedOut, output, start: 0, chunkLen: seqLen);
        unpackWatch.stop();
        final outWatch = Stopwatch()..start();
        final result = MlxArray.fromFloat32List(
          output,
          shape: [1, seqLen, dim],
        );
        outWatch.stop();
        _profile.record(
          layerIndex,
          castMs: castWatch.elapsedMicroseconds / 1000.0,
          copyMs: 0,
          packMs: 0,
          aneMs: aneWatch.elapsedMicroseconds / 1000.0,
          unpackMs: unpackWatch.elapsedMicroseconds / 1000.0,
          outputMs: outWatch.elapsedMicroseconds / 1000.0,
        );
        return result;
      }

      final copyWatch = Stopwatch()..start();
      final flat = inputF32.toFloat32List();
      copyWatch.stop();
      final output = Float32List(seqLen * dim);
      var packUs = 0;
      var aneUs = 0;
      var unpackUs = 0;
      for (var start = 0; start < seqLen; start += lane) {
        final chunk = math.min(lane, seqLen - start);
        final packWatch = Stopwatch()..start();
        final packed = _pack(flat, start: start, chunkLen: chunk);
        packWatch.stop();
        packUs += packWatch.elapsedMicroseconds;
        final aneWatch = Stopwatch()..start();
        final packedOut = layer.runPacked(packed);
        aneWatch.stop();
        aneUs += aneWatch.elapsedMicroseconds;
        final unpackWatch = Stopwatch()..start();
        _unpack(packedOut, output, start: start, chunkLen: chunk);
        unpackWatch.stop();
        unpackUs += unpackWatch.elapsedMicroseconds;
      }
      final outWatch = Stopwatch()..start();
      final result = MlxArray.fromFloat32List(output, shape: [1, seqLen, dim]);
      outWatch.stop();
      _profile.record(
        layerIndex,
        castMs: castWatch.elapsedMicroseconds / 1000.0,
        copyMs: copyWatch.elapsedMicroseconds / 1000.0,
        packMs: packUs / 1000.0,
        aneMs: aneUs / 1000.0,
        unpackMs: unpackUs / 1000.0,
        outputMs: outWatch.elapsedMicroseconds / 1000.0,
      );
      return result;
    } finally {
      inputF32.close();
    }
  }

  Float32List _pack(
    Float32List values, {
    required int start,
    required int chunkLen,
  }) {
    final packed = Float32List(dim * lane);
    for (var token = 0; token < chunkLen; token++) {
      final srcBase = (start + token) * dim;
      for (var channel = 0; channel < dim; channel++) {
        packed[channel * lane + token] = values[srcBase + channel];
      }
    }
    return packed;
  }

  void _unpack(
    Float32List packed,
    Float32List dst, {
    required int start,
    required int chunkLen,
  }) {
    for (var token = 0; token < chunkLen; token++) {
      final dstBase = (start + token) * dim;
      for (var channel = 0; channel < dim; channel++) {
        dst[dstBase + channel] = packed[channel * lane + token];
      }
    }
  }

  void close() {
    _profile.dump();
    for (final layer in layers.values) {
      layer.close();
    }
  }
}
