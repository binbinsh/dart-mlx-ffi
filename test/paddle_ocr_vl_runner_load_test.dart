/// Unit tests for the MLX weight loader at
/// `lib/src/models/paddle_ocr_vl/runner_load.dart`.
///
/// These tests pin the *current* full-model load behaviour: the loader
/// requires BOTH `language_model.*` decoder weights AND `visual.*` vision
/// encoder weights to be present in the snapshot. This is the baseline that
/// the hybrid OCR refactor (issue #1) will diverge from in commit #4 by
/// adding a `keepVisionWeights` flag — once that flag exists, the negative
/// "missing visual weight throws" assertion below will be re-pointed at the
/// flag-on configuration so the flag-off path can be asserted to succeed.
///
/// Strategy: build a tiny synthetic snapshot directory on disk containing
/// `config.json` + a single `weights.safetensors` produced via
/// [MlxIo.saveSafetensors]. Drive the loader through the public
/// [PaddleOcrVlRunner.load] entry point. Assertions are limited to what is
/// observable from the package's public surface (load completes / throws,
/// `runner.config` round-trips), which is sufficient to baseline the weight
/// routing contract — any missing key trips a `StateError` deep inside
/// `_LinearBase.load` or `_firstExistingTensorPrefix`.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/mlx.dart';
import 'package:dart_inference/models.dart';
import 'package:test/test.dart';

void main() {
  group('PaddleOcrVlRunner.load — synthetic snapshot', () {
    test('routes language_model.* and visual.* weights end-to-end', () {
      final snap = _SyntheticSnapshot(_tinyConfig());
      addTearDown(snap.dispose);
      snap.write();

      final runner = PaddleOcrVlRunner.load(snap.path);
      addTearDown(runner.close);

      // Config round-tripped from snapshot reflects the synthetic shape.
      expect(runner.config.numHiddenLayers, _tinyConfig().numHiddenLayers);
      expect(runner.config.hiddenSize, _tinyConfig().hiddenSize);
      expect(runner.config.vocabSize, _tinyConfig().vocabSize);
      expect(runner.config.tieWordEmbeddings, isFalse);
    });

    test('tied lm_head: load succeeds without language_model.lm_head.weight',
        () {
      final cfg = _tinyConfig().copyWith(tieWordEmbeddings: true);
      final snap = _SyntheticSnapshot(cfg, includeLmHead: false);
      addTearDown(snap.dispose);
      snap.write();

      final runner = PaddleOcrVlRunner.load(snap.path);
      addTearDown(runner.close);

      expect(runner.config.tieWordEmbeddings, isTrue);
      expect(runner.config.numHiddenLayers, cfg.numHiddenLayers);
    });

    test(
      'parameterised layer count: numHiddenLayers controls decoder size '
      '(load completes for each value)',
      () {
        for (final n in const [1, 2, 3]) {
          final cfg = _tinyConfig().copyWith(numHiddenLayers: n);
          final snap = _SyntheticSnapshot(cfg);
          addTearDown(snap.dispose);
          snap.write();

          final runner = PaddleOcrVlRunner.load(snap.path);
          addTearDown(runner.close);
          expect(runner.config.numHiddenLayers, n);
        }
      },
    );

    test(
      'baseline: missing a visual.* weight makes load throw '
      '(commit #4 will introduce keepVisionWeights:false to skip these)',
      () {
        final snap = _SyntheticSnapshot(_tinyConfig());
        addTearDown(snap.dispose);
        snap.write(omitTensorKeys: const {'visual.post_layernorm.weight'});

        expect(
          () => PaddleOcrVlRunner.load(snap.path),
          throwsA(isA<Error>()),
        );
      },
    );

    test('baseline: missing a language_model.* layer weight makes load throw',
        () {
      final snap = _SyntheticSnapshot(_tinyConfig());
      addTearDown(snap.dispose);
      snap.write(
        omitTensorKeys: const {
          'language_model.model.layers.0.self_attn.q_proj.weight',
        },
      );

      expect(
        () => PaddleOcrVlRunner.load(snap.path),
        throwsA(isA<Error>()),
      );
    });
  });
}

// ---------------------------------------------------------------------------
// Synthetic config + snapshot writer
// ---------------------------------------------------------------------------

class _SyntheticConfig {
  const _SyntheticConfig({
    required this.hiddenSize,
    required this.intermediateSize,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.vocabSize,
    required this.tieWordEmbeddings,
    required this.visionHiddenSize,
    required this.visionIntermediateSize,
    required this.visionNumHiddenLayers,
    required this.visionNumAttentionHeads,
    required this.visionPatchSize,
    required this.visionImageSize,
    required this.visionSpatialMergeSize,
  });

  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int vocabSize;
  final bool tieWordEmbeddings;

  final int visionHiddenSize;
  final int visionIntermediateSize;
  final int visionNumHiddenLayers;
  final int visionNumAttentionHeads;
  final int visionPatchSize;
  final int visionImageSize;
  final int visionSpatialMergeSize;

  int get qOutDim => numAttentionHeads * headDim;
  int get kvOutDim => numKeyValueHeads * headDim;

  int get visionQkvOutDim => 3 * visionHiddenSize;
  int get visionProjectorInputDim =>
      visionHiddenSize * visionSpatialMergeSize * visionSpatialMergeSize;

  _SyntheticConfig copyWith({
    int? numHiddenLayers,
    bool? tieWordEmbeddings,
  }) =>
      _SyntheticConfig(
        hiddenSize: hiddenSize,
        intermediateSize: intermediateSize,
        numHiddenLayers: numHiddenLayers ?? this.numHiddenLayers,
        numAttentionHeads: numAttentionHeads,
        numKeyValueHeads: numKeyValueHeads,
        headDim: headDim,
        vocabSize: vocabSize,
        tieWordEmbeddings: tieWordEmbeddings ?? this.tieWordEmbeddings,
        visionHiddenSize: visionHiddenSize,
        visionIntermediateSize: visionIntermediateSize,
        visionNumHiddenLayers: visionNumHiddenLayers,
        visionNumAttentionHeads: visionNumAttentionHeads,
        visionPatchSize: visionPatchSize,
        visionImageSize: visionImageSize,
        visionSpatialMergeSize: visionSpatialMergeSize,
      );

  Map<String, Object?> toConfigJson() => {
        'architectures': ['PaddleOCRVLForConditionalGeneration'],
        'head_dim': headDim,
        'hidden_size': hiddenSize,
        'image_token_id': 100295,
        'intermediate_size': intermediateSize,
        'num_attention_heads': numAttentionHeads,
        'num_hidden_layers': numHiddenLayers,
        'num_key_value_heads': numKeyValueHeads,
        'rms_norm_eps': 1e-5,
        'rope_scaling': {
          'mrope_section': [16, 24, 24],
          'rope_type': 'default',
        },
        'rope_theta': 500000,
        'tie_word_embeddings': tieWordEmbeddings,
        'vision_start_token_id': 101305,
        'vision_end_token_id': 101306,
        'eos_token_id': 2,
        'vocab_size': vocabSize,
        'vision_config': {
          'hidden_size': visionHiddenSize,
          'image_size': visionImageSize,
          'intermediate_size': visionIntermediateSize,
          'num_attention_heads': visionNumAttentionHeads,
          'num_channels': 3,
          'num_hidden_layers': visionNumHiddenLayers,
          'patch_size': visionPatchSize,
          'spatial_merge_size': visionSpatialMergeSize,
          'layer_norm_eps': 1e-6,
        },
      };
}

_SyntheticConfig _tinyConfig() => const _SyntheticConfig(
      hiddenSize: 8,
      intermediateSize: 16,
      numHiddenLayers: 2,
      numAttentionHeads: 2,
      numKeyValueHeads: 1,
      headDim: 4,
      vocabSize: 32,
      tieWordEmbeddings: false,
      visionHiddenSize: 8,
      visionIntermediateSize: 16,
      visionNumHiddenLayers: 2,
      visionNumAttentionHeads: 2,
      visionPatchSize: 4,
      visionImageSize: 16,
      visionSpatialMergeSize: 2,
    );

class _SyntheticSnapshot {
  _SyntheticSnapshot(this.config, {this.includeLmHead = true});

  final _SyntheticConfig config;
  final bool includeLmHead;
  final Directory _dir =
      Directory.systemTemp.createTempSync('paddle-ocr-vl-loadtest-');
  bool _disposed = false;

  String get path => _dir.path;

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    try {
      _dir.deleteSync(recursive: true);
    } catch (_) {}
  }

  void write({Set<String> omitTensorKeys = const {}}) {
    File('${_dir.path}/config.json')
        .writeAsStringSync(jsonEncode(config.toConfigJson()));

    final tensors = <String, MlxArray>{};
    try {
      _buildTensors(tensors, omitTensorKeys);
      MlxIo.saveSafetensors('${_dir.path}/weights.safetensors', tensors);
    } finally {
      for (final t in tensors.values) {
        t.close();
      }
    }
  }

  void _buildTensors(Map<String, MlxArray> out, Set<String> omit) {
    void put(String key, List<int> shape) {
      if (omit.contains(key)) return;
      out[key] = MlxArray.zeros(shape, dtype: MlxDType.MLX_FLOAT32);
    }

    // ── Language model decoder ────────────────────────────────────────────
    const lm = 'language_model.model.';
    put('${lm}embed_tokens.weight', [config.vocabSize, config.hiddenSize]);
    put('${lm}norm.weight', [config.hiddenSize]);
    if (includeLmHead && !config.tieWordEmbeddings) {
      put('language_model.lm_head.weight',
          [config.vocabSize, config.hiddenSize]);
    }
    for (var i = 0; i < config.numHiddenLayers; i++) {
      final p = '${lm}layers.$i.';
      put('${p}input_layernorm.weight', [config.hiddenSize]);
      put('${p}post_attention_layernorm.weight', [config.hiddenSize]);
      // Q/K/V have differing first dims (so fusion still concats along axis 0)
      // but matching trailing shape `[hiddenSize]`.
      put('${p}self_attn.q_proj.weight', [config.qOutDim, config.hiddenSize]);
      put('${p}self_attn.k_proj.weight', [config.kvOutDim, config.hiddenSize]);
      put('${p}self_attn.v_proj.weight', [config.kvOutDim, config.hiddenSize]);
      put('${p}self_attn.o_proj.weight', [config.hiddenSize, config.qOutDim]);
      put('${p}mlp.gate_proj.weight',
          [config.intermediateSize, config.hiddenSize]);
      put('${p}mlp.up_proj.weight',
          [config.intermediateSize, config.hiddenSize]);
      put('${p}mlp.down_proj.weight',
          [config.hiddenSize, config.intermediateSize]);
    }

    // ── Vision encoder ────────────────────────────────────────────────────
    const vp = 'visual.';
    // Conv2d kernel: [outChannels, inChannels, pH, pW].
    put('${vp}patch_embedding.weight',
        [config.visionHiddenSize, 3, config.visionPatchSize, config.visionPatchSize]);
    // Position embedding modelled as a linear weight (per loader).
    final numPatches =
        (config.visionImageSize ~/ config.visionPatchSize);
    put('${vp}position_embedding.weight',
        [numPatches * numPatches, config.visionHiddenSize]);

    for (var i = 0; i < config.visionNumHiddenLayers; i++) {
      final bp = '${vp}blocks.$i.';
      put('${bp}layer_norm1.weight', [config.visionHiddenSize]);
      put('${bp}layer_norm1.bias', [config.visionHiddenSize]);
      put('${bp}layer_norm2.weight', [config.visionHiddenSize]);
      put('${bp}layer_norm2.bias', [config.visionHiddenSize]);
      put('${bp}attn.qkv.weight',
          [config.visionQkvOutDim, config.visionHiddenSize]);
      put('${bp}attn.out_proj.weight',
          [config.visionHiddenSize, config.visionHiddenSize]);
      put('${bp}mlp.fc1.weight',
          [config.visionIntermediateSize, config.visionHiddenSize]);
      put('${bp}mlp.fc2.weight',
          [config.visionHiddenSize, config.visionIntermediateSize]);
    }
    put('${vp}post_layernorm.weight', [config.visionHiddenSize]);
    put('${vp}post_layernorm.bias', [config.visionHiddenSize]);

    // Spatial-merge projector (`visual.merger.*`).
    const mp = '${vp}merger.';
    put('${mp}pre_norm.weight', [config.visionProjectorInputDim]);
    put('${mp}pre_norm.bias', [config.visionProjectorInputDim]);
    put('${mp}linear_1.weight',
        [config.hiddenSize, config.visionProjectorInputDim]);
    put('${mp}linear_2.weight', [config.hiddenSize, config.hiddenSize]);
  }
}
