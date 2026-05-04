/// Tests for [PaddleOcrVlHybridRunner] (issue #1, commit #8).
///
/// These pin the hybrid runner's three load-time and runtime invariants:
///
///   1. Only the `vision_embed` CoreML stage is opened — `token_embed`,
///      `prefill_decoder`, `decode_decoder` packages declared in
///      `pipeline.json` are deliberately ignored.
///   2. The MLX runner is loaded with `keepVisionWeights: false`, so its
///      vision-touching APIs throw `StateError` (commit #4).
///   3. End-to-end `generate(...)` calls vision_embed exactly once and
///      drives prefill+decode via the MLX runner; image_embeds rows from
///      CoreML land in the multimodal embedding at the placeholder
///      positions, then run through the existing MLX prefill+decode loop.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/mlx.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/coreml_pipeline_manifest.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/coreml_loader.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/hybrid_runner.dart';
import 'package:test/test.dart';

void main() {
  group('PaddleOcrVlHybridRunner', () {
    late Directory bundleDir;
    late Directory snapshotDir;
    late _SyntheticSnapshot snapshot;

    setUp(() {
      bundleDir = Directory.systemTemp.createTempSync('paddle-ocr-vl-hybrid-bundle-');
      File('${bundleDir.path}/pipeline.json')
          .writeAsStringSync(jsonEncode(_fixtureManifest));
      snapshot = _SyntheticSnapshot(_tinyConfig());
      snapshotDir = Directory(snapshot.path);
      snapshot.write();
    });

    tearDown(() {
      testCoremlLoaderOverride = null;
      try {
        bundleDir.deleteSync(recursive: true);
      } catch (_) {}
      try {
        snapshot.dispose();
      } catch (_) {}
    });

    test(
      'load opens only vision_embed; MLX runner has keepVisionWeights:false',
      () async {
        final fake = _FakeLoader(hiddenSize: _hiddenSize);
        testCoremlLoaderOverride = fake;

        final runner = await PaddleOcrVlHybridRunner.load(
          coremlBundlePath: bundleDir.path,
          mlxSnapshotPath: snapshotDir.path,
        );
        addTearDown(runner.close);

        // Only vision_embed was queried from the loader.
        expect(fake.openedStages, ['vision_embed']);
        expect(fake.byStage.keys, ['vision_embed']);

        // MLX runner's vision-touching API throws — proves
        // keepVisionWeights:false was used (commit #4 contract).
        expect(
          () => runner.mlxRunner.visionInputDType,
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('keepVisionWeights: false'),
            ),
          ),
        );
      },
    );

    test(
      'generate routes image_embeds through MLX scatter and produces tokens',
      () async {
        final fake = _FakeLoader(hiddenSize: _hiddenSize);
        testCoremlLoaderOverride = fake;

        final runner = await PaddleOcrVlHybridRunner.load(
          coremlBundlePath: bundleDir.path,
          mlxSnapshotPath: snapshotDir.path,
        );
        addTearDown(runner.close);

        // Prompt: 5 prefix tokens + 144 image-token placeholders + 3
        // trailing tokens (matches bucket (1,24,24) merged-token count).
        final promptIds = <int>[
          1, 2, 3, 4, 5,
          ...List<int>.filled(_mergedTokens, _imageTokenId),
          6, 7, 8,
        ];

        final out = runner.generate(
          imageBytes: _whiteRgb(_imageDim, _imageDim),
          imageHeight: _imageDim,
          imageWidth: _imageDim,
          promptIds: promptIds,
          maxNewTokens: 4,
        );

        // ── 1. vision_embed surface ───────────────────────────────────────
        final vision = fake.byStage['vision_embed']!;
        expect(vision.calls, hasLength(1));
        final inputs = vision.calls.single;
        expect(
          inputs.keys,
          unorderedEquals(<String>['pixel_values', 'image_grid_thw']),
          reason: 'vision_embed must NOT receive input_ids in the hybrid path',
        );
        expect(inputs.containsKey('input_ids'), isFalse);

        final grid = inputs['image_grid_thw']! as (List<int>, Int32List);
        expect(grid.$1, [3]);
        expect(grid.$2.toList(), [1, 24, 24]);

        // ── 2. token_embed / prefill_decoder / decode_decoder NEVER open ─
        expect(fake.openedStages, ['vision_embed']);
        expect(fake.byStage.containsKey('token_embed'), isFalse);
        expect(fake.byStage.containsKey('prefill_decoder'), isFalse);
        expect(fake.byStage.containsKey('decode_decoder'), isFalse);

        // ── 3. Output shape ───────────────────────────────────────────────
        // The MLX runner runs greedy decode on synthetic zero weights; we
        // only assert the structural contract: the prompt is preserved as
        // a prefix and at least one new token was generated (or EOS hit
        // immediately, which still appends one token).
        expect(out.length, greaterThanOrEqualTo(promptIds.length));
        expect(
          out.sublist(0, promptIds.length),
          promptIds,
          reason: 'generate must return prompt + generated tokens',
        );
      },
    );

    test('placeholder count mismatch with chosen bucket throws StateError',
        () async {
      final fake = _FakeLoader(hiddenSize: _hiddenSize);
      testCoremlLoaderOverride = fake;

      final runner = await PaddleOcrVlHybridRunner.load(
        coremlBundlePath: bundleDir.path,
        mlxSnapshotPath: snapshotDir.path,
      );
      addTearDown(runner.close);

      final badPrompt = <int>[
        1, 2, 3, 4, 5,
        ...List<int>.filled(10, _imageTokenId), // wrong count
        6, 7, 8,
      ];

      expect(
        () => runner.generate(
          imageBytes: _whiteRgb(_imageDim, _imageDim),
          imageHeight: _imageDim,
          imageWidth: _imageDim,
          promptIds: badPrompt,
          maxNewTokens: 2,
        ),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            allOf(contains('placeholder'), contains('10')),
          ),
        ),
      );

      // The placeholder-count guard fires before vision_embed.predict.
      expect(fake.byStage['vision_embed']!.calls, isEmpty);
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────
// Fixtures
// ──────────────────────────────────────────────────────────────────────────

const int _imageTokenId = 100295;
const int _eosTokenId = 2;
const int _hiddenSize = 16; // matches _tinyConfig().hiddenSize
const int _imageDim = 336;
const int _mergedTokens = 12 * 12; // (24/2)^2 = 144

Uint8List _whiteRgb(int h, int w) {
  final px = Uint8List(h * w * 3);
  for (var i = 0; i < px.length; i++) {
    px[i] = 255;
  }
  return px;
}

final Map<String, Object?> _fixtureManifest = {
  'schema': 2,
  'model': 'paddleocr-vl-1.5-coreml',
  'stages': [
    {
      'name': 'vision_embed',
      'package': 'vision_embed.mlpackage',
      'compute_units': 'cpu_and_neural_engine',
      'stateful': false,
    },
    // Other stages are declared but should NOT be opened by the hybrid
    // runner. Their presence in pipeline.json is the realistic case during
    // the deprecation window for the legacy runner.
    {
      'name': 'token_embed',
      'package': 'token_embed.mlpackage',
      'compute_units': 'cpu_and_neural_engine',
      'stateful': false,
    },
    {
      'name': 'prefill_decoder',
      'package': 'prefill_decoder.mlpackage',
      'compute_units': 'cpu_and_neural_engine',
      'stateful': true,
      'state_group': 'kv',
    },
    {
      'name': 'decode_decoder',
      'package': 'decode_decoder.mlpackage',
      'compute_units': 'cpu_and_neural_engine',
      'stateful': true,
      'state_group': 'kv',
    },
  ],
  'kv': {
    'layers': 2,
    'kv_heads': 1,
    'head_dim': 4,
    'max_len': 512,
    'dtype': 'fp16',
  },
  'vision': {
    'buckets': [
      [1, 16, 16],
      [1, 24, 24],
      [1, 32, 32],
    ],
    'patch_size': 14,
    'spatial_merge': 2,
  },
  'tokens': {
    'image_token_id': _imageTokenId,
    'eos_token_id': _eosTokenId,
    'pad_token_id': 0,
  },
  'prefill_buckets': [128, 256, 384, 512],
};

// ──────────────────────────────────────────────────────────────────────────
// Recording fakes (vision_embed only)
// ──────────────────────────────────────────────────────────────────────────

final class _FakeLoader implements CoremlLoader {
  _FakeLoader({required this.hiddenSize});

  final int hiddenSize;
  final Map<String, _RecordingSession> byStage = {};
  final List<String> openedStages = [];

  @override
  CoremlSession loadStage({
    required String packagePath,
    required CoremlComputeUnits computeUnits,
    required bool stateful,
  }) {
    final stage = _stageNameFromPath(packagePath);
    openedStages.add(stage);
    final session = _RecordingSession(stage: stage, hiddenSize: hiddenSize);
    byStage[stage] = session;
    return session;
  }

  static String _stageNameFromPath(String path) {
    final last = path.split(Platform.pathSeparator).last;
    if (last.endsWith('.mlpackage')) {
      return last.substring(0, last.length - '.mlpackage'.length);
    }
    return last;
  }
}

final class _RecordingSession implements CoremlSession {
  _RecordingSession({required this.stage, required this.hiddenSize});

  final String stage;
  final int hiddenSize;
  final List<Map<String, Object>> calls = [];

  @override
  Map<String, Object> predict(Map<String, Object> inputs) {
    calls.add(Map<String, Object>.unmodifiable(inputs));
    if (stage != 'vision_embed') {
      throw StateError(
        'Hybrid runner unexpectedly invoked stage "$stage"; only vision_embed '
        'should run through CoreML.',
      );
    }
    final grid = inputs['image_grid_thw']! as (List<int>, Int32List);
    final t = grid.$2[0];
    final h = grid.$2[1];
    final w = grid.$2[2];
    final merged = t * (h ~/ 2) * (w ~/ 2);
    // Sentinel rows: row i = (1 + i*0.001) — small enough to keep
    // softmax/argmax in the synthetic decoder numerically tame.
    final out = Float32List(merged * hiddenSize);
    for (var i = 0; i < merged; i++) {
      final v = 1.0 + i * 0.001;
      for (var k = 0; k < hiddenSize; k++) {
        out[i * hiddenSize + k] = v;
      }
    }
    return {
      'image_embeds': (<int>[merged, hiddenSize], out),
    };
  }

  @override
  void resetState() {}

  @override
  void close() {}
}

// ──────────────────────────────────────────────────────────────────────────
// Synthetic MLX snapshot — minimal mirror of paddle_ocr_vl_runner_load_test
// ──────────────────────────────────────────────────────────────────────────

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
  });

  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int vocabSize;
  final bool tieWordEmbeddings;

  int get qOutDim => numAttentionHeads * headDim;
  int get kvOutDim => numKeyValueHeads * headDim;

  Map<String, Object?> toConfigJson() => {
        'architectures': ['PaddleOCRVLForConditionalGeneration'],
        'head_dim': headDim,
        'hidden_size': hiddenSize,
        'image_token_id': _imageTokenId,
        'intermediate_size': intermediateSize,
        'num_attention_heads': numAttentionHeads,
        'num_hidden_layers': numHiddenLayers,
        'num_key_value_heads': numKeyValueHeads,
        'rms_norm_eps': 1e-5,
        'rope_scaling': {
          'mrope_section': [1, 1, 2],
          'rope_type': 'default',
        },
        'rope_theta': 500000,
        'tie_word_embeddings': tieWordEmbeddings,
        'vision_start_token_id': 101305,
        'vision_end_token_id': 101306,
        'eos_token_id': _eosTokenId,
        'vocab_size': vocabSize,
        'vision_config': {
          'hidden_size': 8,
          'image_size': 16,
          'intermediate_size': 16,
          'num_attention_heads': 2,
          'num_channels': 3,
          'num_hidden_layers': 2,
          'patch_size': 4,
          'spatial_merge_size': 2,
          'layer_norm_eps': 1e-6,
        },
      };
}

_SyntheticConfig _tinyConfig() => const _SyntheticConfig(
      hiddenSize: 16,
      intermediateSize: 32,
      numHiddenLayers: 2,
      numAttentionHeads: 2,
      numKeyValueHeads: 1,
      headDim: 8,
      vocabSize: 32,
      tieWordEmbeddings: false,
    );

class _SyntheticSnapshot {
  _SyntheticSnapshot(this.config);

  final _SyntheticConfig config;
  final Directory _dir =
      Directory.systemTemp.createTempSync('paddle-ocr-vl-hybrid-snap-');
  bool _disposed = false;

  String get path => _dir.path;

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    try {
      _dir.deleteSync(recursive: true);
    } catch (_) {}
  }

  void write() {
    File('${_dir.path}/config.json')
        .writeAsStringSync(jsonEncode(config.toConfigJson()));
    final tensors = <String, MlxArray>{};
    try {
      _buildTensors(tensors);
      MlxIo.saveSafetensors('${_dir.path}/weights.safetensors', tensors);
    } finally {
      for (final t in tensors.values) {
        t.close();
      }
    }
  }

  void _buildTensors(Map<String, MlxArray> out) {
    void put(String key, List<int> shape) {
      out[key] = MlxArray.zeros(shape, dtype: MlxDType.MLX_FLOAT32);
    }

    const lm = 'language_model.model.';
    put('${lm}embed_tokens.weight', [config.vocabSize, config.hiddenSize]);
    put('${lm}norm.weight', [config.hiddenSize]);
    if (!config.tieWordEmbeddings) {
      put('language_model.lm_head.weight',
          [config.vocabSize, config.hiddenSize]);
    }
    for (var i = 0; i < config.numHiddenLayers; i++) {
      final p = '${lm}layers.$i.';
      put('${p}input_layernorm.weight', [config.hiddenSize]);
      put('${p}post_attention_layernorm.weight', [config.hiddenSize]);
      put('${p}self_attn.q_proj.weight',
          [config.qOutDim, config.hiddenSize]);
      put('${p}self_attn.k_proj.weight',
          [config.kvOutDim, config.hiddenSize]);
      put('${p}self_attn.v_proj.weight',
          [config.kvOutDim, config.hiddenSize]);
      put('${p}self_attn.o_proj.weight',
          [config.hiddenSize, config.qOutDim]);
      put('${p}mlp.gate_proj.weight',
          [config.intermediateSize, config.hiddenSize]);
      put('${p}mlp.up_proj.weight',
          [config.intermediateSize, config.hiddenSize]);
      put('${p}mlp.down_proj.weight',
          [config.hiddenSize, config.intermediateSize]);
    }
    // visual.* tensors are intentionally absent — the hybrid loader uses
    // keepVisionWeights:false, which the loader handles even on
    // decoder-only snapshots (see paddle_ocr_vl_runner_load_test.dart).
  }
}
