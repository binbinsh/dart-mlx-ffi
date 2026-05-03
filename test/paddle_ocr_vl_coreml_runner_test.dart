/// Tests for the PaddleOCR-VL CoreML pipeline runner.
///
/// Most assertions that touch real `.mlpackage` files are gated with
/// `skip:` because Phase 1 (the bundle itself) and Phase 2 (the FFI runtime)
/// are not yet wired in. Once both land, flip the skips off — the assertions
/// already encode the intended behaviour.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/src/models/paddle_ocr_vl/coreml_image.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/coreml_pipeline_manifest.dart';
import 'package:test/test.dart';

void main() {
  group('CoremlPipelineManifest', () {
    test('parses schema v2 fixture with all 4 stages', () {
      final tmp = Directory.systemTemp.createTempSync('pipeline-manifest-');
      addTearDown(() => tmp.deleteSync(recursive: true));
      final path = '${tmp.path}/pipeline.json';
      File(path).writeAsStringSync(jsonEncode(_fixtureManifest));

      final m = CoremlPipelineManifest.loadFile(path);
      expect(m.schema, 2);
      expect(m.modelId, 'paddleocr-vl-1.5-coreml');
      expect(m.stages.map((s) => s.name).toList(), [
        'vision_embed',
        'token_embed',
        'prefill_decoder',
        'decode_decoder',
      ]);
      expect(m.stage('prefill_decoder').stateful, isTrue);
      expect(m.stage('prefill_decoder').stateGroup, 'kv');
      expect(m.stage('decode_decoder').stateGroup, 'kv');
      expect(
        m.stage('vision_embed').computeUnits,
        CoremlComputeUnits.cpuAndNeuralEngine,
      );
      expect(m.kv.layers, 28);
      expect(m.kv.kvHeads, 4);
      expect(m.kv.headDim, 128);
      expect(m.vision.patchSize, 14);
      expect(m.vision.spatialMerge, 2);
      expect(m.vision.buckets, contains((1, 16, 16)));
      expect(m.tokens.imageTokenId, 100295);
      expect(m.tokens.eosTokenId, 2);
      expect(m.prefillBuckets, [128, 256, 384, 512, 768]);
    });

    test('pickPrefillBucket selects smallest bucket >= promptLen', () {
      final tmp = Directory.systemTemp.createTempSync('pipeline-bucket-');
      addTearDown(() => tmp.deleteSync(recursive: true));
      final path = '${tmp.path}/pipeline.json';
      File(path).writeAsStringSync(jsonEncode(_fixtureManifest));
      final m = CoremlPipelineManifest.loadFile(path);

      expect(m.pickPrefillBucket(1), 128);
      expect(m.pickPrefillBucket(128), 128);
      expect(m.pickPrefillBucket(129), 256);
      expect(m.pickPrefillBucket(500), 512);
      expect(m.pickPrefillBucket(768), 768);
      expect(m.pickPrefillBucket(9999), 768); // overflow falls to largest
    });

    test('missing stage throws StateError with helpful message', () {
      final tmp = Directory.systemTemp.createTempSync('pipeline-missing-');
      addTearDown(() => tmp.deleteSync(recursive: true));
      final path = '${tmp.path}/pipeline.json';
      final partial = Map<String, Object?>.from(_fixtureManifest);
      partial['stages'] = [
        (_fixtureManifest['stages']! as List).first,
      ]; // only vision_embed
      File(path).writeAsStringSync(jsonEncode(partial));
      final m = CoremlPipelineManifest.loadFile(path);
      expect(
        () => m.stage('decode_decoder'),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('decode_decoder'),
        )),
      );
    });
  });

  group('pickImageBucket', () {
    final buckets = <(int, int, int)>[
      (1, 16, 16), // 256 patches
      (1, 24, 24), // 576
      (1, 16, 32), // 512
      (1, 32, 16), // 512
      (1, 32, 32), // 1024
    ];

    test('chooses smallest bucket whose area fits', () {
      // Resized 196x196 → 14x14 = 196 patches → smallest fit is 16x16.
      final b = pickImageBucket(
        resizedHeight: 196,
        resizedWidth: 196,
        buckets: buckets,
      );
      expect(b, (1, 16, 16));
    });

    test('aspect ratio breaks ties between equal-area buckets', () {
      // 224x448 → 16x32 patches → exact bucket match prefers 16x32, not 32x16.
      final b = pickImageBucket(
        resizedHeight: 224,
        resizedWidth: 448,
        buckets: buckets,
      );
      expect(b, (1, 16, 32));
    });

    test('falls back to largest bucket when nothing fits', () {
      // 1024x1024 → 73x73 patches = 5329 patches, exceeds all buckets.
      final b = pickImageBucket(
        resizedHeight: 1024,
        resizedWidth: 1024,
        buckets: buckets,
      );
      expect(b, (1, 32, 32));
    });
  });

  group('PaddleOcrVlCoremlRunner — pipeline assembly', () {
    test(
      'load() opens 4 sessions in the order declared by pipeline.json',
      () async {
        // Requires Phase 2 FFI loader. Once available, drop the skip and
        // point bundlePath at a Phase 1 mlpackage bundle on disk.
      },
      skip: 'requires Phase 1 (mlpackages) + Phase 2 (CoreMlRuntime FFI)',
    );

    test('warmup() executes one decode step then resets MLState', () async {},
        skip: 'requires Phase 1 mlpackages');

    test(
      'generate() runs vision → prefill → decode and stops at EOS',
      () async {
        // Synthetic 224x224 white image, prompt with the right number of
        // image-token placeholders for the 16x16 bucket.
        final img = Uint8List(224 * 224 * 3)..fillRange(0, 224 * 224 * 3, 255);
        expect(img.length, 224 * 224 * 3);
      },
      skip: 'requires Phase 1 mlpackages',
    );

    test(
      'generate() rejects prompts whose placeholder count mismatches bucket',
      () async {
        // Negative path is *almost* testable without mlpackages — but
        // CoremlPipelineManifest.loadFile + image preprocess run first, so
        // we'd need a working `vision_embed` session. Defer until Phase 2.
      },
      skip: 'requires Phase 2 fake-session injection (Phase 3 follow-up)',
    );

    test('close() is idempotent', () async {},
        skip: 'requires Phase 2 CoremlSession');
  });
}

/// Minimal but realistic `pipeline.json` v2 — mirrors the layout Phase 1
/// will produce. Patch sizes / bucket lists match ADR §5.1 and §6.
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
    'layers': 28,
    'kv_heads': 4,
    'head_dim': 128,
    'max_len': 4096,
    'dtype': 'fp16',
  },
  'vision': {
    'buckets': [
      [1, 16, 16],
      [1, 24, 24],
      [1, 16, 32],
      [1, 32, 16],
      [1, 32, 32],
    ],
    'patch_size': 14,
    'spatial_merge': 2,
  },
  'tokens': {
    'image_token_id': 100295,
    'eos_token_id': 2,
    'pad_token_id': 0,
  },
  'prefill_buckets': [128, 256, 384, 512, 768],
};
