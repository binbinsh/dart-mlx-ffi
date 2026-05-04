/// Integration test for [PaddleOcrVlCoremlRunner] that drives the full
/// vision -> token_embed -> prefill -> decode pipeline through recording
/// mock [CoremlSession]s. No real `.mlpackage`s, no FFI.
///
/// Asserts the post-commit-#6 host-side scatter contract:
///   - `vision_embed` receives `pixel_values` + `image_grid_thw` (no
///     `input_ids`).
///   - `image_embeds` rows from `vision_embed` are scattered host-side
///     (via `paddleOcrVlScatterImageEmbeddingsFloat32`) into the
///     `inputs_embeds` tensor passed to `prefill_decoder`.
///   - `token_embed` runs once for the prompt and once per decode step.
///   - `prefill_decoder` runs once with the fused embeds at the bucket
///     shape.
///   - `decode_decoder` runs once per decoded token (plus the prompt-replay
///     priming pass).
///   - A bucket/placeholder mismatch raises `StateError`.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/src/models/paddle_ocr_vl/coreml_pipeline_manifest.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/coreml_runner.dart';
import 'package:test/test.dart';

void main() {
  group('PaddleOcrVlCoremlRunner — host-side scatter integration', () {
    late Directory tmp;
    late String bundlePath;

    setUp(() {
      tmp = Directory.systemTemp.createTempSync('paddle-ocr-vl-coreml-int-');
      bundlePath = tmp.path;
      File('$bundlePath/pipeline.json')
          .writeAsStringSync(jsonEncode(_fixtureManifest));
    });

    tearDown(() {
      testCoremlLoaderOverride = null;
      tmp.deleteSync(recursive: true);
    });

    test(
      'vision_embed receives pixel_values and image_grid_thw, not input_ids; '
      'image_embeds are scattered host-side into prefill inputs_embeds',
      () async {
        final fake = _FakeLoader(hiddenSize: _hiddenSize, vocab: _vocab);
        testCoremlLoaderOverride = fake;

        final runner = await PaddleOcrVlCoremlRunner.load(bundlePath);
        addTearDown(runner.close);

        // Tell the decode mock how many priming calls to skip before the
        // real decode loop schedule kicks in.
        fake.byStage['decode_decoder']!.promptLen = _promptIds.length;

        final result = await runner.generate(
          imageBytes: _whiteRgb(_imageDim, _imageDim),
          imageHeight: _imageDim,
          imageWidth: _imageDim,
          promptIds: _promptIds,
          maxNewTokens: 8,
        );

        // Decode emits token 42 every step until EOS. Schedule: prefill→42,
        // decode-step-1→42, decode-step-2→EOS. Loop breaks before
        // appending EOS, so generated = [42 (from prefill), 42 (step 1)].
        expect(result, [42, 42]);

        // ── 1. vision_embed call surface ──────────────────────────────────
        final vision = fake.byStage['vision_embed']!;
        expect(vision.calls, hasLength(1));
        final vinputs = vision.calls.single;
        expect(
          vinputs.keys,
          unorderedEquals(<String>['pixel_values', 'image_grid_thw']),
          reason: 'vision_embed must NOT receive input_ids post-commit-#5',
        );
        expect(vinputs.containsKey('input_ids'), isFalse);

        // image_grid_thw matches the chosen bucket (1, 24, 24) for a 336x336
        // image with patch=14 / merge=2.
        final gridRecord = vinputs['image_grid_thw']! as (List<int>, Int32List);
        expect(gridRecord.$1, [3]);
        expect(gridRecord.$2.toList(), [1, 24, 24]);

        // ── 2. host-side scatter into prefill inputs_embeds ───────────────
        final prefill = fake.byStage['prefill_decoder']!;
        expect(prefill.calls, hasLength(1));
        final pinputs = prefill.calls.single;
        final embedsRecord =
            pinputs['inputs_embeds']! as (List<int>, Float32List);
        // Bucket = 256 (next prefill bucket above promptLen=152).
        expect(embedsRecord.$1, [1, _prefillBucket, _hiddenSize]);
        final embeds = embedsRecord.$2;

        // For every prompt position:
        //  - if it's an image-token placeholder, the row equals the
        //    corresponding image_embeds row (sentinel = 100 + imageIdx).
        //  - otherwise, the row equals the token_embed row sentinel
        //    (token_embed row i = i * 0.01).
        var imageRowIdx = 0;
        for (var i = 0; i < _promptIds.length; i++) {
          final base = i * _hiddenSize;
          final firstFloat = embeds[base];
          if (_promptIds[i] == _imageTokenId) {
            final expected = (100 + imageRowIdx).toDouble();
            expect(
              firstFloat,
              expected,
              reason: 'placeholder slot $i should hold image_embeds row '
                  '$imageRowIdx (= $expected)',
            );
            // spot check whole row
            for (var k = 0; k < _hiddenSize; k++) {
              expect(embeds[base + k], expected);
            }
            imageRowIdx++;
          } else {
            final expected = i * 0.01;
            expect(
              firstFloat,
              closeTo(expected, 1e-6),
              reason: 'non-image slot $i should hold text_embeds sentinel '
                  '($expected)',
            );
          }
        }
        expect(imageRowIdx, _mergedTokens);

        // Padding rows beyond promptLen should be zero.
        final padBase = _promptIds.length * _hiddenSize;
        expect(embeds[padBase], 0.0);
        expect(embeds[embeds.length - 1], 0.0);

        // ── 3. token_embed called for prompt + per decode step ───────────
        final tokenEmbed = fake.byStage['token_embed']!;
        // 1 (prompt) + 2 (one per decode-loop iteration; the loop runs
        // until decode_decoder emits EOS, which is at step 2 — token_embed
        // is invoked at the *top* of each loop iteration before
        // decode_decoder, so it fires for both steps).
        expect(tokenEmbed.calls, hasLength(3));
        // First call: full prompt of length 152.
        final firstCall =
            tokenEmbed.calls.first['input_id']! as (List<int>, Int32List);
        expect(firstCall.$1, [1, _promptIds.length]);
        expect(firstCall.$2.length, _promptIds.length);
        // Subsequent calls: single-token shape [1, 1].
        for (final c in tokenEmbed.calls.skip(1)) {
          final rec = c['input_id']! as (List<int>, Int32List);
          expect(rec.$1, [1, 1]);
          expect(rec.$2.length, 1);
        }

        // ── 4. decode_decoder cadence ────────────────────────────────────
        // primeDecodeState replays the prompt (152 calls) then the decode
        // loop runs until EOS (2 calls). Total = 154.
        final decode = fake.byStage['decode_decoder']!;
        expect(decode.calls, hasLength(_promptIds.length + 2));
        // Last 2 calls are the actual decode steps; their input shape is
        // [1, 1, hidden].
        for (final c in decode.calls.skip(_promptIds.length)) {
          final rec = c['token_embed']! as (List<int>, Float32List);
          expect(rec.$1, [1, 1, _hiddenSize]);
        }
      },
    );

    test(
      'placeholder count mismatch with chosen bucket throws StateError',
      () async {
        final fake = _FakeLoader(hiddenSize: _hiddenSize, vocab: _vocab);
        testCoremlLoaderOverride = fake;

        final runner = await PaddleOcrVlCoremlRunner.load(bundlePath);
        addTearDown(runner.close);

        // Bucket (1, 24, 24) requires 144 image-token placeholders. Pass
        // only 10 → expect StateError naming the mismatch.
        final badPrompt = <int>[
          1, 1, 1, 1, 1,
          ...List<int>.filled(10, _imageTokenId),
          9, 9, 9,
        ];
        expect(
          () => runner.generate(
            imageBytes: _whiteRgb(_imageDim, _imageDim),
            imageHeight: _imageDim,
            imageWidth: _imageDim,
            promptIds: badPrompt,
            maxNewTokens: 4,
          ),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              allOf(contains('placeholder'), contains('10')),
            ),
          ),
        );

        // vision_embed must NOT have been called: the placeholder-count
        // guard fires after preprocessing but before vision_embed.predict.
        expect(fake.byStage['vision_embed']!.calls, isEmpty);
        expect(fake.byStage['prefill_decoder']!.calls, isEmpty);
      },
    );
  });
}

// ──────────────────────────────────────────────────────────────────────────
// Fixtures
// ──────────────────────────────────────────────────────────────────────────

const int _imageTokenId = 100295;
const int _eosTokenId = 2;
const int _hiddenSize = 8;
const int _vocab = 100;
const int _imageDim = 336; // → smartResize keeps 336x336 → bucket (1,24,24)
const int _mergedTokens = 12 * 12; // (24/2)^2 = 144
const int _prefillBucket = 256;

/// Prompt: 5 prefix tokens, 144 image placeholders, 3 trailing → length 152.
final List<int> _promptIds = <int>[
  10, 11, 12, 13, 14,
  ...List<int>.filled(_mergedTokens, _imageTokenId),
  20, 21, 22,
];

/// White HxWx3 RGB image.
Uint8List _whiteRgb(int h, int w) {
  final px = Uint8List(h * w * 3);
  for (var i = 0; i < px.length; i++) {
    px[i] = 255;
  }
  return px;
}

/// Inline pipeline.json fixture — mirrors the one in
/// `paddle_ocr_vl_coreml_runner_test.dart` but with vocab/hidden tuned for
/// the integration test.
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
    'image_token_id': _imageTokenId,
    'eos_token_id': _eosTokenId,
    'pad_token_id': 0,
  },
  'prefill_buckets': [128, 256, 384, 512, 768],
};

// ──────────────────────────────────────────────────────────────────────────
// Recording fakes
// ──────────────────────────────────────────────────────────────────────────

/// CoremlLoader that returns a [_RecordingSession] keyed by stage name.
final class _FakeLoader implements CoremlLoader {
  _FakeLoader({required this.hiddenSize, required this.vocab});

  final int hiddenSize;
  final int vocab;
  final Map<String, _RecordingSession> byStage = {};

  @override
  CoremlSession loadStage({
    required String packagePath,
    required CoremlComputeUnits computeUnits,
    required bool stateful,
  }) {
    final stage = _stageNameFromPath(packagePath);
    final session = _RecordingSession(
      stage: stage,
      hiddenSize: hiddenSize,
      vocab: vocab,
    );
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

/// Records every `predict` call's input map and returns a canned tensor
/// matching the runner's expectations for the stage.
final class _RecordingSession implements CoremlSession {
  _RecordingSession({
    required this.stage,
    required this.hiddenSize,
    required this.vocab,
  });

  final String stage;
  final int hiddenSize;
  final int vocab;
  final List<Map<String, Object>> calls = [];
  int _decodeCallsSeen = 0;
  // For prefill we want argmax to land on a chosen token.
  int prefillNextToken = 42;
  // Decode steps schedule: token 42, then EOS.
  final List<int> _decodeSchedule = const [42, _eosTokenId];

  @override
  Map<String, Object> predict(Map<String, Object> inputs) {
    // Record an immutable snapshot so later mutation can't corrupt it.
    calls.add(Map<String, Object>.unmodifiable(inputs));
    switch (stage) {
      case 'vision_embed':
        return _visionOutput(inputs);
      case 'token_embed':
        return _tokenEmbedOutput(inputs);
      case 'prefill_decoder':
        return _prefillOutput();
      case 'decode_decoder':
        return _decodeOutput();
      default:
        throw StateError('unknown stage $stage');
    }
  }

  Map<String, Object> _visionOutput(Map<String, Object> inputs) {
    final grid = inputs['image_grid_thw']! as (List<int>, Int32List);
    final t = grid.$2[0];
    final h = grid.$2[1];
    final w = grid.$2[2];
    // Spatial merge = 2 (we know this from the fixture); assert via shape.
    final merged = t * (h ~/ 2) * (w ~/ 2);
    final out = Float32List(merged * hiddenSize);
    // Row i is filled with sentinel (100 + i) so the scatter assertion can
    // identify which row landed where.
    for (var i = 0; i < merged; i++) {
      final v = (100 + i).toDouble();
      for (var k = 0; k < hiddenSize; k++) {
        out[i * hiddenSize + k] = v;
      }
    }
    return {
      'image_embeds': (<int>[merged, hiddenSize], out),
    };
  }

  Map<String, Object> _tokenEmbedOutput(Map<String, Object> inputs) {
    final ids = inputs['input_id']! as (List<int>, Int32List);
    final n = ids.$2.length;
    final out = Float32List(n * hiddenSize);
    // Row i sentinel = i * 0.01 (kept distinct from image sentinel range).
    for (var i = 0; i < n; i++) {
      final v = i * 0.01;
      for (var k = 0; k < hiddenSize; k++) {
        out[i * hiddenSize + k] = v;
      }
    }
    return {
      'token_embed': (<int>[1, n, hiddenSize], out),
    };
  }

  Map<String, Object> _prefillOutput() {
    final logits = Float32List(vocab);
    logits[prefillNextToken] = 10.0;
    return {
      'last_logits': (<int>[1, vocab], logits),
    };
  }

  /// Length of the prompt replay (= promptIds.length). Set by the test;
  /// the first [promptLen] decode predicts are the priming pass and emit
  /// dummy logits, while subsequent calls follow [_decodeSchedule].
  int promptLen = 0;

  Map<String, Object> _decodeOutput() {
    _decodeCallsSeen++;
    final logits = Float32List(vocab);
    if (_decodeCallsSeen <= promptLen) {
      // Prime pass — logits are unused by the runner.
      logits[0] = 1.0;
    } else {
      final idx = _decodeCallsSeen - promptLen - 1;
      final tok = idx < _decodeSchedule.length
          ? _decodeSchedule[idx]
          : _eosTokenId;
      logits[tok] = 10.0;
    }
    return {
      'logits': (<int>[1, vocab], logits),
    };
  }

  @override
  void resetState() {}

  @override
  void close() {}
}
