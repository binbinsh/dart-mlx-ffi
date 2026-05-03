/// PaddleOCR-VL-1.5 CoreML pipeline runner (Phase 3 of the re-architecture).
///
/// Drives the 4-mlpackage bundle produced by Phase 1 — `vision_embed`,
/// `token_embed`, `prefill_decoder`, `decode_decoder` — through the stateful
/// MLState backend exposed by Phase 2 (`CoreMlRuntime.resetState`).
///
/// **Pipeline cadence per `generate()` call:**
///
/// 1. `vision_embed` runs **once** to fuse pixel patches + token embeds.
/// 2. `prefill_decoder` runs **once** to seed the KV cache.
/// 3. `decode_decoder` runs **N times** (N ≤ `maxNewTokens`, capped by EOS)
///    sharing the same `MLState` with `prefill_decoder`.
///
/// See:
///   - `docs/adr/0001-paddleocr-vl-coreml-rearchitecture.md` §1, §2, §7, §8
///   - `docs/plans/paddleocr-vl-coreml-rebuild.md` Phase 3
///   - `coreml_mrope.dart` for the mRoPE algorithm
///   - `coreml_image.dart` for `smart_resize` and patchify
///   - `coreml_pipeline_manifest.dart` for `pipeline.json` parsing
library;

import 'dart:typed_data';

import 'coreml_image.dart';
import 'coreml_mrope.dart';
import 'coreml_pipeline_manifest.dart';

// -----------------------------------------------------------------------
// Phase 2 FFI surface (documented Dart API).
//
// These are the symbols we depend on from
// `lib/src/native/runtime/coreml_runtime.dart` once Phase 2 lands:
//
//   class CoreMlRuntime {
//     static CoreMlSession loadStage({
//       required String packagePath,
//       required CoremlComputeUnits computeUnits,
//       required bool stateful,
//     });
//     static void resetState(CoreMlSession session);
//     static void releaseSession(CoreMlSession session);
//   }
//   class CoreMlSession {
//     Map<String, Object> predict(Map<String, Object> inputs);
//     // For state_group: "kv" sharing, both prefill+decode point at the
//     // same underlying MLState — handled inside the native session.
//   }
//
// To keep this file compilable in isolation we declare a thin facade type
// here. When Phase 2 lands, the import will switch to the real binding and
// the facade will be removed.
// -----------------------------------------------------------------------

/// Facade for the Phase 2 CoreML session handle. Replace with the real
/// import (`package:dart_inference/src/native/runtime/coreml_runtime.dart`)
/// once Phase 2 lands.
abstract interface class CoremlSession {
  /// Run one inference. Inputs are name → tensor; tensors are
  /// `Float32List` / `Int32List` / `(shape, data)` records — the exact
  /// envelope is defined by Phase 2.
  Map<String, Object> predict(Map<String, Object> inputs);

  /// Release the underlying MLModel + MLState.
  void close();

  /// Drop the cached MLState so the next `predict` materialises a fresh
  /// one. No-op for non-stateful sessions.
  void resetState();
}

/// Phase 2 loader facade. The real implementation will live in
/// `lib/src/native/runtime/coreml_runtime.dart` as `CoreMlRuntime`.
abstract class CoremlLoader {
  CoremlSession loadStage({
    required String packagePath,
    required CoremlComputeUnits computeUnits,
    required bool stateful,
  });
}

/// Hook so the runner can be unit-tested with a fake loader. Defaults to
/// the real native loader once Phase 2 publishes one.
CoremlLoader? testCoremlLoaderOverride;

/// Default tensor-naming scheme between Dart and the mlpackages, matching
/// ADR §2. Centralised so Phase 1 and Phase 3 stay in sync.
abstract final class _CoremlIo {
  static const visionPixels = 'pixel_values';
  static const visionInputIds = 'input_ids';
  static const visionGridThw = 'image_grid_thw';
  static const visionInputsEmbeds = 'inputs_embeds';

  static const tokenEmbedInputId = 'input_id';
  static const tokenEmbedOut = 'token_embed';

  static const prefillEmbeds = 'inputs_embeds';
  static const prefillPositionIds = 'position_ids';
  static const prefillCausalMask = 'causal_mask';
  static const prefillLogits = 'last_logits';

  static const decodeTokenEmbed = 'token_embed';
  static const decodePositionIds = 'position_ids';
  static const decodePastKvLen = 'past_kv_len';
  static const decodeLogits = 'logits';
}

// -----------------------------------------------------------------------
// PaddleOcrVlCoremlRunner
// -----------------------------------------------------------------------

/// New CoreML runner replacing the stateless 4-stage stub. See ADR §7 for
/// the public contract this is implementing.
final class PaddleOcrVlCoremlRunner {
  PaddleOcrVlCoremlRunner._({
    required this.bundlePath,
    required this.manifest,
    required CoremlSession visionEmbed,
    required CoremlSession tokenEmbed,
    required CoremlSession prefill,
    required CoremlSession decode,
  })  : _visionEmbed = visionEmbed,
        _tokenEmbed = tokenEmbed,
        _prefill = prefill,
        _decode = decode;

  final String bundlePath;
  final CoremlPipelineManifest manifest;

  final CoremlSession _visionEmbed;
  final CoremlSession _tokenEmbed;
  final CoremlSession _prefill;
  final CoremlSession _decode;

  bool _closed = false;

  /// Load a Phase 1 bundle directory and instantiate all 4 sessions.
  static Future<PaddleOcrVlCoremlRunner> load(String bundlePath) async {
    final manifest = CoremlPipelineManifest.loadFile('$bundlePath/pipeline.json');
    final loader = testCoremlLoaderOverride ?? _defaultLoader();

    CoremlSession open(String name) {
      final stage = manifest.stage(name);
      return loader.loadStage(
        packagePath: '$bundlePath/${stage.package}',
        computeUnits: stage.computeUnits,
        stateful: stage.stateful,
      );
    }

    return PaddleOcrVlCoremlRunner._(
      bundlePath: bundlePath,
      manifest: manifest,
      visionEmbed: open('vision_embed'),
      tokenEmbed: open('token_embed'),
      prefill: open('prefill_decoder'),
      decode: open('decode_decoder'),
    );
  }

  /// One dummy decode step to JIT-compile ANE kernels and materialise the
  /// MLState. Should be called once after `load()` so the first real
  /// `generate()` doesn't pay the warmup cost.
  void warmup() {
    _ensureOpen();
    // Run a single decode step with a zeroed token embedding and position 0.
    // We deliberately reset state right after so the first `generate()`
    // doesn't see the dummy KV.
    final hidden = manifest.kv.headDim * manifest.kv.kvHeads * 4; // upper bound
    final dummyEmbed = Float32List(hidden);
    final dummyPos = Float32List.fromList([0, 0, 0]);
    final dummyPastLen = Int32List.fromList([0]);
    try {
      _decode.predict({
        _CoremlIo.decodeTokenEmbed: ([1, 1, hidden], dummyEmbed),
        _CoremlIo.decodePositionIds: ([3, 1, 1], dummyPos),
        _CoremlIo.decodePastKvLen: ([1], dummyPastLen),
      });
    } finally {
      _decode.resetState();
      _prefill.resetState();
    }
  }

  /// Run the full vision → prefill → decode pipeline.
  ///
  /// [imageBytes] is HWC RGB `Uint8List` of length `imageHeight*imageWidth*3`.
  /// The decoded image source (file, network, camera frame) is the caller's
  /// responsibility — this runner does not depend on a specific image codec.
  ///
  /// [promptIds] **must already contain the right number of `imageTokenId`
  /// placeholders** (matching the chosen bucket's merged-token count). The
  /// caller is expected to construct the prompt from the chat template +
  /// computed token count, mirroring the MLX path's contract.
  Future<List<int>> generate({
    required Uint8List imageBytes,
    required int imageHeight,
    required int imageWidth,
    required List<int> promptIds,
    required int maxNewTokens,
    int? eosTokenId,
    void Function(int tokenId, int index)? onToken,
  }) async {
    _ensureOpen();
    final eos = eosTokenId ?? manifest.tokens.eosTokenId;

    // ── 1. Image preprocessing ────────────────────────────────────────────
    final resized = smartResize(
      height: imageHeight,
      width: imageWidth,
      factor: manifest.vision.patchSize * manifest.vision.spatialMerge,
    );
    final bucket = pickImageBucket(
      resizedHeight: resized.height,
      resizedWidth: resized.width,
      buckets: manifest.vision.buckets,
      patchSize: manifest.vision.patchSize,
    );
    final pre = preprocessImage(
      imageRgb: imageBytes,
      imageHeight: imageHeight,
      imageWidth: imageWidth,
      bucket: bucket,
      patchSize: manifest.vision.patchSize,
      spatialMergeSize: manifest.vision.spatialMerge,
    );

    // Sanity: prompt must contain merged-token count of placeholders.
    final mergedCount = (bucket.$1 *
            (bucket.$2 ~/ manifest.vision.spatialMerge) *
            (bucket.$3 ~/ manifest.vision.spatialMerge));
    final placeholderCount =
        promptIds.where((id) => id == manifest.tokens.imageTokenId).length;
    if (placeholderCount != mergedCount) {
      throw StateError(
        'prompt has $placeholderCount image-token placeholders but bucket '
        '$bucket requires $mergedCount',
      );
    }

    // ── 2. Reset KV state for a clean generation ─────────────────────────
    _prefill.resetState();
    _decode.resetState();

    // ── 3. Vision + embed: fused inputs_embeds in one shot ───────────────
    final inputIdsI32 =
        Int32List.fromList(promptIds); // CoreML int input is i32
    final gridI32 = Int32List.fromList([bucket.$1, bucket.$2, bucket.$3]);
    final visionOut = _visionEmbed.predict({
      _CoremlIo.visionPixels: (
        [1, 3, pre.resizedHeight, pre.resizedWidth],
        pre.pixelValues,
      ),
      _CoremlIo.visionInputIds: ([1, promptIds.length], inputIdsI32),
      _CoremlIo.visionGridThw: ([3], gridI32),
    });
    final fusedRecord =
        visionOut[_CoremlIo.visionInputsEmbeds]! as (List<int>, Float32List);
    final fusedShape = fusedRecord.$1;
    final fusedEmbeds = fusedRecord.$2;
    final hiddenSize = fusedShape.last; // 1024

    // ── 4. mRoPE positions for prefill ───────────────────────────────────
    final prefillPositions = computeMRopePositionIds(
      inputIds: promptIds,
      imageGridThw: bucket,
      imageTokenId: manifest.tokens.imageTokenId,
      spatialMergeSize: manifest.vision.spatialMerge,
    );
    final anchor = lastPositionTriple(prefillPositions, promptIds.length);

    // ── 5. Pad embeds + positions + mask to the chosen prefill bucket ─────
    final seqBucket = manifest.pickPrefillBucket(promptIds.length);
    if (promptIds.length > seqBucket) {
      throw StateError(
        'prompt length ${promptIds.length} exceeds largest prefill bucket '
        '$seqBucket; fall back to MLX runner',
      );
    }
    final paddedEmbeds = _padEmbedsToBucket(
      fusedEmbeds,
      promptLen: promptIds.length,
      hiddenSize: hiddenSize,
      bucket: seqBucket,
    );
    final paddedPositions = _padPositionsToBucket(
      prefillPositions,
      promptLen: promptIds.length,
      bucket: seqBucket,
    );
    final causalMask = _buildCausalMask(
      promptLen: promptIds.length,
      bucket: seqBucket,
    );

    // ── 6. Prefill: returns last-position logits only ────────────────────
    final prefillOut = _prefill.predict({
      _CoremlIo.prefillEmbeds: ([1, seqBucket, hiddenSize], paddedEmbeds),
      _CoremlIo.prefillPositionIds: ([3, 1, seqBucket], paddedPositions),
      _CoremlIo.prefillCausalMask: ([1, 1, seqBucket, seqBucket], causalMask),
    });
    final firstLogits =
        (prefillOut[_CoremlIo.prefillLogits]! as (List<int>, Float32List)).$2;
    var nextToken = _argmax(firstLogits);

    final generated = <int>[];
    if (nextToken == eos) return generated;
    generated.add(nextToken);
    onToken?.call(nextToken, 0);

    // ── 7. Decode loop ───────────────────────────────────────────────────
    var pastKvLen = promptIds.length;
    for (var step = 1; step < maxNewTokens; step++) {
      // 7a. Embed the previously sampled token (token_embed mlpackage).
      final tokenIn = Int32List.fromList([nextToken]);
      final tokOut = _tokenEmbed.predict({
        _CoremlIo.tokenEmbedInputId: ([1, 1], tokenIn),
      });
      final tokEmbed =
          (tokOut[_CoremlIo.tokenEmbedOut]! as (List<int>, Float32List)).$2;

      // 7b. mRoPE single-token position.
      final pos = computeDecodePositionIds(
        newTokenIndex: step - 1,
        anchorPosition: anchor,
      );

      // 7c. Stateful decode step.
      final out = _decode.predict({
        _CoremlIo.decodeTokenEmbed: ([1, 1, hiddenSize], tokEmbed),
        _CoremlIo.decodePositionIds: ([3, 1, 1], pos),
        _CoremlIo.decodePastKvLen: ([1], Int32List.fromList([pastKvLen])),
      });
      final logits =
          (out[_CoremlIo.decodeLogits]! as (List<int>, Float32List)).$2;
      nextToken = _argmax(logits);
      pastKvLen += 1;

      if (nextToken == eos) break;
      generated.add(nextToken);
      onToken?.call(nextToken, step);
    }

    return generated;
  }

  /// Release native sessions + MLState. Idempotent.
  void close() {
    if (_closed) return;
    _closed = true;
    _visionEmbed.close();
    _tokenEmbed.close();
    _prefill.close();
    _decode.close();
  }

  void _ensureOpen() {
    if (_closed) throw StateError('PaddleOcrVlCoremlRunner used after close()');
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  /// Pad `[1, promptLen, hidden]` → `[1, bucket, hidden]` with zeros.
  static Float32List _padEmbedsToBucket(
    Float32List src, {
    required int promptLen,
    required int hiddenSize,
    required int bucket,
  }) {
    if (promptLen == bucket) return src;
    final padded = Float32List(bucket * hiddenSize);
    padded.setRange(0, promptLen * hiddenSize, src);
    return padded;
  }

  /// Pad `[3, 1, promptLen]` → `[3, 1, bucket]` with the last real position
  /// repeated (so RoPE on padding doesn't blow up; values are masked anyway).
  static Float32List _padPositionsToBucket(
    Float32List src, {
    required int promptLen,
    required int bucket,
  }) {
    if (promptLen == bucket) return src;
    final padded = Float32List(3 * bucket);
    for (var stream = 0; stream < 3; stream++) {
      final srcBase = stream * promptLen;
      final dstBase = stream * bucket;
      for (var i = 0; i < promptLen; i++) {
        padded[dstBase + i] = src[srcBase + i];
      }
      final fill = src[srcBase + promptLen - 1];
      for (var i = promptLen; i < bucket; i++) {
        padded[dstBase + i] = fill;
      }
    }
    return padded;
  }

  /// Build `[1, 1, bucket, bucket]` additive causal mask in fp16-safe FP32:
  ///   - 0.0 where attention is allowed (lower triangle of the prompt block)
  ///   - large negative (-1e4) elsewhere — both above-diagonal and any
  ///     padded positions.
  static Float32List _buildCausalMask({
    required int promptLen,
    required int bucket,
  }) {
    final mask = Float32List(bucket * bucket);
    const neg = -1e4;
    for (var i = 0; i < bucket; i++) {
      for (var j = 0; j < bucket; j++) {
        final inPrompt = i < promptLen && j < promptLen;
        final causal = j <= i;
        mask[i * bucket + j] = (inPrompt && causal) ? 0.0 : neg;
      }
    }
    return mask;
  }

  /// Greedy argmax over the last logits row. Tiny (~100k vocab) so a plain
  /// linear scan is faster than reaching for an isolate.
  static int _argmax(Float32List logits) {
    var bestIdx = 0;
    var bestVal = logits[0];
    for (var i = 1; i < logits.length; i++) {
      final v = logits[i];
      if (v > bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    }
    return bestIdx;
  }
}

/// Default real-FFI loader. Stubbed until Phase 2 lands — throws so any
/// accidental call without setting a test override fails fast.
CoremlLoader _defaultLoader() {
  return _UnimplementedCoremlLoader();
}

class _UnimplementedCoremlLoader implements CoremlLoader {
  @override
  CoremlSession loadStage({
    required String packagePath,
    required CoremlComputeUnits computeUnits,
    required bool stateful,
  }) {
    throw UnimplementedError(
      'PaddleOcrVlCoremlRunner default loader requires Phase 2 '
      '(coreml_runtime.dart). Set testCoremlLoaderOverride to inject a fake '
      'loader for tests, or wait for Phase 2 to land.',
    );
  }
}
