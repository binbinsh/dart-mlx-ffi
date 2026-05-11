/// Hybrid CoreML-vision + MLX-decoder runner for PaddleOCR-VL-1.5
/// (issue #1, commit #8 — the centerpiece of the hybrid OCR refactor).
///
/// **Pipeline cadence per [PaddleOcrVlHybridRunner.generate] call:**
///
/// 1. CoreML `vision_embed.mlpackage` runs **once** to turn the raw image
///    bytes into a rank-2 `image_embeds [num_image_tokens, hidden]` tensor
///    (Float32List). This is the only CoreML stage this runner ever opens —
///    `token_embed`, `prefill_decoder`, `decode_decoder` are deliberately
///    NOT loaded, even if they are present in `pipeline.json`.
/// 2. The Float32List is wrapped into an [MlxArray] and handed to the MLX
///    runner's pre-existing [PaddleOcrVlRunner.generateFromVisionFeaturesDetailed]
///    entry point, which:
///      a. Runs the LM `embed_tokens` table over the prompt token IDs to
///         produce text embeddings.
///      b. Scatters `image_embeds` rows into the placeholder positions via
///         the [paddleOcrVlScatterImageEmbeddings] helper extracted in
///         commit #2 (which transparently `astype()`s the image features to
///         match the text-embedding dtype — bf16/fp16/fp32 all work).
///      c. Runs prefill + greedy decode on MLX through the established
///         `_prefillFromEmbeddingWithCache` + `_forwardWithCache` paths.
/// 3. Returns the generated token IDs.
///
/// The MLX runner is loaded with `keepVisionWeights: false` (commit #4) so
/// the ViT tower never enters MLX RAM — the visual.* tensors are dropped
/// immediately after the safetensors file is parsed. On the real
/// PaddleOCR-VL-1.5 weights this saves roughly 385 MB of GPU memory.
///
/// The legacy 4-stage `PaddleOcrVlCoremlRunner` was removed in commit #11
/// once this hybrid path was validated end-to-end. The CoreML loader/session
/// abstractions it used to expose now live in `coreml_loader.dart`.
library;

import 'dart:typed_data';

import 'package:dart_inference/mlx.dart';

import 'coreml_image.dart';
import 'coreml_loader.dart'
    show
        CoremlLoader,
        CoremlSession,
        defaultCoremlLoader,
        testCoremlLoaderOverride;
import 'coreml_pipeline_manifest.dart';
import 'paddle_ocr_vl.dart';

/// Hybrid runner: CoreML for the ViT, MLX for the decoder.
///
/// Held resources:
///   - one [CoremlSession] for `vision_embed`;
///   - one [PaddleOcrVlRunner] for the language-model decoder, loaded with
///     `keepVisionWeights: false`.
///
/// Both must be released via [close]. The runner is single-shot in the
/// sense that no warmup is needed — the very first `generate` triggers ANE
/// kernel compilation just like the legacy runner's first call, and the
/// MLX side runs the same cold prefill it always has.
final class PaddleOcrVlHybridRunner {
  PaddleOcrVlHybridRunner._({
    required this.bundlePath,
    required this.snapshotPath,
    required this.manifest,
    required CoremlSession visionEmbed,
    required PaddleOcrVlRunner mlxRunner,
  }) : _visionEmbed = visionEmbed,
       _mlxRunner = mlxRunner;

  /// Directory containing `pipeline.json` + the `vision_embed.mlpackage`.
  /// Other stages (`token_embed`, `prefill_decoder`, `decode_decoder`) may
  /// also be present in this directory but are intentionally ignored.
  final String bundlePath;

  /// Directory containing `config.json` + decoder safetensors. May or may
  /// not contain `visual.*` tensors — they will be dropped on load.
  final String snapshotPath;

  /// Parsed `pipeline.json`. Kept for image preprocessing (bucket list,
  /// patch / merge sizes) and special-token IDs.
  final CoremlPipelineManifest manifest;

  final CoremlSession _visionEmbed;
  final PaddleOcrVlRunner _mlxRunner;

  bool _closed = false;

  /// MLX runner accessor — exposed primarily for tests / advanced callers
  /// that want to reach the decoder's config or run additional generations
  /// in text-only mode without going through `generate`.
  PaddleOcrVlRunner get mlxRunner {
    _ensureOpen();
    return _mlxRunner;
  }

  /// Load both halves of the hybrid pipeline.
  ///
  /// - [coremlBundlePath] points at a Phase 1 CoreML bundle directory.
  ///   Only its `pipeline.json` and `vision_embed.mlpackage` are touched;
  ///   the other three mlpackages may exist but are not opened.
  /// - [mlxSnapshotPath] points at an MLX weights snapshot directory
  ///   (`config.json` + `*.safetensors`). The visual.* tensors, if any,
  ///   are dropped during load.
  static Future<PaddleOcrVlHybridRunner> load({
    required String coremlBundlePath,
    required String mlxSnapshotPath,
  }) async {
    final manifest = CoremlPipelineManifest.loadFile(
      '$coremlBundlePath/pipeline.json',
    );
    final stage = manifest.stage('vision_embed');
    final loader = testCoremlLoaderOverride ?? _defaultLoader();
    final visionEmbed = loader.loadStage(
      packagePath: '$coremlBundlePath/${stage.package}',
      computeUnits: stage.computeUnits,
      stateful: stage.stateful,
    );

    final mlxRunner = PaddleOcrVlRunner.load(
      mlxSnapshotPath,
      keepVisionWeights: false,
    );

    return PaddleOcrVlHybridRunner._(
      bundlePath: coremlBundlePath,
      snapshotPath: mlxSnapshotPath,
      manifest: manifest,
      visionEmbed: visionEmbed,
      mlxRunner: mlxRunner,
    );
  }

  /// Run the full vision -> prefill -> decode pipeline.
  ///
  /// - [imageBytes] is RGB HWC `Uint8List` of length `imageHeight*imageWidth*3`.
  /// - [promptIds] must contain exactly the merged-token count of
  ///   `imageTokenId` placeholders for the chosen bucket; mismatches throw
  ///   `StateError` from the MLX runner's position-id builder.
  /// - Returns prompt + generated tokens (matching the MLX runner's
  ///   `generateFromVisionFeaturesDetailed.fullTokenIds` convention).
  ///
  /// `onStage` is forwarded to the MLX runner's existing tracing hook for
  /// debugging.
  List<int> generate({
    required Uint8List imageBytes,
    required int imageHeight,
    required int imageWidth,
    required List<int> promptIds,
    int maxNewTokens = 512,
    void Function(String message)? onStage,
  }) {
    _ensureOpen();

    // ── 1. Image preprocessing. ────────────────────────────────────────
    // Uses the public helpers from `coreml_image.dart` (smartResize,
    // pickImageBucket, preprocessImage). The legacy 4-stage runner that
    // used to share this preprocessing was removed in commit #11.
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

    final mergedCount =
        bucket.$1 *
        (bucket.$2 ~/ manifest.vision.spatialMerge) *
        (bucket.$3 ~/ manifest.vision.spatialMerge);
    final placeholderCount = promptIds
        .where((id) => id == manifest.tokens.imageTokenId)
        .length;
    if (placeholderCount != mergedCount) {
      throw StateError(
        'prompt has $placeholderCount image-token placeholders but bucket '
        '$bucket requires $mergedCount',
      );
    }

    // ── 2. CoreML vision_embed — only stage we open. ─────────────────────
    final numPatches = bucket.$1 * bucket.$2 * bucket.$3;
    final gridI32 = Int32List.fromList([bucket.$1, bucket.$2, bucket.$3]);
    final visionOut = _visionEmbed.predict({
      'pixel_values': (
        [
          1,
          numPatches,
          3 * manifest.vision.patchSize * manifest.vision.patchSize,
        ],
        pre.pixelValues,
      ),
      'image_grid_thw': ([3], gridI32),
    });
    final imageRecord = visionOut['image_embeds']! as (List<int>, Float32List);
    final imageEmbedShape = imageRecord.$1;
    final imageEmbedFloats = imageRecord.$2;
    if (imageEmbedShape.length != 2) {
      throw StateError(
        'vision_embed produced rank-${imageEmbedShape.length} image_embeds '
        '(shape $imageEmbedShape); expected rank-2 [num_image_tokens, hidden].',
      );
    }
    final numImageTokens = imageEmbedShape[0];
    final imageHiddenSize = imageEmbedShape[1];
    if (numImageTokens != mergedCount) {
      throw StateError(
        'vision_embed returned $numImageTokens image tokens; bucket $bucket '
        'expects $mergedCount.',
      );
    }
    if (imageHiddenSize != _mlxRunner.config.hiddenSize) {
      throw StateError(
        'vision_embed hidden size $imageHiddenSize does not match MLX '
        'decoder hidden size ${_mlxRunner.config.hiddenSize}; CoreML and '
        'MLX snapshots are out of sync.',
      );
    }
    if (imageEmbedFloats.length != numImageTokens * imageHiddenSize) {
      throw StateError(
        'vision_embed produced ${imageEmbedFloats.length} floats; expected '
        '${numImageTokens * imageHiddenSize} for shape $imageEmbedShape.',
      );
    }

    onStage?.call(
      'hybrid: vision_embed done shape=$imageEmbedShape bucket=$bucket',
    );

    // ── 3. Float32List -> MlxArray. ──────────────────────────────────────
    // We construct an fp32 MlxArray; the MLX-side scatter (commit #2) calls
    // `imageHidden.astype(textEmbed.dtype)` so a bf16 / fp16 decoder
    // snapshot is handled transparently. There is no dtype-mismatch
    // blocker on the hybrid boundary.
    final imageHidden = MlxArray.fromFloat32List(
      imageEmbedFloats,
      shape: [numImageTokens, imageHiddenSize],
    );

    try {
      // ── 4 + 5 + 6. Drive prompt embed + scatter + prefill + decode through
      // the MLX runner's existing vision-features entry point. The MLX
      // runner already knows how to expand placeholders, build mRoPE
      // positions, run prefill and greedy decode loop.
      final result = _mlxRunner.generateFromVisionFeaturesDetailed(
        promptIds,
        imageHidden,
        gridHeight: bucket.$2,
        gridWidth: bucket.$3,
        maxNewTokens: maxNewTokens,
        onStage: onStage,
      );
      return result.fullTokenIds;
    } finally {
      imageHidden.close();
    }
  }

  /// Debug seam used by the hybrid benchmark: runs the CoreML
  /// `vision_embed` stage exactly like [generate], then returns only the
  /// last-token prefill logits prefix `[1, width]` (defaults to 16) as a
  /// fp32 [MlxArray]. No decode loop is run.
  ///
  /// This mirrors `PaddleOcrVlRunner.debugPrefillLogitsPrefixFromPixelValues`
  /// but with the ViT replaced by CoreML, giving the publish bench a
  /// directly-comparable scalar signal between the pure-MLX path and the
  /// hybrid path.
  ///
  /// Caller owns the returned array and must close it.
  MlxArray debugFirstTokenLogitsPrefix({
    required Uint8List imageBytes,
    required int imageHeight,
    required int imageWidth,
    required List<int> promptIds,
    int width = 16,
  }) {
    _ensureOpen();

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
    final mergedCount =
        bucket.$1 *
        (bucket.$2 ~/ manifest.vision.spatialMerge) *
        (bucket.$3 ~/ manifest.vision.spatialMerge);
    final placeholderCount = promptIds
        .where((id) => id == manifest.tokens.imageTokenId)
        .length;
    if (placeholderCount != mergedCount) {
      throw StateError(
        'prompt has $placeholderCount image-token placeholders but bucket '
        '$bucket requires $mergedCount',
      );
    }

    final gridI32 = Int32List.fromList([bucket.$1, bucket.$2, bucket.$3]);
    final numPatches = bucket.$1 * bucket.$2 * bucket.$3;
    final visionOut = _visionEmbed.predict({
      'pixel_values': (
        [
          1,
          numPatches,
          3 * manifest.vision.patchSize * manifest.vision.patchSize,
        ],
        pre.pixelValues,
      ),
      'image_grid_thw': ([3], gridI32),
    });
    final imageRecord = visionOut['image_embeds']! as (List<int>, Float32List);
    final imageEmbedShape = imageRecord.$1;
    final imageEmbedFloats = imageRecord.$2;
    if (imageEmbedShape.length != 2) {
      throw StateError(
        'vision_embed produced rank-${imageEmbedShape.length} image_embeds',
      );
    }
    final numImageTokens = imageEmbedShape[0];
    final imageHiddenSize = imageEmbedShape[1];

    final imageHidden = MlxArray.fromFloat32List(
      imageEmbedFloats,
      shape: [numImageTokens, imageHiddenSize],
    );
    try {
      return _mlxRunner.debugPrefillLogitsPrefixFromVisionFeatures(
        promptIds,
        imageHidden,
        gridHeight: bucket.$2,
        gridWidth: bucket.$3,
        width: width,
      );
    } finally {
      imageHidden.close();
    }
  }

  /// Release both the CoreML session and the MLX runner. Idempotent.
  void close() {
    if (_closed) return;
    _closed = true;
    _visionEmbed.close();
    _mlxRunner.close();
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('PaddleOcrVlHybridRunner used after close()');
    }
  }
}

CoremlLoader _defaultLoader() => defaultCoremlLoader();
