/// mRoPE (Multimodal 3D Rotary Position Embedding) position-id computation
/// for the PaddleOCR-VL-1.5 CoreML pipeline.
///
/// This is a **self-contained** Dart port of HuggingFace's `get_rope_index`
/// from `transformers` (`modeling_paddleocr_vl.py`, lines ~1902–2123 in the
/// snapshot at
/// `models--PaddlePaddle--PaddleOCR-VL-1.5/.../modeling_paddleocr_vl.py`).
///
/// The CoreML pipeline ships position ids from Dart instead of computing them
/// inside the graph (the original PyTorch code uses `torch.argwhere` /
/// dynamic Python loops that don't trace into MLProgram cleanly). The
/// algorithm is single-batch, single-image — multi-image / video paths are
/// out of scope (ADR §12).
///
/// Output layout matches HF: `[3, 1, seq_len]`, row-major, where the 3
/// streams are `(temporal, height, width)`. We return a `Float32List` so it
/// can feed CoreML directly without a dtype convert step (CoreML positions
/// are FP32 in the prefill/decode mlpackages emitted by Phase 1).
library;

import 'dart:typed_data';

/// `(t, h, w)` triple for an `image_grid_thw` value.
///
/// `t` = temporal patches (always 1 for a single still image),
/// `h` and `w` are the *unmerged* patch grid dimensions
/// (i.e. before `spatial_merge_size` is applied).
typedef ImageGridThw = (int t, int h, int w);

/// `mrope_section` from `config.rope_scaling`. Default `(16, 24, 24)`
/// for ERNIE-4.5 with `head_dim=128` (sum × 2 = 128).
typedef MropeSection = (int text, int height, int width);

/// Compute the prefill mRoPE position ids for an `inputIds` sequence that
/// contains exactly one image span.
///
/// Returns a flat `Float32List` of length `3 * seqLen` representing the
/// `[3, 1, seqLen]` tensor in row-major order:
///
/// ```
/// [t_0, t_1, …, t_{S-1},  h_0, h_1, …, h_{S-1},  w_0, w_1, …, w_{S-1}]
/// ```
///
/// Algorithm (mirrors `get_rope_index` for `image_token_id` only):
///
/// 1. Text tokens **before** the image use sequential 1-D positions
///    `[0, 1, …, st_idx-1]` on all 3 streams.
/// 2. The image span occupies `(h*w) / (spatial_merge**2)` slots. For each
///    image token at flat index `i`:
///      - `t = st_idx`  (always 0 for still images, since `t == 1` and
///        `second_per_grid_t` is `0` for images)
///      - `h = st_idx + (i / merged_w)`
///      - `w = st_idx + (i % merged_w)`
///    where `st_idx` is the start position right after the preceding text.
/// 3. Trailing text resumes at `max(image_positions) + 1`.
///
/// **Important:** `imageTokenId` must occur *contiguously* in `inputIds` and
/// the count must equal `(t * h * w) / (spatial_merge ** 2)`. Caller is
/// responsible for inserting that many `<image>` placeholders.
///
/// If `inputIds` contains no `imageTokenId`, returns sequential text-only
/// positions on all 3 streams (matches the HF "no image_grid_thw" branch).
Float32List computeMRopePositionIds({
  required List<int> inputIds,
  required ImageGridThw imageGridThw,
  required int imageTokenId,
  int spatialMergeSize = 2,
}) {
  final seqLen = inputIds.length;
  final out = Float32List(3 * seqLen);

  // Locate the contiguous image span.
  var imageStart = -1;
  for (var i = 0; i < seqLen; i++) {
    if (inputIds[i] == imageTokenId) {
      imageStart = i;
      break;
    }
  }

  if (imageStart < 0) {
    // Text-only: HF falls into the `else` branch and emits arange on all 3.
    for (var i = 0; i < seqLen; i++) {
      out[i] = i.toDouble();             // temporal
      out[seqLen + i] = i.toDouble();    // height
      out[2 * seqLen + i] = i.toDouble(); // width
    }
    return out;
  }

  var imageEnd = imageStart;
  while (imageEnd < seqLen && inputIds[imageEnd] == imageTokenId) {
    imageEnd++;
  }
  final imageCount = imageEnd - imageStart;

  final (t, h, w) = imageGridThw;
  final llmGridT = t;
  final llmGridH = h ~/ spatialMergeSize;
  final llmGridW = w ~/ spatialMergeSize;
  final expectedImageCount = llmGridT * llmGridH * llmGridW;
  if (imageCount != expectedImageCount) {
    throw ArgumentError(
      'image-token count $imageCount does not match grid '
      '($llmGridT, $llmGridH, $llmGridW) → $expectedImageCount',
    );
  }

  // 1. Pre-image text: positions [0 .. imageStart).
  for (var i = 0; i < imageStart; i++) {
    out[i] = i.toDouble();
    out[seqLen + i] = i.toDouble();
    out[2 * seqLen + i] = i.toDouble();
  }
  final stIdx = imageStart; // == max(prev positions)+1, prev max = imageStart-1

  // 2. Image span. For still images `second_per_grid_t = 0` ⇒ t_index is
  //    all zeros, then we add `text_len + st_idx = stIdx` to every stream.
  for (var i = 0; i < imageCount; i++) {
    final tokenIdx = imageStart + i;
    final hi = i ~/ llmGridW;
    final wi = i % llmGridW;
    out[tokenIdx] = stIdx.toDouble();             // t_index = 0 + stIdx
    out[seqLen + tokenIdx] = (stIdx + hi).toDouble();
    out[2 * seqLen + tokenIdx] = (stIdx + wi).toDouble();
  }

  // Compute max image position to seed trailing text.
  // max_t = stIdx; max_h = stIdx + llmGridH - 1; max_w = stIdx + llmGridW - 1.
  // HF uses max across all 3 streams.
  final imageMax = stIdx +
      (llmGridH > llmGridW ? llmGridH - 1 : llmGridW - 1);
  // text_len for image block = 0 (we already counted the image), trailing
  // text starts at imageMax + 1.
  final trailingBase = imageMax + 1;
  for (var i = imageEnd; i < seqLen; i++) {
    final v = (trailingBase + (i - imageEnd)).toDouble();
    out[i] = v;
    out[seqLen + i] = v;
    out[2 * seqLen + i] = v;
  }

  return out;
}

/// Compute the single-token mRoPE position id for autoregressive decode.
///
/// In the decode loop the model sees one new text token per step. The HF
/// reference advances all 3 mRoPE streams by `+1` from the previous step's
/// max — for trailing text after an image, this is just
/// `(maxImagePos + 1) + stepIndex`.
///
/// [anchorPosition] is the `(t, h, w)` triple for the **last** position
/// emitted by `computeMRopePositionIds` (i.e. the position of the prompt's
/// final non-padding token). The first decode step adds 1 to each.
///
/// Returns a flat `Float32List` of length `3` (shape `[3, 1, 1]` flattened).
Float32List computeDecodePositionIds({
  required int newTokenIndex,
  required ImageGridThw anchorPosition,
}) {
  final (t, h, w) = anchorPosition;
  final step = newTokenIndex + 1;
  return Float32List.fromList([
    (t + step).toDouble(),
    (h + step).toDouble(),
    (w + step).toDouble(),
  ]);
}

/// Helper: extract the last `(t, h, w)` triple from a flat position-id buffer
/// produced by [computeMRopePositionIds]. Useful for seeding decode.
ImageGridThw lastPositionTriple(Float32List flat, int seqLen) {
  final last = seqLen - 1;
  return (
    flat[last].toInt(),
    flat[seqLen + last].toInt(),
    flat[2 * seqLen + last].toInt(),
  );
}
