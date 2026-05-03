/// Image preprocessing for the PaddleOCR-VL CoreML pipeline.
///
/// Ports the relevant parts of HF's
/// `image_processing_paddleocr_vl.PaddleOCRVLImageProcessor` (snapshot
/// commit 89769a87…) — specifically `smart_resize` and the
/// rescale/normalize/patchify chain — into pure Dart with no external image
/// codec dependency.
///
/// Inputs to [preprocessImage] are **already-decoded RGB pixel bytes** in
/// HWC `Uint8List` order (one byte per channel, channels=3, no alpha). The
/// caller (e.g. `package:image` in Flutter, or stb_image at the FFI layer)
/// owns image decode + EXIF orientation correction. We do *not* do BGR↔RGB
/// flips; HF's processor calls `convert("RGB")` so we assume RGB input.
///
/// Normalization: as instructed by Phase 3 spec, we use
/// `mean = std = [0.5, 0.5, 0.5]` (mapping `[0, 1]` → `[-1, 1]`). This
/// matches the canonical SigLIP recipe used by PaddleOCR-VL's vision tower
/// after the `do_rescale` step. **Note:** the HF default `image_mean` /
/// `image_std` are `OPENAI_CLIP_MEAN` / `OPENAI_CLIP_STD` for backwards
/// compat, but the deployed PaddleOCR-VL preprocessor overrides these to
/// `[0.5, 0.5, 0.5]` (see `preprocessor_config.json` shipped with the
/// snapshot). Phase 1 must verify by reading `preprocessor_config.json`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Result of [smartResize].
typedef SmartResizeResult = ({int height, int width});

/// Verbatim port of `smart_resize` from `image_processing_paddleocr_vl.py`
/// (snapshot lines 128–173).
///
/// - Both output dims are multiples of [factor].
/// - Total pixel count is clamped to `[minPixels, maxPixels]`.
/// - Aspect ratio is preserved as closely as possible.
/// - Throws if aspect ratio > 200.
///
/// Defaults match HF: `factor = 28` (= patch_size 14 × merge_size 2),
/// `min_pixels = 28*28*130`, `max_pixels = 28*28*1280`.
SmartResizeResult smartResize({
  required int height,
  required int width,
  int factor = 28,
  int minPixels = 28 * 28 * 130,
  int maxPixels = 28 * 28 * 1280,
}) {
  var h = height;
  var w = width;
  if (h < factor) {
    w = ((w * factor) / h).round();
    h = factor;
  }
  if (w < factor) {
    h = ((h * factor) / w).round();
    w = factor;
  }
  final aspect = math.max(h, w) / math.min(h, w);
  if (aspect > 200) {
    throw ArgumentError(
      'absolute aspect ratio must be smaller than 200, got $aspect',
    );
  }

  // round() in Dart uses banker's rounding for .5 — Python `round` uses the
  // same. For the typical OCR aspect ratios this matters only for exact
  // half-pixel cases, which `smart_resize` then re-snaps to the factor grid
  // anyway. Acceptable.
  var hBar = (h / factor).round() * factor;
  var wBar = (w / factor).round() * factor;
  if (hBar * wBar > maxPixels) {
    final beta = math.sqrt((h * w) / maxPixels);
    hBar = ((h / beta) / factor).floor() * factor;
    wBar = ((w / beta) / factor).floor() * factor;
  } else if (hBar * wBar < minPixels) {
    final beta = math.sqrt(minPixels / (h * w));
    hBar = ((h * beta) / factor).ceil() * factor;
    wBar = ((w * beta) / factor).ceil() * factor;
  }
  return (height: hBar, width: wBar);
}

/// Bicubic-ish bilinear resize (cheap and good enough for FP16 ViT). HF uses
/// PIL bicubic, which is more accurate; if Phase 4 golden gates show drift
/// we can swap in a true bicubic at the cost of ~3× CPU. For now bilinear
/// is the deliberate trade-off — vision is FP16 anyway.
///
/// Input/output are RGB HWC `Uint8List`s.
Uint8List bilinearResizeRgb(
  Uint8List src, {
  required int srcHeight,
  required int srcWidth,
  required int dstHeight,
  required int dstWidth,
}) {
  final dst = Uint8List(dstHeight * dstWidth * 3);
  final scaleY = srcHeight / dstHeight;
  final scaleX = srcWidth / dstWidth;
  for (var y = 0; y < dstHeight; y++) {
    final sy = (y + 0.5) * scaleY - 0.5;
    final y0 = sy.floor().clamp(0, srcHeight - 1);
    final y1 = (y0 + 1).clamp(0, srcHeight - 1);
    final wy = sy - y0;
    for (var x = 0; x < dstWidth; x++) {
      final sx = (x + 0.5) * scaleX - 0.5;
      final x0 = sx.floor().clamp(0, srcWidth - 1);
      final x1 = (x0 + 1).clamp(0, srcWidth - 1);
      final wx = sx - x0;
      final dstBase = (y * dstWidth + x) * 3;
      for (var c = 0; c < 3; c++) {
        final p00 = src[(y0 * srcWidth + x0) * 3 + c];
        final p01 = src[(y0 * srcWidth + x1) * 3 + c];
        final p10 = src[(y1 * srcWidth + x0) * 3 + c];
        final p11 = src[(y1 * srcWidth + x1) * 3 + c];
        final v = (p00 * (1 - wx) + p01 * wx) * (1 - wy) +
            (p10 * (1 - wx) + p11 * wx) * wy;
        dst[dstBase + c] = v.round().clamp(0, 255);
      }
    }
  }
  return dst;
}

/// Result of [preprocessImage].
final class PreprocessedImage {
  const PreprocessedImage({
    required this.pixelValues,
    required this.gridThw,
    required this.resizedHeight,
    required this.resizedWidth,
  });

  /// Flat `Float32List` of pre-normalized patches, layout
  /// `[numPatches, 3, patchSize, patchSize]` row-major.
  ///
  /// The exact shape depends on the chosen image grid bucket — Phase 1
  /// emits `EnumeratedShapes` over `(1, 3, h_pix, w_pix)` per ADR §5.1.
  final Float32List pixelValues;

  /// `(t, h, w)` matching the chosen bucket, where h and w are *unmerged*
  /// patch counts (so `h_pix = h * patch_size`).
  final (int t, int h, int w) gridThw;

  /// Pixel dimensions actually fed to the model after smart_resize +
  /// bucket-snap. May differ from the raw image dimensions.
  final int resizedHeight;
  final int resizedWidth;
}

/// Run the full HF-equivalent preprocessing chain:
///
///   smart_resize → bilinear resize → rescale `/255` → normalize → patchify.
///
/// [imageRgb] must be HWC `Uint8List` with `imageHeight * imageWidth * 3`
/// bytes (RGB, no alpha). [bucket] is the chosen `(t, h, w)` triple from the
/// Phase 1 enumerated set; we resize *to that bucket's exact pixel size*
/// (= `h * patchSize` × `w * patchSize`) so the CoreML EnumeratedShape
/// matches byte-for-byte.
PreprocessedImage preprocessImage({
  required Uint8List imageRgb,
  required int imageHeight,
  required int imageWidth,
  required (int t, int h, int w) bucket,
  int patchSize = 14,
  int spatialMergeSize = 2,
  List<double> mean = const [0.5, 0.5, 0.5],
  List<double> std = const [0.5, 0.5, 0.5],
}) {
  if (imageRgb.length != imageHeight * imageWidth * 3) {
    throw ArgumentError(
      'imageRgb length ${imageRgb.length} != H*W*3 = '
      '${imageHeight * imageWidth * 3}',
    );
  }
  final (t, gridH, gridW) = bucket;
  final dstH = gridH * patchSize;
  final dstW = gridW * patchSize;

  final resized = (dstH == imageHeight && dstW == imageWidth)
      ? imageRgb
      : bilinearResizeRgb(
          imageRgb,
          srcHeight: imageHeight,
          srcWidth: imageWidth,
          dstHeight: dstH,
          dstWidth: dstW,
        );

  // Patchify directly into NCHW-per-patch layout
  // [num_patches, 3, patchSize, patchSize], rescaling and normalizing
  // in the same pass to avoid two big allocations.
  final numPatches = t * gridH * gridW;
  final out = Float32List(numPatches * 3 * patchSize * patchSize);
  final inv255 = 1.0 / 255.0;
  final m0 = mean[0], m1 = mean[1], m2 = mean[2];
  final s0 = std[0], s1 = std[1], s2 = std[2];

  for (var pr = 0; pr < gridH; pr++) {
    for (var pc = 0; pc < gridW; pc++) {
      final patchBase = (pr * gridW + pc) * 3 * patchSize * patchSize;
      for (var ph = 0; ph < patchSize; ph++) {
        final srcRow = pr * patchSize + ph;
        final srcRowBase = srcRow * dstW * 3;
        for (var pw = 0; pw < patchSize; pw++) {
          final srcPx = srcRowBase + (pc * patchSize + pw) * 3;
          final r = (resized[srcPx] * inv255 - m0) / s0;
          final g = (resized[srcPx + 1] * inv255 - m1) / s1;
          final b = (resized[srcPx + 2] * inv255 - m2) / s2;
          out[patchBase + 0 * patchSize * patchSize + ph * patchSize + pw] = r;
          out[patchBase + 1 * patchSize * patchSize + ph * patchSize + pw] = g;
          out[patchBase + 2 * patchSize * patchSize + ph * patchSize + pw] = b;
        }
      }
    }
  }

  return PreprocessedImage(
    pixelValues: out,
    gridThw: bucket,
    resizedHeight: dstH,
    resizedWidth: dstW,
  );
}

/// Pick the smallest `(t, h, w)` bucket from [buckets] that fits the
/// post-`smart_resize` shape. "Fits" means `bucket.h * bucket.w >=
/// fitH * fitW` AND aspect-ratio L2 distance is minimised among ties.
///
/// If no bucket has enough pixels, returns the largest bucket (Phase 1
/// owners must add a bigger bucket — we log a warning at the call site).
(int, int, int) pickImageBucket({
  required int resizedHeight,
  required int resizedWidth,
  required List<(int, int, int)> buckets,
  int patchSize = 14,
}) {
  final fitH = resizedHeight ~/ patchSize;
  final fitW = resizedWidth ~/ patchSize;
  final fitArea = fitH * fitW;
  final fitAspect = fitH / fitW;

  (int, int, int)? best;
  double bestScore = double.infinity;
  for (final b in buckets) {
    final (_, bh, bw) = b;
    final bArea = bh * bw;
    if (bArea < fitArea) continue;
    final aspect = bh / bw;
    final aspectErr = (aspect - fitAspect).abs();
    // Primary: minimise area waste; secondary: aspect match.
    final score = (bArea - fitArea).toDouble() + aspectErr * 100.0;
    if (score < bestScore) {
      bestScore = score;
      best = b;
    }
  }
  if (best != null) return best;

  // Fallback: largest bucket by area.
  var largest = buckets.first;
  var largestArea = largest.$2 * largest.$3;
  for (final b in buckets.skip(1)) {
    final a = b.$2 * b.$3;
    if (a > largestArea) {
      largest = b;
      largestArea = a;
    }
  }
  return largest;
}
