/// Image preprocessing for the LivePortrait extractors.
///
/// Both `appearance_extractor.onnx` and `motion_extractor.onnx`
/// expect input shape `[1, 3, 256, 256]` float32 in the [0, 1] range
/// (Ditto's reference impl: `image / 255.0`, no further mean/std).
/// Layout is NCHW with channel order **RGB** (matches Ditto's
/// `cv2.cvtColor(BGR2RGB)` in `face_align.py`).
///
/// Input from face_crop is a packed RGB uint8 buffer at 512×512;
/// we downsample to 256×256 with bilinear filtering. (The face crop
/// itself is 4-DOF aligned, so any upstream rotation is already
/// removed and the downsample is plain.)
library;

import 'dart:typed_data';

/// Side length expected by both extractor ONNX inputs.
const int kExtractorInputSize = 256;

/// Downsample a packed [Uint8List] RGB image of size
/// [sourceSize]×[sourceSize] to NCHW float32 RGB at
/// [kExtractorInputSize]×[kExtractorInputSize], normalised by 1/255.
///
/// Bilinear filter; takes ~1.2 ms on a single thread for the canonical
/// 512→256 case. Allocates a single output [Float32List] of length
/// `3 * 256 * 256 = 196608`.
Float32List rgb512ToExtractorNchw({
  required Uint8List sourceRgb,
  int sourceSize = 512,
  int outSize = kExtractorInputSize,
}) {
  if (sourceRgb.length != sourceSize * sourceSize * 3) {
    throw ArgumentError(
      'rgb512ToExtractorNchw: sourceRgb length ${sourceRgb.length} '
      '!= ${sourceSize}x${sourceSize}x3',
    );
  }
  final out = Float32List(3 * outSize * outSize);
  final scale = sourceSize / outSize;
  // Channel strides in the NCHW output buffer.
  final planeR = 0;
  final planeG = outSize * outSize;
  final planeB = 2 * outSize * outSize;
  const inv255 = 1.0 / 255.0;
  final maxIdx = sourceSize - 1;
  for (var y = 0; y < outSize; y++) {
    final sy = (y + 0.5) * scale - 0.5;
    final y0 = sy.floor().clamp(0, maxIdx);
    final y1 = (y0 + 1 > maxIdx) ? y0 : y0 + 1;
    final fy = (sy - y0).clamp(0.0, 1.0);
    for (var x = 0; x < outSize; x++) {
      final sx = (x + 0.5) * scale - 0.5;
      final x0 = sx.floor().clamp(0, maxIdx);
      final x1 = (x0 + 1 > maxIdx) ? x0 : x0 + 1;
      final fx = (sx - x0).clamp(0.0, 1.0);
      final w00 = (1 - fx) * (1 - fy);
      final w01 = fx * (1 - fy);
      final w10 = (1 - fx) * fy;
      final w11 = fx * fy;
      final i00 = (y0 * sourceSize + x0) * 3;
      final i01 = (y0 * sourceSize + x1) * 3;
      final i10 = (y1 * sourceSize + x0) * 3;
      final i11 = (y1 * sourceSize + x1) * 3;
      final outIdx = y * outSize + x;
      // Channel 0 (R)
      final r =
          sourceRgb[i00] * w00 +
          sourceRgb[i01] * w01 +
          sourceRgb[i10] * w10 +
          sourceRgb[i11] * w11;
      // Channel 1 (G)
      final g =
          sourceRgb[i00 + 1] * w00 +
          sourceRgb[i01 + 1] * w01 +
          sourceRgb[i10 + 1] * w10 +
          sourceRgb[i11 + 1] * w11;
      // Channel 2 (B)
      final b =
          sourceRgb[i00 + 2] * w00 +
          sourceRgb[i01 + 2] * w01 +
          sourceRgb[i10 + 2] * w10 +
          sourceRgb[i11 + 2] * w11;
      out[planeR + outIdx] = r * inv255;
      out[planeG + outIdx] = g * inv255;
      out[planeB + outIdx] = b * inv255;
    }
  }
  return out;
}
