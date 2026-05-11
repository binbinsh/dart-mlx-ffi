/// Bilinear resampler — warps a source RGB image into a 512x512 crop
/// using a 2x3 affine that maps crop coordinates to source coordinates.
///
/// Per-output-pixel cost: 4 RGB texel reads + 12 float multiplies. At
/// 512x512 = 262144 pixels this is ~3M float ops, well under 5 ms on
/// a single thread. No need for SIMD or threading at v1.
library;

import 'dart:typed_data';

import 'similarity.dart';

/// Warp [sourceRgb] into a [cropSize]x[cropSize] RGB buffer using
/// the affine [cropToSource] (which maps crop pixel coords -> source
/// pixel coords). Out-of-bounds reads return black.
Uint8List warpRgbBilinear({
  required Uint8List sourceRgb,
  required int sourceWidth,
  required int sourceHeight,
  required Affine2x3 cropToSource,
  int cropSize = 512,
}) {
  if (sourceRgb.length != sourceWidth * sourceHeight * 3) {
    throw ArgumentError(
      'sourceRgb length ${sourceRgb.length} does not match '
      '${sourceWidth}x${sourceHeight}x3',
    );
  }
  final out = Uint8List(cropSize * cropSize * 3);
  final m0 = cropToSource[0];
  final m1 = cropToSource[1];
  final m2 = cropToSource[2];
  final m3 = cropToSource[3];
  final m4 = cropToSource[4];
  final m5 = cropToSource[5];
  final maxX = sourceWidth - 1;
  final maxY = sourceHeight - 1;
  for (var y = 0; y < cropSize; y++) {
    for (var x = 0; x < cropSize; x++) {
      final sx = m0 * x + m1 * y + m2;
      final sy = m3 * x + m4 * y + m5;
      final outIdx = (y * cropSize + x) * 3;
      if (sx < 0 || sy < 0 || sx > maxX || sy > maxY) {
        out[outIdx + 0] = 0;
        out[outIdx + 1] = 0;
        out[outIdx + 2] = 0;
        continue;
      }
      final x0 = sx.floor();
      final y0 = sy.floor();
      final x1 = x0 + 1 > maxX ? x0 : x0 + 1;
      final y1 = y0 + 1 > maxY ? y0 : y0 + 1;
      final fx = sx - x0;
      final fy = sy - y0;
      final w00 = (1 - fx) * (1 - fy);
      final w01 = fx * (1 - fy);
      final w10 = (1 - fx) * fy;
      final w11 = fx * fy;
      final i00 = (y0 * sourceWidth + x0) * 3;
      final i01 = (y0 * sourceWidth + x1) * 3;
      final i10 = (y1 * sourceWidth + x0) * 3;
      final i11 = (y1 * sourceWidth + x1) * 3;
      for (var c = 0; c < 3; c++) {
        final v =
            sourceRgb[i00 + c] * w00 +
            sourceRgb[i01 + c] * w01 +
            sourceRgb[i10 + c] * w10 +
            sourceRgb[i11 + c] * w11;
        out[outIdx + c] = v.round().clamp(0, 255);
      }
    }
  }
  return out;
}
