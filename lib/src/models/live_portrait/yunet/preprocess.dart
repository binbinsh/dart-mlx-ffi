/// Preprocess for YuNet face detector.
///
/// YuNet expects:
///   * `[1, 3, 640, 640]` NCHW float32
///   * Pixel range **0..255** (no mean/std normalization — OpenCV's
///     reference impl just casts uint8 -> float32 directly).
///   * Channel order **BGR** (OpenCV native). Source caller hands us
///     RGB so we swap channels here.
///
/// Letterboxes the source image to fit within 640x640 preserving
/// aspect ratio, padding with black. Stores the inverse mapping on
/// [YuNetLetterbox.toSource] so detections can be remapped to source
/// pixel coords.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Preprocessed YuNet input tensor + reverse mapping.
final class YuNetPreprocessed {
  const YuNetPreprocessed({required this.nchw, required this.letterbox});

  /// Float32 NCHW data, length = 3 * 640 * 640.
  final Float32List nchw;
  final YuNetLetterbox letterbox;
}

/// Records the letterbox geometry so detector outputs (in 0..640 input
/// pixels) can be mapped back to source pixel coordinates.
final class YuNetLetterbox {
  const YuNetLetterbox({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.sourceWidth,
    required this.sourceHeight,
  });

  /// Multiplier applied to source pixels to get input-space pixels.
  /// `inputPx = sourcePx * scale + pad`.
  final double scale;
  final double padX;
  final double padY;
  final int sourceWidth;
  final int sourceHeight;

  /// Map an input-pixel coordinate (0..640) to source-pixel coords.
  ({double x, double y}) toSource(double xPx, double yPx) {
    return (x: (xPx - padX) / scale, y: (yPx - padY) / scale);
  }
}

const int kYuNetInputSize = 640;

YuNetPreprocessed preprocessForYuNet({
  required Uint8List sourceRgb,
  required int sourceWidth,
  required int sourceHeight,
}) {
  if (sourceRgb.length != sourceWidth * sourceHeight * 3) {
    throw ArgumentError(
      'sourceRgb length ${sourceRgb.length} does not match '
      '${sourceWidth}x${sourceHeight}x3',
    );
  }
  const size = kYuNetInputSize;
  final scale = math.min(size / sourceWidth, size / sourceHeight);
  final fitW = (sourceWidth * scale).round();
  final fitH = (sourceHeight * scale).round();
  final padX = ((size - fitW) / 2).floor();
  final padY = ((size - fitH) / 2).floor();
  // NCHW layout: 3 planes of size*size each.
  final out = Float32List(3 * size * size);
  final planeStride = size * size;
  for (var y = 0; y < size; y++) {
    final fy = y - padY;
    for (var x = 0; x < size; x++) {
      final fx = x - padX;
      final outOffset = y * size + x;
      if (fx < 0 || fy < 0 || fx >= fitW || fy >= fitH) {
        out[outOffset] = 0; // B
        out[planeStride + outOffset] = 0; // G
        out[2 * planeStride + outOffset] = 0; // R
        continue;
      }
      final sx = (fx / scale).floor().clamp(0, sourceWidth - 1);
      final sy = (fy / scale).floor().clamp(0, sourceHeight - 1);
      final srcIdx = (sy * sourceWidth + sx) * 3;
      // Source is RGB; YuNet wants BGR.
      out[outOffset] = sourceRgb[srcIdx + 2].toDouble(); // B
      out[planeStride + outOffset] = sourceRgb[srcIdx + 1].toDouble(); // G
      out[2 * planeStride + outOffset] = sourceRgb[srcIdx].toDouble(); // R
    }
  }
  return YuNetPreprocessed(
    nchw: out,
    letterbox: YuNetLetterbox(
      scale: scale,
      padX: padX.toDouble(),
      padY: padY.toDouble(),
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
    ),
  );
}
