/// 5-point similarity (Umeyama) transform for face alignment.
///
/// Given a source image with 5 detected face landmarks (left/right
/// eyes, nose, left/right mouth corners), compute the affine that
/// maps the canonical 512x512 LivePortrait crop coordinates to source
/// image coordinates. The inverse maps source -> crop and is what the
/// resampler walks per output pixel.
///
/// "Similarity" = uniform scale + rotation + translation only, no
/// shear. Encodes a 4-DOF transform with 2x3 matrix `[a, -b, tx;
/// b, a, ty]`. Robust against landmark noise compared to general
/// affine.
library;

import 'dart:typed_data';

/// Canonical 5-point template in 512x512 crop space. Eyes on the
/// upper-third horizontal line, mouth slightly above the lower-third.
/// These are the standard ArcFace template points scaled from 112x112
/// to 512x512.
///
/// Convention is **viewer perspective** (a.k.a. the layout you see in
/// the rendered image): index 0 is on the viewer's LEFT side (lower
/// x), which corresponds to the SUBJECT's RIGHT eye. Matches both the
/// InsightFace reference template and the YuNet keypoint order, so we
/// can pass YuNet's 5 points straight through without re-ordering.
const List<({double x, double y})> kLivePortraitCanonical512 = [
  (x: 168.0, y: 217.6), // viewer-left eye  = subject right eye
  (x: 344.0, y: 217.6), // viewer-right eye = subject left eye
  (x: 256.0, y: 314.6), // nose tip
  (x: 187.6, y: 410.4), // viewer-left mouth corner
  (x: 324.4, y: 410.4), // viewer-right mouth corner
];

/// 2x3 row-major affine: `[m0 m1 m2; m3 m4 m5]`.
typedef Affine2x3 = Float32List;

/// Solve the similarity transform mapping [from] -> [to] (least
/// squares, closed form Umeyama 1991). Returns a 2x3 matrix that, when
/// applied to a homogeneous (x,y,1) `from` point, produces the `to`
/// point.
Affine2x3 solveSimilarity({
  required List<({double x, double y})> from,
  required List<({double x, double y})> to,
}) {
  if (from.length != to.length || from.length < 2) {
    throw ArgumentError(
      'solveSimilarity needs same-length point lists with >= 2 points; '
      'got from=${from.length}, to=${to.length}',
    );
  }
  final n = from.length;
  // Centroids.
  var fx = 0.0, fy = 0.0, tx = 0.0, ty = 0.0;
  for (var i = 0; i < n; i++) {
    fx += from[i].x;
    fy += from[i].y;
    tx += to[i].x;
    ty += to[i].y;
  }
  fx /= n;
  fy /= n;
  tx /= n;
  ty /= n;
  // Cross-covariance terms + source variance.
  var sxy = 0.0; // sum of (fx_c * ty_c - fy_c * tx_c)
  var sxx = 0.0; // sum of (fx_c * tx_c + fy_c * ty_c)
  var sff = 0.0; // sum of (fx_c^2 + fy_c^2)
  for (var i = 0; i < n; i++) {
    final dfx = from[i].x - fx;
    final dfy = from[i].y - fy;
    final dtx = to[i].x - tx;
    final dty = to[i].y - ty;
    sxx += dfx * dtx + dfy * dty;
    sxy += dfx * dty - dfy * dtx;
    sff += dfx * dfx + dfy * dfy;
  }
  if (sff <= 1e-12) {
    throw StateError(
      'solveSimilarity: source points are degenerate (zero variance)',
    );
  }
  // For similarity transform: a = sxx/sff, b = sxy/sff.
  final a = sxx / sff;
  final b = sxy / sff;
  // Translation so centroids align after rotation/scale.
  final txOut = tx - (a * fx - b * fy);
  final tyOut = ty - (b * fx + a * fy);
  return Float32List.fromList([a, -b, txOut, b, a, tyOut]);
}

/// Invert a 2x3 affine. Throws if the linear part is singular.
Affine2x3 invertAffine(Affine2x3 m) {
  final det = m[0] * m[4] - m[1] * m[3];
  if (det.abs() < 1e-12) {
    throw StateError('invertAffine: singular matrix');
  }
  final inv00 = m[4] / det;
  final inv01 = -m[1] / det;
  final inv10 = -m[3] / det;
  final inv11 = m[0] / det;
  final invTx = -(inv00 * m[2] + inv01 * m[5]);
  final invTy = -(inv10 * m[2] + inv11 * m[5]);
  return Float32List.fromList([inv00, inv01, invTx, inv10, inv11, invTy]);
}

/// Apply 2x3 affine to a single (x, y) point.
({double x, double y}) applyAffine(Affine2x3 m, double x, double y) {
  return (x: m[0] * x + m[1] * y + m[2], y: m[3] * x + m[4] * y + m[5]);
}

/// Pass YuNet's 5 keypoints through unchanged — its output order
/// already matches the LivePortrait/ArcFace template.
///
/// YuNet output (per OpenCV's `face_detection_yunet/demo.py`):
///   0 = subject right eye  (viewer's left,  lower x)
///   1 = subject left eye   (viewer's right, higher x)
///   2 = nose tip
///   3 = subject right mouth corner
///   4 = subject left mouth corner
///
/// LivePortrait/ArcFace template (`kLivePortraitCanonical512`) uses
/// the same viewer-perspective ordering, so no re-shuffle is needed.
/// We keep this helper around so callers don't have to know the
/// detector convention; it also serves as a single place to validate
/// length and adjust if we ever swap detectors.
List<({double x, double y})> yunetKeypointsToTemplate5(
  List<({double x, double y})> yunetPoints,
) {
  if (yunetPoints.length != 5) {
    throw ArgumentError(
      'YuNet keypoints must have exactly 5 points; '
      'got ${yunetPoints.length}',
    );
  }
  return yunetPoints;
}
