/// Face crop & alignment for LivePortrait source portraits.
///
/// **Phase 1 implementation.** Replaces InsightFace's `det_10g.onnx` +
/// `2d106det.onnx` (research-only license) with **YuNet**
/// (`face_detection_yunet_2023mar.onnx`, OpenCV Apache-2.0). YuNet
/// handles the small-face-in-large-frame regime that BlazeFace
/// short-range can't (e.g. our full-body buddy portraits where the
/// face is only ~12% of the source frame).
///
/// Pipeline per source portrait:
///
///   1. Letterbox source RGB into 640x640 NCHW float32 (BGR, 0..255).
///   2. Run YuNet ONNX -> 12 outputs (cls/obj/bbox/kps × 3 strides).
///   3. Decode FCOS-style + NMS.
///   4. Reorder YuNet's 5 keypoints into LivePortrait template order.
///   5. Solve Umeyama similarity: source 5pts -> canonical 5pts.
///   6. Bilinear-warp source -> 512x512 crop using the inverse.
///
/// The forward (crop -> source) affine is preserved on
/// [FaceCropResult.toSourceTransform] so `Engine.paste` can blend the
/// generated face back onto the original frame.
///
/// Implementation is split across `yunet/` for readability:
///   * `preprocess.dart` -- letterbox + NCHW + BGR conversion.
///   * `detector.dart` -- ORT session + decode + NMS.
///   * `similarity.dart` -- 5-point Umeyama solver, canonical template.
///   * `resample.dart` -- bilinear warp.
library;

import 'dart:typed_data';

import 'yunet/detector.dart';
import 'yunet/resample.dart';
import 'yunet/similarity.dart';

/// 5-point landmark set in source-image pixel coordinates.
///
/// Order matches the LivePortrait/ArcFace template (viewer
/// perspective): `leftEye` is on the viewer's left side of the image
/// = the subject's right eye.
final class FaceLandmarks5 {
  const FaceLandmarks5({
    required this.leftEye,
    required this.rightEye,
    required this.noseTip,
    required this.leftMouth,
    required this.rightMouth,
  });

  final ({double x, double y}) leftEye;
  final ({double x, double y}) rightEye;
  final ({double x, double y}) noseTip;
  final ({double x, double y}) leftMouth;
  final ({double x, double y}) rightMouth;
}

/// Affine transform from canonical 512x512 face crop coords to source
/// image coords. Stored as `[a, b, tx, c, d, ty]` (2x3 row-major).
final class FaceCropTransform {
  const FaceCropTransform(this.matrix);
  final Float32List matrix;
}

/// Result of a single source-portrait crop operation.
final class FaceCropResult {
  const FaceCropResult({
    required this.cropRgb,
    required this.cropWidth,
    required this.cropHeight,
    required this.landmarks,
    required this.toSourceTransform,
    required this.detectionScore,
  });

  /// 512x512 RGB pixels, packed RGBRGB... uint8.
  final Uint8List cropRgb;
  final int cropWidth;
  final int cropHeight;
  final FaceLandmarks5 landmarks;

  /// Forward affine: crop pixel -> source pixel. Inverse of the matrix
  /// the resampler used internally.
  final FaceCropTransform toSourceTransform;

  /// Detection confidence in [0,1] for the chosen face.
  final double detectionScore;
}

/// Stateless face-crop service. One instance loads the detector once
/// and is shared across all portrait loads.
abstract class FaceCropService {
  /// Detect the dominant face in [sourceRgb] and produce a 512x512 crop
  /// aligned to LivePortrait's canonical eye/mouth template.
  ///
  /// Throws [StateError] when no face is detected -- caller should
  /// surface this as "pick a different portrait" UX.
  FaceCropResult cropPortrait({
    required Uint8List sourceRgb,
    required int sourceWidth,
    required int sourceHeight,
  });

  /// Construct the YuNet-backed default. Loads ONNX immediately;
  /// the returned service holds an open ORT session until [close] is
  /// called.
  factory FaceCropService.yunet({required String onnxPath}) =>
      _YuNetCropService.load(onnxPath);

  /// Free underlying ML session(s).
  void close();
}

class _YuNetCropService implements FaceCropService {
  _YuNetCropService._(this._detector);

  factory _YuNetCropService.load(String onnxPath) {
    final detector = YuNetDetector.load(onnxPath: onnxPath);
    return _YuNetCropService._(detector);
  }

  final YuNetDetector _detector;

  @override
  FaceCropResult cropPortrait({
    required Uint8List sourceRgb,
    required int sourceWidth,
    required int sourceHeight,
  }) {
    final detections = _detector.detect(
      sourceRgb: sourceRgb,
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
      topK: 1,
    );
    if (detections.isEmpty) {
      throw StateError(
        'YuNet: no face detected ($sourceWidth x $sourceHeight '
        'source). Try a clearer portrait.',
      );
    }
    final det = detections.first;
    // YuNet already gives 5 keypoints; just reorder to LivePortrait's
    // template (eyes, nose, mouth corners).
    final fivePts = yunetKeypointsToTemplate5(det.keypoints);
    // Solve source -> canonical (this is the "crop_to_source" inverse).
    final sourceToCrop = solveSimilarity(
      from: fivePts,
      to: kLivePortraitCanonical512,
    );
    // The resampler walks the output 512x512 grid and reads source
    // pixels -- it needs the crop -> source direction.
    final cropToSource = invertAffine(sourceToCrop);
    final cropRgb = warpRgbBilinear(
      sourceRgb: sourceRgb,
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
      cropToSource: cropToSource,
      cropSize: 512,
    );
    return FaceCropResult(
      cropRgb: cropRgb,
      cropWidth: 512,
      cropHeight: 512,
      landmarks: FaceLandmarks5(
        leftEye: fivePts[0],
        rightEye: fivePts[1],
        noseTip: fivePts[2],
        leftMouth: fivePts[3],
        rightMouth: fivePts[4],
      ),
      toSourceTransform: FaceCropTransform(cropToSource),
      detectionScore: det.score,
    );
  }

  @override
  void close() => _detector.close();
}
