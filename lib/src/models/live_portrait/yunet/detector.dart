/// YuNet face detector — anchor-free FCOS-style decoder + NMS.
///
/// YuNet (face_detection_yunet_2023mar.onnx) is a 232 KB single-stage
/// face detector trained by OpenCV. Compared to MediaPipe BlazeFace
/// short-range, it locks onto faces across a much wider range of
/// scales — including faces that are only ~10% of the input frame
/// (which is what we hit on full-body buddy portraits).
///
/// ## I/O contract
///
/// Input: `[1, 3, 640, 640]` NCHW float32, **BGR** channel order,
/// pixel range 0..255 (no normalization).
///
/// Outputs: 12 tensors across 3 strides s in {8, 16, 32}. For each
/// stride the grid is `640/s` cells wide and tall:
///   * `cls_s   [1, N, 1]`  — face classification logit
///   * `obj_s   [1, N, 1]`  — objectness logit
///   * `bbox_s  [1, N, 4]`  — `[dx, dy, log_w, log_h]` in stride units
///   * `kps_s   [1, N, 10]` — 5 keypoints `(x, y)` in stride units
///
/// where `N = (640/s)^2` (6400, 1600, 400). The grid cell at index
/// `i = gy*W + gx` decodes as:
///
///   center_x = (gx + bbox[0]) * s
///   center_y = (gy + bbox[1]) * s
///   width    = exp(bbox[2]) * s
///   height   = exp(bbox[3]) * s
///   kp_k.x   = (gx + kps[2k]) * s
///   kp_k.y   = (gy + kps[2k+1]) * s
///   score    = sigmoid(cls) * sigmoid(obj)
///
/// Keypoint order (from OpenCV reference impl):
///   0 = right eye, 1 = left eye, 2 = nose tip,
///   3 = right mouth corner, 4 = left mouth corner.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

import 'preprocess.dart';

/// One detected face in *source-image pixel* coordinates.
final class YuNetDetection {
  const YuNetDetection({
    required this.score,
    required this.bbox,
    required this.keypoints,
  });

  /// Combined sigmoid(cls)*sigmoid(obj) score in [0,1].
  final double score;

  /// (x0, y0, x1, y1) in source-image pixel coords.
  final ({double x0, double y0, double x1, double y1}) bbox;

  /// 5 keypoints in source-image pixel coords. See library comment for
  /// ordering.
  final List<({double x, double y})> keypoints;
}

/// Strides used by YuNet 2023mar. Order matches OpenCV reference impl.
const List<int> _kStrides = [8, 16, 32];

final class YuNetDetector {
  YuNetDetector._({
    required DartOnnxSession session,
    required String inputName,
    required Map<int, _StrideOutputs> outputsByStride,
  }) : _session = session,
       _inputName = inputName,
       _outputsByStride = outputsByStride;

  factory YuNetDetector.load({
    required String onnxPath,
    int numThreads = 2,
  }) {
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: onnxPath,
        id: 'live_portrait_yunet',
        family: 'face_detection',
        provider: 'cpu',
        requireProvider: false,
        numThreads: numThreads,
      ),
    );
    final diag = session.diagnostics;
    final inputs = (diag['input_metadata'] as List).cast<Map>();
    final outputs = (diag['output_metadata'] as List).cast<Map>();
    if (inputs.length != 1) {
      session.close();
      throw StateError(
        'YuNet ONNX expected 1 input, got ${inputs.length}: '
        '${inputs.map((m) => m['name']).toList()}',
      );
    }
    final inputName = inputs.first['name'] as String;
    final outputNames = outputs
        .map((m) => m['name'] as String)
        .toSet();
    final byStride = <int, _StrideOutputs>{};
    for (final s in _kStrides) {
      final cls = 'cls_$s';
      final obj = 'obj_$s';
      final bbox = 'bbox_$s';
      final kps = 'kps_$s';
      for (final required in [cls, obj, bbox, kps]) {
        if (!outputNames.contains(required)) {
          session.close();
          throw StateError(
            'YuNet ONNX missing expected output "$required". '
            'Got outputs: ${outputNames.toList()}',
          );
        }
      }
      byStride[s] = _StrideOutputs(
        clsName: cls,
        objName: obj,
        bboxName: bbox,
        kpsName: kps,
      );
    }
    return YuNetDetector._(
      session: session,
      inputName: inputName,
      outputsByStride: byStride,
    );
  }

  final DartOnnxSession _session;
  final String _inputName;
  final Map<int, _StrideOutputs> _outputsByStride;

  /// Detect faces in [sourceRgb], returning detections sorted by
  /// descending score after NMS. Pass [topK] = 1 if you only need the
  /// dominant face.
  List<YuNetDetection> detect({
    required Uint8List sourceRgb,
    required int sourceWidth,
    required int sourceHeight,
    double scoreThreshold = 0.6,
    double iouThreshold = 0.3,
    int topK = 5,
  }) {
    final pre = preprocessForYuNet(
      sourceRgb: sourceRgb,
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
    );
    final inputTensor = RuntimeTensor.float32(
      [1, 3, kYuNetInputSize, kYuNetInputSize],
      pre.nchw,
    );
    final result = _session.run({_inputName: inputTensor});
    final List<YuNetDetection> detections;
    try {
      detections = <YuNetDetection>[];
      for (final entry in _outputsByStride.entries) {
        final stride = entry.key;
        final names = entry.value;
        final cls = (result.outputs[names.clsName] as RuntimeTensor)
            .asFloat32List();
        final obj = (result.outputs[names.objName] as RuntimeTensor)
            .asFloat32List();
        final bbox = (result.outputs[names.bboxName] as RuntimeTensor)
            .asFloat32List();
        final kps = (result.outputs[names.kpsName] as RuntimeTensor)
            .asFloat32List();
        _decodeStride(
          stride: stride,
          cls: cls,
          obj: obj,
          bbox: bbox,
          kps: kps,
          letterbox: pre.letterbox,
          scoreThreshold: scoreThreshold,
          out: detections,
        );
      }
    } finally {
      result.close();
    }
    detections.sort((a, b) => b.score.compareTo(a.score));
    final kept = _nms(detections, iouThreshold: iouThreshold);
    return kept.take(topK).toList(growable: false);
  }

  /// Free the underlying ORT session. The detector is unusable after.
  void close() => _session.close();

  void _decodeStride({
    required int stride,
    required Float32List cls,
    required Float32List obj,
    required Float32List bbox,
    required Float32List kps,
    required YuNetLetterbox letterbox,
    required double scoreThreshold,
    required List<YuNetDetection> out,
  }) {
    final gridW = kYuNetInputSize ~/ stride;
    final gridH = kYuNetInputSize ~/ stride;
    final n = gridW * gridH;
    if (cls.length != n) {
      throw StateError(
        'YuNet stride $stride: cls length ${cls.length} != expected $n '
        '(grid ${gridW}x$gridH)',
      );
    }
    for (var i = 0; i < n; i++) {
      final clsScore = _sigmoid(cls[i]);
      final objScore = _sigmoid(obj[i]);
      final score = math.sqrt(clsScore * objScore); // OpenCV uses geo mean
      if (score < scoreThreshold) continue;
      final gy = i ~/ gridW;
      final gx = i % gridW;
      final bx = bbox[i * 4 + 0];
      final by = bbox[i * 4 + 1];
      final bw = bbox[i * 4 + 2];
      final bh = bbox[i * 4 + 3];
      final cxPx = (gx + bx) * stride;
      final cyPx = (gy + by) * stride;
      final wPx = math.exp(bw) * stride;
      final hPx = math.exp(bh) * stride;
      final x0Px = cxPx - wPx / 2;
      final y0Px = cyPx - hPx / 2;
      final x1Px = cxPx + wPx / 2;
      final y1Px = cyPx + hPx / 2;
      final srcTL = letterbox.toSource(x0Px, y0Px);
      final srcBR = letterbox.toSource(x1Px, y1Px);
      final keypoints = <({double x, double y})>[];
      for (var k = 0; k < 5; k++) {
        final kxPx = (gx + kps[i * 10 + k * 2 + 0]) * stride;
        final kyPx = (gy + kps[i * 10 + k * 2 + 1]) * stride;
        keypoints.add(letterbox.toSource(kxPx, kyPx));
      }
      out.add(
        YuNetDetection(
          score: score,
          bbox: (x0: srcTL.x, y0: srcTL.y, x1: srcBR.x, y1: srcBR.y),
          keypoints: keypoints,
        ),
      );
    }
  }

  static double _sigmoid(double x) {
    final c = x.clamp(-100.0, 100.0);
    return 1.0 / (1.0 + math.exp(-c));
  }

  List<YuNetDetection> _nms(
    List<YuNetDetection> sorted, {
    required double iouThreshold,
  }) {
    final kept = <YuNetDetection>[];
    final suppressed = List<bool>.filled(sorted.length, false);
    for (var i = 0; i < sorted.length; i++) {
      if (suppressed[i]) continue;
      kept.add(sorted[i]);
      for (var j = i + 1; j < sorted.length; j++) {
        if (suppressed[j]) continue;
        if (_iou(sorted[i].bbox, sorted[j].bbox) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  static double _iou(
    ({double x0, double y0, double x1, double y1}) a,
    ({double x0, double y0, double x1, double y1}) b,
  ) {
    final ix0 = math.max(a.x0, b.x0);
    final iy0 = math.max(a.y0, b.y0);
    final ix1 = math.min(a.x1, b.x1);
    final iy1 = math.min(a.y1, b.y1);
    final iw = math.max(0.0, ix1 - ix0);
    final ih = math.max(0.0, iy1 - iy0);
    final inter = iw * ih;
    final ua = (a.x1 - a.x0) * (a.y1 - a.y0);
    final ub = (b.x1 - b.x0) * (b.y1 - b.y0);
    final union = ua + ub - inter;
    return union <= 0 ? 0 : inter / union;
  }
}

class _StrideOutputs {
  const _StrideOutputs({
    required this.clsName,
    required this.objName,
    required this.bboxName,
    required this.kpsName,
  });
  final String clsName;
  final String objName;
  final String bboxName;
  final String kpsName;
}
