/// Motion Extractor (`motion_extractor.onnx`).
///
/// Single-shot keypoint + head-pose + expression extraction from a
/// 256×256 face crop. Backbone is ConvNeXt-V2-Tiny; outputs 7 tensors.
///
/// ## Output shapes
///
///   * `pitch`  `[1, 66]`  bin distribution, range -99° .. +99°, step 3°
///   * `yaw`    `[1, 66]`  same
///   * `roll`   `[1, 66]`  same
///   * `t`      `[1, 3]`   head translation (tx, ty, tz)
///   * `exp`    `[1, 63]`  expression delta, reshaped as [21, 3]
///   * `scale`  `[1, 1]`   global scale scalar
///   * `kp`     `[1, 63]`  canonical keypoints, reshaped as [21, 3]
///
/// We expose the parsed Euler angles in degrees + the rotation matrix
/// + the canonical keypoints + scale + translation + expression as
/// [MotionDescriptor]. The bin66 distributions are also kept around in
/// case downstream code wants to ctrl-bias them (see Ditto's
/// `ctrl_motion`/`bin66_to_degree`).
///
/// ## Soft-argmax
///
/// Mirrors Ditto `bin66_to_degree`:
///
/// ```
/// degree = sum(softmax(pred) * arange(66)) * 3 - 97.5
/// ```
///
/// ## Rotation matrix
///
/// Mirrors Ditto `get_rotation_matrix`:
///
/// ```
/// R = (Rz @ Ry @ Rx).T
/// ```
///
/// where Rx/Ry/Rz are the standard right-handed rotation matrices for
/// pitch/yaw/roll (in radians). The transpose at the end is critical —
/// Ditto's keypoint update is `kp @ R` not `R @ kp`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

const String _kMotionFamily = 'live_portrait_motion';
const int _kPoseBins = 66;

/// Parsed motion-extractor output. All matrices are row-major Float32.
final class MotionDescriptor {
  const MotionDescriptor({
    required this.pitchDeg,
    required this.yawDeg,
    required this.rollDeg,
    required this.rotation,
    required this.translation,
    required this.expression,
    required this.scale,
    required this.canonicalKeypoints,
    required this.pitchBins,
    required this.yawBins,
    required this.rollBins,
  });

  /// Euler angles in degrees (post soft-argmax).
  final double pitchDeg;
  final double yawDeg;
  final double rollDeg;

  /// 3×3 rotation matrix, row-major, length 9.
  /// `R = (Rz @ Ry @ Rx).T` per Ditto convention.
  final Float32List rotation;

  /// Length-3 translation `(tx, ty, tz)`.
  final Float32List translation;

  /// `[21, 3]` flattened expression delta, length 63.
  final Float32List expression;

  /// Scalar global scale.
  final double scale;

  /// `[21, 3]` flattened canonical keypoints, length 63.
  final Float32List canonicalKeypoints;

  /// Raw 66-bin distributions (softmax not applied — kept as-is from
  /// the network for callers that want to manipulate them directly).
  final Float32List pitchBins;
  final Float32List yawBins;
  final Float32List rollBins;
}

/// Wraps the ORT session for `motion_extractor.onnx`.
final class MotionExtractor {
  MotionExtractor._({
    required DartOnnxSession session,
    required String inputName,
  }) : _session = session,
       _inputName = inputName;

  factory MotionExtractor.load({
    required String onnxPath,
    int numThreads = 2,
  }) {
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: onnxPath,
        id: _kMotionFamily,
        family: _kMotionFamily,
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
        'MotionExtractor: expected 1 input, got ${inputs.length}',
      );
    }
    final outNames = outputs.map((m) => m['name'] as String).toSet();
    const expected = {'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'};
    final missing = expected.difference(outNames);
    if (missing.isNotEmpty) {
      session.close();
      throw StateError(
        'MotionExtractor: ONNX missing expected outputs $missing. '
        'Got: $outNames',
      );
    }
    return MotionExtractor._(
      session: session,
      inputName: inputs.first['name'] as String,
    );
  }

  final DartOnnxSession _session;
  final String _inputName;

  /// Run the extractor on a 256×256 RGB NCHW float32 crop.
  MotionDescriptor extract(Float32List crop256Nchw) {
    if (crop256Nchw.length != 3 * 256 * 256) {
      throw ArgumentError(
        'MotionExtractor.extract: input length ${crop256Nchw.length} '
        '!= 3*256*256',
      );
    }
    final inputTensor = RuntimeTensor.float32(
      [1, 3, 256, 256],
      crop256Nchw,
    );
    final result = _session.run({_inputName: inputTensor});
    try {
      final pitch = _read(result, 'pitch', _kPoseBins);
      final yaw = _read(result, 'yaw', _kPoseBins);
      final roll = _read(result, 'roll', _kPoseBins);
      final t = _read(result, 't', 3);
      final exp = _read(result, 'exp', 63);
      final scale = _read(result, 'scale', 1);
      final kp = _read(result, 'kp', 63);

      final pitchDeg = bin66ToDegree(pitch);
      final yawDeg = bin66ToDegree(yaw);
      final rollDeg = bin66ToDegree(roll);
      final rotation = ditttoRotationMatrix(
        pitchDeg: pitchDeg,
        yawDeg: yawDeg,
        rollDeg: rollDeg,
      );
      return MotionDescriptor(
        pitchDeg: pitchDeg,
        yawDeg: yawDeg,
        rollDeg: rollDeg,
        rotation: rotation,
        translation: t,
        expression: exp,
        scale: scale[0],
        canonicalKeypoints: kp,
        pitchBins: pitch,
        yawBins: yaw,
        rollBins: roll,
      );
    } finally {
      result.close();
    }
  }

  Float32List _read(DartOnnxResult r, String name, int expectedLen) {
    final tensor = r.outputs[name] as RuntimeTensor;
    final flat = tensor.asFloat32List();
    if (flat.length != expectedLen) {
      throw StateError(
        'MotionExtractor: output "$name" length ${flat.length} '
        '!= expected $expectedLen',
      );
    }
    return Float32List.fromList(flat);
  }

  void close() => _session.close();
}

/// Soft-argmax over a 66-bin pose distribution. Matches Ditto's
/// `bin66_to_degree` exactly.
///
/// ```
/// degree = sum(softmax(pred) * arange(66)) * 3 - 97.5
/// ```
double bin66ToDegree(Float32List pred) {
  if (pred.length != _kPoseBins) {
    throw ArgumentError(
      'bin66ToDegree expects length 66; got ${pred.length}',
    );
  }
  // Stable softmax.
  var maxVal = pred[0];
  for (var i = 1; i < _kPoseBins; i++) {
    if (pred[i] > maxVal) maxVal = pred[i];
  }
  var sum = 0.0;
  final exp = Float32List(_kPoseBins);
  for (var i = 0; i < _kPoseBins; i++) {
    final e = math.exp(pred[i] - maxVal);
    exp[i] = e;
    sum += e;
  }
  var weighted = 0.0;
  for (var i = 0; i < _kPoseBins; i++) {
    weighted += (exp[i] / sum) * i;
  }
  return weighted * 3.0 - 97.5;
}

/// Build a 3×3 rotation matrix from Euler angles in degrees.
///
/// Mirrors Ditto's `get_rotation_matrix`: produces `(Rz @ Ry @ Rx).T`.
/// Returned as row-major length-9 `Float32List`.
Float32List ditttoRotationMatrix({
  required double pitchDeg,
  required double yawDeg,
  required double rollDeg,
}) {
  final px = pitchDeg * math.pi / 180.0;
  final py = yawDeg * math.pi / 180.0;
  final pz = rollDeg * math.pi / 180.0;
  final cx = math.cos(px), sx = math.sin(px);
  final cy = math.cos(py), sy = math.sin(py);
  final cz = math.cos(pz), sz = math.sin(pz);

  // Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
  // Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
  // Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
  // M = Rz @ Ry @ Rx; out = M.T

  // Compute M = Rz @ Ry @ Rx, row-major 3x3.
  // First Ry @ Rx:
  //   ry_rx[0] = [cy,        sy*sx,         sy*cx       ]
  //   ry_rx[1] = [0,         cx,            -sx         ]
  //   ry_rx[2] = [-sy,       cy*sx,         cy*cx       ]
  final m00 = cz * cy + (-sz) * 0.0 + 0.0 * (-sy);
  final m01 = cz * (sy * sx) + (-sz) * cx + 0.0 * (cy * sx);
  final m02 = cz * (sy * cx) + (-sz) * (-sx) + 0.0 * (cy * cx);
  final m10 = sz * cy + cz * 0.0 + 0.0 * (-sy);
  final m11 = sz * (sy * sx) + cz * cx + 0.0 * (cy * sx);
  final m12 = sz * (sy * cx) + cz * (-sx) + 0.0 * (cy * cx);
  final m20 = 0.0 * cy + 0.0 * 0.0 + 1.0 * (-sy);
  final m21 = 0.0 * (sy * sx) + 0.0 * cx + 1.0 * (cy * sx);
  final m22 = 0.0 * (sy * cx) + 0.0 * (-sx) + 1.0 * (cy * cx);

  // Transpose and pack row-major.
  return Float32List.fromList([
    m00, m10, m20,
    m01, m11, m21,
    m02, m12, m22,
  ]);
}

/// Transform canonical keypoints by pose + expression + scale + translation.
///
/// Mirrors Ditto's `transform_keypoint`:
///   kp_xfm = (kp.reshape(K,3) @ R) + exp.reshape(K,3)
///   kp_xfm *= scale
///   kp_xfm[:, 0:2] += t[0:2]
///
/// Inputs:
///   * [canonicalKp]  flat `[K, 3]` length 3K
///   * [rotation]     row-major 3×3 length 9
///   * [expression]   flat `[K, 3]` length 3K
///   * [scale]        scalar
///   * [translation]  length 3
///
/// Returns flat `[K, 3]` length 3K transformed keypoints.
Float32List transformKeypoints({
  required Float32List canonicalKp,
  required Float32List rotation,
  required Float32List expression,
  required double scale,
  required Float32List translation,
}) {
  if (canonicalKp.length != expression.length) {
    throw ArgumentError(
      'transformKeypoints: canonicalKp ${canonicalKp.length} '
      '!= expression ${expression.length}',
    );
  }
  if (canonicalKp.length % 3 != 0) {
    throw ArgumentError(
      'transformKeypoints: canonicalKp length ${canonicalKp.length} '
      'not a multiple of 3',
    );
  }
  if (rotation.length != 9) {
    throw ArgumentError('rotation must be length 9; got ${rotation.length}');
  }
  if (translation.length != 3) {
    throw ArgumentError(
      'translation must be length 3; got ${translation.length}',
    );
  }
  final k = canonicalKp.length ~/ 3;
  final out = Float32List(canonicalKp.length);
  // R is row-major. kp @ R means each kp row (1x3) times R (3x3) -> 1x3.
  // out[i,j] = sum_m kp[i,m] * R[m,j]
  for (var i = 0; i < k; i++) {
    final base = i * 3;
    final x = canonicalKp[base];
    final y = canonicalKp[base + 1];
    final z = canonicalKp[base + 2];
    final ox = x * rotation[0] + y * rotation[3] + z * rotation[6];
    final oy = x * rotation[1] + y * rotation[4] + z * rotation[7];
    final oz = x * rotation[2] + y * rotation[5] + z * rotation[8];
    var fx = ox + expression[base];
    var fy = oy + expression[base + 1];
    var fz = oz + expression[base + 2];
    fx *= scale;
    fy *= scale;
    fz *= scale;
    fx += translation[0];
    fy += translation[1];
    out[base] = fx;
    out[base + 1] = fy;
    out[base + 2] = fz;
  }
  return out;
}
