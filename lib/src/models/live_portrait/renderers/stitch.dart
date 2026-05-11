/// Stitching MLP (`stitch_network.onnx`).
///
/// Per Ditto / LivePortrait: this is **not** a pixel-space stitcher.
/// It's an MLP that adjusts the driving keypoints `x_d` so that the
/// decoded face stays aligned with the source crop boundary,
/// suppressing seam artifacts. The pixel paste-back is a separate
/// step done in image space (we do it in [PortraitRenderer] using
/// the YuNet crop transform).
///
/// I/O (verified):
///   * input  `kp_source`  `[1, 21, 3]`
///   * input  `kp_driving` `[1, 21, 3]`
///   * output `out`         `[1, 21, 3]` — adjusted driving kps
library;

import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

const String _kFamily = 'live_portrait_stitch';

final class StitchNetwork {
  StitchNetwork._({
    required DartOnnxSession session,
    required this.kpSourceInput,
    required this.kpDrivingInput,
    required this.outputName,
  }) : _session = session;

  factory StitchNetwork.load({
    required String onnxPath,
    int numThreads = 1,
  }) {
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: onnxPath,
        id: _kFamily,
        family: _kFamily,
        provider: 'cpu',
        requireProvider: false,
        numThreads: numThreads,
      ),
    );
    final diag = session.diagnostics;
    final inputs = (diag['input_metadata'] as List).cast<Map>();
    final outputs = (diag['output_metadata'] as List).cast<Map>();
    final names = inputs.map((m) => m['name'] as String).toSet();
    const expected = {'kp_source', 'kp_driving'};
    if (!expected.every(names.contains) || outputs.length != 1) {
      session.close();
      throw StateError(
        'StitchNetwork: I/O mismatch. inputs=$names outputs='
        '${outputs.map((m) => m['name']).toList()}',
      );
    }
    return StitchNetwork._(
      session: session,
      kpSourceInput: 'kp_source',
      kpDrivingInput: 'kp_driving',
      outputName: outputs.first['name'] as String,
    );
  }

  final DartOnnxSession _session;
  final String kpSourceInput;
  final String kpDrivingInput;
  final String outputName;

  /// Adjust [kpDriving] given [kpSource]. Both flat length 63.
  /// Returns flat length 63 adjusted driving kps.
  Float32List stitch({
    required Float32List kpSource,
    required Float32List kpDriving,
  }) {
    if (kpSource.length != 63 || kpDriving.length != 63) {
      throw ArgumentError(
        'StitchNetwork: kp lengths must be 63; got '
        '${kpSource.length}/${kpDriving.length}',
      );
    }
    final inputs = <String, Object?>{
      kpSourceInput: RuntimeTensor.float32(const [1, 21, 3], kpSource),
      kpDrivingInput: RuntimeTensor.float32(const [1, 21, 3], kpDriving),
    };
    final result = _session.run(inputs);
    try {
      final out = (result.outputs[outputName] as RuntimeTensor)
          .asFloat32List();
      if (out.length != 63) {
        throw StateError(
          'StitchNetwork: output length ${out.length} != 63',
        );
      }
      return Float32List.fromList(out);
    } finally {
      result.close();
    }
  }

  void close() => _session.close();
}
