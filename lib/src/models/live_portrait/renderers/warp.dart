/// Warping Network (`warp_network_v2.onnx`).
///
/// This is the GridSample3D-rewritten version of Ditto's
/// `warp_network.onnx`: the two custom `GridSample3D` ops (domain `''`)
/// were replaced with standard `ai.onnx::GridSample` at opset 20,
/// `align_corners=0, mode=linear, padding_mode=zeros`. See
/// `tool/rewrite_warp_gridsample3d.py`.
///
/// Inputs:
///   * `feature_3d` `[1, 32, 16, 64, 64]` — appearance volume from
///     [AppearanceExtractor].
///   * `kp_source`  `[1, 21, 3]`           — transformed source kps.
///   * `kp_driving` `[1, 21, 3]`           — transformed driving kps
///     (post-stitching adjust).
///
/// Output:
///   * `out` `[1, 256, 64, 64]` — warped feature consumed by the
///     SPADE generator.
///
/// CPU first-inference is ~1.2 s on M1 (warm should drop). Phase 4.5
/// will switch the EP to CoreML.
library;

import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

const String _kFamily = 'live_portrait_warp_v2';

final class WarpNetwork {
  WarpNetwork._({
    required DartOnnxSession session,
    required this.featureInput,
    required this.kpSourceInput,
    required this.kpDrivingInput,
    required this.outputName,
  }) : _session = session;

  factory WarpNetwork.load({
    required String onnxPath,
    int numThreads = 4,
  }) {
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: onnxPath,
        id: _kFamily,
        family: _kFamily,
        provider: 'coreml',
        requireProvider: false,
        numThreads: numThreads,
      ),
    );
    final diag = session.diagnostics;
    final inputs = (diag['input_metadata'] as List).cast<Map>();
    final outputs = (diag['output_metadata'] as List).cast<Map>();
    final names = inputs.map((m) => m['name'] as String).toSet();
    const expected = {'feature_3d', 'kp_source', 'kp_driving'};
    if (!expected.every(names.contains)) {
      session.close();
      throw StateError(
        'WarpNetwork: ONNX inputs $names != expected $expected',
      );
    }
    if (outputs.length != 1) {
      session.close();
      throw StateError(
        'WarpNetwork: expected 1 output, got ${outputs.length}',
      );
    }
    return WarpNetwork._(
      session: session,
      featureInput: 'feature_3d',
      kpSourceInput: 'kp_source',
      kpDrivingInput: 'kp_driving',
      outputName: outputs.first['name'] as String,
    );
  }

  final DartOnnxSession _session;
  final String featureInput;
  final String kpSourceInput;
  final String kpDrivingInput;
  final String outputName;

  /// Run warp. Inputs are flat float32:
  ///   * [appearanceVolume]: length 32*16*64*64 = 2_097_152
  ///   * [kpSource]: length 21*3 = 63
  ///   * [kpDriving]: length 21*3 = 63
  ///
  /// Returns flat `[1, 256, 64, 64]` length 1_048_576.
  Float32List run({
    required Float32List appearanceVolume,
    required Float32List kpSource,
    required Float32List kpDriving,
  }) {
    if (appearanceVolume.length != 32 * 16 * 64 * 64) {
      throw ArgumentError(
        'WarpNetwork: appearanceVolume length '
        '${appearanceVolume.length} != 2_097_152',
      );
    }
    if (kpSource.length != 63 || kpDriving.length != 63) {
      throw ArgumentError(
        'WarpNetwork: kpSource/kpDriving must be length 63 (21*3); '
        'got ${kpSource.length}/${kpDriving.length}',
      );
    }
    final inputs = <String, Object?>{
      featureInput: RuntimeTensor.float32(
        const [1, 32, 16, 64, 64],
        appearanceVolume,
      ),
      kpSourceInput: RuntimeTensor.float32(const [1, 21, 3], kpSource),
      kpDrivingInput: RuntimeTensor.float32(const [1, 21, 3], kpDriving),
    };
    final result = _session.run(inputs);
    try {
      final out = (result.outputs[outputName] as RuntimeTensor)
          .asFloat32List();
      return Float32List.fromList(out);
    } finally {
      result.close();
    }
  }

  void close() => _session.close();
}
