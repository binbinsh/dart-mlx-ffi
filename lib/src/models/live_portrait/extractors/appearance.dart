/// Appearance Feature Extractor (`appearance_extractor.onnx`).
///
/// Single-shot 3D appearance volume extraction from a 256×256 face crop.
///
/// Input  : `image` `[1, 3, 256, 256]` float32 RGB in [0,1]
/// Output : `pred`  `[1, 32, 16, 64, 64]` float32 — the 5D feature
///          volume consumed by the warping module.
///
/// Total output size = 1 × 32 × 16 × 64 × 64 = 2_097_152 floats =
/// 8 MiB. Held in a [Float32List] inside [SourceState] for the
/// duration of one talking session.
library;

import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

const String _kAppearanceFamily = 'live_portrait_appearance';

/// Wraps the ORT session for `appearance_extractor.onnx`.
///
/// Stateless given an input crop. Hold one per [LivePortraitEngine];
/// call [close] when shutting down.
final class AppearanceExtractor {
  AppearanceExtractor._({
    required DartOnnxSession session,
    required String inputName,
    required String outputName,
  }) : _session = session,
       _inputName = inputName,
       _outputName = outputName;

  factory AppearanceExtractor.load({
    required String onnxPath,
    int numThreads = 2,
  }) {
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: onnxPath,
        id: _kAppearanceFamily,
        family: _kAppearanceFamily,
        provider: 'coreml',
        requireProvider: false,
        numThreads: numThreads,
      ),
    );
    final diag = session.diagnostics;
    final inputs = (diag['input_metadata'] as List).cast<Map>();
    final outputs = (diag['output_metadata'] as List).cast<Map>();
    if (inputs.length != 1 || outputs.length != 1) {
      session.close();
      throw StateError(
        'AppearanceExtractor: expected 1 input + 1 output, got '
        'inputs=${inputs.length} outputs=${outputs.length}',
      );
    }
    return AppearanceExtractor._(
      session: session,
      inputName: inputs.first['name'] as String,
      outputName: outputs.first['name'] as String,
    );
  }

  final DartOnnxSession _session;
  final String _inputName;
  final String _outputName;

  /// Run inference. [crop256Nchw] must be length 3*256*256 = 196608.
  /// Returns the flattened `[1,32,16,64,64]` volume (length 2_097_152).
  Float32List extract(Float32List crop256Nchw) {
    if (crop256Nchw.length != 3 * 256 * 256) {
      throw ArgumentError(
        'AppearanceExtractor.extract: input length ${crop256Nchw.length} '
        '!= 3*256*256',
      );
    }
    final inputTensor = RuntimeTensor.float32(
      [1, 3, 256, 256],
      crop256Nchw,
    );
    final result = _session.run({_inputName: inputTensor});
    try {
      final out = (result.outputs[_outputName] as RuntimeTensor)
          .asFloat32List();
      // Copy out — the underlying tensor memory is freed when result.close()
      // runs. Float32List.fromList copies into a fresh typed-data backing.
      return Float32List.fromList(out);
    } finally {
      result.close();
    }
  }

  void close() => _session.close();
}
