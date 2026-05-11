/// SPADE Generator (`decoder.onnx`).
///
/// Decodes the warped 4D feature `[1, 256, 64, 64]` into an RGB
/// image `[1, 3, 512, 512]`.
///
/// Output is in LivePortrait's `[-1, 1]` convention (`tanh` head).
/// [decode] returns it unchanged; the caller (renderer) is
/// responsible for `(x * 0.5 + 0.5) * 255` denorm and uint8 packing.
library;

import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

const String _kFamily = 'live_portrait_decoder';

final class SpadeDecoder {
  SpadeDecoder._({
    required DartOnnxSession session,
    required this.inputName,
    required this.outputName,
  }) : _session = session;

  factory SpadeDecoder.load({
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
    if (inputs.length != 1 || outputs.length != 1) {
      session.close();
      throw StateError(
        'SpadeDecoder: expected 1 input + 1 output, got '
        'inputs=${inputs.length} outputs=${outputs.length}',
      );
    }
    return SpadeDecoder._(
      session: session,
      inputName: inputs.first['name'] as String,
      outputName: outputs.first['name'] as String,
    );
  }

  final DartOnnxSession _session;
  final String inputName;
  final String outputName;

  /// Run the generator on a flat warped feature
  /// (length 256*64*64 = 1_048_576). Returns the flat NCHW output
  /// `[1, 3, 512, 512]` length 786_432 in `[-1, 1]` (approximately —
  /// activations can clip slightly outside).
  Float32List decode(Float32List warpedFeature) {
    if (warpedFeature.length != 256 * 64 * 64) {
      throw ArgumentError(
        'SpadeDecoder: warpedFeature length ${warpedFeature.length} '
        '!= 1_048_576',
      );
    }
    final inputs = <String, Object?>{
      inputName: RuntimeTensor.float32(
        const [1, 256, 64, 64],
        warpedFeature,
      ),
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

/// Convert a flat `[1, 3, H, W]` float NCHW image in `[-1, 1]` to
/// `[H, W, 3]` packed uint8 RGB. Clips out-of-range to `[0, 255]`.
///
/// This is shared between the renderer (live frames) and the smoke
/// tests (PNG dump), so it lives next to [SpadeDecoder].
Uint8List nchwTanhToRgb({
  required Float32List nchw,
  required int width,
  required int height,
}) {
  final expected = 3 * width * height;
  if (nchw.length != expected) {
    throw ArgumentError(
      'nchwTanhToRgb: nchw length ${nchw.length} != $expected '
      '(3*$width*$height)',
    );
  }
  final out = Uint8List(width * height * 3);
  final plane = width * height;
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final src = y * width + x;
      final dst = (y * width + x) * 3;
      for (var c = 0; c < 3; c++) {
        final v = nchw[c * plane + src];
        // (v*0.5 + 0.5) * 255 — Ditto/LivePortrait clamp to [0, 1].
        var u = (v * 0.5 + 0.5) * 255.0;
        if (u < 0) u = 0;
        if (u > 255) u = 255;
        out[dst + c] = u.toInt();
      }
    }
  }
  return out;
}
