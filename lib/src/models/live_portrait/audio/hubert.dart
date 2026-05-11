/// HuBERT audio encoder ORT wrapper.
///
/// Wraps `hubert.onnx` (`HubertStreaming` in Ditto). Input is a mono
/// 16 kHz float32 waveform; output is `[T, 1024]` features at **25 Hz**
/// (the network itself emits 50 Hz pairs that we mean-pool, matching
/// Ditto's `Wav2FeatHubert.__call__`).
///
/// ## Chunking math (mirrors Ditto)
///
/// Default `chunksize=(3, 5, 2)`:
///   * each forward call consumes `sum(chunksize)*0.04*16000 + 80 = 6480`
///     PCM samples
///   * the network's raw output (50 Hz) is sliced
///     `[-(5+2)*2 : -2*2] = [-14 : -4]` → 10 frames
///   * those 10 frames are reshaped `[5, 2, 1024]` and mean-pooled along
///     dim 1 → 5 valid frames at 25 Hz
///   * stride between calls is `chunksize[1] = 5` (25 Hz frames) =
///     `5 * 0.04 * 16000 = 3200` samples
///
/// ## Offline mode
///
/// `encode(...)` runs the streaming chunker over a finite waveform with
/// the standard pre/post pad and returns exactly
/// `ceil(seconds * 25)` 1024-dim frames. Suitable for the "buddy speaks
/// a TTS clip" use case where the full WAV is in hand before rendering.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

const String _kHubertFamily = 'live_portrait_hubert';

/// HuBERT feature dim (1024 for `hubert.onnx` in Ditto's pack).
const int kHubertFeatureDim = 1024;

/// Motion frame rate the audio→motion stack runs at.
const int kMotionFps = 25;

/// Sample rate the HuBERT ONNX expects.
const int kHubertSampleRate = 16000;

/// One-side chunk parameters: (preFrames, validFrames, postFrames) in
/// 25 Hz units. Defaults match Ditto's `chunksize=(3, 5, 2)`.
final class HubertChunkSize {
  const HubertChunkSize({this.pre = 3, this.valid = 5, this.post = 2});
  final int pre;
  final int valid;
  final int post;
  int get total => pre + valid + post;
}

/// Wraps the ORT session for `hubert.onnx`.
final class HubertEncoder {
  HubertEncoder._({
    required DartOnnxSession session,
    required String inputName,
    required String outputName,
  }) : _session = session,
       _inputName = inputName,
       _outputName = outputName;

  factory HubertEncoder.load({
    required String onnxPath,
    int numThreads = 2,
  }) {
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: onnxPath,
        id: _kHubertFamily,
        family: _kHubertFamily,
        provider: 'cpu',
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
        'HubertEncoder: expected 1 input + 1 output; '
        'got ${inputs.length} / ${outputs.length}',
      );
    }
    return HubertEncoder._(
      session: session,
      inputName: inputs.first['name'] as String,
      outputName: outputs.first['name'] as String,
    );
  }

  final DartOnnxSession _session;
  final String _inputName;
  final String _outputName;

  /// Encode a finite mono 16 kHz waveform into HuBERT features.
  ///
  /// Output is row-major `[frameCount, 1024]` with
  /// `frameCount = ceil(waveform.length / sampleRate * 25)`.
  /// Returned [Float32List] length is `frameCount * 1024`.
  ///
  /// Mirrors `Wav2FeatHubert.wav2feat` exactly:
  ///   1. pad pre with `splitLen - validSamples` zeros
  ///   2. pad post with `splitLen` zeros
  ///   3. step `chunksize.valid * 0.04 * 16000 = 3200` samples per call
  ///   4. concatenate per-call valid windows → trim to `frameCount`
  ({Float32List features, int frameCount}) encode(
    Float32List waveform16k, {
    HubertChunkSize chunkSize = const HubertChunkSize(),
  }) {
    final n = waveform16k.length;
    final secondsTimes25 = n / kHubertSampleRate * kMotionFps;
    final frameCount = (secondsTimes25).ceil();

    // splitLen = sum(chunksize) * 0.04 * 16000 + 80
    final splitLen = (chunkSize.total * 0.04 * kHubertSampleRate).round() + 80;
    // validSamples in 16 kHz = (chunksize.valid + chunksize.post) * 0.04 * 16000
    final tailSamples =
        ((chunkSize.valid + chunkSize.post) * 0.04 * kHubertSampleRate).round();
    final prePad = splitLen - tailSamples;
    final postPad = splitLen;

    final padded = Float32List(prePad + n + postPad);
    // prePad already zeros
    padded.setRange(prePad, prePad + n, waveform16k);
    // postPad already zeros

    final out = Float32List(frameCount * kHubertFeatureDim);
    var producedFrames = 0;
    var stepIdx = 0;
    final stepSamples = (chunkSize.valid * 0.04 * kHubertSampleRate).round();
    while (producedFrames < frameCount) {
      final sss = stepIdx * stepSamples;
      final eee = sss + splitLen;
      if (eee > padded.length) {
        // Should not happen given postPad sizing, but defensively pad.
        break;
      }
      final chunk = Float32List(splitLen);
      chunk.setRange(0, splitLen, padded.sublist(sss, eee));

      final raw = _runOne(chunk); // [Tnet, 1024] flat
      _appendValid(
        raw: raw,
        chunkSize: chunkSize,
        out: out,
        outFrameOffset: producedFrames,
        framesRemaining: frameCount - producedFrames,
      );

      producedFrames += chunkSize.valid;
      stepIdx++;
    }
    return (features: out, frameCount: frameCount);
  }

  Float32List _runOne(Float32List chunk) {
    final inputTensor = RuntimeTensor.float32(
      [1, chunk.length],
      chunk,
    );
    final result = _session.run({_inputName: inputTensor});
    try {
      final tensor = result.outputs[_outputName] as RuntimeTensor;
      final flat = tensor.asFloat32List();
      // Output shape is [Tnet, 1024]; ORT reports it via tensor.shape but
      // we can infer Tnet = flat.length / 1024.
      return Float32List.fromList(flat);
    } finally {
      result.close();
    }
  }

  /// Slice the network's full output, mean-pool 50 Hz pairs, and write
  /// up to `framesRemaining` frames into `out` at `outFrameOffset`.
  ///
  /// Network output layout: `[Tnet, 1024]` flat row-major.
  /// We take frames `[Tnet - (valid+post)*2 : Tnet - post*2]`, reshape
  /// `[valid, 2, 1024]`, mean over the inner dim → `[valid, 1024]`.
  static void _appendValid({
    required Float32List raw,
    required HubertChunkSize chunkSize,
    required Float32List out,
    required int outFrameOffset,
    required int framesRemaining,
  }) {
    const featDim = kHubertFeatureDim;
    final tnet = raw.length ~/ featDim;
    final sliceStart = tnet - (chunkSize.valid + chunkSize.post) * 2;
    final sliceEnd = tnet - chunkSize.post * 2;
    if (sliceStart < 0 || sliceEnd > tnet) {
      throw StateError(
        'HubertEncoder: unexpected raw frame count $tnet for chunkSize '
        '(${chunkSize.pre}, ${chunkSize.valid}, ${chunkSize.post})',
      );
    }
    final emit = math.min(chunkSize.valid, framesRemaining);
    for (var i = 0; i < emit; i++) {
      final aIdx = (sliceStart + i * 2) * featDim;
      final bIdx = (sliceStart + i * 2 + 1) * featDim;
      final outBase = (outFrameOffset + i) * featDim;
      for (var d = 0; d < featDim; d++) {
        out[outBase + d] = (raw[aIdx + d] + raw[bIdx + d]) * 0.5;
      }
    }
  }

  void close() => _session.close();
}
