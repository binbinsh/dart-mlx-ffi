// Voice prompt path for CosyVoice2 zero-shot synthesis.
//
// Given a prompt waveform (float32 PCM, mono) and its sample rate,
// produces a `SpeakerPrompt` containing:
//
//   * `speakerEmbedding`   (float32 [192]) — campplus output.
//   * `promptSpeechTokens` (int32  [N])    — speech_tokenizer_v2 ids.
//   * `promptSpeechFeat`   (float32 [F, 80]) — matcha 80-mel features
//                                              for the flow encoder.
//
// All three are derived in the same pipeline shape as upstream
// `CosyVoiceFrontEnd.frontend_zero_shot`:
//
//   1. resample to 16 kHz (for tokenizer + speaker encoder).
//   2. resample to 24 kHz (for matcha 80-mel feature path).
//   3. compute whisper 128-mel  -> speech_tokenizer_v2  -> tokens.
//   4. compute kaldi 80-fbank   -> CMN -> campplus       -> embedding.
//   5. compute matcha 80-mel    -> transposed (F, 80)    -> speech feat.
//   6. enforce upstream token/feat length parity:
//        token_len = min(feat_frames // 2, token_count)
//        feat       = feat[:2*token_len]
//        tokens     = tokens[:token_len]
//
// Resampling is intentionally rate-agnostic (linear interpolation):
// CosyVoice2's voice prompt is rarely > 30 s and the extractors
// dominate the runtime, so a fancy polyphase filter is overkill here
// and would pull in extra dependencies. If the upstream input is
// already at the target rate the native helper performs a straight native copy.

import 'dart:typed_data';

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/native_tensor_buffers.dart';
import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart' show RuntimeTensor;
import 'cosyvoice2.dart';
import 'cosyvoice2_native.dart';
import 'cosyvoice2_prompt_native.dart';

class SpeakerPrompt {
  SpeakerPrompt({
    required Float32List speakerEmbedding,
    required Int32List promptSpeechTokens,
    required Float32List promptSpeechFeat,
    required this.promptSpeechFeatFrames,
  }) : _speakerEmbedding = speakerEmbedding,
       _promptSpeechTokens = promptSpeechTokens,
       _promptSpeechFeat = promptSpeechFeat,
       _speakerEmbeddingBuffer = null,
       _promptSpeechTokensBuffer = null,
       _promptSpeechFeatBuffer = null;

  SpeakerPrompt._native({
    required NativeTensorBuffer speakerEmbedding,
    required NativeTensorBuffer promptSpeechTokens,
    required NativeTensorBuffer promptSpeechFeat,
    required this.promptSpeechFeatFrames,
  }) : _speakerEmbedding = null,
       _promptSpeechTokens = null,
       _promptSpeechFeat = null,
       _speakerEmbeddingBuffer = speakerEmbedding,
       _promptSpeechTokensBuffer = promptSpeechTokens,
       _promptSpeechFeatBuffer = promptSpeechFeat;

  /// `[192]` float32 from campplus.
  Float32List get speakerEmbedding =>
      _speakerEmbedding ?? _speakerEmbeddingBuffer!.asFloat32List();

  Object get speakerEmbeddingSource =>
      _speakerEmbeddingBuffer ?? speakerEmbedding;

  /// `[N]` int32 speech tokens.
  Int32List get promptSpeechTokens =>
      _promptSpeechTokens ?? _promptSpeechTokensBuffer!.asInt32List();

  Object get promptSpeechTokensSource =>
      _promptSpeechTokensBuffer ?? promptSpeechTokens;

  int get promptSpeechTokenCount =>
      _promptSpeechTokens?.length ?? _promptSpeechTokensBuffer!.byteLength ~/ 4;

  /// Matcha 80-mel features with shape `(F, 80)` row-major. Already
  /// transposed from the (80, F) raw mel layout — flow encoder takes
  /// time-major.
  Float32List get promptSpeechFeat =>
      _promptSpeechFeat ?? _promptSpeechFeatBuffer!.asFloat32List();

  Object get promptSpeechFeatSource =>
      _promptSpeechFeatBuffer ?? promptSpeechFeat;

  /// `F` (time dimension of `promptSpeechFeat`).
  final int promptSpeechFeatFrames;

  final Float32List? _speakerEmbedding;
  final Int32List? _promptSpeechTokens;
  final Float32List? _promptSpeechFeat;
  final NativeTensorBuffer? _speakerEmbeddingBuffer;
  final NativeTensorBuffer? _promptSpeechTokensBuffer;
  final NativeTensorBuffer? _promptSpeechFeatBuffer;

  void close() {
    _speakerEmbeddingBuffer?.close();
    _promptSpeechTokensBuffer?.close();
    _promptSpeechFeatBuffer?.close();
  }
}

class SpeakerPromptExtractor {
  SpeakerPromptExtractor({required this.bundle})
    : _promptPlan = CosyPromptNativePlan();

  final CosyVoice2PartialOnnxBundle bundle;
  final CosyPromptNativePlan _promptPlan;

  /// Run the full prompt-extraction pipeline. The two ONNX components
  /// (`speech_tokenizer_v2`, `campplus`) must already be loaded into
  /// the bundle; this method doesn't load them on demand.
  SpeakerPrompt extract(Float32List audio, int sampleRate) {
    if (audio.isEmpty) {
      throw ArgumentError('prompt audio is empty');
    }
    final at16k = cosyResampleLinearBuffer(
      audio,
      srcRate: sampleRate,
      dstRate: 16000,
    );
    final at24k = cosyResampleLinearBuffer(
      audio,
      srcRate: sampleRate,
      dstRate: 24000,
    );
    try {
      // 1. speech tokens via 128-mel + speech_tokenizer_v2.
      //    Inputs: feats[1, 128, T] f32, feats_length[1] i32.
      final mel128 = cosyPromptMelSpectrogramBuffer(
        at16k,
        kind: CosyPromptMelKind.whisper128,
        plan: _promptPlan,
      );
      final featLength = NativeTensorBuffer.int32([1]);
      featLength.asInt32List()[0] = mel128.frames;
      final DartOnnxResult tokRes;
      try {
        tokRes = bundle.runComponent('speech_tokenizer_v2', {
          'feats': mel128.data.tensorView(
            shape: [1, mel128.bins, mel128.frames],
            byteLength: mel128.data.byteLength,
          ),
          'feats_length': featLength.tensor,
        });
      } finally {
        featLength.close();
        mel128.close();
      }
      final NativeTensorBuffer tokens;
      try {
        tokens = _readInt32Buffer(tokRes, 'indices');
      } finally {
        tokRes.close();
      }
      NativeTensorBuffer? speakerEmbedding;
      var keepSpeakerEmbedding = false;
      try {
        // 2. speaker embedding via kaldi 80-fbank + CMN + campplus.
        //    Inputs: input[1, T, 80] f32.
        final fb = cosyPromptKaldiFbankBuffer(at16k, plan: _promptPlan);
        final DartOnnxResult campRes;
        try {
          cosyPromptCepstralMeanNormalizeInPlace(
            fb.data,
            frames: fb.frames,
            bins: fb.bins,
          );
          campRes = bundle.runComponent('campplus', {
            'input': fb.data.tensorView(
              shape: [1, fb.frames, fb.bins],
              byteLength: fb.data.byteLength,
            ),
          });
        } finally {
          fb.close();
        }
        try {
          speakerEmbedding = _readFloat32Buffer(campRes, 'output');
          if (speakerEmbedding.byteLength ~/ 4 != 192) {
            throw StateError(
              'campplus output expected length 192, got '
              '${speakerEmbedding.byteLength ~/ 4}',
            );
          }
        } finally {
          campRes.close();
        }

        // 3. matcha 80-mel feature for the flow encoder.
        final mel80 = cosyPromptMelSpectrogramBuffer(
          at24k,
          kind: CosyPromptMelKind.matcha80,
          plan: _promptPlan,
        );
        // Layout is (80, F). Transpose to (F, 80).
        final feFrames = mel80.frames;
        final NativeTensorBuffer feat;
        try {
          feat = cosyTransposeFloat32Buffer(
            mel80.data,
            rows: mel80.bins,
            cols: mel80.frames,
          );
        } finally {
          mel80.close();
        }
        try {
          // 4. upstream length parity: cosyvoice2 forces feat % 2 == token,
          //    i.e. each speech token corresponds to 2 feature frames.
          final tokenLen = ((feFrames ~/ 2) < (tokens.byteLength ~/ 4))
              ? feFrames ~/ 2
              : tokens.byteLength ~/ 4;
          final clipped = cosyClipPromptBuffers(
            feat: feat,
            tokens: tokens,
            tokenLen: tokenLen,
            melBins: 80,
          );
          var keepClipped = false;
          try {
            final prompt = SpeakerPrompt._native(
              speakerEmbedding: speakerEmbedding,
              promptSpeechTokens: clipped.tokens,
              promptSpeechFeat: clipped.feat,
              promptSpeechFeatFrames: clipped.featFrames,
            );
            keepSpeakerEmbedding = true;
            keepClipped = true;
            return prompt;
          } finally {
            if (!keepClipped) {
              clipped.close();
            }
          }
        } finally {
          feat.close();
        }
      } finally {
        if (!keepSpeakerEmbedding) {
          speakerEmbedding?.close();
        }
        tokens.close();
      }
    } finally {
      at24k.close();
      at16k.close();
    }
  }

  void close() {
    _promptPlan.close();
  }
}

// --- helpers ----------------------------------------------------------

Float32List _readFloat32(DartOnnxResult r, String name) {
  final v = r.outputs[name];
  if (v is Float32List) return v;
  if (v is RuntimeTensor) return float32View(v);
  if (v is List<double>) return Float32List.fromList(v);
  throw StateError('output "$name" has unexpected type ${v.runtimeType}');
}

NativeTensorBuffer _readFloat32Buffer(DartOnnxResult r, String name) {
  return nativeFloat32Buffer(_readFloat32(r, name));
}

Int32List _readInt32(DartOnnxResult r, String name) {
  final v = r.outputs[name];
  if (v is Int32List) return v;
  if (v is RuntimeTensor) return v.asInt32List();
  if (v is List<int>) return Int32List.fromList(v);
  throw StateError('output "$name" has unexpected type ${v.runtimeType}');
}

NativeTensorBuffer _readInt32Buffer(DartOnnxResult r, String name) {
  return nativeInt32Buffer(_readInt32(r, name));
}
