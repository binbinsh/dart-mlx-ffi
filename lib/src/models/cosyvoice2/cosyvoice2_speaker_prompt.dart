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
// already at the target rate we skip resampling entirely.

import 'dart:typed_data';

import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart' show RuntimeTensor;
import 'cosyvoice2.dart';
import 'cosyvoice2_kaldi_fbank.dart';
import 'cosyvoice2_mel.dart';

class SpeakerPrompt {
  SpeakerPrompt({
    required this.speakerEmbedding,
    required this.promptSpeechTokens,
    required this.promptSpeechFeat,
    required this.promptSpeechFeatFrames,
  });

  /// `[192]` float32 from campplus.
  final Float32List speakerEmbedding;

  /// `[N]` int32 speech tokens.
  final Int32List promptSpeechTokens;

  /// Matcha 80-mel features with shape `(F, 80)` row-major. Already
  /// transposed from the (80, F) raw mel layout — flow encoder takes
  /// time-major.
  final Float32List promptSpeechFeat;

  /// `F` (time dimension of `promptSpeechFeat`).
  final int promptSpeechFeatFrames;
}

class SpeakerPromptExtractor {
  SpeakerPromptExtractor({required this.bundle});

  final CosyVoice2PartialOnnxBundle bundle;

  /// Run the full prompt-extraction pipeline. The two ONNX components
  /// (`speech_tokenizer_v2`, `campplus`) must already be loaded into
  /// the bundle; this method doesn't load them on demand.
  SpeakerPrompt extract(Float32List audio, int sampleRate) {
    if (audio.isEmpty) {
      throw ArgumentError('prompt audio is empty');
    }
    final at16k = _resampleLinear(audio, sampleRate, 16000);
    final at24k = _resampleLinear(audio, sampleRate, 24000);

    // 1. speech tokens via 128-mel + speech_tokenizer_v2.
    //    Inputs: feats[1, 128, T] f32, feats_length[1] i32.
    final mel128 =
        computeMelSpectrogram(at16k, MelConfig.whisper128);
    final tokRes = bundle.runComponent('speech_tokenizer_v2', {
      'feats': float32Tensor(
          mel128.data, [1, mel128.numMels, mel128.nFrames]),
      'feats_length':
          RuntimeTensor.int32([1], Int32List.fromList([mel128.nFrames])),
    });
    final tokens = _readInt32(tokRes, 'indices');

    // 2. speaker embedding via kaldi 80-fbank + CMN + campplus.
    //    Inputs: input[1, T, 80] f32.
    final fb = computeKaldiFbank(at16k, const KaldiFbankConfig());
    cepstralMeanNormalize(fb.data, fb.nFrames, fb.numMelBins);
    final campRes = bundle.runComponent('campplus', {
      'input':
          float32Tensor(fb.data, [1, fb.nFrames, fb.numMelBins]),
    });
    final emb = _readFloat32(campRes, 'output');
    if (emb.length != 192) {
      throw StateError(
          'campplus output expected length 192, got ${emb.length}');
    }

    // 3. matcha 80-mel feature for the flow encoder
    final mel80 = computeMelSpectrogram(at24k, MelConfig.matcha80);
    // Layout is (80, F). Transpose to (F, 80).
    final feat = _transpose(mel80.data, mel80.numMels, mel80.nFrames);

    // 4. upstream length parity: cosyvoice2 forces feat % 2 == token,
    //    i.e. each speech token corresponds to 2 feature frames.
    final feFrames = mel80.nFrames;
    final tokenLen = ((feFrames ~/ 2) < tokens.length)
        ? feFrames ~/ 2
        : tokens.length;
    final clippedFeat = feat.sublist(0, 2 * tokenLen * 80);
    final clippedTokens =
        Int32List.fromList(tokens.sublist(0, tokenLen));

    return SpeakerPrompt(
      speakerEmbedding: emb,
      promptSpeechTokens: clippedTokens,
      promptSpeechFeat: Float32List.fromList(clippedFeat),
      promptSpeechFeatFrames: 2 * tokenLen,
    );
  }
}

// --- helpers ----------------------------------------------------------

Float32List _transpose(Float32List src, int rows, int cols) {
  final out = Float32List(rows * cols);
  for (var r = 0; r < rows; r += 1) {
    for (var c = 0; c < cols; c += 1) {
      out[c * rows + r] = src[r * cols + c];
    }
  }
  return out;
}

Float32List _resampleLinear(Float32List x, int srcRate, int dstRate) {
  if (srcRate == dstRate) return x;
  if (x.length < 2) return x;
  final ratio = srcRate / dstRate;
  final dstLen = (x.length / ratio).floor();
  final out = Float32List(dstLen);
  for (var i = 0; i < dstLen; i += 1) {
    final srcPos = i * ratio;
    final srcIdx = srcPos.floor();
    final frac = srcPos - srcIdx;
    if (srcIdx + 1 >= x.length) {
      out[i] = x[x.length - 1];
    } else {
      out[i] = (x[srcIdx] * (1.0 - frac) + x[srcIdx + 1] * frac).toDouble();
    }
  }
  return out;
}

Float32List _readFloat32(DartOnnxResult r, String name) {
  final v = r.outputs[name];
  if (v is Float32List) return v;
  if (v is RuntimeTensor) return float32View(v);
  if (v is List<double>) return Float32List.fromList(v);
  throw StateError('output "$name" has unexpected type ${v.runtimeType}');
}

Int32List _readInt32(DartOnnxResult r, String name) {
  final v = r.outputs[name];
  if (v is Int32List) return v;
  if (v is RuntimeTensor) return v.asInt32List();
  if (v is List<int>) return Int32List.fromList(v);
  throw StateError('output "$name" has unexpected type ${v.runtimeType}');
}
