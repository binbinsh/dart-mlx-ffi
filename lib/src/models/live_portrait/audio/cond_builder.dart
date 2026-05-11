/// Build the LMDM `cond` tensor `[1, seqFrames, 1103]` from HuBERT
/// audio features + per-source aux fields.
///
/// Mirrors Ditto's `ConditionHandler.__call__` (concat order):
///   1. aud_feat       (1024)
///   2. emo_seq         (8)   default = "Surprise"=6 averaged via softmax
///   3. eye_open_seq    (2)   default = (0.6, 0.6) (eyes open)
///   4. eye_ball_seq    (6)   default = zeros (centered gaze)
///   5. sc_seq         (63)   source canonical keypoints, broadcast per-frame
///                          ----
///                          1103
///
/// Defaults provide a "no extra signals" baseline so callers can ship
/// LivePortrait talking out of the box without an emotion/eye estimator.
/// The buddy app can override `emoOneHot`, `eyeOpen`, `eyeBall` to inject
/// expression / gaze later.
library;

import 'dart:math' as math;
import 'dart:typed_data';

const int kEmoDim = 8;
const int kEyeOpenDim = 2;
const int kEyeBallDim = 6;
const int kSourceCanonicalDim = 63;

/// 8-class softmax average heavily weighted toward [emoIdx]. Matches
/// Ditto's `_get_emo_avg(idx, weight=8)` exactly.
Float32List defaultEmoVector({int emoIdx = 4}) {
  if (emoIdx < 0 || emoIdx >= kEmoDim) {
    throw ArgumentError('emoIdx must be in [0, 8)');
  }
  final logits = Float32List(kEmoDim);
  logits[emoIdx] = 8.0;
  // softmax
  var maxV = logits[0];
  for (var i = 1; i < kEmoDim; i++) {
    if (logits[i] > maxV) maxV = logits[i];
  }
  var sum = 0.0;
  final out = Float32List(kEmoDim);
  for (var i = 0; i < kEmoDim; i++) {
    out[i] = math.exp(logits[i] - maxV);
    sum += out[i];
  }
  for (var i = 0; i < kEmoDim; i++) {
    out[i] /= sum;
  }
  return out;
}

/// Build `[seqFrames, 1103]` flat condition tensor.
///
/// Inputs:
///   * [hubert]     `[seqFrames, 1024]` flat HuBERT features. If shorter
///                  than seqFrames, the missing frames are zero-padded.
///   * [sourceCanonical] length-63 source canonical keypoints (from
///     [SourceState.canonicalKeypoints]).
///   * [emoOneHot]  optional length-8 emotion vector. Defaults to
///                  [defaultEmoVector] (Neutral=4 weighted softmax).
///   * [eyeOpen]    optional length-2 eye-open per side. Defaults to
///                  (0.6, 0.6).
///   * [eyeBall]    optional length-6 eye-ball gaze. Defaults to zeros.
Float32List buildAudioCondTensor({
  required Float32List hubert,
  required Float32List sourceCanonical,
  int seqFrames = 80,
  int hubertDim = 1024,
  Float32List? emoOneHot,
  Float32List? eyeOpen,
  Float32List? eyeBall,
}) {
  if (sourceCanonical.length != kSourceCanonicalDim) {
    throw ArgumentError(
      'sourceCanonical length ${sourceCanonical.length} != '
      '$kSourceCanonicalDim',
    );
  }
  final emo = emoOneHot ?? defaultEmoVector();
  if (emo.length != kEmoDim) {
    throw ArgumentError('emoOneHot length ${emo.length} != $kEmoDim');
  }
  final eo = eyeOpen ?? Float32List.fromList(const [0.6, 0.6]);
  if (eo.length != kEyeOpenDim) {
    throw ArgumentError('eyeOpen length ${eo.length} != $kEyeOpenDim');
  }
  final eb = eyeBall ?? Float32List(kEyeBallDim);
  if (eb.length != kEyeBallDim) {
    throw ArgumentError('eyeBall length ${eb.length} != $kEyeBallDim');
  }

  final perFrameDim =
      hubertDim + kEmoDim + kEyeOpenDim + kEyeBallDim + kSourceCanonicalDim;
  final out = Float32List(seqFrames * perFrameDim);

  final hubertFrames = hubert.length ~/ hubertDim;
  for (var f = 0; f < seqFrames; f++) {
    final base = f * perFrameDim;
    var off = base;
    // 1. hubert (zero pad if past available frames)
    if (f < hubertFrames) {
      final src = f * hubertDim;
      for (var d = 0; d < hubertDim; d++) {
        out[off + d] = hubert[src + d];
      }
    }
    // (else already zero)
    off += hubertDim;
    // 2. emo
    for (var d = 0; d < kEmoDim; d++) {
      out[off + d] = emo[d];
    }
    off += kEmoDim;
    // 3. eye_open
    for (var d = 0; d < kEyeOpenDim; d++) {
      out[off + d] = eo[d];
    }
    off += kEyeOpenDim;
    // 4. eye_ball
    for (var d = 0; d < kEyeBallDim; d++) {
      out[off + d] = eb[d];
    }
    off += kEyeBallDim;
    // 5. sc (source canonical kp)
    for (var d = 0; d < kSourceCanonicalDim; d++) {
      out[off + d] = sourceCanonical[d];
    }
  }
  return out;
}
