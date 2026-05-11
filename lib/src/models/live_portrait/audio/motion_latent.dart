/// LMDM motion-latent layout helpers.
///
/// The 265-dim per-frame motion latent that LMDM consumes/produces is
/// defined by Ditto's `_cvt_LP_motion_info`:
///
///   scale  (1)   = scale - 1   (i.e. delta from 1.0)
///   pitch (66)   raw bin distribution
///   yaw   (66)
///   roll  (66)
///   t      (3)
///   exp   (63)
///                ----
///                265
///
/// `kp` (the canonical keypoints) is **excluded** because it's a
/// per-source constant baked into [SourceState].
library;

import 'dart:typed_data';

const int kMotionLatentDim = 265;

/// Field offsets into the flat 265-dim layout.
class MotionLatentOffsets {
  static const scale = 0;
  static const pitch = 1;
  static const yaw = pitch + 66;
  static const roll = yaw + 66;
  static const t = roll + 66;
  static const exp = t + 3;
  static const total = exp + 63;
}

/// Pack the source's per-frame motion fields into the 265-dim latent.
/// Used to seed `kp_cond` for the first LMDM clip.
Float32List packSourceMotionLatent({
  required double scale,
  required Float32List pitchBins,
  required Float32List yawBins,
  required Float32List rollBins,
  required Float32List translation,
  required Float32List expression,
}) {
  if (pitchBins.length != 66 ||
      yawBins.length != 66 ||
      rollBins.length != 66) {
    throw ArgumentError('pitch/yaw/roll bins must each be length 66');
  }
  if (translation.length != 3) {
    throw ArgumentError('translation must be length 3');
  }
  if (expression.length != 63) {
    throw ArgumentError('expression must be length 63');
  }
  final out = Float32List(kMotionLatentDim);
  out[MotionLatentOffsets.scale] = scale - 1.0;
  for (var i = 0; i < 66; i++) {
    out[MotionLatentOffsets.pitch + i] = pitchBins[i];
    out[MotionLatentOffsets.yaw + i] = yawBins[i];
    out[MotionLatentOffsets.roll + i] = rollBins[i];
  }
  for (var i = 0; i < 3; i++) {
    out[MotionLatentOffsets.t + i] = translation[i];
  }
  for (var i = 0; i < 63; i++) {
    out[MotionLatentOffsets.exp + i] = expression[i];
  }
  return out;
}

/// Slice one frame out of a flat `[seqFrames, 265]` motion latent
/// buffer. Returns a view-style copy (length 265).
Float32List sliceMotionFrame(Float32List packed, int frameIdx) {
  final base = frameIdx * kMotionLatentDim;
  return Float32List.fromList(
    packed.sublist(base, base + kMotionLatentDim),
  );
}

/// Container exposing the field views without copying. Useful for
/// downstream rotation-matrix construction & keypoint transformation.
final class UnpackedMotionFrame {
  const UnpackedMotionFrame({
    required this.scale,
    required this.pitchBins,
    required this.yawBins,
    required this.rollBins,
    required this.translation,
    required this.expression,
  });

  final double scale;
  final Float32List pitchBins;
  final Float32List yawBins;
  final Float32List rollBins;
  final Float32List translation;
  final Float32List expression;
}

/// Unpack a single 265-dim frame.
UnpackedMotionFrame unpackMotionLatent(Float32List frame) {
  if (frame.length != kMotionLatentDim) {
    throw ArgumentError(
      'unpackMotionLatent expects length $kMotionLatentDim; got ${frame.length}',
    );
  }
  final pitch = Float32List(66);
  final yaw = Float32List(66);
  final roll = Float32List(66);
  for (var i = 0; i < 66; i++) {
    pitch[i] = frame[MotionLatentOffsets.pitch + i];
    yaw[i] = frame[MotionLatentOffsets.yaw + i];
    roll[i] = frame[MotionLatentOffsets.roll + i];
  }
  final t = Float32List(3);
  for (var i = 0; i < 3; i++) {
    t[i] = frame[MotionLatentOffsets.t + i];
  }
  final exp = Float32List(63);
  for (var i = 0; i < 63; i++) {
    exp[i] = frame[MotionLatentOffsets.exp + i];
  }
  return UnpackedMotionFrame(
    scale: frame[MotionLatentOffsets.scale] + 1.0,
    pitchBins: pitch,
    yawBins: yaw,
    rollBins: roll,
    translation: t,
    expression: exp,
  );
}
