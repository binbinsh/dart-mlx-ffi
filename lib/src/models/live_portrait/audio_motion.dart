/// Audio → motion-latent pipeline for LivePortrait (Ditto fork).
///
/// **Phase 3.5: offline batch.** This wires the real [HubertEncoder]
/// and [LmdmSampler] in a finite, single-shot mode:
///
///   pushAudio(pcm) → HuBERT.encode → buildAudioCondTensor →
///       LmdmSampler.sample(seqFrames=80) → `List<MotionFrame>`
///
/// Streaming sliding-window emission across multiple LMDM windows is
/// the next phase — for now the pipeline assumes the whole utterance
/// is pushed in one call. Audio shorter than 80 frames (3.2s @ 25Hz)
/// is zero-padded; longer audio is chunked into back-to-back 80-frame
/// windows with `kpCond` carried forward from the previous window's
/// last frame (matches Ditto's autoregressive seeding).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'audio/cond_builder.dart';
import 'audio/hubert.dart';
import 'audio/lmdm.dart';
import 'audio/motion_latent.dart';
import 'config.dart';
import 'renderer.dart';

/// One frame of motion latent emitted by the LMDM sampler.
///
/// Wraps the raw 265-dim latent so the renderer can reconstruct the
/// driving info via [Driving.fromLatent]. Direct `rotation` /
/// `expression` accessors are deferred — keep one source of truth.
final class MotionFrame {
  const MotionFrame(this.latent);

  /// Length [kMotionLatentDim] (265) flat motion latent.
  /// Layout: [MotionLatentOffsets].
  final Float32List latent;
}

/// Audio → motion driver. Holds the source's canonical kp + initial
/// kpCond seed so callers only need to push PCM.
abstract class AudioMotionPipeline {
  /// Create a pipeline tied to a specific [source]. Reset the pipeline
  /// (or rebuild it) when the source portrait changes.
  factory AudioMotionPipeline.create({
    required LivePortraitConfig config,
    required HubertEncoder hubert,
    required LmdmSampler sampler,
    required SourceState source,
    int samplingTimesteps,
    math.Random? rng,
  }) = _AudioMotionPipelineImpl;

  /// Push raw 16 kHz mono PCM. Returns motion frames covering the
  /// entire pushed audio (may include trailing zero-padded frames if
  /// audio length isn't a multiple of one LMDM window).
  List<MotionFrame> pushAudio(Float32List pcm16k);

  /// Discard buffered audio + motion state (resets `kpCond` to the
  /// source seed).
  void reset();
}

class _AudioMotionPipelineImpl implements AudioMotionPipeline {
  _AudioMotionPipelineImpl({
    required this.config,
    required this.hubert,
    required this.sampler,
    required this.source,
    this.samplingTimesteps = 10,
    math.Random? rng,
  }) : _rng = rng ?? math.Random(),
       _seed = packSourceMotionLatent(
         scale: source.scale,
         pitchBins: source.pitchBins,
         yawBins: source.yawBins,
         rollBins: source.rollBins,
         translation: source.translation,
         expression: source.expression,
       );

  final LivePortraitConfig config;
  final HubertEncoder hubert;
  final LmdmSampler sampler;
  final SourceState source;
  final int samplingTimesteps;
  final math.Random _rng;

  /// Source-seeded motion latent, used as `kpCond` for the first
  /// window. Subsequent windows use the previous window's last frame.
  final Float32List _seed;
  Float32List? _kpCond;

  @override
  List<MotionFrame> pushAudio(Float32List pcm16k) {
    final encoded = hubert.encode(pcm16k);
    final hubertFeatures = encoded.features;
    final hubertFrames = encoded.frameCount;
    if (hubertFrames == 0) return const [];

    final windowFrames = sampler.seqFrames;
    final motionDim = sampler.motionDim;
    final hubertDim = hubertFeatures.length ~/ hubertFrames;

    final frames = <MotionFrame>[];
    var kpCond = _kpCond ?? _seed;

    for (var winStart = 0;
        winStart < hubertFrames;
        winStart += windowFrames) {
      final winEnd = math.min(winStart + windowFrames, hubertFrames);
      final winLen = winEnd - winStart;

      // Slice hubert features for this window (zero-pad to windowFrames).
      final winHubert = Float32List(windowFrames * hubertDim);
      for (var f = 0; f < winLen; f++) {
        final src = (winStart + f) * hubertDim;
        final dst = f * hubertDim;
        for (var d = 0; d < hubertDim; d++) {
          winHubert[dst + d] = hubertFeatures[src + d];
        }
      }

      final cond = buildAudioCondTensor(
        hubert: winHubert,
        sourceCanonical: source.canonicalKeypoints,
        seqFrames: windowFrames,
        hubertDim: hubertDim,
      );
      final motionFlat = sampler.sample(
        kpCond: kpCond,
        audioCond: cond,
        samplingTimesteps: samplingTimesteps,
        rng: _rng,
      );

      // Emit only the frames that correspond to real audio (skip zero-pad).
      for (var f = 0; f < winLen; f++) {
        final base = f * motionDim;
        final latent = Float32List(motionDim);
        for (var d = 0; d < motionDim; d++) {
          latent[d] = motionFlat[base + d];
        }
        frames.add(MotionFrame(latent));
      }
      // Carry last produced frame forward as kpCond for the next window.
      final lastBase = (winLen - 1) * motionDim;
      kpCond = Float32List(motionDim);
      for (var d = 0; d < motionDim; d++) {
        kpCond[d] = motionFlat[lastBase + d];
      }
    }
    _kpCond = kpCond;
    return frames;
  }

  @override
  void reset() {
    _kpCond = null;
  }
}
