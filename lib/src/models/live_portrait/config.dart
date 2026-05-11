/// Static config + on-disk layout descriptor for the LivePortrait engine.
///
/// Mirrors the JSON shape produced by `tool/convert_live_portrait_weights.py`:
///
/// ```
/// {
///   "kind": "cmdspace.live_portrait.snapshot",
///   "version": 1,
///   "source": "ditto-talkinghead",
///   "weights": {
///     "appearance":  "appearance_feature_extractor.safetensors",
///     "motion":      "motion_extractor.safetensors",
///     "warp":        "warping_module.safetensors",
///     "decoder":     "spade_generator.safetensors",
///     "stitch":      "stitching_module.safetensors",
///     "hubert":      "hubert.safetensors",
///     "lmdm":        "lmdm_v0_4_hubert.safetensors"
///   },
///   "audio": {
///     "sampleRate": 16000,
///     "hopFrames":  320,            // 50 Hz HuBERT frame rate
///     "featureDim": 768
///   },
///   "render": {
///     "frameWidth":   512,
///     "frameHeight":  512,
///     "internalRes":  256,          // generator native res
///     "fpsTarget":    25
///   },
///   "motion": {
///     "keypointCount": 21,
///     "appearanceVolume": [1, 32, 16, 64, 64]
///   },
///   "sampler": {
///     "kind":          "ddim",
///     "steps":         10,           // streaming default; 50 = quality
///     "guidance":      1.5,
///     "windowFrames":  20            // sliding-window LMDM
///   }
/// }
/// ```
library;

import 'dart:convert';

const String kLivePortraitSchemaVersion = 'cmdspace.live_portrait.snapshot.v1';

final class LivePortraitWeightPaths {
  const LivePortraitWeightPaths({
    required this.appearance,
    required this.motion,
    required this.warp,
    required this.decoder,
    required this.stitch,
    required this.hubert,
    required this.lmdm,
    required this.faceDetector,
  });

  factory LivePortraitWeightPaths.fromJson(Map<String, Object?> json) =>
      LivePortraitWeightPaths(
        appearance: json['appearance'] as String,
        motion: json['motion'] as String,
        warp: json['warp'] as String,
        decoder: json['decoder'] as String,
        stitch: json['stitch'] as String,
        hubert: json['hubert'] as String,
        lmdm: json['lmdm'] as String,
        faceDetector:
            (json['face_detector'] ?? json['faceDetector']) as String? ??
            'onnx/yunet.onnx',
      );

  final String appearance;
  final String motion;
  final String warp;
  final String decoder;
  final String stitch;
  final String hubert;
  final String lmdm;

  /// YuNet face detector ONNX. Used by [FaceCropService.yunet]. May
  /// fall back to the conventional `onnx/yunet.onnx` location when the
  /// manifest omits it (older snapshots).
  final String faceDetector;
}

final class LivePortraitAudioConfig {
  const LivePortraitAudioConfig({
    required this.sampleRate,
    required this.hopFrames,
    required this.featureDim,
  });

  factory LivePortraitAudioConfig.fromJson(Map<String, Object?> json) =>
      LivePortraitAudioConfig(
        sampleRate: (json['sampleRate'] as num).toInt(),
        hopFrames: (json['hopFrames'] as num).toInt(),
        featureDim: (json['featureDim'] as num).toInt(),
      );

  final int sampleRate;
  final int hopFrames;
  final int featureDim;
}

final class LivePortraitRenderConfig {
  const LivePortraitRenderConfig({
    required this.frameWidth,
    required this.frameHeight,
    required this.internalRes,
    required this.fpsTarget,
  });

  factory LivePortraitRenderConfig.fromJson(Map<String, Object?> json) =>
      LivePortraitRenderConfig(
        frameWidth: (json['frameWidth'] as num).toInt(),
        frameHeight: (json['frameHeight'] as num).toInt(),
        internalRes: (json['internalRes'] as num).toInt(),
        fpsTarget: (json['fpsTarget'] as num).toInt(),
      );

  final int frameWidth;
  final int frameHeight;
  final int internalRes;
  final int fpsTarget;
}

final class LivePortraitMotionConfig {
  const LivePortraitMotionConfig({
    required this.keypointCount,
    required this.appearanceVolume,
  });

  factory LivePortraitMotionConfig.fromJson(Map<String, Object?> json) =>
      LivePortraitMotionConfig(
        keypointCount: (json['keypointCount'] as num).toInt(),
        appearanceVolume:
            (json['appearanceVolume'] as List<Object?>)
                .map((e) => (e as num).toInt())
                .toList(growable: false),
      );

  final int keypointCount;
  final List<int> appearanceVolume;
}

final class LivePortraitSamplerConfig {
  const LivePortraitSamplerConfig({
    required this.kind,
    required this.steps,
    required this.guidance,
    required this.windowFrames,
  });

  factory LivePortraitSamplerConfig.fromJson(Map<String, Object?> json) =>
      LivePortraitSamplerConfig(
        kind: json['kind'] as String,
        steps: (json['steps'] as num).toInt(),
        guidance: (json['guidance'] as num).toDouble(),
        windowFrames: (json['windowFrames'] as num).toInt(),
      );

  final String kind;
  final int steps;
  final double guidance;
  final int windowFrames;
}

final class LivePortraitConfig {
  const LivePortraitConfig({
    required this.weights,
    required this.audio,
    required this.render,
    required this.motion,
    required this.sampler,
    required this.source,
  });

  factory LivePortraitConfig.fromJson(Map<String, Object?> json) {
    final kind = json['kind'] as String?;
    if (kind != kLivePortraitSchemaVersion) {
      throw FormatException(
        'unexpected kind=$kind, want $kLivePortraitSchemaVersion',
      );
    }
    return LivePortraitConfig(
      source: (json['source'] as String?) ?? 'unknown',
      weights: LivePortraitWeightPaths.fromJson(
        json['weights']! as Map<String, Object?>,
      ),
      audio: LivePortraitAudioConfig.fromJson(
        json['audio']! as Map<String, Object?>,
      ),
      render: LivePortraitRenderConfig.fromJson(
        json['render']! as Map<String, Object?>,
      ),
      motion: LivePortraitMotionConfig.fromJson(
        json['motion']! as Map<String, Object?>,
      ),
      sampler: LivePortraitSamplerConfig.fromJson(
        json['sampler']! as Map<String, Object?>,
      ),
    );
  }

  static LivePortraitConfig fromJsonString(String text) {
    return LivePortraitConfig.fromJson(jsonDecode(text) as Map<String, Object?>);
  }

  final String source;
  final LivePortraitWeightPaths weights;
  final LivePortraitAudioConfig audio;
  final LivePortraitRenderConfig render;
  final LivePortraitMotionConfig motion;
  final LivePortraitSamplerConfig sampler;
}
