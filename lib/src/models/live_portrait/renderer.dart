/// LivePortrait warp + decode renderer.
///
/// Per-frame inference path:
///
///   x_s        = transform_keypoint(kp_c, R_s, exp_s, scale_s, t_s)
///   x_d        = transform_keypoint(kp_c, R_d, exp_d, scale_d, t_d)
///   x_d_adj    = StitchNetwork(x_s, x_d)
///   warped     = WarpNetwork(f_s, x_s, x_d_adj)
///   rgb_tanh   = SpadeDecoder(warped)
///   rgb_uint8  = (rgb_tanh * 0.5 + 0.5) * 255
///
/// All sub-networks are independent ORT sessions:
///   * `warp_network_v2.onnx` (~190 MB; rewritten GridSample3D → opset 20 GridSample)
///   * `decoder.onnx`         (~212 MB)
///   * `stitch_network.onnx`  (~2.3 MB)
///
/// Pixel-space paste-back into the original full frame is **not**
/// done here — the caller handles it via the YuNet crop transform
/// stored on [SourceState]'s sibling `FaceCropResult`. This renderer
/// only emits the 512×512 face crop.
library;

import 'dart:typed_data';

import 'audio/motion_latent.dart';
import 'config.dart';
import 'extractors/appearance.dart';
import 'extractors/image_preprocess.dart';
import 'extractors/motion.dart' as motion_ex;
import 'loader.dart';
import 'renderers/decoder.dart';
import 'renderers/stitch.dart';
import 'renderers/warp.dart';

/// Cached source-portrait state. Built once per portrait via
/// [SourceState.bake] and reused for every output frame.
final class SourceState {
  const SourceState({
    required this.appearanceVolume,
    required this.canonicalKeypoints,
    required this.rotation,
    required this.translation,
    required this.expression,
    required this.scale,
    required this.pitchDeg,
    required this.yawDeg,
    required this.rollDeg,
    required this.pitchBins,
    required this.yawBins,
    required this.rollBins,
  });

  final Float32List appearanceVolume;
  final Float32List canonicalKeypoints;
  final Float32List rotation;
  final Float32List translation;
  final Float32List expression;
  final double scale;
  final double pitchDeg;
  final double yawDeg;
  final double rollDeg;
  final Float32List pitchBins;
  final Float32List yawBins;
  final Float32List rollBins;

  /// Build a [SourceState] from an aligned 512×512 RGB face crop.
  static SourceState bake({
    required Uint8List crop512Rgb,
    required AppearanceExtractor appearance,
    required motion_ex.MotionExtractor motion,
  }) {
    final nchw = rgb512ToExtractorNchw(sourceRgb: crop512Rgb);
    final volume = appearance.extract(nchw);
    final desc = motion.extract(nchw);
    return SourceState(
      appearanceVolume: volume,
      canonicalKeypoints: desc.canonicalKeypoints,
      rotation: desc.rotation,
      translation: desc.translation,
      expression: desc.expression,
      scale: desc.scale,
      pitchDeg: desc.pitchDeg,
      yawDeg: desc.yawDeg,
      rollDeg: desc.rollDeg,
      pitchBins: desc.pitchBins,
      yawBins: desc.yawBins,
      rollBins: desc.rollBins,
    );
  }

  /// Compute the transformed source keypoints `x_s` once per source.
  /// Cached on the renderer to avoid recomputation across frames.
  Float32List computeSourceKeypoints() {
    return motion_ex.transformKeypoints(
      canonicalKp: canonicalKeypoints,
      rotation: rotation,
      expression: expression,
      scale: scale,
      translation: translation,
    );
  }
}

/// One rendered output frame (face crop, pre paste-back).
final class RenderedFrame {
  const RenderedFrame({
    required this.rgb,
    required this.width,
    required this.height,
  });

  final Uint8List rgb;
  final int width;
  final int height;
}

/// Driving motion info for a single frame. Construct from a
/// [MotionFrame] (LMDM output) via [Driving.fromLatent], or directly
/// for tests.
final class Driving {
  const Driving({
    required this.pitchDeg,
    required this.yawDeg,
    required this.rollDeg,
    required this.translation,
    required this.expression,
    required this.scale,
  });

  final double pitchDeg;
  final double yawDeg;
  final double rollDeg;
  final Float32List translation;
  final Float32List expression;
  final double scale;

  /// Build from a 265-dim motion latent (LMDM output, post
  /// [unpackMotionLatent]).
  factory Driving.fromLatent(Float32List latent265) {
    final u = unpackMotionLatent(latent265);
    return Driving(
      pitchDeg: motion_ex.bin66ToDegree(u.pitchBins),
      yawDeg: motion_ex.bin66ToDegree(u.yawBins),
      rollDeg: motion_ex.bin66ToDegree(u.rollBins),
      translation: u.translation,
      expression: u.expression,
      scale: u.scale,
    );
  }

  /// Build from a [SourceState] (identity render — drive == source).
  /// Useful for smoke-testing the renderer pipeline with a known
  /// expected output (the source crop, modulo warp/decode error).
  factory Driving.identity(SourceState s) => Driving(
    pitchDeg: s.pitchDeg,
    yawDeg: s.yawDeg,
    rollDeg: s.rollDeg,
    translation: Float32List.fromList(s.translation),
    expression: Float32List.fromList(s.expression),
    scale: s.scale,
  );
}

/// Per-frame renderer.
abstract class PortraitRenderer {
  RenderedFrame render({required SourceState source, required Driving drive});

  /// Whether stitching is applied. Defaults true (matches Ditto
  /// `flag_stitching=True`). Disable for ablation/debugging.
  bool get stitchEnabled;

  void close();

  factory PortraitRenderer.mlx({
    required LivePortraitConfig config,
    required LivePortraitSnapshot snapshot,
    bool stitchEnabled,
  }) = _OrtPortraitRenderer.fromSnapshot;
}

class _OrtPortraitRenderer implements PortraitRenderer {
  _OrtPortraitRenderer._({
    required this.config,
    required WarpNetwork warp,
    required SpadeDecoder decoder,
    required StitchNetwork stitch,
    required this.stitchEnabled,
  }) : _warp = warp,
       _decoder = decoder,
       _stitch = stitch;

  factory _OrtPortraitRenderer.fromSnapshot({
    required LivePortraitConfig config,
    required LivePortraitSnapshot snapshot,
    bool stitchEnabled = true,
  }) {
    return _OrtPortraitRenderer._(
      config: config,
      warp: WarpNetwork.load(
        onnxPath: snapshot.pathFor(LivePortraitModule.warp),
      ),
      decoder: SpadeDecoder.load(
        onnxPath: snapshot.pathFor(LivePortraitModule.decoder),
      ),
      stitch: StitchNetwork.load(
        onnxPath: snapshot.pathFor(LivePortraitModule.stitch),
      ),
      stitchEnabled: stitchEnabled,
    );
  }

  final LivePortraitConfig config;
  final WarpNetwork _warp;
  final SpadeDecoder _decoder;
  final StitchNetwork _stitch;

  @override
  final bool stitchEnabled;

  @override
  RenderedFrame render({
    required SourceState source,
    required Driving drive,
  }) {
    // 1. Source kps (could be cached per source — keep simple for now).
    final xS = source.computeSourceKeypoints();

    // 2. Driving kps using SOURCE canonical kps (LivePortrait shares
    //    the canonical basis between source and driving — only R/exp/
    //    scale/t differ frame-to-frame).
    final driveR = motion_ex.ditttoRotationMatrix(
      pitchDeg: drive.pitchDeg,
      yawDeg: drive.yawDeg,
      rollDeg: drive.rollDeg,
    );
    final xD = motion_ex.transformKeypoints(
      canonicalKp: source.canonicalKeypoints,
      rotation: driveR,
      expression: drive.expression,
      scale: drive.scale,
      translation: drive.translation,
    );

    // 3. Stitch (kp adjust, not pixel paste).
    final xDFinal = stitchEnabled
        ? _stitch.stitch(kpSource: xS, kpDriving: xD)
        : xD;

    // 4. Warp.
    final warped = _warp.run(
      appearanceVolume: source.appearanceVolume,
      kpSource: xS,
      kpDriving: xDFinal,
    );

    // 5. Decode → tanh-RGB.
    final tanh = _decoder.decode(warped);

    // 6. Denorm to uint8 RGB 512×512.
    final rgb = nchwTanhToRgb(nchw: tanh, width: 512, height: 512);

    return RenderedFrame(rgb: rgb, width: 512, height: 512);
  }

  @override
  void close() {
    _warp.close();
    _decoder.close();
    _stitch.close();
  }
}
