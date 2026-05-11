/// Top-level LivePortrait engine.
///
/// Wires the four subsystems:
///
///   * [FaceCropService]      — one-time per source portrait
///   * [AppearanceExtractor]  — bake source appearance volume
///   * [MotionExtractor]      — bake source pose/expression
///   * [HubertEncoder]        — audio → 25 Hz features
///   * [LmdmSampler]          — features → motion latents
///   * [PortraitRenderer]     — per-frame warp + decode
///
/// Designed for cmdspace-app's "buddy halfBody mode": one source
/// portrait file (e.g. `assets/buddies/girlfriend/looks/look_02/portrait.png`),
/// audio comes from the buddy's TTS pipeline, output frames replace
/// the layered rig view while the buddy is speaking.
library;

import 'dart:async';
import 'dart:typed_data';

import 'audio/hubert.dart';
import 'audio/lmdm.dart';
import 'audio_motion.dart';
import 'extractors/appearance.dart';
import 'extractors/motion.dart';
import 'face_crop.dart';
import 'loader.dart';
import 'renderer.dart';

final class LivePortraitEngine {
  LivePortraitEngine._({
    required this.snapshot,
    required FaceCropService faceCrop,
    required AppearanceExtractor appearance,
    required MotionExtractor motion,
    required HubertEncoder hubert,
    required LmdmSampler lmdm,
    required PortraitRenderer renderer,
  }) : _faceCrop = faceCrop,
       _appearance = appearance,
       _motion = motion,
       _hubert = hubert,
       _lmdm = lmdm,
       _renderer = renderer;

  /// Load all sessions from a converted snapshot directory.
  factory LivePortraitEngine.load({
    required String snapshotDir,
    required String faceDetectorOnnxPath,
  }) {
    final snapshot = LivePortraitSnapshot.open(snapshotDir);
    final faceCrop = FaceCropService.yunet(onnxPath: faceDetectorOnnxPath);
    final appearance = AppearanceExtractor.load(
      onnxPath: snapshot.pathFor(LivePortraitModule.appearance),
    );
    final motion = MotionExtractor.load(
      onnxPath: snapshot.pathFor(LivePortraitModule.motion),
    );
    final hubert = HubertEncoder.load(
      onnxPath: snapshot.pathFor(LivePortraitModule.hubert),
    );
    final lmdm = LmdmSampler.load(
      onnxPath: snapshot.pathFor(LivePortraitModule.lmdm),
    );
    final renderer = PortraitRenderer.mlx(
      config: snapshot.config,
      snapshot: snapshot,
    );
    return LivePortraitEngine._(
      snapshot: snapshot,
      faceCrop: faceCrop,
      appearance: appearance,
      motion: motion,
      hubert: hubert,
      lmdm: lmdm,
      renderer: renderer,
    );
  }

  final LivePortraitSnapshot snapshot;
  final FaceCropService _faceCrop;
  final AppearanceExtractor _appearance;
  final MotionExtractor _motion;
  final HubertEncoder _hubert;
  final LmdmSampler _lmdm;
  final PortraitRenderer _renderer;

  SourceState? _activeSource;
  AudioMotionPipeline? _pipeline;
  StreamController<RenderedFrame>? _frameController;

  /// Bake a portrait into a reusable [SourceState].
  Future<SourceState> bakePortrait({
    required Uint8List portraitRgb,
    required int width,
    required int height,
  }) async {
    final crop = _faceCrop.cropPortrait(
      sourceRgb: portraitRgb,
      sourceWidth: width,
      sourceHeight: height,
    );
    return SourceState.bake(
      crop512Rgb: crop.cropRgb,
      appearance: _appearance,
      motion: _motion,
    );
  }

  /// Set the active portrait. Subsequent [pushAudio] calls will warp
  /// from this source. Resets any in-flight motion pipeline.
  void setActiveSource(SourceState source) {
    _activeSource = source;
    _pipeline?.reset();
    _pipeline = AudioMotionPipeline.create(
      config: snapshot.config,
      hubert: _hubert,
      sampler: _lmdm,
      source: source,
    );
  }

  /// Stream of rendered frames driven by [pushAudio].
  Stream<RenderedFrame> animate() {
    final controller = StreamController<RenderedFrame>();
    _frameController = controller;
    controller.onCancel = () {
      _pipeline?.reset();
      if (identical(_frameController, controller)) {
        _frameController = null;
      }
    };
    return controller.stream;
  }

  /// Push a chunk of mono 16 kHz PCM. Synchronously runs HuBERT + LMDM
  /// on the chunk, then renders each emitted motion frame and pushes
  /// the result onto the [animate] stream.
  ///
  /// Phase 3.5 limitation: this batches the whole chunk through one
  /// LMDM window pass — call once per utterance rather than streaming.
  ///
  /// [maxRenderFrames]: optional cap on how many of the produced motion
  /// frames are rendered + emitted. The remaining frames are
  /// discarded. Useful for smoke tests; for real usage leave null.
  void pushAudio(Float32List pcm16k, {int? maxRenderFrames}) {
    final source = _activeSource;
    final pipeline = _pipeline;
    if (source == null || pipeline == null) {
      throw StateError(
        'LivePortraitEngine: setActiveSource must be called before pushAudio',
      );
    }
    final motionFrames = pipeline.pushAudio(pcm16k);
    final controller = _frameController;
    final cap = maxRenderFrames ?? motionFrames.length;
    final n = cap < motionFrames.length ? cap : motionFrames.length;
    for (var i = 0; i < n; i++) {
      final drive = Driving.fromLatent(motionFrames[i].latent);
      final frame = _renderer.render(source: source, drive: drive);
      controller?.add(frame);
    }
  }

  /// Render a single frame for [drive] without going through the audio
  /// pipeline. Useful for tests.
  RenderedFrame renderFrame({required Driving drive}) {
    final source = _activeSource;
    if (source == null) {
      throw StateError('LivePortraitEngine.renderFrame: no active source');
    }
    return _renderer.render(source: source, drive: drive);
  }

  /// Discard buffered audio + motion state.
  void reset() {
    _pipeline?.reset();
  }

  /// Release ORT sessions held by the engine.
  void dispose() {
    _frameController?.close();
    _frameController = null;
    _appearance.close();
    _motion.close();
    _hubert.close();
    _lmdm.close();
    _renderer.close();
  }
}
