/// LivePortrait audio-driven portrait animation engine.
///
/// **Status: Phase 1 partial.** Face crop is real (BlazeFace via ORT);
/// appearance/motion extraction, audio→motion sampling, and the
/// renderer still throw `UnimplementedError`.
///
/// This package targets a Dart-native MLX/CoreML port of the **Ditto**
/// fork of LivePortrait (Apache-2.0, ONNX-shipped, streaming-friendly):
///
///   https://github.com/antgroup/ditto-talkinghead
///   https://huggingface.co/digital-avatar/ditto-talkinghead
///
/// Why Ditto over the original KwaiVGI LivePortrait or JoyVASA:
///   * Apache-2.0 (cleaner commercial bundling than MIT-with-InsightFace).
///   * Already factored into independent `.onnx` modules for piecewise
///     conversion to MLX/CoreML.
///   * Designed for realtime streaming with HuBERT audio + Latent Motion
///     Diffusion Model (LMDM); no SVD backbone tax.
///
/// ## Pipeline (one source portrait + audio stream → animated frames)
///
/// One-time per portrait (cached on disk):
///   1. Face detect + crop (BlazeFace, NOT InsightFace) → 512×512 aligned.
///   2. AppearanceFeatureExtractor → `f_s` ≈ `[1,32,16,64,64]`.
///   3. MotionExtractor → canonical kp `x_c_s`, `R_s`, `t_s`, `δ_s`,
///      scale `s_s`. Compose source kp `x_s = s_s·(x_c_s @ R_s + δ_s) + t_s`.
///
/// Per audio chunk (streaming):
///   4. 16 kHz waveform → HuBERT → audio features `[T_a, 768]` @ 50 Hz.
///
/// Per output frame (target 25–30 fps):
///   5. LMDM samples motion latent → `(R_d, t_d, δ_d)`.
///   6. Compose driving kp `x_d`.
///   7. WarpingModule(f_s, x_s, x_d) → warped feature volume.
///   8. SpadeGenerator(warped) → RGB frame.
///   9. Stitching MLP pastes frame back into source crop.
///
/// ## Known blockers (see `docs/live_portrait_integration.md`)
///   * 3D `grid_sample` op missing in MLX — needs custom Metal kernel.
///   * SPADE generator's spatial normalization needs hand-fused ops.
///   * LMDM diffusion sampler latency — must implement KV-cached
///     streaming sampler; naive 50-step DDIM is too slow for realtime.
///
/// ## Non-goals for this scaffold
///   * No actual MLX inference yet — every method throws
///     `UnimplementedError` with a pointer to the missing piece.
///   * No video-driven path. Audio-only. The original LivePortrait video
///     driver is irrelevant for cmdspace's use case.
///   * No retargeting controls (eye/lip explicit gain) in v1.
library;

export 'audio/cond_builder.dart';
export 'audio/hubert.dart';
export 'audio/lmdm.dart';
export 'audio/motion_latent.dart';
export 'audio_motion.dart';
export 'config.dart';
export 'engine.dart';
export 'face_crop.dart';
export 'loader.dart';
export 'renderer.dart';
