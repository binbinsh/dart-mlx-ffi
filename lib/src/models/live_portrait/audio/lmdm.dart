/// Latent Motion Diffusion Model (LMDM) DDIM sampler.
///
/// Wraps `lmdm.onnx` and implements the cosine-beta DDIM loop from
/// Ditto's `core/models/lmdm.py`.
///
/// ## ONNX signature
///
///   inputs:
///     x          [1, 80, 265]   current motion latent
///     cond_frame [1,    265]    previous-frame motion (kp_cond)
///     cond       [1, 80, 1103]  per-frame audio+aux condition
///     time_cond  [1]            int64 DDIM timestep
///   outputs:
///     pred_noise [1, 80, 265]
///     x_start    [1, 80, 265]
///
/// ## Schedule
///
/// `n_timestep = 1000`, cosine schedule with `cosine_s = 8e-3`,
/// `eta = 1` (DDPM-style sampling, not pure DDIM `eta=0`).
/// Default `sampling_timesteps = 50`; we expose it so the realtime
/// path can drop to ~10.
///
/// ## Output layout
///
/// Flat row-major `[seqFrames, motionLatentDim]` (default `[80, 265]`).
/// The 265-dim layout per frame is:
///
///   scale  (1)  [delta from 1.0]
///   pitch (66)  bin distribution
///   yaw   (66)
///   roll  (66)
///   t     (3)
///   exp  (63)
///                                                = 265
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

import 'motion_latent.dart' show kMotionLatentDim;
export 'motion_latent.dart' show kMotionLatentDim;

const String _kLmdmFamily = 'live_portrait_lmdm';

/// Audio+aux condition dim (1024 hubert + 8 emo + 2 eye_open + 6 eye_ball + 63 sc).
const int kAudioCondDim = 1103;

/// Default sliding-window length in 25 Hz frames (3.2 s).
const int kSeqFrames = 80;

/// Total diffusion timesteps the model was trained with.
const int kDiffusionTimesteps = 1000;

/// One DDIM step pre-computed from the cosine schedule.
final class _DdimStep {
  const _DdimStep({
    required this.timeCond,
    required this.alphaNextSqrt,
    required this.c,
    required this.sigma,
    required this.lastStep,
  });
  final int timeCond;
  final double alphaNextSqrt;
  final double c;
  final double sigma;
  final bool lastStep;
}

/// LMDM ORT wrapper + DDIM sampler.
final class LmdmSampler {
  LmdmSampler._({
    required DartOnnxSession session,
    required this.seqFrames,
    required this.motionDim,
    required this.audioCondDim,
  }) : _session = session;

  factory LmdmSampler.load({
    required String onnxPath,
    int seqFrames = kSeqFrames,
    int motionDim = kMotionLatentDim,
    int audioCondDim = kAudioCondDim,
    int numThreads = 2,
  }) {
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: onnxPath,
        id: _kLmdmFamily,
        family: _kLmdmFamily,
        provider: 'cpu',
        requireProvider: false,
        numThreads: numThreads,
      ),
    );
    final diag = session.diagnostics;
    final inputs = (diag['input_metadata'] as List).cast<Map>();
    final outputs = (diag['output_metadata'] as List).cast<Map>();
    final inNames = inputs.map((m) => m['name'] as String).toSet();
    const expectedIn = {'x', 'cond_frame', 'cond', 'time_cond'};
    if (!expectedIn.every(inNames.contains)) {
      session.close();
      throw StateError(
        'LmdmSampler: ONNX missing expected inputs '
        '${expectedIn.difference(inNames)} (got $inNames)',
      );
    }
    final outNames = outputs.map((m) => m['name'] as String).toSet();
    const expectedOut = {'pred_noise', 'x_start'};
    if (!expectedOut.every(outNames.contains)) {
      session.close();
      throw StateError(
        'LmdmSampler: ONNX missing expected outputs '
        '${expectedOut.difference(outNames)} (got $outNames)',
      );
    }
    return LmdmSampler._(
      session: session,
      seqFrames: seqFrames,
      motionDim: motionDim,
      audioCondDim: audioCondDim,
    );
  }

  final DartOnnxSession _session;
  final int seqFrames;
  final int motionDim;
  final int audioCondDim;

  // Cached schedule, regenerated when `samplingTimesteps` changes.
  int? _cachedSamplingTimesteps;
  late List<_DdimStep> _schedule;
  late Float32List _alphasCumprod; // length kDiffusionTimesteps

  /// Sample one full window of motion latents.
  ///
  /// Inputs:
  ///   * [kpCond] — previous frame motion latent, length [motionDim]
  ///     (use the source motion or last produced frame).
  ///   * [audioCond] — `[seqFrames, audioCondDim]` flat.
  ///   * [samplingTimesteps] — DDIM steps (default 50, realtime ~10).
  ///   * [rng] — optional seeded RNG for deterministic tests.
  ///
  /// Returns flat `[seqFrames, motionDim]` motion latents.
  Float32List sample({
    required Float32List kpCond,
    required Float32List audioCond,
    int samplingTimesteps = 50,
    math.Random? rng,
  }) {
    if (kpCond.length != motionDim) {
      throw ArgumentError(
        'sample: kpCond length ${kpCond.length} != motionDim $motionDim',
      );
    }
    final expectedAud = seqFrames * audioCondDim;
    if (audioCond.length != expectedAud) {
      throw ArgumentError(
        'sample: audioCond length ${audioCond.length} != $expectedAud',
      );
    }
    _ensureSchedule(samplingTimesteps);
    final r = rng ?? math.Random();

    // x ~ N(0, I)
    var x = _gaussian(seqFrames * motionDim, r);

    Float32List? xStart;
    for (var i = 0; i < _schedule.length; i++) {
      final step = _schedule[i];
      final result = _runOne(
        x: x,
        kpCond: kpCond,
        audioCond: audioCond,
        timeCond: step.timeCond,
      );
      final predNoise = result.predNoise;
      xStart = result.xStart;
      if (step.lastStep) {
        x = xStart;
        continue;
      }
      // x = x_start * alpha_next_sqrt + c * pred_noise + sigma * noise
      final noise = _gaussian(x.length, r);
      final nx = Float32List(x.length);
      final aSqrt = step.alphaNextSqrt;
      final c = step.c;
      final sigma = step.sigma;
      for (var j = 0; j < x.length; j++) {
        nx[j] = xStart[j] * aSqrt + c * predNoise[j] + sigma * noise[j];
      }
      x = nx;
    }
    return x;
  }

  void _ensureSchedule(int samplingTimesteps) {
    if (_cachedSamplingTimesteps == samplingTimesteps) return;
    _cachedSamplingTimesteps = samplingTimesteps;
    _alphasCumprod = _makeAlphasCumprod();

    // times = linspace(-1, T-1, sampling_timesteps+1).int(), reversed
    final pts = <int>[];
    final n = samplingTimesteps + 1;
    for (var i = 0; i < n; i++) {
      final v = -1.0 + (kDiffusionTimesteps - 1 + 1) * (i / (n - 1));
      pts.add(v.toInt());
    }
    final times = pts.reversed.toList();
    final pairs = <(int, int)>[];
    for (var i = 0; i < times.length - 1; i++) {
      pairs.add((times[i], times[i + 1]));
    }

    const eta = 1.0;
    final schedule = <_DdimStep>[];
    for (final p in pairs) {
      final time = p.$1;
      final timeNext = p.$2;
      if (timeNext < 0) {
        schedule.add(
          _DdimStep(
            timeCond: time,
            alphaNextSqrt: 0,
            c: 0,
            sigma: 0,
            lastStep: true,
          ),
        );
        continue;
      }
      final alpha = _alphasCumprod[time];
      final alphaNext = _alphasCumprod[timeNext];
      final sigma = eta *
          math.sqrt((1 - alpha / alphaNext) * (1 - alphaNext) / (1 - alpha));
      final c = math.sqrt(1 - alphaNext - sigma * sigma);
      schedule.add(
        _DdimStep(
          timeCond: time,
          alphaNextSqrt: math.sqrt(alphaNext),
          c: c,
          sigma: sigma,
          lastStep: false,
        ),
      );
    }
    _schedule = schedule;
  }

  /// Cosine alphas_cumprod, identical to `make_beta` in Ditto.
  Float32List _makeAlphasCumprod() {
    const n = kDiffusionTimesteps;
    const cosineS = 8e-3;
    final alphasBar = Float32List(n + 1);
    for (var i = 0; i <= n; i++) {
      final t = (i / n + cosineS) / (1 + cosineS) * math.pi / 2;
      final c = math.cos(t);
      alphasBar[i] = c * c;
    }
    final base = alphasBar[0];
    for (var i = 0; i <= n; i++) {
      alphasBar[i] = alphasBar[i] / base;
    }
    // betas = 1 - alphas[1:] / alphas[:-1]
    // alphas = 1 - betas
    // alphas_cumprod = cumprod(alphas)
    final out = Float32List(n);
    var prod = 1.0;
    for (var i = 0; i < n; i++) {
      var beta = 1.0 - alphasBar[i + 1] / alphasBar[i];
      if (beta < 0) beta = 0;
      if (beta > 0.999) beta = 0.999;
      final a = 1.0 - beta;
      prod *= a;
      out[i] = prod;
    }
    return out;
  }

  ({Float32List predNoise, Float32List xStart}) _runOne({
    required Float32List x,
    required Float32List kpCond,
    required Float32List audioCond,
    required int timeCond,
  }) {
    final xT = RuntimeTensor.float32([1, seqFrames, motionDim], x);
    final kpT = RuntimeTensor.float32([1, motionDim], kpCond);
    final cT = RuntimeTensor.float32([1, seqFrames, audioCondDim], audioCond);
    final tT = RuntimeTensor.int64([1], Int64List.fromList([timeCond]));
    final result = _session.run({
      'x': xT,
      'cond_frame': kpT,
      'cond': cT,
      'time_cond': tT,
    });
    try {
      final pn = (result.outputs['pred_noise'] as RuntimeTensor).asFloat32List();
      final xs = (result.outputs['x_start'] as RuntimeTensor).asFloat32List();
      return (
        predNoise: Float32List.fromList(pn),
        xStart: Float32List.fromList(xs),
      );
    } finally {
      result.close();
    }
  }

  static Float32List _gaussian(int n, math.Random r) {
    final out = Float32List(n);
    var i = 0;
    while (i < n) {
      // Box–Muller
      final u1 = math.max(r.nextDouble(), 1e-12);
      final u2 = r.nextDouble();
      final mag = math.sqrt(-2.0 * math.log(u1));
      final z0 = mag * math.cos(2 * math.pi * u2);
      out[i++] = z0;
      if (i < n) {
        final z1 = mag * math.sin(2 * math.pi * u2);
        out[i++] = z1;
      }
    }
    return out;
  }

  void close() => _session.close();
}
