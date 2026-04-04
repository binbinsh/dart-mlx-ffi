library;

import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'config.dart';
import 'decoder_vocoder.dart';
import 'espeak_g2p.dart';
import 'loader.dart';
import 'model.dart';
import 'npz.dart';
import 'preprocess.dart';
import 'stft.dart';
import 'vocoder_core.dart';

/// End-to-end KittenTTS engine: text → audio float array.
///
/// Wires together G2P, front runner, source/STFT, decoder, and iSTFT
/// into a single synthesise call. All components use Dart-native MLX
/// operations (no Python-exported functions).
///
/// Usage:
/// ```dart
/// final engine = KittenTtsEngine.load(
///   snapshotPath: '/path/to/model',
///   espeakLibPath: '/path/to/libespeak-ng.dylib',
///   espeakDataPath: '/path/to/espeak-ng-data',
/// );
/// final result = engine.synthesise('Hello world');
/// // result.audio: MlxArray [1, numSamples]
/// // result.sampleRate: 24000
/// result.close();
/// engine.dispose();
/// ```
final class KittenTtsEngine {
  KittenTtsEngine._({
    required this.config,
    required this.front,
    required this.decoder,
    required this.voices,
    required this.g2p,
  });

  /// Loads all model components from a snapshot directory.
  ///
  /// [snapshotPath] — directory containing `model.safetensors`, `config.json`,
  ///   and `voices.npz`.
  /// [espeakLibPath] — path to `libespeak-ng.dylib`.
  /// [espeakDataPath] — path to the `espeak-ng-data` directory.
  /// [voice] — eSpeak voice identifier (default `en-us`).
  factory KittenTtsEngine.load({
    required String snapshotPath,
    required String espeakLibPath,
    required String espeakDataPath,
    String voice = 'en-us',
  }) {
    final config = ModelConfig.fromSnapshot(snapshotPath);
    final tensors = loadKittenTensors(snapshotPath);
    final voices = loadKittenVoices(snapshotPath, config);

    final front = KittenFrontRunner.fromTensors(
      config: config,
      tensors: tensors,
    );

    final activationQuant = config.activationQuantModules.contains('decoder');
    final decoder = KittenDecoder.load(
      tensors,
      prefix: 'decoder',
      config: config,
      activationQuant: activationQuant,
    );

    final g2p = EspeakG2p(
      libraryPath: espeakLibPath,
      dataPath: espeakDataPath,
      voice: voice,
    );

    return KittenTtsEngine._(
      config: config,
      front: front,
      decoder: decoder,
      voices: voices,
      g2p: g2p,
    );
  }

  final ModelConfig config;
  final KittenFrontRunner front;
  final KittenDecoder decoder;
  final Map<String, NpyArray> voices;
  final EspeakG2p g2p;

  /// Available voice names from the loaded voices.npz.
  List<String> get voiceNames {
    final resolved = <String>{...voices.keys};
    for (final entry in config.voiceAliases.entries) {
      if (voices.containsKey(entry.value)) {
        resolved.add(entry.key);
      }
    }
    return resolved.toList(growable: false)..sort();
  }

  /// Synthesises speech from [text].
  ///
  /// [voiceName] selects the reference voice (default: first available).
  /// [speed] controls speaking rate (1.0 = normal, lower = slower).
  ///
  /// Returns a [KittenTtsResult] containing the audio waveform and metadata.
  /// The caller must call [KittenTtsResult.close] when done.
  KittenTtsResult synthesise(
    String text, {
    String? voiceName,
    double speed = 1.0,
  }) {
    final refS = _loadRefS(voiceName);
    try {
      return _synthesiseWithRef(text, refS, speed: speed);
    } finally {
      refS.close();
    }
  }

  /// Synthesises using a pre-built reference style vector `[1, 256]`.
  KittenTtsResult synthesiseWithRef(
    String text,
    MlxArray refS, {
    double speed = 1.0,
  }) => _synthesiseWithRef(text, refS, speed: speed);

  KittenTtsResult _synthesiseWithRef(
    String text,
    MlxArray refS, {
    required double speed,
  }) {
    // 1. Text → phonemes → token IDs → MlxArray [1, seqLen].
    final inputIds = buildInputArrayFromText(text, g2p: g2p);

    // 2. Front runner: tokens + refS → asr, f0, noise, style, predDur.
    final frontResult = front.run(inputIds, refS, speed: speed);
    inputIds.close();

    // 3. Compute harmonic source from f0.
    final source = decoder.generator.sourceFromF0(frontResult.f0Pred);

    // 4. Forward STFT on the harmonic source to get har.
    final har = _buildHar(source);
    source.close();

    // 5. Decoder: asr + f0 + noise + style + har → projection(spec, phase).
    final projection = decoder.forwardProjection(
      asr: frontResult.asr,
      f0Curve: frontResult.f0Pred,
      noise: frontResult.nPred,
      style: frontResult.style,
      har: har,
    );
    har.close();
    frontResult.close();

    // 6. Inverse STFT: projection → audio waveform.
    final istftConfig = config.istftnetConfig;
    final audio = istftFromProjection(
      projection,
      nfft: istftConfig.genIstftNfft,
      hopSize: istftConfig.genIstftHopSize,
    );
    projection.close();

    // Force evaluation so the full graph is computed.
    audio.eval();

    return KittenTtsResult(audio: audio, sampleRate: config.sampleRate);
  }

  /// Builds the `har` tensor from the harmonic source output.
  ///
  /// The source module produces `sineMerge` of shape `[batch, samples, 1]`.
  /// We squeeze to `[batch, samples]`, compute the forward STFT to get
  /// `(harSpec, harPhase)`, concatenate along the frequency axis, and
  /// transpose to `[batch, frames, freqBins*2]` for the decoder.
  MlxArray _buildHar(KittenSourceOutput source) {
    final istftConfig = config.istftnetConfig;
    final nfft = istftConfig.genIstftNfft;
    final hopSize = istftConfig.genIstftHopSize;

    // sineMerge: [batch, samples, 1] → squeeze to [batch, samples].
    final sineMerge = source.sineMerge;
    final squeezed = sineMerge.ndim == 3
        ? sineMerge.reshape([sineMerge.shape[0], sineMerge.shape[1]])
        : sineMerge;

    // Forward STFT → (spec, phase) each [batch, freqBins, frames].
    final stftResult = stftTransform(squeezed, nfft: nfft, hopSize: hopSize);
    if (!identical(squeezed, sineMerge)) {
      squeezed.close();
    }

    // Concatenate spec and phase along freq axis: [batch, freqBins*2, frames].
    final harConcat = mx.concatenate([
      stftResult.spec,
      stftResult.phase,
    ], axis: 1);
    stftResult.spec.close();
    stftResult.phase.close();

    // Transpose to [batch, frames, freqBins*2] for the decoder.
    final har = mx.transposeAxes(harConcat, [0, 2, 1]);
    harConcat.close();
    return har;
  }

  /// Resolves a voice name to a reference style vector `[1, 256]`.
  MlxArray _loadRefS(String? voiceName) {
    var name = voiceName ?? voices.keys.first;
    // Resolve aliases.
    if (!voices.containsKey(name) && config.voiceAliases.containsKey(name)) {
      name = config.voiceAliases[name]!;
    }
    final npy = voices[name];
    if (npy == null) {
      throw ArgumentError(
        'Voice "$name" not found. Available: ${voices.keys.join(", ")}',
      );
    }
    // voices.npz entries are typically [256]. Reshape to [1, 256].
    final raw = npy.toMlxArray();
    if (raw.ndim == 1) {
      final reshaped = raw.reshape([1, raw.shape[0]]);
      raw.close();
      return reshaped;
    }
    return raw;
  }

  /// Resolves the effective speed for a voice, applying speed_priors if set.
  double effectiveSpeed(String? voiceName, {double baseSpeed = 1.0}) {
    if (voiceName == null) return baseSpeed;
    final prior = config.speedPriors[voiceName];
    if (prior == null) return baseSpeed;
    return baseSpeed * prior;
  }

  /// Releases all resources.
  void dispose() {
    g2p.dispose();
  }
}

/// Result of a KittenTTS synthesis call.
final class KittenTtsResult {
  const KittenTtsResult({required this.audio, required this.sampleRate});

  /// Audio waveform, shape `[1, numSamples]` (float32, range ~[-1, 1]).
  final MlxArray audio;

  /// Sample rate in Hz (typically 24000).
  final int sampleRate;

  /// Number of audio samples.
  int get numSamples => audio.shape.last;

  /// Duration of the audio in seconds.
  double get durationSeconds => numSamples / sampleRate;

  /// Extracts audio as a `Float32List` for playback.
  ///
  /// Returns a flat list of float32 samples.
  Float32List toFloat32List() {
    final flat = audio.reshape([audio.size]);
    try {
      return flat.toFloat32List();
    } finally {
      flat.close();
    }
  }

  /// Releases the audio MlxArray.
  void close() {
    audio.close();
  }
}
