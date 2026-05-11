part of 'cosyvoice2_mlx.dart';

final class CosyVoice2LowerBundle {
  CosyVoice2LowerBundle._({
    required this.bundlePath,
    required this.sampleRate,
    required MlxImportedFunction tokensToAudio,
  }) : _tokensToAudio = tokensToAudio;

  factory CosyVoice2LowerBundle.load(String bundlePath) {
    final metaFile = File('$bundlePath/meta.json');
    if (!metaFile.existsSync()) {
      throw StateError('Missing CosyVoice2 meta.json in $bundlePath');
    }
    final meta =
        jsonDecode(metaFile.readAsStringSync()) as Map<String, Object?>;
    final sampleRate = (meta['sample_rate'] as num?)?.toInt() ?? 24000;
    final tokensToAudio = MlxExport.importFunction(
      '$bundlePath/tokens_to_audio.mlxfn',
    );
    return CosyVoice2LowerBundle._(
      bundlePath: bundlePath,
      sampleRate: sampleRate,
      tokensToAudio: tokensToAudio,
    );
  }

  final String bundlePath;
  final int sampleRate;
  final MlxImportedFunction _tokensToAudio;

  CosyVoice2LowerResult synthesise(List<int> tokens, {int? seed}) {
    if (tokens.isEmpty) {
      throw ArgumentError('Speech tokens must not be empty.');
    }
    if (seed != null) {
      MlxRuntime.seed(seed);
    }
    final ids = MlxArray.fromInt32List(tokens, shape: [1, tokens.length]);
    try {
      final outputs = _tokensToAudio.call([ids]);
      if (outputs.length != 1) {
        throw StateError(
          'Expected 1 output from tokens_to_audio, got ${outputs.length}.',
        );
      }
      final audio = outputs[0].astype(MlxDType.MLX_FLOAT32);
      outputs[0].close();
      MlxRuntime.evalAll([audio]);
      return CosyVoice2LowerResult(audio: audio, sampleRate: sampleRate);
    } finally {
      ids.close();
    }
  }

  void close() {
    _tokensToAudio.close();
  }
}

final class CosyVoice2LowerResult {
  const CosyVoice2LowerResult({required this.audio, required this.sampleRate});

  final MlxArray audio;
  final int sampleRate;

  Float32List toFloat32List() {
    final flat = audio.reshape([audio.size]);
    try {
      return flat.toFloat32List();
    } finally {
      flat.close();
    }
  }

  void close() {
    audio.close();
  }
}

final class CosyVoice2FlowBundle {
  CosyVoice2FlowBundle._({
    required this.bundlePath,
    required this.bucketTokens,
    required this.promptMelFrames,
    required MlxImportedFunction tokensToMel,
  }) : _tokensToMel = tokensToMel;

  factory CosyVoice2FlowBundle.load(
    String bundlePath, {
    required int bucketTokens,
  }) {
    final metaFile = File('$bundlePath/meta.json');
    if (!metaFile.existsSync()) {
      throw StateError('Missing CosyVoice2 meta.json in $bundlePath');
    }
    final meta =
        jsonDecode(metaFile.readAsStringSync()) as Map<String, Object?>;
    final flowFile = File('$bundlePath/tokens_to_mel_$bucketTokens.mlxfn');
    if (!flowFile.existsSync()) {
      throw StateError(
        'Missing CosyVoice2 flow bucket export: ${flowFile.path}',
      );
    }
    final tokensToMel = MlxExport.importFunction(flowFile.path);
    return CosyVoice2FlowBundle._(
      bundlePath: bundlePath,
      bucketTokens: bucketTokens,
      promptMelFrames: (meta['prompt_mel_len'] as num?)?.toInt() ?? 0,
      tokensToMel: tokensToMel,
    );
  }

  final String bundlePath;
  final int bucketTokens;
  final int promptMelFrames;
  final MlxImportedFunction _tokensToMel;

  Float32List synthesiseTokens(List<int> tokens, {int? seed}) {
    if (tokens.isEmpty) {
      throw ArgumentError('Speech tokens must not be empty.');
    }
    if (tokens.length > bucketTokens) {
      throw StateError(
        'Speech token count ${tokens.length} exceeds flow bucket $bucketTokens.',
      );
    }
    if (seed != null) {
      MlxRuntime.seed(seed);
    }
    final padded = <int>[
      ...tokens,
      ...List<int>.filled(bucketTokens - tokens.length, 0),
    ];
    final ids = MlxArray.fromInt32List(padded, shape: [1, bucketTokens]);
    final tokenLen = MlxArray.fromInt32List([tokens.length], shape: [1]);
    final noise = mx.random.normal([1, 80, promptMelFrames + bucketTokens * 2]);
    try {
      final outputs = _tokensToMel.call([ids, tokenLen, noise]);
      if (outputs.length != 1) {
        throw StateError(
          'Expected 1 output from tokens_to_mel, got ${outputs.length}.',
        );
      }
      final full = outputs[0].astype(MlxDType.MLX_FLOAT32);
      outputs[0].close();
      final melFrames = tokens.length * 2;
      final trimmed = full.slice(start: [0, 0, 0], stop: [1, 80, melFrames]);
      full.close();
      MlxRuntime.evalAll([trimmed]);
      try {
        return trimmed.toFloat32List();
      } finally {
        trimmed.close();
      }
    } finally {
      ids.close();
      tokenLen.close();
      noise.close();
    }
  }

  void close() {
    _tokensToMel.close();
  }
}

final class CosyVoice2VocoderBundle {
  CosyVoice2VocoderBundle._({
    required this.bundlePath,
    required this.sampleRate,
    required this.hopSize,
    required this.maxFrames,
    required MlxImportedFunction melToAudio,
  }) : _melToAudio = melToAudio;

  factory CosyVoice2VocoderBundle.load(String bundlePath, {int? bucketFrames}) {
    final metaFile = File('$bundlePath/meta.json');
    if (!metaFile.existsSync()) {
      throw StateError('Missing CosyVoice2 meta.json in $bundlePath');
    }
    final meta =
        jsonDecode(metaFile.readAsStringSync()) as Map<String, Object?>;
    final sampleRate = (meta['sample_rate'] as num?)?.toInt() ?? 24000;
    final hopSize = (meta['hop_size'] as num?)?.toInt() ?? 480;
    final buckets =
        (meta['vocoder_buckets'] as List<Object?>?)
            ?.map((value) => (value as num).toInt())
            .toList(growable: false) ??
        const <int>[4096];
    final maxFrames = bucketFrames ?? buckets.last;
    final vocoderFile = File('$bundlePath/mel_to_audio_$maxFrames.mlxfn');
    final melToAudio = MlxExport.importFunction(
      vocoderFile.existsSync()
          ? vocoderFile.path
          : '$bundlePath/mel_to_audio.mlxfn',
    );
    return CosyVoice2VocoderBundle._(
      bundlePath: bundlePath,
      sampleRate: sampleRate,
      hopSize: hopSize,
      maxFrames: maxFrames,
      melToAudio: melToAudio,
    );
  }

  final String bundlePath;
  final int sampleRate;
  final int hopSize;
  final int maxFrames;
  final MlxImportedFunction _melToAudio;

  CosyVoice2LowerResult synthesiseMel(
    Float32List mel, {
    required List<int> shape,
  }) {
    final frames = shape.length >= 3 ? shape[2] : 0;
    if (frames <= 0) {
      throw ArgumentError('Mel shape must be [1, 80, T].');
    }
    if (frames > maxFrames) {
      throw StateError(
        'Mel frame count $frames exceeds exported vocoder maxFrames $maxFrames.',
      );
    }
    final melArray = MlxArray.fromFloat32List(mel, shape: shape);
    try {
      MlxArray input = melArray;
      if (frames < maxFrames) {
        final padding = mx.zeros([shape[0], shape[1], maxFrames - frames]);
        input = mx.concatenate([melArray, padding], axis: 2);
        padding.close();
      }
      final outputs = _melToAudio.call([input]);
      if (!identical(input, melArray)) {
        input.close();
      }
      if (outputs.length != 1) {
        throw StateError(
          'Expected 1 output from mel_to_audio, got ${outputs.length}.',
        );
      }
      final expectedSamples = frames * hopSize;
      final trimmed = outputs[0].slice(
        start: [0, 0],
        stop: [1, expectedSamples],
      );
      final audio = trimmed.astype(MlxDType.MLX_FLOAT32);
      trimmed.close();
      outputs[0].close();
      MlxRuntime.evalAll([audio]);
      return CosyVoice2LowerResult(audio: audio, sampleRate: sampleRate);
    } finally {
      melArray.close();
    }
  }

  void close() {
    _melToAudio.close();
  }
}
