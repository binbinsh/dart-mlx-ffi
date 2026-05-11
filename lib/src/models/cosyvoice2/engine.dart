part of 'cosyvoice2_mlx.dart';

final class CosyVoice2Engine {
  const CosyVoice2Engine({this.modelId = defaultModelId});

  static const String defaultModelId = 'mlx-community/CosyVoice2-0.5B-4bit';
  static final Map<String, CosyVoice2UpperRunner> _runnerCache =
      <String, CosyVoice2UpperRunner>{};
  static final Map<String, CosyVoice2FlowBundle> _flowCache =
      <String, CosyVoice2FlowBundle>{};
  static final Map<String, CosyVoice2VocoderBundle> _vocoderCache =
      <String, CosyVoice2VocoderBundle>{};
  static const List<int> _defaultFlowBuckets = <int>[
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    88,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    124,
    128,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    544,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
    800,
    832,
    864,
    896,
    928,
    960,
    992,
    1024,
  ];
  static const List<int> _defaultVocoderBuckets = <int>[
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    36,
    38,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    58,
    60,
    62,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    252,
    256,
    288,
    304,
    320,
    336,
    352,
    368,
    384,
    400,
    416,
    432,
    448,
    464,
    480,
    496,
    512,
    576,
    640,
    704,
    768,
    832,
    896,
    960,
    1024,
    1088,
    1152,
    1216,
    1280,
    1344,
    1408,
    1472,
    1536,
    1600,
    1664,
    1728,
    1792,
    1856,
    1920,
    1984,
    2048,
  ];

  final String modelId;

  static String promptCacheKey({
    String modelId = defaultModelId,
    required String refAudioPath,
    required String refText,
  }) {
    return _stableHash('$modelId\n$refAudioPath\n$refText');
  }

  static String promptRootPath({String? home}) {
    final resolvedHome = home ??
        Platform.environment['HOME'] ??
        Platform.environment['USERPROFILE'] ??
        Directory.current.path;
    return _joinPath(
      _joinPath(_joinPath(resolvedHome, '.cmdspace'), 'models'),
      'cosyvoice2_prompts',
    );
  }

  static String promptBundleDir({
    String modelId = defaultModelId,
    String? home,
    required String refAudioPath,
    required String refText,
  }) {
    return _joinPath(
      promptRootPath(home: home),
      promptCacheKey(
        modelId: modelId,
        refAudioPath: refAudioPath,
        refText: refText,
      ),
    );
  }

  Future<CosyVoice2Result> synthesise(
    String text, {
    required String refAudioPath,
    required String refText,
    String? instructText,
    double speed = 1.0,
  }) async {
    final source = text.trim();
    final refCaption = refText.trim();
    if (source.isEmpty) {
      throw ArgumentError('Text must not be empty.');
    }
    if (refCaption.isEmpty) {
      throw ArgumentError('refText must not be empty.');
    }

    final promptDir = await _resolvePromptBundleDir(refAudioPath.trim(), refCaption);
    final meta =
        jsonDecode(await File(_joinPath(promptDir, 'meta.json')).readAsString())
            as Map<String, dynamic>;
    final snapshotPath = meta['snapshot_path']?.toString() ?? '';
    if (snapshotPath.isEmpty) {
      throw StateError('CosyVoice2 prompt bundle missing snapshot_path.');
    }

    final prompt = CosyVoice2PromptBundle.load(promptDir);
    final promptSpeechTokens = prompt.promptSpeechToken
        .reshape([prompt.promptSpeechToken.size])
        .toList()
        .cast<int>();
    prompt.close();

    final runner = _runnerCache.putIfAbsent(
      snapshotPath,
      () => CosyVoice2UpperRunner.load(snapshotPath),
    );
    final speechTokens = runner.generateSpeechTokens(
      text: source,
      refText: refCaption,
      promptSpeechTokens: promptSpeechTokens,
    );
    if (speechTokens.isEmpty) {
      throw StateError('CosyVoice2 upper runtime produced no speech tokens.');
    }

    final flowBucket = _pickFlowBucket(meta, speechTokens.length);
    if (flowBucket == null) {
      throw StateError(
        'Speech token count ${speechTokens.length} exceeds exported flow buckets.',
      );
    }
    final flowKey = '$promptDir#$flowBucket';
    final flow = _flowCache.putIfAbsent(
      flowKey,
      () => CosyVoice2FlowBundle.load(promptDir, bucketTokens: flowBucket),
    );
    final mel = flow.synthesiseTokens(speechTokens, seed: 0);
    final melShape = <int>[1, 80, speechTokens.length * 2];

    final vocoderBucket = _pickVocoderBucket(meta, melShape[2]);
    if (vocoderBucket == null) {
      throw StateError(
        'Mel frame count ${melShape[2]} exceeds exported vocoder buckets.',
      );
    }
    final vocoderKey = '$promptDir#$vocoderBucket';
    final vocoder = _vocoderCache.putIfAbsent(
      vocoderKey,
      () => CosyVoice2VocoderBundle.load(promptDir, bucketFrames: vocoderBucket),
    );
    final audioResult = vocoder.synthesiseMel(mel, shape: melShape);
    try {
      return CosyVoice2Result(
        audio: audioResult.toFloat32List(),
        sampleRate: audioResult.sampleRate,
      );
    } finally {
      audioResult.close();
    }
  }

  Future<String> _resolvePromptBundleDir(String refAudioPath, String refText) async {
    final dir = promptBundleDir(
      modelId: modelId,
      refAudioPath: refAudioPath,
      refText: refText,
    );
    final metaPath = _joinPath(dir, 'meta.json');
    final promptPath = _joinPath(dir, 'prompt.safetensors');
    final flowPath = _joinPath(dir, 'tokens_to_mel_1024.mlxfn');
    final vocoderPath = _joinPath(dir, 'mel_to_audio_2048.mlxfn');
    if (await File(metaPath).exists() &&
        await File(promptPath).exists() &&
        await File(flowPath).exists() &&
        await File(vocoderPath).exists()) {
      return dir;
    }
    throw StateError(
      'CosyVoice2 prompt bundle is missing at $dir. '
      'Provide a native prompt bundle before synthesis.',
    );
  }

  int? _pickFlowBucket(Map<String, dynamic> meta, int tokens) {
    final buckets = (meta['flow_buckets'] as List<Object?>?)
            ?.map((value) => (value as num).toInt())
            .toList(growable: false) ??
        _defaultFlowBuckets;
    for (final bucket in buckets) {
      if (tokens <= bucket) {
        return bucket;
      }
    }
    return null;
  }

  int? _pickVocoderBucket(Map<String, dynamic> meta, int frames) {
    final buckets = (meta['vocoder_buckets'] as List<Object?>?)
            ?.map((value) => (value as num).toInt())
            .toList(growable: false) ??
        _defaultVocoderBuckets;
    for (final bucket in buckets) {
      if (frames <= bucket) {
        return bucket;
      }
    }
    return null;
  }
}

String _stableHash(String text) {
  var hash = 0xcbf29ce484222325;
  for (final code in text.codeUnits) {
    hash ^= code;
    hash = (hash * 0x100000001b3) & 0x7fffffffffffffff;
  }
  return hash.toRadixString(16);
}

final class CosyVoice2Result {
  const CosyVoice2Result({required this.audio, required this.sampleRate});

  final Float32List audio;
  final int sampleRate;

  int get numSamples => audio.length;

  double get durationSeconds => sampleRate <= 0 ? 0 : audio.length / sampleRate;
}

String _joinPath(String base, String leaf) {
  if (base.endsWith(Platform.pathSeparator)) {
    return '$base$leaf';
  }
  return '$base${Platform.pathSeparator}$leaf';
}
