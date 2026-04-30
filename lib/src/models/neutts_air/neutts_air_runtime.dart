import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart' show RuntimeTensor, RuntimeTensorDataType;
import '../cosyvoice2/cosyvoice2_audio.dart' show encodeWavPcm16Tensor;
import '../cosyvoice2/qwen2_tokenizer.dart';
import 'neutts_air.dart';
import 'neutts_air_native.dart';

final class NeuttsAirDecodeResult {
  const NeuttsAirDecodeResult({
    required this.text,
    required this.audioWavBytes,
    required this.sampleRate,
    required this.codecTokens,
    required this.codecTokenCount,
    required this.promptTokenIds,
    required this.promptTokenCount,
    required this.promptElapsedMicroseconds,
    required this.decodeElapsedMicroseconds,
    required this.codecDecoderProvider,
    required this.codecDecoderFrameCapacity,
    required this.lmElapsedMicroseconds,
    required this.lmProvider,
    required this.generatedFromLm,
    required this.lmInputTokenCount,
  });

  final String text;
  final Uint8List audioWavBytes;
  final int sampleRate;
  final List<int> codecTokens;
  final int codecTokenCount;
  final List<int> promptTokenIds;
  final int promptTokenCount;
  final int promptElapsedMicroseconds;
  final int decodeElapsedMicroseconds;
  final String codecDecoderProvider;
  final int? codecDecoderFrameCapacity;
  final int lmElapsedMicroseconds;
  final String? lmProvider;
  final bool generatedFromLm;
  final int lmInputTokenCount;
}

final class NeuttsAirLmLogits {
  const NeuttsAirLmLogits({required this.tensor, required this.owner});

  final RuntimeTensor tensor;
  final DartOnnxResult owner;

  void close() {
    owner.close();
  }
}

final class NeuttsAirDartRuntime {
  NeuttsAirDartRuntime._({
    required this.paths,
    required NeuttsAirSpecialTokenIds tokenIds,
    required Qwen2BpeTokenizer? tokenizer,
    required DartOnnxSession? lm,
    required DartOnnxSession? codecDecoder,
  }) : _tokenIds = tokenIds,
       _tokenizer = tokenizer,
       _lm = lm,
       _codecDecoder = codecDecoder;

  final NeuttsAirPaths paths;
  final NeuttsAirSpecialTokenIds _tokenIds;
  final Qwen2BpeTokenizer? _tokenizer;
  final DartOnnxSession? _lm;
  final DartOnnxSession? _codecDecoder;

  bool get tokenizerLoaded => _tokenizer != null;

  bool get lmLoaded => _lm != null;

  bool get codecDecoderLoaded => _codecDecoder != null;

  int? get codecDecoderFrameCapacity {
    final decoder = _codecDecoder;
    if (decoder == null) {
      return null;
    }
    return _staticOnnxInputDim(decoder.diagnostics, name: 'codes', axis: 2);
  }

  List<String> get loadedComponentNames => [
    if (_tokenizer != null) 'tokenizer',
    if (_lm != null) 'neutts_air_lm',
    if (_codecDecoder != null) 'neucodec_decoder',
  ];

  List<String> get selectedProviders => [
    if (_lm != null) 'neutts_air_lm:${_lm.selectedProvider}',
    if (_codecDecoder != null)
      'neucodec_decoder:${_codecDecoder.selectedProvider}',
  ];

  static Future<NeuttsAirDartRuntime> load({
    required NeuttsAirPaths paths,
    required String provider,
    required int deviceId,
    required bool requireProvider,
    required int numThreads,
    Map<String, Object?> backendOptions = const {},
    bool loadTokenizer = true,
    bool loadLm = true,
    bool loadCodecDecoder = true,
  }) async {
    final blockers = [
      if (loadTokenizer && !File(paths.tokenizerJson).existsSync())
        'tokenizer.json is missing at ${paths.tokenizerJson}',
      if (loadTokenizer && !File(paths.tokenizerConfigJson).existsSync())
        'tokenizer_config.json is missing at ${paths.tokenizerConfigJson}',
      if (loadLm && !File(paths.lmOnnx).existsSync())
        'neutts_air_lm ONNX is missing at ${paths.lmOnnx}',
      if (loadCodecDecoder && !File(paths.neucodecDecoderOnnx).existsSync())
        'neucodec_decoder ONNX is missing at ${paths.neucodecDecoderOnnx}',
    ];
    if (blockers.isNotEmpty) {
      throw StateError(
        'NeuTTS Air runtime is not loadable: ${blockers.join('; ')}',
      );
    }

    final tokenIds = loadTokenizer
        ? NeuttsAirSpecialTokenIds.fromTokenizerConfigFile(
            paths.tokenizerConfigJson,
          )
        : NeuttsAirSpecialTokenIds.defaults;
    Qwen2BpeTokenizer? tokenizer;
    DartOnnxSession? lm;
    DartOnnxSession? codecDecoder;
    try {
      if (loadTokenizer) {
        tokenizer = await loadNeuttsAirTokenizer(paths, tokenIds: tokenIds);
      }
      if (loadLm) {
        lm = DartOnnxSession.load(
          DartOnnxConfig(
            modelPath: paths.lmOnnx,
            id: 'neutts_air_lm',
            family: 'neutts_air',
            provider: provider,
            deviceId: deviceId,
            requireProvider: requireProvider,
            numThreads: numThreads,
            backendOptions: backendOptions,
          ),
        );
      }
      if (loadCodecDecoder) {
        codecDecoder = DartOnnxSession.load(
          DartOnnxConfig(
            modelPath: paths.neucodecDecoderOnnx,
            id: 'neutts_air_neucodec_decoder',
            family: 'neutts_air',
            provider: provider,
            deviceId: deviceId,
            requireProvider: requireProvider,
            numThreads: numThreads,
            backendOptions: backendOptions,
          ),
        );
      }
      return NeuttsAirDartRuntime._(
        paths: paths,
        tokenIds: tokenIds,
        tokenizer: tokenizer,
        lm: lm,
        codecDecoder: codecDecoder,
      );
    } catch (_) {
      tokenizer?.close();
      lm?.close();
      codecDecoder?.close();
      rethrow;
    }
  }

  List<int> buildPromptTokenIds({
    required String referencePhones,
    required String inputPhones,
    required List<int> referenceCodes,
  }) {
    final tokenizer = _tokenizer;
    if (tokenizer == null) {
      throw StateError('NeuTTS Air tokenizer is not loaded.');
    }
    return buildNeuttsAirPromptTokenIds(
      tokenizer: tokenizer,
      referencePhones: referencePhones,
      inputPhones: inputPhones,
      referenceCodes: referenceCodes,
      tokenIds: _tokenIds,
    );
  }

  NativeTensorBuffer buildPromptTokenIdsBuffer({
    required String referencePhones,
    required String inputPhones,
    required Object referenceCodes,
  }) {
    final tokenizer = _tokenizer;
    if (tokenizer == null) {
      throw StateError('NeuTTS Air tokenizer is not loaded.');
    }
    return buildNeuttsAirPromptTokenIdsBuffer(
      tokenizer: tokenizer,
      referencePhones: referencePhones,
      inputPhones: inputPhones,
      referenceCodes: referenceCodes,
      tokenIds: _tokenIds,
    );
  }

  Future<NeuttsAirDecodeResult> decodeCodecTokens({
    required String text,
    required Object codecTokens,
    List<int>? codecTokenValues,
    String referencePhones = '',
    String inputPhones = '',
    List<int> referenceCodes = const [],
    bool includeCodecMetadata = true,
    bool includePromptMetadata = true,
    bool validateCodecTokens = true,
  }) async {
    final codecTokenCount = _int32TokenCount(codecTokens);
    if (codecTokenCount == 0) {
      throw ArgumentError('NeuTTS Air codecTokens must not be empty.');
    }
    if (validateCodecTokens) {
      _validateCodecTokens(codecTokens);
    }
    final codecTokenList = includeCodecMetadata
        ? codecTokenValues ?? _int32TokensToUnmodifiableList(codecTokens)
        : const <int>[];
    final promptTimer = Stopwatch()..start();
    NativeTensorBuffer? promptTokenBuffer;
    var promptTokenCount = 0;
    var promptTokenIds = const <int>[];
    try {
      if (referencePhones.isNotEmpty || inputPhones.isNotEmpty) {
        promptTokenBuffer = buildPromptTokenIdsBuffer(
          referencePhones: referencePhones,
          inputPhones: inputPhones,
          referenceCodes: referenceCodes,
        );
        promptTokenCount = promptTokenBuffer.byteLength ~/ 4;
        if (includePromptMetadata) {
          promptTokenIds = List<int>.unmodifiable(
            promptTokenBuffer.asInt32List(),
          );
        }
      }
    } finally {
      promptTimer.stop();
      promptTokenBuffer?.close();
    }

    final decodeTimer = Stopwatch()..start();
    final audioWavBytes = _decodeCodecTokensToWav(codecTokens);
    decodeTimer.stop();
    return NeuttsAirDecodeResult(
      text: text,
      audioWavBytes: audioWavBytes,
      sampleRate: neuttsAirSampleRate,
      codecTokens: codecTokenList,
      codecTokenCount: codecTokenCount,
      promptTokenIds: promptTokenIds,
      promptTokenCount: promptTokenCount,
      promptElapsedMicroseconds: promptTimer.elapsedMicroseconds,
      decodeElapsedMicroseconds: decodeTimer.elapsedMicroseconds,
      codecDecoderProvider: _codecDecoder?.selectedProvider ?? 'unloaded',
      codecDecoderFrameCapacity: codecDecoderFrameCapacity,
      lmElapsedMicroseconds: 0,
      lmProvider: _lm?.selectedProvider,
      generatedFromLm: false,
      lmInputTokenCount: 0,
    );
  }

  Future<NeuttsAirDecodeResult> synthesizeText({
    required String text,
    String referencePhones = '',
    String inputPhones = '',
    List<int> referenceCodes = const [],
    int? maxGeneratedTokens,
    int seed = 0,
    double temperature = 1.0,
    double topP = 1.0,
    int topK = 50,
    bool includeCodecMetadata = true,
    bool includePromptMetadata = true,
  }) async {
    final lm = _lm;
    if (lm == null) {
      throw StateError(
        'NeuTTS Air LM is not loaded; export/load neutts_air_lm.onnx first.',
      );
    }
    if (_codecDecoder == null) {
      throw StateError(
        'NeuTTS Air NeuCodec decoder is not loaded; export '
        'neucodec_decoder.onnx first.',
      );
    }
    final frameCapacity = codecDecoderFrameCapacity;
    final maxTokens = maxGeneratedTokens ?? frameCapacity ?? 50;
    if (maxTokens <= 0) {
      throw RangeError.range(maxTokens, 1, null, 'maxGeneratedTokens');
    }
    if (frameCapacity != null && maxTokens != frameCapacity) {
      throw ArgumentError(
        'Current NeuCodec decoder ONNX expects exactly $frameCapacity generated '
        'codec token(s); got maxGeneratedTokens=$maxTokens. Re-export '
        'neucodec_decoder.onnx with --frames $maxTokens or pass '
        '--max-generated-tokens $frameCapacity.',
      );
    }
    final effectiveInputPhones = inputPhones.trim().isNotEmpty
        ? inputPhones
        : text;
    final promptTimer = Stopwatch()..start();
    final promptTokenBuffer = buildPromptTokenIdsBuffer(
      referencePhones: referencePhones,
      inputPhones: effectiveInputPhones,
      referenceCodes: referenceCodes,
    );
    final promptTokenCount = promptTokenBuffer.byteLength ~/ 4;
    final promptTokenIds = includePromptMetadata
        ? List<int>.unmodifiable(promptTokenBuffer.asInt32List())
        : const <int>[];
    promptTimer.stop();

    final lmTimer = Stopwatch()..start();
    final inputTokenBuffer = NativeTensorBuffer.int64([
      neuttsAirMaxContextTokens,
    ]);
    final generatedCodeBuffer = NativeTensorBuffer.int32([maxTokens]);
    final rng = math.Random(seed);
    var tokenCount = 0;
    var generatedCount = 0;
    var lastInputTokenCount = 0;
    try {
      tokenCount = initNeuttsAirDecodeInputIdsNative(
        promptTokenIds: promptTokenBuffer,
        inputIds: inputTokenBuffer,
      );
      lastInputTokenCount = tokenCount;
      while (generatedCount < maxTokens &&
          tokenCount < neuttsAirMaxContextTokens) {
        lastInputTokenCount = tokenCount;
        final logits = _runLmNoCache(lm, inputTokenBuffer, tokenCount);
        try {
          final nextTokenizerId = _sampleNextSpeechTokenizerId(
            logits.tensor,
            seqLen: tokenCount,
            rng: rng,
            tokenIds: _tokenIds,
            temperature: temperature,
            topP: topP,
            topK: topK,
            allowEndToken: frameCapacity == null && generatedCount > 0,
          );
          if (nextTokenizerId == _tokenIds.speechGenerationEndId) {
            break;
          }
          appendNeuttsAirDecodeSpeechTokenNative(
            inputIds: inputTokenBuffer,
            inputLength: tokenCount,
            generatedCodes: generatedCodeBuffer,
            generatedLength: generatedCount,
            tokenizerId: nextTokenizerId,
            speechBaseId: _tokenIds.speechTokenBaseId,
            speechVocabSize: _tokenIds.speechVocabSize,
          );
          generatedCount += 1;
          tokenCount += 1;
        } finally {
          logits.close();
        }
      }
      lmTimer.stop();
      if (generatedCount == 0) {
        throw StateError('NeuTTS Air LM produced no speech tokens.');
      }
      if (frameCapacity != null && generatedCount != frameCapacity) {
        throw StateError(
          'NeuTTS Air LM produced $generatedCount speech token(s), '
          'but the fixed decoder graph requires $frameCapacity.',
        );
      }
      final codecTokens = generatedCodeBuffer.tensorView(
        shape: [generatedCount],
        byteLength: generatedCount * 4,
      );
      final decodeTimer = Stopwatch()..start();
      final audioWavBytes = _decodeCodecTokensToWav(codecTokens);
      decodeTimer.stop();
      final generatedCodes = includeCodecMetadata
          ? List<int>.unmodifiable(
              Int32List.sublistView(
                generatedCodeBuffer.asInt32List(),
                0,
                generatedCount,
              ),
            )
          : const <int>[];
      return NeuttsAirDecodeResult(
        text: text,
        audioWavBytes: audioWavBytes,
        sampleRate: neuttsAirSampleRate,
        codecTokens: generatedCodes,
        codecTokenCount: generatedCount,
        promptTokenIds: promptTokenIds,
        promptTokenCount: promptTokenCount,
        promptElapsedMicroseconds: promptTimer.elapsedMicroseconds,
        decodeElapsedMicroseconds: decodeTimer.elapsedMicroseconds,
        codecDecoderProvider: _codecDecoder.selectedProvider,
        codecDecoderFrameCapacity: frameCapacity,
        lmElapsedMicroseconds: lmTimer.elapsedMicroseconds,
        lmProvider: lm.selectedProvider,
        generatedFromLm: true,
        lmInputTokenCount: lastInputTokenCount,
      );
    } finally {
      if (lmTimer.isRunning) {
        lmTimer.stop();
      }
      promptTokenBuffer.close();
      generatedCodeBuffer.close();
      inputTokenBuffer.close();
    }
  }

  NeuttsAirLmLogits _runLmNoCache(
    DartOnnxSession lm,
    NativeTensorBuffer inputIds,
    int tokenCount,
  ) {
    final result = lm.run({
      'input_ids': inputIds.tensorView(
        shape: [1, tokenCount],
        byteLength: tokenCount * 8,
      ),
    });
    var keepResult = false;
    try {
      final logits = _singleFloat32Output(result, label: 'NeuTTS Air LM');
      keepResult = true;
      return NeuttsAirLmLogits(tensor: logits, owner: result);
    } finally {
      if (!keepResult) {
        result.close();
      }
    }
  }

  Uint8List _decodeCodecTokensToWav(Object codecTokens) {
    final decoder = _codecDecoder;
    if (decoder == null) {
      throw StateError(
        'NeuTTS Air NeuCodec decoder is not loaded; export '
        'neucodec_decoder.onnx first.',
      );
    }
    final tokenCount = _int32TokenCount(codecTokens);
    final frameCapacity = codecDecoderFrameCapacity;
    if (frameCapacity != null && tokenCount != frameCapacity) {
      throw ArgumentError(
        'NeuCodec decoder ONNX expects exactly $frameCapacity codec token(s); '
        'got $tokenCount. Re-export neucodec_decoder.onnx with '
        '--frames $tokenCount, or provide a graph whose fixed frame budget '
        'matches the token count.',
      );
    }
    final owned = _nativeInt32TokenBuffer(codecTokens);
    try {
      final codes = _int32TokenTensor(
        owned ?? codecTokens,
        shape: [1, 1, tokenCount],
      );
      final result = decoder.run({'codes': codes});
      try {
        final output = _singleFloat32Output(result);
        return encodeWavPcm16Tensor(output, sampleRate: neuttsAirSampleRate);
      } finally {
        result.close();
      }
    } finally {
      owned?.close();
    }
  }

  void close() {
    _tokenizer?.close();
    _lm?.close();
    _codecDecoder?.close();
  }
}

int _sampleNextSpeechTokenizerId(
  RuntimeTensor logits, {
  required int seqLen,
  required math.Random rng,
  required NeuttsAirSpecialTokenIds tokenIds,
  required double temperature,
  required double topP,
  required int topK,
  required bool allowEndToken,
}) {
  if (logits.dtype != RuntimeTensorDataType.float32) {
    throw StateError('NeuTTS Air LM logits must be float32.');
  }
  if (logits.shape.length != 3 || logits.shape[0] != 1) {
    throw StateError('NeuTTS Air LM logits shape must be [1, S, V].');
  }
  final vocabSize = logits.shape[2];
  if (seqLen <= 0 || seqLen > logits.shape[1]) {
    throw RangeError.range(seqLen, 1, logits.shape[1], 'seqLen');
  }
  final temp = temperature.isFinite && temperature > 0 ? temperature : 1.0;
  return sampleNeuttsAirSpeechTokenizerIdNative(
    logits: logits,
    logitsOffset: (seqLen - 1) * vocabSize,
    vocabSize: vocabSize,
    speechBaseId: tokenIds.speechTokenBaseId,
    speechVocabSize: tokenIds.speechVocabSize,
    eosId: tokenIds.speechGenerationEndId,
    allowEos: allowEndToken,
    temperature: temp,
    topP: topP,
    topK: topK,
    randomDraw: rng.nextDouble(),
  );
}

int? _staticOnnxInputDim(
  Map<String, Object?> diagnostics, {
  required String name,
  required int axis,
}) {
  final metadata = diagnostics['input_metadata'];
  if (metadata is! List) {
    return null;
  }
  for (final raw in metadata) {
    if (raw is! Map) {
      continue;
    }
    final inputName = '${raw['name'] ?? ''}';
    if (inputName != name) {
      continue;
    }
    final shape = raw['shape'];
    if (shape is! List || axis >= shape.length) {
      return null;
    }
    final dim = int.tryParse('${shape[axis]}') ?? -1;
    return dim > 0 ? dim : null;
  }
  return null;
}

List<int> codecTokensFromNeuttsAirRequest({
  required List<int> codecTokens,
  required String codecTokenText,
}) {
  if (codecTokens.isNotEmpty) {
    _validateCodecTokens(codecTokens);
    return List<int>.unmodifiable(codecTokens);
  }
  if (codecTokenText.trim().isEmpty) {
    return const [];
  }
  final parsed = parseNeuttsAirSpeechTokens(codecTokenText);
  if (parsed.isEmpty) {
    throw ArgumentError(
      'codecTokenText did not contain NeuTTS Air speech tokens.',
    );
  }
  return parsed.toList(growable: false);
}

final class NeuttsAirCodecTokenSource {
  NeuttsAirCodecTokenSource._({
    required this.tokens,
    required this.tokenCount,
    required this.source,
    NativeTensorBuffer? owner,
  }) : _owner = owner;

  final List<int> tokens;
  final int tokenCount;
  final Object source;
  final NativeTensorBuffer? _owner;

  bool get isEmpty => tokenCount == 0;

  void close() {
    _owner?.close();
  }
}

NeuttsAirCodecTokenSource codecTokenSourceFromNeuttsAirRequest({
  required List<int> codecTokens,
  required String codecTokenText,
  bool includeTokenMetadata = true,
}) {
  if (codecTokens.isNotEmpty) {
    final source = NativeTensorBuffer.int32([codecTokens.length]);
    try {
      source.asInt32List().setAll(0, codecTokens);
      _validateCodecTokens(source);
      return NeuttsAirCodecTokenSource._(
        tokens: includeTokenMetadata
            ? List<int>.unmodifiable(codecTokens)
            : const <int>[],
        tokenCount: codecTokens.length,
        source: source,
        owner: source,
      );
    } catch (_) {
      source.close();
      rethrow;
    }
  }
  if (codecTokenText.trim().isEmpty) {
    return NeuttsAirCodecTokenSource._(
      tokens: const [],
      tokenCount: 0,
      source: const <int>[],
    );
  }
  final parsed = parseNeuttsAirSpeechTokensBuffer(codecTokenText);
  try {
    final tokenCount = parsed.byteLength ~/ 4;
    if (parsed.byteLength == 0) {
      throw ArgumentError(
        'codecTokenText did not contain NeuTTS Air speech tokens.',
      );
    }
    return NeuttsAirCodecTokenSource._(
      tokens: includeTokenMetadata
          ? List<int>.unmodifiable(parsed.asInt32List())
          : const <int>[],
      tokenCount: tokenCount,
      source: parsed,
      owner: parsed,
    );
  } catch (_) {
    parsed.close();
    rethrow;
  }
}

void _validateCodecTokens(Object codecTokens) {
  if (codecTokens is NativeTensorBuffer || codecTokens is RuntimeTensor) {
    validateNeuttsAirSpeechTokensNative(codecTokens);
    return;
  }
  for (final token in _int32TokenValues(codecTokens)) {
    if (token < 0 || token >= neuttsAirSpeechVocabSize) {
      throw RangeError.range(
        token,
        0,
        neuttsAirSpeechVocabSize - 1,
        'codecTokens',
      );
    }
  }
}

int _int32TokenCount(Object tokens) {
  if (tokens is NativeTensorBuffer) {
    _checkInt32Tokens(tokens.dtype);
    return tokens.byteLength ~/ 4;
  }
  if (tokens is RuntimeTensor) {
    _checkInt32Tokens(tokens.dtype);
    return tokens.bytes.lengthInBytes ~/ 4;
  }
  if (tokens is Int32List) {
    return tokens.length;
  }
  if (tokens is List<int>) {
    return tokens.length;
  }
  throw ArgumentError.value(
    tokens,
    'tokens',
    'expected NativeTensorBuffer/RuntimeTensor/Int32List/List<int>',
  );
}

Iterable<int> _int32TokenValues(Object tokens) {
  if (tokens is NativeTensorBuffer) {
    _checkInt32Tokens(tokens.dtype);
    return tokens.asInt32List();
  }
  if (tokens is RuntimeTensor) {
    _checkInt32Tokens(tokens.dtype);
    return tokens.asInt32List();
  }
  if (tokens is Int32List) {
    return tokens;
  }
  if (tokens is List<int>) {
    return tokens;
  }
  throw ArgumentError.value(
    tokens,
    'tokens',
    'expected NativeTensorBuffer/RuntimeTensor/Int32List/List<int>',
  );
}

List<int> _int32TokensToUnmodifiableList(Object tokens) =>
    List<int>.unmodifiable(_int32TokenValues(tokens));

NativeTensorBuffer? _nativeInt32TokenBuffer(Object tokens) {
  if (tokens is NativeTensorBuffer || tokens is RuntimeTensor) {
    return null;
  }
  final values = _int32TokenValues(tokens);
  final out = NativeTensorBuffer.int32([_int32TokenCount(tokens)]);
  out.asInt32List().setAll(0, values);
  return out;
}

RuntimeTensor _int32TokenTensor(Object tokens, {required List<int> shape}) {
  final byteLength = _int32TokenCount(tokens) * 4;
  if (tokens is NativeTensorBuffer) {
    _checkInt32Tokens(tokens.dtype);
    return tokens.tensorView(shape: shape, byteLength: byteLength);
  }
  if (tokens is RuntimeTensor) {
    _checkInt32Tokens(tokens.dtype);
    return RuntimeTensor(
      dtype: RuntimeTensorDataType.int32,
      shape: shape,
      bytes: tokens.bytes,
      nativeData: tokens.nativeData,
      owner: tokens,
    );
  }
  throw ArgumentError.value(
    tokens,
    'tokens',
    'expected NativeTensorBuffer/RuntimeTensor',
  );
}

void _checkInt32Tokens(RuntimeTensorDataType dtype) {
  if (dtype != RuntimeTensorDataType.int32) {
    throw StateError('Expected int32 codec tokens, got ${dtype.name}.');
  }
}

RuntimeTensor _singleFloat32Output(
  DartOnnxResult result, {
  String label = 'NeuCodec decoder',
}) {
  for (final entry in result.outputs.entries) {
    final value = entry.value;
    if (value is RuntimeTensor) {
      if (value.dtype != RuntimeTensorDataType.float32) {
        throw StateError(
          '$label output "${entry.key}" has dtype ${value.dtype.name}; '
          'expected float32.',
        );
      }
      return value;
    }
  }
  throw StateError('$label produced no native float32 output.');
}
