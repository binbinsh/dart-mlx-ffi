import 'dart:math';
import 'dart:io';
import 'dart:typed_data';

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/onnx.dart' show canonicalOnnxExecutionProvider;
import '../../runtime/runtime.dart' show RuntimeTensor, RuntimeTensorDataType;
import '../cosyvoice2/cosyvoice2.dart';
import '../cosyvoice2/cosyvoice2_audio.dart';
import '../cosyvoice2/cosyvoice2_flow_driver.dart';
import '../cosyvoice2/cosyvoice2_speaker_prompt.dart';
import 'sarashina2.dart';
import 'sarashina2_flow_options.dart';
import 'sarashina2_llm_driver.dart';
import 'sarashina2_llm_options.dart';
import 'sarashina2_native.dart';

final class Sarashina2DecodeResult {
  const Sarashina2DecodeResult({
    required this.text,
    required this.audio,
    required this.audioSampleCount,
    required this.audioWavBytes,
    required this.sampleRate,
    required this.semanticTokens,
    required this.decodedSemanticTokens,
    required this.semanticTokenCount,
    required this.decodedSemanticTokenCount,
    required this.melFrames,
    required this.promptElapsedMicroseconds,
    required this.semanticElapsedMicroseconds,
    required this.decodeElapsedMicroseconds,
    required this.usedPrompt,
    this.flowPrepareElapsedMicroseconds = 0,
    this.flowEncoderElapsedMicroseconds = 0,
    this.flowSetupElapsedMicroseconds = 0,
    this.flowDiffuseElapsedMicroseconds = 0,
    this.flowVocodeElapsedMicroseconds = 0,
  });

  final String text;
  final Float32List audio;
  final int audioSampleCount;
  final Uint8List audioWavBytes;
  final int sampleRate;
  final List<int> semanticTokens;
  final List<int> decodedSemanticTokens;
  final int semanticTokenCount;
  final int decodedSemanticTokenCount;
  final int melFrames;
  final int promptElapsedMicroseconds;
  final int semanticElapsedMicroseconds;
  final int decodeElapsedMicroseconds;
  final bool usedPrompt;
  final int flowPrepareElapsedMicroseconds;
  final int flowEncoderElapsedMicroseconds;
  final int flowSetupElapsedMicroseconds;
  final int flowDiffuseElapsedMicroseconds;
  final int flowVocodeElapsedMicroseconds;
}

enum Sarashina2SemanticSource { text, promptTokenIds, external }

final class Sarashina2SynthesisRequest {
  const Sarashina2SynthesisRequest({
    required this.text,
    this.promptAudio,
    this.prompt,
    this.promptText = '',
    this.semanticTokens = const [],
    this.semanticTokenText = '',
    this.promptTokenIds = const [],
    this.maxGeneratedTokens = 2048,
    this.latencyTokens = 1,
    this.seed = 0,
    this.temperature = sarashina2DefaultTemperature,
    this.topP = sarashina2DefaultTopP,
    this.frequencyPenalty = sarashina2DefaultFrequencyPenalty,
    this.includeFloatOutputs = true,
    this.includeTokenMetadata = true,
  });

  final String text;
  final PcmAudio? promptAudio;
  final Sarashina2Prompt? prompt;
  final String promptText;
  final List<int> semanticTokens;
  final String semanticTokenText;
  final List<int> promptTokenIds;
  final int maxGeneratedTokens;
  final int latencyTokens;
  final int seed;
  final double temperature;
  final double topP;
  final double frequencyPenalty;
  final bool includeFloatOutputs;
  final bool includeTokenMetadata;

  bool get hasExternalSemanticTokens =>
      semanticTokens.isNotEmpty || semanticTokenText.trim().isNotEmpty;

  bool get needsLlm => !hasExternalSemanticTokens;
}

final class Sarashina2SynthesisResult {
  const Sarashina2SynthesisResult({
    required this.decode,
    required this.semanticSource,
    required this.promptTokenGenerationElapsedMicroseconds,
  });

  final Sarashina2DecodeResult decode;
  final Sarashina2SemanticSource semanticSource;
  final int promptTokenGenerationElapsedMicroseconds;

  String get semanticSourceName => switch (semanticSource) {
    Sarashina2SemanticSource.text => 'text',
    Sarashina2SemanticSource.promptTokenIds => 'prompt_token_ids',
    Sarashina2SemanticSource.external => 'external',
  };

  Uint8List get audioWavBytes => decode.audioWavBytes;

  int get ttsElapsedMicroseconds =>
      decode.semanticElapsedMicroseconds +
      promptTokenGenerationElapsedMicroseconds +
      decode.decodeElapsedMicroseconds;

  Map<String, Object?> toJson({String? outputWav}) {
    final json = <String, Object?>{
      'audioBytes': audioWavBytes.length,
      'sampleRate': decode.sampleRate,
      'audioSamples': decode.audioSampleCount,
      'semanticSource': semanticSourceName,
      if (decode.semanticTokens.isNotEmpty)
        'semanticTokens': decode.semanticTokens,
      if (decode.decodedSemanticTokens.isNotEmpty)
        'decodedSemanticTokens': decode.decodedSemanticTokens,
      if (decode.semanticTokens.isNotEmpty)
        'semanticTokenText': sarashina2SemanticTokensToText(
          decode.semanticTokens,
        ),
      if (decode.decodedSemanticTokens.isNotEmpty)
        'decodedSemanticTokenText': sarashina2SemanticTokensToText(
          decode.decodedSemanticTokens,
        ),
      'semanticTokenCount': decode.semanticTokenCount,
      'decodedSemanticTokenCount': decode.decodedSemanticTokenCount,
      'melFrames': decode.melFrames,
      'usedPrompt': decode.usedPrompt,
      'promptElapsedMs': decode.promptElapsedMicroseconds / 1000.0,
      'semanticElapsedMs':
          (decode.semanticElapsedMicroseconds +
              promptTokenGenerationElapsedMicroseconds) /
          1000.0,
      'decodeElapsedMs': decode.decodeElapsedMicroseconds / 1000.0,
      'flowPrepareElapsedMs': decode.flowPrepareElapsedMicroseconds / 1000.0,
      'flowEncoderElapsedMs': decode.flowEncoderElapsedMicroseconds / 1000.0,
      'flowSetupElapsedMs': decode.flowSetupElapsedMicroseconds / 1000.0,
      'flowDiffuseElapsedMs': decode.flowDiffuseElapsedMicroseconds / 1000.0,
      'flowVocodeElapsedMs': decode.flowVocodeElapsedMicroseconds / 1000.0,
      'ttsElapsedMs': ttsElapsedMicroseconds / 1000.0,
      'watermark': 'not_embedded',
    };
    if (outputWav != null) {
      json['outputWav'] = outputWav;
    }
    return json;
  }
}

int _int32TokenCount(Object tokens) {
  if (tokens is NativeTensorBuffer) {
    if (tokens.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tokens, got ${tokens.dtype.name}.');
    }
    return tokens.byteLength ~/ 4;
  }
  if (tokens is RuntimeTensor) {
    if (tokens.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tokens, got ${tokens.dtype.name}.');
    }
    return tokens.byteLength ~/ 4;
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
    'expected NativeTensorBuffer/Int32List/List<int>',
  );
}

Iterable<int> _int32TokenValues(Object tokens) {
  if (tokens is NativeTensorBuffer) {
    if (tokens.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tokens, got ${tokens.dtype.name}.');
    }
    return tokens.asInt32List();
  }
  if (tokens is RuntimeTensor) {
    if (tokens.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tokens, got ${tokens.dtype.name}.');
    }
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
    'expected NativeTensorBuffer/Int32List/List<int>',
  );
}

void _validateSemanticTokenSource(Object tokens) {
  validateSarashinaSemanticTokensNative(tokens);
}

final class _SemanticTokenSource {
  _SemanticTokenSource._({
    required this.source,
    required this.tokens,
    NativeTensorBuffer? owner,
  }) : _owner = owner;

  final Object source;
  final List<int> tokens;
  final NativeTensorBuffer? _owner;

  void close() {
    _owner?.close();
  }
}

final class Sarashina2Prompt {
  const Sarashina2Prompt._({
    required SpeakerPrompt prompt,
    required this.extractElapsedMicroseconds,
  }) : _prompt = prompt;

  final SpeakerPrompt _prompt;
  final int extractElapsedMicroseconds;

  Int32List get promptSemanticTokens => _prompt.promptSpeechTokens;

  int get promptSemanticTokenCount => _prompt.promptSpeechTokenCount;

  int get promptSpeechFeatFrames => _prompt.promptSpeechFeatFrames;

  void close() {
    _prompt.close();
  }
}

final class _PreparedPrompt {
  const _PreparedPrompt({
    required this.prompt,
    required this.elapsedMicroseconds,
    this.ownsPrompt = false,
  });

  final SpeakerPrompt? prompt;
  final int elapsedMicroseconds;
  final bool ownsPrompt;

  void close() {
    if (ownsPrompt) {
      prompt?.close();
    }
  }
}

final class Sarashina2DartRuntime {
  Sarashina2DartRuntime._({
    required this.paths,
    required CosyVoice2PartialOnnxBundle bundle,
    required Sarashina2LlmDriver? llm,
    required Sarashina2BaseTokenizer? tokenizer,
    required CosyVoice2FlowDriver flow,
    required this.llmPrecision,
  }) : _bundle = bundle,
       _llm = llm,
       _tokenizer = tokenizer,
       _flow = flow,
       _promptExtractor = SpeakerPromptExtractor(bundle: bundle);

  final Sarashina2TtsPaths paths;
  final String llmPrecision;
  final CosyVoice2PartialOnnxBundle _bundle;
  final Sarashina2LlmDriver? _llm;
  final Sarashina2BaseTokenizer? _tokenizer;
  final CosyVoice2FlowDriver _flow;
  final SpeakerPromptExtractor _promptExtractor;

  List<String> get loadedComponentNames => _bundle.loadedComponentNames;

  bool get llmLoaded => _llm != null;

  bool get tokenizerLoaded => _tokenizer != null;

  List<String> get selectedProviders => [
    for (final component in _bundle.loadedComponents)
      '${component.name}:${component.selectedProvider}',
  ];

  Sarashina2Prompt extractPrompt(PcmAudio promptAudio) {
    final timer = Stopwatch()..start();
    final prompt = _promptExtractor.extract(
      promptAudio.samples,
      promptAudio.sampleRate,
    );
    timer.stop();
    return Sarashina2Prompt._(
      prompt: prompt,
      extractElapsedMicroseconds: timer.elapsedMicroseconds,
    );
  }

  static Future<Sarashina2DartRuntime> load({
    required Sarashina2TtsPaths paths,
    required String provider,
    required int deviceId,
    required bool requireProvider,
    required int numThreads,
    Map<String, Object?> backendOptions = const {},
    bool loadLlm = false,
  }) async {
    final effectiveBackendOptions = _sarashina2BackendOptions(
      provider,
      backendOptions,
    );
    final componentProviders = _sarashina2ComponentProviders(
      provider,
      effectiveBackendOptions,
    );
    final llmProvider = componentProviders['llm_prefill'] ?? provider;
    final cosyPaths = CosyVoice2Paths(modelDir: paths.modelDir);
    final llmPrecision = loadLlm
        ? resolveSarashina2LlmPrecision(paths, effectiveBackendOptions)
        : 'none';
    final useFusedDecodeHead =
        loadLlm &&
        _useFusedDecodeHead(paths, effectiveBackendOptions, llmPrecision);
    final useFusedFlowLoop = sarashina2UseFusedFlowLoop(
      paths,
      effectiveBackendOptions,
    );
    final isTensorRtProvider =
        canonicalOnnxExecutionProvider(provider) == 'TensorrtExecutionProvider';
    final useFusedFlowStep =
        !useFusedFlowLoop &&
        sarashina2UseFusedFlowStep(paths, effectiveBackendOptions);
    final useDeviceFlowLoop =
        useFusedFlowStep &&
        resolveSarashina2UseDeviceFlowLoop(
          effectiveBackendOptions,
          provider: provider,
        );
    final flowStepPrecision = useFusedFlowStep
        ? resolveSarashina2FlowStepPrecision(
            paths,
            effectiveBackendOptions,
            provider: provider,
          )
        : 'fp32';
    final flowStepContextPath =
        isTensorRtProvider && useFusedFlowStep && flowStepPrecision == 'fp32'
        ? resolveSarashina2TensorRtFlowStepContextPath(
            paths,
            effectiveBackendOptions,
          )
        : null;
    final useFlowDecoderOutputViews =
        !isTensorRtProvider &&
        _backendBool(
          effectiveBackendOptions,
          'sarashina2FlowDecoderUseOutputViews',
          true,
        );
    final components = {
      'campplus',
      'speech_tokenizer_v2',
      'flow_encoder_fp32',
      if (useFusedFlowLoop)
        'flow_decoder_loop_fp32'
      else if (useFusedFlowStep) ...[
        'flow_decoder_step_fp32',
        if (useDeviceFlowLoop) 'flow_decoder_step_final_fp32',
      ] else
        'flow_decoder_estimator_fp32',
      'hift',
      if (loadLlm) 'llm_prefill',
      if (loadLlm && useFusedDecodeHead) 'llm_decode_head',
      if (loadLlm && !useFusedDecodeHead) 'llm_decode',
      if (loadLlm) 'llm_decoder_head',
    };
    final componentOptions = {
      ...sarashina2BaseComponentBackendOptions(effectiveBackendOptions),
      if (useFusedFlowLoop && useFlowDecoderOutputViews)
        'flow_decoder_loop_fp32': const {'useOutputViews': true},
      if (useFusedFlowStep && !useDeviceFlowLoop && useFlowDecoderOutputViews)
        'flow_decoder_step_fp32': const {'useOutputViews': true},
      if (useDeviceFlowLoop) ...{
        'flow_decoder_step_fp32': {
          'useIoBinding': true,
          'useDeviceOutputs': true,
          'deviceOutputNames': 'next_x',
          if (!_backendBool(
            effectiveBackendOptions,
            'sarashina2FlowStepSyncOutputs',
            false,
          ))
            'syncBoundOutputs': false,
          if (_backendBool(
            effectiveBackendOptions,
            'sarashina2FlowStepCacheBoundOutputs',
            true,
          ))
            'cacheBoundOutputs': true,
          'prepackedWeightsKey':
              'sarashina2-flow-step:${paths.modelDir}:$provider:$deviceId:$flowStepPrecision',
        },
        'flow_decoder_step_final_fp32': {
          'useIoBinding': true,
          'useOutputViews': true,
          'prepackedWeightsKey':
              'sarashina2-flow-step:${paths.modelDir}:$provider:$deviceId:$flowStepPrecision',
        },
      },
      if (loadLlm)
        ...sarashina2LlmComponentBackendOptions(
          paths: paths,
          provider: llmProvider,
          deviceId: deviceId,
          precision: llmPrecision,
          backendOptions: effectiveBackendOptions,
        ),
    };
    final bundle = CosyVoice2PartialOnnxBundle.load(
      paths: cosyPaths,
      provider: provider,
      deviceId: deviceId,
      requireProvider: requireProvider,
      numThreads: numThreads,
      backendOptions: effectiveBackendOptions,
      componentBackendOptions: componentOptions,
      componentProviders: componentProviders,
      componentPathOverrides: {
        if (flowStepContextPath != null) ...{
          'flow_decoder_step_fp32': flowStepContextPath,
          if (useDeviceFlowLoop)
            'flow_decoder_step_final_fp32': flowStepContextPath,
        },
        if (useFusedFlowStep && flowStepPrecision == 'fp16') ...{
          'flow_decoder_step_fp32': paths.flowDecoderStepFp16Onnx,
          if (useDeviceFlowLoop)
            'flow_decoder_step_final_fp32': paths.flowDecoderStepFp16Onnx,
        },
        if (loadLlm)
          ...sarashina2LlmPathOverrides(
            paths,
            llmPrecision,
            provider: llmProvider,
            deviceId: deviceId,
            backendOptions: effectiveBackendOptions,
          ),
      },
      componentNames: components,
    );
    try {
      final blockers = [
        for (final status in bundle.statuses)
          if (components.contains(status.file.name) &&
              (!status.exists || status.error != null))
            status.error == null
                ? '${status.file.name} is missing'
                : '${status.file.name}: ${status.error}',
        if (!cosyPaths.supportAssets().any(
          (asset) => asset.name == 'flow_support' && asset.exists,
        ))
          'flow_support.npz is missing',
        if (loadLlm &&
            !cosyPaths.supportAssets().any(
              (asset) => asset.name == 'llm_embeddings' && asset.exists,
            ))
          'llm_embeddings.npz is missing',
        if (loadLlm && !File(paths.addedTokensJson).existsSync())
          'added_tokens.json is missing',
        if (loadLlm && !File(paths.tokenizerSidecar).existsSync())
          'tokenizer.sara2tok is missing',
        ..._providerBlockers(
          statuses: bundle.statuses,
          components: components,
          provider: provider,
          componentProviders: componentProviders,
          requireProvider: requireProvider,
        ),
      ];
      if (blockers.isNotEmpty) {
        throw StateError(
          'Sarashina2 decoder runtime is not loadable: ${blockers.join('; ')}',
        );
      }
      Sarashina2LlmDriver? llm;
      Sarashina2BaseTokenizer? tokenizer;
      try {
        if (loadLlm) {
          llm = await Sarashina2LlmDriver.load(bundle: bundle, paths: paths);
          tokenizer = Sarashina2BaseTokenizer.fromFile(
            paths.tokenizerSidecar,
            tokenMap: llm.tokenMap,
          );
        }
        final flow = await CosyVoice2FlowDriver.load(
          bundle: bundle,
          paths: cosyPaths,
          diffusionSteps: resolveSarashina2FlowSteps(effectiveBackendOptions),
        );
        return Sarashina2DartRuntime._(
          paths: paths,
          bundle: bundle,
          llm: llm,
          tokenizer: tokenizer,
          flow: flow,
          llmPrecision: llmPrecision,
        );
      } catch (_) {
        tokenizer?.close();
        llm?.close();
        rethrow;
      }
    } catch (_) {
      bundle.close();
      rethrow;
    }
  }

  List<int> generateSemanticTokensFromPromptTokenIds({
    required Object promptTokenIds,
    int maxGeneratedTokens = 2048,
    int seed = 0,
    double temperature = sarashina2DefaultTemperature,
    double topP = sarashina2DefaultTopP,
    double frequencyPenalty = sarashina2DefaultFrequencyPenalty,
  }) {
    final generated = _generateSemanticTokenSourceFromPromptTokenIds(
      promptTokenIds: promptTokenIds,
      maxGeneratedTokens: maxGeneratedTokens,
      seed: seed,
      temperature: temperature,
      topP: topP,
      frequencyPenalty: frequencyPenalty,
      includeTokenMetadata: true,
    );
    try {
      return generated.tokens;
    } finally {
      generated.close();
    }
  }

  _SemanticTokenSource _generateSemanticTokenSourceFromPromptTokenIds({
    required Object promptTokenIds,
    required int maxGeneratedTokens,
    required int seed,
    required double temperature,
    required double topP,
    required double frequencyPenalty,
    required bool includeTokenMetadata,
  }) {
    final llm = _llm;
    if (llm == null) {
      throw StateError(
        'Sarashina2 LLM components are not loaded; pass loadLlm: true.',
      );
    }
    if (_int32TokenCount(promptTokenIds) == 0) {
      throw ArgumentError('promptTokenIds must not be empty.');
    }
    if (maxGeneratedTokens < 1) {
      throw ArgumentError('maxGeneratedTokens must be positive.');
    }
    if (!temperature.isFinite || temperature < 0.0) {
      throw ArgumentError('temperature must be non-negative.');
    }
    if (!topP.isFinite || topP <= 0.0) {
      throw ArgumentError('topP must be positive.');
    }
    if (!frequencyPenalty.isFinite) {
      throw ArgumentError('frequencyPenalty must be finite.');
    }
    final state = llm.prefillTokenIds(promptTokenIds);
    Sarashina2SemanticSamplerState? samplerState;
    NativeTensorBuffer? generatedBuffer;
    NativeTensorBuffer? nextEmbed;
    var keepGeneratedBuffer = false;
    final rng = Random(seed);
    try {
      generatedBuffer = NativeTensorBuffer.int32([maxGeneratedTokens]);
      nextEmbed = llm.createTokenEmbeddingBuffer();
      final nextEmbedTensor = nextEmbed.tensor;
      var generatedCount = 0;
      samplerState = Sarashina2SemanticSamplerState(tokenMap: llm.tokenMap);
      for (var step = 0; step < maxGeneratedTokens; step += 1) {
        final tokenId = llm.sampleNextSemanticTokenizerId(
          state: state,
          generatedSemanticTokens: const <int>[],
          samplerState: samplerState,
          eosId: sarashina2EosTokenId,
          temperature: temperature,
          topP: topP,
          frequencyPenalty: frequencyPenalty,
          randomDraw: rng.nextDouble(),
        );
        if (tokenId == sarashina2EosTokenId) {
          break;
        }
        final semanticId = llm.tokenMap.semanticIdForTokenizerId(tokenId)!;
        samplerState.appendSemanticId(
          generated: generatedBuffer,
          generatedLength: generatedCount,
          semanticId: semanticId,
        );
        generatedCount += 1;
        if (step + 1 == maxGeneratedTokens) {
          break;
        }
        llm.fillTokenEmbeddingBuffer(tokenId: tokenId, out: nextEmbed);
        llm.decodeStep(state: state, nextEmbed: nextEmbedTensor);
      }
      final tokens = includeTokenMetadata
          ? List<int>.unmodifiable(
              generatedBuffer.asInt32List().sublist(0, generatedCount),
            )
          : const <int>[];
      keepGeneratedBuffer = true;
      return _SemanticTokenSource._(
        source: generatedBuffer.tensorView(
          shape: [generatedCount],
          byteLength: generatedCount * 4,
        ),
        tokens: tokens,
        owner: generatedBuffer,
      );
    } finally {
      if (!keepGeneratedBuffer) {
        generatedBuffer?.close();
      }
      nextEmbed?.close();
      samplerState?.close();
      state.close();
    }
  }

  List<int> generateSemanticTokensFromText({
    required String text,
    String promptText = '',
    List<int> promptSemanticTokens = const [],
    int maxGeneratedTokens = 2048,
    int seed = 0,
    double temperature = sarashina2DefaultTemperature,
    double topP = sarashina2DefaultTopP,
    double frequencyPenalty = sarashina2DefaultFrequencyPenalty,
  }) {
    final generated = _generateSemanticTokenSourceFromText(
      text: text,
      promptText: promptText,
      promptSemanticTokens: promptSemanticTokens,
      maxGeneratedTokens: maxGeneratedTokens,
      seed: seed,
      temperature: temperature,
      topP: topP,
      frequencyPenalty: frequencyPenalty,
      includeTokenMetadata: true,
    );
    try {
      return generated.tokens;
    } finally {
      generated.close();
    }
  }

  _SemanticTokenSource _generateSemanticTokenSourceFromText({
    required String text,
    String promptText = '',
    Object promptSemanticTokens = const <int>[],
    required int maxGeneratedTokens,
    required int seed,
    required double temperature,
    required double topP,
    required double frequencyPenalty,
    required bool includeTokenMetadata,
  }) {
    final tokenizer = _tokenizer;
    if (tokenizer == null) {
      throw StateError(
        'Sarashina2 tokenizer is not loaded; generate tokenizer.sara2tok and pass loadLlm: true.',
      );
    }
    if (text.trim().isEmpty) {
      throw ArgumentError('Sarashina2 text must not be empty');
    }
    final effectivePromptText = promptText.trim();
    final promptSemanticTokenCount = _int32TokenCount(promptSemanticTokens);
    if (effectivePromptText.isNotEmpty && promptSemanticTokenCount == 0) {
      throw ArgumentError(
        'promptSemanticTokens are required when promptText is provided.',
      );
    }
    if (effectivePromptText.isEmpty && promptSemanticTokenCount != 0) {
      throw ArgumentError(
        'promptText is required when promptSemanticTokens are provided.',
      );
    }
    final promptTokenIds = tokenizer.encodePromptTokenIdsBuffer(
      text: text,
      promptText: effectivePromptText,
      promptTokens: promptSemanticTokens,
    );
    try {
      return _generateSemanticTokenSourceFromPromptTokenIds(
        promptTokenIds: promptTokenIds,
        maxGeneratedTokens: maxGeneratedTokens,
        seed: seed,
        temperature: temperature,
        topP: topP,
        frequencyPenalty: frequencyPenalty,
        includeTokenMetadata: includeTokenMetadata,
      );
    } finally {
      promptTokenIds.close();
    }
  }

  Future<Sarashina2DecodeResult> synthesizeText({
    required String text,
    PcmAudio? promptAudio,
    Sarashina2Prompt? prompt,
    String promptText = '',
    int maxGeneratedTokens = 2048,
    int latencyTokens = 1,
    int seed = 0,
    double temperature = sarashina2DefaultTemperature,
    double topP = sarashina2DefaultTopP,
    double frequencyPenalty = sarashina2DefaultFrequencyPenalty,
    bool includeFloatOutputs = true,
    bool includeTokenMetadata = true,
  }) async {
    if (text.trim().isEmpty) {
      throw ArgumentError('Sarashina2 text must not be empty');
    }
    final effectivePromptText = promptText.trim();
    if (effectivePromptText.isNotEmpty &&
        promptAudio == null &&
        prompt == null) {
      throw ArgumentError(
        'Sarashina2 promptText requires promptAudio or prompt.',
      );
    }

    final preparedPrompt = _preparePrompt(
      promptAudio: promptAudio,
      prompt: prompt,
    );
    try {
      final promptSemanticTokens = effectivePromptText.isEmpty
          ? const <int>[]
          : preparedPrompt.prompt!.promptSpeechTokens;
      final semanticTimer = Stopwatch()..start();
      final semanticTokens = _generateSemanticTokenSourceFromText(
        text: text,
        promptText: effectivePromptText,
        promptSemanticTokens: promptSemanticTokens,
        maxGeneratedTokens: maxGeneratedTokens,
        seed: seed,
        temperature: temperature,
        topP: topP,
        frequencyPenalty: frequencyPenalty,
        includeTokenMetadata: includeTokenMetadata,
      );
      semanticTimer.stop();
      try {
        return _decodePreparedSemanticTokens(
          text: text,
          semanticTokenSource: semanticTokens.source,
          semanticTokenValues: semanticTokens.tokens,
          prompt: preparedPrompt.prompt,
          latencyTokens: latencyTokens,
          promptElapsedMicroseconds: preparedPrompt.elapsedMicroseconds,
          semanticElapsedMicroseconds: semanticTimer.elapsedMicroseconds,
          includeFloatOutputs: includeFloatOutputs,
          includeTokenMetadata: includeTokenMetadata,
        );
      } finally {
        semanticTokens.close();
      }
    } finally {
      preparedPrompt.close();
    }
  }

  Future<Sarashina2SynthesisResult> synthesizeFrom({
    required String text,
    PcmAudio? promptAudio,
    Sarashina2Prompt? prompt,
    String promptText = '',
    List<int> semanticTokens = const [],
    String semanticTokenText = '',
    List<int> promptTokenIds = const [],
    int maxGeneratedTokens = 2048,
    int latencyTokens = 1,
    int seed = 0,
    double temperature = sarashina2DefaultTemperature,
    double topP = sarashina2DefaultTopP,
    double frequencyPenalty = sarashina2DefaultFrequencyPenalty,
    bool includeFloatOutputs = true,
    bool includeTokenMetadata = true,
  }) {
    return synthesize(
      Sarashina2SynthesisRequest(
        text: text,
        promptAudio: promptAudio,
        prompt: prompt,
        promptText: promptText,
        semanticTokens: semanticTokens,
        semanticTokenText: semanticTokenText,
        promptTokenIds: promptTokenIds,
        maxGeneratedTokens: maxGeneratedTokens,
        latencyTokens: latencyTokens,
        seed: seed,
        temperature: temperature,
        topP: topP,
        frequencyPenalty: frequencyPenalty,
        includeFloatOutputs: includeFloatOutputs,
        includeTokenMetadata: includeTokenMetadata,
      ),
    );
  }

  Future<Sarashina2SynthesisResult> synthesize(
    Sarashina2SynthesisRequest request,
  ) async {
    final externalTokens = _semanticTokenSourceFromInputs(
      semanticTokens: request.semanticTokens,
      semanticTokenText: request.semanticTokenText,
      includeTokenMetadata: request.includeTokenMetadata,
    );
    _SemanticTokenSource? generatedFromPromptTokenIds;
    try {
      if (externalTokens != null && request.promptTokenIds.isNotEmpty) {
        throw ArgumentError(
          'Pass either semantic tokens or prompt token ids, not both.',
        );
      }
      final promptTokenTimer = Stopwatch();
      generatedFromPromptTokenIds =
          externalTokens == null && request.promptTokenIds.isNotEmpty
          ? _semanticTokenSourceFromPromptTokenIds(
              promptTokenIds: request.promptTokenIds,
              maxGeneratedTokens: request.maxGeneratedTokens,
              seed: request.seed,
              temperature: request.temperature,
              topP: request.topP,
              frequencyPenalty: request.frequencyPenalty,
              includeTokenMetadata: request.includeTokenMetadata,
              timer: promptTokenTimer,
            )
          : null;
      final source = externalTokens != null
          ? Sarashina2SemanticSource.external
          : generatedFromPromptTokenIds != null
          ? Sarashina2SemanticSource.promptTokenIds
          : Sarashina2SemanticSource.text;
      final decode = source == Sarashina2SemanticSource.text
          ? await synthesizeText(
              text: request.text,
              promptAudio: request.promptAudio,
              prompt: request.prompt,
              promptText: request.promptText,
              maxGeneratedTokens: request.maxGeneratedTokens,
              latencyTokens: request.latencyTokens,
              seed: request.seed,
              temperature: request.temperature,
              topP: request.topP,
              frequencyPenalty: request.frequencyPenalty,
              includeFloatOutputs: request.includeFloatOutputs,
              includeTokenMetadata: request.includeTokenMetadata,
            )
          : externalTokens != null
          ? await _decodeSemanticTokenSource(
              text: request.text,
              semanticTokenSource: externalTokens.source,
              semanticTokenValues: externalTokens.tokens,
              promptAudio: request.promptAudio,
              prompt: request.prompt,
              latencyTokens: request.latencyTokens,
              includeFloatOutputs: request.includeFloatOutputs,
              includeTokenMetadata: request.includeTokenMetadata,
            )
          : await _decodeSemanticTokenSource(
              text: request.text,
              semanticTokenSource: generatedFromPromptTokenIds!.source,
              semanticTokenValues: generatedFromPromptTokenIds.tokens,
              promptAudio: request.promptAudio,
              prompt: request.prompt,
              latencyTokens: request.latencyTokens,
              includeFloatOutputs: request.includeFloatOutputs,
              includeTokenMetadata: request.includeTokenMetadata,
            );
      return Sarashina2SynthesisResult(
        decode: decode,
        semanticSource: source,
        promptTokenGenerationElapsedMicroseconds:
            promptTokenTimer.elapsedMicroseconds,
      );
    } finally {
      externalTokens?.close();
      generatedFromPromptTokenIds?.close();
    }
  }

  Future<Sarashina2DecodeResult> decodeSemanticTokens({
    required String text,
    required List<int> semanticTokens,
    PcmAudio? promptAudio,
    Sarashina2Prompt? prompt,
    int latencyTokens = 1,
    bool includeFloatOutputs = true,
    bool includeTokenMetadata = true,
  }) async {
    validateSarashina2SemanticTokens(semanticTokens);
    return _decodeSemanticTokenSource(
      text: text,
      semanticTokenSource: semanticTokens,
      semanticTokenValues: semanticTokens,
      promptAudio: promptAudio,
      prompt: prompt,
      latencyTokens: latencyTokens,
      includeFloatOutputs: includeFloatOutputs,
      includeTokenMetadata: includeTokenMetadata,
    );
  }

  Future<Sarashina2DecodeResult> _decodeSemanticTokenSource({
    required String text,
    required Object semanticTokenSource,
    required List<int> semanticTokenValues,
    PcmAudio? promptAudio,
    Sarashina2Prompt? prompt,
    required int latencyTokens,
    required bool includeFloatOutputs,
    required bool includeTokenMetadata,
  }) async {
    final preparedPrompt = _preparePrompt(
      promptAudio: promptAudio,
      prompt: prompt,
    );
    try {
      return _decodePreparedSemanticTokens(
        text: text,
        semanticTokenSource: semanticTokenSource,
        semanticTokenValues: semanticTokenValues,
        prompt: preparedPrompt.prompt,
        latencyTokens: latencyTokens,
        promptElapsedMicroseconds: preparedPrompt.elapsedMicroseconds,
        semanticElapsedMicroseconds: 0,
        includeFloatOutputs: includeFloatOutputs,
        includeTokenMetadata: includeTokenMetadata,
      );
    } finally {
      preparedPrompt.close();
    }
  }

  _SemanticTokenSource? _semanticTokenSourceFromInputs({
    required List<int> semanticTokens,
    required String semanticTokenText,
    required bool includeTokenMetadata,
  }) {
    final text = semanticTokenText.trim();
    NativeTensorBuffer? parsed;
    NativeTensorBuffer? source;
    try {
      if (text.isNotEmpty) {
        parsed = parseSarashina2SemanticTokensBuffer(text);
        if (parsed.byteLength == 0) {
          throw ArgumentError(
            'semanticTokenText did not contain Sarashina2 semantic tokens.',
          );
        }
      }
      final parsedCount = parsed == null ? 0 : parsed.byteLength ~/ 4;
      final tokenCount = semanticTokens.length + parsedCount;
      if (tokenCount == 0) {
        parsed?.close();
        return null;
      }
      source = NativeTensorBuffer.int32([tokenCount]);
      final values = source.asInt32List();
      var offset = 0;
      if (semanticTokens.isNotEmpty) {
        values.setAll(0, semanticTokens);
        offset = semanticTokens.length;
      }
      if (parsed != null) {
        values.setAll(offset, parsed.asInt32List());
      }
      if (semanticTokens.isNotEmpty) {
        _validateSemanticTokenSource(source);
      }
      final tokens = includeTokenMetadata
          ? List<int>.unmodifiable(values)
          : const <int>[];
      return _SemanticTokenSource._(
        source: source,
        tokens: tokens,
        owner: source,
      );
    } catch (_) {
      source?.close();
      rethrow;
    } finally {
      parsed?.close();
    }
  }

  _SemanticTokenSource _semanticTokenSourceFromPromptTokenIds({
    required Object promptTokenIds,
    required int maxGeneratedTokens,
    required int seed,
    required double temperature,
    required double topP,
    required double frequencyPenalty,
    required bool includeTokenMetadata,
    required Stopwatch timer,
  }) {
    timer.start();
    try {
      return _generateSemanticTokenSourceFromPromptTokenIds(
        promptTokenIds: promptTokenIds,
        maxGeneratedTokens: maxGeneratedTokens,
        seed: seed,
        temperature: temperature,
        topP: topP,
        frequencyPenalty: frequencyPenalty,
        includeTokenMetadata: includeTokenMetadata,
      );
    } finally {
      timer.stop();
    }
  }

  _PreparedPrompt _preparePrompt({
    required PcmAudio? promptAudio,
    required Sarashina2Prompt? prompt,
  }) {
    if (promptAudio != null && prompt != null) {
      throw ArgumentError('Pass either promptAudio or prompt, not both.');
    }
    if (prompt != null) {
      return _PreparedPrompt(prompt: prompt._prompt, elapsedMicroseconds: 0);
    }
    if (promptAudio == null) {
      return const _PreparedPrompt(prompt: null, elapsedMicroseconds: 0);
    }
    final extracted = extractPrompt(promptAudio);
    return _PreparedPrompt(
      prompt: extracted._prompt,
      elapsedMicroseconds: extracted.extractElapsedMicroseconds,
      ownsPrompt: true,
    );
  }

  Sarashina2DecodeResult _decodePreparedSemanticTokens({
    required String text,
    required Object semanticTokenSource,
    required List<int> semanticTokenValues,
    required SpeakerPrompt? prompt,
    required int latencyTokens,
    required int promptElapsedMicroseconds,
    required int semanticElapsedMicroseconds,
    required bool includeFloatOutputs,
    required bool includeTokenMetadata,
  }) {
    if (latencyTokens < 0) {
      throw RangeError.range(latencyTokens, 0, null, 'latencyTokens');
    }
    final semanticTokenCount = _int32TokenCount(semanticTokenSource);
    if (semanticTokenCount <= latencyTokens) {
      throw ArgumentError(
        'semanticTokens must contain more than $latencyTokens latency token(s)',
      );
    }
    final decodedSemanticTokenCount = semanticTokenCount - latencyTokens;
    final semanticTokens = includeTokenMetadata
        ? List<int>.unmodifiable(
            semanticTokenValues.isEmpty
                ? _int32TokenValues(semanticTokenSource)
                : semanticTokenValues,
          )
        : const <int>[];
    final decodedTokens = includeTokenMetadata
        ? semanticTokens.skip(latencyTokens).toList(growable: false)
        : const <int>[];
    final decodeTimer = Stopwatch()..start();
    final flow = _flow.synthesizeTokens(
      prompt: prompt,
      generatedSpeechTokens: semanticTokenSource,
      generatedTokenOffset: latencyTokens,
      generatedTokenCount: decodedSemanticTokenCount,
      includeFloatOutputs: includeFloatOutputs,
    );
    decodeTimer.stop();
    return Sarashina2DecodeResult(
      text: text,
      audio: flow.audio,
      audioSampleCount: flow.audioSampleCount,
      audioWavBytes: flow.audioWavBytes,
      sampleRate: cosyvoice2SampleRate,
      semanticTokens: semanticTokens,
      decodedSemanticTokens: decodedTokens,
      semanticTokenCount: semanticTokenCount,
      decodedSemanticTokenCount: decodedSemanticTokenCount,
      melFrames: flow.melFrames,
      promptElapsedMicroseconds: promptElapsedMicroseconds,
      semanticElapsedMicroseconds: semanticElapsedMicroseconds,
      decodeElapsedMicroseconds: decodeTimer.elapsedMicroseconds,
      usedPrompt: prompt != null,
      flowPrepareElapsedMicroseconds: flow.prepareElapsedMicroseconds,
      flowEncoderElapsedMicroseconds: flow.encoderElapsedMicroseconds,
      flowSetupElapsedMicroseconds: flow.setupElapsedMicroseconds,
      flowDiffuseElapsedMicroseconds: flow.diffuseElapsedMicroseconds,
      flowVocodeElapsedMicroseconds: flow.vocodeElapsedMicroseconds,
    );
  }

  void close() {
    _promptExtractor.close();
    _flow.close();
    _tokenizer?.close();
    _llm?.close();
    _bundle.close();
  }
}

Map<String, Object?> _sarashina2BackendOptions(
  String provider,
  Map<String, Object?> backendOptions,
) {
  final additions = <String, Object?>{};
  if (canonicalOnnxExecutionProvider(provider) == 'TensorrtExecutionProvider' &&
      _backendBool(backendOptions, 'sarashina2TensorRtFlowStepProfile', true) &&
      !backendOptions.containsKey('trtProfileMinShapes') &&
      !backendOptions.containsKey('trtProfileOptShapes') &&
      !backendOptions.containsKey('trtProfileMaxShapes')) {
    final minFrames = max(
      1,
      _backendInt(backendOptions, 'sarashina2TensorRtMinFrames', 16),
    );
    final optFrames = max(
      minFrames,
      _backendInt(backendOptions, 'sarashina2TensorRtOptFrames', 384),
    );
    final maxFrames = max(
      optFrames,
      _backendInt(backendOptions, 'sarashina2TensorRtMaxFrames', 1024),
    );
    additions['trtProfileMinShapes'] = _flowStepProfileShapes(minFrames);
    additions['trtProfileOptShapes'] = _flowStepProfileShapes(optFrames);
    additions['trtProfileMaxShapes'] = _flowStepProfileShapes(maxFrames);
  }
  final useTunableOps = _backendBool(
    backendOptions,
    'sarashina2UseCudaTunableOps',
    false,
  );
  if (!useTunableOps) {
    return additions.isEmpty
        ? backendOptions
        : {...additions, ...backendOptions};
  }
  return {
    ...additions,
    'cudaTunableOpEnable': true,
    'cudaTunableOpTuningEnable': true,
    'cudaTunableOpMaxTuningMs': 10,
    ...backendOptions,
  };
}

String _flowStepProfileShapes(int frames) {
  return [
    'x:1x80x$frames',
    'mask:2x1x$frames',
    'mu:2x80x$frames',
    't:2',
    'spks:2x80',
    'cond:2x80x$frames',
    'dt:1',
  ].join(',');
}

int _backendInt(Map<String, Object?> backendOptions, String key, int fallback) {
  final value = backendOptions[key];
  if (value is int) {
    return value;
  }
  if (value is String && value.trim().isNotEmpty) {
    return int.parse(value.trim());
  }
  return fallback;
}

List<String> _providerBlockers({
  required List<CosyVoice2ComponentStatus> statuses,
  required Set<String> components,
  required String provider,
  required Map<String, String> componentProviders,
  required bool requireProvider,
}) {
  if (!requireProvider) {
    return const [];
  }
  final blockers = <String>[];
  for (final status in statuses) {
    if (!components.contains(status.file.name) || !status.loaded) {
      continue;
    }
    final expectedProvider = canonicalOnnxExecutionProvider(
      componentProviders[status.file.name] ?? provider,
    );
    if (expectedProvider == null ||
        status.selectedProvider == expectedProvider) {
      continue;
    }
    blockers.add(
      '${status.file.name} selected provider '
      '${status.selectedProvider ?? 'unknown'}, expected $expectedProvider',
    );
  }
  return blockers;
}

Map<String, String> _sarashina2ComponentProviders(
  String provider,
  Map<String, Object?> backendOptions,
) {
  if (canonicalOnnxExecutionProvider(provider) != 'TensorrtExecutionProvider') {
    return const {};
  }
  if (!_backendBool(backendOptions, 'sarashina2TensorRtHybrid', true)) {
    return const {};
  }
  final fallbackProvider = _backendString(
    backendOptions,
    'sarashina2TensorRtFallbackProvider',
    'cuda',
  );
  return {
    for (final component in const [
      'campplus',
      'speech_tokenizer_v2',
      'flow_encoder_fp32',
      'hift',
      'llm_prefill',
      'llm_decode',
      'llm_decode_head',
      'llm_decoder_head',
    ])
      component: fallbackProvider,
  };
}

String _backendString(
  Map<String, Object?> backendOptions,
  String key,
  String fallback,
) {
  final value = backendOptions[key];
  if (value is String && value.trim().isNotEmpty) {
    return value.trim();
  }
  return fallback;
}

bool _useFusedDecodeHead(
  Sarashina2TtsPaths paths,
  Map<String, Object?> backendOptions,
  String precision,
) {
  if (!_backendBool(backendOptions, 'sarashina2LlmUseFusedDecodeHead', false)) {
    return false;
  }
  return sarashina2LlmDecodeHeadExists(paths, precision);
}

bool _backendBool(
  Map<String, Object?> backendOptions,
  String key,
  bool fallback,
) {
  final value = backendOptions[key];
  if (value is bool) {
    return value;
  }
  if (value is String) {
    final normalized = value.trim().toLowerCase();
    if (normalized == '1' || normalized == 'true' || normalized == 'yes') {
      return true;
    }
    if (normalized == '0' || normalized == 'false' || normalized == 'no') {
      return false;
    }
  }
  return fallback;
}
