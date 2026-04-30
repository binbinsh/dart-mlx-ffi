import 'dart:math';
import 'dart:typed_data';

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import 'cosyvoice2.dart';
import 'cosyvoice2_audio.dart';
import 'cosyvoice2_flow_driver.dart';
import 'cosyvoice2_llm_driver.dart';
import 'cosyvoice2_ras_sampler.dart';
import 'cosyvoice2_speaker_prompt.dart';

const int cosyvoice2EosToken = 6561;

final class CosyVoice2SynthesisResult {
  const CosyVoice2SynthesisResult({
    required this.text,
    required this.audio,
    required this.audioSampleCount,
    required this.audioWavBytes,
    required this.sampleRate,
    required this.generatedSpeechTokens,
    required this.generatedSpeechTokenCount,
    required this.melFrames,
    required this.llmElapsedMicroseconds,
    required this.flowElapsedMicroseconds,
    required this.promptElapsedMicroseconds,
    required this.usedPrompt,
    required this.usedStreamingHift,
  });

  final String text;
  final Float32List audio;
  final int audioSampleCount;
  final Uint8List audioWavBytes;
  final int sampleRate;
  final List<int> generatedSpeechTokens;
  final int generatedSpeechTokenCount;
  final int melFrames;
  final int llmElapsedMicroseconds;
  final int flowElapsedMicroseconds;
  final int promptElapsedMicroseconds;
  final bool usedPrompt;
  final bool usedStreamingHift;
}

final class CosyVoice2DartRuntime {
  CosyVoice2DartRuntime._({
    required this.paths,
    required CosyVoice2PartialOnnxBundle bundle,
    required CosyVoice2LlmDriver llm,
    required CosyVoice2FlowDriver flow,
  }) : _bundle = bundle,
       _llm = llm,
       _flow = flow,
       _promptExtractor = SpeakerPromptExtractor(bundle: bundle);

  final CosyVoice2Paths paths;
  final CosyVoice2PartialOnnxBundle _bundle;
  final CosyVoice2LlmDriver _llm;
  final CosyVoice2FlowDriver _flow;
  final SpeakerPromptExtractor _promptExtractor;

  List<String> get loadedComponentNames => _bundle.loadedComponentNames;

  List<String> get selectedProviders => [
    for (final component in _bundle.loadedComponents)
      '${component.name}:${component.selectedProvider}',
  ];

  static Future<CosyVoice2DartRuntime> load({
    required CosyVoice2Paths paths,
    required String provider,
    required int deviceId,
    required bool requireProvider,
    required int numThreads,
    Map<String, Object?> backendOptions = const {},
    bool loadStreamingHift = true,
  }) async {
    final components = {
      'campplus',
      'speech_tokenizer_v2',
      'llm_prefill',
      'llm_decode',
      'llm_decoder_head',
      'flow_encoder_fp32',
      'flow_decoder_estimator_fp32',
      'hift',
      if (loadStreamingHift) 'hift_streaming',
    };
    final bundle = CosyVoice2PartialOnnxBundle.load(
      paths: paths,
      provider: provider,
      deviceId: deviceId,
      requireProvider: requireProvider,
      numThreads: numThreads,
      backendOptions: backendOptions,
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
        for (final blocker in bundle.blockers)
          if (blocker.contains('support asset')) blocker,
      ];
      if (blockers.isNotEmpty) {
        throw StateError(
          'CosyVoice2 runtime is not loadable: ${blockers.join('; ')}',
        );
      }
      final llm = await CosyVoice2LlmDriver.load(bundle: bundle, paths: paths);
      try {
        final flow = await CosyVoice2FlowDriver.load(
          bundle: bundle,
          paths: paths,
        );
        return CosyVoice2DartRuntime._(
          paths: paths,
          bundle: bundle,
          llm: llm,
          flow: flow,
        );
      } catch (_) {
        llm.close();
        rethrow;
      }
    } catch (_) {
      bundle.close();
      rethrow;
    }
  }

  Future<CosyVoice2SynthesisResult> synthesize({
    required String text,
    PcmAudio? promptAudio,
    String promptText = '',
    int? maxGeneratedTokens,
    int rasSeed = 0,
    bool useStreamingHift = false,
    bool includeFloatOutputs = true,
    bool includeTokenMetadata = true,
  }) async {
    if (text.trim().isEmpty) {
      throw ArgumentError('CosyVoice2 text must not be empty');
    }
    final promptTimer = Stopwatch()..start();
    final SpeakerPrompt? prompt = promptAudio == null
        ? null
        : _promptExtractor.extract(promptAudio.samples, promptAudio.sampleRate);
    promptTimer.stop();
    try {
      final llmTimer = Stopwatch()..start();
      final promptTokens = prompt?.promptSpeechTokensSource ?? const <int>[];
      final prefill = _llm.buildPrefillEmbeddingBufferFromText(
        text: text,
        promptText: promptText.trim().isEmpty ? '' : promptText,
        promptSpeechTokens: promptTokens,
      );
      if (prefill.targetTextTokenCount == 0) {
        prefill.close();
        throw StateError(
          'CosyVoice2 tokenizer produced no target text tokens.',
        );
      }
      final CosyVoice2LlmState state;
      try {
        state = _llm.prefill(
          inputsEmbeds: prefill.tensor,
          seqLen: prefill.seqLen,
        );
      } finally {
        prefill.close();
      }
      final generated = _decodeSpeechTokens(
        state: state,
        targetTextTokenCount: prefill.targetTextTokenCount,
        maxGeneratedTokens: maxGeneratedTokens,
        rasSeed: rasSeed,
        includeTokenMetadata: includeTokenMetadata,
      );
      llmTimer.stop();

      final CosyVoice2FlowOutput flow;
      final flowTimer = Stopwatch()..start();
      try {
        flow = _flow.synthesizeTokens(
          prompt: prompt,
          generatedSpeechTokens: generated.tokenSource,
          useStreamingHift: useStreamingHift,
          includeFloatOutputs: includeFloatOutputs,
        );
      } finally {
        generated.close();
      }
      flowTimer.stop();

      return CosyVoice2SynthesisResult(
        text: text,
        audio: flow.audio,
        audioSampleCount: flow.audioSampleCount,
        audioWavBytes: flow.audioWavBytes,
        sampleRate: cosyvoice2SampleRate,
        generatedSpeechTokens: generated.tokens,
        generatedSpeechTokenCount: generated.tokenCount,
        melFrames: flow.melFrames,
        llmElapsedMicroseconds: llmTimer.elapsedMicroseconds,
        flowElapsedMicroseconds: flowTimer.elapsedMicroseconds,
        promptElapsedMicroseconds: promptTimer.elapsedMicroseconds,
        usedPrompt: prompt != null,
        usedStreamingHift: useStreamingHift,
      );
    } finally {
      prompt?.close();
    }
  }

  void close() {
    _promptExtractor.close();
    _flow.close();
    _llm.close();
    _bundle.close();
  }

  _DecodedSpeechTokens _decodeSpeechTokens({
    required CosyVoice2LlmState state,
    required int targetTextTokenCount,
    required int? maxGeneratedTokens,
    required int rasSeed,
    required bool includeTokenMetadata,
  }) {
    final minLen = max(1, targetTextTokenCount * 2);
    final maxLen =
        maxGeneratedTokens ?? max(minLen + 1, targetTextTokenCount * 20);
    if (maxLen < 1) {
      throw ArgumentError('maxGeneratedTokens must be positive');
    }
    RasDecodeBuffer? sampler;
    NativeTensorBuffer? nextEmbed;
    var keepSampler = false;
    try {
      sampler = RasDecodeBuffer(maxTokens: maxLen, rng: Random(rasSeed));
      nextEmbed = _llm.createSpeechTokenEmbeddingBuffer();
      final nextEmbedTensor = nextEmbed.tensor;
      for (var step = 0; step < maxLen; step += 1) {
        final logits = _llm.headLogitsTensor(state.lastHidden);
        final int token;
        try {
          token = sampler.sampleAndAppendNonEos(
            logits.tensor,
            eosToken: cosyvoice2EosToken,
            ignoreEos: step < minLen,
          );
        } finally {
          logits.close();
        }
        if (token == cosyvoice2EosToken) {
          break;
        }
        _llm.fillSpeechTokenEmbeddingBuffer(id: token, out: nextEmbed);
        _llm.decodeStep(state: state, nextEmbed: nextEmbedTensor);
      }
      final generatedCount = sampler.length;
      if (generatedCount == 0) {
        throw StateError('CosyVoice2 LLM produced no speech tokens.');
      }
      final generated = includeTokenMetadata ? sampler.toList() : const <int>[];
      keepSampler = true;
      return _DecodedSpeechTokens(
        tokens: generated,
        tokenCount: generatedCount,
        tokenSource: sampler.tokensTensor(),
        owner: sampler,
      );
    } finally {
      nextEmbed?.close();
      if (!keepSampler) {
        sampler?.close();
      }
      state.close();
    }
  }
}

final class _DecodedSpeechTokens {
  _DecodedSpeechTokens({
    required this.tokens,
    required this.tokenCount,
    required this.tokenSource,
    required RasDecodeBuffer owner,
  }) : _owner = owner;

  final List<int> tokens;
  final int tokenCount;
  final Object tokenSource;
  final RasDecodeBuffer _owner;

  void close() {
    _owner.close();
  }
}
