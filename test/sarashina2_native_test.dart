import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';
import 'package:dart_inference/src/models/sarashina2/sarashina2_flow_options.dart'
    show
        resolveSarashina2TensorRtFlowStepContextPath,
        resolveSarashina2UseDeviceFlowLoop,
        resolveSarashina2FlowSteps,
        resolveSarashina2FlowStepPrecision,
        sarashina2UseFusedFlowLoop,
        sarashina2UseFusedFlowStep;
import 'package:dart_inference/src/models/sarashina2/sarashina2_llm_options.dart'
    show
        sarashina2BaseComponentBackendOptions,
        sarashina2LlmComponentBackendOptions,
        sarashina2LlmPathOverrides;
import 'package:dart_inference/src/models/sarashina2/sarashina2_native.dart'
    show validateSarashinaSemanticTokensNative;
import 'package:test/test.dart';

void main() {
  test('Sarashina2 semantic token helpers use the native FFI path', () {
    final parsed = parseSarashina2SemanticTokens(
      'noise<|semantic_0|><|semantic_42|><|semantic_6560|>',
    );
    final parsedBuffer = parseSarashina2SemanticTokensBuffer(
      'noise<|semantic_0|><|semantic_42|><|semantic_6560|>',
    );

    try {
      expect(parsed, [0, 42, 6560]);
      expect(parsedBuffer.asInt32List(), parsed);
      expect(
        sarashina2SemanticTokensToText(parsed),
        '<|semantic_0|><|semantic_42|><|semantic_6560|>',
      );
      expect(
        () => sarashina2SemanticTokensToText([sarashina2SemanticVocabSize]),
        throwsStateError,
      );
      expect(
        () => validateSarashina2SemanticTokens([0, 42, 6560]),
        returnsNormally,
      );
      expect(
        () => validateSarashinaSemanticTokensNative(parsedBuffer),
        returnsNormally,
      );
      expect(
        () => validateSarashina2SemanticTokens([sarashina2SemanticVocabSize]),
        throwsRangeError,
      );
      final invalidBuffer = NativeTensorBuffer.int32([1]);
      try {
        invalidBuffer.asInt32List()[0] = sarashina2SemanticVocabSize;
        expect(
          () => validateSarashinaSemanticTokensNative(invalidBuffer),
          throwsStateError,
        );
      } finally {
        invalidBuffer.close();
      }
    } finally {
      parsedBuffer.close();
    }
  });

  test('Sarashina2 prompt builder mirrors upstream prompt layout', () {
    expect(
      buildSarashina2Prompt(
        text: '東京（かなざわ）',
        promptText: '参照文',
        promptTokens: const [1, 23],
      ),
      '参照文東京「かなざわ」<|speech_start|><|semantic_1|><|semantic_23|>',
    );
    expect(
      buildSarashina2Prompt(text: '**hello**', preprocessText: true),
      'hello<|speech_start|>',
    );
  });

  test('Sarashina2 paths point at the UniFrontend provider snapshot', () {
    final paths = Sarashina2TtsPaths.fromUniFrontendRoot('/repo/');

    expect(
      paths.modelSafetensors,
      '/repo/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts/model.safetensors',
    );
    expect(
      paths.inspect().map((status) => status.name),
      contains('flow_checkpoint'),
    );
    expect(
      paths.inspect().map((status) => status.name),
      contains('llm_decode_head_onnx'),
    );
  });

  test('Sarashina2 LLM options keep decode hidden on CUDA handles', () {
    final options = sarashina2LlmComponentBackendOptions(
      paths: const Sarashina2TtsPaths(modelDir: '/models/sarashina2'),
      provider: 'cuda',
      deviceId: 0,
      precision: 'bf16',
      backendOptions: const {'sarashina2LlmUseOptimizedGraphs': false},
    );
    final prefillNames = options['llm_prefill']!['deviceOutputNames'] as String;
    final decodeNames = options['llm_decode']!['deviceOutputNames'] as String;

    expect(prefillNames, contains('present_key_0'));
    expect(prefillNames, contains('present_value_23'));
    expect(prefillNames.split(','), isNot(contains('hidden')));
    expect(decodeNames.split(','), containsAll(['hidden', 'present_key_0']));
    expect(options['llm_prefill']!['useEnvAllocators'], isTrue);
    expect(options['llm_decode']!.containsKey('cpuOutputNames'), isFalse);
    expect(options['llm_decode']!['syncBoundOutputs'], isFalse);
    expect(options['llm_decode']!['cacheBoundOutputs'], isTrue);
    expect(
      options['llm_decoder_head']!.containsKey('useDeviceOutputs'),
      isFalse,
    );
    final cpuHiddenOptions = sarashina2LlmComponentBackendOptions(
      paths: const Sarashina2TtsPaths(modelDir: '/models/sarashina2'),
      provider: 'cuda',
      deviceId: 0,
      precision: 'bf16',
      backendOptions: const {
        'sarashina2LlmUseOptimizedGraphs': false,
        'sarashina2LlmUseDeviceHidden': false,
      },
    );
    expect(cpuHiddenOptions['llm_decode']!['cpuOutputNames'], 'hidden');
    expect(
      cpuHiddenOptions['llm_decode']!.containsKey('syncBoundOutputs'),
      isFalse,
    );
    expect(
      cpuHiddenOptions['llm_decode']!.containsKey('cacheBoundOutputs'),
      isFalse,
    );

    final cudaGraphOptions = sarashina2LlmComponentBackendOptions(
      paths: const Sarashina2TtsPaths(modelDir: '/models/sarashina2'),
      provider: 'cuda',
      deviceId: 0,
      precision: 'bf16',
      backendOptions: const {
        'sarashina2LlmUseOptimizedGraphs': false,
        'sarashina2LlmCudaGraph': true,
        'sarashina2LlmCudaGraphId': 7,
        'sarashina2LlmDecodeSyncOutputs': false,
        'sarashina2LlmDecodeCacheBoundOutputs': true,
      },
    );
    expect(cudaGraphOptions['llm_decode']!['cudaEnableGraph'], isTrue);
    expect(cudaGraphOptions['llm_decode']!['cudaGraphId'], 7);
    expect(cudaGraphOptions['llm_decode']!['syncBoundOutputs'], isFalse);
    expect(cudaGraphOptions['llm_decode']!['cacheBoundOutputs'], isTrue);
  });

  test('Sarashina2 fused decode-head keeps logits on CPU and KV on device', () {
    final dir = Directory.systemTemp.createTempSync('sarashina2-llm-options-');
    try {
      final paths = Sarashina2TtsPaths(modelDir: dir.path);
      File(paths.llmDecodeHeadBf16Onnx).writeAsBytesSync([1]);
      final options = sarashina2LlmComponentBackendOptions(
        paths: paths,
        provider: 'cuda',
        deviceId: 0,
        precision: 'bf16',
        backendOptions: const {'sarashina2LlmUseOptimizedGraphs': false},
      );
      final names = options['llm_decode_head']!['deviceOutputNames'] as String;

      expect(
        names.split(','),
        containsAll(['present_key_0', 'present_value_23']),
      );
      expect(names.split(','), isNot(contains('hidden')));
      expect(names.split(','), isNot(contains('logits')));
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('Sarashina2 skips optimized cache for external-data LLM graphs', () {
    final dir = Directory.systemTemp.createTempSync('sarashina2-llm-cache-');
    try {
      final paths = Sarashina2TtsPaths(modelDir: dir.path);
      File(paths.llmPrefillBf16Onnx).writeAsBytesSync([1]);
      File(paths.llmDecodeBf16Onnx).writeAsBytesSync([1]);
      File(paths.llmDecoderHeadBf16Onnx).writeAsBytesSync([1]);
      File(paths.llmDecodeHeadBf16Onnx).writeAsBytesSync([1]);
      File('${paths.llmDecodeHeadBf16Onnx}.data').writeAsBytesSync([1]);

      final overrides = sarashina2LlmPathOverrides(
        paths,
        'bf16',
        provider: 'cuda',
        deviceId: 0,
        backendOptions: const {},
      );
      final options = sarashina2LlmComponentBackendOptions(
        paths: paths,
        provider: 'cuda',
        deviceId: 0,
        precision: 'bf16',
        backendOptions: const {},
      );

      expect(overrides['llm_decode_head'], paths.llmDecodeHeadBf16Onnx);
      expect(
        options['llm_decode_head']!.containsKey('optimizedModelFilePath'),
        isFalse,
      );
      expect(
        options['llm_decode']!['optimizedModelFilePath'],
        contains('.dart_inference_ort_cache'),
      );
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('Sarashina2 prefers last-hidden prefill export when present', () {
    final dir = Directory.systemTemp.createTempSync('sarashina2-llm-prefill-');
    try {
      final paths = Sarashina2TtsPaths(modelDir: dir.path);
      File(paths.llmPrefillBf16Onnx).writeAsBytesSync([1]);
      File(paths.llmPrefillLastBf16Onnx).writeAsBytesSync([1]);
      File(paths.llmDecodeBf16Onnx).writeAsBytesSync([1]);
      File(paths.llmDecoderHeadBf16Onnx).writeAsBytesSync([1]);

      final overrides = sarashina2LlmPathOverrides(
        paths,
        'bf16',
        provider: 'cuda',
        deviceId: 0,
        backendOptions: const {'sarashina2LlmUseOptimizedGraphs': false},
      );
      expect(overrides['llm_prefill'], paths.llmPrefillLastBf16Onnx);
      final options = sarashina2LlmComponentBackendOptions(
        paths: paths,
        provider: 'cuda',
        deviceId: 0,
        precision: 'bf16',
        backendOptions: const {'sarashina2LlmUseOptimizedGraphs': false},
      );
      expect(
        (options['llm_prefill']!['deviceOutputNames'] as String).split(','),
        contains('hidden'),
      );
      final hostHidden = sarashina2LlmComponentBackendOptions(
        paths: paths,
        provider: 'cuda',
        deviceId: 0,
        precision: 'bf16',
        backendOptions: const {
          'sarashina2LlmUseOptimizedGraphs': false,
          'sarashina2LlmUsePrefillDeviceHidden': false,
        },
      );
      expect(
        (hostHidden['llm_prefill']!['deviceOutputNames'] as String).split(','),
        isNot(contains('hidden')),
      );

      final disabled = sarashina2LlmPathOverrides(
        paths,
        'bf16',
        provider: 'cuda',
        deviceId: 0,
        backendOptions: const {
          'sarashina2LlmUseOptimizedGraphs': false,
          'sarashina2LlmUsePrefillLastHidden': false,
        },
      );
      expect(disabled['llm_prefill'], paths.llmPrefillBf16Onnx);
      final deviceHidden = sarashina2LlmComponentBackendOptions(
        paths: paths,
        provider: 'cuda',
        deviceId: 0,
        precision: 'bf16',
        backendOptions: const {
          'sarashina2LlmUseOptimizedGraphs': false,
          'sarashina2LlmUsePrefillDeviceHidden': true,
        },
      );
      expect(
        (deviceHidden['llm_prefill']!['deviceOutputNames'] as String).split(
          ',',
        ),
        contains('hidden'),
      );
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('Sarashina2 base ONNX components use CPU output views', () {
    expect(sarashina2BaseComponentBackendOptions(const {}), isEmpty);

    final options = sarashina2BaseComponentBackendOptions(const {
      'sarashina2UseOutputViews': true,
    });

    expect(options['flow_decoder_estimator_fp32']!['useOutputViews'], isTrue);
    expect(options['hift']!['useOutputViews'], isTrue);
    expect(
      sarashina2BaseComponentBackendOptions(const {
        'sarashina2UseOutputViews': false,
      }),
      isEmpty,
    );
  });

  test('Sarashina2 flow options prefer FP16 step on CUDA auto', () {
    final dir = Directory.systemTemp.createTempSync('sarashina2-flow-options-');
    try {
      final paths = Sarashina2TtsPaths(modelDir: dir.path);

      expect(sarashina2UseFusedFlowStep(paths, const {}), isFalse);
      expect(sarashina2UseFusedFlowLoop(paths, const {}), isFalse);
      expect(resolveSarashina2FlowSteps(const {}), 10);
      expect(resolveSarashina2FlowSteps(const {'sarashina2FlowSteps': 6}), 6);
      expect(resolveSarashina2FlowSteps(const {'sarashina2FlowSteps': '8'}), 8);
      expect(resolveSarashina2UseDeviceFlowLoop(const {}), isFalse);
      expect(
        resolveSarashina2UseDeviceFlowLoop(const {}, provider: 'tensorrt'),
        isFalse,
      );
      expect(
        resolveSarashina2UseDeviceFlowLoop(const {
          'sarashina2UseDeviceFlowLoop': true,
        }, provider: 'tensorrt'),
        isTrue,
      );
      final contextPath = '${dir.path}/flow_step_ctx.onnx';
      expect(
        resolveSarashina2TensorRtFlowStepContextPath(paths, const {
          'sarashina2TensorRtFlowStepContextPath': '',
        }),
        isNull,
      );
      expect(
        () => resolveSarashina2TensorRtFlowStepContextPath(paths, {
          'sarashina2TensorRtFlowStepContextPath': contextPath,
        }),
        throwsStateError,
      );
      File(contextPath).writeAsBytesSync([1]);
      expect(
        resolveSarashina2TensorRtFlowStepContextPath(paths, {
          'sarashina2TensorRtFlowStepContextPath': contextPath,
        }),
        contextPath,
      );
      File(paths.flowDecoderStepTensorRtContextOnnx).writeAsBytesSync([1]);
      expect(
        resolveSarashina2TensorRtFlowStepContextPath(paths, const {}),
        paths.flowDecoderStepTensorRtContextOnnx,
      );
      expect(
        resolveSarashina2TensorRtFlowStepContextPath(paths, const {
          'sarashina2TensorRtUseFlowStepContext': false,
        }),
        isNull,
      );
      expect(
        () => resolveSarashina2FlowSteps(const {'sarashina2FlowSteps': 0}),
        throwsRangeError,
      );
      expect(resolveSarashina2FlowStepPrecision(paths, const {}), 'fp32');

      File(paths.flowDecoderStepOnnx).writeAsBytesSync([1]);
      expect(sarashina2UseFusedFlowStep(paths, const {}), isTrue);
      expect(
        () => resolveSarashina2FlowStepPrecision(paths, const {
          'sarashina2FlowStepPrecision': 'fp16',
        }),
        throwsStateError,
      );

      File(paths.flowDecoderStepFp16Onnx).writeAsBytesSync([1]);
      expect(resolveSarashina2FlowStepPrecision(paths, const {}), 'fp32');
      expect(
        resolveSarashina2FlowStepPrecision(paths, const {}, provider: 'cuda'),
        'fp16',
      );
      expect(
        resolveSarashina2FlowStepPrecision(paths, const {
          'sarashina2FlowStepPrecision': 'fp16',
        }),
        'fp16',
      );
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('Sarashina2 token map resolves semantic tokenizer ids', () {
    final dir = Directory.systemTemp.createTempSync('sarashina2-map-');
    try {
      final path = '${dir.path}/added_tokens.json';
      File(path).writeAsStringSync(
        jsonEncode({
          '<|semantic_0|>': sarashina2SemanticTokenBaseId,
          '<|semantic_${sarashina2SemanticVocabSize - 1}|>':
              sarashina2SemanticTokenBaseId + sarashina2SemanticVocabSize - 1,
          sarashina2SpeechStartToken: sarashina2SpeechStartTokenId,
          '<|speech_end|>': sarashina2SpeechEndTokenId,
        }),
      );

      final tokenMap = Sarashina2TokenMap.fromAddedTokensFile(path);

      expect(tokenMap.tokenizerIdForSemantic(0), sarashina2SemanticTokenBaseId);
      expect(tokenMap.tokenizerIdForSemantic(42), 102442);
      expect(tokenMap.semanticIdForTokenizerId(102442), 42);
      expect(
        tokenMap.semanticIdForTokenizerId(sarashina2SpeechStartTokenId),
        isNull,
      );
      expect(tokenMap.speechStartTokenId, sarashina2SpeechStartTokenId);
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('Sarashina2 base tokenizer runs through native sidecar runtime', () {
    final sidecar = [
      'sara2tok\t1',
      'meta\tunk_id\t0',
      'meta\treplacement_hex\te29681',
      'meta\tprepend_scheme\tnever',
      'tok\t0\t-100\t3c756e6b3e',
      'tok\t1\t1\t68',
      'tok\t2\t1\t65',
      'tok\t3\t1\t6c',
      'tok\t4\t1\t6f',
      'tok\t5\t4\t6865',
      'tok\t6\t4\t6c6c6f',
      'tok\t7\t1\te29681',
      'add\t99\t0\t3c7c7370656563685f73746172747c3e',
      'add\t101\t0\t3c7c73656d616e7469635f317c3e',
      'add\t123\t0\t3c7c73656d616e7469635f32337c3e',
      '',
    ].join('\n');
    final tokenizer = Sarashina2BaseTokenizer.fromBytes(
      utf8.encode(sidecar),
      tokenMap: const Sarashina2TokenMap(
        semanticTokenBaseId: 100,
        semanticVocabSize: 32,
        speechStartTokenId: 99,
        speechEndTokenId: 98,
      ),
    );
    try {
      expect(tokenizer.encode('hello <|speech_start|>'), [5, 6, 7, 99]);
      final encoded = tokenizer.encodeBuffer('hello <|speech_start|>');
      try {
        expect(encoded.shape, [4]);
        expect(encoded.asInt32List(), [5, 6, 7, 99]);
      } finally {
        encoded.close();
      }
      final expectedPromptIds = tokenizer
          .encode('hehello<|speech_start|><|semantic_1|><|semantic_23|>')
          .toList(growable: false);
      expect(
        tokenizer.encodePromptTokenIds(
          text: 'hello',
          promptText: 'he',
          promptTokens: const [1, 23],
        ),
        expectedPromptIds,
      );
      final promptTokens = NativeTensorBuffer.int32([2]);
      promptTokens.asInt32List().setAll(0, [1, 23]);
      final promptIds = tokenizer.encodePromptTokenIdsBuffer(
        text: 'hello',
        promptText: 'he',
        promptTokens: promptTokens,
      );
      try {
        expect(promptIds.asInt32List(), expectedPromptIds);
      } finally {
        promptIds.close();
        promptTokens.close();
      }
      expect(
        () => tokenizer.encodePromptTokenIds(
          text: 'hello',
          promptTokens: const [1],
        ),
        throwsArgumentError,
      );
      expect(
        () => tokenizer.encodePromptTokenIds(text: 'hello', promptText: 'he'),
        throwsArgumentError,
      );
      expect(tokenizer.encode('x'), [0]);
    } finally {
      tokenizer.close();
    }
  });

  test('Sarashina2 base tokenizer uses Unigram byte fallback ids', () {
    final sidecar = [
      'sara2tok\t1',
      'meta\tunk_id\t0',
      'meta\tbyte_fallback\t1',
      'meta\treplacement_hex\te29681',
      'meta\tprepend_scheme\tnever',
      'tok\t0\t-100\t3c756e6b3e',
      for (var value = 0; value < 256; value += 1)
        'tok\t${200 + value}\t0\t${_hex('<0x${value.toRadixString(16).padLeft(2, '0').toUpperCase()}>')}',
      '',
    ].join('\n');
    final tokenizer = Sarashina2BaseTokenizer.fromBytes(utf8.encode(sidecar));
    try {
      expect(tokenizer.encode('\t\n'), [209, 210]);
      final encoded = tokenizer.encodeBuffer('\t\n');
      try {
        expect(encoded.asInt32List(), [209, 210]);
      } finally {
        encoded.close();
      }
      expect(tokenizer.encode('x'), [320]);
    } finally {
      tokenizer.close();
    }
  });

  test('Sarashina2 semantic sampler runs frequency penalty in native', () {
    final logits = Float32List(20);
    logits[sarashina2EosTokenId] = 0;
    logits[10] = 1;
    logits[11] = 4;
    logits[12] = 3;
    const tokenMap = Sarashina2TokenMap(
      semanticTokenBaseId: 10,
      semanticVocabSize: 3,
      speechStartTokenId: sarashina2SpeechStartTokenId,
      speechEndTokenId: sarashina2SpeechEndTokenId,
    );

    expect(
      sampleSarashina2SemanticTokenizerId(
        logits: logits,
        generatedSemanticTokens: const [],
        tokenMap: tokenMap,
        eosId: sarashina2EosTokenId,
        temperature: 0,
        topP: 1,
        frequencyPenalty: 0,
        randomDraw: 0,
      ),
      11,
    );
    expect(
      sampleSarashina2SemanticTokenizerId(
        logits: logits,
        generatedSemanticTokens: const [1, 1],
        tokenMap: tokenMap,
        eosId: sarashina2EosTokenId,
        temperature: 0,
        topP: 1,
        frequencyPenalty: 2,
        randomDraw: 0,
      ),
      12,
    );
    final samplerState = Sarashina2SemanticSamplerState(tokenMap: tokenMap);
    try {
      expect(
        sampleSarashina2SemanticTokenizerIdWithState(
          logits: logits,
          samplerState: samplerState,
          eosId: sarashina2EosTokenId,
          temperature: 0,
          topP: 1,
          frequencyPenalty: 2,
          randomDraw: 0,
        ),
        11,
      );
      samplerState.recordSemanticId(1);
      samplerState.recordSemanticId(1);
      expect(
        sampleSarashina2SemanticTokenizerIdWithState(
          logits: logits,
          samplerState: samplerState,
          eosId: sarashina2EosTokenId,
          temperature: 0,
          topP: 1,
          frequencyPenalty: 2,
          randomDraw: 0,
        ),
        12,
      );
    } finally {
      samplerState.close();
    }
  });

  test('Sarashina2 semantic sampler accepts native-backed logits', () {
    final logits = NativeTensorBuffer.float32([20]);
    try {
      final values = logits.asFloat32List();
      values[sarashina2EosTokenId] = 0;
      values[10] = 1;
      values[11] = 4;
      values[12] = 3;

      expect(
        sampleSarashina2SemanticTokenizerId(
          logits: logits.tensor,
          generatedSemanticTokens: const [1, 1],
          tokenMap: const Sarashina2TokenMap(
            semanticTokenBaseId: 10,
            semanticVocabSize: 3,
            speechStartTokenId: sarashina2SpeechStartTokenId,
            speechEndTokenId: sarashina2SpeechEndTokenId,
          ),
          eosId: sarashina2EosTokenId,
          temperature: 0,
          topP: 1,
          frequencyPenalty: 2,
          randomDraw: 0,
        ),
        12,
      );
    } finally {
      logits.close();
    }
  });

  test('Sarashina2 semantic sampler state appends generated ids in native', () {
    final tokenMap = const Sarashina2TokenMap(
      semanticTokenBaseId: 10,
      semanticVocabSize: 3,
      speechStartTokenId: sarashina2SpeechStartTokenId,
      speechEndTokenId: sarashina2SpeechEndTokenId,
    );
    final generated = NativeTensorBuffer.int32([2]);
    final samplerState = Sarashina2SemanticSamplerState(tokenMap: tokenMap);
    try {
      samplerState.appendSemanticId(
        generated: generated,
        generatedLength: 0,
        semanticId: 1,
      );
      samplerState.appendSemanticId(
        generated: generated,
        generatedLength: 1,
        semanticId: 1,
      );
      expect(generated.asInt32List(), [1, 1]);
      final logits = Float32List(20)
        ..[sarashina2EosTokenId] = 0
        ..[10] = 1
        ..[11] = 4
        ..[12] = 3;
      expect(
        sampleSarashina2SemanticTokenizerIdWithState(
          logits: logits,
          samplerState: samplerState,
          eosId: sarashina2EosTokenId,
          temperature: 0,
          topP: 1,
          frequencyPenalty: 2,
          randomDraw: 0,
        ),
        12,
      );
      expect(
        () => samplerState.appendSemanticId(
          generated: generated,
          generatedLength: 2,
          semanticId: 0,
        ),
        throwsStateError,
      );
    } finally {
      samplerState.close();
      generated.close();
    }
  });

  test('Sarashina2 semantic sampler rejects invalid sampling params', () {
    final logits = Float32List(20);
    logits[sarashina2EosTokenId] = 0;
    logits[10] = 1;
    const tokenMap = Sarashina2TokenMap(
      semanticTokenBaseId: 10,
      semanticVocabSize: 1,
      speechStartTokenId: sarashina2SpeechStartTokenId,
      speechEndTokenId: sarashina2SpeechEndTokenId,
    );

    expect(
      () => sampleSarashina2SemanticTokenizerId(
        logits: logits,
        generatedSemanticTokens: const [],
        tokenMap: tokenMap,
        eosId: sarashina2EosTokenId,
        temperature: double.nan,
        topP: 1,
        frequencyPenalty: 0,
        randomDraw: 0,
      ),
      throwsStateError,
    );
    expect(
      () => sampleSarashina2SemanticTokenizerId(
        logits: logits,
        generatedSemanticTokens: const [],
        tokenMap: tokenMap,
        eosId: sarashina2EosTokenId,
        temperature: 1,
        topP: 1,
        frequencyPenalty: 0,
        randomDraw: 1.1,
      ),
      throwsStateError,
    );
    logits[10] = double.nan;
    expect(
      () => sampleSarashina2SemanticTokenizerId(
        logits: logits,
        generatedSemanticTokens: const [],
        tokenMap: tokenMap,
        eosId: sarashina2EosTokenId,
        temperature: 0,
        topP: 1,
        frequencyPenalty: 0,
        randomDraw: 0,
      ),
      throwsStateError,
    );
  });

  test('Sarashina2 synthesis result exposes direct metadata', () {
    final result = Sarashina2SynthesisResult(
      decode: Sarashina2DecodeResult(
        text: 'hello',
        audio: Float32List.fromList([0, 0.5]),
        audioSampleCount: 2,
        audioWavBytes: Uint8List.fromList([1, 2, 3]),
        sampleRate: 24000,
        semanticTokens: const [1, 2, 3],
        decodedSemanticTokens: const [2, 3],
        semanticTokenCount: 3,
        decodedSemanticTokenCount: 2,
        melFrames: 4,
        promptElapsedMicroseconds: 1000,
        semanticElapsedMicroseconds: 2000,
        decodeElapsedMicroseconds: 3000,
        usedPrompt: true,
      ),
      semanticSource: Sarashina2SemanticSource.promptTokenIds,
      promptTokenGenerationElapsedMicroseconds: 4000,
    );

    expect(result.semanticSourceName, 'prompt_token_ids');
    expect(result.audioWavBytes, [1, 2, 3]);
    expect(result.ttsElapsedMicroseconds, 9000);
    expect(
      result.toJson(outputWav: 'out.wav'),
      containsPair(
        'semanticTokenText',
        '<|semantic_1|><|semantic_2|><|semantic_3|>',
      ),
    );
    expect(result.toJson(), containsPair('watermark', 'not_embedded'));
  });

  test('Sarashina2 direct synthesis request classifies LLM need', () {
    expect(const Sarashina2SynthesisRequest(text: 'hello').needsLlm, isTrue);
    expect(
      const Sarashina2SynthesisRequest(
        text: 'hello',
        promptTokenIds: [1, 2],
      ).needsLlm,
      isTrue,
    );
    expect(
      const Sarashina2SynthesisRequest(
        text: '',
        semanticTokens: [1, 2],
      ).needsLlm,
      isFalse,
    );
    expect(
      const Sarashina2SynthesisRequest(
        text: '',
        semanticTokenText: '<|semantic_1|>',
      ).hasExternalSemanticTokens,
      isTrue,
    );
  });

  test('Sarashina2 decoder runtime reports missing exported assets', () async {
    await expectLater(
      Sarashina2DartRuntime.load(
        paths: const Sarashina2TtsPaths(modelDir: '/missing/sarashina2.2-tts'),
        provider: 'cpu',
        deviceId: 0,
        requireProvider: false,
        numThreads: 1,
      ),
      throwsA(
        isA<StateError>().having(
          (error) => '$error',
          'message',
          contains('Sarashina2 decoder runtime is not loadable'),
        ),
      ),
    );
  });
}

String _hex(String value) => utf8.encode(value).map(_hexByte).join();

String _hexByte(int value) => value.toRadixString(16).padLeft(2, '0');
