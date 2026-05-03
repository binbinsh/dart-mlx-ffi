import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';
import 'package:dart_inference/src/models/qwen3_asr/mel_cpu.dart';

void main() {
  test('CPU mel frontend produces Whisper-shaped finite features', () {
    final frontend = Qwen3AsrCpuMelFrontend();
    final audio = Float32List(16000);
    for (var i = 0; i < audio.length; i++) {
      audio[i] = math.sin(2 * math.pi * 440 * i / 16000).toDouble();
    }

    final mel = frontend.compute(audio);

    expect(mel.shape, [1, 100, 128]);
    expect(mel.data.length, 100 * 128);
    expect(mel.data.every((value) => value.isFinite), isTrue);
  });

  test('config parser accepts Core ML export metadata', () {
    final temp = Directory.systemTemp.createTempSync('qwen3_asr_config_test_');
    addTearDown(() => temp.deleteSync(recursive: true));
    File('${temp.path}/config.json').writeAsStringSync(
      jsonEncode({
        'model_type': 'qwen3-asr-decoder-coreml',
        'hidden_size': 2048,
        'intermediate_size': 6144,
        'num_layers': 28,
        'num_heads': 16,
        'num_kv_heads': 8,
        'head_dim': 128,
        'vocab_size': 151936,
        'quantization': 'int8_palettize',
      }),
    );

    final config = Qwen3AsrConfig.fromSnapshot(temp.path);

    expect(config.textHiddenSize, 2048);
    expect(config.textNumLayers, 28);
    expect(config.textNumHeads, 16);
    expect(config.textNumKvHeads, 8);
    expect(config.quantMode, 'int8_palettize');
  });

  test(
    'native runner orchestrates encoder, decoder init, and decoder step',
    () {
      final temp = Directory.systemTemp.createTempSync(
        'qwen3_asr_native_test_',
      );
      addTearDown(() => temp.deleteSync(recursive: true));
      File('${temp.path}/vocab.json').writeAsStringSync(jsonEncode({'A': 42}));
      File('${temp.path}/merges.txt').writeAsStringSync('#version: 0.2\n');
      final tokenizer = Qwen3AsrBpeTokenizer.load(temp.path);
      final config = _testConfig();
      final embeddings = Float32List(
        config.textVocabSize * config.textHiddenSize,
      );
      for (var i = 0; i < config.textHiddenSize; i++) {
        embeddings[(42 * config.textHiddenSize) + i] = 10 + i.toDouble();
      }

      late List<int> promptIds;
      const encoderInput = 'serving_default_input_features:0';
      const encoderOutput = 'StatefulPartitionedCall/audio_features:0';
      final encoder = _FakeSession(
        diagnostics: const {
          'input_names': [encoderInput],
          'output_names': [encoderOutput],
        },
        onRun: (inputs) {
          final mel = inputs.values[encoderInput] as RuntimeTensor;
          expect(mel.shape, [1, 128, 12]);
          return ModelOutputs({
            encoderOutput: RuntimeTensor.float32([
              1,
              2,
              config.textHiddenSize,
            ], Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8])),
          });
        },
      );
      const initInputIds = 'serving_default_input_ids:0';
      const initPositionIds = 'serving_default_position_ids:0';
      const initAudioFeatures = 'serving_default_audio_features:0';
      const initAudioOffset = 'serving_default_audio_offset:0';
      const initLogits = 'StatefulPartitionedCall/logits:0';
      const initKeys = 'StatefulPartitionedCall/present_keys:0';
      const initValues = 'StatefulPartitionedCall/present_values:0';
      final decoderInit = _FakeSession(
        diagnostics: const {
          'input_names': [
            initInputIds,
            initPositionIds,
            initAudioFeatures,
            initAudioOffset,
          ],
          'output_names': [initLogits, initKeys, initValues],
        },
        onRun: (inputs) {
          promptIds = (inputs.values[initInputIds] as RuntimeTensor)
              .asInt64List()
              .toList();
          final offset = (inputs.values[initAudioOffset] as RuntimeTensor)
              .asInt64List()
              .single;
          expect(
            promptIds.where((id) => id == config.audioPadTokenId),
            hasLength(2),
          );
          expect(promptIds[offset], config.audioPadTokenId);
          final logits = Float32List(promptIds.length * config.textVocabSize);
          logits[((promptIds.length - 1) * config.textVocabSize) + 42] = 99;
          return ModelOutputs({
            initLogits: RuntimeTensor.float32([
              1,
              promptIds.length,
              config.textVocabSize,
            ], logits),
            initKeys: RuntimeTensor.float32([1, 1, 1, 1, 1], Float32List(1)),
            initValues: RuntimeTensor.float32([1, 1, 1, 1, 1], Float32List(1)),
          });
        },
      );
      const stepInputEmbeds = 'serving_default_input_embeds:0';
      const stepPositionIds = 'serving_default_position_ids:0';
      const stepPastKeys = 'serving_default_past_keys:0';
      const stepPastValues = 'serving_default_past_values:0';
      const stepLogits = 'StatefulPartitionedCall/logits:0';
      const stepKeys = 'StatefulPartitionedCall/present_keys:0';
      const stepValues = 'StatefulPartitionedCall/present_values:0';
      final decoderStep = _FakeSession(
        diagnostics: const {
          'input_names': [
            stepInputEmbeds,
            stepPositionIds,
            stepPastKeys,
            stepPastValues,
          ],
          'output_names': [stepLogits, stepKeys, stepValues],
        },
        onRun: (inputs) {
          final embed = (inputs.values[stepInputEmbeds] as RuntimeTensor)
              .asFloat32List();
          expect(embed, [10, 11, 12, 13]);
          final position = (inputs.values[stepPositionIds] as RuntimeTensor)
              .asInt64List()
              .single;
          expect(position, promptIds.length);
          final logits = Float32List(config.textVocabSize);
          logits[2] = 99;
          return ModelOutputs({
            stepLogits: RuntimeTensor.float32([
              1,
              1,
              config.textVocabSize,
            ], logits),
            stepKeys: RuntimeTensor.float32([1, 1, 1, 2, 1], Float32List(2)),
            stepValues: RuntimeTensor.float32([1, 1, 1, 2, 1], Float32List(2)),
          });
        },
      );

      final runner = Qwen3AsrNativeRunner(
        config: config,
        tokenizer: tokenizer,
        encoder: encoder,
        decoderInit: decoderInit,
        decoderStep: decoderStep,
        embeddings: Qwen3AsrEmbeddingTable.fromFloat32Rows(
          vocabSize: config.textVocabSize,
          hiddenSize: config.textHiddenSize,
          values: embeddings,
        ),
        encoderMelFrames: 12,
      );
      addTearDown(runner.close);

      expect(runner.transcribe(Float32List(1600), maxNewTokens: 4), 'A');
      final diagnostics = runner.componentDiagnostics();
      expect(diagnostics['model_level_runner'], 'Qwen3AsrNativeRunner');
      expect(diagnostics['encoder'], isA<Map<String, Object?>>());
      expect(diagnostics['decoder_init'], isA<Map<String, Object?>>());
      expect(diagnostics['decoder_step'], isA<Map<String, Object?>>());
    },
  );

  test('Core ML runner feeds stateful decoder one token at a time', () {
    final temp = Directory.systemTemp.createTempSync('qwen3_asr_coreml_test_');
    addTearDown(() => temp.deleteSync(recursive: true));
    File('${temp.path}/vocab.json').writeAsStringSync(jsonEncode({'A': 42}));
    File('${temp.path}/merges.txt').writeAsStringSync('#version: 0.2\n');
    final tokenizer = Qwen3AsrBpeTokenizer.load(temp.path);
    final config = _testConfig();

    final encoder = _FakeSession(
      diagnostics: const {
        'input_names': ['mel'],
        'output_names': ['audio_embeddings'],
      },
      onRun: (inputs) {
        final mel = inputs.values['mel'] as RuntimeTensor;
        expect(mel.shape[0], 1);
        expect(mel.shape[1], 128);
        return ModelOutputs({
          'audio_embeddings': RuntimeTensor.float32([
            1,
            2,
            config.textHiddenSize,
          ], Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8])),
        });
      },
    );
    final embedding = _FakeSession(
      diagnostics: const {
        'input_names': ['token_id'],
        'output_names': ['embedding'],
      },
      onRun: (inputs) {
        final tokenId = (inputs.values['token_id'] as RuntimeTensor)
            .asInt32List()
            .single;
        final values = tokenId == 42
            ? Float32List.fromList([10, 11, 12, 13])
            : Float32List(config.textHiddenSize);
        return ModelOutputs({
          'embedding': RuntimeTensor.float32([
            1,
            1,
            config.textHiddenSize,
          ], values),
        });
      },
    );
    var audioRowsSeen = 0;
    ModelSession decoderFactory() => _FakeSession(
      diagnostics: const {
        'input_names': ['input_embeds', 'position', 'attention_mask'],
        'output_names': ['logits'],
      },
      onRun: (inputs) {
        final embed = (inputs.values['input_embeds'] as RuntimeTensor)
            .asFloat32List();
        if (embed[0] == 1 || embed[0] == 5) {
          audioRowsSeen += 1;
        }
        final position = (inputs.values['position'] as RuntimeTensor)
            .asInt32List()
            .single;
        final mask = inputs.values['attention_mask'] as RuntimeTensor;
        expect(mask.shape, [1, 1, 1, 1024]);
        expect(position, greaterThanOrEqualTo(0));
        final logits = Float32List(config.textVocabSize);
        logits[embed[0] == 10 ? 2 : 42] = 99;
        return ModelOutputs({
          'logits': RuntimeTensor.float32([1, 1, config.textVocabSize], logits),
        });
      },
    );

    final runner = Qwen3AsrCoreMlRunner(
      config: config,
      tokenizer: tokenizer,
      encoder: encoder,
      embedding: embedding,
      decoderFactory: decoderFactory,
    );
    addTearDown(runner.close);

    expect(runner.transcribe(Float32List(1600), maxNewTokens: 4), 'A');
    expect(audioRowsSeen, 2);
  });

  test('LiteRT bundle factory loads component sessions', () {
    final temp = Directory.systemTemp.createTempSync('qwen3_asr_litert_test_');
    addTearDown(() => temp.deleteSync(recursive: true));
    _writeLiteRtBundleFixture(temp.path);
    final loaded = <String>[];
    final runtime = _FakeRuntime(
      RuntimeEngine.litert,
      onLoad: (bundle, _) {
        loaded.add(bundle.artifactPath.split('/').last);
        return _FakeSession(
          diagnostics: const {},
          onRun: (_) => const ModelOutputs({}),
        );
      },
    );

    final runner = Qwen3AsrNativeRunner.loadLiteRtBundle(
      temp.path,
      runtime: runtime,
    );
    addTearDown(runner.close);

    expect(loaded, [
      'encoder.tflite',
      'decoder_init.tflite',
      'decoder_step.tflite',
    ]);
  });
}

Qwen3AsrConfig _testConfig() {
  return Qwen3AsrConfig(
    audioEncoderDModel: 4,
    audioEncoderLayers: 1,
    audioEncoderHeads: 1,
    audioEncoderFfnDim: 4,
    audioEncoderMelBins: 128,
    audioDownsampleHidden: 4,
    audioOutputDim: 4,
    audioNWindow: 50,
    audioNWindowInfer: 800,
    audioMaxSourcePositions: 1500,
    textHiddenSize: 4,
    textIntermediateSize: 8,
    textNumLayers: 1,
    textNumHeads: 1,
    textNumKvHeads: 1,
    textHeadDim: 4,
    textVocabSize: 64,
    textRmsNormEps: 1e-6,
    textRopeTheta: 1e6,
    textMropeSections: const [24, 20, 20],
    tieWordEmbeddings: true,
    quantGroupSize: 64,
    quantBits: 8,
    quantMode: 'affine',
    audioPadTokenId: 151676,
    audioStartTokenId: 151669,
    audioEndTokenId: 151670,
    asrTextTokenId: 151704,
    imStartTokenId: 151644,
    imEndTokenId: 151645,
    newlineTokenId: 198,
    eosTokenIds: const [2],
  );
}

void _writeLiteRtBundleFixture(String path) {
  File('$path/config.json').writeAsStringSync(
    jsonEncode({
      'hidden_size': 4,
      'intermediate_size': 8,
      'num_hidden_layers': 1,
      'num_attention_heads': 1,
      'num_key_value_heads': 1,
      'head_dim': 4,
      'vocab_size': 64,
    }),
  );
  File('$path/vocab.json').writeAsStringSync(jsonEncode({'A': 42}));
  File('$path/merges.txt').writeAsStringSync('#version: 0.2\n');
  for (final name in [
    'encoder.tflite',
    'decoder_init.tflite',
    'decoder_step.tflite',
  ]) {
    File('$path/$name').writeAsBytesSync(const [0]);
  }
  File(
    '$path/embed_tokens.bin',
  ).writeAsBytesSync(Uint8List(64 * 4 * Float32List.bytesPerElement));
}

final class _FakeRuntime implements ModelRuntime {
  const _FakeRuntime(this.engine, {required this.onLoad});

  final RuntimeEngine engine;
  final ModelSession Function(ModelBundle bundle, RuntimeOptions options)
  onLoad;

  @override
  RuntimeCapabilities get capabilities =>
      RuntimeCapabilities(engine: engine, platform: RuntimePlatform.android);

  @override
  ModelSession load(ModelBundle bundle, RuntimeOptions options) {
    expect(bundle.artifact.engine, engine);
    return onLoad(bundle, options);
  }
}

final class _FakeSession implements ModelSession {
  _FakeSession({required this.diagnostics, required this.onRun});

  @override
  final Map<String, Object?> diagnostics;

  final ModelOutputs Function(ModelInputs inputs) onRun;

  @override
  ModelOutputs run(ModelInputs inputs) => onRun(inputs);

  @override
  Stream<ModelOutputs> stream(ModelInputs inputs) async* {
    yield run(inputs);
  }

  @override
  void close() {}
}
