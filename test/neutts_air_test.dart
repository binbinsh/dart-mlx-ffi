import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/models.dart';
import 'package:dart_inference/runtime.dart';
import 'package:dart_inference/src/models/neutts_air/neutts_air_native.dart'
    show
        appendNeuttsAirDecodeSpeechTokenNative,
        initNeuttsAirDecodeInputIdsNative,
        sampleNeuttsAirSpeechTokenizerIdNative,
        validateNeuttsAirSpeechTokensNative;
import 'package:test/test.dart';

void main() {
  test('NeuTTS Air speech token helpers use the native FFI path', () {
    final parsed = parseNeuttsAirSpeechTokens(
      'noise<|speech_0|><|speech_42|><|speech_65535|>',
    );
    final parsedBuffer = parseNeuttsAirSpeechTokensBuffer(
      'noise<|speech_0|><|speech_42|><|speech_65535|>',
    );

    try {
      expect(parsed, [0, 42, 65535]);
      expect(parsedBuffer.asInt32List(), parsed);
      expect(
        neuttsAirSpeechTokensToText(parsed),
        '<|speech_0|><|speech_42|><|speech_65535|>',
      );
      expect(
        () => validateNeuttsAirSpeechTokensNative(parsedBuffer),
        returnsNormally,
      );
      expect(
        () => neuttsAirSpeechTokensToText([neuttsAirSpeechVocabSize]),
        throwsStateError,
      );
      final invalidBuffer = NativeTensorBuffer.int32([1]);
      try {
        invalidBuffer.asInt32List()[0] = neuttsAirSpeechVocabSize;
        expect(
          () => validateNeuttsAirSpeechTokensNative(invalidBuffer),
          throwsStateError,
        );
      } finally {
        invalidBuffer.close();
      }
    } finally {
      parsedBuffer.close();
    }
  });

  test('NeuTTS Air prompt builder mirrors upstream GGUF prompt layout', () {
    expect(
      buildNeuttsAirPrompt(
        referencePhones: 'R EH F',
        inputPhones: 'T EH S T',
        referenceCodes: const [1, 23],
      ),
      'user: Convert the text to speech:<|TEXT_PROMPT_START|>'
      'R EH F T EH S T'
      '<|TEXT_PROMPT_END|>\n'
      'assistant:<|SPEECH_GENERATION_START|>'
      '<|speech_1|><|speech_23|>',
    );
  });

  test('NeuTTS Air special token ids load from tokenizer_config.json', () {
    final dir = Directory.systemTemp.createTempSync('neutts-config-');
    try {
      final path = '${dir.path}/tokenizer_config.json';
      File(path).writeAsStringSync(
        jsonEncode({
          'added_tokens_decoder': {
            '151665': {'content': neuttsAirTextReplaceToken},
            '151666': {'content': neuttsAirTextPromptStartToken},
            '151667': {'content': neuttsAirTextPromptEndToken},
            '151668': {'content': neuttsAirSpeechReplaceToken},
            '151669': {'content': neuttsAirSpeechGenerationStartToken},
            '151670': {'content': neuttsAirSpeechGenerationEndToken},
            '151671': {'content': '<|speech_0|>'},
            '217206': {'content': '<|speech_${neuttsAirSpeechVocabSize - 1}|>'},
          },
        }),
      );

      final ids = NeuttsAirSpecialTokenIds.fromTokenizerConfigFile(path);

      expect(ids.textReplaceId, 151665);
      expect(ids.speechTokenBaseId, neuttsAirSpeechTokenBaseId);
      expect(ids.tokenizerIdForSpeechCode(42), 151713);
      expect(ids.speechCodeForTokenizerId(151713), 42);
      expect(ids.speechCodeForTokenizerId(ids.speechGenerationStartId), isNull);
      expect(
        ids.tokenizerSpecials.map((token) => token.text),
        contains(neuttsAirSpeechGenerationStartToken),
      );
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('NeuTTS Air prompt token id builder replaces sentinels natively', () {
    final ids = buildNeuttsAirPromptTokenIdsFromParts(
      chatIds: const [7, 151665, 8, 151668, 9],
      textPromptIds: const [21, 22],
      referenceCodes: const [1, 23],
    );
    final buffer = buildNeuttsAirPromptTokenIdsFromPartsBuffer(
      chatIds: const [7, 151665, 8, 151668, 9],
      textPromptIds: const [21, 22],
      referenceCodes: const [1, 23],
    );

    try {
      expect(ids, [7, 151666, 21, 22, 151667, 8, 151669, 151672, 151694, 9]);
      expect(buffer.asInt32List(), ids);
    } finally {
      buffer.close();
    }
  });

  test('NeuTTS Air prompt tokenizer stays inside the Dart tokenizer path', () {
    final tokenIds = NeuttsAirSpecialTokenIds.defaults;
    final tokenizer = Qwen2BpeTokenizer.fromSidecarBytes(
      utf8.encode(_qwen2Sidecar(tokenIds.tokenizerSpecials)),
    );
    try {
      expect(tokenizer.vocabSize, 217207);
      final expected = buildNeuttsAirPromptTokenIdsFromParts(
        chatIds: tokenizer.encode(neuttsAirChatTemplate),
        textPromptIds: tokenizer.encode('R EH F T EH S T'),
        referenceCodes: const [1, 23],
      );

      final actual = buildNeuttsAirPromptTokenIds(
        tokenizer: tokenizer,
        referencePhones: 'R EH F',
        inputPhones: 'T EH S T',
        referenceCodes: const [1, 23],
        tokenIds: tokenIds,
      );
      final actualBuffer = buildNeuttsAirPromptTokenIdsBuffer(
        tokenizer: tokenizer,
        referencePhones: 'R EH F',
        inputPhones: 'T EH S T',
        referenceCodes: const [1, 23],
        tokenIds: tokenIds,
      );
      try {
        expect(actual, expected);
        expect(actualBuffer.asInt32List(), expected);
      } finally {
        actualBuffer.close();
      }
    } finally {
      tokenizer.close();
    }
  });

  test('NeuTTS Air speech sampler filters logits in native', () {
    final logits = Float32List.fromList([100, 90, 50, 7, 8, 9, 10, 200]);

    expect(
      sampleNeuttsAirSpeechTokenizerIdNative(
        logits: logits,
        logitsOffset: 0,
        vocabSize: logits.length,
        speechBaseId: 3,
        speechVocabSize: 4,
        eosId: 2,
        allowEos: false,
        temperature: 0,
        topP: 1,
        topK: 2,
        randomDraw: 0.5,
      ),
      6,
    );
    expect(
      sampleNeuttsAirSpeechTokenizerIdNative(
        logits: logits,
        logitsOffset: 0,
        vocabSize: logits.length,
        speechBaseId: 3,
        speechVocabSize: 4,
        eosId: 2,
        allowEos: true,
        temperature: 0,
        topP: 1,
        topK: 2,
        randomDraw: 0.5,
      ),
      2,
    );
    expect(
      () => sampleNeuttsAirSpeechTokenizerIdNative(
        logits: logits,
        logitsOffset: 0,
        vocabSize: logits.length,
        speechBaseId: 6,
        speechVocabSize: 4,
        eosId: 2,
        allowEos: true,
        temperature: 1,
        topP: 1,
        topK: 2,
        randomDraw: 0.5,
      ),
      throwsStateError,
    );
  });

  test('NeuTTS Air decode token buffers append in native', () {
    final inputIds = NativeTensorBuffer.int64([3]);
    final generatedCodes = NativeTensorBuffer.int32([1]);
    try {
      final promptCount = initNeuttsAirDecodeInputIdsNative(
        promptTokenIds: Int32List.fromList([151665, 151669]),
        inputIds: inputIds,
      );
      expect(promptCount, 2);
      expect(inputIds.asInt64List().sublist(0, 2), [151665, 151669]);

      appendNeuttsAirDecodeSpeechTokenNative(
        inputIds: inputIds,
        inputLength: promptCount,
        generatedCodes: generatedCodes,
        generatedLength: 0,
        tokenizerId: neuttsAirSpeechTokenBaseId + 42,
        speechBaseId: neuttsAirSpeechTokenBaseId,
        speechVocabSize: neuttsAirSpeechVocabSize,
      );

      expect(inputIds.asInt64List()[2], neuttsAirSpeechTokenBaseId + 42);
      expect(generatedCodes.asInt32List()[0], 42);
      expect(
        () => appendNeuttsAirDecodeSpeechTokenNative(
          inputIds: inputIds,
          inputLength: promptCount,
          generatedCodes: generatedCodes,
          generatedLength: 0,
          tokenizerId: 42,
          speechBaseId: neuttsAirSpeechTokenBaseId,
          speechVocabSize: neuttsAirSpeechVocabSize,
        ),
        throwsStateError,
      );
    } finally {
      generatedCodes.close();
      inputIds.close();
    }
  });

  test(
    'Qwen2BpeTokenizer.load prefers the native sidecar when present',
    () async {
      final dir = Directory.systemTemp.createTempSync('qwen2-sidecar-load-');
      try {
        await File(
          '${dir.path}/tokenizer.qwen2bpe',
        ).writeAsString(_qwen2Sidecar(kCosyVoice2DefaultSpecials));

        final tokenizer = await Qwen2BpeTokenizer.load(dir.path);
        try {
          expect(tokenizer.vocabSize, 217207);
          expect(tokenizer.encode('!'), [33]);
          expect(tokenizer.encode('<|im_start|>'), [151644]);
          final ids = tokenizer.encodeInt32Buffer('<|im_start|>!');
          try {
            expect(ids.shape, [2]);
            expect(ids.asInt32List(), [151644, 33]);
          } finally {
            ids.close();
          }
          expect(
            () => tokenizer.encodeInt32Buffer('!', maxLength: 0),
            throwsStateError,
          );
        } finally {
          tokenizer.close();
        }
      } finally {
        dir.deleteSync(recursive: true);
      }
    },
  );

  test(
    'Qwen2BpeTokenizer loads HuggingFace tokenizer.json BPE specials',
    () async {
      final dir = Directory.systemTemp.createTempSync('neutts-tokenizer-');
      try {
        final path = '${dir.path}/tokenizer.json';
        File(path).writeAsStringSync(
          jsonEncode({
            'model': {
              'type': 'BPE',
              'vocab': {},
              'merges': [
                ['a', 'b'],
              ],
            },
            'added_tokens': [
              {'id': 151665, 'content': neuttsAirTextReplaceToken},
              {'id': 217206, 'content': '<|speech_65535|>'},
            ],
          }),
        );

        final tokenizer = await Qwen2BpeTokenizer.loadFromTokenizerJson(
          path,
          specials: const [
            Qwen2SpecialToken(neuttsAirTextReplaceToken, 151665),
          ],
        );
        try {
          expect(tokenizer.vocabSize, 217207);
          expect(tokenizer.encode(neuttsAirTextReplaceToken), [151665]);
        } finally {
          tokenizer.close();
        }
      } finally {
        dir.deleteSync(recursive: true);
      }
    },
  );

  test('NeuTTS Air request codec-token parser validates native text', () {
    expect(
      codecTokensFromNeuttsAirRequest(
        codecTokens: const [],
        codecTokenText: '<|speech_1|><|speech_23|>',
      ),
      [1, 23],
    );
    expect(
      () => codecTokensFromNeuttsAirRequest(
        codecTokens: const [neuttsAirSpeechVocabSize],
        codecTokenText: '',
      ),
      throwsRangeError,
    );
    final source = codecTokenSourceFromNeuttsAirRequest(
      codecTokens: const [],
      codecTokenText: '<|speech_1|><|speech_23|>',
    );
    try {
      expect(source.tokens, [1, 23]);
      expect(source.tokenCount, 2);
      expect(source.source, isA<NativeTensorBuffer>());
    } finally {
      source.close();
    }
    final listSource = codecTokenSourceFromNeuttsAirRequest(
      codecTokens: const [1, 23],
      codecTokenText: '',
    );
    try {
      expect(listSource.tokens, [1, 23]);
      expect(listSource.tokenCount, 2);
      expect(listSource.source, isA<NativeTensorBuffer>());
    } finally {
      listSource.close();
    }
    final metadataFree = codecTokenSourceFromNeuttsAirRequest(
      codecTokens: const [],
      codecTokenText: '<|speech_1|><|speech_23|>',
      includeTokenMetadata: false,
    );
    try {
      expect(metadataFree.tokens, isEmpty);
      expect(metadataFree.tokenCount, 2);
      expect(metadataFree.isEmpty, isFalse);
      expect(metadataFree.source, isA<NativeTensorBuffer>());
    } finally {
      metadataFree.close();
    }
  });

  test('NeuTTS Air runtime reports missing exported decoder assets', () async {
    final dir = Directory.systemTemp.createTempSync('neutts-runtime-');
    try {
      final paths = NeuttsAirPaths(providerDir: dir.path);

      await expectLater(
        NeuttsAirDartRuntime.load(
          paths: paths,
          provider: 'cpu',
          deviceId: 0,
          requireProvider: false,
          numThreads: 1,
        ),
        throwsA(
          isA<StateError>().having(
            (error) => '$error',
            'message',
            allOf(
              contains('NeuTTS Air runtime is not loadable'),
              contains('tokenizer.json is missing'),
              contains('neucodec_decoder ONNX is missing'),
            ),
          ),
        ),
      );
    } finally {
      dir.deleteSync(recursive: true);
    }
  });
}

String _qwen2Sidecar(List<Qwen2SpecialToken> specials) {
  final lines = <String>['qwen2bpe\t1', 'meta\tdeclared_vocab_size\t217207'];
  for (var code = 33; code <= 126; code += 1) {
    lines.add('v\t$code\t${_hex(String.fromCharCode(code))}');
  }
  for (var code = 161; code <= 172; code += 1) {
    lines.add('v\t$code\t${_hex(String.fromCharCode(code))}');
  }
  for (var code = 174; code <= 255; code += 1) {
    lines.add('v\t$code\t${_hex(String.fromCharCode(code))}');
  }
  lines
    ..add('v\t300\t${_hex('Ġ')}')
    ..add('v\t310\t${_hex('Ċ')}');
  for (final special in specials) {
    lines.add('s\t${special.id}\t${_hex(special.text)}');
  }
  return '${lines.join('\n')}\n';
}

String _hex(String value) {
  return utf8.encode(value).map((byte) {
    return byte.toRadixString(16).padLeft(2, '0');
  }).join();
}
