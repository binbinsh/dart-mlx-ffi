import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../../runtime/native_ffi.dart' as dz;

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../cosyvoice2/qwen2_tokenizer.dart';
import 'neutts_air_native.dart';

const neuttsAirProvider = 'neutts-air';
const neuttsAirModelId = 'neuphonic/neutts-air';
const neuttsAirSampleRate = 24000;
const neuttsAirMaxContextTokens = 2048;
const neuttsAirCodecHopLength = 480;
const neuttsAirSpeechVocabSize = 65536;
const neuttsAirSpeechTokenBaseId = 151671;

const neuttsAirTextReplaceToken = '<|TEXT_REPLACE|>';
const neuttsAirTextPromptStartToken = '<|TEXT_PROMPT_START|>';
const neuttsAirTextPromptEndToken = '<|TEXT_PROMPT_END|>';
const neuttsAirSpeechReplaceToken = '<|SPEECH_REPLACE|>';
const neuttsAirSpeechGenerationStartToken = '<|SPEECH_GENERATION_START|>';
const neuttsAirSpeechGenerationEndToken = '<|SPEECH_GENERATION_END|>';

const neuttsAirChatTemplate =
    'user: Convert the text to speech:<|TEXT_REPLACE|>\n'
    'assistant:<|SPEECH_REPLACE|>';

final class NeuttsAirPaths {
  const NeuttsAirPaths({required this.providerDir});

  factory NeuttsAirPaths.fromUniFrontendRoot(String root) {
    final normalized = root.endsWith('/')
        ? root.substring(0, root.length - 1)
        : root;
    return NeuttsAirPaths(
      providerDir: '$normalized/src/ttsbackends/providers/neutts-air',
    );
  }

  final String providerDir;

  String get modelDir => '$providerDir/models';
  String get configJson => '$modelDir/config.json';
  String get generationConfigJson => '$modelDir/generation_config.json';
  String get tokenizerJson => '$modelDir/tokenizer.json';
  String get tokenizerSidecar => '$modelDir/tokenizer.qwen2bpe';
  String get tokenizerConfigJson => '$modelDir/tokenizer_config.json';
  String get vocabJson => '$modelDir/vocab.json';
  String get specialTokensMapJson => '$modelDir/special_tokens_map.json';
  String get modelSafetensors => '$modelDir/model.safetensors';
  String get modelGguf => '$modelDir/neutss-air-BF16.gguf';
  String get neucodecSource =>
      '$modelDir/models--neuphonic--neucodec/snapshots/c92ba97d538f2a0baa9118c21ea5de4cdad4e02a/pytorch_model.bin';
  String get lmOnnx => '$modelDir/onnx/neutts_air_lm.onnx';
  String get neucodecDecoderOnnx => '$modelDir/onnx/neucodec_decoder.onnx';

  List<NeuttsAirAssetStatus> inspect() => [
    NeuttsAirAssetStatus.required('config_json', configJson),
    NeuttsAirAssetStatus.required('tokenizer_json', tokenizerJson),
    NeuttsAirAssetStatus.required('tokenizer_config_json', tokenizerConfigJson),
    NeuttsAirAssetStatus.required('model_safetensors', modelSafetensors),
    NeuttsAirAssetStatus.required('neucodec_source', neucodecSource),
    NeuttsAirAssetStatus.optional(
      'generation_config_json',
      generationConfigJson,
    ),
    NeuttsAirAssetStatus.optional('vocab_json', vocabJson),
    NeuttsAirAssetStatus.optional('tokenizer_sidecar', tokenizerSidecar),
    NeuttsAirAssetStatus.optional(
      'special_tokens_map_json',
      specialTokensMapJson,
    ),
    NeuttsAirAssetStatus.optional('model_gguf', modelGguf),
    NeuttsAirAssetStatus.optional('lm_onnx', lmOnnx),
    NeuttsAirAssetStatus.optional('neucodec_decoder_onnx', neucodecDecoderOnnx),
  ];
}

final class NeuttsAirAssetStatus {
  const NeuttsAirAssetStatus({
    required this.name,
    required this.path,
    required this.required,
  });

  const NeuttsAirAssetStatus.required(String name, String path)
    : this(name: name, path: path, required: true);

  const NeuttsAirAssetStatus.optional(String name, String path)
    : this(name: name, path: path, required: false);

  final String name;
  final String path;
  final bool required;

  bool get exists => File(path).existsSync();

  int? get sizeBytes => exists ? File(path).lengthSync() : null;

  Map<String, Object?> toJson() => {
    'name': name,
    'path': path,
    'required': required,
    'exists': exists,
    if (sizeBytes != null) 'sizeBytes': sizeBytes,
  };
}

final class NeuttsAirSpecialTokenIds {
  const NeuttsAirSpecialTokenIds({
    required this.textReplaceId,
    required this.textPromptStartId,
    required this.textPromptEndId,
    required this.speechReplaceId,
    required this.speechGenerationStartId,
    required this.speechGenerationEndId,
    this.speechTokenBaseId = neuttsAirSpeechTokenBaseId,
    this.speechVocabSize = neuttsAirSpeechVocabSize,
  });

  factory NeuttsAirSpecialTokenIds.fromTokenizerConfigFile(String path) {
    final decoded = jsonDecode(File(path).readAsStringSync());
    if (decoded is! Map) {
      throw FormatException(
        'NeuTTS Air tokenizer_config.json must be an object.',
      );
    }
    final entries = _addedTokenDecoderEntries(decoded['added_tokens_decoder']);
    return NeuttsAirSpecialTokenIds(
      textReplaceId: _idForContent(entries, neuttsAirTextReplaceToken),
      textPromptStartId: _idForContent(entries, neuttsAirTextPromptStartToken),
      textPromptEndId: _idForContent(entries, neuttsAirTextPromptEndToken),
      speechReplaceId: _idForContent(entries, neuttsAirSpeechReplaceToken),
      speechGenerationStartId: _idForContent(
        entries,
        neuttsAirSpeechGenerationStartToken,
      ),
      speechGenerationEndId: _idForContent(
        entries,
        neuttsAirSpeechGenerationEndToken,
      ),
      speechTokenBaseId: _idForContent(entries, '<|speech_0|>'),
      speechVocabSize: _speechVocabSize(entries),
    );
  }

  factory NeuttsAirSpecialTokenIds.fromPaths(NeuttsAirPaths paths) {
    return NeuttsAirSpecialTokenIds.fromTokenizerConfigFile(
      paths.tokenizerConfigJson,
    );
  }

  static const defaults = NeuttsAirSpecialTokenIds(
    textReplaceId: 151665,
    textPromptStartId: 151666,
    textPromptEndId: 151667,
    speechReplaceId: 151668,
    speechGenerationStartId: 151669,
    speechGenerationEndId: 151670,
  );

  final int textReplaceId;
  final int textPromptStartId;
  final int textPromptEndId;
  final int speechReplaceId;
  final int speechGenerationStartId;
  final int speechGenerationEndId;
  final int speechTokenBaseId;
  final int speechVocabSize;

  List<Qwen2SpecialToken> get tokenizerSpecials => [
    Qwen2SpecialToken(neuttsAirTextReplaceToken, textReplaceId),
    Qwen2SpecialToken(neuttsAirTextPromptStartToken, textPromptStartId),
    Qwen2SpecialToken(neuttsAirTextPromptEndToken, textPromptEndId),
    Qwen2SpecialToken(neuttsAirSpeechReplaceToken, speechReplaceId),
    Qwen2SpecialToken(
      neuttsAirSpeechGenerationStartToken,
      speechGenerationStartId,
    ),
    Qwen2SpecialToken(neuttsAirSpeechGenerationEndToken, speechGenerationEndId),
  ];

  int tokenizerIdForSpeechCode(int code) {
    if (code < 0 || code >= speechVocabSize) {
      throw RangeError.range(code, 0, speechVocabSize - 1, 'code');
    }
    return speechTokenBaseId + code;
  }

  int? speechCodeForTokenizerId(int tokenizerId) {
    final code = tokenizerId - speechTokenBaseId;
    if (code < 0 || code >= speechVocabSize) {
      return null;
    }
    return code;
  }
}

Future<Qwen2BpeTokenizer> loadNeuttsAirTokenizer(
  NeuttsAirPaths paths, {
  NeuttsAirSpecialTokenIds? tokenIds,
}) async {
  if (File(paths.tokenizerSidecar).existsSync()) {
    return Qwen2BpeTokenizer.loadFromSidecar(paths.tokenizerSidecar);
  }
  final ids =
      tokenIds ??
      NeuttsAirSpecialTokenIds.fromTokenizerConfigFile(
        paths.tokenizerConfigJson,
      );
  return Qwen2BpeTokenizer.loadFromTokenizerJson(
    paths.tokenizerJson,
    specials: ids.tokenizerSpecials,
  );
}

String buildNeuttsAirPrompt({
  required String referencePhones,
  required String inputPhones,
  List<int> referenceCodes = const [],
}) {
  return buildNeuttsAirPromptNative(
    referencePhones: referencePhones,
    inputPhones: inputPhones,
    referenceCodes: referenceCodes,
  );
}

List<int> buildNeuttsAirPromptTokenIds({
  required Qwen2BpeTokenizer tokenizer,
  required String referencePhones,
  required String inputPhones,
  required List<int> referenceCodes,
  NeuttsAirSpecialTokenIds tokenIds = NeuttsAirSpecialTokenIds.defaults,
}) {
  return buildNeuttsAirPromptTokenIdsWithTokenizerNative(
    tokenizer: tokenizer,
    referencePhones: referencePhones,
    inputPhones: inputPhones,
    referenceCodes: referenceCodes,
    textReplaceId: tokenIds.textReplaceId,
    textPromptStartId: tokenIds.textPromptStartId,
    textPromptEndId: tokenIds.textPromptEndId,
    speechReplaceId: tokenIds.speechReplaceId,
    speechGenerationStartId: tokenIds.speechGenerationStartId,
    speechTokenBaseId: tokenIds.speechTokenBaseId,
    speechVocabSize: tokenIds.speechVocabSize,
  );
}

NativeTensorBuffer buildNeuttsAirPromptTokenIdsBuffer({
  required Qwen2BpeTokenizer tokenizer,
  required String referencePhones,
  required String inputPhones,
  required Object referenceCodes,
  NeuttsAirSpecialTokenIds tokenIds = NeuttsAirSpecialTokenIds.defaults,
  dz.NativeFfi? ffiRuntime,
}) {
  return buildNeuttsAirPromptTokenIdsWithTokenizerBufferNative(
    tokenizer: tokenizer,
    referencePhones: referencePhones,
    inputPhones: inputPhones,
    referenceCodes: referenceCodes,
    textReplaceId: tokenIds.textReplaceId,
    textPromptStartId: tokenIds.textPromptStartId,
    textPromptEndId: tokenIds.textPromptEndId,
    speechReplaceId: tokenIds.speechReplaceId,
    speechGenerationStartId: tokenIds.speechGenerationStartId,
    speechTokenBaseId: tokenIds.speechTokenBaseId,
    speechVocabSize: tokenIds.speechVocabSize,
    ffiRuntime: ffiRuntime,
  );
}

Int32List buildNeuttsAirPromptTokenIdsFromParts({
  required List<int> chatIds,
  required List<int> textPromptIds,
  required List<int> referenceCodes,
  NeuttsAirSpecialTokenIds tokenIds = NeuttsAirSpecialTokenIds.defaults,
}) {
  return buildNeuttsAirPromptTokenIdsNative(
    chatIds: chatIds,
    textPromptIds: textPromptIds,
    referenceCodes: referenceCodes,
    textReplaceId: tokenIds.textReplaceId,
    textPromptStartId: tokenIds.textPromptStartId,
    textPromptEndId: tokenIds.textPromptEndId,
    speechReplaceId: tokenIds.speechReplaceId,
    speechGenerationStartId: tokenIds.speechGenerationStartId,
    speechTokenBaseId: tokenIds.speechTokenBaseId,
    speechVocabSize: tokenIds.speechVocabSize,
  );
}

NativeTensorBuffer buildNeuttsAirPromptTokenIdsFromPartsBuffer({
  required Object chatIds,
  required Object textPromptIds,
  required Object referenceCodes,
  NeuttsAirSpecialTokenIds tokenIds = NeuttsAirSpecialTokenIds.defaults,
  dz.NativeFfi? ffiRuntime,
}) {
  return buildNeuttsAirPromptTokenIdsBufferNative(
    chatIds: chatIds,
    textPromptIds: textPromptIds,
    referenceCodes: referenceCodes,
    textReplaceId: tokenIds.textReplaceId,
    textPromptStartId: tokenIds.textPromptStartId,
    textPromptEndId: tokenIds.textPromptEndId,
    speechReplaceId: tokenIds.speechReplaceId,
    speechGenerationStartId: tokenIds.speechGenerationStartId,
    speechTokenBaseId: tokenIds.speechTokenBaseId,
    speechVocabSize: tokenIds.speechVocabSize,
    ffiRuntime: ffiRuntime,
  );
}

Int32List parseNeuttsAirSpeechTokens(String generatedText) {
  return parseNeuttsAirSpeechTokensNative(generatedText);
}

NativeTensorBuffer parseNeuttsAirSpeechTokensBuffer(
  String generatedText, {
  dz.NativeFfi? ffiRuntime,
}) {
  return parseNeuttsAirSpeechTokensBufferNative(
    generatedText,
    ffiRuntime: ffiRuntime,
  );
}

String neuttsAirSpeechTokensToText(List<int> tokens) {
  return formatNeuttsAirSpeechTokensNative(tokens);
}

Map<String, int> _addedTokenDecoderEntries(Object? raw) {
  if (raw is! Map) {
    throw FormatException(
      'NeuTTS Air tokenizer_config.json added_tokens_decoder must be an object.',
    );
  }
  final entries = <String, int>{};
  for (final item in raw.entries) {
    final id = int.tryParse('${item.key}');
    final value = item.value;
    if (id == null || value is! Map || value['content'] is! String) {
      throw FormatException('Invalid NeuTTS Air added token entry: $item');
    }
    entries[value['content'] as String] = id;
  }
  return entries;
}

int _idForContent(Map<String, int> entries, String token) {
  final id = entries[token];
  if (id == null) {
    throw FormatException('NeuTTS Air special token "$token" is missing.');
  }
  return id;
}

int _speechVocabSize(Map<String, int> entries) {
  final base = _idForContent(entries, '<|speech_0|>');
  final last = _idForContent(
    entries,
    '<|speech_${neuttsAirSpeechVocabSize - 1}|>',
  );
  final expectedLast = base + neuttsAirSpeechVocabSize - 1;
  if (last != expectedLast) {
    throw StateError(
      'NeuTTS Air speech token ids must be contiguous: '
      '<|speech_0|>=$base, '
      '<|speech_${neuttsAirSpeechVocabSize - 1}|>=$last.',
    );
  }
  return neuttsAirSpeechVocabSize;
}
