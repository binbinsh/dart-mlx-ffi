import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../../runtime/native_ffi.dart' as dz;

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/runtime.dart' show RuntimeTensorDataType;
import 'sarashina2_native.dart';

const sarashina2Provider = 'sarashina2-tts';
const sarashina2ModelId = 'sbintuitions/sarashina2.2-tts';
const sarashina2SemanticVocabSize = 6561;
const sarashina2SpeechStartToken = '<|speech_start|>';
const sarashina2EosTokenId = 2;
const sarashina2SemanticTokenBaseId = 102400;
const sarashina2SpeechStartTokenId = 108961;
const sarashina2SpeechEndTokenId = 108962;
const sarashina2DefaultTemperature = 0.9;
const sarashina2DefaultTopP = 0.95;
const sarashina2DefaultFrequencyPenalty = 1.0;

final class Sarashina2TtsPaths {
  const Sarashina2TtsPaths({required this.modelDir});

  factory Sarashina2TtsPaths.fromUniFrontendRoot(String root) {
    final normalized = root.endsWith('/')
        ? root.substring(0, root.length - 1)
        : root;
    return Sarashina2TtsPaths(
      modelDir:
          '$normalized/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts',
    );
  }

  final String modelDir;

  String get configJson => '$modelDir/config.json';
  String get generationConfigJson => '$modelDir/generation_config.json';
  String get tokenizerJson => '$modelDir/tokenizer.json';
  String get tokenizerSidecar => '$modelDir/tokenizer.sara2tok';
  String get tokenizerModel => '$modelDir/tokenizer.model';
  String get addedTokensJson => '$modelDir/added_tokens.json';
  String get specialTokensMapJson => '$modelDir/special_tokens_map.json';
  String get modelSafetensors => '$modelDir/model.safetensors';
  String get flowCheckpoint => '$modelDir/flow.pt';
  String get hiftCheckpoint => '$modelDir/hift.pt';
  String get campplusCheckpoint => '$modelDir/campplus_cn_common.bin';

  String get speechTokenizerOnnx => '$modelDir/speech_tokenizer_v2.onnx';
  String get campplusOnnx => '$modelDir/campplus.onnx';
  String get llmPrefillOnnx => '$modelDir/llm_prefill.onnx';
  String get llmPrefillLastOnnx => '$modelDir/llm_prefill_last.onnx';
  String get llmDecodeOnnx => '$modelDir/llm_decode.onnx';
  String get llmDecodeHeadOnnx => '$modelDir/llm_decode_head.onnx';
  String get llmDecoderHeadOnnx => '$modelDir/llm_decoder_head.onnx';
  String get llmPrefillFp16Onnx => '$modelDir/llm_prefill.fp16.onnx';
  String get llmPrefillLastFp16Onnx => '$modelDir/llm_prefill_last.fp16.onnx';
  String get llmDecodeFp16Onnx => '$modelDir/llm_decode.fp16.onnx';
  String get llmDecodeHeadFp16Onnx => '$modelDir/llm_decode_head.fp16.onnx';
  String get llmDecoderHeadFp16Onnx => '$modelDir/llm_decoder_head.fp16.onnx';
  String get llmPrefillBf16Onnx => '$modelDir/llm_prefill.bf16.onnx';
  String get llmPrefillLastBf16Onnx => '$modelDir/llm_prefill_last.bf16.onnx';
  String get llmDecodeBf16Onnx => '$modelDir/llm_decode.bf16.onnx';
  String get llmDecodeHeadBf16Onnx => '$modelDir/llm_decode_head.bf16.onnx';
  String get llmDecoderHeadBf16Onnx => '$modelDir/llm_decoder_head.bf16.onnx';
  String get llmEmbeddingsNpz => '$modelDir/llm_embeddings.npz';
  String get flowEncoderOnnx => '$modelDir/flow.encoder.fp32.onnx';
  String get flowDecoderEstimatorOnnx =>
      '$modelDir/flow.decoder.estimator.fp32.onnx';
  String get flowDecoderStepOnnx => '$modelDir/flow.decoder.step.fp32.onnx';
  String get flowDecoderStepTensorRtContextOnnx =>
      '$modelDir/flow.decoder.step.fp32.trt_ctx.onnx';
  String get flowDecoderStepFp16Onnx => '$modelDir/flow.decoder.step.fp16.onnx';
  String get flowDecoderLoopOnnx => '$modelDir/flow.decoder.loop.fp32.onnx';
  String get flowSupportNpz => '$modelDir/flow_support.npz';
  String get hiftOnnx => '$modelDir/hift.onnx';

  List<Sarashina2AssetStatus> inspect() => [
    Sarashina2AssetStatus.required('config_json', configJson),
    Sarashina2AssetStatus.required('tokenizer_json', tokenizerJson),
    Sarashina2AssetStatus.required('model_safetensors', modelSafetensors),
    Sarashina2AssetStatus.required('flow_checkpoint', flowCheckpoint),
    Sarashina2AssetStatus.required('hift_checkpoint', hiftCheckpoint),
    Sarashina2AssetStatus.required('campplus_checkpoint', campplusCheckpoint),
    Sarashina2AssetStatus.optional(
      'generation_config_json',
      generationConfigJson,
    ),
    Sarashina2AssetStatus.optional('tokenizer_model', tokenizerModel),
    Sarashina2AssetStatus.optional('tokenizer_sidecar', tokenizerSidecar),
    Sarashina2AssetStatus.optional('added_tokens_json', addedTokensJson),
    Sarashina2AssetStatus.optional(
      'special_tokens_map_json',
      specialTokensMapJson,
    ),
    Sarashina2AssetStatus.optional('llm_prefill_onnx', llmPrefillOnnx),
    Sarashina2AssetStatus.optional('llm_prefill_last_onnx', llmPrefillLastOnnx),
    Sarashina2AssetStatus.optional('llm_decode_onnx', llmDecodeOnnx),
    Sarashina2AssetStatus.optional('llm_decode_head_onnx', llmDecodeHeadOnnx),
    Sarashina2AssetStatus.optional('llm_decoder_head_onnx', llmDecoderHeadOnnx),
    Sarashina2AssetStatus.optional('llm_embeddings_npz', llmEmbeddingsNpz),
    Sarashina2AssetStatus.optional(
      'speech_tokenizer_onnx',
      speechTokenizerOnnx,
    ),
    Sarashina2AssetStatus.optional('campplus_onnx', campplusOnnx),
    Sarashina2AssetStatus.optional('flow_encoder_onnx', flowEncoderOnnx),
    Sarashina2AssetStatus.optional(
      'flow_decoder_estimator_onnx',
      flowDecoderEstimatorOnnx,
    ),
    Sarashina2AssetStatus.optional(
      'flow_decoder_step_onnx',
      flowDecoderStepOnnx,
    ),
    Sarashina2AssetStatus.optional(
      'flow_decoder_step_tensorrt_context_onnx',
      flowDecoderStepTensorRtContextOnnx,
    ),
    Sarashina2AssetStatus.optional(
      'flow_decoder_step_fp16_onnx',
      flowDecoderStepFp16Onnx,
    ),
    Sarashina2AssetStatus.optional(
      'flow_decoder_loop_onnx',
      flowDecoderLoopOnnx,
    ),
    Sarashina2AssetStatus.optional('flow_support_npz', flowSupportNpz),
    Sarashina2AssetStatus.optional('hift_onnx', hiftOnnx),
  ];
}

final class Sarashina2AssetStatus {
  const Sarashina2AssetStatus({
    required this.name,
    required this.path,
    required this.required,
  });

  const Sarashina2AssetStatus.required(String name, String path)
    : this(name: name, path: path, required: true);

  const Sarashina2AssetStatus.optional(String name, String path)
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

final class Sarashina2TokenMap {
  const Sarashina2TokenMap({
    required this.semanticTokenBaseId,
    required this.semanticVocabSize,
    required this.speechStartTokenId,
    required this.speechEndTokenId,
  });

  factory Sarashina2TokenMap.fromAddedTokensFile(String path) {
    final decoded = jsonDecode(File(path).readAsStringSync());
    if (decoded is! Map) {
      throw FormatException('Sarashina2 added_tokens.json must be an object.');
    }
    final values = decoded.map((key, value) => MapEntry('$key', value));
    final semantic0 = _intEntry(values, '<|semantic_0|>');
    final semanticLast = _intEntry(
      values,
      '<|semantic_${sarashina2SemanticVocabSize - 1}|>',
    );
    final expectedLast = semantic0 + sarashina2SemanticVocabSize - 1;
    if (semanticLast != expectedLast) {
      throw StateError(
        'Sarashina2 semantic token ids must be contiguous: '
        '<|semantic_0|>=$semantic0, '
        '<|semantic_${sarashina2SemanticVocabSize - 1}|>=$semanticLast.',
      );
    }
    return Sarashina2TokenMap(
      semanticTokenBaseId: semantic0,
      semanticVocabSize: sarashina2SemanticVocabSize,
      speechStartTokenId: _intEntry(values, sarashina2SpeechStartToken),
      speechEndTokenId: _intEntry(values, '<|speech_end|>'),
    );
  }

  factory Sarashina2TokenMap.fromPaths(Sarashina2TtsPaths paths) {
    return Sarashina2TokenMap.fromAddedTokensFile(paths.addedTokensJson);
  }

  static const defaults = Sarashina2TokenMap(
    semanticTokenBaseId: sarashina2SemanticTokenBaseId,
    semanticVocabSize: sarashina2SemanticVocabSize,
    speechStartTokenId: sarashina2SpeechStartTokenId,
    speechEndTokenId: sarashina2SpeechEndTokenId,
  );

  final int semanticTokenBaseId;
  final int semanticVocabSize;
  final int speechStartTokenId;
  final int speechEndTokenId;

  int tokenizerIdForSemantic(int semanticId) {
    if (semanticId < 0 || semanticId >= semanticVocabSize) {
      throw RangeError.range(
        semanticId,
        0,
        semanticVocabSize - 1,
        'semanticId',
      );
    }
    return semanticTokenBaseId + semanticId;
  }

  int? semanticIdForTokenizerId(int tokenizerId) {
    final semanticId = tokenizerId - semanticTokenBaseId;
    if (semanticId < 0 || semanticId >= semanticVocabSize) {
      return null;
    }
    return semanticId;
  }

  bool isSemanticTokenizerId(int tokenizerId) {
    return semanticIdForTokenizerId(tokenizerId) != null;
  }
}

final class Sarashina2SemanticSamplerState {
  Sarashina2SemanticSamplerState({this.tokenMap = Sarashina2TokenMap.defaults})
    : _counts = NativeTensorBuffer.int32([tokenMap.semanticVocabSize]) {
    _countsView = _counts.asInt32List();
    _countsView.fillRange(0, _countsView.length, 0);
  }

  final Sarashina2TokenMap tokenMap;
  final NativeTensorBuffer _counts;
  late final Int32List _countsView;

  void recordSemanticId(int semanticId) {
    _checkOpen();
    if (semanticId < 0 || semanticId >= tokenMap.semanticVocabSize) {
      throw RangeError.range(
        semanticId,
        0,
        tokenMap.semanticVocabSize - 1,
        'semanticId',
      );
    }
    if (_countsView[semanticId] < 0x7fffffff) {
      _countsView[semanticId] += 1;
    }
  }

  void appendSemanticId({
    required NativeTensorBuffer generated,
    required int generatedLength,
    required int semanticId,
  }) {
    _checkOpen();
    if (generated.dtype != RuntimeTensorDataType.int32) {
      throw StateError(
        'Expected int32 generated, got ${generated.dtype.name}.',
      );
    }
    final generatedCapacity = generated.byteLength ~/ 4;
    final countsLength = _counts.byteLength ~/ 4;
    if (generatedLength < 0 || generatedLength > generatedCapacity) {
      throw RangeError.range(
        generatedLength,
        0,
        generatedCapacity,
        'generatedLength',
      );
    }
    if (semanticId < 0 || semanticId >= countsLength) {
      throw RangeError.range(semanticId, 0, countsLength - 1, 'semanticId');
    }
    appendSarashinaSemanticIdNative(
      generated: generated,
      generatedLength: generatedLength,
      semanticCounts: _counts,
      semanticId: semanticId,
    );
  }

  int sampleTokenizerId({
    required Object logits,
    required int eosId,
    required double temperature,
    required double topP,
    required double frequencyPenalty,
    required double randomDraw,
  }) {
    _checkOpen();
    return sampleSarashinaSemanticTokenizerIdFromCounts(
      logits: logits,
      semanticCounts: _counts,
      semanticBaseId: tokenMap.semanticTokenBaseId,
      semanticVocabSize: tokenMap.semanticVocabSize,
      eosId: eosId,
      temperature: temperature,
      topP: topP,
      frequencyPenalty: frequencyPenalty,
      randomDraw: randomDraw,
    );
  }

  void close() {
    _counts.close();
  }

  void _checkOpen() {
    if (_counts.isClosed) {
      throw StateError('Sarashina2 semantic sampler state is closed.');
    }
  }
}

final class Sarashina2BaseTokenizer {
  Sarashina2BaseTokenizer._(this._handle, this.tokenMap);

  final SarashinaTokenizerHandle _handle;
  final Sarashina2TokenMap tokenMap;

  factory Sarashina2BaseTokenizer.fromFile(
    String sidecarPath, {
    Sarashina2TokenMap tokenMap = Sarashina2TokenMap.defaults,
  }) {
    return Sarashina2BaseTokenizer._(
      SarashinaTokenizerHandle.fromFile(sidecarPath),
      tokenMap,
    );
  }

  factory Sarashina2BaseTokenizer.fromBytes(
    List<int> sidecarBytes, {
    Sarashina2TokenMap tokenMap = Sarashina2TokenMap.defaults,
  }) {
    return Sarashina2BaseTokenizer._(
      SarashinaTokenizerHandle.fromBytes(sidecarBytes),
      tokenMap,
    );
  }

  Int32List encode(String text) => _handle.encode(text);

  NativeTensorBuffer encodeBuffer(String text, {dz.NativeFfi? ffiRuntime}) {
    return _handle.encodeBuffer(text, ffiRuntime: ffiRuntime);
  }

  List<int> encodePromptTokenIds({
    required String text,
    String promptText = '',
    Object promptTokens = const <int>[],
    bool preprocessText = true,
    dz.NativeFfi? ffiRuntime,
  }) {
    final buffer = encodePromptTokenIdsBuffer(
      text: text,
      promptText: promptText,
      promptTokens: promptTokens,
      preprocessText: preprocessText,
      ffiRuntime: ffiRuntime,
    );
    try {
      return Int32List.fromList(buffer.asInt32List());
    } finally {
      buffer.close();
    }
  }

  NativeTensorBuffer encodePromptTokenIdsBuffer({
    required String text,
    String promptText = '',
    Object promptTokens = const <int>[],
    bool preprocessText = true,
    dz.NativeFfi? ffiRuntime,
  }) {
    final promptTokenCount = _int32SourceLength(promptTokens);
    if ((promptText.isEmpty) != (promptTokenCount == 0)) {
      throw ArgumentError(
        'promptText and promptTokens must either both be empty or both be set.',
      );
    }
    final baseText =
        '$promptText${preprocessText ? preprocessSarashinaText(text) : text}';
    return _handle.encodePromptTokenIdsBuffer(
      text: baseText,
      promptTokens: promptTokens,
      speechStartTokenId: tokenMap.speechStartTokenId,
      semanticBaseId: tokenMap.semanticTokenBaseId,
      semanticVocabSize: tokenMap.semanticVocabSize,
      ffiRuntime: ffiRuntime,
    );
  }

  void close() {
    _handle.close();
  }
}

int _int32SourceLength(Object source) {
  if (source is NativeTensorBuffer) {
    if (source.dtype != RuntimeTensorDataType.int32) {
      throw StateError(
        'Expected int32 token buffer, got ${source.dtype.name}.',
      );
    }
    return source.byteLength ~/ 4;
  }
  if (source is Int32List) {
    return source.length;
  }
  if (source is List<int>) {
    return source.length;
  }
  throw ArgumentError.value(
    source,
    'source',
    'expected NativeTensorBuffer/Int32List/List<int>',
  );
}

String preprocessSarashinaText(String text) {
  var value = text;
  value = value.replaceAllMapped(
    RegExp(r'\*{3}(.+?)\*{3}|_{3}(.+?)_{3}'),
    (match) => match.group(1) ?? match.group(2) ?? '',
  );
  value = value.replaceAllMapped(
    RegExp(r'\*{2}(.+?)\*{2}|_{2}(.+?)_{2}'),
    (match) => match.group(1) ?? match.group(2) ?? '',
  );
  value = value.replaceAllMapped(
    RegExp(r'(?<!\w)\*(.+?)\*(?!\w)|(?<!\w)_(.+?)_(?!\w)'),
    (match) => match.group(1) ?? match.group(2) ?? '',
  );
  value = value.replaceAllMapped(
    RegExp(r'~~(.+?)~~|`([^`]+)`'),
    (match) => match.group(1) ?? match.group(2) ?? '',
  );
  value = value.replaceAll(
    RegExp(r'^[\s]*([-*_])\1{2,}[\s]*$', multiLine: true),
    '',
  );
  value = value.replaceAll(RegExp(r'^#{1,6}\s+', multiLine: true), '');
  value = value.replaceAll(RegExp(r'^>\s?', multiLine: true), '');
  value = value.replaceAll(RegExp(r'^[\s]*[-*+]\s+', multiLine: true), '');
  value = value.replaceAll('（', '「').replaceAll('）', '」');
  value = value.replaceAll('(', '\u201c').replaceAll(')', '\u201d');
  return value;
}

int _intEntry(Map<String, Object?> values, String key) {
  final value = values[key];
  if (value is int) {
    return value;
  }
  throw FormatException('Sarashina2 added token "$key" is missing or invalid.');
}

String buildSarashina2Prompt({
  required String text,
  String promptText = '',
  Object promptTokens = const <int>[],
  bool preprocessText = true,
}) {
  return buildSarashinaPromptNative(
    text: preprocessText ? preprocessSarashinaText(text) : text,
    promptText: promptText,
    promptTokens: promptTokens,
  );
}

Int32List parseSarashina2SemanticTokens(String generatedText) {
  return parseSarashinaSemanticTokens(generatedText);
}

NativeTensorBuffer parseSarashina2SemanticTokensBuffer(String generatedText) {
  return parseSarashinaSemanticTokensBuffer(generatedText);
}

String sarashina2SemanticTokensToText(List<int> tokens) {
  return formatSarashinaSemanticTokens(tokens);
}

void validateSarashina2SemanticTokens(List<int> tokens) {
  for (final token in tokens) {
    if (token < 0 || token >= sarashina2SemanticVocabSize) {
      throw RangeError.range(
        token,
        0,
        sarashina2SemanticVocabSize - 1,
        'semanticTokens',
      );
    }
  }
}

int sampleSarashina2SemanticTokenizerId({
  required Object logits,
  required List<int> generatedSemanticTokens,
  Sarashina2TokenMap tokenMap = Sarashina2TokenMap.defaults,
  int eosId = sarashina2EosTokenId,
  double temperature = sarashina2DefaultTemperature,
  double topP = sarashina2DefaultTopP,
  double frequencyPenalty = sarashina2DefaultFrequencyPenalty,
  double randomDraw = 0,
}) {
  return sampleSarashinaSemanticTokenizerId(
    logits: logits,
    generatedSemanticTokens: generatedSemanticTokens,
    semanticBaseId: tokenMap.semanticTokenBaseId,
    semanticVocabSize: tokenMap.semanticVocabSize,
    eosId: eosId,
    temperature: temperature,
    topP: topP,
    frequencyPenalty: frequencyPenalty,
    randomDraw: randomDraw,
  );
}

int sampleSarashina2SemanticTokenizerIdWithState({
  required Object logits,
  required Sarashina2SemanticSamplerState samplerState,
  int eosId = sarashina2EosTokenId,
  double temperature = sarashina2DefaultTemperature,
  double topP = sarashina2DefaultTopP,
  double frequencyPenalty = sarashina2DefaultFrequencyPenalty,
  double randomDraw = 0,
}) {
  samplerState._checkOpen();
  return samplerState.sampleTokenizerId(
    logits: logits,
    eosId: eosId,
    temperature: temperature,
    topP: topP,
    frequencyPenalty: frequencyPenalty,
    randomDraw: randomDraw,
  );
}
