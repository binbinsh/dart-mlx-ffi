import 'dart:ffi' as ffi;
import 'dart:math' as math;
import 'dart:typed_data';

import '../../runtime/native_ffi.dart' as dz;

import '../../runtime/native_float32_source.dart';
import '../../runtime/native_int32_source.dart';
import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/runtime.dart' show RuntimeTensorDataType;
import '../cosyvoice2/qwen2_tokenizer.dart';

const _speechVocabSize = 65536;
const _speechPrefix = '<|speech_';
const _speechSuffix = '|>';
const _chatTemplate =
    'user: Convert the text to speech:<|TEXT_REPLACE|>\n'
    'assistant:<|SPEECH_REPLACE|>';
const _promptPrefix = 'user: Convert the text to speech:<|TEXT_PROMPT_START|>';
const _promptMiddle =
    '<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>';

Int32List parseNeuttsAirSpeechTokensNative(String text) {
  final tokens = _parseTokenText(
    text,
    prefix: _speechPrefix,
    suffix: _speechSuffix,
    vocabSize: _speechVocabSize,
  );
  return Int32List.fromList(tokens);
}

NativeTensorBuffer parseNeuttsAirSpeechTokensBufferNative(
  String text, {
  dz.NativeFfi? ffiRuntime,
}) {
  final tokens = parseNeuttsAirSpeechTokensNative(text);
  final out = _allocateInt32Buffer(tokens.length, ffiRuntime);
  out.asInt32List().setAll(0, tokens);
  return out;
}

String formatNeuttsAirSpeechTokensNative(List<int> tokens) {
  validateNeuttsAirSpeechTokensNative(tokens);
  return tokens.map((token) => '$_speechPrefix$token$_speechSuffix').join();
}

void validateNeuttsAirSpeechTokensNative(Object tokens) {
  final values = _int32Values(tokens);
  for (final token in values) {
    if (token < 0 || token >= _speechVocabSize) {
      throw StateError(
        'NeuTTS Air speech token id is outside the codec vocabulary.',
      );
    }
  }
}

int sampleNeuttsAirSpeechTokenizerIdNative({
  required Object logits,
  required int logitsOffset,
  required int vocabSize,
  required int speechBaseId,
  required int speechVocabSize,
  required int eosId,
  required bool allowEos,
  required double temperature,
  required double topP,
  required int topK,
  required double randomDraw,
}) {
  if (logitsOffset < 0 ||
      vocabSize <= 0 ||
      speechBaseId < 0 ||
      speechVocabSize <= 0 ||
      eosId < 0 ||
      topK <= 0 ||
      !temperature.isFinite ||
      !topP.isFinite ||
      !randomDraw.isFinite ||
      temperature < 0 ||
      topP <= 0 ||
      randomDraw < 0 ||
      randomDraw > 1) {
    throw StateError('NeuTTS Air native helper received invalid input.');
  }
  final values = _float32Values(logits);
  if (logitsOffset > values.length ||
      vocabSize > values.length - logitsOffset ||
      speechBaseId >= vocabSize ||
      speechBaseId + speechVocabSize > vocabSize ||
      (allowEos && eosId >= vocabSize)) {
    throw StateError('NeuTTS Air native helper received invalid input.');
  }
  final limit = math.min(topK, speechVocabSize + (allowEos ? 1 : 0));
  final candidates = <_Candidate>[];
  void insert(int tokenId) {
    final logit = values[logitsOffset + tokenId];
    if (!logit.isFinite) {
      throw StateError('NeuTTS Air native helper received invalid input.');
    }
    final candidate = _Candidate(tokenId, logit.toDouble());
    if (candidates.length < limit) {
      candidates.add(candidate);
      return;
    }
    var worst = 0;
    for (var i = 1; i < candidates.length; i += 1) {
      if (_candidateWorse(candidates[i], candidates[worst])) {
        worst = i;
      }
    }
    if (_candidateBetter(candidate, candidates[worst])) {
      candidates[worst] = candidate;
    }
  }

  if (allowEos) {
    insert(eosId);
  }
  for (var code = 0; code < speechVocabSize; code += 1) {
    insert(speechBaseId + code);
  }
  return _sampleTopP(candidates, temperature, topP, randomDraw);
}

int initNeuttsAirDecodeInputIdsNative({
  required Object promptTokenIds,
  required NativeTensorBuffer inputIds,
}) {
  _checkBuffer(inputIds, RuntimeTensorDataType.int64, 'inputIds');
  final prompt = _int32Values(promptTokenIds);
  final out = inputIds.asInt64List();
  if (out.length < prompt.length) {
    throw StateError('NeuTTS Air native helper output buffer is too small.');
  }
  for (var i = 0; i < prompt.length; i += 1) {
    if (prompt[i] < 0) {
      throw StateError('NeuTTS Air native helper received invalid input.');
    }
    out[i] = prompt[i];
  }
  return prompt.length;
}

void appendNeuttsAirDecodeSpeechTokenNative({
  required NativeTensorBuffer inputIds,
  required int inputLength,
  required NativeTensorBuffer generatedCodes,
  required int generatedLength,
  required int tokenizerId,
  required int speechBaseId,
  required int speechVocabSize,
}) {
  _checkBuffer(inputIds, RuntimeTensorDataType.int64, 'inputIds');
  _checkBuffer(generatedCodes, RuntimeTensorDataType.int32, 'generatedCodes');
  final input = inputIds.asInt64List();
  final generated = generatedCodes.asInt32List();
  if (inputLength < 0 ||
      inputLength >= input.length ||
      generatedLength < 0 ||
      generatedLength >= generated.length ||
      tokenizerId < 0 ||
      speechBaseId < 0 ||
      speechVocabSize <= 0) {
    throw StateError('NeuTTS Air native helper received invalid input.');
  }
  final code = tokenizerId - speechBaseId;
  if (code < 0 || code >= speechVocabSize) {
    throw StateError(
      'NeuTTS Air speech token id is outside the codec vocabulary.',
    );
  }
  input[inputLength] = tokenizerId;
  generated[generatedLength] = code;
}

String buildNeuttsAirPromptNative({
  required String referencePhones,
  required String inputPhones,
  List<int> referenceCodes = const [],
}) {
  return '$_promptPrefix$referencePhones $inputPhones$_promptMiddle'
      '${formatNeuttsAirSpeechTokensNative(referenceCodes)}';
}

Int32List buildNeuttsAirPromptTokenIdsNative({
  required List<int> chatIds,
  required List<int> textPromptIds,
  required List<int> referenceCodes,
  required int textReplaceId,
  required int textPromptStartId,
  required int textPromptEndId,
  required int speechReplaceId,
  required int speechGenerationStartId,
  required int speechTokenBaseId,
  required int speechVocabSize,
  dz.NativeFfi? ffiRuntime,
}) {
  return Int32List.fromList(
    _buildPromptTokenIds(
      chatIds: chatIds,
      textPromptIds: textPromptIds,
      referenceCodes: referenceCodes,
      textReplaceId: textReplaceId,
      textPromptStartId: textPromptStartId,
      textPromptEndId: textPromptEndId,
      speechReplaceId: speechReplaceId,
      speechGenerationStartId: speechGenerationStartId,
      speechTokenBaseId: speechTokenBaseId,
      speechVocabSize: speechVocabSize,
    ),
  );
}

NativeTensorBuffer buildNeuttsAirPromptTokenIdsBufferNative({
  required Object chatIds,
  required Object textPromptIds,
  required Object referenceCodes,
  required int textReplaceId,
  required int textPromptStartId,
  required int textPromptEndId,
  required int speechReplaceId,
  required int speechGenerationStartId,
  required int speechTokenBaseId,
  required int speechVocabSize,
  dz.NativeFfi? ffiRuntime,
}) {
  final ids = _buildPromptTokenIds(
    chatIds: _int32Values(chatIds),
    textPromptIds: _int32Values(textPromptIds),
    referenceCodes: _int32Values(referenceCodes),
    textReplaceId: textReplaceId,
    textPromptStartId: textPromptStartId,
    textPromptEndId: textPromptEndId,
    speechReplaceId: speechReplaceId,
    speechGenerationStartId: speechGenerationStartId,
    speechTokenBaseId: speechTokenBaseId,
    speechVocabSize: speechVocabSize,
  );
  final out = _allocateInt32Buffer(ids.length, ffiRuntime);
  out.asInt32List().setAll(0, ids);
  return out;
}

Int32List buildNeuttsAirPromptTokenIdsWithTokenizerNative({
  required Qwen2BpeTokenizer tokenizer,
  required String referencePhones,
  required String inputPhones,
  required List<int> referenceCodes,
  required int textReplaceId,
  required int textPromptStartId,
  required int textPromptEndId,
  required int speechReplaceId,
  required int speechGenerationStartId,
  required int speechTokenBaseId,
  required int speechVocabSize,
  dz.NativeFfi? ffiRuntime,
}) {
  final buffer = buildNeuttsAirPromptTokenIdsWithTokenizerBufferNative(
    tokenizer: tokenizer,
    referencePhones: referencePhones,
    inputPhones: inputPhones,
    referenceCodes: referenceCodes,
    textReplaceId: textReplaceId,
    textPromptStartId: textPromptStartId,
    textPromptEndId: textPromptEndId,
    speechReplaceId: speechReplaceId,
    speechGenerationStartId: speechGenerationStartId,
    speechTokenBaseId: speechTokenBaseId,
    speechVocabSize: speechVocabSize,
    ffiRuntime: ffiRuntime,
  );
  try {
    return Int32List.fromList(buffer.asInt32List());
  } finally {
    buffer.close();
  }
}

NativeTensorBuffer buildNeuttsAirPromptTokenIdsWithTokenizerBufferNative({
  required Qwen2BpeTokenizer tokenizer,
  required String referencePhones,
  required String inputPhones,
  required Object referenceCodes,
  required int textReplaceId,
  required int textPromptStartId,
  required int textPromptEndId,
  required int speechReplaceId,
  required int speechGenerationStartId,
  required int speechTokenBaseId,
  required int speechVocabSize,
  dz.NativeFfi? ffiRuntime,
}) {
  return buildNeuttsAirPromptTokenIdsBufferNative(
    chatIds: tokenizer.encode(_chatTemplate),
    textPromptIds: tokenizer.encode('$referencePhones $inputPhones'),
    referenceCodes: referenceCodes,
    textReplaceId: textReplaceId,
    textPromptStartId: textPromptStartId,
    textPromptEndId: textPromptEndId,
    speechReplaceId: speechReplaceId,
    speechGenerationStartId: speechGenerationStartId,
    speechTokenBaseId: speechTokenBaseId,
    speechVocabSize: speechVocabSize,
    ffiRuntime: ffiRuntime,
  );
}

List<int> _buildPromptTokenIds({
  required List<int> chatIds,
  required List<int> textPromptIds,
  required List<int> referenceCodes,
  required int textReplaceId,
  required int textPromptStartId,
  required int textPromptEndId,
  required int speechReplaceId,
  required int speechGenerationStartId,
  required int speechTokenBaseId,
  required int speechVocabSize,
}) {
  if (textReplaceId < 0 ||
      textPromptStartId < 0 ||
      textPromptEndId < 0 ||
      speechReplaceId < 0 ||
      speechGenerationStartId < 0 ||
      speechTokenBaseId < 0 ||
      speechVocabSize <= 0) {
    throw StateError('NeuTTS Air native helper received invalid input.');
  }
  var sawText = false;
  var sawSpeech = false;
  final out = <int>[];
  for (final id in chatIds) {
    if (id == textReplaceId) {
      if (sawText) {
        throw StateError('NeuTTS Air native helper received invalid input.');
      }
      sawText = true;
      out
        ..add(textPromptStartId)
        ..addAll(textPromptIds)
        ..add(textPromptEndId);
    } else if (id == speechReplaceId) {
      if (sawSpeech) {
        throw StateError('NeuTTS Air native helper received invalid input.');
      }
      sawSpeech = true;
      out.add(speechGenerationStartId);
      for (final code in referenceCodes) {
        if (code < 0 || code >= speechVocabSize) {
          throw StateError(
            'NeuTTS Air speech token id is outside the codec vocabulary.',
          );
        }
        out.add(speechTokenBaseId + code);
      }
    } else {
      out.add(id);
    }
  }
  if (!sawText || !sawSpeech) {
    throw StateError('NeuTTS Air native helper received invalid input.');
  }
  return out;
}

List<int> _parseTokenText(
  String text, {
  required String prefix,
  required String suffix,
  required int vocabSize,
}) {
  final tokens = <int>[];
  var cursor = 0;
  while (true) {
    final start = text.indexOf(prefix, cursor);
    if (start < 0) break;
    final numberStart = start + prefix.length;
    final end = text.indexOf(suffix, numberStart);
    if (end < 0) break;
    final id = int.tryParse(text.substring(numberStart, end));
    if (id != null) {
      if (id < 0 || id >= vocabSize) {
        throw StateError(
          'NeuTTS Air speech token id is outside the codec vocabulary.',
        );
      }
      tokens.add(id);
    }
    cursor = end + suffix.length;
  }
  return tokens;
}

List<int> _int32Values(Object source) {
  return withNativeInt32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return const <int>[];
    return List<int>.from(pointer.asTypedList(length), growable: false);
  });
}

Float32List _float32Values(Object source) {
  return withNativeFloat32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return Float32List(0);
    return Float32List.fromList(pointer.asTypedList(length));
  });
}

NativeTensorBuffer _allocateInt32Buffer(int count, dz.NativeFfi? ffiRuntime) {
  return NativeTensorBuffer.int32([count], ffiRuntime: ffiRuntime);
}

void _checkBuffer(
  NativeTensorBuffer buffer,
  RuntimeTensorDataType dtype,
  String name,
) {
  if (buffer.dtype != dtype) {
    throw StateError('Expected ${dtype.name} $name, got ${buffer.dtype.name}.');
  }
}

final class _Candidate {
  _Candidate(this.id, this.logit);
  final int id;
  final double logit;
  double prob = 0;
}

bool _candidateBetter(_Candidate a, _Candidate b) {
  if (a.logit > b.logit) return true;
  if (a.logit < b.logit) return false;
  return a.id < b.id;
}

bool _candidateWorse(_Candidate a, _Candidate b) {
  if (a.logit < b.logit) return true;
  if (a.logit > b.logit) return false;
  return a.id > b.id;
}

int _sampleTopP(
  List<_Candidate> candidates,
  double temperature,
  double topP,
  double randomDraw,
) {
  if (candidates.isEmpty) {
    throw StateError('NeuTTS Air native helper received invalid input.');
  }
  if (temperature == 0) {
    var best = candidates.first;
    for (final candidate in candidates.skip(1)) {
      if (_candidateBetter(candidate, best)) best = candidate;
    }
    return best.id;
  }
  var maxLogit = candidates.first.logit;
  for (final candidate in candidates.skip(1)) {
    if (candidate.logit > maxLogit) maxLogit = candidate.logit;
  }
  var sum = 0.0;
  for (final candidate in candidates) {
    candidate.prob = math.exp((candidate.logit - maxLogit) / temperature);
    sum += candidate.prob;
  }
  if (!(sum > 0)) {
    throw StateError('NeuTTS Air native helper received invalid input.');
  }
  for (final candidate in candidates) {
    candidate.prob /= sum;
  }
  candidates.sort((a, b) {
    final byProb = b.prob.compareTo(a.prob);
    return byProb != 0 ? byProb : a.id.compareTo(b.id);
  });
  final nucleusP = topP > 1 ? 1.0 : topP;
  var nucleusCount = 0;
  var cumulative = 0.0;
  while (nucleusCount < candidates.length && cumulative < nucleusP) {
    cumulative += candidates[nucleusCount].prob;
    nucleusCount += 1;
  }
  final slice = candidates.take(math.max(1, nucleusCount)).toList();
  final target =
      randomDraw.clamp(0.0, 0.9999999999999999) *
      slice.fold<double>(0, (sum, candidate) => sum + candidate.prob);
  cumulative = 0.0;
  for (final candidate in slice) {
    cumulative += candidate.prob;
    if (target < cumulative) return candidate.id;
  }
  return slice.last.id;
}
