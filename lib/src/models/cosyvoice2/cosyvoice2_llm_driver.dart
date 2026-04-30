// CosyVoice2 LLM autoregressive driver.
//
// Owns the three split-LLM ONNX sessions (`llm_prefill`, `llm_decode`,
// `llm_decoder_head`), the embedding table NPZ (`text_embedding`,
// `llm_embedding`, `speech_embedding`), and a Qwen2 BPE tokenizer.
//
// The ONNX I/O contract is:
//   * llm_prefill:  inputs_embeds[B,S,896], attention_mask[B,S]
//                   -> hidden[B,S,896], present_key_{0..23}, present_value_{0..23}
//   * llm_decode:   inputs_embeds[B,1,896], attention_mask[B,total],
//                   past_key_*, past_value_*
//                   -> hidden[B,1,896], present_key_*, present_value_*
//   * llm_decoder_head: hidden[B,S,896] -> logits[B,S,6564]
//
// This driver intentionally exposes raw primitives (`prefill`, `decodeStep`,
// `headLogits`) rather than a sampling policy.  Callers (Step 9 smoke /
// production serving) compose those with their own sampling logic.

import 'dart:typed_data';

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart' show RuntimeTensor, RuntimeTensorDataType;
import '../kokoro/kokoro.dart' show NpyArray, loadNpz;
import 'cosyvoice2.dart';
import 'cosyvoice2_native.dart';
import 'qwen2_tokenizer.dart';

/// Shape constants for the cosyvoice2 0.5B LLM as currently exported.
const int cosyvoice2LlmHiddenDim = 896;
const int cosyvoice2LlmNumLayers = 24;
const int cosyvoice2LlmKvHeads = 2;
const int cosyvoice2LlmKvHeadDim = 64;
const int cosyvoice2SpeechVocabSize = 6564;
const int cosyvoice2TextVocabSize = 151936;

/// Mutable per-stream state carried across `prefill` -> `decodeStep`
/// invocations.  Holds the latest hidden state, cached sequence length, and
/// the 24-layer KV cache.
final class CosyVoice2LlmState {
  CosyVoice2LlmState._({
    required this.totalSeq,
    required this.kvKeys,
    required this.kvValues,
    required this.lastHidden,
    required DartOnnxResult owner,
    required CosyLlmAttentionMaskCache attentionMask,
    NativeTensorBuffer? ownedLastHidden,
  }) : _owner = owner,
       _attentionMask = attentionMask,
       _ownedLastHidden = ownedLastHidden,
       _decodeInputs = _newLlmDecodeInputMap(kvKeys, kvValues);

  /// Total cached sequence length so far.
  int totalSeq;

  /// Per-layer KV cache. These are native ORT output tensors with shape
  /// `[1, 2, total_seq, 64]`; keeping them native avoids a 48-tensor
  /// Dart heap round trip on every decode step.
  List<RuntimeTensor> kvKeys;
  List<RuntimeTensor> kvValues;

  /// Hidden state of the last produced position, `[1, 1, 896]`.
  Object lastHidden;

  DartOnnxResult _owner;
  final CosyLlmAttentionMaskCache _attentionMask;
  NativeTensorBuffer? _ownedLastHidden;
  final Map<String, Object?> _decodeInputs;

  void _replaceOwner(DartOnnxResult nextOwner, RuntimeTensor nextHidden) {
    final previous = _owner;
    final previousHidden = _ownedLastHidden;
    _owner = nextOwner;
    lastHidden = nextHidden;
    _ownedLastHidden = null;
    previousHidden?.close();
    previous.close();
  }

  Map<String, Object?> _decodeInputMap({
    required RuntimeTensor nextEmbed,
    required RuntimeTensor attentionMask,
  }) {
    _decodeInputs['inputs_embeds'] = nextEmbed;
    _decodeInputs['attention_mask'] = attentionMask;
    _refreshDecodeKvInputs();
    return _decodeInputs;
  }

  void _refreshDecodeKvInputs() {
    for (var l = 0; l < cosyvoice2LlmNumLayers; l += 1) {
      _decodeInputs['past_key_$l'] = kvKeys[l];
      _decodeInputs['past_value_$l'] = kvValues[l];
    }
  }

  void _clearDecodeInputMap() {
    _decodeInputs.clear();
  }

  void close() {
    _clearDecodeInputMap();
    _ownedLastHidden?.close();
    _ownedLastHidden = null;
    _attentionMask.close();
    _owner.close();
  }

  RuntimeTensor attentionMaskTensor(int seqLen) =>
      _attentionMask.tensor(seqLen);
}

final class CosyVoice2LlmHeadLogits {
  CosyVoice2LlmHeadLogits._({required this.tensor, required this.owner});

  final RuntimeTensor tensor;
  final DartOnnxResult owner;

  void close() {
    owner.close();
  }
}

/// Pure-Dart driver around the split-LLM ONNX bundle.
///
/// Lifecycle:
///   1. [CosyVoice2LlmDriver.load] from a partial ONNX bundle (must
///      contain `llm_prefill`, `llm_decode`, `llm_decoder_head`) plus
///      the tokenizer + embedding NPZ.
///   2. Call [prefill] with a pre-built embedding tensor (callers can
///      embed text token ids via [embedTextTokens] and prepend the
///      `<|start_of_text|>` / speaker / instruction structure as needed).
///   3. Call [decodeStep] in a loop; each step returns the predicted
///      hidden vector and updates the KV cache in place.
///   4. Use [headLogits] to convert a hidden vector to speech-token
///      logits for sampling.
final class CosyVoice2LlmDriver {
  CosyVoice2LlmDriver._({
    required this.tokenizer,
    required this.textEmbedding,
    required this.llmEmbedding,
    required this.speechEmbedding,
    required CosyVoice2LoadedComponent prefill,
    required CosyVoice2LoadedComponent decode,
    required CosyVoice2LoadedComponent head,
  }) : _prefill = prefill,
       _decode = decode,
       _head = head;

  final Qwen2BpeTokenizer tokenizer;

  /// `[151936, 896]` float32 input-embedding table for text tokens.
  final NpyArray textEmbedding;

  /// `[2, 896]` float32 special-token embedding (BOS / task tokens).
  final NpyArray llmEmbedding;

  /// `[6564, 896]` float32 speech-token embedding table for the
  /// autoregressive loop body.
  final NpyArray speechEmbedding;

  final CosyVoice2LoadedComponent _prefill;
  final CosyVoice2LoadedComponent _decode;
  final CosyVoice2LoadedComponent _head;

  static Future<CosyVoice2LlmDriver> load({
    required CosyVoice2PartialOnnxBundle bundle,
    required CosyVoice2Paths paths,
  }) async {
    final prefill = bundle.requireLoadedComponent('llm_prefill');
    final decode = bundle.requireLoadedComponent('llm_decode');
    final head = bundle.requireLoadedComponent('llm_decoder_head');
    final tokenizer = await Qwen2BpeTokenizer.load(paths.qwen2TokenizerDir);
    Map<String, NpyArray>? embeddings;
    final taken = <NpyArray>[];
    try {
      final loadedEmbeddings = await loadNpz(paths.llmEmbeddingsNpz);
      embeddings = loadedEmbeddings;
      final text = loadedEmbeddings.remove('text_embedding');
      final llm = loadedEmbeddings.remove('llm_embedding');
      final speech = loadedEmbeddings.remove('speech_embedding');
      if (text != null) {
        taken.add(text);
      }
      if (llm != null) {
        taken.add(llm);
      }
      if (speech != null) {
        taken.add(speech);
      }
      if (text == null || llm == null || speech == null) {
        throw StateError(
          'llm_embeddings.npz is missing required arrays '
          '(have ${loadedEmbeddings.keys.toList()}).',
        );
      }
      for (final value in loadedEmbeddings.values) {
        value.close();
      }
      _expectShape(text.shape, [
        cosyvoice2TextVocabSize,
        cosyvoice2LlmHiddenDim,
      ], 'text_embedding');
      _expectShape(llm.shape, [2, cosyvoice2LlmHiddenDim], 'llm_embedding');
      _expectShape(speech.shape, [
        cosyvoice2SpeechVocabSize,
        cosyvoice2LlmHiddenDim,
      ], 'speech_embedding');
      return CosyVoice2LlmDriver._(
        tokenizer: tokenizer,
        textEmbedding: text,
        llmEmbedding: llm,
        speechEmbedding: speech,
        prefill: prefill,
        decode: decode,
        head: head,
      );
    } catch (_) {
      for (final value in taken) {
        value.close();
      }
      for (final value in embeddings?.values ?? const <NpyArray>[]) {
        value.close();
      }
      tokenizer.close();
      rethrow;
    }
  }

  /// Embeds a list of text token ids into a `[1, ids.length, 896]`
  /// flattened float32 buffer using the text-embedding table.
  Float32List embedTextTokens(List<int> ids) {
    final out = Float32List(ids.length * cosyvoice2LlmHiddenDim);
    for (var i = 0; i < ids.length; i += 1) {
      final row = textEmbedding.row(ids[i]);
      out.setRange(
        i * cosyvoice2LlmHiddenDim,
        (i + 1) * cosyvoice2LlmHiddenDim,
        row,
      );
    }
    return out;
  }

  /// Embeds a single speech token id into a `[1, 1, 896]` flattened
  /// float32 buffer using the speech-embedding table.
  Float32List embedSpeechToken(int id) {
    final buffer = embedSpeechTokenBuffer(id);
    try {
      return Float32List.fromList(buffer.asFloat32List());
    } finally {
      buffer.close();
    }
  }

  NativeTensorBuffer embedSpeechTokenBuffer(int id) {
    return cosyLlmEmbedSpeechToken(
      token: id,
      speechEmbedding: speechEmbedding,
      speechVocabSize: cosyvoice2SpeechVocabSize,
      dim: cosyvoice2LlmHiddenDim,
    );
  }

  NativeTensorBuffer createSpeechTokenEmbeddingBuffer() =>
      cosyLlmSpeechTokenBuffer(dim: cosyvoice2LlmHiddenDim);

  void fillSpeechTokenEmbeddingBuffer({
    required int id,
    required NativeTensorBuffer out,
  }) {
    cosyLlmEmbedSpeechTokenInto(
      token: id,
      speechEmbedding: speechEmbedding,
      speechVocabSize: cosyvoice2SpeechVocabSize,
      dim: cosyvoice2LlmHiddenDim,
      out: out,
    );
  }

  /// Embeds a CosyVoice2 LLM special token row. Row 0 is SOS/EOS and
  /// row 1 is the task marker used before speech tokens.
  Float32List embedLlmSpecial(int row) {
    final out = Float32List(cosyvoice2LlmHiddenDim);
    out.setRange(0, cosyvoice2LlmHiddenDim, llmEmbedding.row(row));
    return out;
  }

  /// Builds the upstream unistream CosyVoice2 prompt:
  /// `[sos, prompt_text + text, task_id, prompt_speech_tokens]`.
  Float32List buildPrefillEmbeddings({
    required Object textTokens,
    Object promptSpeechTokens = const <int>[],
  }) {
    final buffer = buildPrefillEmbeddingBuffer(
      textTokens: textTokens,
      promptSpeechTokens: promptSpeechTokens,
    );
    try {
      return Float32List.fromList(buffer.asFloat32List());
    } finally {
      buffer.close();
    }
  }

  NativeTensorBuffer buildPrefillEmbeddingBuffer({
    required Object textTokens,
    Object promptSpeechTokens = const <int>[],
  }) {
    return cosyLlmBuildPrefillEmbeddings(
      textTokens: textTokens,
      promptSpeechTokens: promptSpeechTokens,
      textEmbedding: textEmbedding,
      llmEmbedding: llmEmbedding,
      speechEmbedding: speechEmbedding,
      textVocabSize: cosyvoice2TextVocabSize,
      speechVocabSize: cosyvoice2SpeechVocabSize,
      dim: cosyvoice2LlmHiddenDim,
    );
  }

  CosyLlmPrefillTextPlan buildPrefillEmbeddingBufferFromText({
    required String text,
    String promptText = '',
    Object promptSpeechTokens = const <int>[],
  }) {
    return cosyLlmBuildPrefillEmbeddingsFromText(
      tokenizer: tokenizer,
      text: text,
      promptText: promptText,
      promptSpeechTokens: promptSpeechTokens,
      textEmbedding: textEmbedding,
      llmEmbedding: llmEmbedding,
      speechEmbedding: speechEmbedding,
      textVocabSize: cosyvoice2TextVocabSize,
      speechVocabSize: cosyvoice2SpeechVocabSize,
      dim: cosyvoice2LlmHiddenDim,
    );
  }

  /// Runs a prefill pass with `inputsEmbeds` of shape `[1, S, 896]`
  /// (flattened) and returns the initialized state.  `seqLen` must
  /// match `inputsEmbeds.length / 896`.
  CosyVoice2LlmState prefill({
    required Object inputsEmbeds,
    required int seqLen,
  }) {
    final expected = seqLen * cosyvoice2LlmHiddenDim;
    final length = _float32InputLength(inputsEmbeds);
    if (length != expected) {
      throw ArgumentError(
        'inputsEmbeds length $length != seqLen*$cosyvoice2LlmHiddenDim ($expected)',
      );
    }
    final mask = CosyLlmAttentionMaskCache(seqLen);
    var keepMask = false;
    try {
      final inputs = <String, Object?>{
        'inputs_embeds': _float32InputTensor(inputsEmbeds, [
          1,
          seqLen,
          cosyvoice2LlmHiddenDim,
        ]),
        'attention_mask': mask.tensor(seqLen),
      };
      final result = _prefill.run(inputs);
      var keepResult = false;
      NativeTensorBuffer? lastHidden;
      try {
        lastHidden = cosyLlmSliceLastHidden(
          hidden: _readFloat32Tensor(result, 'hidden'),
          seqLen: seqLen,
          dim: cosyvoice2LlmHiddenDim,
        );
        final keys = <RuntimeTensor>[];
        final values = <RuntimeTensor>[];
        for (var l = 0; l < cosyvoice2LlmNumLayers; l += 1) {
          keys.add(_readFloat32Tensor(result, 'present_key_$l'));
          values.add(_readFloat32Tensor(result, 'present_value_$l'));
        }
        keepResult = true;
        keepMask = true;
        return CosyVoice2LlmState._(
          totalSeq: seqLen,
          kvKeys: keys,
          kvValues: values,
          lastHidden: lastHidden,
          owner: result,
          attentionMask: mask,
          ownedLastHidden: lastHidden,
        );
      } finally {
        if (!keepResult) {
          lastHidden?.close();
          result.close();
        }
      }
    } finally {
      if (!keepMask) {
        mask.close();
      }
    }
  }

  /// One autoregressive step: feeds the current hidden's downstream
  /// embedding (caller-provided) into the decode session, refreshes the
  /// KV cache and `lastHidden`.
  void decodeStep({
    required CosyVoice2LlmState state,
    required Object nextEmbed,
  }) {
    final plan = cosyLlmDecodeStepPlan(
      pastSeq: state.totalSeq,
      hiddenDim: cosyvoice2LlmHiddenDim,
      layerCount: cosyvoice2LlmNumLayers,
    );
    final nextLength = _float32InputLength(nextEmbed);
    if (nextLength != plan.expectedEmbedFloats) {
      throw ArgumentError(
        'nextEmbed length $nextLength != ${plan.expectedEmbedFloats}',
      );
    }
    final inputs = state._decodeInputMap(
      nextEmbed: _float32InputTensor(nextEmbed, [1, 1, cosyvoice2LlmHiddenDim]),
      attentionMask: state.attentionMaskTensor(plan.nextSeq),
    );
    final result = _decode.run(inputs);
    var keepResult = false;
    try {
      final hidden = _readFloat32Tensor(result, 'hidden');
      final keys = <RuntimeTensor>[];
      final values = <RuntimeTensor>[];
      for (var l = 0; l < cosyvoice2LlmNumLayers; l += 1) {
        keys.add(_readFloat32Tensor(result, 'present_key_$l'));
        values.add(_readFloat32Tensor(result, 'present_value_$l'));
      }
      if (keys.length + values.length != plan.kvTensorCount) {
        throw StateError(
          'llm_decode emitted ${keys.length + values.length} KV tensors, '
          'expected ${plan.kvTensorCount}',
        );
      }
      state.kvKeys = keys;
      state.kvValues = values;
      state.totalSeq = plan.nextSeq;
      state._refreshDecodeKvInputs();
      keepResult = true;
      state._replaceOwner(result, hidden);
    } finally {
      if (!keepResult) {
        result.close();
      }
    }
  }

  /// Projects a `[896]` hidden vector to `[6564]` speech-token logits.
  Float32List headLogits(Object hidden) {
    final logits = headLogitsTensor(hidden);
    try {
      return Float32List.fromList(float32View(logits.tensor));
    } finally {
      logits.close();
    }
  }

  /// Projects hidden state to native-backed logits. The returned owner must be
  /// closed after sampling.
  CosyVoice2LlmHeadLogits headLogitsTensor(Object hidden) {
    final hiddenLength = _float32InputLength(hidden);
    if (hiddenLength % cosyvoice2LlmHiddenDim != 0) {
      throw ArgumentError(
        'hidden length $hiddenLength not a multiple of $cosyvoice2LlmHiddenDim',
      );
    }
    final seq = hiddenLength ~/ cosyvoice2LlmHiddenDim;
    final result = _head.run({
      'hidden': _float32InputTensor(hidden, [1, seq, cosyvoice2LlmHiddenDim]),
    });
    var keepResult = false;
    try {
      final tensor = _readFloat32Tensor(result, 'logits');
      keepResult = true;
      return CosyVoice2LlmHeadLogits._(tensor: tensor, owner: result);
    } finally {
      if (!keepResult) {
        result.close();
      }
    }
  }

  void close() {
    tokenizer.close();
    textEmbedding.close();
    llmEmbedding.close();
    speechEmbedding.close();
  }
}

void _expectShape(List<int> got, List<int> want, String name) {
  if (got.length != want.length) {
    throw StateError(
      '$name: expected rank ${want.length}, got rank ${got.length}',
    );
  }
  for (var i = 0; i < got.length; i += 1) {
    if (got[i] != want[i]) {
      throw StateError('$name: expected shape $want, got $got');
    }
  }
}

RuntimeTensor _float32InputTensor(Object value, List<int> shape) {
  if (value is NativeTensorBuffer) {
    _checkFloat32Buffer(value);
    return value.tensorView(shape: shape, byteLength: value.byteLength);
  }
  if (value is RuntimeTensor) {
    if (value.dtype != RuntimeTensorDataType.float32) {
      throw StateError('Expected float32 tensor, got ${value.dtype.name}.');
    }
    return value;
  }
  if (value is Float32List) {
    return float32Tensor(value, shape);
  }
  throw ArgumentError.value(
    value,
    'value',
    'expected NativeTensorBuffer/RuntimeTensor/Float32List',
  );
}

Map<String, Object?> _newLlmDecodeInputMap(
  List<RuntimeTensor> keys,
  List<RuntimeTensor> values,
) {
  final inputs = <String, Object?>{
    'inputs_embeds': null,
    'attention_mask': null,
  };
  for (var l = 0; l < cosyvoice2LlmNumLayers; l += 1) {
    inputs['past_key_$l'] = keys[l];
    inputs['past_value_$l'] = values[l];
  }
  return inputs;
}

int _float32InputLength(Object value) {
  if (value is NativeTensorBuffer) {
    _checkFloat32Buffer(value);
    return value.byteLength ~/ 4;
  }
  if (value is RuntimeTensor) {
    if (value.dtype != RuntimeTensorDataType.float32) {
      throw StateError('Expected float32 tensor, got ${value.dtype.name}.');
    }
    return value.bytes.lengthInBytes ~/ 4;
  }
  if (value is Float32List) {
    return value.length;
  }
  throw ArgumentError.value(
    value,
    'value',
    'expected NativeTensorBuffer/RuntimeTensor/Float32List',
  );
}

void _checkFloat32Buffer(NativeTensorBuffer value) {
  if (value.dtype != RuntimeTensorDataType.float32) {
    throw StateError('Expected float32 tensor, got ${value.dtype.name}.');
  }
}

RuntimeTensor _readFloat32Tensor(DartOnnxResult result, String name) {
  final value = result.outputs[name];
  if (value is RuntimeTensor) {
    if (value.dtype != RuntimeTensorDataType.float32) {
      throw StateError('LLM session output "$name" has dtype ${value.dtype}');
    }
    return value;
  }
  if (value == null) {
    throw StateError('LLM session output is missing "$name"');
  }
  throw StateError(
    'LLM session output "$name" is not native-backed: ${value.runtimeType}',
  );
}
