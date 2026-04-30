import 'dart:typed_data';

import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart' show RuntimeTensor, RuntimeTensorDataType;
import '../cosyvoice2/cosyvoice2.dart';
import '../cosyvoice2/cosyvoice2_native.dart';
import '../kokoro/kokoro.dart' show NpyArray, loadNpz;
import 'sarashina2.dart';

const int sarashina2LlmHiddenDim = 1280;
const int sarashina2LlmNumLayers = 24;
const int sarashina2LlmKvHeads = 8;
const int sarashina2LlmKvHeadDim = 80;
const int sarashina2LlmVocabSize = 108986;

final class Sarashina2LlmState {
  Sarashina2LlmState._({
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

  int totalSeq;
  List<RuntimeTensor> kvKeys;
  List<RuntimeTensor> kvValues;
  Object? lastHidden;
  RuntimeTensor? lastLogits;

  DartOnnxResult _owner;
  final CosyLlmAttentionMaskCache _attentionMask;
  NativeTensorBuffer? _ownedLastHidden;
  final Map<String, Object?> _decodeInputs;

  void _replaceOwner(
    DartOnnxResult nextOwner, {
    RuntimeTensor? nextHidden,
    RuntimeTensor? nextLogits,
  }) {
    if (nextHidden == null && nextLogits == null) {
      throw ArgumentError('decode result must contain hidden or logits.');
    }
    final previous = _owner;
    final previousHidden = _ownedLastHidden;
    _owner = nextOwner;
    lastHidden = nextHidden;
    lastLogits = nextLogits;
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
    for (var layer = 0; layer < sarashina2LlmNumLayers; layer += 1) {
      _decodeInputs['past_key_$layer'] = kvKeys[layer];
      _decodeInputs['past_value_$layer'] = kvValues[layer];
    }
  }

  void close() {
    _decodeInputs.clear();
    _ownedLastHidden?.close();
    _ownedLastHidden = null;
    _attentionMask.close();
    _owner.close();
  }

  RuntimeTensor attentionMaskTensor(int seqLen) =>
      _attentionMask.tensor(seqLen);
}

final class Sarashina2LlmHeadLogits {
  Sarashina2LlmHeadLogits._({required this.tensor, required this.owner});

  final RuntimeTensor tensor;
  final DartOnnxResult owner;

  void close() {
    owner.close();
  }
}

final class Sarashina2LlmDriver {
  Sarashina2LlmDriver._({
    required this.paths,
    required this.tokenMap,
    required this.tokenEmbedding,
    required CosyVoice2LoadedComponent prefill,
    required CosyVoice2LoadedComponent? decode,
    required CosyVoice2LoadedComponent? decodeHead,
    required CosyVoice2LoadedComponent head,
  }) : _prefill = prefill,
       _decode = decode,
       _decodeHead = decodeHead,
       _head = head;

  final Sarashina2TtsPaths paths;
  final Sarashina2TokenMap tokenMap;
  final NpyArray tokenEmbedding;
  final CosyVoice2LoadedComponent _prefill;
  final CosyVoice2LoadedComponent? _decode;
  final CosyVoice2LoadedComponent? _decodeHead;
  final CosyVoice2LoadedComponent _head;

  static Future<Sarashina2LlmDriver> load({
    required CosyVoice2PartialOnnxBundle bundle,
    required Sarashina2TtsPaths paths,
    Sarashina2TokenMap? tokenMap,
  }) async {
    final prefill = bundle.requireLoadedComponent('llm_prefill');
    final decode = bundle.loadedComponent('llm_decode');
    final decodeHead = bundle.loadedComponent('llm_decode_head');
    if (decode == null && decodeHead == null) {
      throw StateError(
        'CosyVoice2 ONNX component is not loaded: llm_decode or llm_decode_head',
      );
    }
    final head = bundle.requireLoadedComponent('llm_decoder_head');
    Map<String, NpyArray>? embeddings;
    NpyArray? tokenEmbedding;
    try {
      embeddings = await loadNpz(paths.llmEmbeddingsNpz);
      tokenEmbedding =
          embeddings.remove('token_embedding') ??
          embeddings.remove('model_embed_tokens') ??
          embeddings.remove('embed_tokens');
      if (tokenEmbedding == null) {
        throw StateError(
          'llm_embeddings.npz is missing token_embedding '
          '(have ${embeddings.keys.toList()}).',
        );
      }
      for (final value in embeddings.values) {
        value.close();
      }
      _expectShape(tokenEmbedding.shape, [
        sarashina2LlmVocabSize,
        sarashina2LlmHiddenDim,
      ], 'token_embedding');
      return Sarashina2LlmDriver._(
        paths: paths,
        tokenMap: tokenMap ?? Sarashina2TokenMap.fromPaths(paths),
        tokenEmbedding: tokenEmbedding,
        prefill: prefill,
        decode: decode,
        decodeHead: decodeHead,
        head: head,
      );
    } catch (_) {
      tokenEmbedding?.close();
      for (final value in embeddings?.values ?? const <NpyArray>[]) {
        value.close();
      }
      rethrow;
    }
  }

  NativeTensorBuffer embedTokenIdsBuffer(Object tokenIds) {
    _validateTokenIds(tokenIds);
    return cosyFlowEmbedTokens(
      tokens: tokenIds,
      inputEmbedding: tokenEmbedding,
      vocabSize: sarashina2LlmVocabSize,
      dim: sarashina2LlmHiddenDim,
    );
  }

  NativeTensorBuffer embedTokenIdBuffer(int tokenId) {
    _validateTokenId(tokenId);
    return cosyFlowEmbedOneToken(
      token: tokenId,
      inputEmbedding: tokenEmbedding,
      vocabSize: sarashina2LlmVocabSize,
      dim: sarashina2LlmHiddenDim,
    );
  }

  NativeTensorBuffer createTokenEmbeddingBuffer() =>
      NativeTensorBuffer.float32([1, 1, sarashina2LlmHiddenDim]);

  void fillTokenEmbeddingBuffer({
    required int tokenId,
    required NativeTensorBuffer out,
  }) {
    _validateTokenId(tokenId);
    cosyFlowEmbedOneTokenInto(
      token: tokenId,
      inputEmbedding: tokenEmbedding,
      vocabSize: sarashina2LlmVocabSize,
      dim: sarashina2LlmHiddenDim,
      out: out,
    );
  }

  Sarashina2LlmState prefillTokenIds(Object tokenIds) {
    final embeddings = embedTokenIdsBuffer(tokenIds);
    try {
      return prefill(
        inputsEmbeds: embeddings,
        seqLen: _int32InputLength(tokenIds),
      );
    } finally {
      embeddings.close();
    }
  }

  Sarashina2LlmState prefill({
    required Object inputsEmbeds,
    required int seqLen,
  }) {
    final expected = seqLen * sarashina2LlmHiddenDim;
    final length = _float32InputLength(inputsEmbeds);
    if (length != expected) {
      throw ArgumentError(
        'inputsEmbeds length $length != seqLen*$sarashina2LlmHiddenDim ($expected)',
      );
    }
    final mask = CosyLlmAttentionMaskCache(seqLen);
    var keepMask = false;
    try {
      final result = _prefill.run({
        'inputs_embeds': _float32InputTensor(inputsEmbeds, [
          1,
          seqLen,
          sarashina2LlmHiddenDim,
        ]),
        'attention_mask': mask.tensor(seqLen),
      });
      var keepResult = false;
      Object? lastHidden;
      NativeTensorBuffer? ownedLastHidden;
      try {
        final hidden = _readFloat32Tensor(result, 'hidden');
        final hiddenLength = _float32InputLength(hidden);
        if (hiddenLength == sarashina2LlmHiddenDim) {
          lastHidden = hidden;
        } else if (hiddenLength == seqLen * sarashina2LlmHiddenDim) {
          ownedLastHidden = cosyLlmSliceLastHidden(
            hidden: hidden,
            seqLen: seqLen,
            dim: sarashina2LlmHiddenDim,
          );
          lastHidden = ownedLastHidden;
        } else {
          throw StateError(
            'llm_prefill emitted hidden length $hiddenLength, expected '
            '$sarashina2LlmHiddenDim or ${seqLen * sarashina2LlmHiddenDim}.',
          );
        }
        final keys = <RuntimeTensor>[];
        final values = <RuntimeTensor>[];
        for (var layer = 0; layer < sarashina2LlmNumLayers; layer += 1) {
          keys.add(_readFloat32Tensor(result, 'present_key_$layer'));
          values.add(_readFloat32Tensor(result, 'present_value_$layer'));
        }
        keepResult = true;
        keepMask = true;
        return Sarashina2LlmState._(
          totalSeq: seqLen,
          kvKeys: keys,
          kvValues: values,
          lastHidden: lastHidden,
          owner: result,
          attentionMask: mask,
          ownedLastHidden: ownedLastHidden,
        );
      } finally {
        if (!keepResult) {
          ownedLastHidden?.close();
          result.close();
        }
      }
    } finally {
      if (!keepMask) {
        mask.close();
      }
    }
  }

  void decodeTokenId({
    required Sarashina2LlmState state,
    required int tokenId,
  }) {
    final embedding = embedTokenIdBuffer(tokenId);
    try {
      decodeStep(state: state, nextEmbed: embedding);
    } finally {
      embedding.close();
    }
  }

  void decodeStep({
    required Sarashina2LlmState state,
    required Object nextEmbed,
  }) {
    final plan = cosyLlmDecodeStepPlan(
      pastSeq: state.totalSeq,
      hiddenDim: sarashina2LlmHiddenDim,
      layerCount: sarashina2LlmNumLayers,
    );
    final nextLength = _float32InputLength(nextEmbed);
    if (nextLength != plan.expectedEmbedFloats) {
      throw ArgumentError(
        'nextEmbed length $nextLength != ${plan.expectedEmbedFloats}',
      );
    }
    final inputs = state._decodeInputMap(
      nextEmbed: _float32InputTensor(nextEmbed, [1, 1, sarashina2LlmHiddenDim]),
      attentionMask: state.attentionMaskTensor(plan.nextSeq),
    );
    final fusedDecodeHead = _decodeHead;
    final decode = fusedDecodeHead ?? _decode;
    if (decode == null) {
      throw StateError('Sarashina2 LLM decode component is not loaded.');
    }
    final result = decode.run(inputs);
    var keepResult = false;
    try {
      final hidden = fusedDecodeHead == null
          ? _readFloat32Tensor(result, 'hidden')
          : null;
      final logits = fusedDecodeHead == null
          ? null
          : _readFloat32Tensor(result, 'logits');
      final keys = <RuntimeTensor>[];
      final values = <RuntimeTensor>[];
      for (var layer = 0; layer < sarashina2LlmNumLayers; layer += 1) {
        keys.add(_readFloat32Tensor(result, 'present_key_$layer'));
        values.add(_readFloat32Tensor(result, 'present_value_$layer'));
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
      keepResult = true;
      state._replaceOwner(result, nextHidden: hidden, nextLogits: logits);
    } finally {
      if (!keepResult) {
        result.close();
      }
    }
  }

  Float32List headLogits(Object hidden) {
    final logits = headLogitsTensor(hidden);
    try {
      return Float32List.fromList(float32View(logits.tensor));
    } finally {
      logits.close();
    }
  }

  Sarashina2LlmHeadLogits headLogitsTensor(Object hidden) {
    final length = _float32InputLength(hidden);
    if (length % sarashina2LlmHiddenDim != 0) {
      throw ArgumentError(
        'hidden length $length not a multiple of $sarashina2LlmHiddenDim',
      );
    }
    final seq = length ~/ sarashina2LlmHiddenDim;
    final result = _head.run({
      'hidden': _float32InputTensor(hidden, [1, seq, sarashina2LlmHiddenDim]),
    });
    var keepResult = false;
    try {
      final tensor = _readFloat32Tensor(result, 'logits');
      keepResult = true;
      return Sarashina2LlmHeadLogits._(tensor: tensor, owner: result);
    } finally {
      if (!keepResult) {
        result.close();
      }
    }
  }

  int sampleSemanticTokenizerId({
    required Object hidden,
    required List<int> generatedSemanticTokens,
    Sarashina2SemanticSamplerState? samplerState,
    required int eosId,
    required double temperature,
    required double topP,
    required double frequencyPenalty,
    required double randomDraw,
  }) {
    final logits = headLogitsTensor(hidden);
    try {
      return sampleSemanticTokenizerIdFromLogits(
        logits: logits.tensor,
        generatedSemanticTokens: generatedSemanticTokens,
        samplerState: samplerState,
        eosId: eosId,
        temperature: temperature,
        topP: topP,
        frequencyPenalty: frequencyPenalty,
        randomDraw: randomDraw,
      );
    } finally {
      logits.close();
    }
  }

  int sampleNextSemanticTokenizerId({
    required Sarashina2LlmState state,
    required List<int> generatedSemanticTokens,
    Sarashina2SemanticSamplerState? samplerState,
    required int eosId,
    required double temperature,
    required double topP,
    required double frequencyPenalty,
    required double randomDraw,
  }) {
    final logits = state.lastLogits;
    if (logits != null) {
      return sampleSemanticTokenizerIdFromLogits(
        logits: logits,
        generatedSemanticTokens: generatedSemanticTokens,
        samplerState: samplerState,
        eosId: eosId,
        temperature: temperature,
        topP: topP,
        frequencyPenalty: frequencyPenalty,
        randomDraw: randomDraw,
      );
    }
    final hidden = state.lastHidden;
    if (hidden == null) {
      throw StateError('Sarashina2 LLM state has no hidden or logits tensor.');
    }
    return sampleSemanticTokenizerId(
      hidden: hidden,
      generatedSemanticTokens: generatedSemanticTokens,
      samplerState: samplerState,
      eosId: eosId,
      temperature: temperature,
      topP: topP,
      frequencyPenalty: frequencyPenalty,
      randomDraw: randomDraw,
    );
  }

  int sampleSemanticTokenizerIdFromLogits({
    required RuntimeTensor logits,
    required List<int> generatedSemanticTokens,
    Sarashina2SemanticSamplerState? samplerState,
    required int eosId,
    required double temperature,
    required double topP,
    required double frequencyPenalty,
    required double randomDraw,
  }) {
    final state = samplerState;
    if (state != null) {
      return sampleSarashina2SemanticTokenizerIdWithState(
        logits: logits,
        samplerState: state,
        eosId: eosId,
        temperature: temperature,
        topP: topP,
        frequencyPenalty: frequencyPenalty,
        randomDraw: randomDraw,
      );
    }
    return sampleSarashina2SemanticTokenizerId(
      logits: logits,
      generatedSemanticTokens: generatedSemanticTokens,
      tokenMap: tokenMap,
      eosId: eosId,
      temperature: temperature,
      topP: topP,
      frequencyPenalty: frequencyPenalty,
      randomDraw: randomDraw,
    );
  }

  void close() {
    tokenEmbedding.close();
  }

  void _validateTokenIds(Object tokenIds) {
    final values = _int32InputView(tokenIds);
    for (final tokenId in values) {
      _validateTokenId(tokenId);
    }
  }

  void _validateTokenId(int tokenId) {
    if (tokenId < 0 || tokenId >= sarashina2LlmVocabSize) {
      throw RangeError.range(tokenId, 0, sarashina2LlmVocabSize - 1, 'tokenId');
    }
  }
}

int _int32InputLength(Object value) {
  if (value is NativeTensorBuffer) {
    _checkInt32Buffer(value);
    return value.byteLength ~/ 4;
  }
  if (value is RuntimeTensor) {
    if (value.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tensor, got ${value.dtype.name}.');
    }
    return value.byteLength ~/ 4;
  }
  if (value is Int32List) {
    return value.length;
  }
  if (value is List<int>) {
    return value.length;
  }
  throw ArgumentError.value(
    value,
    'value',
    'expected NativeTensorBuffer/RuntimeTensor/Int32List/List<int>',
  );
}

Iterable<int> _int32InputView(Object value) {
  if (value is NativeTensorBuffer) {
    _checkInt32Buffer(value);
    return value.asInt32List();
  }
  if (value is RuntimeTensor) {
    if (value.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tensor, got ${value.dtype.name}.');
    }
    return value.asInt32List();
  }
  if (value is Int32List) {
    return value;
  }
  if (value is List<int>) {
    return value;
  }
  throw ArgumentError.value(
    value,
    'value',
    'expected NativeTensorBuffer/RuntimeTensor/Int32List/List<int>',
  );
}

void _checkInt32Buffer(NativeTensorBuffer value) {
  if (value.dtype != RuntimeTensorDataType.int32) {
    throw StateError('Expected int32 tensor, got ${value.dtype.name}.');
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
  for (var layer = 0; layer < sarashina2LlmNumLayers; layer += 1) {
    inputs['past_key_$layer'] = keys[layer];
    inputs['past_value_$layer'] = values[layer];
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
    return value.byteLength ~/ 4;
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
