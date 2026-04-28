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

import '../../runtime/onnx.dart';
import '../../runtime/runtime.dart' show RuntimeTensor;
import '../kokoro/kokoro.dart' show NpyArray, loadNpz;
import 'cosyvoice2.dart';
import 'qwen2_tokenizer.dart';

/// Shape constants for the cosyvoice2 0.5B LLM as currently exported.
const int cosyvoice2LlmHiddenDim = 896;
const int cosyvoice2LlmNumLayers = 24;
const int cosyvoice2LlmKvHeads = 2;
const int cosyvoice2LlmKvHeadDim = 64;
const int cosyvoice2SpeechVocabSize = 6564;
const int cosyvoice2TextVocabSize = 151936;

/// Mutable per-stream state carried across `prefill` -> `decodeStep`
/// invocations.  Holds the latest hidden state, the running attention
/// mask, and the 24-layer KV cache.
final class CosyVoice2LlmState {
  CosyVoice2LlmState._({
    required this.attentionMask,
    required this.kvKeys,
    required this.kvValues,
    required this.lastHidden,
  });

  /// `[1, total_seq]` int64 attention mask (all ones in the trivial case).
  Int64List attentionMask;

  /// Per-layer KV cache: `kvKeys[L]` and `kvValues[L]` are flat float32
  /// buffers of shape `[1, 2, total_seq, 64]`.
  final List<Float32List> kvKeys;
  final List<Float32List> kvValues;

  /// Total cached sequence length so far.
  int get totalSeq => attentionMask.length;

  /// Hidden state of the last produced position, `[1, 1, 896]` flattened.
  Float32List lastHidden;
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
  })  : _prefill = prefill,
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
    final embeddings = await loadNpz(paths.llmEmbeddingsNpz);
    final text = embeddings['text_embedding'];
    final llm = embeddings['llm_embedding'];
    final speech = embeddings['speech_embedding'];
    if (text == null || llm == null || speech == null) {
      throw StateError(
        'llm_embeddings.npz is missing required arrays '
        '(have ${embeddings.keys.toList()}).',
      );
    }
    _expectShape(text.shape, [cosyvoice2TextVocabSize, cosyvoice2LlmHiddenDim],
        'text_embedding');
    _expectShape(llm.shape, [2, cosyvoice2LlmHiddenDim], 'llm_embedding');
    _expectShape(
        speech.shape,
        [cosyvoice2SpeechVocabSize, cosyvoice2LlmHiddenDim],
        'speech_embedding');
    return CosyVoice2LlmDriver._(
      tokenizer: tokenizer,
      textEmbedding: text,
      llmEmbedding: llm,
      speechEmbedding: speech,
      prefill: prefill,
      decode: decode,
      head: head,
    );
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
    final out = Float32List(cosyvoice2LlmHiddenDim);
    out.setRange(0, cosyvoice2LlmHiddenDim, speechEmbedding.row(id));
    return out;
  }

  /// Runs a prefill pass with `inputsEmbeds` of shape `[1, S, 896]`
  /// (flattened) and returns the initialized state.  `seqLen` must
  /// match `inputsEmbeds.length / 896`.
  CosyVoice2LlmState prefill({
    required Float32List inputsEmbeds,
    required int seqLen,
  }) {
    final expected = seqLen * cosyvoice2LlmHiddenDim;
    if (inputsEmbeds.length != expected) {
      throw ArgumentError(
        'inputsEmbeds length ${inputsEmbeds.length} != seqLen*$cosyvoice2LlmHiddenDim ($expected)',
      );
    }
    final mask = Int64List(seqLen);
    for (var i = 0; i < seqLen; i += 1) {
      mask[i] = 1;
    }
    final inputs = <String, Object?>{
      'inputs_embeds':
          float32Tensor(inputsEmbeds, [1, seqLen, cosyvoice2LlmHiddenDim]),
      'attention_mask': int64Tensor(mask, [1, seqLen]),
    };
    final result = _prefill.run(inputs);
    try {
      final hidden = _readFloat32(result, 'hidden');
      // Last position only (`[1, 1, 896]`) — that's what feeds the head /
      // first decode step.
      final lastOff = (seqLen - 1) * cosyvoice2LlmHiddenDim;
      final lastHidden = Float32List(cosyvoice2LlmHiddenDim);
      lastHidden.setRange(0, cosyvoice2LlmHiddenDim,
          hidden.sublist(lastOff, lastOff + cosyvoice2LlmHiddenDim));
      final keys = <Float32List>[];
      final values = <Float32List>[];
      for (var l = 0; l < cosyvoice2LlmNumLayers; l += 1) {
        keys.add(Float32List.fromList(_readFloat32(result, 'present_key_$l')));
        values.add(
            Float32List.fromList(_readFloat32(result, 'present_value_$l')));
      }
      return CosyVoice2LlmState._(
        attentionMask: mask,
        kvKeys: keys,
        kvValues: values,
        lastHidden: lastHidden,
      );
    } finally {
      result.close();
    }
  }

  /// One autoregressive step: feeds the current hidden's downstream
  /// embedding (caller-provided) into the decode session, refreshes the
  /// KV cache and `lastHidden`.
  void decodeStep({
    required CosyVoice2LlmState state,
    required Float32List nextEmbed,
  }) {
    if (nextEmbed.length != cosyvoice2LlmHiddenDim) {
      throw ArgumentError(
        'nextEmbed length ${nextEmbed.length} != $cosyvoice2LlmHiddenDim',
      );
    }
    final past = state.totalSeq;
    final newMask = Int64List(past + 1);
    newMask.setRange(0, past, state.attentionMask);
    newMask[past] = 1;

    final inputs = <String, Object?>{
      'inputs_embeds': float32Tensor(
          nextEmbed, [1, 1, cosyvoice2LlmHiddenDim]),
      'attention_mask': int64Tensor(newMask, [1, past + 1]),
    };
    for (var l = 0; l < cosyvoice2LlmNumLayers; l += 1) {
      inputs['past_key_$l'] = float32Tensor(state.kvKeys[l],
          [1, cosyvoice2LlmKvHeads, past, cosyvoice2LlmKvHeadDim]);
      inputs['past_value_$l'] = float32Tensor(state.kvValues[l],
          [1, cosyvoice2LlmKvHeads, past, cosyvoice2LlmKvHeadDim]);
    }
    final result = _decode.run(inputs);
    try {
      final hidden = _readFloat32(result, 'hidden');
      state.lastHidden = Float32List.fromList(hidden);
      for (var l = 0; l < cosyvoice2LlmNumLayers; l += 1) {
        state.kvKeys[l] =
            Float32List.fromList(_readFloat32(result, 'present_key_$l'));
        state.kvValues[l] =
            Float32List.fromList(_readFloat32(result, 'present_value_$l'));
      }
      state.attentionMask = newMask;
    } finally {
      result.close();
    }
  }

  /// Projects a `[896]` hidden vector to `[6564]` speech-token logits.
  Float32List headLogits(Float32List hidden) {
    if (hidden.length % cosyvoice2LlmHiddenDim != 0) {
      throw ArgumentError(
        'hidden length ${hidden.length} not a multiple of $cosyvoice2LlmHiddenDim',
      );
    }
    final seq = hidden.length ~/ cosyvoice2LlmHiddenDim;
    final result = _head.run({
      'hidden':
          float32Tensor(hidden, [1, seq, cosyvoice2LlmHiddenDim]),
    });
    try {
      return Float32List.fromList(_readFloat32(result, 'logits'));
    } finally {
      result.close();
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
    throw StateError('$name: expected rank ${want.length}, got rank ${got.length}');
  }
  for (var i = 0; i < got.length; i += 1) {
    if (got[i] != want[i]) {
      throw StateError('$name: expected shape $want, got $got');
    }
  }
}

Float32List _readFloat32(DartOnnxResult result, String name) {
  final value = result.outputs[name];
  if (value == null) {
    throw StateError('LLM session output is missing "$name"');
  }
  if (value is Float32List) return value;
  if (value is RuntimeTensor) {
    return float32View(value);
  }
  if (value is List<double>) {
    return Float32List.fromList(value);
  }
  throw StateError('LLM session output "$name" has unexpected type: ${value.runtimeType}');
}
