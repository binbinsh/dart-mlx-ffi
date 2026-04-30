import 'dart:ffi' as ffi;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

import '../kokoro/kokoro.dart' show NpyArray;
import 'qwen2_tokenizer.dart';

final class CosyLlmPrefillTextPlan {
  const CosyLlmPrefillTextPlan({
    required this.buffer,
    required this.seqLen,
    required this.targetTextTokenCount,
  });

  final NativeTensorBuffer buffer;
  final int seqLen;
  final int targetTextTokenCount;

  RuntimeTensor get tensor => buffer.tensorView(
    shape: [1, seqLen, buffer.shape[2]],
    byteLength: byteLength,
  );

  int get byteLength => seqLen * buffer.shape[2] * 4;

  void close() {
    buffer.close();
  }
}

final class CosyLlmDecodeStepPlan {
  const CosyLlmDecodeStepPlan({
    required this.nextSeq,
    required this.attentionCapacity,
    required this.expectedEmbedFloats,
    required this.kvTensorCount,
  });

  final int nextSeq;
  final int attentionCapacity;
  final int expectedEmbedFloats;
  final int kvTensorCount;
}

int cosyLlmAttentionCapacity(int seqLen) {
  if (seqLen < 1) {
    throw RangeError.range(seqLen, 1, null, 'seqLen');
  }
  var capacity = 1;
  while (capacity < seqLen) {
    capacity *= 2;
  }
  return capacity;
}

CosyLlmDecodeStepPlan cosyLlmDecodeStepPlan({
  required int pastSeq,
  required int hiddenDim,
  required int layerCount,
}) {
  if (pastSeq < 1) {
    throw RangeError.range(pastSeq, 1, null, 'pastSeq');
  }
  if (hiddenDim < 1) {
    throw RangeError.range(hiddenDim, 1, null, 'hiddenDim');
  }
  if (layerCount < 1) {
    throw RangeError.range(layerCount, 1, null, 'layerCount');
  }
  final nextSeq = pastSeq + 1;
  return CosyLlmDecodeStepPlan(
    nextSeq: nextSeq,
    attentionCapacity: cosyLlmAttentionCapacity(nextSeq),
    expectedEmbedFloats: hiddenDim,
    kvTensorCount: layerCount * 2,
  );
}

NativeTensorBuffer cosyFlowEmbedTokens({
  required Object tokens,
  required NpyArray inputEmbedding,
  required int vocabSize,
  required int dim,
}) {
  final tokenValues = _int32Values(tokens);
  final out = NativeTensorBuffer.float32([tokenValues.length, dim]);
  try {
    _copyRowsClamped(
      tokens: tokenValues,
      table: inputEmbedding.data,
      vocabSize: vocabSize,
      dim: dim,
      out: out.asFloat32List(),
    );
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyFlowEmbedTokenParts({
  required Object promptTokens,
  required Object generatedTokens,
  int generatedTokenOffset = 0,
  int? generatedTokenCount,
  required NpyArray inputEmbedding,
  required int vocabSize,
  required int dim,
}) {
  final promptValues = _int32Values(promptTokens);
  final generatedValues = _int32Values(generatedTokens);
  final selectedGeneratedTokenCount = _checkedGeneratedTokenSliceLength(
    generatedSourceTokenCount: generatedValues.length,
    generatedTokenOffset: generatedTokenOffset,
    generatedTokenCount: generatedTokenCount,
  );
  final tokenCount = promptValues.length + selectedGeneratedTokenCount;
  if (tokenCount == 0) {
    throw ArgumentError('at least one flow token is required');
  }
  final out = NativeTensorBuffer.float32([tokenCount, dim]);
  try {
    final outValues = out.asFloat32List();
    var offset = 0;
    offset = _copyRowsClamped(
      tokens: promptValues,
      table: inputEmbedding.data,
      vocabSize: vocabSize,
      dim: dim,
      out: outValues,
      outOffset: offset,
    );
    _copyRowsClamped(
      tokens: Int32List.sublistView(
        generatedValues,
        generatedTokenOffset,
        generatedTokenOffset + selectedGeneratedTokenCount,
      ),
      table: inputEmbedding.data,
      vocabSize: vocabSize,
      dim: dim,
      out: outValues,
      outOffset: offset,
    );
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

int _checkedGeneratedTokenSliceLength({
  required int generatedSourceTokenCount,
  required int generatedTokenOffset,
  required int? generatedTokenCount,
}) {
  if (generatedTokenOffset < 0 ||
      generatedTokenOffset > generatedSourceTokenCount) {
    throw RangeError.range(
      generatedTokenOffset,
      0,
      generatedSourceTokenCount,
      'generatedTokenOffset',
    );
  }
  final selectedGeneratedTokenCount =
      generatedTokenCount ?? generatedSourceTokenCount - generatedTokenOffset;
  if (selectedGeneratedTokenCount < 0 ||
      selectedGeneratedTokenCount >
          generatedSourceTokenCount - generatedTokenOffset) {
    throw RangeError.range(
      selectedGeneratedTokenCount,
      0,
      generatedSourceTokenCount - generatedTokenOffset,
      'generatedTokenCount',
    );
  }
  return selectedGeneratedTokenCount;
}

NativeTensorBuffer cosyFlowEmbedOneToken({
  required int token,
  required NpyArray inputEmbedding,
  required int vocabSize,
  required int dim,
}) {
  final out = NativeTensorBuffer.float32([1, dim]);
  try {
    _copyOneRowClamped(
      token: token,
      table: inputEmbedding.data,
      vocabSize: vocabSize,
      dim: dim,
      out: out.asFloat32List(),
    );
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

void cosyFlowEmbedOneTokenInto({
  required int token,
  required NpyArray inputEmbedding,
  required int vocabSize,
  required int dim,
  required NativeTensorBuffer out,
}) {
  _checkFloat32Buffer(out, 'out');
  final outLength = out.byteLength ~/ 4;
  if (outLength != dim) {
    throw StateError('out length is $outLength, expected $dim.');
  }
  _copyOneRowClamped(
    token: token,
    table: inputEmbedding.data,
    vocabSize: vocabSize,
    dim: dim,
    out: out.asFloat32List(),
  );
}

NativeTensorBuffer cosyResampleLinearBuffer(
  Object source, {
  required int srcRate,
  required int dstRate,
}) {
  if (srcRate <= 0) {
    throw RangeError.value(srcRate, 'srcRate', 'must be positive');
  }
  if (dstRate <= 0) {
    throw RangeError.value(dstRate, 'dstRate', 'must be positive');
  }
  final values = _float32Values(source);
  final outLength = srcRate == dstRate || values.length < 2
      ? values.length
      : (values.length / (srcRate / dstRate)).floor();
  final out = NativeTensorBuffer.float32([outLength]);
  try {
    final target = out.asFloat32List();
    if (srcRate == dstRate || values.length < 2) {
      target.setAll(0, values);
      return out;
    }
    final ratio = srcRate / dstRate;
    for (var i = 0; i < outLength; i += 1) {
      final srcPos = i * ratio;
      final srcIndex = srcPos.floor();
      final frac = srcPos - srcIndex;
      if (srcIndex + 1 >= values.length) {
        target[i] = values.last;
      } else {
        target[i] =
            values[srcIndex] * (1.0 - frac) + values[srcIndex + 1] * frac;
      }
    }
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyTransposeFloat32Buffer(
  Object source, {
  required int rows,
  required int cols,
}) {
  if (rows <= 0) {
    throw RangeError.value(rows, 'rows', 'must be positive');
  }
  if (cols <= 0) {
    throw RangeError.value(cols, 'cols', 'must be positive');
  }
  final values = _float32Values(source);
  final expectedLength = rows * cols;
  if (values.length != expectedLength) {
    throw StateError(
      'transpose source length is ${values.length}, expected $expectedLength.',
    );
  }
  final out = NativeTensorBuffer.float32([expectedLength]);
  try {
    final target = out.asFloat32List();
    for (var row = 0; row < rows; row += 1) {
      for (var col = 0; col < cols; col += 1) {
        target[col * rows + row] = values[row * cols + col];
      }
    }
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyCepstralMeanNormalizeBuffer(
  Float32List feat, {
  required int frames,
  required int melBins,
}) {
  if (frames < 0) {
    throw RangeError.value(frames, 'frames', 'must be non-negative');
  }
  if (melBins <= 0) {
    throw RangeError.value(melBins, 'melBins', 'must be positive');
  }
  final expectedLength = frames * melBins;
  if (feat.length != expectedLength) {
    throw StateError(
      'CMN input length is ${feat.length}, expected $expectedLength.',
    );
  }
  final out = NativeTensorBuffer.float32([expectedLength]);
  try {
    final target = out.asFloat32List();
    target.setAll(0, feat);
    _cepstralMeanNormalize(target, frames: frames, melBins: melBins);
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

final class CosyPromptClipBuffers {
  CosyPromptClipBuffers._({
    required this.feat,
    required this.tokens,
    required this.tokenLen,
    required this.melBins,
  });

  final NativeTensorBuffer feat;
  final NativeTensorBuffer tokens;
  final int tokenLen;
  final int melBins;

  int get featFrames => tokenLen * 2;

  void close() {
    tokens.close();
    feat.close();
  }
}

CosyPromptClipBuffers cosyClipPromptBuffers({
  required Object feat,
  required Object tokens,
  required int tokenLen,
  required int melBins,
}) {
  if (tokenLen < 0) {
    throw RangeError.value(tokenLen, 'tokenLen', 'must be non-negative');
  }
  if (melBins <= 0) {
    throw RangeError.value(melBins, 'melBins', 'must be positive');
  }
  final featValues = _float32Values(feat);
  final tokenValues = _int32Values(tokens);
  final clippedFeatLength = tokenLen * 2 * melBins;
  if (clippedFeatLength > featValues.length) {
    throw StateError(
      'prompt feature length is ${featValues.length}, expected at least '
      '$clippedFeatLength.',
    );
  }
  if (tokenLen > tokenValues.length) {
    throw StateError(
      'prompt token count is ${tokenValues.length}, expected at least '
      '$tokenLen.',
    );
  }
  final featOut = NativeTensorBuffer.float32([clippedFeatLength]);
  final tokensOut = NativeTensorBuffer.int32([tokenLen]);
  try {
    featOut.asFloat32List().setAll(
      0,
      Float32List.sublistView(featValues, 0, clippedFeatLength),
    );
    tokensOut.asInt32List().setAll(
      0,
      Int32List.sublistView(tokenValues, 0, tokenLen),
    );
    return CosyPromptClipBuffers._(
      feat: featOut,
      tokens: tokensOut,
      tokenLen: tokenLen,
      melBins: melBins,
    );
  } catch (_) {
    tokensOut.close();
    featOut.close();
    rethrow;
  }
}

final class CosyFlowTimeStepBuffer {
  CosyFlowTimeStepBuffer() : buffer = NativeTensorBuffer.float32([2]) {
    tensor = buffer.tensor;
  }

  final NativeTensorBuffer buffer;

  late final RuntimeTensor tensor;

  double setStep({required int step, required int totalSteps}) {
    if (step < 1) {
      throw RangeError.range(step, 1, null, 'step');
    }
    if (totalSteps < 1) {
      throw RangeError.range(totalSteps, 1, null, 'totalSteps');
    }
    if (step > totalSteps) {
      throw RangeError.range(step, 1, totalSteps, 'step');
    }
    final previous = _cosineSchedule(step - 1, totalSteps);
    final next = _cosineSchedule(step, totalSteps);
    final values = buffer.asFloat32List();
    values[0] = previous;
    values[1] = previous;
    return next - previous;
  }

  void close() {
    buffer.close();
  }
}

NativeTensorBuffer cosyLlmBuildPrefillEmbeddings({
  required Object textTokens,
  required Object promptSpeechTokens,
  required NpyArray textEmbedding,
  required NpyArray llmEmbedding,
  required NpyArray speechEmbedding,
  required int textVocabSize,
  required int speechVocabSize,
  required int dim,
}) {
  final textValues = _int32Values(textTokens);
  final promptValues = _int32Values(promptSpeechTokens);
  final seqLen = 2 + textValues.length + promptValues.length;
  final out = NativeTensorBuffer.float32([1, seqLen, dim]);
  try {
    _buildLlmPrefill(
      textTokens: textValues,
      promptSpeechTokens: promptValues,
      textTable: textEmbedding.data,
      llmTable: llmEmbedding.data,
      speechTable: speechEmbedding.data,
      textVocabSize: textVocabSize,
      speechVocabSize: speechVocabSize,
      dim: dim,
      out: out.asFloat32List(),
    );
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

CosyLlmPrefillTextPlan cosyLlmBuildPrefillEmbeddingsFromText({
  required Qwen2BpeTokenizer tokenizer,
  required String text,
  String promptText = '',
  required Object promptSpeechTokens,
  required NpyArray textEmbedding,
  required NpyArray llmEmbedding,
  required NpyArray speechEmbedding,
  required int textVocabSize,
  required int speechVocabSize,
  required int dim,
}) {
  final promptTextIds = promptText.isEmpty
      ? const <int>[]
      : tokenizer.encode(promptText);
  final targetTextIds = tokenizer.encode(text);
  final textIds = Int32List.fromList([...promptTextIds, ...targetTextIds]);
  final promptValues = _int32Values(promptSpeechTokens);
  final seqLen = 2 + textIds.length + promptValues.length;
  final out = NativeTensorBuffer.float32([1, seqLen, dim]);
  try {
    _buildLlmPrefill(
      textTokens: textIds,
      promptSpeechTokens: promptValues,
      textTable: textEmbedding.data,
      llmTable: llmEmbedding.data,
      speechTable: speechEmbedding.data,
      textVocabSize: textVocabSize,
      speechVocabSize: speechVocabSize,
      dim: dim,
      out: out.asFloat32List(),
    );
    return CosyLlmPrefillTextPlan(
      buffer: out,
      seqLen: seqLen,
      targetTextTokenCount: targetTextIds.length,
    );
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyLlmEmbedSpeechToken({
  required int token,
  required NpyArray speechEmbedding,
  required int speechVocabSize,
  required int dim,
}) {
  return cosyFlowEmbedOneToken(
    token: token,
    inputEmbedding: speechEmbedding,
    vocabSize: speechVocabSize,
    dim: dim,
  );
}

NativeTensorBuffer cosyLlmSpeechTokenBuffer({required int dim}) {
  if (dim <= 0) {
    throw RangeError.range(dim, 1, null, 'dim');
  }
  return NativeTensorBuffer.float32([1, 1, dim]);
}

void cosyLlmEmbedSpeechTokenInto({
  required int token,
  required NpyArray speechEmbedding,
  required int speechVocabSize,
  required int dim,
  required NativeTensorBuffer out,
}) {
  cosyFlowEmbedOneTokenInto(
    token: token,
    inputEmbedding: speechEmbedding,
    vocabSize: speechVocabSize,
    dim: dim,
    out: out,
  );
}

NativeTensorBuffer cosyLlmSliceLastHidden({
  required Object hidden,
  required int seqLen,
  required int dim,
}) {
  if (seqLen <= 0) {
    throw RangeError.range(seqLen, 1, null, 'seqLen');
  }
  if (dim <= 0) {
    throw RangeError.range(dim, 1, null, 'dim');
  }
  final values = _float32Values(hidden);
  final expectedLength = seqLen * dim;
  if (values.length != expectedLength) {
    throw StateError(
      'hidden length is ${values.length}, expected $expectedLength.',
    );
  }
  final out = NativeTensorBuffer.float32([1, 1, dim]);
  try {
    out.asFloat32List().setAll(
      0,
      Float32List.sublistView(values, expectedLength - dim, expectedLength),
    );
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyLlmAttentionMask(int seqLen) {
  final out = NativeTensorBuffer.int64([1, seqLen]);
  try {
    cosyLlmFillAttentionMask(out: out, seqLen: seqLen);
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

void cosyLlmFillAttentionMask({
  required NativeTensorBuffer out,
  required int seqLen,
}) {
  _checkInt64Buffer(out, 'attentionMask');
  if (seqLen < 1) {
    throw RangeError.range(seqLen, 1, null, 'seqLen');
  }
  final values = out.asInt64List();
  if (values.length < seqLen) {
    throw ArgumentError(
      'attentionMask length ${values.length} is smaller than seqLen $seqLen',
    );
  }
  values.fillRange(0, seqLen, 1);
}

final class CosyLlmAttentionMaskCache {
  CosyLlmAttentionMaskCache(int initialSeqLen) {
    if (initialSeqLen < 1) {
      throw RangeError.range(initialSeqLen, 1, null, 'initialSeqLen');
    }
    _capacity = _nextCapacity(initialSeqLen);
    _buffer = NativeTensorBuffer.int64([1, _capacity]);
    try {
      cosyLlmFillAttentionMask(out: _buffer, seqLen: _capacity);
    } catch (_) {
      _buffer.close();
      rethrow;
    }
  }

  late NativeTensorBuffer _buffer;
  late int _capacity;

  RuntimeTensor tensor(int seqLen) {
    if (seqLen < 1) {
      throw RangeError.range(seqLen, 1, null, 'seqLen');
    }
    _ensureCapacity(seqLen);
    return _buffer.tensorView(shape: [1, seqLen], byteLength: seqLen * 8);
  }

  void close() {
    _buffer.close();
  }

  void _ensureCapacity(int seqLen) {
    if (seqLen <= _capacity) {
      return;
    }
    final nextCapacity = _nextCapacity(seqLen);
    final next = NativeTensorBuffer.int64([1, nextCapacity]);
    try {
      cosyLlmFillAttentionMask(out: next, seqLen: nextCapacity);
    } catch (_) {
      next.close();
      rethrow;
    }
    final previous = _buffer;
    _buffer = next;
    _capacity = nextCapacity;
    previous.close();
  }

  static int _nextCapacity(int seqLen) {
    return cosyLlmAttentionCapacity(seqLen);
  }
}

NativeTensorBuffer cosyFlowProjectEncoder({
  required Object encoded,
  required int frames,
  required int dim,
  required NpyArray weight,
  required NpyArray bias,
  required int melBins,
}) {
  if (frames <= 0) {
    throw RangeError.range(frames, 1, null, 'frames');
  }
  if (dim <= 0) {
    throw RangeError.range(dim, 1, null, 'dim');
  }
  if (melBins <= 0) {
    throw RangeError.range(melBins, 1, null, 'melBins');
  }
  final encodedValues = _float32Values(encoded);
  final out = NativeTensorBuffer.float32([melBins, frames]);
  try {
    _projectEncoder(
      encoded: encodedValues,
      frames: frames,
      dim: dim,
      weight: weight.data,
      bias: bias.data,
      melBins: melBins,
      out: out.asFloat32List(),
    );
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyFlowSpeakerCondition({
  required Object? speakerEmbedding,
  required NpyArray weight,
  required NpyArray bias,
  required int melBins,
  required int speakerDim,
}) {
  if (melBins <= 0) {
    throw RangeError.range(melBins, 1, null, 'melBins');
  }
  if (speakerDim <= 0) {
    throw RangeError.range(speakerDim, 1, null, 'speakerDim');
  }
  final speakerValues = speakerEmbedding == null
      ? Float32List(0)
      : _float32Values(speakerEmbedding);
  final out = NativeTensorBuffer.float32([melBins]);
  try {
    _speakerCondition(
      speaker: speakerValues,
      weight: weight.data,
      bias: bias.data,
      melBins: melBins,
      speakerDim: speakerDim,
      out: out.asFloat32List(),
    );
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyFlowConditioning({
  required Object promptFeat,
  required int promptFrames,
  required int melBins,
  required int totalFrames,
}) {
  if (promptFrames < 0) {
    throw RangeError.range(promptFrames, 0, null, 'promptFrames');
  }
  if (melBins <= 0) {
    throw RangeError.range(melBins, 1, null, 'melBins');
  }
  if (totalFrames <= 0 || promptFrames > totalFrames) {
    throw RangeError.range(totalFrames, promptFrames, null, 'totalFrames');
  }
  final promptValues = _float32Values(promptFeat);
  final out = NativeTensorBuffer.float32([melBins, totalFrames]);
  try {
    final target = out.asFloat32List();
    if (promptValues.length < promptFrames * melBins) {
      throw StateError(
        'prompt feature length is ${promptValues.length}, expected at least '
        '${promptFrames * melBins}.',
      );
    }
    target.fillRange(0, target.length, 0);
    for (var frame = 0; frame < promptFrames; frame += 1) {
      for (var mel = 0; mel < melBins; mel += 1) {
        target[mel * totalFrames + frame] = promptValues[frame * melBins + mel];
      }
    }
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyFlowInitialNoise({
  required NpyArray randNoise,
  required int randFrames,
  required int melBins,
  required int frames,
}) {
  if (randFrames <= 0) {
    throw RangeError.range(randFrames, 1, null, 'randFrames');
  }
  if (melBins <= 0) {
    throw RangeError.range(melBins, 1, null, 'melBins');
  }
  if (frames <= 0 || frames > randFrames) {
    throw RangeError.range(frames, 1, randFrames, 'frames');
  }
  final source = randNoise.data;
  final expectedLength = melBins * randFrames;
  if (source.length != expectedLength) {
    throw StateError(
      'noise length is ${source.length}, expected $expectedLength.',
    );
  }
  final out = NativeTensorBuffer.float32([melBins, frames]);
  try {
    final target = out.asFloat32List();
    for (var mel = 0; mel < melBins; mel += 1) {
      target.setRange(
        mel * frames,
        mel * frames + frames,
        source,
        mel * randFrames,
      );
    }
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

NativeTensorBuffer cosyFlowSliceMel({
  required Object mel,
  required int melBins,
  required int frames,
  required int startFrame,
}) {
  if (melBins <= 0) {
    throw RangeError.range(melBins, 1, null, 'melBins');
  }
  if (frames <= 0) {
    throw RangeError.range(frames, 1, null, 'frames');
  }
  if (startFrame < 0 || startFrame > frames) {
    throw RangeError.range(startFrame, 0, frames, 'startFrame');
  }
  final values = _float32Values(mel);
  final outFrames = frames - startFrame;
  final expectedLength = melBins * frames;
  if (values.length != expectedLength) {
    throw StateError(
      'mel length is ${values.length}, expected $expectedLength.',
    );
  }
  final out = NativeTensorBuffer.float32([melBins, outFrames]);
  try {
    final target = out.asFloat32List();
    for (var bin = 0; bin < melBins; bin += 1) {
      target.setRange(
        bin * outFrames,
        bin * outFrames + outFrames,
        values,
        bin * frames + startFrame,
      );
    }
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

final class CosyFlowEstimatorInputs {
  CosyFlowEstimatorInputs._({
    required this.mask,
    required this.mu,
    required this.spk,
    required this.cond,
  }) : maskTensor = mask.tensor,
       muTensor = mu.tensor,
       spkTensor = spk.tensor,
       condTensor = cond.tensor;

  final NativeTensorBuffer mask;
  final NativeTensorBuffer mu;
  final NativeTensorBuffer spk;
  final NativeTensorBuffer cond;
  final RuntimeTensor maskTensor;
  final RuntimeTensor muTensor;
  final RuntimeTensor spkTensor;
  final RuntimeTensor condTensor;

  void close() {
    cond.close();
    spk.close();
    mu.close();
    mask.close();
  }
}

CosyFlowEstimatorInputs cosyFlowEstimatorStaticInputs({
  required NativeTensorBuffer mu,
  required NativeTensorBuffer spk,
  required NativeTensorBuffer cond,
  required int frames,
  required int melBins,
}) {
  _checkFloat32Buffer(mu, 'mu');
  _checkFloat32Buffer(spk, 'spk');
  _checkFloat32Buffer(cond, 'cond');
  if (frames <= 0) {
    throw RangeError.range(frames, 1, null, 'frames');
  }
  if (melBins <= 0) {
    throw RangeError.range(melBins, 1, null, 'melBins');
  }
  final half = frames * melBins;
  if (mu.byteLength ~/ 4 != half ||
      spk.byteLength ~/ 4 != melBins ||
      cond.byteLength ~/ 4 != half) {
    throw StateError('CosyVoice2 flow static input shapes do not match.');
  }
  final maskOut = NativeTensorBuffer.float32([2, 1, frames]);
  final muOut = NativeTensorBuffer.float32([2, melBins, frames]);
  final spkOut = NativeTensorBuffer.float32([2, melBins]);
  final condOut = NativeTensorBuffer.float32([2, melBins, frames]);
  try {
    maskOut.asFloat32List().fillRange(0, maskOut.byteLength ~/ 4, 1);
    muOut.asFloat32List().fillRange(0, muOut.byteLength ~/ 4, 0);
    spkOut.asFloat32List().fillRange(0, spkOut.byteLength ~/ 4, 0);
    condOut.asFloat32List().fillRange(0, condOut.byteLength ~/ 4, 0);
    muOut.asFloat32List().setRange(0, half, mu.asFloat32List());
    spkOut.asFloat32List().setRange(0, melBins, spk.asFloat32List());
    condOut.asFloat32List().setRange(0, half, cond.asFloat32List());
    return CosyFlowEstimatorInputs._(
      mask: maskOut,
      mu: muOut,
      spk: spkOut,
      cond: condOut,
    );
  } catch (_) {
    condOut.close();
    spkOut.close();
    muOut.close();
    maskOut.close();
    rethrow;
  }
}

NativeTensorBuffer cosyFlowDuplicateBatch(NativeTensorBuffer source) {
  final out = NativeTensorBuffer.float32([2, source.byteLength ~/ 4]);
  try {
    cosyFlowDuplicateBatchInto(source: source, out: out);
    return out;
  } catch (_) {
    out.close();
    rethrow;
  }
}

void cosyFlowDuplicateBatchInto({
  required NativeTensorBuffer source,
  required NativeTensorBuffer out,
}) {
  _checkFloat32Buffer(source, 'source');
  _checkFloat32Buffer(out, 'out');
  final sourceValues = source.asFloat32List();
  final outValues = out.asFloat32List();
  if (outValues.length != sourceValues.length * 2) {
    throw StateError(
      'duplicate output length is ${outValues.length}, expected '
      '${sourceValues.length * 2}.',
    );
  }
  outValues.setRange(0, sourceValues.length, sourceValues);
  outValues.setRange(
    sourceValues.length,
    sourceValues.length * 2,
    sourceValues,
  );
}

void cosyFlowApplyGuidance({
  required NativeTensorBuffer x,
  required Object estimator,
  required double dt,
  required double condScale,
}) {
  _checkFloat32Buffer(x, 'x');
  final values = x.asFloat32List();
  final estimatorValues = _float32Values(estimator);
  if (values.isEmpty) {
    throw StateError('CosyVoice2 flow x buffer is empty.');
  }
  if (estimatorValues.length < values.length * 2) {
    throw StateError('CosyVoice2 estimator buffer is too small.');
  }
  final uncondScale = 1.0 - condScale;
  for (var i = 0; i < values.length; i += 1) {
    final guided =
        condScale * estimatorValues[i] +
        uncondScale * estimatorValues[values.length + i];
    values[i] += dt * guided;
  }
}

int cosyRasNucleusSample({
  required Object logits,
  required int eosToken,
  required bool ignoreEos,
  required double topP,
  required int topK,
  required double randomDraw,
}) {
  return _nucleusSample(
    _float32Values(logits),
    eosToken: eosToken,
    ignoreEos: ignoreEos,
    topP: topP,
    topK: topK,
    randomDraw: randomDraw,
  );
}

final class CosyRasNucleusResult {
  const CosyRasNucleusResult({
    required this.token,
    required this.shouldFallback,
    this.appended = false,
  });

  final int token;
  final bool shouldFallback;
  final bool appended;
}

CosyRasNucleusResult cosyRasNucleusSampleWithRepetition({
  required Object logits,
  required NativeTensorBuffer history,
  required int historyLength,
  required int eosToken,
  required bool ignoreEos,
  required double topP,
  required int topK,
  required int winSize,
  required double tauR,
  required double randomDraw,
}) {
  _checkInt32History(history, historyLength);
  final token = cosyRasNucleusSample(
    logits: logits,
    eosToken: eosToken,
    ignoreEos: ignoreEos,
    topP: topP,
    topK: topK,
    randomDraw: randomDraw,
  );
  final shouldFallback = _shouldFallback(
    token,
    Int32List.sublistView(history.asInt32List(), 0, historyLength),
    winSize: winSize,
    tauR: tauR,
  );
  return CosyRasNucleusResult(token: token, shouldFallback: shouldFallback);
}

CosyRasNucleusResult cosyRasNucleusSampleWithRepetitionAppend({
  required Object logits,
  required NativeTensorBuffer history,
  required int historyLength,
  required int eosToken,
  required bool ignoreEos,
  required double topP,
  required int topK,
  required int winSize,
  required double tauR,
  required double randomDraw,
}) {
  _checkInt32History(history, historyLength);
  final result = cosyRasNucleusSampleWithRepetition(
    logits: logits,
    history: history,
    historyLength: historyLength,
    eosToken: eosToken,
    ignoreEos: ignoreEos,
    topP: topP,
    topK: topK,
    winSize: winSize,
    tauR: tauR,
    randomDraw: randomDraw,
  );
  final shouldAppend =
      result.token != eosToken && historyLength < history.asInt32List().length;
  if (shouldAppend) {
    history.asInt32List()[historyLength] = result.token;
  }
  return CosyRasNucleusResult(
    token: result.token,
    shouldFallback: result.shouldFallback,
    appended: shouldAppend,
  );
}

void cosyRasHistoryAppend({
  required NativeTensorBuffer history,
  required int historyLength,
  required int token,
}) {
  _checkInt32History(history, historyLength);
  final values = history.asInt32List();
  if (historyLength >= values.length) {
    throw StateError('CosyVoice2 RAS history buffer is full.');
  }
  values[historyLength] = token;
}

int cosyRasMultinomialSample({
  required Object logits,
  required int eosToken,
  required bool ignoreEos,
  required double randomDraw,
}) {
  return _multinomialSample(
    _float32Values(logits),
    eosToken: eosToken,
    ignoreEos: ignoreEos,
    randomDraw: randomDraw,
  );
}

void _checkInt64Buffer(NativeTensorBuffer buffer, String name) {
  if (buffer.dtype != RuntimeTensorDataType.int64) {
    throw StateError('Expected int64 $name, got ${buffer.dtype.name}.');
  }
}

void _checkFloat32Buffer(NativeTensorBuffer buffer, String name) {
  if (buffer.dtype != RuntimeTensorDataType.float32) {
    throw StateError('Expected float32 $name, got ${buffer.dtype.name}.');
  }
}

void _checkInt32History(NativeTensorBuffer history, int historyLength) {
  if (history.dtype != RuntimeTensorDataType.int32) {
    throw StateError(
      'Expected int32 history buffer, got ${history.dtype.name}.',
    );
  }
  final historyCapacity = history.byteLength ~/ 4;
  if (historyLength < 0 || historyLength > historyCapacity) {
    throw RangeError.range(historyLength, 0, historyCapacity, 'historyLength');
  }
}

Float32List _float32Values(Object source) {
  return withNativeFloat32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return Float32List(0);
    return Float32List.fromList(pointer.asTypedList(length));
  });
}

Int32List _int32Values(Object source) {
  return withNativeInt32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return Int32List(0);
    return Int32List.fromList(pointer.asTypedList(length));
  });
}

int _copyRowsClamped({
  required Int32List tokens,
  required Float32List table,
  required int vocabSize,
  required int dim,
  required Float32List out,
  int outOffset = 0,
}) {
  _validateTable(table, vocabSize, dim);
  final requiredLength = outOffset + tokens.length * dim;
  if (out.length < requiredLength) {
    throw StateError('embedding output buffer is too small.');
  }
  var offset = outOffset;
  for (final token in tokens) {
    _copyOneRowClamped(
      token: token,
      table: table,
      vocabSize: vocabSize,
      dim: dim,
      out: out,
      outOffset: offset,
    );
    offset += dim;
  }
  return offset;
}

void _copyOneRowClamped({
  required int token,
  required Float32List table,
  required int vocabSize,
  required int dim,
  required Float32List out,
  int outOffset = 0,
}) {
  _validateTable(table, vocabSize, dim);
  if (out.length < outOffset + dim) {
    throw StateError('embedding output buffer is too small.');
  }
  final row = token.clamp(0, vocabSize - 1).toInt();
  out.setRange(outOffset, outOffset + dim, table, row * dim);
}

void _validateTable(Float32List table, int rows, int dim) {
  if (rows <= 0) {
    throw RangeError.range(rows, 1, null, 'rows');
  }
  if (dim <= 0) {
    throw RangeError.range(dim, 1, null, 'dim');
  }
  final expectedLength = rows * dim;
  if (table.length != expectedLength) {
    throw StateError(
      'table length is ${table.length}, expected $expectedLength.',
    );
  }
}

void _cepstralMeanNormalize(
  Float32List feat, {
  required int frames,
  required int melBins,
}) {
  if (melBins <= 0) {
    throw RangeError.range(melBins, 1, null, 'melBins');
  }
  if (feat.length != frames * melBins) {
    throw StateError(
      'CMN input length is ${feat.length}, expected ${frames * melBins}.',
    );
  }
  if (frames == 0) return;
  for (var bin = 0; bin < melBins; bin += 1) {
    var sum = 0.0;
    for (var frame = 0; frame < frames; frame += 1) {
      sum += feat[frame * melBins + bin];
    }
    final mean = sum / frames;
    for (var frame = 0; frame < frames; frame += 1) {
      final index = frame * melBins + bin;
      feat[index] = feat[index] - mean;
    }
  }
}

double _cosineSchedule(int index, int totalSteps) {
  final phase = index / totalSteps * 0.5 * math.pi;
  return 1.0 - math.cos(phase);
}

void _buildLlmPrefill({
  required Int32List textTokens,
  required Int32List promptSpeechTokens,
  required Float32List textTable,
  required Float32List llmTable,
  required Float32List speechTable,
  required int textVocabSize,
  required int speechVocabSize,
  required int dim,
  required Float32List out,
}) {
  _validateTable(textTable, textVocabSize, dim);
  _validateTable(llmTable, 2, dim);
  _validateTable(speechTable, speechVocabSize, dim);
  final expectedLength =
      (2 + textTokens.length + promptSpeechTokens.length) * dim;
  if (out.length != expectedLength) {
    throw StateError(
      'prefill output length is ${out.length}, expected $expectedLength.',
    );
  }
  var offset = 0;
  out.setRange(offset, offset + dim, llmTable, 0);
  offset += dim;
  offset = _copyRowsClamped(
    tokens: textTokens,
    table: textTable,
    vocabSize: textVocabSize,
    dim: dim,
    out: out,
    outOffset: offset,
  );
  out.setRange(offset, offset + dim, llmTable, dim);
  offset += dim;
  _copyRowsClamped(
    tokens: promptSpeechTokens,
    table: speechTable,
    vocabSize: speechVocabSize,
    dim: dim,
    out: out,
    outOffset: offset,
  );
}

void _projectEncoder({
  required Float32List encoded,
  required int frames,
  required int dim,
  required Float32List weight,
  required Float32List bias,
  required int melBins,
  required Float32List out,
}) {
  if (encoded.length != frames * dim ||
      weight.length != melBins * dim ||
      bias.length != melBins ||
      out.length != melBins * frames) {
    throw StateError('CosyVoice2 flow projection shapes do not match.');
  }
  for (var frame = 0; frame < frames; frame += 1) {
    final inputOffset = frame * dim;
    for (var mel = 0; mel < melBins; mel += 1) {
      var sum = bias[mel];
      final weightOffset = mel * dim;
      for (var k = 0; k < dim; k += 1) {
        sum += encoded[inputOffset + k] * weight[weightOffset + k];
      }
      out[mel * frames + frame] = sum;
    }
  }
}

void _speakerCondition({
  required Float32List speaker,
  required Float32List weight,
  required Float32List bias,
  required int melBins,
  required int speakerDim,
  required Float32List out,
}) {
  if (weight.length != melBins * speakerDim ||
      bias.length != melBins ||
      out.length != melBins) {
    throw StateError('CosyVoice2 speaker projection shapes do not match.');
  }
  var norm = 0.0;
  for (final value in speaker) {
    norm += value * value;
  }
  final scale = norm > 0.0 ? 1.0 / math.sqrt(norm) : 0.0;
  final count = math.min(speaker.length, speakerDim);
  for (var mel = 0; mel < melBins; mel += 1) {
    var sum = bias[mel];
    final weightOffset = mel * speakerDim;
    for (var k = 0; k < count; k += 1) {
      sum += speaker[k] * scale * weight[weightOffset + k];
    }
    out[mel] = sum;
  }
}

final class _Candidate {
  _Candidate(this.id, this.logit);

  final int id;
  final double logit;
  double prob = 0;
}

int _nucleusSample(
  Float32List logits, {
  required int eosToken,
  required bool ignoreEos,
  required double topP,
  required int topK,
  required double randomDraw,
}) {
  if (logits.isEmpty ||
      eosToken < 0 ||
      eosToken >= logits.length ||
      topK <= 0 ||
      topP < 0 ||
      randomDraw < 0) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  if (topP == 0) {
    return _bestLogit(logits, eosToken: eosToken, ignoreEos: ignoreEos);
  }
  final candidates = _softmaxCandidates(
    logits,
    eosToken: eosToken,
    ignoreEos: ignoreEos,
  );
  candidates.sort((a, b) {
    final byProb = b.prob.compareTo(a.prob);
    return byProb != 0 ? byProb : a.id.compareTo(b.id);
  });
  final top = candidates.take(math.min(topK, candidates.length)).toList();
  return _sampleSortedProbTopP(top, topP: topP, randomDraw: randomDraw);
}

int _multinomialSample(
  Float32List logits, {
  required int eosToken,
  required bool ignoreEos,
  required double randomDraw,
}) {
  if (logits.isEmpty ||
      eosToken < 0 ||
      eosToken >= logits.length ||
      randomDraw < 0) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  final candidates = _softmaxCandidates(
    logits,
    eosToken: eosToken,
    ignoreEos: ignoreEos,
  );
  var cumulative = 0.0;
  final target = randomDraw >= 1.0 ? 0.9999999999999999 : randomDraw;
  var lastValid = candidates.last.id;
  for (final candidate in candidates) {
    lastValid = candidate.id;
    cumulative += candidate.prob;
    if (target < cumulative) return candidate.id;
  }
  return lastValid;
}

List<_Candidate> _softmaxCandidates(
  Float32List logits, {
  required int eosToken,
  required bool ignoreEos,
}) {
  final ids = [
    for (var id = 0; id <= eosToken; id += 1)
      if (!(ignoreEos && id == eosToken)) id,
  ];
  if (ids.isEmpty) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  var maxLogit = logits[ids.first].toDouble();
  for (final id in ids.skip(1)) {
    final logit = logits[id].toDouble();
    if (!logit.isFinite) {
      throw StateError('CosyVoice2 RAS helper received invalid input.');
    }
    if (logit > maxLogit) maxLogit = logit;
  }
  var sum = 0.0;
  final candidates = <_Candidate>[];
  for (final id in ids) {
    final logit = logits[id].toDouble();
    if (!logit.isFinite) {
      throw StateError('CosyVoice2 RAS helper received invalid input.');
    }
    final prob = math.exp(logit - maxLogit);
    sum += prob;
    candidates.add(_Candidate(id, logit)..prob = prob);
  }
  if (sum <= 0.0) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  for (final candidate in candidates) {
    candidate.prob /= sum;
  }
  return candidates;
}

int _sampleSortedProbTopP(
  List<_Candidate> candidates, {
  required double topP,
  required double randomDraw,
}) {
  final nucleusP = topP > 1.0 ? 1.0 : topP;
  var count = 0;
  var cumulative = 0.0;
  while (count < candidates.length && cumulative < nucleusP) {
    cumulative += candidates[count].prob;
    count += 1;
  }
  final selected = candidates.take(math.max(1, count)).toList();
  return _draw(selected, randomDraw);
}

int _draw(List<_Candidate> candidates, double randomDraw) {
  final targetDraw = randomDraw >= 1.0 ? 0.9999999999999999 : randomDraw;
  final sum = candidates.fold<double>(0, (value, item) => value + item.prob);
  final target = targetDraw * sum;
  var cumulative = 0.0;
  for (final candidate in candidates) {
    cumulative += candidate.prob;
    if (target < cumulative) return candidate.id;
  }
  return candidates.last.id;
}

int _bestLogit(
  Float32List logits, {
  required int eosToken,
  required bool ignoreEos,
}) {
  var found = false;
  var winnerId = 0;
  var winnerLogit = 0.0;
  for (var id = 0; id <= eosToken; id += 1) {
    if (ignoreEos && id == eosToken) continue;
    final value = logits[id].toDouble();
    if (!value.isFinite) {
      throw StateError('CosyVoice2 RAS helper received invalid input.');
    }
    if (!found ||
        value > winnerLogit ||
        (value == winnerLogit && id < winnerId)) {
      winnerId = id;
      winnerLogit = value;
      found = true;
    }
  }
  if (!found) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  return winnerId;
}

bool _shouldFallback(
  int candidate,
  Iterable<int> history, {
  required int winSize,
  required double tauR,
}) {
  if (winSize <= 0 || tauR < 0 || !tauR.isFinite) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  final items = history.toList(growable: false);
  final start = items.length > winSize ? items.length - winSize : 0;
  var repetitions = 0;
  for (var i = start; i < items.length; i += 1) {
    if (items[i] == candidate) repetitions += 1;
  }
  return repetitions >= winSize * tauR;
}
