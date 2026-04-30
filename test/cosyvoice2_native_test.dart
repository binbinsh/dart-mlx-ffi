import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';
import 'package:dart_inference/src/models/cosyvoice2/cosyvoice2_native.dart';
import 'package:dart_inference/src/models/cosyvoice2/qwen2_tokenizer.dart';
import 'package:dart_inference/src/models/kokoro/kokoro.dart';
import 'package:test/test.dart';

void main() {
  test('native CosyVoice2 flow embedding clamps token ids', () {
    final table = NpyArray(
      shape: const [3, 2],
      data: Float32List.fromList([1, 2, 3, 4, 5, 6]),
    );
    final out = cosyFlowEmbedTokens(
      tokens: const [-1, 1, 99],
      inputEmbedding: table,
      vocabSize: 3,
      dim: 2,
    );
    final parts = cosyFlowEmbedTokenParts(
      promptTokens: const [1],
      generatedTokens: const [-1, 99],
      inputEmbedding: table,
      vocabSize: 3,
      dim: 2,
    );
    final sliced = cosyFlowEmbedTokenParts(
      promptTokens: const [1],
      generatedTokens: const [-1, 0, 99],
      generatedTokenOffset: 2,
      generatedTokenCount: 1,
      inputEmbedding: table,
      vocabSize: 3,
      dim: 2,
    );
    final one = cosyFlowEmbedOneToken(
      token: 99,
      inputEmbedding: table,
      vocabSize: 3,
      dim: 2,
    );
    try {
      expect(out.asFloat32List(), [1, 2, 3, 4, 5, 6]);
      expect(parts.asFloat32List(), [3, 4, 1, 2, 5, 6]);
      expect(sliced.asFloat32List(), [3, 4, 5, 6]);
      expect(one.shape, [1, 2]);
      expect(one.asFloat32List(), [5, 6]);
    } finally {
      one.close();
      sliced.close();
      parts.close();
      out.close();
      table.close();
    }
  });

  test('native CosyVoice2 prompt preprocess helpers match Dart layouts', () {
    final audio = Float32List.fromList([0, 10, 20, 30]);
    final downsampled = cosyResampleLinearBuffer(audio, srcRate: 4, dstRate: 2);
    final upsampled = cosyResampleLinearBuffer(audio, srcRate: 2, dstRate: 4);
    final transposed = cosyTransposeFloat32Buffer(
      Float32List.fromList([1, 2, 3, 4, 5, 6]),
      rows: 2,
      cols: 3,
    );
    final cmn = cosyCepstralMeanNormalizeBuffer(
      Float32List.fromList([1, 2, 3, 4, 5, 6]),
      frames: 3,
      melBins: 2,
    );
    final clipped = cosyClipPromptBuffers(
      feat: transposed,
      tokens: Int32List.fromList([8, 9, 10]),
      tokenLen: 2,
      melBins: 1,
    );
    try {
      expect(downsampled.asFloat32List(), [0, 20]);
      expect(upsampled.asFloat32List(), [0, 5, 10, 15, 20, 25, 30, 30]);
      expect(transposed.asFloat32List(), [1, 4, 2, 5, 3, 6]);
      expect(cmn.asFloat32List(), [-2, -2, 0, 0, 2, 2]);
      expect(clipped.feat.asFloat32List(), [1, 4, 2, 5]);
      expect(clipped.tokens.asInt32List(), [8, 9]);
      expect(clipped.featFrames, 4);
    } finally {
      clipped.close();
      cmn.close();
      transposed.close();
      upsampled.close();
      downsampled.close();
    }
  });

  test('native CosyVoice2 flow helpers preserve Python layout contracts', () {
    final weight = NpyArray(
      shape: const [2, 2],
      data: Float32List.fromList([1, 10, 100, 1000]),
    );
    final bias = NpyArray(
      shape: const [2],
      data: Float32List.fromList([0.5, -0.5]),
    );
    final encoded = Float32List.fromList([1, 2, 3, 4]);
    final projected = cosyFlowProjectEncoder(
      encoded: encoded,
      frames: 2,
      dim: 2,
      weight: weight,
      bias: bias,
      melBins: 2,
    );
    final speaker = cosyFlowSpeakerCondition(
      speakerEmbedding: Float32List.fromList([3, 4]),
      weight: weight,
      bias: bias,
      melBins: 2,
      speakerDim: 2,
    );
    final cond = cosyFlowConditioning(
      promptFeat: Float32List.fromList([10, 11, 20, 21]),
      promptFrames: 2,
      melBins: 2,
      totalFrames: 3,
    );
    final randNoise = NpyArray(
      shape: const [1, 2, 4],
      data: Float32List.fromList([1, 2, 3, 4, 10, 20, 30, 40]),
    );
    final noise = cosyFlowInitialNoise(
      randNoise: randNoise,
      randFrames: 4,
      melBins: 2,
      frames: 3,
    );
    final sliced = cosyFlowSliceMel(
      mel: Float32List.fromList([1, 2, 3, 4, 10, 20, 30, 40]),
      melBins: 2,
      frames: 4,
      startFrame: 1,
    );
    final timeStep = CosyFlowTimeStepBuffer();
    try {
      expect(projected.asFloat32List(), [21.5, 43.5, 2099.5, 4299.5]);
      final speakerValues = speaker.asFloat32List();
      expect(speakerValues[0], closeTo(9.1, 1e-5));
      expect(speakerValues[1], closeTo(859.5, 1e-4));
      expect(cond.asFloat32List(), [10, 20, 0, 11, 21, 0]);
      expect(noise.asFloat32List(), [1, 2, 3, 10, 20, 30]);
      expect(sliced.asFloat32List(), [2, 3, 4, 20, 30, 40]);
      final dt = timeStep.setStep(step: 1, totalSteps: 10);
      expect(timeStep.buffer.asFloat32List(), [0, 0]);
      expect(dt, closeTo(0.0123116594, 1e-7));
    } finally {
      timeStep.close();
      sliced.close();
      noise.close();
      randNoise.close();
      cond.close();
      speaker.close();
      projected.close();
      bias.close();
      weight.close();
    }
  });

  test('native CosyVoice2 LLM prefill helper composes embedding rows', () {
    final textEmbedding = NpyArray(
      shape: const [3, 2],
      data: Float32List.fromList([1, 2, 3, 4, 5, 6]),
    );
    final llmEmbedding = NpyArray(
      shape: const [2, 2],
      data: Float32List.fromList([10, 20, 30, 40]),
    );
    final speechEmbedding = NpyArray(
      shape: const [3, 2],
      data: Float32List.fromList([100, 200, 300, 400, 500, 600]),
    );
    final prefill = cosyLlmBuildPrefillEmbeddings(
      textTokens: const [-1, 1, 99],
      promptSpeechTokens: const [2, 99],
      textEmbedding: textEmbedding,
      llmEmbedding: llmEmbedding,
      speechEmbedding: speechEmbedding,
      textVocabSize: 3,
      speechVocabSize: 3,
      dim: 2,
    );
    final speech = cosyLlmEmbedSpeechToken(
      token: 99,
      speechEmbedding: speechEmbedding,
      speechVocabSize: 3,
      dim: 2,
    );
    final reusableSpeech = cosyLlmSpeechTokenBuffer(dim: 2);
    cosyLlmEmbedSpeechTokenInto(
      token: 1,
      speechEmbedding: speechEmbedding,
      speechVocabSize: 3,
      dim: 2,
      out: reusableSpeech,
    );
    final lastHidden = cosyLlmSliceLastHidden(
      hidden: Float32List.fromList([1, 2, 3, 4, 5, 6]),
      seqLen: 3,
      dim: 2,
    );
    final mask = cosyLlmAttentionMask(4);
    final maskCache = CosyLlmAttentionMaskCache(2);
    final decodePlan = cosyLlmDecodeStepPlan(
      pastSeq: 4,
      hiddenDim: 2,
      layerCount: 3,
    );
    try {
      expect(prefill.shape, [1, 7, 2]);
      expect(prefill.asFloat32List(), [
        10,
        20,
        1,
        2,
        3,
        4,
        5,
        6,
        30,
        40,
        500,
        600,
        500,
        600,
      ]);
      expect(speech.asFloat32List(), [500, 600]);
      expect(reusableSpeech.asFloat32List(), [300, 400]);
      expect(lastHidden.shape, [1, 1, 2]);
      expect(lastHidden.asFloat32List(), [5, 6]);
      expect(mask.shape, [1, 4]);
      expect(mask.asInt64List(), [1, 1, 1, 1]);
      expect(cosyLlmAttentionCapacity(5), 8);
      expect(decodePlan.nextSeq, 5);
      expect(decodePlan.attentionCapacity, 8);
      expect(decodePlan.expectedEmbedFloats, 2);
      expect(decodePlan.kvTensorCount, 6);
      expect(maskCache.tensor(2).asInt64List(), [1, 1]);
      expect(maskCache.tensor(5).asInt64List(), [1, 1, 1, 1, 1]);
    } finally {
      maskCache.close();
      mask.close();
      lastHidden.close();
      reusableSpeech.close();
      speech.close();
      prefill.close();
      speechEmbedding.close();
      llmEmbedding.close();
      textEmbedding.close();
    }
  });

  test('native CosyVoice2 LLM prefill helper tokenizes in native', () {
    final tokenizer = Qwen2BpeTokenizer.fromSidecarBytes(
      utf8.encode(_tinyQwen2Sidecar()),
    );
    final textEmbedding = NpyArray(
      shape: const [64, 2],
      data: Float32List.fromList([
        for (var i = 0; i < 64; i += 1) ...[i * 10.0, i * 10.0 + 1],
      ]),
    );
    final llmEmbedding = NpyArray(
      shape: const [2, 2],
      data: Float32List.fromList([1000, 1001, 2000, 2001]),
    );
    final speechEmbedding = NpyArray(
      shape: const [3, 2],
      data: Float32List.fromList([3000, 3001, 4000, 4001, 5000, 5001]),
    );
    final plan = cosyLlmBuildPrefillEmbeddingsFromText(
      tokenizer: tokenizer,
      text: '!',
      promptSpeechTokens: const [2],
      textEmbedding: textEmbedding,
      llmEmbedding: llmEmbedding,
      speechEmbedding: speechEmbedding,
      textVocabSize: 64,
      speechVocabSize: 3,
      dim: 2,
    );
    try {
      expect(plan.seqLen, 4);
      expect(plan.targetTextTokenCount, 1);
      expect(plan.tensor.shape, [1, 4, 2]);
      expect(plan.tensor.asFloat32List(), [
        1000,
        1001,
        330,
        331,
        2000,
        2001,
        5000,
        5001,
      ]);
    } finally {
      plan.close();
      speechEmbedding.close();
      llmEmbedding.close();
      textEmbedding.close();
      tokenizer.close();
    }
  });

  test('native CosyVoice2 flow estimator helpers update diffusion state', () {
    final mu = NativeTensorBuffer.float32([2, 2]);
    final spk = NativeTensorBuffer.float32([2]);
    final cond = NativeTensorBuffer.float32([2, 2]);
    final x = NativeTensorBuffer.float32([2]);
    try {
      mu.asFloat32List().setAll(0, [21.5, 43.5, 2099.5, 4299.5]);
      spk.asFloat32List().setAll(0, [9.1, 859.5]);
      cond.asFloat32List().setAll(0, [10, 20, 0, 0]);
      x.asFloat32List().setAll(0, [1, 2]);

      final staticInputs = cosyFlowEstimatorStaticInputs(
        mu: mu,
        spk: spk,
        cond: cond,
        frames: 2,
        melBins: 2,
      );
      final duplicated = cosyFlowDuplicateBatch(x);
      final reusedDup = NativeTensorBuffer.float32([2, 2]);
      cosyFlowDuplicateBatchInto(source: x, out: reusedDup);
      try {
        expect(staticInputs.mask.asFloat32List(), [1, 1, 1, 1]);
        expect(staticInputs.mu.asFloat32List(), [
          21.5,
          43.5,
          2099.5,
          4299.5,
          0,
          0,
          0,
          0,
        ]);
        final spkValues = staticInputs.spk.asFloat32List();
        expect(spkValues[0], closeTo(9.1, 1e-5));
        expect(spkValues[1], closeTo(859.5, 1e-5));
        expect(spkValues[2], 0);
        expect(spkValues[3], 0);
        expect(staticInputs.cond.asFloat32List(), [10, 20, 0, 0, 0, 0, 0, 0]);
        expect(duplicated.asFloat32List(), [1, 2, 1, 2]);
        expect(reusedDup.asFloat32List(), [1, 2, 1, 2]);

        cosyFlowApplyGuidance(
          x: x,
          estimator: Float32List.fromList([10, 20, 1, 2]),
          dt: 0.5,
          condScale: 1.7,
        );
        final values = x.asFloat32List();
        expect(values[0], closeTo(9.15, 1e-5));
        expect(values[1], closeTo(18.3, 1e-5));
      } finally {
        reusedDup.close();
        duplicated.close();
        staticInputs.close();
      }
    } finally {
      x.close();
      cond.close();
      spk.close();
      mu.close();
    }
  });
}

String _tinyQwen2Sidecar() {
  return [
    'qwen2bpe\t1',
    'meta\tdeclared_vocab_size\t64',
    'v\t33\t21',
    '',
  ].join('\n');
}
