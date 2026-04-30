import 'dart:typed_data';

import '../../runtime/onnx.dart';
import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/runtime.dart' show RuntimeTensor, RuntimeTensorDataType;
import '../kokoro/kokoro.dart' show NpyArray, loadNpz;
import 'cosyvoice2.dart';
import 'cosyvoice2_audio.dart';
import 'cosyvoice2_native.dart';
import 'cosyvoice2_speaker_prompt.dart';

const int cosyvoice2FlowTokenVocabSize = 6561;
const int cosyvoice2FlowInputDim = 512;
const int cosyvoice2MelBins = 80;
const int cosyvoice2SpeakerEmbeddingDim = 192;
const int cosyvoice2FlowRandFrames = 15000;
const int cosyvoice2SampleRate = 24000;
const int cosyvoice2HiftHopLength = 480;

final class CosyVoice2FlowOutput {
  const CosyVoice2FlowOutput({
    required this.mel,
    required this.melFrames,
    required this.audio,
    required this.audioSampleCount,
    required this.audioWavBytes,
    required this.prepareElapsedMicroseconds,
    required this.encoderElapsedMicroseconds,
    required this.setupElapsedMicroseconds,
    required this.diffuseElapsedMicroseconds,
    required this.vocodeElapsedMicroseconds,
  });

  final Float32List mel;
  final int melFrames;
  final Float32List audio;
  final int audioSampleCount;
  final Uint8List audioWavBytes;
  final int prepareElapsedMicroseconds;
  final int encoderElapsedMicroseconds;
  final int setupElapsedMicroseconds;
  final int diffuseElapsedMicroseconds;
  final int vocodeElapsedMicroseconds;
}

final class CosyVoice2FlowDriver {
  CosyVoice2FlowDriver._({
    required CosyVoice2PartialOnnxBundle bundle,
    required this.inputEmbedding,
    required this.encoderProjWeight,
    required this.encoderProjBias,
    required this.spkAffineWeight,
    required this.spkAffineBias,
    required this.randNoise,
    required this.diffusionSteps,
  }) : _bundle = bundle;

  final CosyVoice2PartialOnnxBundle _bundle;
  final NativeTensorBuffer _emptyFloat = NativeTensorBuffer.float32([0]);
  final NpyArray inputEmbedding;
  final NpyArray encoderProjWeight;
  final NpyArray encoderProjBias;
  final NpyArray spkAffineWeight;
  final NpyArray spkAffineBias;
  final NpyArray randNoise;
  final int diffusionSteps;

  static Future<CosyVoice2FlowDriver> load({
    required CosyVoice2PartialOnnxBundle bundle,
    required CosyVoice2Paths paths,
    int diffusionSteps = 10,
  }) async {
    if (diffusionSteps < 1) {
      throw RangeError.range(diffusionSteps, 1, null, 'diffusionSteps');
    }
    bundle.requireLoadedComponent('flow_encoder_fp32');
    if (bundle.loadedComponent('flow_decoder_loop_fp32') == null &&
        bundle.loadedComponent('flow_decoder_step_fp32') == null &&
        bundle.loadedComponent('flow_decoder_estimator_fp32') == null) {
      throw StateError(
        'CosyVoice2 ONNX component is not loaded: '
        'flow_decoder_loop_fp32, flow_decoder_step_fp32, or '
        'flow_decoder_estimator_fp32',
      );
    }
    bundle.requireLoadedComponent('hift');
    final support = await loadNpz(paths.flowSupportNpz);
    final taken = <NpyArray>[];
    NpyArray take(String name) {
      final value = support.remove(name);
      if (value == null) {
        for (final item in support.values) {
          item.close();
        }
        throw StateError(
          'flow_support.npz is missing "$name" (have ${support.keys.toList()})',
        );
      }
      taken.add(value);
      return value;
    }

    try {
      final inputEmbedding = take('input_embedding');
      final encoderProjWeight = take('encoder_proj_weight');
      final encoderProjBias = take('encoder_proj_bias');
      final spkAffineWeight = take('spk_embed_affine_weight');
      final spkAffineBias = take('spk_embed_affine_bias');
      final randNoise = take('rand_noise');
      for (final item in support.values) {
        item.close();
      }
      _expectShape(inputEmbedding.shape, [
        cosyvoice2FlowTokenVocabSize,
        cosyvoice2FlowInputDim,
      ], 'input_embedding');
      _expectShape(encoderProjWeight.shape, [
        cosyvoice2MelBins,
        cosyvoice2FlowInputDim,
      ], 'encoder_proj_weight');
      _expectShape(encoderProjBias.shape, [
        cosyvoice2MelBins,
      ], 'encoder_proj_bias');
      _expectShape(spkAffineWeight.shape, [
        cosyvoice2MelBins,
        cosyvoice2SpeakerEmbeddingDim,
      ], 'spk_embed_affine_weight');
      _expectShape(spkAffineBias.shape, [
        cosyvoice2MelBins,
      ], 'spk_embed_affine_bias');
      _expectShape(randNoise.shape, [
        1,
        cosyvoice2MelBins,
        cosyvoice2FlowRandFrames,
      ], 'rand_noise');
      return CosyVoice2FlowDriver._(
        bundle: bundle,
        inputEmbedding: inputEmbedding,
        encoderProjWeight: encoderProjWeight,
        encoderProjBias: encoderProjBias,
        spkAffineWeight: spkAffineWeight,
        spkAffineBias: spkAffineBias,
        randNoise: randNoise,
        diffusionSteps: diffusionSteps,
      );
    } catch (_) {
      for (final item in taken) {
        item.close();
      }
      for (final item in support.values) {
        item.close();
      }
      rethrow;
    }
  }

  CosyVoice2FlowOutput synthesizeTokens({
    required SpeakerPrompt? prompt,
    required Object generatedSpeechTokens,
    int generatedTokenOffset = 0,
    int? generatedTokenCount,
    bool useStreamingHift = false,
    bool includeFloatOutputs = true,
  }) {
    final generatedSourceTokenCount = _int32TokenLength(generatedSpeechTokens);
    final selectedGeneratedTokenCount = _int32TokenSliceLength(
      generatedSourceTokenCount: generatedSourceTokenCount,
      generatedTokenOffset: generatedTokenOffset,
      generatedTokenCount: generatedTokenCount,
    );
    if (selectedGeneratedTokenCount == 0) {
      throw ArgumentError('generatedSpeechTokens must not be empty');
    }
    final promptTokens = prompt?.promptSpeechTokensSource ?? Int32List(0);
    final promptFrames = prompt?.promptSpeechFeatFrames ?? 0;
    final promptTokenCount = prompt?.promptSpeechTokenCount ?? 0;
    final tokenCount = promptTokenCount + selectedGeneratedTokenCount;
    final totalFrames = tokenCount * 2;
    if (promptFrames > totalFrames) {
      throw StateError(
        'prompt frames $promptFrames exceed flow frames $totalFrames',
      );
    }
    if (totalFrames > cosyvoice2FlowRandFrames) {
      throw StateError(
        'CosyVoice2 flow request needs $totalFrames mel frames, '
        'but flow_support.npz rand_noise only has $cosyvoice2FlowRandFrames',
      );
    }

    final prepareTimer = Stopwatch()..start();
    final embeddings = _embedFlowTokenParts(
      promptTokens,
      generatedSpeechTokens,
      generatedTokenOffset: generatedTokenOffset,
      generatedTokenCount: selectedGeneratedTokenCount,
    );
    final xsLens = NativeTensorBuffer.int32([1]);
    xsLens.asInt32List()[0] = tokenCount;
    prepareTimer.stop();
    try {
      final encoderTimer = Stopwatch()..start();
      final encoderResult = _bundle.runComponent('flow_encoder_fp32', {
        'xs': embeddings.tensorView(
          shape: [1, tokenCount, cosyvoice2FlowInputDim],
          byteLength: embeddings.byteLength,
        ),
        'xs_lens': xsLens.tensor,
      });
      encoderTimer.stop();
      try {
        NativeTensorBuffer? mu;
        NativeTensorBuffer? spk;
        NativeTensorBuffer? cond;
        try {
          final setupTimer = Stopwatch()..start();
          mu = _projectEncoder(encoderResult, totalFrames);
          spk = _speakerCondition(prompt?.speakerEmbeddingSource);
          cond = _conditioning(prompt, totalFrames);
          setupTimer.stop();
          final diffuseTimer = Stopwatch()..start();
          final mel = _diffuse(
            mu: mu,
            spk: spk,
            cond: cond,
            frames: totalFrames,
          );
          diffuseTimer.stop();
          final generatedFrames = totalFrames - promptFrames;
          final generatedMel = _sliceMel(mel, totalFrames, promptFrames);
          try {
            final vocodeTimer = Stopwatch()..start();
            final audio = _vocode(
              generatedMel,
              generatedFrames,
              streaming: useStreamingHift,
              includeSamples: includeFloatOutputs,
            );
            vocodeTimer.stop();
            return CosyVoice2FlowOutput(
              mel: includeFloatOutputs
                  ? Float32List.fromList(generatedMel.asFloat32List())
                  : Float32List(0),
              melFrames: generatedFrames,
              audio: audio.samples,
              audioSampleCount: audio.sampleCount,
              audioWavBytes: audio.wavBytes,
              prepareElapsedMicroseconds: prepareTimer.elapsedMicroseconds,
              encoderElapsedMicroseconds: encoderTimer.elapsedMicroseconds,
              setupElapsedMicroseconds: setupTimer.elapsedMicroseconds,
              diffuseElapsedMicroseconds: diffuseTimer.elapsedMicroseconds,
              vocodeElapsedMicroseconds: vocodeTimer.elapsedMicroseconds,
            );
          } finally {
            generatedMel.close();
            mel.close();
          }
        } finally {
          cond?.close();
          spk?.close();
          mu?.close();
        }
      } finally {
        encoderResult.close();
      }
    } finally {
      xsLens.close();
      embeddings.close();
    }
  }

  void close() {
    inputEmbedding.close();
    encoderProjWeight.close();
    encoderProjBias.close();
    spkAffineWeight.close();
    spkAffineBias.close();
    randNoise.close();
    _emptyFloat.close();
  }

  NativeTensorBuffer _embedFlowTokenParts(
    Object promptTokens,
    Object generatedTokens, {
    required int generatedTokenOffset,
    required int generatedTokenCount,
  }) {
    return cosyFlowEmbedTokenParts(
      promptTokens: promptTokens,
      generatedTokens: generatedTokens,
      generatedTokenOffset: generatedTokenOffset,
      generatedTokenCount: generatedTokenCount,
      inputEmbedding: inputEmbedding,
      vocabSize: cosyvoice2FlowTokenVocabSize,
      dim: cosyvoice2FlowInputDim,
    );
  }

  NativeTensorBuffer _projectEncoder(
    DartOnnxResult result,
    int expectedFrames,
  ) {
    final value = result.outputs['encoder_out'];
    final length = _float32Length(value, 'encoder_out');
    if (length % cosyvoice2FlowInputDim != 0) {
      throw StateError(
        'flow_encoder_fp32 emitted $length values, not divisible by '
        '$cosyvoice2FlowInputDim',
      );
    }
    final frames = length ~/ cosyvoice2FlowInputDim;
    if (frames != expectedFrames) {
      throw StateError(
        'flow_encoder_fp32 emitted $frames frames, expected $expectedFrames',
      );
    }
    return cosyFlowProjectEncoder(
      encoded: value!,
      frames: frames,
      dim: cosyvoice2FlowInputDim,
      weight: encoderProjWeight,
      bias: encoderProjBias,
      melBins: cosyvoice2MelBins,
    );
  }

  NativeTensorBuffer _speakerCondition(Object? speakerEmbedding) {
    return cosyFlowSpeakerCondition(
      speakerEmbedding: speakerEmbedding,
      weight: spkAffineWeight,
      bias: spkAffineBias,
      melBins: cosyvoice2MelBins,
      speakerDim: cosyvoice2SpeakerEmbeddingDim,
    );
  }

  NativeTensorBuffer _conditioning(SpeakerPrompt? prompt, int frames) {
    return cosyFlowConditioning(
      promptFeat: prompt?.promptSpeechFeatSource ?? _emptyFloat,
      promptFrames: prompt?.promptSpeechFeatFrames ?? 0,
      melBins: cosyvoice2MelBins,
      totalFrames: frames,
    );
  }

  NativeTensorBuffer _diffuse({
    required NativeTensorBuffer mu,
    required NativeTensorBuffer spk,
    required NativeTensorBuffer cond,
    required int frames,
  }) {
    final fusedLoop = _bundle.loadedComponent('flow_decoder_loop_fp32');
    if (fusedLoop != null) {
      return _diffuseFusedLoop(
        mu: mu,
        spk: spk,
        cond: cond,
        frames: frames,
        loop: fusedLoop,
      );
    }
    final fusedStep = _bundle.loadedComponent('flow_decoder_step_fp32');
    if (fusedStep != null) {
      return _diffuseFusedStep(
        mu: mu,
        spk: spk,
        cond: cond,
        frames: frames,
        step: fusedStep,
        finalStep: _bundle.loadedComponent('flow_decoder_step_final_fp32'),
      );
    }
    return _diffuseEstimator(mu: mu, spk: spk, cond: cond, frames: frames);
  }

  NativeTensorBuffer _diffuseFusedLoop({
    required NativeTensorBuffer mu,
    required NativeTensorBuffer spk,
    required NativeTensorBuffer cond,
    required int frames,
    required CosyVoice2LoadedComponent loop,
  }) {
    final x = _initialNoise(frames);
    final staticInputs = cosyFlowEstimatorStaticInputs(
      mu: mu,
      spk: spk,
      cond: cond,
      frames: frames,
      melBins: cosyvoice2MelBins,
    );
    var keepX = false;
    try {
      final result = loop.run({
        'x': x.tensorView(
          shape: [1, cosyvoice2MelBins, frames],
          byteLength: x.byteLength,
        ),
        'mask': staticInputs.maskTensor,
        'mu': staticInputs.muTensor,
        'spks': staticInputs.spkTensor,
        'cond': staticInputs.condTensor,
      });
      try {
        final next = result.outputs['next_x'];
        final nextLength = _float32Length(next, 'next_x');
        final expectedLength = x.byteLength ~/ 4;
        if (nextLength != expectedLength) {
          throw StateError(
            'flow_decoder_loop_fp32 emitted $nextLength values, '
            'expected $expectedLength',
          );
        }
        x.copyFrom(_readFloat32Value(next, 'next_x'));
        keepX = true;
        return x;
      } finally {
        result.close();
      }
    } finally {
      staticInputs.close();
      if (!keepX) {
        x.close();
      }
    }
  }

  NativeTensorBuffer _diffuseEstimator({
    required NativeTensorBuffer mu,
    required NativeTensorBuffer spk,
    required NativeTensorBuffer cond,
    required int frames,
  }) {
    final x = _initialNoise(frames);
    final staticInputs = cosyFlowEstimatorStaticInputs(
      mu: mu,
      spk: spk,
      cond: cond,
      frames: frames,
      melBins: cosyvoice2MelBins,
    );
    var keepX = false;
    final timeStep = CosyFlowTimeStepBuffer();
    final xIn = NativeTensorBuffer.float32([2, x.byteLength ~/ 4]);
    final xInTensor = xIn.tensorView(
      shape: [2, cosyvoice2MelBins, frames],
      byteLength: xIn.byteLength,
    );
    final estimatorInputs = <String, Object?>{
      'x': xInTensor,
      'mask': staticInputs.maskTensor,
      'mu': staticInputs.muTensor,
      't': timeStep.tensor,
      'spks': staticInputs.spkTensor,
      'cond': staticInputs.condTensor,
    };
    try {
      final totalSteps = diffusionSteps;
      for (var step = 1; step <= totalSteps; step += 1) {
        final dt = timeStep.setStep(step: step, totalSteps: totalSteps);
        cosyFlowDuplicateBatchInto(source: x, out: xIn);
        final result = _bundle.runComponent(
          'flow_decoder_estimator_fp32',
          estimatorInputs,
        );
        try {
          final estimator = result.outputs['estimator_out'];
          final estimatorLength = _float32Length(estimator, 'estimator_out');
          final expectedLength = (x.byteLength ~/ 4) * 2;
          if (estimatorLength < expectedLength) {
            throw StateError(
              'flow_decoder_estimator_fp32 emitted $estimatorLength values, '
              'expected at least $expectedLength',
            );
          }
          cosyFlowApplyGuidance(
            x: x,
            estimator: estimator!,
            dt: dt,
            condScale: 1.7,
          );
        } finally {
          result.close();
        }
      }
      keepX = true;
      return x;
    } finally {
      xIn.close();
      timeStep.close();
      staticInputs.close();
      if (!keepX) {
        x.close();
      }
    }
  }

  NativeTensorBuffer _diffuseFusedStep({
    required NativeTensorBuffer mu,
    required NativeTensorBuffer spk,
    required NativeTensorBuffer cond,
    required int frames,
    required CosyVoice2LoadedComponent step,
    required CosyVoice2LoadedComponent? finalStep,
  }) {
    final x = _initialNoise(frames);
    final staticInputs = cosyFlowEstimatorStaticInputs(
      mu: mu,
      spk: spk,
      cond: cond,
      frames: frames,
      melBins: cosyvoice2MelBins,
    );
    var keepX = false;
    final timeStep = CosyFlowTimeStepBuffer();
    final dt = NativeTensorBuffer.float32([1]);
    final dtValues = dt.asFloat32List();
    Object xValue = x.tensorView(
      shape: [1, cosyvoice2MelBins, frames],
      byteLength: x.byteLength,
    );
    DartOnnxResult? xOwner;
    final stepInputs = <String, Object?>{
      'x': xValue,
      'mask': staticInputs.maskTensor,
      'mu': staticInputs.muTensor,
      't': timeStep.tensor,
      'spks': staticInputs.spkTensor,
      'cond': staticInputs.condTensor,
      'dt': dt.tensor,
    };
    try {
      final totalSteps = diffusionSteps;
      for (var index = 1; index <= totalSteps; index += 1) {
        dtValues[0] = timeStep.setStep(step: index, totalSteps: totalSteps);
        stepInputs['x'] = xValue;
        final isFinalStep = index == totalSteps && finalStep != null;
        final runner = isFinalStep ? finalStep : step;
        final result = runner.run(stepInputs);
        try {
          final next = result.outputs['next_x'];
          final nextLength = _float32Length(next, 'next_x');
          final expectedLength = x.byteLength ~/ 4;
          if (nextLength != expectedLength) {
            throw StateError(
              'flow_decoder_step_fp32 emitted $nextLength values, '
              'expected $expectedLength',
            );
          }
          if (next is RuntimeTensor && next.isNativeHandle) {
            if (isFinalStep) {
              throw StateError(
                'flow_decoder_step_final_fp32 returned a native handle.',
              );
            }
            xOwner?.close();
            xOwner = result;
            xValue = next;
          } else {
            x.copyFrom(_readFloat32Value(next, 'next_x'));
            result.close();
            xOwner?.close();
            xOwner = null;
            xValue = x.tensorView(
              shape: [1, cosyvoice2MelBins, frames],
              byteLength: x.byteLength,
            );
          }
        } catch (_) {
          result.close();
          rethrow;
        }
      }
      keepX = true;
      return x;
    } finally {
      xOwner?.close();
      dt.close();
      timeStep.close();
      staticInputs.close();
      if (!keepX) {
        x.close();
      }
    }
  }

  NativeTensorBuffer _initialNoise(int frames) {
    return cosyFlowInitialNoise(
      randNoise: randNoise,
      randFrames: cosyvoice2FlowRandFrames,
      melBins: cosyvoice2MelBins,
      frames: frames,
    );
  }

  NativeTensorBuffer _sliceMel(
    NativeTensorBuffer mel,
    int frames,
    int startFrame,
  ) {
    return cosyFlowSliceMel(
      mel: mel,
      melBins: cosyvoice2MelBins,
      frames: frames,
      startFrame: startFrame,
    );
  }

  ({Float32List samples, int sampleCount, Uint8List wavBytes}) _vocode(
    NativeTensorBuffer mel,
    int frames, {
    required bool streaming,
    required bool includeSamples,
  }) {
    final componentName =
        streaming && _bundle.loadedComponent('hift_streaming') != null
        ? 'hift_streaming'
        : 'hift';
    final inputs = <String, Object?>{
      'speech_feat': mel.tensorView(
        shape: [1, cosyvoice2MelBins, frames],
        byteLength: mel.byteLength,
      ),
    };
    if (componentName == 'hift_streaming') {
      inputs['cache_source'] = _emptyFloat.tensorView(
        shape: [1, 1, 0],
        byteLength: 0,
      );
    }
    final result = _bundle.runComponent(componentName, inputs);
    try {
      final audioValue = result.outputs['audio'];
      final rawSampleCount = _float32Length(audioValue, 'audio');
      final expectedSampleCount = frames * cosyvoice2HiftHopLength;
      final sampleCount = rawSampleCount > expectedSampleCount
          ? expectedSampleCount
          : rawSampleCount;
      final audioSource = _float32AudioSource(audioValue, 'audio');
      final wavBytes = encodeWavPcm16Source(
        audioSource,
        sampleRate: cosyvoice2SampleRate,
        sampleCount: sampleCount,
      );
      final samples = includeSamples
          ? copyFloat32Prefix(audioSource, sampleCount)
          : Float32List(0);
      return (samples: samples, sampleCount: sampleCount, wavBytes: wavBytes);
    } finally {
      result.close();
    }
  }
}

void _expectShape(List<int> got, List<int> want, String name) {
  if (got.length != want.length) {
    throw StateError('$name: expected rank ${want.length}, got ${got.length}');
  }
  for (var i = 0; i < got.length; i += 1) {
    if (got[i] != want[i]) {
      throw StateError('$name: expected shape $want, got $got');
    }
  }
}

int _int32TokenLength(Object value) {
  if (value is NativeTensorBuffer) {
    if (value.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tokens, got ${value.dtype.name}.');
    }
    return value.byteLength ~/ 4;
  }
  if (value is RuntimeTensor) {
    if (value.dtype != RuntimeTensorDataType.int32) {
      throw StateError('Expected int32 tokens, got ${value.dtype.name}.');
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

int _int32TokenSliceLength({
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

Float32List _readFloat32Value(Object? value, String name) {
  if (value is Float32List) {
    return value;
  }
  if (value is RuntimeTensor) {
    return value.asFloat32List();
  }
  if (value is List<double>) {
    return Float32List.fromList(value);
  }
  throw StateError('output "$name" has unexpected type ${value.runtimeType}');
}

Object _float32AudioSource(Object? value, String name) {
  if (value is Float32List || value is RuntimeTensor) {
    return value!;
  }
  if (value is List<double>) {
    return Float32List.fromList(value);
  }
  throw StateError('output "$name" has unexpected type ${value.runtimeType}');
}

int _float32Length(Object? value, String name) {
  if (value is Float32List) {
    return value.length;
  }
  if (value is RuntimeTensor) {
    if (value.dtype != RuntimeTensorDataType.float32) {
      throw StateError('output "$name" has dtype ${value.dtype.name}');
    }
    return value.byteLength ~/ 4;
  }
  if (value is List<double>) {
    return value.length;
  }
  throw StateError('output "$name" has unexpected type ${value.runtimeType}');
}
