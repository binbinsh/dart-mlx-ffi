/// Machine-readable model manifest.
///
/// A [ModelManifest] is a JSON-serialisable registry of all model families
/// shipped with (or supported by) this package.  It is consumed by
/// benchmarks, examples, documentation generators, and CI tooling.
library;

import 'dart:convert';

import 'model_spec.dart';
import 'runtime_metadata.dart';

// ---------------------------------------------------------------------------
// ModelManifest
// ---------------------------------------------------------------------------

/// A collection of [ModelSpec] entries with JSON round-trip support.
///
/// ```dart
/// final manifest = ModelManifest.builtIn();
/// final json = manifest.toJsonString();
/// final restored = ModelManifest.fromJson(jsonDecode(json));
/// ```
final class ModelManifest {
  const ModelManifest(this.models);

  /// All registered model specs.
  final List<ModelSpec> models;

  /// Built-in manifest containing every model family in this package.
  factory ModelManifest.builtIn() => ModelManifest(builtInSpecs);

  /// Look up a spec by its [ModelSpec.id].  Returns `null` if not found.
  ModelSpec? operator [](String id) {
    for (final spec in models) {
      if (spec.id == id) return spec;
    }
    return null;
  }

  /// All specs matching a given modality.
  List<ModelSpec> byModality(ModelModality modality) =>
      models.where((s) => s.modalities.contains(modality)).toList();

  /// Specs at a given support level.
  List<ModelSpec> bySupportLevel(SupportLevel level) =>
      models.where((s) => s.supportLevel == level).toList();

  /// Production-ready specs exposed by default model discovery.
  List<ModelSpec> get productionModels =>
      bySupportLevel(SupportLevel.production);

  /// Apply a runtime promotion patch produced by `benchmark/runtime/promote.py`.
  ///
  /// Unknown model ids are ignored so a patch can be generated for a broader
  /// model set than the current package exposes.
  ModelManifest withRuntimeValidation(Map<String, Object?> patch) {
    final patchModels = patch['models'];
    if (patchModels is! List<Object?>) return this;
    final updates = <String, Map<String, Object?>>{};
    for (final entry in patchModels) {
      final object = _objectMap(entry);
      final id = object['id'] as String?;
      if (id != null) updates[id] = object;
    }
    if (updates.isEmpty) return this;
    return ModelManifest([
      for (final spec in models) _applyRuntimePatch(spec, updates[spec.id]),
    ]);
  }

  /// Serialise to a JSON-friendly map.
  Map<String, Object?> toJson() => {
    'version': 1,
    'models': models.map((m) => m.toJson()).toList(),
  };

  /// Pretty-printed JSON string.
  String toJsonString() => const JsonEncoder.withIndent('  ').convert(toJson());

  /// Deserialise from a JSON map (as produced by [toJson]).
  factory ModelManifest.fromJson(Map<String, Object?> json) {
    final modelsJson = json['models'] as List<Object?>;
    final models = modelsJson.map((e) {
      final m = _objectMap(e);
      return ModelSpec(
        id: m['id'] as String,
        family: m['family'] as String,
        modalities: (m['modalities'] as List<Object?>)
            .whereType<String>()
            .map(
              (name) => ModelModality.values.firstWhere((v) => v.name == name),
            )
            .toList(),
        description: m['description'] as String? ?? '',
        version: m['version'] as String?,
        requiredFiles:
            (m['requiredFiles'] as List<Object?>?)
                ?.whereType<String>()
                .toList() ??
            const ['config.json'],
        optionalFiles:
            (m['optionalFiles'] as List<Object?>?)
                ?.whereType<String>()
                .toList() ??
            const <String>[],
        requiredTags:
            (m['requiredTags'] as List<Object?>?)
                ?.whereType<String>()
                .toList() ??
            const <String>[],
        sizeHint: (m['sizeHint'] as num?)?.toInt(),
        metadata: _objectMap(m['metadata']),
        supportLevel: parseSupportLevel(m['supportLevel'] as String?),
        platformArtifacts: _objectMap(m['platformArtifacts']).map((key, value) {
          final engine = RuntimeEngine.values.firstWhere(
            (candidate) => candidate.name == key,
            orElse: () => RuntimeEngine.mlx,
          );
          return MapEntry(engine, RuntimeArtifact.fromJson(_objectMap(value)));
        }),
        validationStatus: _objectMap(m['validationStatus']).map((key, value) {
          return MapEntry(
            key,
            RuntimeValidationStatus.fromJson(_objectMap(value)),
          );
        }),
      );
    }).toList();
    return ModelManifest(models);
  }
}

ModelSpec _applyRuntimePatch(ModelSpec spec, Map<String, Object?>? patch) {
  if (patch == null) return spec;
  final status = _objectMap(patch['validationStatus']).map((key, value) {
    return MapEntry(key, RuntimeValidationStatus.fromJson(_objectMap(value)));
  });
  final platformArtifacts = Map<RuntimeEngine, RuntimeArtifact>.from(
    spec.platformArtifacts,
  );
  final artifactPatch = _objectMap(patch['platformArtifacts']);
  for (final entry in artifactPatch.entries) {
    final engine = RuntimeEngine.values.firstWhere(
      (candidate) => candidate.name == entry.key,
      orElse: () => RuntimeEngine.mlx,
    );
    final artifact = RuntimeArtifact.fromJson(_objectMap(entry.value));
    platformArtifacts[engine] = artifact;
  }
  return spec.copyWith(
    supportLevel: patch.containsKey('supportLevel')
        ? parseSupportLevel(patch['supportLevel'] as String?)
        : spec.supportLevel,
    validationStatus: status.isEmpty ? spec.validationStatus : status,
    platformArtifacts: artifactPatch.isEmpty
        ? spec.platformArtifacts
        : platformArtifacts,
  );
}

Map<String, Object?> _objectMap(Object? value) {
  if (value is Map) {
    return Map<String, Object?>.from(value);
  }
  return const <String, Object?>{};
}

// ---------------------------------------------------------------------------
// Built-in model specs
// ---------------------------------------------------------------------------

/// Canonical list of every model family in this package.
///
/// Each model's runner should reference its spec from here.
final List<ModelSpec> builtInSpecs = [
  ModelSpec(
    id: 'qwen2_5',
    family: 'Qwen2.5',
    modalities: const [ModelModality.textGeneration],
    description: 'Qwen 2.5 text-generation LLM (quantised)',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'qwen2_5',
      sourceModel: 'Qwen/Qwen2.5-0.5B-Instruct',
      mlxRepo: 'mlx-community/Qwen2.5-0.5B-Instruct-4bit',
      mlxArtifact: '.',
      coremlRepo: 'finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit',
      coremlArtifact: 'Qwen2.5-0.5B-Instruct-4bit.mlmodelc',
      onnxRepo: 'onnx-community/Qwen2.5-0.5B-Instruct',
      onnxArtifact: 'onnx/model_q4f16.onnx',
      litertRepo: 'litert-community/Qwen2.5-0.5B-Instruct',
      litertArtifact: 'Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite',
    ),
  ),
  ModelSpec(
    id: 'qwen3_5',
    family: 'Qwen3.5',
    modalities: const [
      ModelModality.textGeneration,
      ModelModality.visionLanguage,
    ],
    description: 'Qwen 3.5 hybrid LLM with optional vision (Mamba + attention)',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'qwen3_5',
      sourceModel: 'Qwen/Qwen3.5-0.8B',
      mlxRepo: 'mlx-community/Qwen3.5-0.8B-4bit',
      mlxArtifact: '.',
      coremlRepo: 'mlboydaisuke/qwen3.5-0.8B-CoreML',
      coremlArtifact: 'qwen3_5_0_8b_decode_int8_mseq128.mlpackage',
      onnxRepo: 'onnx-community/Qwen3.5-0.8B-ONNX',
      onnxArtifact: 'onnx/decoder_model_merged_q4f16.onnx',
      litertRepo: 'Yoursmiling/Qwen3.5-0.8B-LiteRT',
      litertArtifact: 'model_multimodal.litertlm',
    ),
  ),
  ModelSpec(
    id: 'paddle_ocr_vl',
    family: 'PaddleOCR-VL',
    modalities: const [ModelModality.visionLanguage],
    description: 'ERNIE-4.5 based OCR vision-language model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'paddle_ocr_vl',
      sourceModel: 'PaddlePaddle/PaddleOCR-VL-1.5',
      mlxRepo: 'mlx-community/PaddleOCR-VL-1.5-8bit',
      mlxArtifact: '.',
      onnxRepo: 'lbm364dl/PaddleOCR-VL-1.5-ONNX',
      onnxArtifact: 'onnx/decoder_model_merged.onnx',
    ),
  ),
  ModelSpec(
    id: 'qwen3_asr',
    family: 'Qwen3-ASR',
    modalities: const [ModelModality.speechToText],
    description: 'Qwen3 automatic speech recognition',
    requiredFiles: const ['config.json'],
    optionalFiles: const [
      'vocab.json',
      'merges.txt',
      'tokenizer.json',
      'added_tokens.json',
    ],
    platformArtifacts: {
      ..._runtimeArtifacts(
        id: 'qwen3_asr',
        sourceModel: 'Qwen/Qwen3-ASR-1.7B',
        mlxRepo: 'mlx-community/Qwen3-ASR-1.7B-8bit',
        mlxArtifact: '.',
      ),
      RuntimeEngine.coreml: _artifact(
        engine: RuntimeEngine.coreml,
        repo: 'UniMocha/Qwen3-ASR-1.7B-CoreML-INT8',
        artifact: '.',
        modelId: 'qwen3_asr',
        sourceModel: 'Qwen/Qwen3-ASR-1.7B',
        format: 'coreml-asr-components',
        platforms: const ['ios', 'macos'],
        accelerators: const [Accelerator.ane, Accelerator.gpu, Accelerator.cpu],
        extraMetadata: const {
          'runtimeScope': 'model-level-coreml-stateful',
          'runner': 'Qwen3AsrCoreMlRunner.loadCoreMlBundle',
          'tokenizerRequired': true,
          'tokenizerSourceRepo': 'andrewleech/qwen3-asr-1.7b-onnx',
          'componentArtifacts': [
            'encoder.mlmodelc',
            'embedding.mlmodelc',
            'decoder.mlmodelc',
          ],
        },
      ),
      RuntimeEngine.onnx: _artifact(
        engine: RuntimeEngine.onnx,
        repo: 'andrewleech/qwen3-asr-1.7b-onnx',
        artifact: '.',
        modelId: 'qwen3_asr',
        sourceModel: 'Qwen/Qwen3-ASR-1.7B',
        format: 'onnx-asr-components',
        platforms: const ['linux', 'android'],
        accelerators: const [Accelerator.npu, Accelerator.gpu, Accelerator.cpu],
        extraMetadata: const {
          'runtimeScope': 'model-level-asr-components',
          'runner': 'Qwen3AsrNativeRunner.loadOnnxBundle',
          'componentArtifacts': [
            'encoder.int4.onnx',
            'decoder_init.int4.onnx',
            'decoder_step.int4.onnx',
            'embed_tokens.bin',
          ],
        },
      ),
    },
  ),
  ModelSpec(
    id: 'kitten_tts',
    family: 'KittenTTS',
    modalities: const [ModelModality.textToSpeech],
    description: 'Lightweight on-device text-to-speech engine',
    requiredFiles: const ['config.json'],
    platformArtifacts: _runtimeArtifacts(
      id: 'kitten_tts',
      sourceModel: 'KittenML/kitten-tts-mini-0.8',
      mlxRepo: 'mlx-community/kitten-tts-mini-0.8-8bit',
      mlxArtifact: '.',
      coremlRepo: 'alexwengg/kittentts-coreml',
      coremlArtifact: 'mini/kittentts_mini_5s.mlmodelc',
      onnxRepo: 'onnx-community/KittenTTS-Mini-v0.8-ONNX',
      onnxArtifact: 'onnx/model.onnx',
    ),
  ),
  ModelSpec(
    id: 'silero_vad',
    family: 'Silero VAD',
    modalities: const [ModelModality.voiceActivityDetection],
    description: 'Voice activity detection (Silero)',
    platformArtifacts: _runtimeArtifacts(
      id: 'silero_vad',
      sourceModel: 'snakers4/silero-vad',
      mlxRepo: 'aufklarer/Silero-VAD-v5-MLX',
      mlxArtifact: '.',
      coremlRepo: 'FluidInference/silero-vad-coreml',
      coremlArtifact: 'silero-vad-unified-v6.0.0.mlmodelc',
      onnxRepo: 'onnx-community/silero-vad',
      onnxArtifact: 'onnx/model.onnx',
    ),
  ),
  ModelSpec(
    id: 'qwen3_vl',
    family: 'Qwen3-VL',
    modalities: const [
      ModelModality.textGeneration,
      ModelModality.visionLanguage,
    ],
    description: 'Qwen3 vision-language model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'qwen3_vl',
      sourceModel: 'Qwen/Qwen3-VL-2B-Instruct',
      mlxRepo: 'mlx-community/Qwen3-VL-2B-Instruct-4bit',
      mlxArtifact: '.',
      coremlRepo: 'mlboydaisuke/qwen3-vl-2b-coreml',
      coremlArtifact: 'qwen3_vl_2b_decode_chunks',
      onnxRepo: 'onnx-community/Qwen3-VL-2B-Instruct-ONNX',
      onnxArtifact: 'onnx/decoder_model_merged_q4f16.onnx',
    ),
  ),
  ModelSpec(
    id: 'gemma4',
    family: 'Gemma 4',
    modalities: const [ModelModality.textGeneration],
    description: 'Gemma 4 E4B instruction-tuned text model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'gemma4',
      sourceModel: 'google/gemma-4-E4B-it',
      mlxRepo: 'mlx-community/gemma-4-e4b-it-4bit',
      mlxArtifact: '.',
      coremlRepo: 'mlboydaisuke/gemma-4-E4B-coreml',
      coremlArtifact: '.',
      onnxRepo: 'huggingworld/gemma-4-E4B-it-ONNX',
      onnxArtifact: 'onnx/decoder_model_merged_q4f16.onnx',
      litertRepo: 'litert-community/gemma-4-E4B-it-litert-lm',
      litertArtifact: 'gemma-4-E4B-it.litertlm',
    ),
  ),
  ModelSpec(
    id: 'function_gemma',
    family: 'FunctionGemma',
    modalities: const [ModelModality.textGeneration],
    description: 'Gemma function-calling model for structured tool output',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'function_gemma',
      sourceModel: 'google/functiongemma-270m-it',
      mlxRepo: 'mlx-community/functiongemma-270m-it-4bit',
      mlxArtifact: '.',
      coremlRepo: 'mlboydaisuke/functiongemma-270m-coreml',
      coremlArtifact: '.',
      onnxRepo: 'onnx-community/functiongemma-270m-it-ONNX-GQA',
      onnxArtifact: 'onnx/model_q4f16.onnx',
      litertRepo: 'litert-community/functiongemma-270m-ft-mobile-actions',
      litertArtifact: 'mobile_actions_q8_ekv1024.litertlm',
    ),
  ),
  ModelSpec(
    id: 'embedding_gemma',
    family: 'EmbeddingGemma',
    modalities: const [ModelModality.embedding],
    description: 'Gemma embedding model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'embedding_gemma',
      sourceModel: 'google/embeddinggemma-300m',
      mlxRepo: 'mlx-community/embeddinggemma-300m-4bit',
      mlxArtifact: '.',
      coremlRepo: 'mlboydaisuke/embeddinggemma-300m-coreml',
      coremlArtifact: 'encoder.mlmodelc',
      onnxRepo: 'onnx-community/embeddinggemma-300m-ONNX',
      onnxArtifact: 'onnx/model_q4f16.onnx',
      litertRepo: 'litert-community/embeddinggemma-300m',
      litertArtifact: 'embeddinggemma-300M_seq512_mixed-precision.tflite',
    ),
  ),
  ModelSpec(
    id: 'qwen3_6_27b',
    family: 'Qwen3.6 27B',
    modalities: const [ModelModality.textGeneration],
    description: 'Qwen3.6 27B publish-time MLX benchmark model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos'],
      blockedPlatforms: const {
        'linux':
            'ONNX Runtime GenAI conversion is wired and the config probe passes, but no validated 27B ONNX artifact has been exported yet.',
        'android':
            'LiteRT patched conversion is wired and the config probe passes, but no validated 27B LiteRT artifact has been exported yet.',
      },
    ),
    platformArtifacts: _runtimeArtifacts(
      id: 'qwen3_6_27b',
      sourceModel: 'Qwen/Qwen3.6-27B',
      mlxRepo: 'mlx-community/Qwen3.6-27B-4bit',
      mlxArtifact: '.',
    ),
  ),
  ModelSpec(
    id: 'translategemma_4b_it',
    family: 'TranslateGemma 4B IT',
    modalities: const [ModelModality.textGeneration],
    description: 'TranslateGemma 4B instruction-tuned translation model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'translategemma_4b_it',
      sourceModel: 'google/translategemma-4b-it',
      mlxRepo: 'mlx-community/translategemma-4b-it-4bit',
      mlxArtifact: '.',
      coremlRepo: 'Skyline23/translategemma-4b-it-coreml',
      coremlArtifact: 'StatefulTranslateGemma4BITInt4PerChannel.mlpackage',
      onnxRepo: 'onnx-community/translategemma-text-4b-it-ONNX',
      onnxArtifact: 'onnx/model_q4f16.onnx',
      litertRepo: 'litert-community/TranslateGemma-4B-IT',
      litertArtifact: 'translategemma-4b-it-int8-web.task',
    ),
  ),
  ModelSpec(
    id: 'minicpm_o_4_5',
    family: 'MiniCPM-o 4.5',
    modalities: const [
      ModelModality.textGeneration,
      ModelModality.visionLanguage,
      ModelModality.speechToText,
      ModelModality.textToSpeech,
    ],
    description:
        'MiniCPM-o 4.5 omni multimodal publish-time MLX benchmark model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos'],
      blockedPlatforms: const {
        'linux':
            'Only token2wav/CosyVoice ONNX components and a Core ML vision sidecar found; no full ONNX Runtime artifact found.',
        'android':
            'No directly loadable LiteRT artifact found on Hugging Face; only non-LiteRT component sidecars were found.',
      },
    ),
    platformArtifacts: _runtimeArtifacts(
      id: 'minicpm_o_4_5',
      sourceModel: 'openbmb/MiniCPM-o-4_5',
      mlxRepo: 'mlx-community/MiniCPM-o-4_5-4bit',
      mlxArtifact: '.',
    ),
  ),
  ModelSpec(
    id: 'gemma_sea_lion_v4_4b_vl',
    family: 'Gemma SEA-LION v4 4B VL',
    modalities: const [
      ModelModality.textGeneration,
      ModelModality.visionLanguage,
    ],
    description:
        'Gemma SEA-LION v4 4B vision-language publish-time MLX benchmark model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos'],
      blockedPlatforms: const {
        'linux':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'android':
            'No directly loadable LiteRT artifact found on Hugging Face.',
      },
    ),
    platformArtifacts: _runtimeArtifacts(
      id: 'gemma_sea_lion_v4_4b_vl',
      sourceModel: 'aisingapore/Gemma-SEA-LION-v4-4B-VL',
      mlxRepo: 'mlx-community/Gemma-SEA-LION-v4-4B-VL-mlx-3bit',
      mlxArtifact: '.',
    ),
  ),
  ModelSpec(
    id: 'ming_omni_tts_0_5b',
    family: 'Ming-omni TTS 0.5B',
    modalities: const [ModelModality.textToSpeech],
    description: 'Ming-omni TTS 0.5B publish-time MLX benchmark model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: {
      ..._partialRuntimeMigration(
        migratedPlatforms: const ['ios', 'macos'],
        blockedPlatforms: const {
          'linux':
              'Only campplus.onnx component sidecar found; no full ONNX Runtime TTS artifact found.',
          'android':
              'Ming Omni TTS LLM, flowloss_dit_step, linear_proj_audio, stop_head, and audio_decode_chunk submodels export and load as LiteRT/XNNPACK, but full TTS still needs streaming sampler/orchestration.',
        },
      ),
      'runtimeComponentEvidence': {
        'android': {
          'litert_llm_artifact':
              'benchmark/artifacts_local/converted/ming_omni_tts_0_5b/native_components/components/llm_litert/model.litertlm',
          'litert_flowloss_dit_step_artifact':
              'benchmark/artifacts_local/converted/ming_omni_tts_0_5b/native_components/components/litert/flowloss_dit_step/model.tflite',
          'litert_linear_proj_audio_artifact':
              'benchmark/artifacts_local/converted/ming_omni_tts_0_5b/native_components/components/litert/linear_proj_audio/model.tflite',
          'litert_stop_head_artifact':
              'benchmark/artifacts_local/converted/ming_omni_tts_0_5b/native_components/components/litert/stop_head/model.tflite',
          'litert_audio_decode_chunk_artifact':
              'benchmark/artifacts_local/converted/ming_omni_tts_0_5b/native_components/components/litert/audio_decode_chunk/model.tflite',
          'onnx_audio_decode_chunk_artifact':
              'benchmark/artifacts_local/converted/ming_omni_tts_0_5b/native_components/components/onnx/audio_decode_chunk.onnx',
          'audio_decode_chunk_litert_report':
              'benchmark/artifacts_local/converted/ming_omni_tts_0_5b/native_components/components/litert/audio_decode_chunk/onnx_to_litert_report.json',
          'audio_decode_chunk_host_litert_smoke_report':
              'benchmark/out_local/runtime/ming_omni_tts_0_5b/host/audio_decode_chunk_tflite_smoke.json',
          'audio_decode_chunk_host_litert_parity_report':
              'benchmark/out_local/runtime/ming_omni_tts_0_5b/host/audio_decode_chunk_litert_onnx_parity.json',
          'flowloss_dit_step_device_smoke_report':
              'benchmark/out_local/runtime/ming_omni_tts_0_5b/android/flowloss_dit_step_litert_device_smoke_xnnpack.json',
          'linear_proj_audio_device_smoke_report':
              'benchmark/out_local/runtime/ming_omni_tts_0_5b/android/linear_proj_audio_litert_device_smoke_xnnpack.json',
          'stop_head_device_smoke_report':
              'benchmark/out_local/runtime/ming_omni_tts_0_5b/android/stop_head_litert_device_smoke_xnnpack.json',
          'audio_decode_chunk_device_smoke_report':
              'benchmark/out_local/runtime/ming_omni_tts_0_5b/android/audio_decode_chunk_litert_device_smoke_xnnpack.json',
          'audio_decode_chunk_native_device_report':
              'benchmark/out_local/runtime/ming_omni_tts_0_5b/android/audio_decode_chunk_native_litert_report.json',
          'delegate': 'xnnpack',
          'flowloss_dit_step_peak_memory_bytes': 263455744,
          'linear_proj_audio_peak_memory_bytes': 260816896,
          'stop_head_peak_memory_bytes': 53060608,
          'audio_decode_chunk_load_peak_memory_bytes': 2298078208,
          'audio_decode_chunk_invoke_peak_memory_bytes': 2307913728,
          'audio_decode_chunk_end_to_end_ms': 476.2753125,
          'audio_decode_chunk_litert_onnx_max_abs_error': 6.149709224700928e-05,
          'missing_native_components': const [
            'streaming TTS sampler/orchestration',
          ],
        },
      },
    },
    platformArtifacts: _runtimeArtifacts(
      id: 'ming_omni_tts_0_5b',
      sourceModel: 'inclusionAI/Ming-omni-tts-0.5B',
      mlxRepo: 'mlx-community/Ming-omni-tts-0.5B-4bit',
      mlxArtifact: '.',
    ),
  ),
];

Map<String, Object?> _partialRuntimeMigration({
  required List<String> migratedPlatforms,
  required Map<String, String> blockedPlatforms,
}) {
  return {
    'runtimeMigration': {
      'status': 'partial',
      'checkedAt': '2026-04-24',
      'migratedPlatforms': migratedPlatforms,
      'blockedPlatforms': blockedPlatforms,
    },
  };
}

Map<RuntimeEngine, RuntimeArtifact> _runtimeArtifacts({
  required String id,
  String? sourceModel,
  String? mlxRepo,
  String? mlxArtifact,
  String? coremlRepo,
  String? coremlArtifact,
  String? onnxRepo,
  String? onnxArtifact,
  String? litertRepo,
  String? litertArtifact,
}) => {
  if (mlxRepo != null && mlxArtifact != null)
    RuntimeEngine.mlx: _artifact(
      engine: RuntimeEngine.mlx,
      repo: mlxRepo,
      artifact: mlxArtifact,
      modelId: id,
      sourceModel: sourceModel,
      format: 'mlx-safetensors',
      platforms: const ['ios', 'macos'],
      accelerators: const [Accelerator.gpu, Accelerator.cpu],
    ),
  if (coremlRepo != null && coremlArtifact != null)
    RuntimeEngine.coreml: _artifact(
      engine: RuntimeEngine.coreml,
      repo: coremlRepo,
      artifact: coremlArtifact,
      modelId: id,
      sourceModel: sourceModel,
      format: 'coreml-bundle',
      platforms: const ['ios', 'macos'],
      accelerators: const [Accelerator.ane, Accelerator.gpu, Accelerator.cpu],
    ),
  if (onnxRepo != null && onnxArtifact != null)
    RuntimeEngine.onnx: _artifact(
      engine: RuntimeEngine.onnx,
      repo: onnxRepo,
      artifact: onnxArtifact,
      modelId: id,
      sourceModel: sourceModel,
      format: 'onnx',
      platforms: const ['linux', 'android'],
      accelerators: const [Accelerator.gpu, Accelerator.cpu],
    ),
  if (litertRepo != null && litertArtifact != null)
    RuntimeEngine.litert: _artifact(
      engine: RuntimeEngine.litert,
      repo: litertRepo,
      artifact: litertArtifact,
      modelId: id,
      sourceModel: sourceModel,
      format: 'tflite',
      platforms: const ['android'],
      accelerators: const [Accelerator.npu, Accelerator.gpu, Accelerator.cpu],
    ),
};

RuntimeArtifact _artifact({
  required RuntimeEngine engine,
  required String repo,
  required String artifact,
  required String modelId,
  String? sourceModel,
  required String format,
  required List<String> platforms,
  required List<Accelerator> accelerators,
  Map<String, Object?> extraMetadata = const {},
}) {
  final uri = 'hf://$repo/$artifact';
  final metadata = <String, Object?>{
    'source': 'huggingface',
    'modelId': modelId,
    'repo': repo,
    'artifact': artifact,
    ...extraMetadata,
  };
  if (sourceModel != null) {
    metadata['sourceModel'] = sourceModel;
  }
  return RuntimeArtifact(
    engine: engine,
    path: uri,
    sourceUri: uri,
    format: format,
    targetPlatforms: platforms,
    accelerators: accelerators,
    metadata: metadata,
  );
}
