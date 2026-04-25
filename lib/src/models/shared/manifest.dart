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
  return spec.copyWith(
    supportLevel: patch.containsKey('supportLevel')
        ? parseSupportLevel(patch['supportLevel'] as String?)
        : spec.supportLevel,
    validationStatus: status.isEmpty ? spec.validationStatus : status,
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
    optionalFiles: const ['vocab.json', 'merges.txt'],
    platformArtifacts: _runtimeArtifacts(
      id: 'qwen3_asr',
      sourceModel: 'Qwen/Qwen3-ASR-0.6B',
      mlxRepo: 'mlx-community/Qwen3-ASR-1.7B-8bit',
      mlxArtifact: '.',
      coremlRepo: 'FluidInference/qwen3-asr-0.6b-coreml',
      coremlArtifact: 'int8/qwen3_asr_decoder_stateful.mlmodelc',
      onnxRepo: 'Daumee/Qwen3-ASR-0.6B-ONNX-CPU',
      onnxArtifact: 'onnx_models/decoder_step.int8.onnx',
      litertRepo: 'litert-community/Qwen3-ASR-0.6B',
      litertArtifact: 'qwen3_asr_0.6b_5s_i8.tflite',
    ),
  ),
  ModelSpec(
    id: 'kitten_tts',
    family: 'KittenTTS',
    modalities: const [ModelModality.textToSpeech],
    description: 'Lightweight on-device text-to-speech engine',
    requiredFiles: const ['config.json'],
    platformArtifacts: _runtimeArtifacts(
      id: 'kitten_tts',
      sourceModel: 'KittenML/kitten-tts-nano-0.1',
      mlxRepo: 'mlx-community/kitten-tts-nano-0.8-6bit',
      mlxArtifact: '.',
      coremlRepo: 'alexwengg/kittentts-coreml',
      coremlArtifact: 'nano/kittentts_5s.mlmodelc',
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
    description: 'Gemma 4 E2B instruction-tuned text model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    platformArtifacts: _runtimeArtifacts(
      id: 'gemma4',
      sourceModel: 'google/gemma-4-E2B-it',
      mlxRepo: 'unsloth/gemma-4-E2B-it-UD-MLX-4bit',
      mlxArtifact: '.',
      coremlRepo: 'mlboydaisuke/gemma-4-E2B-coreml',
      coremlArtifact: 'lite-chunks',
      onnxRepo: 'onnx-community/gemma-4-E2B-it-ONNX',
      onnxArtifact: 'onnx/decoder_model_merged_q4f16.onnx',
      litertRepo: 'litert-community/gemma-4-E2B-it-litert-lm',
      litertArtifact: 'gemma-4-E2B-it.litertlm',
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
    id: 'qwen3_5_27b_dwq',
    family: 'Qwen3.5 27B DWQ',
    modalities: const [ModelModality.textGeneration],
    description: 'Qwen3.5 27B publish-time MLX benchmark model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos'],
      blockedPlatforms: const {
        'windows':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'linux':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'android':
            'No directly loadable LiteRT artifact found on Hugging Face.',
      },
    ),
    platformArtifacts: _runtimeArtifacts(
      id: 'qwen3_5_27b_dwq',
      sourceModel: 'Qwen/Qwen3.5-27B',
      mlxRepo: 'mlx-community/Qwen3.5-27B-4bit-DWQ',
      mlxArtifact: '.',
    ),
  ),
  ModelSpec(
    id: 'translategemma_27b_it',
    family: 'TranslateGemma 27B IT',
    modalities: const [ModelModality.textGeneration],
    description: 'TranslateGemma 27B instruction-tuned translation model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos', 'android'],
      blockedPlatforms: const {
        'windows':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'linux':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
      },
    ),
    platformArtifacts: _runtimeArtifacts(
      id: 'translategemma_27b_it',
      sourceModel: 'google/translategemma-27b-it',
      mlxRepo: 'mlx-community/translategemma-27b-it-4bit',
      mlxArtifact: '.',
      litertRepo: 'litert-community/TranslateGemma-27B-IT',
      litertArtifact: 'translategemma-27b-it-int8-web.task',
    ),
  ),
  ModelSpec(
    id: 'nemotron3_nano_30b',
    family: 'NVIDIA Nemotron 3 Nano 30B A3B',
    modalities: const [ModelModality.textGeneration],
    description:
        'NVIDIA Nemotron 3 Nano 30B A3B publish-time MLX benchmark model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos'],
      blockedPlatforms: const {
        'windows':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'linux':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'android':
            'No directly loadable LiteRT artifact found on Hugging Face.',
      },
    ),
    platformArtifacts: _runtimeArtifacts(
      id: 'nemotron3_nano_30b',
      sourceModel: 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
      mlxRepo: 'mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit',
      mlxArtifact: '.',
    ),
  ),
  ModelSpec(
    id: 'glm4_7_flash',
    family: 'GLM-4.7-Flash',
    modalities: const [ModelModality.textGeneration],
    description: 'GLM-4.7-Flash publish-time MLX benchmark model',
    requiredFiles: const ['config.json'],
    requiredTags: const ['mlx'],
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos'],
      blockedPlatforms: const {
        'windows':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'linux':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
        'android':
            'No directly loadable LiteRT artifact found on Hugging Face.',
      },
    ),
    platformArtifacts: _runtimeArtifacts(
      id: 'glm4_7_flash',
      sourceModel: 'zai-org/GLM-4.7-Flash',
      mlxRepo: 'mlx-community/GLM-4.7-Flash-4bit',
      mlxArtifact: '.',
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
        'windows':
            'Only token2wav/CosyVoice ONNX components and a Core ML vision sidecar found; no full ONNX Runtime artifact found.',
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
        'windows':
            'No directly loadable ONNX Runtime artifact found on Hugging Face.',
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
    metadata: _partialRuntimeMigration(
      migratedPlatforms: const ['ios', 'macos'],
      blockedPlatforms: const {
        'windows':
            'Only campplus.onnx component sidecar found; no full ONNX Runtime TTS artifact found.',
        'linux':
            'Only campplus.onnx component sidecar found; no full ONNX Runtime TTS artifact found.',
        'android':
            'No directly loadable LiteRT artifact found on Hugging Face; only campplus.onnx component sidecar was found.',
      },
    ),
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
      platforms: const ['ios', 'macos', 'windows', 'linux', 'android'],
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
}) {
  final uri = 'hf://$repo/$artifact';
  final metadata = <String, Object?>{
    'source': 'huggingface',
    'modelId': modelId,
    'repo': repo,
    'artifact': artifact,
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
