/// Machine-readable model manifest.
///
/// A [ModelManifest] is a JSON-serialisable registry of all model families
/// shipped with (or supported by) this package.  It is consumed by
/// benchmarks, examples, documentation generators, and CI tooling.
library;

import 'dart:convert';

import 'model_spec.dart';

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
      final m = e as Map<String, Object?>;
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
        metadata:
            (m['metadata'] as Map<String, Object?>?) ??
            const <String, Object?>{},
      );
    }).toList();
    return ModelManifest(models);
  }
}

// ---------------------------------------------------------------------------
// Built-in model specs
// ---------------------------------------------------------------------------

/// Canonical list of every model family in this package.
///
/// Each model's runner should reference its spec from here.
final List<ModelSpec> builtInSpecs = [
  const ModelSpec(
    id: 'qwen2_5',
    family: 'Qwen2.5',
    modalities: [ModelModality.textGeneration],
    description: 'Qwen 2.5 text-generation LLM (quantised)',
    requiredFiles: ['config.json'],
    requiredTags: ['mlx'],
  ),
  const ModelSpec(
    id: 'qwen3_5',
    family: 'Qwen3.5',
    modalities: [ModelModality.textGeneration, ModelModality.visionLanguage],
    description: 'Qwen 3.5 hybrid LLM with optional vision (Mamba + attention)',
    requiredFiles: ['config.json'],
    requiredTags: ['mlx'],
  ),
  const ModelSpec(
    id: 'paddle_ocr_vl',
    family: 'PaddleOCR-VL',
    modalities: [ModelModality.visionLanguage],
    description: 'ERNIE-4.5 based OCR vision-language model',
    requiredFiles: ['config.json'],
    requiredTags: ['mlx'],
  ),
  const ModelSpec(
    id: 'qwen3_asr',
    family: 'Qwen3-ASR',
    modalities: [ModelModality.speechToText],
    description: 'Qwen3 automatic speech recognition',
    requiredFiles: ['config.json'],
    optionalFiles: ['vocab.json', 'merges.txt'],
  ),
  const ModelSpec(
    id: 'kitten_tts',
    family: 'KittenTTS',
    modalities: [ModelModality.textToSpeech],
    description: 'Lightweight on-device text-to-speech engine',
    requiredFiles: ['config.json'],
  ),
  const ModelSpec(
    id: 'silero_vad',
    family: 'Silero VAD',
    modalities: [ModelModality.voiceActivityDetection],
    description: 'Voice activity detection (Silero)',
  ),
];
