/// Unified model discovery: [ModelSpec], [SnapshotLocator], [SnapshotValidator].
///
/// Every model family registers a [ModelSpec] describing its identity,
/// required files, and capabilities. [SnapshotLocator] resolves a local
/// snapshot directory from a spec, and [SnapshotValidator] checks whether a
/// snapshot is complete and MLX-compatible.
library;

import 'dart:convert';
import 'dart:io';

// ---------------------------------------------------------------------------
// Model capabilities
// ---------------------------------------------------------------------------

/// Modalities a model can process or produce.
enum ModelModality {
  /// Text-in → text-out (LLM / chat).
  textGeneration,

  /// Image + text-in → text-out (VLM / OCR).
  visionLanguage,

  /// Audio-in → text-out (ASR / STT).
  speechToText,

  /// Text-in → audio-out (TTS).
  textToSpeech,

  /// Audio-in → binary-out (VAD).
  voiceActivityDetection,

  /// Text-in → vector-out (embedding).
  embedding,
}

/// Quantisation scheme detected in a snapshot.
enum QuantScheme {
  /// No quantisation — full-precision or mixed-precision weights.
  none,

  /// Standard MLX quantisation (affine, group-based).
  mlxAffine,

  /// Custom quantisation (e.g. GPTQ, AWQ variants).
  custom,
}

// ---------------------------------------------------------------------------
// ModelSpec
// ---------------------------------------------------------------------------

/// Describes a model family so the runtime can discover, validate, and
/// instantiate it without hard-coding paths.
///
/// A [ModelSpec] is a lightweight, immutable descriptor — it does *not*
/// hold loaded tensors or runtime state.
final class ModelSpec {
  const ModelSpec({
    required this.id,
    required this.family,
    required this.modalities,
    this.description = '',
    this.version,
    this.requiredFiles = const ['config.json'],
    this.optionalFiles = const <String>[],
    this.requiredTags = const <String>[],
    this.sizeHint,
    this.metadata = const <String, Object?>{},
  });

  /// Machine-readable identifier, e.g. `qwen3_5`, `paddle_ocr_vl`.
  final String id;

  /// Human-readable family name, e.g. `Qwen3.5`, `PaddleOCR-VL`.
  final String family;

  /// What this model can do.
  final List<ModelModality> modalities;

  /// Short human-readable description.
  final String description;

  /// Semver-ish version string for the model family implementation.
  final String? version;

  /// Files that **must** exist in a valid snapshot directory.
  ///
  /// Defaults to `['config.json']`.  Most models also need at least one
  /// `.safetensors` file, but that is checked by [SnapshotValidator].
  final List<String> requiredFiles;

  /// Files that *may* exist and provide extra functionality.
  final List<String> optionalFiles;

  /// HuggingFace-style tags that indicate compatibility.
  ///
  /// Example: `['mlx', 'text-generation']`.
  final List<String> requiredTags;

  /// Estimated total weight size in bytes (for download budgeting).
  final int? sizeHint;

  /// Arbitrary extra metadata for tooling / manifests.
  final Map<String, Object?> metadata;

  /// Serialise to a JSON-friendly map.
  Map<String, Object?> toJson() => {
    'id': id,
    'family': family,
    'modalities': modalities.map((m) => m.name).toList(),
    if (description.isNotEmpty) 'description': description,
    if (version != null) 'version': version,
    'requiredFiles': requiredFiles,
    if (optionalFiles.isNotEmpty) 'optionalFiles': optionalFiles,
    if (requiredTags.isNotEmpty) 'requiredTags': requiredTags,
    if (sizeHint != null) 'sizeHint': sizeHint,
    if (metadata.isNotEmpty) 'metadata': metadata,
  };

  @override
  String toString() => 'ModelSpec($id, family=$family)';
}

// ---------------------------------------------------------------------------
// SnapshotLocator
// ---------------------------------------------------------------------------

/// Resolves a local snapshot directory for a given [ModelSpec].
///
/// The locator checks a list of search roots (in priority order) for a
/// directory whose `config.json` model-type tag matches the spec.
final class SnapshotLocator {
  const SnapshotLocator({this.searchPaths = const <String>[]});

  /// Directories to search for model snapshots.
  ///
  /// Each entry is scanned for immediate sub-directories containing a
  /// `config.json`.  Pass HuggingFace cache roots, app bundle paths, etc.
  final List<String> searchPaths;

  /// Find the first valid snapshot directory matching [spec].
  ///
  /// Returns `null` if no match is found.
  String? locate(ModelSpec spec) {
    for (final root in searchPaths) {
      final rootDir = Directory(root);
      if (!rootDir.existsSync()) continue;
      for (final entity in rootDir.listSync()) {
        if (entity is! Directory) continue;
        if (_matches(entity.path, spec)) return entity.path;
      }
    }
    return null;
  }

  /// Find all valid snapshot directories matching [spec].
  List<String> locateAll(ModelSpec spec) {
    final results = <String>[];
    for (final root in searchPaths) {
      final rootDir = Directory(root);
      if (!rootDir.existsSync()) continue;
      for (final entity in rootDir.listSync()) {
        if (entity is! Directory) continue;
        if (_matches(entity.path, spec)) results.add(entity.path);
      }
    }
    return results;
  }

  /// Directly check whether [snapshotPath] is a valid snapshot for [spec].
  bool isMatch(String snapshotPath, ModelSpec spec) =>
      _matches(snapshotPath, spec);

  bool _matches(String path, ModelSpec spec) {
    for (final required in spec.requiredFiles) {
      if (!File('$path/$required').existsSync()) return false;
    }
    // Must contain at least one .safetensors file.
    final dir = Directory(path);
    final hasSafetensors = dir.listSync().whereType<File>().any(
      (f) => f.path.endsWith('.safetensors'),
    );
    if (!hasSafetensors) return false;

    // If tags are required, verify config.json contains model_type or tags.
    if (spec.requiredTags.isNotEmpty) {
      try {
        final config =
            jsonDecode(File('$path/config.json').readAsStringSync())
                as Map<String, Object?>;
        final modelType = config['model_type'] as String? ?? '';
        final tags =
            (config['tags'] as List<Object?>?)?.whereType<String>().toSet() ??
            <String>{};
        for (final tag in spec.requiredTags) {
          if (tag != modelType && !tags.contains(tag)) return false;
        }
      } catch (_) {
        return false;
      }
    }
    return true;
  }
}

// ---------------------------------------------------------------------------
// SnapshotValidator
// ---------------------------------------------------------------------------

/// Validation result with an overall pass/fail and individual check details.
final class SnapshotValidation {
  const SnapshotValidation({required this.checks});

  final List<SnapshotCheck> checks;

  /// `true` when every check passed.
  bool get isValid => checks.every((c) => c.passed);

  /// Human-readable summary of failures.
  String get summary {
    final failures = checks.where((c) => !c.passed);
    if (failures.isEmpty) return 'All checks passed.';
    return failures.map((c) => '- ${c.name}: ${c.message}').join('\n');
  }

  @override
  String toString() =>
      isValid ? 'SnapshotValidation(valid)' : 'SnapshotValidation(FAILED)';
}

/// A single validation check.
final class SnapshotCheck {
  const SnapshotCheck({
    required this.name,
    required this.passed,
    this.message = '',
  });

  final String name;
  final bool passed;
  final String message;
}

/// Validates that a snapshot directory is complete and usable.
///
/// Checks performed:
/// 1. All [ModelSpec.requiredFiles] exist.
/// 2. At least one `.safetensors` file is present.
/// 3. `config.json` is valid JSON.
/// 4. Quantisation metadata (if present) is parseable.
/// 5. Optional: estimated size matches [ModelSpec.sizeHint] within tolerance.
final class SnapshotValidator {
  const SnapshotValidator();

  /// Run all checks against [snapshotPath] using the given [spec].
  SnapshotValidation validate(String snapshotPath, ModelSpec spec) {
    final checks = <SnapshotCheck>[];

    // 1. Required files.
    for (final required in spec.requiredFiles) {
      final exists = File('$snapshotPath/$required').existsSync();
      checks.add(
        SnapshotCheck(
          name: 'required_file:$required',
          passed: exists,
          message: exists ? '' : '$required not found',
        ),
      );
    }

    // 2. Safetensors presence.
    final safetensorsFiles = Directory(snapshotPath)
        .listSync()
        .whereType<File>()
        .where((f) => f.path.endsWith('.safetensors'))
        .toList();
    checks.add(
      SnapshotCheck(
        name: 'safetensors_present',
        passed: safetensorsFiles.isNotEmpty,
        message: safetensorsFiles.isEmpty
            ? 'No .safetensors files found'
            : '${safetensorsFiles.length} file(s)',
      ),
    );

    // 3. Config parseable.
    Map<String, Object?>? configJson;
    try {
      configJson =
          jsonDecode(File('$snapshotPath/config.json').readAsStringSync())
              as Map<String, Object?>;
      checks.add(const SnapshotCheck(name: 'config_json_valid', passed: true));
    } catch (e) {
      checks.add(
        SnapshotCheck(
          name: 'config_json_valid',
          passed: false,
          message: 'Failed to parse config.json: $e',
        ),
      );
    }

    // 4. Quantisation metadata.
    if (configJson != null) {
      final quant = configJson['quantization'];
      if (quant != null) {
        final isMap = quant is Map<String, Object?>;
        checks.add(
          SnapshotCheck(
            name: 'quantization_valid',
            passed: isMap,
            message: isMap ? '' : 'quantization field is not a JSON object',
          ),
        );
        if (isMap) {
          final quantMap = quant;
          final hasBits = quantMap.containsKey('bits');
          final hasGroup = quantMap.containsKey('group_size');
          checks.add(
            SnapshotCheck(
              name: 'quantization_fields',
              passed: hasBits && hasGroup,
              message: (!hasBits || !hasGroup)
                  ? 'Missing bits and/or group_size in quantization'
                  : '',
            ),
          );
        }
      }
    }

    // 5. Size hint (if provided).
    if (spec.sizeHint != null) {
      var totalBytes = 0;
      for (final f in safetensorsFiles) {
        totalBytes += f.lengthSync();
      }
      final ratio = spec.sizeHint! > 0 ? totalBytes / spec.sizeHint! : 1.0;
      final withinTolerance = ratio > 0.5 && ratio < 2.0;
      checks.add(
        SnapshotCheck(
          name: 'size_hint',
          passed: withinTolerance,
          message: withinTolerance
              ? ''
              : 'Expected ~${spec.sizeHint} bytes, got $totalBytes',
        ),
      );
    }

    return SnapshotValidation(checks: checks);
  }

  /// Detect the quantisation scheme from config.json.
  QuantScheme detectQuantScheme(String snapshotPath) {
    try {
      final config =
          jsonDecode(File('$snapshotPath/config.json').readAsStringSync())
              as Map<String, Object?>;
      final quant = config['quantization'] as Map<String, Object?>?;
      if (quant == null) return QuantScheme.none;
      final mode = quant['mode'] as String? ?? '';
      if (mode == 'affine' || mode.isEmpty) return QuantScheme.mlxAffine;
      return QuantScheme.custom;
    } catch (_) {
      return QuantScheme.none;
    }
  }
}
