/// Typed view over `pipeline.json` schema v2 for the PaddleOCR-VL CoreML
/// bundle (ADR §6).
///
/// The bundle layout is:
///
/// ```
/// bundle/
///   pipeline.json
///   vision_embed.mlpackage/
///   token_embed.mlpackage/
///   prefill_decoder.mlpackage/
///   decode_decoder.mlpackage/
///   tokenizer.json   (optional; reuse caller-side BpeTokenizer)
/// ```
library;

import 'dart:convert';
import 'dart:io';

/// Compute-unit hint matching `MLComputeUnits` (ADR §3).
enum CoremlComputeUnits {
  cpuOnly,
  cpuAndGpu,
  cpuAndNeuralEngine,
  all,
}

CoremlComputeUnits _parseUnits(String? raw) {
  switch (raw) {
    case 'cpu_only':
      return CoremlComputeUnits.cpuOnly;
    case 'cpu_and_gpu':
      return CoremlComputeUnits.cpuAndGpu;
    case 'cpu_and_neural_engine':
    case 'cpu_and_ne':
      return CoremlComputeUnits.cpuAndNeuralEngine;
    case 'all':
      return CoremlComputeUnits.all;
    default:
      return CoremlComputeUnits.cpuAndGpu;
  }
}

/// Single stage entry in `pipeline.json`.
final class CoremlStage {
  const CoremlStage({
    required this.name,
    required this.package,
    required this.computeUnits,
    required this.stateful,
    required this.stateGroup,
  });

  final String name;
  final String package;
  final CoremlComputeUnits computeUnits;
  final bool stateful;
  final String? stateGroup;

  factory CoremlStage.fromJson(Map<String, Object?> json) => CoremlStage(
        name: json['name']! as String,
        package: json['package']! as String,
        computeUnits: _parseUnits(json['compute_units'] as String?),
        stateful: json['stateful'] as bool? ?? false,
        stateGroup: json['state_group'] as String?,
      );
}

/// Vision bucket / patch metadata (ADR §5.1, §6).
final class CoremlVisionMeta {
  const CoremlVisionMeta({
    required this.buckets,
    required this.patchSize,
    required this.spatialMerge,
  });

  /// `(t, h, w)` triples; `h`/`w` are unmerged patch counts.
  final List<(int, int, int)> buckets;
  final int patchSize;
  final int spatialMerge;

  factory CoremlVisionMeta.fromJson(Map<String, Object?> json) {
    final raw = (json['buckets'] as List).cast<List>();
    return CoremlVisionMeta(
      buckets: raw
          .map(
            (b) => ((b[0] as num).toInt(), (b[1] as num).toInt(),
                (b[2] as num).toInt()),
          )
          .toList(growable: false),
      patchSize: (json['patch_size'] as num?)?.toInt() ?? 14,
      spatialMerge: (json['spatial_merge'] as num?)?.toInt() ?? 2,
    );
  }
}

/// KV cache shape metadata (ADR §6).
final class CoremlKvMeta {
  const CoremlKvMeta({
    required this.layers,
    required this.kvHeads,
    required this.headDim,
    required this.maxLen,
    required this.dtype,
  });

  final int layers;
  final int kvHeads;
  final int headDim;
  final int maxLen;
  final String dtype;

  factory CoremlKvMeta.fromJson(Map<String, Object?> json) => CoremlKvMeta(
        layers: (json['layers'] as num).toInt(),
        kvHeads: (json['kv_heads'] as num).toInt(),
        headDim: (json['head_dim'] as num).toInt(),
        maxLen: (json['max_len'] as num).toInt(),
        dtype: json['dtype'] as String? ?? 'fp16',
      );
}

/// Special-token table (ADR §6).
final class CoremlTokenMeta {
  const CoremlTokenMeta({
    required this.imageTokenId,
    required this.eosTokenId,
    required this.padTokenId,
  });

  final int imageTokenId;
  final int eosTokenId;
  final int padTokenId;

  factory CoremlTokenMeta.fromJson(Map<String, Object?> json) => CoremlTokenMeta(
        imageTokenId: (json['image_token_id'] as num).toInt(),
        eosTokenId: (json['eos_token_id'] as num).toInt(),
        padTokenId: (json['pad_token_id'] as num?)?.toInt() ?? 0,
      );
}

/// Top-level `pipeline.json` (schema v2).
final class CoremlPipelineManifest {
  const CoremlPipelineManifest({
    required this.schema,
    required this.modelId,
    required this.stages,
    required this.kv,
    required this.vision,
    required this.tokens,
    this.prefillBuckets = const [128, 256, 384, 512, 768],
  });

  final int schema;
  final String modelId;
  final List<CoremlStage> stages;
  final CoremlKvMeta kv;
  final CoremlVisionMeta vision;
  final CoremlTokenMeta tokens;

  /// Allowed prefill `seq_len` values (ADR §5.2). Optional in JSON; we fall
  /// back to the default ladder if absent.
  final List<int> prefillBuckets;

  CoremlStage stage(String name) => stages.firstWhere(
        (s) => s.name == name,
        orElse: () =>
            throw StateError('pipeline.json missing stage "$name"'),
      );

  factory CoremlPipelineManifest.fromJson(Map<String, Object?> json) {
    final stages = (json['stages'] as List)
        .cast<Map<String, Object?>>()
        .map(CoremlStage.fromJson)
        .toList(growable: false);
    final prefill = json['prefill_buckets'] as List?;
    return CoremlPipelineManifest(
      schema: (json['schema'] as num?)?.toInt() ?? 2,
      modelId: json['model'] as String? ?? 'paddleocr-vl-1.5-coreml',
      stages: stages,
      kv: CoremlKvMeta.fromJson(json['kv'] as Map<String, Object?>),
      vision: CoremlVisionMeta.fromJson(json['vision'] as Map<String, Object?>),
      tokens:
          CoremlTokenMeta.fromJson(json['tokens'] as Map<String, Object?>),
      prefillBuckets: prefill == null
          ? const [128, 256, 384, 512, 768]
          : prefill.cast<num>().map((n) => n.toInt()).toList(growable: false),
    );
  }

  factory CoremlPipelineManifest.loadFile(String path) {
    final raw = File(path).readAsStringSync();
    final json = jsonDecode(raw) as Map<String, Object?>;
    return CoremlPipelineManifest.fromJson(json);
  }

  /// Pick the smallest prefill bucket `>= promptLen`. Returns the largest
  /// bucket if none fits (caller should error or fall back to MLX).
  int pickPrefillBucket(int promptLen) {
    for (final b in prefillBuckets) {
      if (b >= promptLen) return b;
    }
    return prefillBuckets.last;
  }
}
