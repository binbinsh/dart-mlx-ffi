/// Shared runtime metadata for model manifests and cross-platform execution.
library;

/// Runtime engines supported by the model-level API.
enum RuntimeEngine {
  /// Apple MLX runtime.
  mlx,

  /// Apple Core ML runtime, including ANE-capable execution.
  coreml,

  /// ONNX Runtime for desktop and server targets.
  onnx,

  /// Google LiteRT runtime for Android.
  litert,
}

/// Hardware accelerators that a runtime may request or report.
enum Accelerator {
  /// CPU execution.
  cpu,

  /// GPU execution.
  gpu,

  /// Apple Neural Engine execution.
  ane,

  /// Android/vendor neural processing unit execution.
  npu,
}

/// Support level exposed by the default model registry.
enum SupportLevel {
  /// Available for development and benchmarking, without full-platform SLA.
  staging,

  /// Passed the required platform correctness, speed, and peak-memory gates.
  production,
}

/// A concrete model artifact for one runtime engine.
final class RuntimeArtifact {
  const RuntimeArtifact({
    required this.engine,
    required this.path,
    this.format,
    this.sourceUri,
    this.targetPlatforms = const <String>[],
    this.accelerators = const <Accelerator>[],
    this.checksum,
    this.metadata = const <String, Object?>{},
  });

  /// Runtime engine that can load this artifact.
  final RuntimeEngine engine;

  /// Path relative to the snapshot/model root, or an absolute path.
  final String path;

  /// Artifact format, e.g. `mlx-safetensors`, `mlmodelc`, `onnx`, `tflite`.
  final String? format;

  /// Canonical remote source URI, e.g. `hf://org/repo/path/to/model.onnx`.
  final String? sourceUri;

  /// Platform names this artifact targets: `ios`, `macos`, `windows`, etc.
  final List<String> targetPlatforms;

  /// Preferred/available accelerators for this artifact.
  final List<Accelerator> accelerators;

  /// Optional checksum used by validation and packaging tools.
  final String? checksum;

  /// Tooling-specific metadata.
  final Map<String, Object?> metadata;

  RuntimeArtifact copyWith({
    RuntimeEngine? engine,
    String? path,
    String? format,
    String? sourceUri,
    List<String>? targetPlatforms,
    List<Accelerator>? accelerators,
    String? checksum,
    Map<String, Object?>? metadata,
  }) {
    return RuntimeArtifact(
      engine: engine ?? this.engine,
      path: path ?? this.path,
      format: format ?? this.format,
      sourceUri: sourceUri ?? this.sourceUri,
      targetPlatforms: targetPlatforms ?? this.targetPlatforms,
      accelerators: accelerators ?? this.accelerators,
      checksum: checksum ?? this.checksum,
      metadata: metadata ?? this.metadata,
    );
  }

  Map<String, Object?> toJson() => {
    'engine': engine.name,
    'path': path,
    if (format != null) 'format': format,
    if (sourceUri != null) 'sourceUri': sourceUri,
    if (targetPlatforms.isNotEmpty) 'targetPlatforms': targetPlatforms,
    if (accelerators.isNotEmpty)
      'accelerators': accelerators.map((a) => a.name).toList(),
    if (checksum != null) 'checksum': checksum,
    if (metadata.isNotEmpty) 'metadata': metadata,
  };

  factory RuntimeArtifact.fromJson(Map<String, Object?> json) {
    return RuntimeArtifact(
      engine: _parseRuntimeEngine(json['engine'] as String?),
      path: json['path'] as String? ?? '',
      format: json['format'] as String?,
      sourceUri: json['sourceUri'] as String?,
      targetPlatforms:
          (json['targetPlatforms'] as List<Object?>?)
              ?.whereType<String>()
              .toList() ??
          const <String>[],
      accelerators:
          (json['accelerators'] as List<Object?>?)
              ?.whereType<String>()
              .map(_parseAccelerator)
              .toList() ??
          const <Accelerator>[],
      checksum: json['checksum'] as String?,
      metadata: _objectMap(json['metadata']),
    );
  }
}

/// Per-platform evidence used to decide whether a model is production-ready.
final class RuntimeValidationStatus {
  const RuntimeValidationStatus({
    required this.platform,
    required this.engine,
    this.identityPassed = false,
    this.correctnessPassed = false,
    this.speedPassed = false,
    this.peakMemoryPassed = false,
    this.deviceProfilePassed = false,
    this.promotionPassed,
    this.reportPath,
    this.speedRatio,
    this.ttftRatio,
    this.endToEndRatio,
    this.peakMemoryRatio,
    this.peakMemoryBytes,
    this.baselinePeakMemoryBytes,
    this.iterationCount,
    this.warmupCount,
    this.latencyMs = const <String, Object?>{},
    this.baselineLatencyMs = const <String, Object?>{},
    this.runConfig = const <String, Object?>{},
    this.inputSignature = const <String, Object?>{},
    this.deviceProfile = const <String, Object?>{},
    this.notes = const <String>[],
  });

  /// Platform name, e.g. `ios`, `macos`, `windows`, `linux`, `android`.
  final String platform;

  /// Runtime engine used for this validation entry.
  final RuntimeEngine engine;

  /// Whether report identity, inputs, and run config matched.
  final bool identityPassed;

  /// Whether deterministic outputs and task-level parity passed.
  final bool correctnessPassed;

  /// Whether throughput/latency ratios passed.
  final bool speedPassed;

  /// Whether peak-memory ratio passed.
  final bool peakMemoryPassed;

  /// Whether provider/delegate/device-placement evidence passed.
  final bool deviceProfilePassed;

  /// Overall verdict emitted by the runtime matrix comparator.
  final bool? promotionPassed;

  /// Path to the full runtime matrix report.
  final String? reportPath;

  /// Candidate throughput divided by baseline throughput.
  final double? speedRatio;

  /// Candidate TTFT divided by baseline TTFT.
  final double? ttftRatio;

  /// Candidate end-to-end latency divided by baseline latency.
  final double? endToEndRatio;

  /// Candidate peak memory divided by baseline peak memory.
  final double? peakMemoryRatio;

  /// Candidate peak memory in bytes.
  final int? peakMemoryBytes;

  /// Baseline peak memory in bytes.
  final int? baselinePeakMemoryBytes;

  /// Number of measured iterations in the candidate run.
  final int? iterationCount;

  /// Number of warmup iterations before candidate measurement.
  final int? warmupCount;

  /// Candidate latency summary: sample count, mean, p50, p95.
  final Map<String, Object?> latencyMs;

  /// Baseline latency summary: sample count, mean, p50, p95.
  final Map<String, Object?> baselineLatencyMs;

  /// Benchmark run configuration used for this validation.
  final Map<String, Object?> runConfig;

  /// Input signature used to prove candidate/baseline input identity.
  final Map<String, Object?> inputSignature;

  /// Raw device/provider profile details.
  final Map<String, Object?> deviceProfile;

  /// Human-readable validation notes.
  final List<String> notes;

  /// All required production gates passed for this platform.
  bool get passed =>
      identityPassed &&
      correctnessPassed &&
      speedPassed &&
      peakMemoryPassed &&
      deviceProfilePassed;

  Map<String, Object?> toJson() => {
    'platform': platform,
    'engine': engine.name,
    'identityPassed': identityPassed,
    'correctnessPassed': correctnessPassed,
    'speedPassed': speedPassed,
    'peakMemoryPassed': peakMemoryPassed,
    'deviceProfilePassed': deviceProfilePassed,
    if (promotionPassed != null) 'promotionPassed': promotionPassed,
    if (reportPath != null) 'reportPath': reportPath,
    if (speedRatio != null) 'speedRatio': speedRatio,
    if (ttftRatio != null) 'ttftRatio': ttftRatio,
    if (endToEndRatio != null) 'endToEndRatio': endToEndRatio,
    if (peakMemoryRatio != null) 'peakMemoryRatio': peakMemoryRatio,
    if (peakMemoryBytes != null) 'peakMemoryBytes': peakMemoryBytes,
    if (baselinePeakMemoryBytes != null)
      'baselinePeakMemoryBytes': baselinePeakMemoryBytes,
    if (iterationCount != null) 'iterationCount': iterationCount,
    if (warmupCount != null) 'warmupCount': warmupCount,
    if (latencyMs.isNotEmpty) 'latencyMs': latencyMs,
    if (baselineLatencyMs.isNotEmpty) 'baselineLatencyMs': baselineLatencyMs,
    if (runConfig.isNotEmpty) 'runConfig': runConfig,
    if (inputSignature.isNotEmpty) 'inputSignature': inputSignature,
    if (deviceProfile.isNotEmpty) 'deviceProfile': deviceProfile,
    if (notes.isNotEmpty) 'notes': notes,
  };

  factory RuntimeValidationStatus.fromJson(Map<String, Object?> json) {
    return RuntimeValidationStatus(
      platform: json['platform'] as String? ?? '',
      engine: _parseRuntimeEngine(json['engine'] as String?),
      identityPassed: json['identityPassed'] as bool? ?? false,
      correctnessPassed: json['correctnessPassed'] as bool? ?? false,
      speedPassed: json['speedPassed'] as bool? ?? false,
      peakMemoryPassed: json['peakMemoryPassed'] as bool? ?? false,
      deviceProfilePassed: json['deviceProfilePassed'] as bool? ?? false,
      promotionPassed: json['promotionPassed'] as bool?,
      reportPath: json['reportPath'] as String?,
      speedRatio: (json['speedRatio'] as num?)?.toDouble(),
      ttftRatio: (json['ttftRatio'] as num?)?.toDouble(),
      endToEndRatio: (json['endToEndRatio'] as num?)?.toDouble(),
      peakMemoryRatio: (json['peakMemoryRatio'] as num?)?.toDouble(),
      peakMemoryBytes: (json['peakMemoryBytes'] as num?)?.toInt(),
      baselinePeakMemoryBytes: (json['baselinePeakMemoryBytes'] as num?)
          ?.toInt(),
      iterationCount: (json['iterationCount'] as num?)?.toInt(),
      warmupCount: (json['warmupCount'] as num?)?.toInt(),
      latencyMs: _objectMap(json['latencyMs']),
      baselineLatencyMs: _objectMap(json['baselineLatencyMs']),
      runConfig: _objectMap(json['runConfig']),
      inputSignature: _objectMap(json['inputSignature']),
      deviceProfile: _objectMap(json['deviceProfile']),
      notes:
          (json['notes'] as List<Object?>?)?.whereType<String>().toList() ??
          const <String>[],
    );
  }
}

RuntimeEngine _parseRuntimeEngine(String? value) {
  return RuntimeEngine.values.firstWhere(
    (engine) => engine.name == value,
    orElse: () => RuntimeEngine.mlx,
  );
}

Accelerator _parseAccelerator(String value) {
  return Accelerator.values.firstWhere(
    (accelerator) => accelerator.name == value,
    orElse: () => Accelerator.cpu,
  );
}

SupportLevel parseSupportLevel(String? value) {
  return SupportLevel.values.firstWhere(
    (level) => level.name == value,
    orElse: () => SupportLevel.staging,
  );
}

Map<String, Object?> _objectMap(Object? value) {
  if (value is Map) {
    return Map<String, Object?>.from(value);
  }
  return const <String, Object?>{};
}
