/// Shared [RuntimeTuning] layer for platform-adaptive model configuration.
///
/// Extracts the env-override → iOS-default → macOS-default pattern from
/// `PaddleOcrVlConfig` into a reusable system.  Each model defines its
/// tuning knobs as a [TuningProfile] and the runtime picks the right
/// value based on the current platform + environment overrides.
library;

import 'dart:io';

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------

/// Supported Apple target platforms.
enum ApplePlatform { ios, macos }

/// Detect the current Apple platform at runtime.
///
/// On real devices this uses the OS; in tests you can override via the
/// `DART_INFERENCE_PLATFORM` env var (`ios` or `macos`).
ApplePlatform get currentPlatform {
  final override = Platform.environment['DART_INFERENCE_PLATFORM'];
  if (override == 'ios') return ApplePlatform.ios;
  if (override == 'macos') return ApplePlatform.macos;
  // Platform.isIOS is only true in Flutter; for pure Dart on macOS, default
  // to macOS.  In a Flutter context the Flutter framework sets this.
  if (Platform.operatingSystem == 'ios') return ApplePlatform.ios;
  return ApplePlatform.macos;
}

// ---------------------------------------------------------------------------
// Tuning knob
// ---------------------------------------------------------------------------

/// A single tuning parameter with per-platform defaults and an optional
/// environment-variable override.
///
/// ```dart
/// final chunkSize = TuningKnob<int>(
///   name: 'prefill_chunk_size',
///   envKey: 'MLX_PREFILL_CHUNK',
///   iosDefault: 256,
///   macosDefault: 1024,
///   parse: int.parse,
/// );
/// ```
final class TuningKnob<T> {
  const TuningKnob({
    required this.name,
    this.envKey,
    required this.iosDefault,
    required this.macosDefault,
    required this.parse,
  });

  /// Human-readable name for logging / debugging.
  final String name;

  /// Environment variable key that overrides the platform default.
  final String? envKey;

  /// Default value on iOS.
  final T iosDefault;

  /// Default value on macOS.
  final T macosDefault;

  /// Parse the environment-variable string into [T].
  final T Function(String raw) parse;

  /// Resolve the effective value for the current platform.
  T resolve([ApplePlatform? platform]) {
    if (envKey != null) {
      final raw = Platform.environment[envKey!];
      if (raw != null && raw.isNotEmpty) {
        try {
          return parse(raw);
        } catch (_) {
          // Fall through to platform default on parse failure.
        }
      }
    }
    return (platform ?? currentPlatform) == ApplePlatform.ios
        ? iosDefault
        : macosDefault;
  }
}

// ---------------------------------------------------------------------------
// TuningProfile
// ---------------------------------------------------------------------------

/// A named collection of [TuningKnob]s for a model family.
///
/// Models register their knobs once and consumers call [resolve] to get
/// the effective values for the current platform + env overrides.
final class TuningProfile {
  TuningProfile({required this.name, required this.knobs});

  /// Model family name (e.g. `paddle_ocr_vl`).
  final String name;

  /// All tuning knobs, keyed by [TuningKnob.name].
  final Map<String, TuningKnob<Object>> knobs;

  /// Resolve all knobs for the current (or given) platform.
  Map<String, Object> resolve([ApplePlatform? platform]) {
    final result = <String, Object>{};
    for (final entry in knobs.entries) {
      result[entry.key] = entry.value.resolve(platform);
    }
    return result;
  }

  /// Resolve a single knob by name.
  T get<T>(String name, [ApplePlatform? platform]) {
    final knob = knobs[name];
    if (knob == null) {
      throw ArgumentError('Unknown tuning knob: $name');
    }
    return knob.resolve(platform) as T;
  }
}

// ---------------------------------------------------------------------------
// RuntimeTuning — global registry
// ---------------------------------------------------------------------------

/// Global registry of per-model [TuningProfile]s.
///
/// Models register their profiles at load time and consumers query via
/// `RuntimeTuning.instance[modelId]`.
final class RuntimeTuning {
  RuntimeTuning._();

  static final RuntimeTuning instance = RuntimeTuning._();

  final Map<String, TuningProfile> _profiles = {};

  /// Register a profile for [modelId].
  void register(String modelId, TuningProfile profile) {
    _profiles[modelId] = profile;
  }

  /// Retrieve the profile for [modelId], or `null`.
  TuningProfile? operator [](String modelId) => _profiles[modelId];

  /// Resolve a single knob for a model.
  T resolve<T>(String modelId, String knobName, [ApplePlatform? platform]) {
    final profile = _profiles[modelId];
    if (profile == null) {
      throw StateError('No tuning profile registered for "$modelId"');
    }
    return profile.get<T>(knobName, platform);
  }

  /// All registered model IDs.
  Iterable<String> get modelIds => _profiles.keys;

  /// Dump all resolved values for debugging.
  Map<String, Map<String, Object>> dumpAll([ApplePlatform? platform]) {
    return _profiles.map(
      (id, profile) => MapEntry(id, profile.resolve(platform)),
    );
  }
}

// ---------------------------------------------------------------------------
// Built-in tuning profiles
// ---------------------------------------------------------------------------

/// Convenience knob constructors for common types.
TuningKnob<int> intKnob({
  required String name,
  String? envKey,
  required int iosDefault,
  required int macosDefault,
}) => TuningKnob<int>(
  name: name,
  envKey: envKey,
  iosDefault: iosDefault,
  macosDefault: macosDefault,
  parse: int.parse,
);

TuningKnob<double> doubleKnob({
  required String name,
  String? envKey,
  required double iosDefault,
  required double macosDefault,
}) => TuningKnob<double>(
  name: name,
  envKey: envKey,
  iosDefault: iosDefault,
  macosDefault: macosDefault,
  parse: double.parse,
);

TuningKnob<bool> boolKnob({
  required String name,
  String? envKey,
  required bool iosDefault,
  required bool macosDefault,
}) => TuningKnob<bool>(
  name: name,
  envKey: envKey,
  iosDefault: iosDefault,
  macosDefault: macosDefault,
  parse: (s) => s.toLowerCase() == 'true' || s == '1',
);

TuningKnob<String> stringKnob({
  required String name,
  String? envKey,
  required String iosDefault,
  required String macosDefault,
}) => TuningKnob<String>(
  name: name,
  envKey: envKey,
  iosDefault: iosDefault,
  macosDefault: macosDefault,
  parse: (s) => s,
);

/// PaddleOCR-VL tuning profile (the most complex one in the codebase).
///
/// This replaces all the `*ForCurrentPlatform` getters in
/// `PaddleOcrVlConfig` with a declarative profile.
final paddleOcrVlTuning = TuningProfile(
  name: 'paddle_ocr_vl',
  knobs: <String, TuningKnob<Object>>{
    'max_pixels': intKnob(
      name: 'max_pixels',
      envKey: 'MLX_MAX_PIXELS',
      iosDefault: 512 * 512,
      macosDefault: 2822400,
    ),
    'vision_attn_chunk': intKnob(
      name: 'vision_attn_chunk',
      envKey: 'MLX_VIS_ATTN_CHUNK',
      iosDefault: 128,
      macosDefault: 0, // 0 = no chunking
    ),
    'enable_vision_layerwise_eval': boolKnob(
      name: 'enable_vision_layerwise_eval',
      envKey: 'MLX_VIS_LAYERWISE',
      iosDefault: true,
      macosDefault: false,
    ),
    'enable_decoder_layerwise_eval': boolKnob(
      name: 'enable_decoder_layerwise_eval',
      envKey: 'MLX_DEC_LAYERWISE',
      iosDefault: true,
      macosDefault: false,
    ),
    'enable_aggressive_cache_clear': boolKnob(
      name: 'enable_aggressive_cache_clear',
      envKey: 'MLX_AGG_CACHE_CLEAR',
      iosDefault: true,
      macosDefault: false,
    ),
    'force_f32_vision': boolKnob(
      name: 'force_f32_vision',
      envKey: 'MLX_FORCE_F32_VISION',
      iosDefault: false,
      macosDefault: false,
    ),
    'memory_limit_bytes': intKnob(
      name: 'memory_limit_bytes',
      envKey: 'MLX_MEM_LIMIT',
      iosDefault: (2.25 * 1024 * 1024 * 1024).toInt(), // 2.25 GB
      macosDefault: 0, // 0 = no limit
    ),
    'cache_limit_bytes': intKnob(
      name: 'cache_limit_bytes',
      envKey: 'MLX_CACHE_LIMIT',
      iosDefault: 256 * 1024 * 1024, // 256 MB
      macosDefault: 0,
    ),
    'prefill_chunk_size': intKnob(
      name: 'prefill_chunk_size',
      envKey: 'MLX_PREFILL_CHUNK',
      iosDefault: 256,
      macosDefault: 0,
    ),
    'kv_cache_quant_bits': intKnob(
      name: 'kv_cache_quant_bits',
      envKey: 'MLX_KV_BITS',
      iosDefault: 8,
      macosDefault: 0,
    ),
    'kv_cache_group_size': intKnob(
      name: 'kv_cache_group_size',
      envKey: 'MLX_KV_GROUP',
      iosDefault: 64,
      macosDefault: 64,
    ),
    'kv_cache_quant_start': intKnob(
      name: 'kv_cache_quant_start',
      envKey: 'MLX_KV_QUANT_START',
      iosDefault: 0,
      macosDefault: 512,
    ),
    'kv_cache_quant_scheme': stringKnob(
      name: 'kv_cache_quant_scheme',
      envKey: 'MLX_KV_SCHEME',
      iosDefault: 'uniform',
      macosDefault: 'none',
    ),
    'turbo_quant_bits': doubleKnob(
      name: 'turbo_quant_bits',
      envKey: 'MLX_TURBO_BITS',
      iosDefault: 4.0,
      macosDefault: 4.0,
    ),
    'turbo_quant_start': intKnob(
      name: 'turbo_quant_start',
      envKey: 'MLX_TURBO_START',
      iosDefault: 0,
      macosDefault: 0,
    ),
    'max_kv_seq_len': intKnob(
      name: 'max_kv_seq_len',
      envKey: 'MLX_MAX_KV_SEQ',
      iosDefault: 1152,
      macosDefault: 4096,
    ),
    'vision_eval_batch': intKnob(
      name: 'vision_eval_batch',
      envKey: 'MLX_VIS_EVAL_BATCH',
      iosDefault: 1,
      macosDefault: 0,
    ),
    'vision_mlp_chunk': intKnob(
      name: 'vision_mlp_chunk',
      envKey: 'MLX_VIS_MLP_CHUNK',
      iosDefault: 512,
      macosDefault: 0,
    ),
    'vision_proj_chunk': intKnob(
      name: 'vision_proj_chunk',
      envKey: 'MLX_VIS_PROJ_CHUNK',
      iosDefault: 256,
      macosDefault: 0,
    ),
    'enable_vision_weight_release': boolKnob(
      name: 'enable_vision_weight_release',
      envKey: 'MLX_VIS_WEIGHT_RELEASE',
      iosDefault: true,
      macosDefault: false,
    ),
    'decode_cache_eval_interval': intKnob(
      name: 'decode_cache_eval_interval',
      envKey: 'MLX_CACHE_EVAL_INT',
      iosDefault: 1,
      macosDefault: 16,
    ),
  },
);

/// Qwen3.5 tuning profile (lighter — fewer platform differences).
final qwen35Tuning = TuningProfile(
  name: 'qwen3_5',
  knobs: <String, TuningKnob<Object>>{
    'prefill_chunk_size': intKnob(
      name: 'prefill_chunk_size',
      envKey: 'MLX_PREFILL_CHUNK',
      iosDefault: 256,
      macosDefault: 0,
    ),
    'max_kv_seq_len': intKnob(
      name: 'max_kv_seq_len',
      envKey: 'MLX_MAX_KV_SEQ',
      iosDefault: 2048,
      macosDefault: 8192,
    ),
  },
);

/// Qwen3-ASR tuning profile.
final qwen3AsrTuning = TuningProfile(
  name: 'qwen3_asr',
  knobs: <String, TuningKnob<Object>>{
    'max_audio_tokens': intKnob(
      name: 'max_audio_tokens',
      envKey: 'MLX_MAX_AUDIO_TOKENS',
      iosDefault: 800,
      macosDefault: 1500,
    ),
    'max_new_tokens': intKnob(
      name: 'max_new_tokens',
      envKey: 'MLX_MAX_NEW_TOKENS',
      iosDefault: 256,
      macosDefault: 512,
    ),
  },
);
