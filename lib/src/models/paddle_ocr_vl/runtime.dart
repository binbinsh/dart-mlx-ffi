part of 'paddle_ocr_vl.dart';

extension PaddleOcrVlConfigRuntime on PaddleOcrVlConfig {
  int get recommendedMaxPixelsForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_MAX_PIXELS'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    if (Platform.isIOS) {
      // iPhone Jetsam logs show the 640-pixel tier can still exceed the
      // foreground hard limit on complex photos, so default to 512.
      return 28 * 28 * 512;
    }
    return 2822400;
  }

  int get recommendedVisionAttentionChunkSizeForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.visionAttentionChunk;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_VISION_ATTENTION_CHUNK'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 16;
    }
    return -1;
  }

  bool get enableVisionLayerwiseEvalForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_VISION_LAYER_EVAL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }

  int get recommendedMemoryLimitBytesForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_MEMORY_LIMIT_MB'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed * 1024 * 1024;
    }
    if (Platform.isIOS) {
      const fallback = 2304 * 1024 * 1024;
      final adaptive = _adaptiveIosMemoryLimitBytes();
      if (adaptive > 0) {
        return math.min(fallback, adaptive);
      }
      return fallback;
    }
    return -1;
  }

  int get recommendedWiredLimitBytesForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_WIRED_LIMIT_MB'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed * 1024 * 1024;
    }
    if (Platform.isIOS) {
      final deviceLimit = _deviceRecommendedWorkingSetBytes();
      final memoryLimit = recommendedMemoryLimitBytesForCurrentPlatform;
      final derived = memoryLimit > 0 ? memoryLimit + (256 * 1024 * 1024) : -1;
      if (deviceLimit > 0 && derived > 0) {
        return math.min(deviceLimit, derived);
      }
      if (derived > 0) return derived;
      return deviceLimit;
    }
    return -1;
  }

  int get recommendedCacheLimitBytesForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_CACHE_LIMIT_MB'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed * 1024 * 1024;
    }
    if (Platform.isIOS) {
      return 256 * 1024 * 1024;
    }
    return -1;
  }

  int get visionEvalBatchSizeForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_VISION_EVAL_BATCH'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    if (Platform.isIOS) {
      return 1;
    }
    return 1;
  }

  int get visionMlpChunkSizeForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.visionMlpChunk;
    if (debug != null) return debug;
    final override = Platform.environment['DART_MLX_PADDLE_VISION_MLP_CHUNK'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 64;
    }
    return -1;
  }

  int get visionProjectorChunkSizeForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.visionProjectorChunk;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_VISION_PROJECTOR_CHUNK'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 32;
    }
    return -1;
  }

  int get prefillChunkSizeForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.prefillChunk;
    if (debug != null) return debug;
    final override = Platform.environment['DART_MLX_PADDLE_PREFILL_CHUNK'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 256;
    }
    return -1;
  }

  bool get enableDecodeCacheStateEvalForCurrentPlatform {
    final override =
        Platform.environment['DART_MLX_PADDLE_DECODE_CACHE_STATE_EVAL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }

  int get decodeCacheStateEvalIntervalForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decodeCacheEvalInterval;
    if (debug != null && debug > 0) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_DECODE_CACHE_EVAL_INTERVAL'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    if (Platform.isIOS && kvCacheQuantSchemeForCurrentPlatform == 'turboquant') {
      return 8;
    }
    if (Platform.isIOS) {
      return 1;
    }
    return 16;
  }

  int get kvCacheQuantBitsForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform == 'turboquant') {
      return -1;
    }
    final debug = PaddleOcrVlDebugOverrides.kvBits;
    if (debug != null) {
      return debug;
    }
    final override = Platform.environment['DART_MLX_PADDLE_KV_BITS'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 8;
    }
    return -1;
  }

  int get kvCacheQuantGroupSizeForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_KV_GROUP_SIZE'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    return 64;
  }

  int get kvCacheQuantizedStartForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform == 'turboquant') {
      return -1;
    }
    final override = Platform.environment['DART_MLX_PADDLE_KV_QUANT_START'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 0;
    }
    return -1;
  }

  String get kvCacheQuantSchemeForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.kvQuantScheme;
    if (debug != null) {
      final normalized = debug.toLowerCase();
      if (normalized == 'uniform' || normalized == 'turboquant') {
        return normalized;
      }
    }
    final override = Platform.environment['DART_MLX_PADDLE_KV_QUANT_SCHEME'];
    if (override != null) {
      final normalized = override.toLowerCase();
      if (normalized == 'uniform' || normalized == 'turboquant') {
        return normalized;
      }
    }
    // Real photo_render_512 runs on iPhone showed that turboquant drifted
    // decode tokens away from the host/reference path, while uniform 8-bit KV
    // both restored the expected prefix and lowered peak memory.
    return 'uniform';
  }

  double? get turboQuantBitsForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return null;
    }
    final debug = PaddleOcrVlDebugOverrides.turboBits;
    if (debug != null && debug >= 1) {
      return debug;
    }
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_BITS'] ??
        Platform.environment['DART_MLX_PADDLE_KV_BITS'];
    if (override != null) {
      final parsed = double.tryParse(override);
      if (parsed != null && parsed >= 1) return parsed;
    }
    if (Platform.isIOS) {
      return 3.5;
    }
    return 4.0;
  }

  int get turboQuantizedStartForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return -1;
    }
    final debug = PaddleOcrVlDebugOverrides.turboStart;
    if (debug != null) return debug;
    final override = Platform.environment['DART_MLX_PADDLE_TURBO_START'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    return 0;
  }

  bool get turboDensePrefillForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return false;
    }
    final debug = PaddleOcrVlDebugOverrides.turboDensePrefill;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_DENSE_PREFILL'];
    if (override == null) return true;
    return override == '1' || override.toLowerCase() == 'true';
  }

  int get turboCompactBudgetForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return -1;
    }
    final debug = PaddleOcrVlDebugOverrides.turboCompactBudget;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_COMPACT_BUDGET'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 640;
    }
    return -1;
  }

  int get turboCompactKeepRecentForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return 0;
    }
    final debug = PaddleOcrVlDebugOverrides.turboCompactKeepRecent;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_COMPACT_KEEP_RECENT'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    return 128;
  }

  int get turboCompactIntervalForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return -1;
    }
    final debug = PaddleOcrVlDebugOverrides.turboCompactInterval;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_COMPACT_INTERVAL'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    return 32;
  }

  int get turboCompactKeepPrefixForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return 0;
    }
    final debug = PaddleOcrVlDebugOverrides.turboCompactKeepPrefix;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_COMPACT_KEEP_PREFIX'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    return 16;
  }

  int get turboCompactHysteresisForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return 0;
    }
    final debug = PaddleOcrVlDebugOverrides.turboCompactHysteresis;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_COMPACT_HYSTERESIS'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    if (Platform.isIOS) {
      return 32;
    }
    return 64;
  }

  bool get uniformQuantizedPrefillForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'uniform') {
      return false;
    }
    final debug = PaddleOcrVlDebugOverrides.uniformQuantizedPrefill;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_UNIFORM_QUANTIZED_PREFILL'];
    if (override == null) return false;
    return override == '1' || override.toLowerCase() == 'true';
  }

  int get maxKvCacheSeqLenForCurrentPlatform {
    final override =
        Platform.environment['DART_MLX_PADDLE_MAX_KV_CACHE_SEQ_LEN'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    if (Platform.isIOS) {
      return 1152;
    }
    return 4096;
  }

  int get turboCapacityStepForCurrentPlatform {
    if (kvCacheQuantSchemeForCurrentPlatform != 'turboquant') {
      return 256;
    }
    final override =
        Platform.environment['DART_MLX_PADDLE_TURBO_CAPACITY_STEP'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    if (Platform.isIOS) {
      return 128;
    }
    return 256;
  }

  bool get enableVisionWeightReleaseForCurrentPlatform {
    final override =
        Platform.environment['DART_MLX_PADDLE_RELEASE_VISION_WEIGHTS'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }
}

int _adaptiveIosMemoryLimitBytes() {
  final value = _deviceRecommendedWorkingSetBytes();
  if (value > 0) {
    return (value.toDouble() * 0.75).floor();
  }
  return -1;
}

int _deviceRecommendedWorkingSetBytes() {
  try {
    final device = MlxDevice.defaultDevice();
    try {
      final value = device.info['max_recommended_working_set_size'];
      if (value is num && value > 0) {
        return value.toInt();
      }
    } finally {
      device.close();
    }
  } catch (_) {}
  return -1;
}
