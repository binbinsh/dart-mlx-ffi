part of 'paddle_ocr_vl.dart';

final class PaddleOcrVlDebugOverrides {
  static String? kvQuantScheme;
  static int? kvBits;
  static double? turboBits;
  static int? turboStart;
  static bool? turboDensePrefill;
  static bool? uniformQuantizedPrefill;
  static bool? turboDisableFusedKv;
  static bool? turboDisableFusedDecode;
  static bool? turboDisableSingleQuant;
  static bool? turboDisableFastScore;
  static bool? turboDisableFastValue;
  static bool? turboUniformLastLayer;
  static bool? turboQuantizeLastLayer;
  static bool? turboUpdateTrace;
  static int? turboCompactBudget;
  static int? turboCompactKeepRecent;
  static int? turboCompactInterval;
  static int? turboCompactKeepPrefix;
  static int? turboCompactHysteresis;
  static int? memoryLimitBytes;
  static int? cacheLimitBytes;
  static int? wiredLimitBytes;
  static bool? decoderLayerEval;
  static bool? forceGpuSynchronizeAfterLayerEval;
  static bool? aggressiveCacheClear;
  static int? decodeCacheEvalInterval;
  static bool? decoderPerLayerTrace;
  static int? decoderPerLayerTraceInterval;
  static bool? decoderSubstepTrace;
  static int? decoderSubstepTraceLayer;
  static bool? decoderTailTrace;
  static bool? explicitLogitsEval;
  static bool? decodeStepExplicitLogitsEval;
  static bool? clearAfterCloseLogits;
  static bool? forceGpuSynchronizePerToken;
  static bool? forceDecodeLmHeadFloat32;
  static bool? turboDetachStateEachUpdate;
  static bool? turboDequantKvToQueryDType;
  static bool? decoderHiddenDetach;
  static bool? decodeLogitsDetach;
  static bool? disableFastSingleTokenMrope;
  static int? visionAttentionChunk;
  static int? visionMlpChunk;
  static int? visionProjectorChunk;
  static int? prefillChunk;
  static String? metalCapturePath;
  static int? metalCaptureStartOffset;
  static int? metalCaptureStopOffset;
  static void Function(String message)? traceSink;

  static void reset() {
    kvQuantScheme = null;
    kvBits = null;
    turboBits = null;
    turboStart = null;
    turboDensePrefill = null;
    uniformQuantizedPrefill = null;
    turboDisableFusedKv = null;
    turboDisableFusedDecode = null;
    turboDisableSingleQuant = null;
    turboDisableFastScore = null;
    turboDisableFastValue = null;
    turboUniformLastLayer = null;
    turboQuantizeLastLayer = null;
    turboUpdateTrace = null;
    turboCompactBudget = null;
    turboCompactKeepRecent = null;
    turboCompactInterval = null;
    turboCompactKeepPrefix = null;
    turboCompactHysteresis = null;
    memoryLimitBytes = null;
    cacheLimitBytes = null;
    wiredLimitBytes = null;
    decoderLayerEval = null;
    forceGpuSynchronizeAfterLayerEval = null;
    aggressiveCacheClear = null;
    decodeCacheEvalInterval = null;
    decoderPerLayerTrace = null;
    decoderPerLayerTraceInterval = null;
    decoderSubstepTrace = null;
    decoderSubstepTraceLayer = null;
    decoderTailTrace = null;
    explicitLogitsEval = null;
    decodeStepExplicitLogitsEval = null;
    clearAfterCloseLogits = null;
    forceGpuSynchronizePerToken = null;
    forceDecodeLmHeadFloat32 = null;
    turboDetachStateEachUpdate = null;
    turboDequantKvToQueryDType = null;
    decoderHiddenDetach = null;
    decodeLogitsDetach = null;
    disableFastSingleTokenMrope = null;
    visionAttentionChunk = null;
    visionMlpChunk = null;
    visionProjectorChunk = null;
    prefillChunk = null;
    metalCapturePath = null;
    metalCaptureStartOffset = null;
    metalCaptureStopOffset = null;
    traceSink = null;
  }
}

extension PaddleOcrVlConfigDebug on PaddleOcrVlConfig {
  bool get enableDecoderLayerwiseEvalForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decoderLayerEval;
    if (debug != null) return debug;
    final override = Platform.environment['DART_MLX_PADDLE_DECODER_LAYER_EVAL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    if (Platform.isIOS && kvCacheQuantSchemeForCurrentPlatform == 'turboquant') {
      return false;
    }
    return Platform.isIOS;
  }

  bool get enableDecoderPerLayerTraceForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decoderPerLayerTrace;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TRACE_DECODER_LAYERS'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return false;
  }

  int get decoderPerLayerTraceIntervalForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decoderPerLayerTraceInterval;
    if (debug != null && debug > 0) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TRACE_DECODER_INTERVAL'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null && parsed > 0) return parsed;
    }
    return 32;
  }

  bool get enableDecoderSubstepTraceForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decoderSubstepTrace;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TRACE_DECODER_SUBSTEPS'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return false;
  }

  int get decoderSubstepTraceLayerForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decoderSubstepTraceLayer;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_TRACE_DECODER_SUBSTEP_LAYER'];
    if (override != null) {
      final parsed = int.tryParse(override);
      if (parsed != null) return parsed;
    }
    return numHiddenLayers - 1;
  }

  bool get enableDecoderTailTraceForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decoderTailTrace;
    if (debug != null) return debug;
    final override = Platform.environment['DART_MLX_PADDLE_TRACE_DECODER_TAIL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return false;
  }

  bool get enableExplicitLogitsEvalForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.explicitLogitsEval;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_EXPLICIT_LOGITS_EVAL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return true;
  }

  bool get enableDecodeStepExplicitLogitsEvalForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decodeStepExplicitLogitsEval;
    if (debug != null) return debug;
    return enableExplicitLogitsEvalForCurrentPlatform;
  }

  bool get forceDecodeLmHeadFloat32ForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.forceDecodeLmHeadFloat32;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_FORCE_DECODE_LM_HEAD_F32'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return Platform.isIOS;
  }

  bool get enableClearAfterCloseLogitsForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.clearAfterCloseLogits;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_CLEAR_AFTER_CLOSE_LOGITS'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    if (Platform.isIOS) {
      return true;
    }
    return false;
  }

  bool get enableForceGpuSynchronizePerTokenForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.forceGpuSynchronizePerToken;
    if (debug != null) return debug;
    final override = Platform.environment['DART_MLX_PADDLE_SYNC_EACH_TOKEN'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return false;
  }

  bool get enableDecoderHiddenDetachForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decoderHiddenDetach;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_DECODER_HIDDEN_DETACH'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    if (Platform.isIOS) {
      // iPhone decode can crash when a lazy layer output still references
      // intermediates that are closed immediately after residual2.
      return true;
    }
    return false;
  }

  bool get enableForceGpuSynchronizeAfterLayerEvalForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.forceGpuSynchronizeAfterLayerEval;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_SYNC_AFTER_LAYER_EVAL'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return false;
  }

  bool get enableDecodeLogitsDetachForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.decodeLogitsDetach;
    if (debug != null) return debug;
    final override =
        Platform.environment['DART_MLX_PADDLE_DECODE_LOGITS_DETACH'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    if (Platform.isIOS) {
      return false;
    }
    return false;
  }

  bool get enableAggressiveCacheClearingForCurrentPlatform {
    final debug = PaddleOcrVlDebugOverrides.aggressiveCacheClear;
    if (debug != null) return debug;
    final override = Platform.environment['DART_MLX_PADDLE_CLEAR_CACHE'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    if (Platform.isIOS && kvCacheQuantSchemeForCurrentPlatform == 'turboquant') {
      return false;
    }
    return Platform.isIOS;
  }

  bool get forceFloat32VisionForCurrentPlatform {
    final override = Platform.environment['DART_MLX_PADDLE_FORCE_VISION_F32'];
    if (override != null) {
      return override == '1' || override.toLowerCase() == 'true';
    }
    return false;
  }
}
