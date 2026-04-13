library;

// ── Shared abstractions ──
export 'src/models/shared/model_spec.dart';
export 'src/models/shared/manifest.dart';
export 'src/models/shared/cache.dart';
export 'src/models/shared/session.dart';
export 'src/models/shared/tuning.dart';
export 'src/models/shared/metal_gate.dart';
export 'src/models/shared/stream_acc.dart';
export 'src/models/shared/kv_store.dart';
export 'src/models/shared/embedding.dart';
export 'src/models/shared/tensor_map.dart';

// ── Model families ──
export 'src/models/kitten_tts/kitten_tts.dart'
    show
        KittenTtsEngine,
        KittenTtsResult,
        EspeakG2p,
        KittenFrontRunner,
        KittenFrontResult,
        KittenDecoder,
        ModelConfig,
        basicEnglishTokenize,
        TextCleaner,
        buildInputIdsFromPhonemes,
        buildInputArrayFromPhonemes,
        buildInputIdsFromText,
        buildInputArrayFromText;
export 'src/models/qwen2_5/qwen2_5.dart' show QwenRunner, QwenConfig;
export 'src/models/paddle_ocr_vl/paddle_ocr_vl.dart'
    show PaddleOcrVlRunner, PaddleOcrVlConfig;
export 'src/models/qwen3_5/qwen3_5.dart'
    show Qwen3_5Runner, Qwen3_5Config, Qwen3_5VisionConfig, Qwen35TopK;
export 'src/models/synthetic/synthetic.dart'
    show
        runSyntheticModelBenchmarks,
        printSyntheticBenchmarkReport,
        readSyntheticBenchArg;
export 'src/models/qwen3_asr/qwen3_asr.dart';
export 'src/models/silero_vad/silero_vad.dart';
