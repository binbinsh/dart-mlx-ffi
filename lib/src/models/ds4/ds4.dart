library;

import '../shared/model_spec.dart';
import '../shared/runtime_metadata.dart';

const String ds4ModelId = 'deepseek_v4_flash_ds4';
const String ds4HuggingFaceRepo = 'antirez/deepseek-v4-gguf';
const String ds4SourceRepository = 'https://github.com/antirez/ds4';

const String ds4Q2File =
    'DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf';
const String ds4Q4File =
    'DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf';
const String ds4MtpFile = 'DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf';

/// The ds4 GGUF quantization families published by antirez/ds4.
enum Ds4Quant {
  /// 2-bit routed experts; intended for 128 GB RAM Apple Silicon machines.
  q2,

  /// 4-bit routed experts; intended for 256 GB+ RAM Apple Silicon machines.
  q4,
}

extension Ds4QuantInfo on Ds4Quant {
  String get cliName => switch (this) {
    Ds4Quant.q2 => 'q2',
    Ds4Quant.q4 => 'q4',
  };

  String get fileName => switch (this) {
    Ds4Quant.q2 => ds4Q2File,
    Ds4Quant.q4 => ds4Q4File,
  };

  int get approximateBytes => switch (this) {
    Ds4Quant.q2 => 81 * 1024 * 1024 * 1024,
    Ds4Quant.q4 => 153 * 1024 * 1024 * 1024,
  };
}

/// Model descriptor for ds4's DeepSeek V4 Flash runtime.
///
/// ds4 is intentionally modeled as a bundled ds4.c FFI runtime. It only runs
/// the GGUF files published for antirez/ds4 and is currently macOS/Metal-only.
final ModelSpec deepSeekV4FlashDs4Spec = ModelSpec(
  id: ds4ModelId,
  family: 'DeepSeek V4 Flash',
  modalities: const [ModelModality.textGeneration],
  description: 'DeepSeek V4 Flash through bundled ds4.c Dart FFI',
  requiredFiles: const ['ds4flash.gguf'],
  requiredTags: const ['ds4', 'ffi', 'gguf', 'metal'],
  sizeHint: Ds4Quant.q2.approximateBytes,
  supportLevel: SupportLevel.staging,
  metadata: const <String, Object?>{
    'runtimeScope': 'bundled-ds4-ffi',
    'sourceRepository': ds4SourceRepository,
    'huggingFaceRepo': ds4HuggingFaceRepo,
    'platform': 'macos',
    'backend': 'metal',
    'minimumRamGb': 128,
  },
);

final class Ds4ModelArtifacts {
  const Ds4ModelArtifacts._();

  static RuntimeArtifact main(Ds4Quant quant) => switch (quant) {
    Ds4Quant.q2 => q2,
    Ds4Quant.q4 => q4,
  };

  static const RuntimeArtifact q2 = RuntimeArtifact(
    engine: RuntimeEngine.mlx,
    path: 'hf://$ds4HuggingFaceRepo/$ds4Q2File',
    sourceUri: 'hf://$ds4HuggingFaceRepo/$ds4Q2File',
    format: 'ds4-gguf',
    targetPlatforms: ['macos'],
    accelerators: [Accelerator.gpu],
    metadata: <String, Object?>{
      'source': 'huggingface',
      'runtimeEngine': 'ds4',
      'nativeBackend': 'metal',
      'modelId': ds4ModelId,
      'repo': ds4HuggingFaceRepo,
      'artifact': ds4Q2File,
      'quant': 'q2',
      'approxBytes': 86973087744,
    },
  );

  static const RuntimeArtifact q4 = RuntimeArtifact(
    engine: RuntimeEngine.mlx,
    path: 'hf://$ds4HuggingFaceRepo/$ds4Q4File',
    sourceUri: 'hf://$ds4HuggingFaceRepo/$ds4Q4File',
    format: 'ds4-gguf',
    targetPlatforms: ['macos'],
    accelerators: [Accelerator.gpu],
    metadata: <String, Object?>{
      'source': 'huggingface',
      'runtimeEngine': 'ds4',
      'nativeBackend': 'metal',
      'modelId': ds4ModelId,
      'repo': ds4HuggingFaceRepo,
      'artifact': ds4Q4File,
      'quant': 'q4',
      'minimumRamGb': 256,
      'approxBytes': 164282499072,
    },
  );

  static const RuntimeArtifact mtp = RuntimeArtifact(
    engine: RuntimeEngine.mlx,
    path: 'hf://$ds4HuggingFaceRepo/$ds4MtpFile',
    sourceUri: 'hf://$ds4HuggingFaceRepo/$ds4MtpFile',
    format: 'ds4-mtp-gguf',
    targetPlatforms: ['macos'],
    accelerators: [Accelerator.gpu],
    metadata: <String, Object?>{
      'source': 'huggingface',
      'runtimeEngine': 'ds4',
      'nativeBackend': 'metal',
      'modelId': ds4ModelId,
      'repo': ds4HuggingFaceRepo,
      'artifact': ds4MtpFile,
      'component': 'mtp',
      'approxBytes': 3758096384,
    },
  );
}

enum Ds4Thinking { disabled, high, max }

enum Ds4ChatChunkKind { content, reasoning, done }

final class Ds4ChatChunk {
  const Ds4ChatChunk({
    required this.kind,
    this.text = '',
    this.payload = const <String, Object?>{},
  });

  final Ds4ChatChunkKind kind;
  final String text;
  final Map<String, Object?> payload;
}

final class Ds4ChatRequest {
  const Ds4ChatRequest({
    required this.messages,
    this.maxTokens = 4096,
    this.temperature = 1.0,
    this.topP,
    this.topK,
    this.minP,
    this.seed,
    this.thinking = Ds4Thinking.high,
  });

  final List<Map<String, Object?>> messages;
  final int maxTokens;
  final double temperature;
  final double? topP;
  final int? topK;
  final double? minP;
  final int? seed;
  final Ds4Thinking thinking;
}

abstract interface class Ds4ChatBackend {
  Stream<Ds4ChatChunk> stream(Ds4ChatRequest request);
  Future<void> close();
}
