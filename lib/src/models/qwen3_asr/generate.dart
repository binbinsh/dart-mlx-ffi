import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'audio_enc.dart';
import 'bpe.dart';
import 'config.dart';
import 'mel.dart';
import 'text_dec.dart';

/// Full Qwen3-ASR inference pipeline.
///
/// Usage:
/// ```dart
/// final runner = Qwen3AsrRunner.load('/path/to/snapshot');
/// final text = runner.transcribe(pcmFloat32);
/// runner.close();
/// ```
final class Qwen3AsrRunner {
  Qwen3AsrRunner._(
    this.config,
    this._audioEncoder,
    this._textDecoder,
    this._mel,
    this._tokenizer,
    this._tensors,
  );

  /// Load all components from a snapshot directory.
  factory Qwen3AsrRunner.load(String snapshotPath) {
    final config = Qwen3AsrConfig.fromSnapshot(snapshotPath);
    final tensors = _loadAllTensors(snapshotPath);
    final audioEncoder = Qwen3AsrAudioEncoder.load(tensors, config);
    final textDecoder = Qwen3AsrTextDecoder.load(tensors, config);
    final mel = Qwen3AsrMelFrontend();
    final tokenizer = Qwen3AsrBpeTokenizer.load(snapshotPath);
    return Qwen3AsrRunner._(
      config,
      audioEncoder,
      textDecoder,
      mel,
      tokenizer,
      tensors,
    );
  }

  final Qwen3AsrConfig config;
  final Qwen3AsrAudioEncoder _audioEncoder;
  final Qwen3AsrTextDecoder _textDecoder;
  final Qwen3AsrMelFrontend _mel;
  final Qwen3AsrBpeTokenizer _tokenizer;
  final Map<String, MlxArray> _tensors;

  /// Transcribe raw PCM audio (16kHz Float32) to text.
  String transcribe(Float32List audio, {int maxNewTokens = 448}) {
    final ids = transcribeToIds(audio, maxNewTokens: maxNewTokens);
    return _tokenizer.decode(ids);
  }

  /// Transcribe and return raw token IDs (before BPE decode).
  List<int> transcribeToIds(Float32List audio, {int maxNewTokens = 448}) {
    // 1. Mel spectrogram.
    final melSpec = _mel.compute(audio);

    // 2. Audio encoding.
    final audioFeatures = _audioEncoder.encode(melSpec);
    melSpec.close();
    MlxRuntime.evalAll([audioFeatures]);

    try {
      // 3. Build prompt and generate.
      return _generate(audioFeatures, maxNewTokens: maxNewTokens);
    } finally {
      audioFeatures.close();
    }
  }

  /// Core generation loop.
  ///
  /// Prompt structure (Qwen3-ASR chat template):
  ///   <|im_start|>system\n<|im_end|>\n
  ///   <|im_start|>user\n<|audio_start|>[audio_features]<|audio_end|><|im_end|>\n
  ///   <|im_start|>assistant\n
  ///
  /// The im_start/im_end/audio_start/audio_end tokens are text embeddings;
  /// audio features are projected encoder outputs injected between
  /// audio_start and audio_end token embeddings.
  List<int> _generate(MlxArray audioFeatures, {required int maxNewTokens}) {
    final cache = _textDecoder.createCache();
    try {
      // Build combined embedding for the prefix.
      final prefixEmb = _buildPrefixEmbedding(audioFeatures);
      MlxRuntime.evalAll([prefixEmb]);

      // Prefill: run the full prefix through the decoder.
      final prefillOut = _textDecoder.forward(prefixEmb, cache: cache);
      prefixEmb.close();
      var logits = _textDecoder.lmHead(prefillOut);
      prefillOut.close();
      MlxRuntime.evalAll([logits]);

      // Autoregressive decode.
      final generated = <int>[];
      for (var step = 0; step < maxNewTokens; step++) {
        final nextId = _greedySample(logits);
        logits.close();
        if (_isEos(nextId)) break;
        generated.add(nextId);

        // Embed the new token and run one decode step.
        final stepEmb = _textDecoder.embed([nextId]);
        final stepOut = _textDecoder.forward(stepEmb, cache: cache);
        stepEmb.close();
        logits = _textDecoder.lmHead(stepOut);
        stepOut.close();
        MlxRuntime.evalAll([logits]);
      }
      logits.close();
      return generated;
    } finally {
      cache.close();
    }
  }

  /// Build the prefix embedding using the Qwen3-ASR chat template.
  ///
  /// Token sequence:
  ///   <|im_start|> "system" \n <|im_end|> \n
  ///   <|im_start|> "user" \n <|audio_start|> [audio_features] <|audio_end|> <|im_end|> \n
  ///   <|im_start|> "assistant" \n
  MlxArray _buildPrefixEmbedding(MlxArray audioFeatures) {
    final c = config;
    // Encode the text tokens for "system", "user", "assistant" via tokenizer.
    final systemIds = _tokenizer.encode('system');
    final userIds = _tokenizer.encode('user');
    final assistantIds = _tokenizer.encode('assistant');
    final nl = c.newlineTokenId;

    // System block: <|im_start|> system \n <|im_end|> \n
    final systemTokens = [
      c.imStartTokenId,
      ...systemIds,
      nl,
      c.imEndTokenId,
      nl,
    ];
    // User block prefix: <|im_start|> user \n <|audio_start|>
    final userPrefix = [c.imStartTokenId, ...userIds, nl, c.audioStartTokenId];
    // User block suffix: <|audio_end|> <|im_end|> \n
    final userSuffix = [c.audioEndTokenId, c.imEndTokenId, nl];
    // Assistant block: <|im_start|> assistant \n
    final assistantTokens = [c.imStartTokenId, ...assistantIds, nl];

    final beforeAudio = [...systemTokens, ...userPrefix];
    final afterAudio = [...userSuffix, ...assistantTokens];

    final beforeEmb = _textDecoder.embed(beforeAudio);
    final afterEmb = _textDecoder.embed(afterAudio);

    final combined = mx.concatenate([
      beforeEmb,
      audioFeatures,
      afterEmb,
    ], axis: 1);
    beforeEmb.close();
    afterEmb.close();
    return combined;
  }

  /// Greedy argmax sampling.
  int _greedySample(MlxArray logits) {
    // logits: [1, vocabSize] or [1, 1, vocabSize].
    final flat = logits.reshape([logits.size]);
    final idx = flat.argmax();
    final idxVal = idx.toList().cast<int>().first;
    idx.close();
    flat.close();
    return idxVal;
  }

  bool _isEos(int tokenId) => config.eosTokenIds.contains(tokenId);

  void close() {
    _audioEncoder.close();
    _textDecoder.close();
    _mel.close();
    for (final tensor in _tensors.values) {
      tensor.close();
    }
  }

  static Map<String, MlxArray> _loadAllTensors(String path) {
    final dir = Directory(path);
    final files =
        dir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.safetensors'))
            .toList()
          ..sort((a, b) => a.path.compareTo(b.path));
    if (files.isEmpty) {
      throw StateError('No safetensors files found in $path');
    }
    final merged = <String, MlxArray>{};
    for (final file in files) {
      final loaded = mx.io.loadSafetensors(file.path);
      for (final entry in loaded.tensors.entries) {
        if (merged.containsKey(entry.key)) {
          throw StateError('Duplicate tensor: ${entry.key}');
        }
        merged[entry.key] = entry.value;
      }
    }
    return merged;
  }
}
