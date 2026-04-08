import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'audio_enc.dart';
import 'bpe.dart';
import 'config.dart';
import 'mel.dart';
import 'mrope.dart';
import 'text_dec.dart';

const String _asrTextTag = '<asr_text>';

/// Result of an incremental streaming decode step.
class AsrStreamDecodeResult {
  const AsrStreamDecodeResult({required this.text, required this.tokenIds});

  /// Decoded text for this chunk.
  final String text;

  /// Raw token IDs generated for this chunk.
  final List<int> tokenIds;
}

/// Persistent state for incremental streaming ASR.
///
/// Holds the KV cache and position counter across chunk decodes.
/// Call [close] when the streaming session ends.
class AsrStreamState {
  AsrStreamState._(this._cache, this._position, this._numLayers);

  AsrKvCache _cache;
  int _position;
  final int _numLayers;

  /// Current KV cache position (total tokens processed so far).
  int get position => _position;

  /// Reset decoder state (e.g. after context trimming).
  void reset() {
    _cache.close();
    _cache = AsrKvCache(_numLayers);
    _position = 0;
  }

  void close() => _cache.close();
}

/// Full Qwen3-ASR inference pipeline.
///
/// Supports both one-shot [transcribe] and incremental streaming via
/// [createStreamState] + [decodeChunk].
///
/// Prompt template matches the reference implementation (moona3k/mlx-qwen3-asr):
///   <|im_start|>system\n{context}<|im_end|>\n
///   <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
///   <|im_start|>assistant\n[language {lang}<asr_text>]
///
/// Audio features are injected into placeholder positions using cumulative-sum
/// indexing (matching _inject_audio_features in the reference).
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

  // ── Tokenizer access ──

  /// Encode text to token IDs.
  List<int> tokenize(String text) => _tokenizer.encode(text);

  /// Decode token IDs to text.
  String detokenize(List<int> ids) => _tokenizer.decode(ids);

  // ── One-shot transcription ──

  /// Transcribe raw PCM audio (16kHz Float32) to text.
  String transcribe(
    Float32List audio, {
    int maxNewTokens = 448,
    String locale = 'auto',
  }) {
    final ids = transcribeToIds(
      audio,
      maxNewTokens: maxNewTokens,
      locale: locale,
    );
    return _tokenizer.decode(ids);
  }

  /// Transcribe and return raw token IDs (before BPE decode).
  List<int> transcribeToIds(
    Float32List audio, {
    int maxNewTokens = 448,
    String locale = 'auto',
  }) {
    final melSpec = _mel.compute(audio);
    final audioFeatures = _audioEncoder.encode(melSpec);
    melSpec.close();
    MlxRuntime.evalAll([audioFeatures]);
    try {
      return _generate(
        audioFeatures,
        maxNewTokens: maxNewTokens,
        locale: locale,
      );
    } finally {
      audioFeatures.close();
    }
  }

  // ── Incremental streaming ──

  /// Create a new streaming state for incremental chunk-based decoding.
  AsrStreamState createStreamState() {
    final cache = _textDecoder.createCache();
    return AsrStreamState._(cache, 0, config.textNumLayers);
  }

  /// Decode a single audio chunk incrementally, reusing KV cache.
  ///
  /// First chunk builds the full prompt (system + user + assistant prefix);
  /// subsequent chunks build a follow-up user turn appended to existing KV.
  AsrStreamDecodeResult decodeChunk(
    Float32List audio, {
    required AsrStreamState state,
    int maxNewTokens = 448,
    String locale = 'auto',
  }) {
    final melSpec = _mel.compute(audio);
    final audioFeatures = _audioEncoder.encode(melSpec);
    melSpec.close();
    MlxRuntime.evalAll([audioFeatures]);
    try {
      final nAudioTokens = audioFeatures.shape[1];
      final isFirst = state._position == 0;

      // Build prompt token IDs with audio_pad placeholders.
      final promptIds = isFirst
          ? _buildPromptTokens(nAudioTokens, locale: locale)
          : _buildFollowupTokens(nAudioTokens, locale: locale);
      final promptLen = promptIds.length;

      // Prefill: embed, inject audio, run decoder.
      final posIds = buildAsrPositionIds(state._position, promptLen);
      final logits = _prefill(promptIds, audioFeatures, posIds, state._cache);
      posIds.close();
      MlxRuntime.evalAll([logits]);

      // Autoregressive decode.
      final startPos = state._position + promptLen;
      final generated = _decodeTokens(
        logits,
        cache: state._cache,
        startPos: startPos,
        maxNewTokens: maxNewTokens,
      );
      logits.close();

      state._position += promptLen + generated.length;
      final text = _tokenizer.decode(generated);
      return AsrStreamDecodeResult(text: text, tokenIds: generated);
    } finally {
      audioFeatures.close();
    }
  }

  // ── Core generation (one-shot) ──

  List<int> _generate(
    MlxArray audioFeatures, {
    required int maxNewTokens,
    required String locale,
  }) {
    final cache = _textDecoder.createCache();
    try {
      final nAudioTokens = audioFeatures.shape[1];
      final promptIds = _buildPromptTokens(nAudioTokens, locale: locale);
      final promptLen = promptIds.length;
      final posIds = buildAsrPositionIds(0, promptLen);
      final logits = _prefill(promptIds, audioFeatures, posIds, cache);
      posIds.close();
      MlxRuntime.evalAll([logits]);

      final generated = _decodeTokens(
        logits,
        cache: cache,
        startPos: promptLen,
        maxNewTokens: maxNewTokens,
      );
      logits.close();
      return generated;
    } finally {
      cache.close();
    }
  }

  // ── Prefill with audio injection ──

  /// Embed prompt tokens, inject audio features at placeholder positions,
  /// then run the decoder to produce logits for the last position.
  MlxArray _prefill(
    List<int> promptIds,
    MlxArray audioFeatures,
    MlxArray positionIds,
    AsrKvCache cache,
  ) {
    final embeds = _textDecoder.embed(promptIds);
    final injected = _injectAudioFeatures(embeds, audioFeatures, promptIds);
    embeds.close();
    MlxRuntime.evalAll([injected]);

    final hidden = _textDecoder.forward(
      injected,
      positionIds: positionIds,
      cache: cache,
    );
    injected.close();
    final logits = _textDecoder.lmHead(hidden);
    hidden.close();
    return logits;
  }

  /// Replace audio_pad placeholder embeddings with actual audio features.
  ///
  /// Uses cumulative-sum indexing matching the reference implementation:
  /// for each placeholder position, cum_idx maps it to the corresponding
  /// audio feature vector.
  MlxArray _injectAudioFeatures(
    MlxArray embeds,
    MlxArray audioFeatures,
    List<int> promptIds,
  ) {
    final padId = config.audioPadTokenId;
    final seqLen = promptIds.length;
    final hiddenDim = embeds.shape[2];

    // Build boolean mask: true where token == audio_pad.
    final maskBools = [for (final id in promptIds) id == padId];
    final audioCount = maskBools.where((b) => b).length;
    if (audioCount == 0) return embeds;

    // Build cumulative index: maps each position to an audio feature index.
    final cumIdx = List<int>.filled(seqLen, 0);
    var cumSum = 0;
    for (var i = 0; i < seqLen; i++) {
      if (maskBools[i]) cumSum++;
      cumIdx[i] = (cumSum - 1).clamp(0, audioCount - 1);
    }

    // Gather audio features for every position using cumIdx.
    // audioFeatures: [1, nAudio, D] → squeeze batch → [nAudio, D]
    final audioFlat = audioFeatures.reshape([
      audioFeatures.shape[1],
      hiddenDim,
    ]);
    final idxArr = MlxArray.fromInt32List(cumIdx, shape: [seqLen]);
    final audioExpanded = audioFlat.take(idxArr, axis: 0); // [seqLen, D]
    idxArr.close();
    audioFlat.close();
    final audioExp3d = audioExpanded.reshape([1, seqLen, hiddenDim]);
    audioExpanded.close();

    // Build 3D boolean mask: [1, seqLen, 1] for broadcast.
    final mask3d = MlxArray.fromBoolList(maskBools, shape: [1, seqLen, 1]);
    final result = mx.where(mask3d, audioExp3d, embeds);
    mask3d.close();
    audioExp3d.close();
    return result;
  }

  // ── Autoregressive token decode ──

  /// Decode tokens autoregressively from initial logits.
  ///
  /// The caller owns [initialLogits] and must close it after this returns.
  /// All internally-produced logits are closed before return.
  List<int> _decodeTokens(
    MlxArray initialLogits, {
    required AsrKvCache cache,
    required int startPos,
    required int maxNewTokens,
  }) {
    final generated = <int>[];
    var position = startPos;

    // Sample first token from prefill logits.
    final firstId = _greedySample(initialLogits);
    if (_isEos(firstId)) return generated;
    generated.add(firstId);

    for (var step = 1; step <= maxNewTokens; step++) {
      if (_detectRepetition(generated)) break;
      final stepEmb = _textDecoder.embed([generated.last]);
      final stepPos = buildAsrPositionIds(position, 1);
      final stepOut = _textDecoder.forward(
        stepEmb,
        positionIds: stepPos,
        cache: cache,
      );
      stepEmb.close();
      stepPos.close();
      final logits = _textDecoder.lmHead(stepOut);
      stepOut.close();
      MlxRuntime.evalAll([logits]);

      final nextId = _greedySample(logits);
      logits.close();
      if (_isEos(nextId)) break;
      generated.add(nextId);
      position += 1;
    }
    return generated;
  }

  // ── Prompt building (placeholder token IDs) ──

  /// First-chunk prompt: system + user turn with audio placeholders.
  ///
  /// Template:
  ///   <|im_start|>system\n<|im_end|>\n
  ///   <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
  ///   <|im_start|>assistant\n[language {lang}<asr_text>]
  List<int> _buildPromptTokens(int nAudioTokens, {required String locale}) {
    final tokens = <int>[];
    // System message (empty context, matching reference).
    tokens.add(config.imStartTokenId);
    tokens.addAll(_tokenizer.encode('system\n'));
    tokens.add(config.imEndTokenId);
    tokens.add(config.newlineTokenId);
    // User message with audio placeholders.
    tokens.add(config.imStartTokenId);
    tokens.addAll(_tokenizer.encode('user\n'));
    tokens.add(config.audioStartTokenId);
    for (var i = 0; i < nAudioTokens; i++) {
      tokens.add(config.audioPadTokenId);
    }
    tokens.add(config.audioEndTokenId);
    tokens.add(config.imEndTokenId);
    tokens.add(config.newlineTokenId);
    // Assistant prefix.
    tokens.add(config.imStartTokenId);
    tokens.addAll(_tokenizer.encode('assistant\n'));
    // Optional language forcing.
    final langTokens = _languageForcingTokens(locale);
    if (langTokens != null) tokens.addAll(langTokens);
    return tokens;
  }

  /// Follow-up prompt: close previous assistant, new user turn with audio.
  ///
  /// Template:
  ///   <|im_end|>\n<|im_start|>user\n
  ///   <|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
  ///   <|im_start|>assistant\n[language {lang}<asr_text>]
  List<int> _buildFollowupTokens(int nAudioTokens, {required String locale}) {
    final tokens = <int>[];
    tokens.add(config.imEndTokenId);
    tokens.add(config.newlineTokenId);
    tokens.add(config.imStartTokenId);
    tokens.addAll(_tokenizer.encode('user\n'));
    tokens.add(config.audioStartTokenId);
    for (var i = 0; i < nAudioTokens; i++) {
      tokens.add(config.audioPadTokenId);
    }
    tokens.add(config.audioEndTokenId);
    tokens.add(config.imEndTokenId);
    tokens.add(config.newlineTokenId);
    tokens.add(config.imStartTokenId);
    tokens.addAll(_tokenizer.encode('assistant\n'));
    final langTokens = _languageForcingTokens(locale);
    if (langTokens != null) tokens.addAll(langTokens);
    return tokens;
  }

  /// Build language forcing suffix tokens, or null if auto-detect.
  List<int>? _languageForcingTokens(String locale) {
    final normalized = locale.trim().toLowerCase();
    final langName = switch (normalized) {
      '' || 'auto' => null,
      'zh' || 'zh-cn' || 'zh-hans' => 'Chinese',
      'zh-tw' || 'zh-hant' => 'Chinese',
      'en' || 'en-us' || 'en-gb' => 'English',
      'ja' || 'ja-jp' => 'Japanese',
      'ko' || 'ko-kr' => 'Korean',
      'de' || 'de-de' => 'German',
      'fr' || 'fr-fr' => 'French',
      'es' || 'es-es' || 'es-419' => 'Spanish',
      'ru' || 'ru-ru' => 'Russian',
      'ar' || 'ar-eg' => 'Arabic',
      'hi' || 'hi-in' => 'Hindi',
      'it' || 'it-it' => 'Italian',
      'pt' || 'pt-br' || 'pt-pt' => 'Portuguese',
      'tr' || 'tr-tr' => 'Turkish',
      'nl' || 'nl-nl' => 'Dutch',
      _ => locale.trim(),
    };
    if (langName == null) return null;
    return _tokenizer.encode('language $langName$_asrTextTag');
  }

  // ── Sampling & EOS ──

  int _greedySample(MlxArray logits) {
    final flat = logits.reshape([logits.size]);
    final idx = flat.argmax();
    final idxVal = idx.toList().cast<int>().first;
    idx.close();
    flat.close();
    return idxVal;
  }

  bool _isEos(int tokenId) => config.eosTokenIds.contains(tokenId);

  // ── Lifecycle ──

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

/// Detect repetitive decode patterns to halt hallucination loops.
///
/// Two-stage check:
/// 1. Single token repeated > 20 consecutive times.
/// 2. N-gram pattern (length 2–10) repeated > threshold/patternLen times.
bool _detectRepetition(List<int> tokens) {
  if (tokens.length < 20) return false;
  // Stage 1: single-token run.
  final last = tokens.last;
  var run = 0;
  for (var i = tokens.length - 1; i >= 0 && tokens[i] == last; i--) {
    run++;
  }
  if (run > 20) return true;
  // Stage 2: n-gram repetition.
  for (var n = 2; n <= 10 && n * 3 <= tokens.length; n++) {
    final pattern = tokens.sublist(tokens.length - n);
    var repeats = 0;
    for (var i = tokens.length - n; i >= n; i -= n) {
      var match = true;
      for (var j = 0; j < n; j++) {
        if (tokens[i - n + j] != pattern[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        repeats++;
      } else {
        break;
      }
    }
    if (repeats >= (30 ~/ n) + 1) return true;
  }
  return false;
}
