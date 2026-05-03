import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import '../../runtime/native_runtime.dart';
import '../../runtime/runtime.dart';
import '../shared/model_spec.dart';
import '../shared/runtime_metadata.dart';
import 'bpe.dart';
import 'config.dart';
import 'mel_cpu.dart';
import 'prompt.dart';

/// Embedding source used by Qwen3-ASR native decoder-step execution.
abstract interface class Qwen3AsrEmbeddingSource {
  int get hiddenSize;

  Float32List lookup(int tokenId);

  void close();
}

/// Memory-mapped-style row lookup for `embed_tokens.bin` sidecars.
final class Qwen3AsrEmbeddingTable implements Qwen3AsrEmbeddingSource {
  Qwen3AsrEmbeddingTable._({
    required int vocabSize,
    required this.hiddenSize,
    required int elementBytes,
    RandomAccessFile? file,
    Float32List? memory,
  }) : _vocabSize = vocabSize,
       _elementBytes = elementBytes,
       _file = file,
       _memory = memory;

  factory Qwen3AsrEmbeddingTable.fromFile(
    String path, {
    required int vocabSize,
    required int hiddenSize,
  }) {
    final file = File(path);
    if (!file.existsSync()) {
      throw StateError('Qwen3-ASR embedding table not found: $path');
    }
    final length = file.lengthSync();
    final rows = vocabSize * hiddenSize;
    final elementBytes = switch (length) {
      final value when value == rows * 2 => 2,
      final value when value == rows * 4 => 4,
      _ => throw StateError(
        'Unexpected Qwen3-ASR embedding size $length for '
        '[$vocabSize, $hiddenSize] at $path.',
      ),
    };
    return Qwen3AsrEmbeddingTable._(
      vocabSize: vocabSize,
      hiddenSize: hiddenSize,
      elementBytes: elementBytes,
      file: file.openSync(),
    );
  }

  factory Qwen3AsrEmbeddingTable.fromFloat32Rows({
    required int vocabSize,
    required int hiddenSize,
    required Float32List values,
  }) {
    if (values.length != vocabSize * hiddenSize) {
      throw ArgumentError.value(
        values.length,
        'values',
        'Expected vocabSize * hiddenSize values.',
      );
    }
    return Qwen3AsrEmbeddingTable._(
      vocabSize: vocabSize,
      hiddenSize: hiddenSize,
      elementBytes: 4,
      memory: values,
    );
  }

  final int _vocabSize;
  @override
  final int hiddenSize;
  final int _elementBytes;
  final RandomAccessFile? _file;
  final Float32List? _memory;

  @override
  Float32List lookup(int tokenId) {
    if (tokenId < 0 || tokenId >= _vocabSize) {
      throw RangeError.range(tokenId, 0, _vocabSize - 1, 'tokenId');
    }
    final memory = _memory;
    if (memory != null) {
      return Float32List.fromList(
        memory.sublist(tokenId * hiddenSize, (tokenId + 1) * hiddenSize),
      );
    }
    final file = _file;
    if (file == null) {
      throw StateError('Qwen3-ASR embedding table is closed.');
    }
    final byteCount = hiddenSize * _elementBytes;
    file.setPositionSync(tokenId * byteCount);
    final raw = file.readSync(byteCount);
    if (_elementBytes == 4) {
      return Float32List.fromList(
        raw.buffer.asFloat32List(raw.offsetInBytes, hiddenSize),
      );
    }
    final bytes = ByteData.sublistView(raw);
    final out = Float32List(hiddenSize);
    for (var i = 0; i < hiddenSize; i++) {
      out[i] = _float16ToFloat32(bytes.getUint16(i * 2, Endian.little));
    }
    return out;
  }

  @override
  void close() {
    _file?.closeSync();
  }
}

/// Native ONNX/Core ML/LiteRT Qwen3-ASR component orchestrator.
///
/// ONNX bundles such as `andrewleech/qwen3-asr-1.7b-onnx` expose encoder,
/// decoder-init, decoder-step, tokenizer, and embedding sidecars instead of a
/// single graph. This runner owns the model-level ASR control flow: CPU mel
/// preprocessing, prompt construction, KV-cache passing, and greedy decoding.
final class Qwen3AsrNativeRunner {
  Qwen3AsrNativeRunner({
    required this.config,
    required Qwen3AsrBpeTokenizer tokenizer,
    required ModelSession encoder,
    required ModelSession decoderInit,
    required ModelSession decoderStep,
    required Qwen3AsrEmbeddingSource embeddings,
    int? encoderMelFrames,
    Qwen3AsrCpuMelFrontend? mel,
  }) : _tokenizer = tokenizer,
       _encoder = encoder,
       _decoderInit = decoderInit,
       _decoderStep = decoderStep,
       _embeddings = embeddings,
       _encoderMelFrames = encoderMelFrames,
       _mel = mel ?? Qwen3AsrCpuMelFrontend();

  factory Qwen3AsrNativeRunner.loadOnnxBundle(
    String bundlePath, {
    RuntimeOptions options = const RuntimeOptions(diagnostics: true),
    bool preferInt4 = true,
    ModelRuntime? runtime,
  }) {
    final config = Qwen3AsrConfig.fromSnapshot(bundlePath);
    final tokenizer = Qwen3AsrBpeTokenizer.load(bundlePath);
    final nativeRuntime = runtime ?? NativeModelRuntime(RuntimeEngine.onnx);
    final effectiveOptions = _defaultOnnxOptions(options);
    final suffix =
        preferInt4 && File('$bundlePath/decoder_step.int4.onnx').existsSync()
        ? '.int4.onnx'
        : '.onnx';
    final encoder = _loadComponent(
      nativeRuntime,
      bundlePath,
      'encoder$suffix',
      RuntimeEngine.onnx,
      effectiveOptions,
    );
    final decoderInit = _loadComponent(
      nativeRuntime,
      bundlePath,
      'decoder_init$suffix',
      RuntimeEngine.onnx,
      effectiveOptions,
    );
    final decoderStep = _loadComponent(
      nativeRuntime,
      bundlePath,
      'decoder_step$suffix',
      RuntimeEngine.onnx,
      effectiveOptions,
    );
    final embeddings = Qwen3AsrEmbeddingTable.fromFile(
      '$bundlePath/embed_tokens.bin',
      vocabSize: config.textVocabSize,
      hiddenSize: config.textHiddenSize,
    );
    return Qwen3AsrNativeRunner(
      config: config,
      tokenizer: tokenizer,
      encoder: encoder,
      decoderInit: decoderInit,
      decoderStep: decoderStep,
      embeddings: embeddings,
    );
  }

  /// Load a same-model LiteRT component bundle.
  ///
  /// Expected files:
  /// - `encoder.tflite`
  /// - `decoder_init.tflite`
  /// - `decoder_step.tflite`
  /// - `embed_tokens.bin`
  /// - tokenizer/config sidecars consumed by [Qwen3AsrConfig] and
  ///   [Qwen3AsrBpeTokenizer]
  factory Qwen3AsrNativeRunner.loadLiteRtBundle(
    String bundlePath, {
    RuntimeOptions options = const RuntimeOptions(
      diagnostics: true,
      prefer: [Accelerator.gpu, Accelerator.cpu],
    ),
    ModelRuntime? runtime,
  }) {
    final config = Qwen3AsrConfig.fromSnapshot(bundlePath);
    final tokenizer = Qwen3AsrBpeTokenizer.load(bundlePath);
    final nativeRuntime = runtime ?? NativeModelRuntime(RuntimeEngine.litert);
    final encoder = _loadComponent(
      nativeRuntime,
      bundlePath,
      'encoder.tflite',
      RuntimeEngine.litert,
      options,
    );
    final decoderInit = _loadComponent(
      nativeRuntime,
      bundlePath,
      'decoder_init.tflite',
      RuntimeEngine.litert,
      options,
    );
    final decoderStep = _loadComponent(
      nativeRuntime,
      bundlePath,
      'decoder_step.tflite',
      RuntimeEngine.litert,
      options,
    );
    final embeddings = Qwen3AsrEmbeddingTable.fromFile(
      '$bundlePath/embed_tokens.bin',
      vocabSize: config.textVocabSize,
      hiddenSize: config.textHiddenSize,
    );
    return Qwen3AsrNativeRunner(
      config: config,
      tokenizer: tokenizer,
      encoder: encoder,
      decoderInit: decoderInit,
      decoderStep: decoderStep,
      embeddings: embeddings,
      encoderMelFrames: _qwen3AsrLiteRtEncoderMelFrames,
    );
  }

  static const int _qwen3AsrLiteRtEncoderMelFrames = 3000;

  final Qwen3AsrConfig config;
  final Qwen3AsrBpeTokenizer _tokenizer;
  final Qwen3AsrCpuMelFrontend _mel;
  final ModelSession _encoder;
  final ModelSession _decoderInit;
  final ModelSession _decoderStep;
  final Qwen3AsrEmbeddingSource _embeddings;
  final int? _encoderMelFrames;

  List<int> tokenize(String text) => _tokenizer.encode(text);

  String detokenize(List<int> ids) => _tokenizer.decode(ids);

  Map<String, Object?> componentDiagnostics() {
    return <String, Object?>{
      'engine': _encoder.diagnostics['engine'],
      'model_level_runner': 'Qwen3AsrNativeRunner',
      'text_vocab_size': config.textVocabSize,
      'text_hidden_size': config.textHiddenSize,
      'audio_max_source_positions': config.audioMaxSourcePositions,
      'encoder': _encoder.diagnostics,
      'decoder_init': _decoderInit.diagnostics,
      'decoder_step': _decoderStep.diagnostics,
    };
  }

  String transcribe(
    Float32List audio, {
    int maxNewTokens = 448,
    String locale = 'auto',
  }) {
    return _tokenizer.decode(
      transcribeToIds(audio, maxNewTokens: maxNewTokens, locale: locale),
    );
  }

  List<int> transcribeToIds(
    Float32List audio, {
    int maxNewTokens = 448,
    String locale = 'auto',
  }) {
    if (maxNewTokens <= 0) return const <int>[];
    final mel = _mel.compute(audio);
    final encoderInputs = {
      _inputName(_encoder, const [
        'input_features',
        'mel',
        'mel_spectrogram',
        'features',
        'audio',
      ]): _channelFirstMelTensor(
        mel,
        targetFrames: _encoderMelFrames,
      ),
    };
    final encoderOut = _encoder.run(ModelInputs(encoderInputs));
    final audioFeatures = _outputTensor(encoderOut, const [
      'audio_features',
      'encoder_output',
      'last_hidden_state',
    ]);
    final nAudioTokens = audioFeatures.shape.length >= 2
        ? audioFeatures.shape[1]
        : 0;
    final promptIds = buildQwen3AsrPromptTokens(
      config,
      _tokenizer,
      nAudioTokens,
      locale: locale,
    );
    final promptLen = promptIds.length;
    final initOut = _decoderInit.run(
      ModelInputs({
        _inputName(_decoderInit, const ['input_ids']): RuntimeTensor.int64([
          1,
          promptLen,
        ], Int64List.fromList(promptIds)),
        _inputName(_decoderInit, const ['position_ids']): RuntimeTensor.int64([
          1,
          promptLen,
        ], _positions(0, promptLen)),
        _inputName(_decoderInit, const ['audio_features']): audioFeatures,
        _inputName(_decoderInit, const ['audio_offset']): RuntimeTensor.int64(
          const [1],
          Int64List.fromList([qwen3AsrAudioOffset(promptIds, config)]),
        ),
      }),
    );

    var pastKeys = _outputTensor(initOut, const ['present_keys']);
    var pastValues = _outputTensor(initOut, const ['present_values']);
    final generated = <int>[];
    var nextId = _greedySampleLast(_outputTensor(initOut, const ['logits']));
    if (_isEos(nextId)) return generated;
    generated.add(nextId);

    var position = promptLen;
    while (generated.length < maxNewTokens) {
      if (detectQwen3AsrRepetition(generated)) break;
      final embedding = _embeddings.lookup(nextId);
      final stepOut = _decoderStep.run(
        ModelInputs({
          _inputName(_decoderStep, const ['input_embeds']):
              RuntimeTensor.float32([1, 1, _embeddings.hiddenSize], embedding),
          _inputName(_decoderStep, const ['position_ids']): RuntimeTensor.int64(
            [1, 1],
            Int64List.fromList([position]),
          ),
          _inputName(_decoderStep, const ['past_keys']): pastKeys,
          _inputName(_decoderStep, const ['past_values']): pastValues,
        }),
      );
      pastKeys = _outputTensor(stepOut, const ['present_keys']);
      pastValues = _outputTensor(stepOut, const ['present_values']);
      nextId = _greedySampleLast(_outputTensor(stepOut, const ['logits']));
      if (_isEos(nextId)) break;
      generated.add(nextId);
      position += 1;
    }
    return generated;
  }

  void close() {
    _encoder.close();
    _decoderInit.close();
    _decoderStep.close();
    _embeddings.close();
  }

  bool _isEos(int tokenId) => config.eosTokenIds.contains(tokenId);
}

/// Stateful Core ML Qwen3-ASR component orchestrator.
///
/// The public 1.7B Core ML bundle uses `encoder.mlmodelc`,
/// `embedding.mlmodelc`, and a stateful single-token `decoder.mlmodelc`.
/// This runner feeds the prompt one token at a time, injects audio encoder rows
/// at `<|audio_pad|>` positions, and lets the native Core ML bridge maintain
/// the decoder KV state through `MLState`.
final class Qwen3AsrCoreMlRunner {
  Qwen3AsrCoreMlRunner({
    required this.config,
    required Qwen3AsrBpeTokenizer tokenizer,
    required ModelSession encoder,
    required ModelSession embedding,
    required ModelSession Function() decoderFactory,
    Qwen3AsrCpuMelFrontend? mel,
    this.maxSequenceLength = 1024,
  }) : _tokenizer = tokenizer,
       _encoder = encoder,
       _embedding = embedding,
       _decoderFactory = decoderFactory,
       _mel = mel ?? Qwen3AsrCpuMelFrontend();

  factory Qwen3AsrCoreMlRunner.loadCoreMlBundle(
    String bundlePath, {
    required String tokenizerPath,
    RuntimeOptions options = const RuntimeOptions(
      diagnostics: true,
      prefer: [Accelerator.ane],
    ),
    ModelRuntime? runtime,
  }) {
    final config = Qwen3AsrConfig.fromSnapshot(bundlePath);
    final tokenizer = Qwen3AsrBpeTokenizer.load(tokenizerPath);
    final nativeRuntime = runtime ?? NativeModelRuntime(RuntimeEngine.coreml);
    final encoder = _loadComponent(
      nativeRuntime,
      bundlePath,
      'encoder.mlmodelc',
      RuntimeEngine.coreml,
      options,
    );
    final embedding = _loadComponent(
      nativeRuntime,
      bundlePath,
      'embedding.mlmodelc',
      RuntimeEngine.coreml,
      options,
    );
    ModelSession createDecoder() => _loadComponent(
      nativeRuntime,
      bundlePath,
      'decoder.mlmodelc',
      RuntimeEngine.coreml,
      options,
    );
    return Qwen3AsrCoreMlRunner(
      config: config,
      tokenizer: tokenizer,
      encoder: encoder,
      embedding: embedding,
      decoderFactory: createDecoder,
    );
  }

  final Qwen3AsrConfig config;
  final int maxSequenceLength;
  final Qwen3AsrBpeTokenizer _tokenizer;
  final Qwen3AsrCpuMelFrontend _mel;
  final ModelSession _encoder;
  final ModelSession _embedding;
  final ModelSession Function() _decoderFactory;

  String detokenize(List<int> ids) => _tokenizer.decode(ids);

  Map<String, Object?> componentDiagnostics({bool includeDecoder = false}) {
    final diagnostics = <String, Object?>{
      'engine': 'coreml',
      'model_level_runner': 'Qwen3AsrCoreMlRunner',
      'text_vocab_size': config.textVocabSize,
      'text_hidden_size': config.textHiddenSize,
      'max_sequence_length': maxSequenceLength,
      'encoder': _encoder.diagnostics,
      'embedding': _embedding.diagnostics,
    };
    if (includeDecoder) {
      final decoder = _decoderFactory();
      try {
        diagnostics['decoder'] = decoder.diagnostics;
      } finally {
        decoder.close();
      }
    }
    return diagnostics;
  }

  String transcribe(
    Float32List audio, {
    int maxNewTokens = 448,
    String locale = 'auto',
  }) {
    return _tokenizer.decode(
      transcribeToIds(audio, maxNewTokens: maxNewTokens, locale: locale),
    );
  }

  List<int> transcribeToIds(
    Float32List audio, {
    int maxNewTokens = 448,
    String locale = 'auto',
  }) {
    if (maxNewTokens <= 0) return const <int>[];
    final mel = _mel.compute(audio);
    final coreMlMel = _transposeMelForCoreMl(mel);
    final encoderOut = _encoder.run(
      ModelInputs({
        _inputName(_encoder, const ['mel', 'input_features']):
            RuntimeTensor.float32([1, 128, mel.frameCount], coreMlMel),
      }),
    );
    final audioFeatures = _outputTensor(encoderOut, const [
      'audio_embeddings',
      'audio_features',
      'encoder_output',
    ]);
    final nAudioTokens = _sequenceLength(audioFeatures);
    final promptIds = buildQwen3AsrPromptTokens(
      config,
      _tokenizer,
      nAudioTokens,
      locale: locale,
    );
    final requestedLength = promptIds.length + maxNewTokens;
    if (requestedLength > maxSequenceLength) {
      throw StateError(
        'Qwen3-ASR Core ML request needs $requestedLength positions, '
        'but this decoder cache supports $maxSequenceLength.',
      );
    }

    final decoder = _decoderFactory();
    try {
      var audioIndex = 0;
      RuntimeTensor? logits;
      for (var position = 0; position < promptIds.length; position++) {
        final tokenId = promptIds[position];
        final embedding = tokenId == config.audioPadTokenId
            ? _audioFeatureRow(audioFeatures, audioIndex++)
            : _embedToken(tokenId);
        logits = _runCoreMlDecodeStep(decoder, embedding, position);
      }
      if (logits == null) return const <int>[];

      final generated = <int>[];
      var nextId = _greedySampleLast(logits);
      if (_isEos(nextId)) return generated;
      generated.add(nextId);

      var position = promptIds.length;
      while (generated.length < maxNewTokens && position < maxSequenceLength) {
        if (detectQwen3AsrRepetition(generated)) break;
        final embedding = _embedToken(nextId);
        logits = _runCoreMlDecodeStep(decoder, embedding, position);
        nextId = _greedySampleLast(logits);
        if (_isEos(nextId)) break;
        generated.add(nextId);
        position += 1;
      }
      return generated;
    } finally {
      decoder.close();
    }
  }

  void close() {
    _encoder.close();
    _embedding.close();
  }

  Float32List _embedToken(int tokenId) {
    final out = _embedding.run(
      ModelInputs({
        _inputName(_embedding, const ['token_id', 'input_ids']):
            RuntimeTensor.int32([1, 1], Int32List.fromList([tokenId])),
      }),
    );
    return _tensorAsFloat32(_outputTensor(out, const ['embedding']));
  }

  RuntimeTensor _runCoreMlDecodeStep(
    ModelSession decoder,
    Float32List embedding,
    int position,
  ) {
    return _outputTensor(
      decoder.run(
        ModelInputs({
          _inputName(decoder, const ['input_embeds']): RuntimeTensor.float32([
            1,
            1,
            config.textHiddenSize,
          ], embedding),
          _inputName(decoder, const ['position']): RuntimeTensor.int32([
            1,
          ], Int32List.fromList([position])),
          _inputName(decoder, const ['attention_mask']): RuntimeTensor.float32([
            1,
            1,
            1,
            maxSequenceLength,
          ], _attentionMask(position, maxSequenceLength)),
        }),
      ),
      const ['logits'],
    );
  }

  bool _isEos(int tokenId) => config.eosTokenIds.contains(tokenId);
}

ModelSession _loadComponent(
  ModelRuntime runtime,
  String root,
  String fileName,
  RuntimeEngine engine,
  RuntimeOptions options,
) {
  final path = '$root/$fileName';
  if (!File(path).existsSync() && !Directory(path).existsSync()) {
    throw StateError('Qwen3-ASR component not found: $path');
  }
  final spec = ModelSpec(
    id: 'qwen3_asr_$fileName',
    family: 'Qwen3-ASR',
    modalities: const [ModelModality.speechToText],
    platformArtifacts: {
      engine: RuntimeArtifact(engine: engine, path: path, format: engine.name),
    },
  );
  return runtime.load(
    ModelBundle(
      spec: spec,
      rootPath: '',
      artifact: spec.platformArtifacts[engine]!,
    ),
    options,
  );
}

RuntimeOptions _defaultOnnxOptions(RuntimeOptions options) {
  if (options.prefer.isNotEmpty) return options;
  return RuntimeOptions(
    engine: options.engine,
    prefer: Platform.isAndroid
        ? const [Accelerator.npu, Accelerator.gpu, Accelerator.cpu]
        : const [Accelerator.gpu, Accelerator.cpu],
    allowFallback: options.allowFallback,
    diagnostics: options.diagnostics,
    numThreads: options.numThreads,
    backendOptions: options.backendOptions,
    artifactResolver: options.artifactResolver,
  );
}

String _inputName(ModelSession session, List<String> preferred) {
  final names = _diagnosticNames(session, 'input_names');
  for (final name in preferred) {
    if (names.contains(name)) return name;
  }
  for (final preferredName in preferred) {
    final canonicalPreferred = _canonicalTensorName(preferredName);
    for (final name in names) {
      final canonicalName = _canonicalTensorName(name);
      if (canonicalName == canonicalPreferred ||
          canonicalName.endsWith(canonicalPreferred) ||
          canonicalName.contains(canonicalPreferred)) {
        return name;
      }
    }
  }
  return names.isNotEmpty ? names.first : preferred.first;
}

RuntimeTensor _outputTensor(ModelOutputs outputs, List<String> preferred) {
  for (final name in preferred) {
    final value = outputs.values[name];
    if (value is RuntimeTensor) return value;
  }
  for (final preferredName in preferred) {
    final canonicalPreferred = _canonicalTensorName(preferredName);
    for (final entry in outputs.values.entries) {
      final canonicalName = _canonicalTensorName(entry.key);
      if (canonicalName == canonicalPreferred ||
          canonicalName.endsWith(canonicalPreferred) ||
          canonicalName.contains(canonicalPreferred)) {
        final value = entry.value;
        if (value is RuntimeTensor) return value;
      }
    }
  }
  for (final value in outputs.values.values) {
    if (value is RuntimeTensor) return value;
  }
  throw StateError('Native Qwen3-ASR component returned no tensor outputs.');
}

String _canonicalTensorName(String value) {
  final withoutPort = value.split(':').first;
  final withoutScope = withoutPort.split('/').last;
  return withoutScope.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), '');
}

List<String> _diagnosticNames(ModelSession session, String key) {
  final raw = session.diagnostics[key];
  if (raw is List) {
    return raw.whereType<String>().toList(growable: false);
  }
  return const <String>[];
}

Int64List _positions(int start, int length) {
  return Int64List.fromList(List<int>.generate(length, (i) => start + i));
}

int _greedySampleLast(RuntimeTensor logits) {
  final values = _tensorAsFloat32(logits);
  if (values.isEmpty) {
    throw StateError('Cannot sample from empty logits.');
  }
  final vocab = logits.shape.isEmpty ? values.length : logits.shape.last;
  final rows = values.length ~/ vocab;
  final offset = math.max(0, rows - 1) * vocab;
  var bestIndex = 0;
  var bestValue = values[offset];
  for (var i = 1; i < vocab; i++) {
    final value = values[offset + i];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
}

Float32List _tensorAsFloat32(RuntimeTensor tensor) {
  if (tensor.dtype == RuntimeTensorDataType.float32) {
    return Float32List.fromList(tensor.asFloat32List());
  }
  if (tensor.dtype == RuntimeTensorDataType.float16) {
    final bytes = ByteData.sublistView(tensor.bytes);
    final out = Float32List(tensor.bytes.lengthInBytes ~/ 2);
    for (var i = 0; i < out.length; i++) {
      out[i] = _float16ToFloat32(bytes.getUint16(i * 2, Endian.little));
    }
    return out;
  }
  throw StateError(
    'Expected float32/float16 tensor, got ${tensor.dtype.name}.',
  );
}

Float32List _transposeMelForCoreMl(Qwen3AsrCpuMelTensor mel) {
  final frames = mel.frameCount;
  final out = Float32List(frames * 128);
  for (var frame = 0; frame < frames; frame++) {
    for (var bin = 0; bin < 128; bin++) {
      out[(bin * frames) + frame] = mel.data[(frame * 128) + bin];
    }
  }
  return out;
}

RuntimeTensor _channelFirstMelTensor(
  Qwen3AsrCpuMelTensor mel, {
  int? targetFrames,
}) {
  final frames = mel.frameCount;
  final outputFrames = targetFrames ?? frames;
  if (outputFrames < 0) {
    throw ArgumentError.value(targetFrames, 'targetFrames');
  }
  final out = Float32List(128 * outputFrames);
  final copiedFrames = math.min(frames, outputFrames);
  for (var frame = 0; frame < copiedFrames; frame++) {
    for (var bin = 0; bin < 128; bin++) {
      out[(bin * outputFrames) + frame] = mel.data[(frame * 128) + bin];
    }
  }
  return RuntimeTensor.float32([1, 128, outputFrames], out);
}

int _sequenceLength(RuntimeTensor tensor) {
  if (tensor.shape.length >= 3) return tensor.shape[1];
  if (tensor.shape.length == 2) return tensor.shape[0];
  return 0;
}

Float32List _audioFeatureRow(RuntimeTensor tensor, int row) {
  final values = _tensorAsFloat32(tensor);
  final hidden = tensor.shape.isNotEmpty ? tensor.shape.last : values.length;
  final sequence = _sequenceLength(tensor);
  if (row < 0 || row >= sequence) {
    throw RangeError.range(row, 0, math.max(0, sequence - 1), 'audioRow');
  }
  final offset = row * hidden;
  return Float32List.fromList(values.sublist(offset, offset + hidden));
}

Float32List _attentionMask(int position, int maxSequenceLength) {
  final out = Float32List(maxSequenceLength);
  for (var i = position + 1; i < out.length; i++) {
    out[i] = -10000.0;
  }
  return out;
}

double _float16ToFloat32(int value) {
  final sign = (value & 0x8000) == 0 ? 1.0 : -1.0;
  final exponent = (value >> 10) & 0x1f;
  final fraction = value & 0x03ff;
  if (exponent == 0) {
    if (fraction == 0) return sign == 1.0 ? 0.0 : -0.0;
    return sign * math.pow(2.0, -14) * (fraction / 1024.0);
  }
  if (exponent == 0x1f) {
    return fraction == 0 ? sign * double.infinity : double.nan;
  }
  return sign * math.pow(2.0, exponent - 15) * (1.0 + fraction / 1024.0);
}
