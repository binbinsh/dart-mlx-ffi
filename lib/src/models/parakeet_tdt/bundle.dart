import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:path/path.dart' as p;

const String parakeetTdtManifestName = 'cmdspace_mlx_parakeet_tdt.json';
const String _parakeetTdtFormat = 'cmdspace-mlx-parakeet-tdt/v2';

class ParakeetTdtManifest {
  const ParakeetTdtManifest({
    required this.rootPath,
    required this.modelId,
    required this.configPath,
    required this.weightsPath,
    required this.sampleRate,
    required this.melFeatures,
    required this.nFft,
    required this.windowSize,
    required this.windowStride,
    required this.normalize,
    required this.preemph,
    required this.padTo,
    required this.padValue,
    required this.hopLength,
    required this.subsamplingFactor,
    required this.frameStepSeconds,
    required this.encoderHidden,
    required this.encoderLayers,
    required this.predictHidden,
    required this.predictLayers,
    required this.predictStateSize,
    required this.blankTokenId,
    required this.maxSymbols,
    required this.vocabulary,
    required this.durations,
    this.nHeads = 8,
    this.convKernelSize = 9,
    this.xScaling = false,
    this.selfAttentionModel = 'rel_pos',
    this.ffExpansionFactor = 4,
  });

  final String rootPath;
  final String modelId;
  final String configPath;
  final String weightsPath;
  final int sampleRate;
  final int melFeatures;
  final int nFft;
  final double windowSize;
  final double windowStride;
  final String normalize;
  final double preemph;
  final int padTo;
  final double padValue;
  final int hopLength;
  final int subsamplingFactor;
  final double frameStepSeconds;
  final int encoderHidden;
  final int encoderLayers;
  final int predictHidden;
  final int predictLayers;
  final int predictStateSize;
  final int blankTokenId;
  final int maxSymbols;
  final List<String> vocabulary;
  final List<int> durations;
  final int nHeads;
  final int convKernelSize;
  final bool xScaling;
  final String selfAttentionModel;
  final int ffExpansionFactor;

  int get winLength => (windowSize * sampleRate).round();
}

class ParakeetTdtBundle {
  ParakeetTdtBundle({
    required this.path,
    required this.manifest,
    required this.tensors,
    required this.metadata,
  });

  final String path;
  final ParakeetTdtManifest manifest;
  final Map<String, MlxArray> tensors;
  final Map<String, String> metadata;

  void close() {
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }
}

Future<ParakeetTdtManifest> readParakeetTdtManifest(String rawPath) async {
  final normalized = p.normalize(rawPath.trim());
  if (normalized.isEmpty) {
    throw StateError('Select a local Parakeet TDT bundle first.');
  }
  final type = FileSystemEntity.typeSync(normalized);
  if (type == FileSystemEntityType.notFound) {
    throw StateError('Parakeet TDT bundle not found: $normalized');
  }

  final rootPath = type == FileSystemEntityType.directory
      ? normalized
      : p.dirname(normalized);
  final manifestFile = File(
    p.basename(normalized) == parakeetTdtManifestName
        ? normalized
        : p.join(rootPath, parakeetTdtManifestName),
  );
  if (!await manifestFile.exists()) {
    throw StateError('Parakeet TDT manifest not found: ${manifestFile.path}');
  }

  final decoded = jsonDecode(await manifestFile.readAsString());
  if (decoded is! Map<String, dynamic>) {
    throw StateError('Invalid Parakeet TDT manifest: ${manifestFile.path}');
  }
  final format = decoded['format']?.toString().trim();
  if (format != _parakeetTdtFormat) {
    throw StateError('Unsupported Parakeet TDT bundle format: $format');
  }

  int requireInt(String key) {
    final value = decoded[key];
    if (value is int) {
      return value;
    }
    if (value is num) {
      return value.toInt();
    }
    final parsed = int.tryParse(value?.toString() ?? '');
    if (parsed == null) {
      throw StateError('Parakeet TDT manifest has invalid "$key".');
    }
    return parsed;
  }

  double requireDouble(String key) {
    final value = decoded[key];
    if (value is double) {
      return value;
    }
    if (value is num) {
      return value.toDouble();
    }
    final parsed = double.tryParse(value?.toString() ?? '');
    if (parsed == null) {
      throw StateError('Parakeet TDT manifest has invalid "$key".');
    }
    return parsed;
  }

  String requirePath(String key) {
    final raw = decoded[key]?.toString().trim() ?? '';
    if (raw.isEmpty) {
      throw StateError('Parakeet TDT manifest is missing "$key".');
    }
    return p.normalize(p.join(rootPath, raw));
  }

  final vocabulary = (decoded['vocabulary'] as List<Object?>?)
      ?.map((value) => value?.toString() ?? '')
      .toList(growable: false);
  final durations = (decoded['durations'] as List<Object?>?)
      ?.map((value) => (value as num).toInt())
      .toList(growable: false);
  if (vocabulary == null || vocabulary.isEmpty) {
    throw StateError('Parakeet TDT manifest is missing vocabulary.');
  }
  if (durations == null || durations.isEmpty) {
    throw StateError('Parakeet TDT manifest is missing durations.');
  }

  return ParakeetTdtManifest(
    rootPath: rootPath,
    modelId: decoded['model_id']?.toString().trim() ?? '',
    configPath: requirePath('config'),
    weightsPath: requirePath('weights'),
    sampleRate: requireInt('sample_rate'),
    melFeatures: requireInt('mel_features'),
    nFft: requireInt('n_fft'),
    windowSize: requireDouble('window_size'),
    windowStride: requireDouble('window_stride'),
    normalize: decoded['normalize']?.toString().trim().isNotEmpty == true
        ? decoded['normalize'].toString().trim()
        : 'per_feature',
    preemph: decoded['preemph'] is num
        ? (decoded['preemph'] as num).toDouble()
        : 0.97,
    padTo: requireInt('pad_to'),
    padValue: requireDouble('pad_value'),
    hopLength: requireInt('hop_length'),
    subsamplingFactor: requireInt('subsampling_factor'),
    frameStepSeconds: requireDouble('frame_step_seconds'),
    encoderHidden: requireInt('encoder_hidden'),
    encoderLayers: requireInt('encoder_layers'),
    predictHidden: requireInt('predict_hidden'),
    predictLayers: requireInt('predict_layers'),
    predictStateSize: requireInt('predict_state_size'),
    blankTokenId: requireInt('blank_token_id'),
    maxSymbols: requireInt('max_symbols'),
    vocabulary: vocabulary,
    durations: durations,
    nHeads: decoded['n_heads'] is int ? decoded['n_heads'] as int : 8,
    convKernelSize: decoded['conv_kernel_size'] is int
        ? decoded['conv_kernel_size'] as int
        : 9,
    xScaling: decoded['x_scaling'] == true,
    selfAttentionModel:
        decoded['self_attention_model']?.toString().trim().isNotEmpty == true
        ? decoded['self_attention_model'].toString().trim()
        : 'rel_pos',
    ffExpansionFactor: decoded['ff_expansion_factor'] is int
        ? decoded['ff_expansion_factor'] as int
        : 4,
  );
}

Future<ParakeetTdtBundle> loadParakeetTdtBundle(String bundlePath) async {
  final manifest = await readParakeetTdtManifest(bundlePath);
  final data = mx.io.loadSafetensors(manifest.weightsPath);
  return ParakeetTdtBundle(
    path: manifest.rootPath,
    manifest: manifest,
    tensors: data.tensors,
    metadata: data.metadata,
  );
}
