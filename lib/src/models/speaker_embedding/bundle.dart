/// ECAPA-TDNN speaker embedding module manifest + bundle loader.
///
/// Mirrors the layout produced by `tool/ecapa_export.py` v2:
///
/// * Conv1d kernels are pre-transposed to MLX layout `(C_out, kW, C_in)`.
/// * BatchNorm1d is pre-fused into `(scale, bias)` pairs.
/// * `fused_keys` in the manifest enumerates every tensor that the runtime
///   needs, in a stable deterministic order.
///
/// The bundle loader keeps every tensor owned by the [EcapaBundle]; callers
/// must call [EcapaBundle.close] to release GPU memory.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/dart_mlx_ffi.dart';
import 'package:path/path.dart' as p;

const String ecapaManifestName = 'cmdspace_mlx_ecapa_tdnn.json';
const String _ecapaFormat = 'cmdspace-mlx-ecapa-tdnn/v2';

class EcapaManifest {
  const EcapaManifest({
    required this.rootPath,
    required this.modelId,
    required this.weightsPath,
    required this.sampleRate,
    required this.nFft,
    required this.winLength,
    required this.hopLength,
    required this.window,
    required this.nMels,
    required this.fMin,
    required this.fMax,
    required this.logScale,
    required this.logFloor,
    required this.meanNorm,
    required this.stdNorm,
    required this.embeddingDim,
    required this.channels,
    required this.kernelSizes,
    required this.dilations,
    required this.attentionChannels,
    required this.res2netScale,
    required this.seChannels,
    required this.bnEps,
    required this.aspEps,
    required this.convPaddingMode,
    required this.fusedKeys,
  });

  final String rootPath;
  final String modelId;
  final String weightsPath;
  final int sampleRate;
  final int nFft;
  final int winLength;
  final int hopLength;
  final String window;
  final int nMels;
  final int fMin;
  final int fMax;
  final double logScale;
  final double logFloor;
  final String meanNorm;
  final bool stdNorm;
  final int embeddingDim;
  final List<int> channels;
  final List<int> kernelSizes;
  final List<int> dilations;
  final int attentionChannels;
  final int res2netScale;
  final int seChannels;
  final double bnEps;
  final double aspEps;
  final String convPaddingMode;
  final List<String> fusedKeys;
}

class EcapaBundle {
  EcapaBundle({
    required this.path,
    required this.manifest,
    required this.tensors,
  });

  final String path;
  final EcapaManifest manifest;
  final Map<String, MlxArray> tensors;

  MlxArray require(String key) {
    final v = tensors[key];
    if (v == null) {
      throw StateError('Missing ECAPA-TDNN tensor: $key');
    }
    return v;
  }

  void close() {
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }
}

Future<EcapaManifest> readEcapaManifest(String rawPath) async {
  final normalized = p.normalize(rawPath.trim());
  if (normalized.isEmpty) {
    throw StateError('Select a local ECAPA-TDNN bundle first.');
  }
  final type = FileSystemEntity.typeSync(normalized);
  if (type == FileSystemEntityType.notFound) {
    throw StateError('ECAPA-TDNN bundle not found: $normalized');
  }

  final rootPath = type == FileSystemEntityType.directory
      ? normalized
      : p.dirname(normalized);
  final manifestFile = File(
    p.basename(normalized) == ecapaManifestName
        ? normalized
        : p.join(rootPath, ecapaManifestName),
  );
  if (!await manifestFile.exists()) {
    throw StateError('ECAPA-TDNN manifest not found: ${manifestFile.path}');
  }

  final decoded = jsonDecode(await manifestFile.readAsString());
  if (decoded is! Map<String, dynamic>) {
    throw StateError('Invalid ECAPA-TDNN manifest: ${manifestFile.path}');
  }
  final format = decoded['format']?.toString().trim();
  if (format != _ecapaFormat) {
    throw StateError(
      'Unsupported ECAPA-TDNN bundle format: $format (want $_ecapaFormat)',
    );
  }

  int requireInt(String key) {
    final value = decoded[key];
    if (value is int) return value;
    if (value is num) return value.toInt();
    throw StateError('ECAPA-TDNN manifest has invalid "$key".');
  }

  double requireDouble(String key) {
    final value = decoded[key];
    if (value is num) return value.toDouble();
    throw StateError('ECAPA-TDNN manifest has invalid "$key".');
  }

  List<int> requireIntList(String key) {
    final value = decoded[key];
    if (value is! List) {
      throw StateError('ECAPA-TDNN manifest missing list "$key".');
    }
    return value.map((e) => (e as num).toInt()).toList(growable: false);
  }

  List<String> requireStringList(String key) {
    final value = decoded[key];
    if (value is! List) {
      throw StateError('ECAPA-TDNN manifest missing list "$key".');
    }
    return value.map((e) => e.toString()).toList(growable: false);
  }

  final weightsRaw = decoded['weights']?.toString().trim() ?? '';
  if (weightsRaw.isEmpty) {
    throw StateError('ECAPA-TDNN manifest is missing "weights".');
  }
  final weightsPath = p.normalize(p.join(rootPath, weightsRaw));

  return EcapaManifest(
    rootPath: rootPath,
    modelId: decoded['model_id']?.toString().trim() ?? '',
    weightsPath: weightsPath,
    sampleRate: requireInt('sample_rate'),
    nFft: requireInt('n_fft'),
    winLength: requireInt('win_length'),
    hopLength: requireInt('hop_length'),
    window: decoded['window']?.toString().trim() ?? 'hamming',
    nMels: requireInt('n_mels'),
    fMin: requireInt('f_min'),
    fMax: requireInt('f_max'),
    logScale: requireDouble('log_scale'),
    logFloor: requireDouble('log_floor'),
    meanNorm: decoded['mean_norm']?.toString().trim() ?? 'sentence',
    stdNorm: decoded['std_norm'] == true,
    embeddingDim: requireInt('embedding_dim'),
    channels: requireIntList('channels'),
    kernelSizes: requireIntList('kernel_sizes'),
    dilations: requireIntList('dilations'),
    attentionChannels: requireInt('attention_channels'),
    res2netScale: requireInt('res2net_scale'),
    seChannels: requireInt('se_channels'),
    bnEps: requireDouble('bn_eps'),
    aspEps: requireDouble('asp_eps'),
    convPaddingMode:
        decoded['conv_padding_mode']?.toString().trim() ?? 'reflect',
    fusedKeys: requireStringList('fused_keys'),
  );
}

Future<EcapaBundle> loadEcapaBundle(String bundlePath) async {
  final manifest = await readEcapaManifest(bundlePath);
  final data = mx.io.loadSafetensors(manifest.weightsPath);
  // Sanity check: every key referenced in the manifest should be present.
  for (final key in manifest.fusedKeys) {
    if (!data.tensors.containsKey(key)) {
      throw StateError('ECAPA-TDNN weights missing fused key: $key');
    }
  }
  return EcapaBundle(
    path: manifest.rootPath,
    manifest: manifest,
    tensors: data.tensors,
  );
}
