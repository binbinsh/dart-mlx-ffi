/// Pyannote Segmentation 3.0 manifest + bundle loader.
///
/// Mirrors the layout produced by `tool/pyannote_seg_export.py`:
///
/// * `weights.safetensors` holds raw PyTorch tensors for the PyanNet model,
///   including the learnable SincNet filterbank parameters
///   (`sincnet.conv1d.0.filterbank.low_hz_`, `band_hz_`, `n_`, `window_`).
/// * The MLX runtime consumes these tensors directly; SincNet filters are
///   recomputed from the 4 stored params on each load (40 filter pairs × 251
///   taps, cheap).
///
/// All tensors are owned by the returned [PyannoteSegBundle]; call
/// [PyannoteSegBundle.close] to release GPU memory.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:path/path.dart' as p;

const String pyannoteSegManifestName = 'cmdspace_mlx_pyannote_seg.json';
const String _pyannoteSegFormat = 'cmdspace-mlx-pyannote-seg/v1';

/// Hyperparameters for SincNet block (first conv1d pair in the stack).
class SincNetParams {
  const SincNetParams({
    required this.sampleRate,
    required this.stride,
    required this.kernelSize,
    required this.nFilters,
  });

  final int sampleRate;
  final int stride;
  final int kernelSize;

  /// `[80, 60, 60]` — output channels of conv1d.0 / conv1d.1 / conv1d.2.
  final List<int> nFilters;
}

/// LSTM stack parameters (4 layers × bidirectional × 128 hidden).
class PyannoteLstmParams {
  const PyannoteLstmParams({
    required this.inputSize,
    required this.hiddenSize,
    required this.numLayers,
    required this.bidirectional,
    required this.dropout,
  });

  final int inputSize;
  final int hiddenSize;
  final int numLayers;
  final bool bidirectional;
  final double dropout;
}

/// Full manifest describing one pyannote segmentation bundle on disk.
class PyannoteSegManifest {
  const PyannoteSegManifest({
    required this.rootPath,
    required this.modelId,
    required this.weightsPath,
    required this.sampleRate,
    required this.windowDurationSeconds,
    required this.windowSamples,
    required this.sincnet,
    required this.lstm,
    required this.linearHiddenSizes,
    required this.numClasses,
    required this.powersetMaxClasses,
    required this.numSpeakers,
    required this.classes,
    required this.powersetIndexLayout,
  });

  final String rootPath;
  final String modelId;
  final String weightsPath;
  final int sampleRate;
  final double windowDurationSeconds;
  final int windowSamples;
  final SincNetParams sincnet;
  final PyannoteLstmParams lstm;

  /// Two post-LSTM Linear+leaky_relu layers: `[256 → 128, 128 → 128]`.
  final List<int> linearHiddenSizes;

  /// Total class count including silence + single + overlap (`7` for 3 spk).
  final int numClasses;
  final int powersetMaxClasses;
  final int numSpeakers;
  final List<String> classes;

  /// Powerset index → list of active speaker indices.  Silence = empty list.
  final List<List<int>> powersetIndexLayout;
}

/// A loaded pyannote segmentation bundle.
class PyannoteSegBundle {
  PyannoteSegBundle({
    required this.path,
    required this.manifest,
    required this.tensors,
  });

  final String path;
  final PyannoteSegManifest manifest;
  final Map<String, MlxArray> tensors;

  MlxArray require(String key) {
    final v = tensors[key];
    if (v == null) {
      throw StateError('Missing pyannote-seg tensor: $key');
    }
    return v;
  }

  void close() {
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }
}

/// Load and validate the manifest JSON for a pyannote-seg bundle.
Future<PyannoteSegManifest> readPyannoteSegManifest(String rawPath) async {
  final normalized = p.normalize(rawPath.trim());
  if (normalized.isEmpty) {
    throw StateError('Select a local pyannote-seg bundle first.');
  }
  final type = FileSystemEntity.typeSync(normalized);
  if (type == FileSystemEntityType.notFound) {
    throw StateError('pyannote-seg bundle not found: $normalized');
  }

  final rootPath = type == FileSystemEntityType.directory
      ? normalized
      : p.dirname(normalized);
  final manifestFile = File(
    p.basename(normalized) == pyannoteSegManifestName
        ? normalized
        : p.join(rootPath, pyannoteSegManifestName),
  );
  if (!await manifestFile.exists()) {
    throw StateError(
      'pyannote-seg manifest not found: ${manifestFile.path}',
    );
  }

  final decoded = jsonDecode(await manifestFile.readAsString());
  if (decoded is! Map<String, dynamic>) {
    throw StateError('Invalid pyannote-seg manifest: ${manifestFile.path}');
  }
  final format = decoded['format']?.toString().trim();
  if (format != _pyannoteSegFormat) {
    throw StateError(
      'Unsupported pyannote-seg bundle format: $format '
      '(want $_pyannoteSegFormat)',
    );
  }

  int requireInt(String key) {
    final value = decoded[key];
    if (value is int) return value;
    if (value is num) return value.toInt();
    throw StateError('pyannote-seg manifest has invalid "$key".');
  }

  double requireDouble(String key) {
    final value = decoded[key];
    if (value is num) return value.toDouble();
    throw StateError('pyannote-seg manifest has invalid "$key".');
  }

  List<int> requireIntList(dynamic value, String key) {
    if (value is! List) {
      throw StateError('pyannote-seg manifest missing list "$key".');
    }
    return value.map((e) => (e as num).toInt()).toList(growable: false);
  }

  List<String> requireStringList(String key) {
    final value = decoded[key];
    if (value is! List) {
      throw StateError('pyannote-seg manifest missing list "$key".');
    }
    return value.map((e) => e.toString()).toList(growable: false);
  }

  final sincRaw = decoded['sincnet'];
  if (sincRaw is! Map<String, dynamic>) {
    throw StateError('pyannote-seg manifest missing "sincnet" block.');
  }
  final sincnet = SincNetParams(
    sampleRate: (sincRaw['sample_rate'] as num).toInt(),
    stride: (sincRaw['stride'] as num).toInt(),
    kernelSize: (sincRaw['kernel_size'] as num).toInt(),
    nFilters: requireIntList(sincRaw['n_filters'], 'sincnet.n_filters'),
  );

  final lstmRaw = decoded['lstm'];
  if (lstmRaw is! Map<String, dynamic>) {
    throw StateError('pyannote-seg manifest missing "lstm" block.');
  }
  final lstm = PyannoteLstmParams(
    inputSize: (lstmRaw['input_size'] as num).toInt(),
    hiddenSize: (lstmRaw['hidden_size'] as num).toInt(),
    numLayers: (lstmRaw['num_layers'] as num).toInt(),
    bidirectional: lstmRaw['bidirectional'] == true,
    dropout: (lstmRaw['dropout'] as num).toDouble(),
  );

  final layout = decoded['powerset_index_layout'];
  if (layout is! List) {
    throw StateError(
      'pyannote-seg manifest missing "powerset_index_layout".',
    );
  }
  final powersetIndexLayout = layout
      .map(
        (row) => (row as List)
            .map((e) => (e as num).toInt())
            .toList(growable: false),
      )
      .toList(growable: false);

  final weightsRaw = decoded['weights']?.toString().trim() ?? '';
  if (weightsRaw.isEmpty) {
    throw StateError('pyannote-seg manifest is missing "weights".');
  }
  final weightsPath = p.normalize(p.join(rootPath, weightsRaw));

  return PyannoteSegManifest(
    rootPath: rootPath,
    modelId: decoded['model_id']?.toString().trim() ?? '',
    weightsPath: weightsPath,
    sampleRate: requireInt('sample_rate'),
    windowDurationSeconds: requireDouble('window_duration_seconds'),
    windowSamples: requireInt('window_samples'),
    sincnet: sincnet,
    lstm: lstm,
    linearHiddenSizes: requireIntList(
      decoded['linear_hidden_sizes'],
      'linear_hidden_sizes',
    ),
    numClasses: requireInt('num_classes'),
    powersetMaxClasses: requireInt('powerset_max_classes'),
    numSpeakers: requireInt('num_speakers'),
    classes: requireStringList('classes'),
    powersetIndexLayout: powersetIndexLayout,
  );
}

/// Load a pyannote-seg bundle (manifest + safetensors).
Future<PyannoteSegBundle> loadPyannoteSegBundle(String bundlePath) async {
  final manifest = await readPyannoteSegManifest(bundlePath);
  final data = mx.io.loadSafetensors(manifest.weightsPath);
  return PyannoteSegBundle(
    path: manifest.rootPath,
    manifest: manifest,
    tensors: data.tensors,
  );
}
