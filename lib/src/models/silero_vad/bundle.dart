import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:path/path.dart' as p;

const String sileroVadManifestName = 'cmdspace_mlx_silero_vad.json';
const String _sileroVadFormat = 'cmdspace-mlx-silero-vad/v1';

class SileroVadManifest {
  const SileroVadManifest({
    required this.rootPath,
    required this.modelId,
    required this.weightsPath,
    required this.sampleRate,
    required this.windowSamples,
    required this.contextSamples,
    required this.hiddenSize,
    required this.nFft,
    required this.hopLength,
  });

  final String rootPath;
  final String modelId;
  final String weightsPath;
  final int sampleRate;
  final int windowSamples;
  final int contextSamples;
  final int hiddenSize;
  final int nFft;
  final int hopLength;
}

class SileroVadBundle {
  SileroVadBundle({
    required this.path,
    required this.manifest,
    required this.tensors,
  });

  final String path;
  final SileroVadManifest manifest;
  final Map<String, MlxArray> tensors;

  void close() {
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }
}

Future<SileroVadManifest> readSileroVadManifest(String rawPath) async {
  final normalized = p.normalize(rawPath.trim());
  if (normalized.isEmpty) {
    throw StateError('Select a local Silero VAD bundle first.');
  }
  final type = FileSystemEntity.typeSync(normalized);
  if (type == FileSystemEntityType.notFound) {
    throw StateError('Silero VAD bundle not found: $normalized');
  }

  final rootPath = type == FileSystemEntityType.directory
      ? normalized
      : p.dirname(normalized);
  final manifestFile = File(
    p.basename(normalized) == sileroVadManifestName
        ? normalized
        : p.join(rootPath, sileroVadManifestName),
  );
  if (!await manifestFile.exists()) {
    throw StateError('Silero VAD manifest not found: ${manifestFile.path}');
  }

  final decoded = jsonDecode(await manifestFile.readAsString());
  if (decoded is! Map<String, dynamic>) {
    throw StateError('Invalid Silero VAD manifest: ${manifestFile.path}');
  }
  final format = decoded['format']?.toString().trim();
  if (format != _sileroVadFormat) {
    throw StateError('Unsupported Silero VAD bundle format: $format');
  }

  int requireInt(String key) {
    final value = decoded[key];
    if (value is int) return value;
    if (value is num) return value.toInt();
    final parsed = int.tryParse(value?.toString() ?? '');
    if (parsed == null) {
      throw StateError('Silero VAD manifest has invalid "$key".');
    }
    return parsed;
  }

  String requirePath(String key) {
    final raw = decoded[key]?.toString().trim() ?? '';
    if (raw.isEmpty) {
      throw StateError('Silero VAD manifest is missing "$key".');
    }
    return p.normalize(p.join(rootPath, raw));
  }

  return SileroVadManifest(
    rootPath: rootPath,
    modelId: decoded['model_id']?.toString().trim() ?? '',
    weightsPath: requirePath('weights'),
    sampleRate: requireInt('sample_rate'),
    windowSamples: requireInt('window_samples'),
    contextSamples: requireInt('context_samples'),
    hiddenSize: requireInt('hidden_size'),
    nFft: requireInt('n_fft'),
    hopLength: requireInt('hop_length'),
  );
}

Future<SileroVadBundle> loadSileroVadBundle(String bundlePath) async {
  final manifest = await readSileroVadManifest(bundlePath);
  final data = mx.io.loadSafetensors(manifest.weightsPath);
  return SileroVadBundle(
    path: manifest.rootPath,
    manifest: manifest,
    tensors: data.tensors,
  );
}
