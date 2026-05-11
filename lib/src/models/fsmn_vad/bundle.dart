import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/dart_mlx_ffi.dart';
import 'package:path/path.dart' as p;

const String fsmnVadManifestName = 'cmdspace_mlx_fsmn_vad.json';
const String _fsmnVadFormat = 'cmdspace-mlx-fsmn-vad/v1';

class FsmnVadManifest {
  const FsmnVadManifest({
    required this.rootPath,
    required this.modelId,
    required this.weightsPath,
    required this.cmvnPath,
    required this.sampleRate,
    required this.numMels,
    required this.frameLengthMs,
    required this.frameShiftMs,
    required this.lfrM,
    required this.lfrN,
    required this.inputDim,
    required this.inputAffineDim,
    required this.fsmnLayers,
    required this.linearDim,
    required this.projDim,
    required this.lorder,
    required this.rorder,
    required this.lstride,
    required this.rstride,
    required this.outputAffineDim,
    required this.outputDim,
    required this.speechThreshold,
    required this.windowSizeMs,
    required this.silToSpeechMs,
    required this.speechToSilMs,
    required this.maxEndSilenceMs,
  });

  final String rootPath;
  final String modelId;
  final String weightsPath;
  final String cmvnPath;
  final int sampleRate;
  final int numMels;
  final int frameLengthMs;
  final int frameShiftMs;
  final int lfrM;
  final int lfrN;
  final int inputDim;
  final int inputAffineDim;
  final int fsmnLayers;
  final int linearDim;
  final int projDim;
  final int lorder;
  final int rorder;
  final int lstride;
  final int rstride;
  final int outputAffineDim;
  final int outputDim;
  final double speechThreshold;
  final int windowSizeMs;
  final int silToSpeechMs;
  final int speechToSilMs;
  final int maxEndSilenceMs;

  int get frameSampleLength => (frameLengthMs * sampleRate) ~/ 1000;
  int get frameShiftSampleLength => (frameShiftMs * sampleRate) ~/ 1000;
  int get fftSize {
    var size = 1;
    while (size < frameSampleLength) {
      size <<= 1;
    }
    return size;
  }

  int get cacheFrames => ((lorder - 1) * lstride) + (rorder * rstride);
}

class FsmnVadCmvn {
  const FsmnVadCmvn({required this.offsets, required this.scales});

  final Float32List offsets;
  final Float32List scales;
}

class FsmnVadBundle {
  FsmnVadBundle({
    required this.path,
    required this.manifest,
    required this.cmvn,
    required this.tensors,
  });

  final String path;
  final FsmnVadManifest manifest;
  final FsmnVadCmvn cmvn;
  final Map<String, MlxArray> tensors;

  void close() {
    for (final tensor in tensors.values) {
      tensor.close();
    }
  }
}

Future<FsmnVadManifest> readFsmnVadManifest(String rawPath) async {
  final normalized = p.normalize(rawPath.trim());
  if (normalized.isEmpty) {
    throw StateError('Select a local FSMN-VAD bundle first.');
  }
  final type = FileSystemEntity.typeSync(normalized);
  if (type == FileSystemEntityType.notFound) {
    throw StateError('FSMN-VAD bundle not found: $normalized');
  }

  final rootPath = type == FileSystemEntityType.directory
      ? normalized
      : p.dirname(normalized);
  final manifestFile = File(
    p.basename(normalized) == fsmnVadManifestName
        ? normalized
        : p.join(rootPath, fsmnVadManifestName),
  );
  if (!await manifestFile.exists()) {
    throw StateError('FSMN-VAD manifest not found: ${manifestFile.path}');
  }

  final decoded = jsonDecode(await manifestFile.readAsString());
  if (decoded is! Map<String, dynamic>) {
    throw StateError('Invalid FSMN-VAD manifest: ${manifestFile.path}');
  }
  final format = decoded['format']?.toString().trim();
  if (format != _fsmnVadFormat) {
    throw StateError('Unsupported FSMN-VAD bundle format: $format');
  }

  int requireInt(String key) {
    final value = decoded[key];
    if (value is int) return value;
    if (value is num) return value.toInt();
    final parsed = int.tryParse(value?.toString() ?? '');
    if (parsed == null) {
      throw StateError('FSMN-VAD manifest has invalid "$key".');
    }
    return parsed;
  }

  double requireDouble(String key) {
    final value = decoded[key];
    if (value is double) return value;
    if (value is num) return value.toDouble();
    final parsed = double.tryParse(value?.toString() ?? '');
    if (parsed == null) {
      throw StateError('FSMN-VAD manifest has invalid "$key".');
    }
    return parsed;
  }

  String requirePath(String key) {
    final raw = decoded[key]?.toString().trim() ?? '';
    if (raw.isEmpty) {
      throw StateError('FSMN-VAD manifest is missing "$key".');
    }
    return p.normalize(p.join(rootPath, raw));
  }

  return FsmnVadManifest(
    rootPath: rootPath,
    modelId: decoded['model_id']?.toString().trim() ?? '',
    weightsPath: requirePath('weights'),
    cmvnPath: requirePath('cmvn'),
    sampleRate: requireInt('sample_rate'),
    numMels: requireInt('num_mels'),
    frameLengthMs: requireInt('frame_length_ms'),
    frameShiftMs: requireInt('frame_shift_ms'),
    lfrM: requireInt('lfr_m'),
    lfrN: requireInt('lfr_n'),
    inputDim: requireInt('input_dim'),
    inputAffineDim: requireInt('input_affine_dim'),
    fsmnLayers: requireInt('fsmn_layers'),
    linearDim: requireInt('linear_dim'),
    projDim: requireInt('proj_dim'),
    lorder: requireInt('lorder'),
    rorder: requireInt('rorder'),
    lstride: requireInt('lstride'),
    rstride: requireInt('rstride'),
    outputAffineDim: requireInt('output_affine_dim'),
    outputDim: requireInt('output_dim'),
    speechThreshold: requireDouble('speech_threshold'),
    windowSizeMs: requireInt('window_size_ms'),
    silToSpeechMs: requireInt('sil_to_speech_ms'),
    speechToSilMs: requireInt('speech_to_sil_ms'),
    maxEndSilenceMs: requireInt('max_end_silence_ms'),
  );
}

Future<FsmnVadCmvn> loadFsmnVadCmvn(String rawPath) async {
  final file = File(rawPath);
  if (!await file.exists()) {
    throw StateError('FSMN-VAD CMVN file not found: $rawPath');
  }
  final lines = await file.readAsLines();
  List<String> parseSection(String header) {
    for (var index = 0; index < lines.length; index += 1) {
      final parts = lines[index].trim().split(RegExp(r'\s+'));
      if (parts.isEmpty || parts.first != header) {
        continue;
      }
      if (index + 1 >= lines.length) {
        break;
      }
      final next = lines[index + 1].trim().split(RegExp(r'\s+'));
      if (next.isEmpty || next.first != '<LearnRateCoef>') {
        break;
      }
      return next.sublist(3, next.length - 1);
    }
    throw StateError('FSMN-VAD CMVN is missing $header section.');
  }

  Float32List parseValues(List<String> raw) {
    final out = Float32List(raw.length);
    for (var i = 0; i < raw.length; i += 1) {
      out[i] = double.parse(raw[i]);
    }
    return out;
  }

  return FsmnVadCmvn(
    offsets: parseValues(parseSection('<AddShift>')),
    scales: parseValues(parseSection('<Rescale>')),
  );
}

Future<FsmnVadBundle> loadFsmnVadBundle(String bundlePath) async {
  final manifest = await readFsmnVadManifest(bundlePath);
  final cmvn = await loadFsmnVadCmvn(manifest.cmvnPath);
  final data = mx.io.loadSafetensors(manifest.weightsPath);
  return FsmnVadBundle(
    path: manifest.rootPath,
    manifest: manifest,
    cmvn: cmvn,
    tensors: data.tensors,
  );
}
