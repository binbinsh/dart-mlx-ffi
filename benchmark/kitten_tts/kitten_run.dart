import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main(List<String> args) {
  if (args.length != 3) {
    stderr.writeln(
      'usage: dart run benchmark/kitten_tts/kitten_run.dart <snapshot_path> <input_ids_json> <ref_s_json>',
    );
    exitCode = 64;
    return;
  }

  final snapshotPath = args[0];
  final inputIdsJson = args[1];
  final refSJson = args[2];
  final exportDir = Directory(
    '${Directory.current.path}/benchmark/out/hub_models/kitten_tts_export',
  );
  final skipExport = Platform.environment['KITTEN_SKIP_EXPORT'] == '1';
  final forceExport = Platform.environment['KITTEN_FORCE_EXPORT'] == '1';
  final fullValues = Platform.environment['KITTEN_FULL_VALUES'] == '1';
  final valuesPath = Platform.environment['KITTEN_VALUES_PATH'];
  final randomPath = Platform.environment['KITTEN_RANDOM_PATH'];
  final warmup = int.tryParse(Platform.environment['KITTEN_WARMUP'] ?? '') ?? 0;
  final iters = int.tryParse(Platform.environment['KITTEN_ITERS'] ?? '') ?? 1;
  final debug = Platform.environment['KITTEN_DEBUG'] == '1';
  final inputIds = _intMatrixToArray(inputIdsJson);
  final refS = _floatMatrixToArray(refSJson);

  final predDurPath = File('${exportDir.path}/pred_dur.mlxfn');
  final fullAlignedPath = File('${exportDir.path}/full_aligned.mlxfn');
  final metaPath = File('${exportDir.path}/meta.json');
  final shouldExport = !skipExport &&
      (forceExport ||
          !predDurPath.existsSync() ||
          !fullAlignedPath.existsSync() ||
          !metaPath.existsSync());

  if (shouldExport) {
    if (debug) {
      stderr.writeln('kitten_run: exporting MLX functions');
    }
    exportDir.createSync(recursive: true);
    final export = Process.runSync('uv', <String>[
      'run',
      'python',
      'benchmark/kitten_tts/mlx_audio_export.py',
      '--snapshot-path',
      snapshotPath,
      '--input-ids-json',
      inputIdsJson,
      '--ref-s-json',
      refSJson,
      '--output-dir',
      exportDir.path,
    ], workingDirectory: Directory.current.path);
    if (export.exitCode != 0) {
      stderr.write(export.stderr);
      stdout.write(export.stdout);
      exit(export.exitCode);
    }
  }

  if (!predDurPath.existsSync() ||
      !fullAlignedPath.existsSync() ||
      !metaPath.existsSync()) {
    stderr.writeln('Missing exported KittenTTS artifacts in ${exportDir.path}.');
    exitCode = 66;
    return;
  }

  final meta = jsonDecode(metaPath.readAsStringSync()) as Map<String, Object?>;
  final harmonicDim = (meta['harmonic_dim'] as num).toInt();
  final sampleRate = (meta['sample_rate'] as num).toInt();

  if (debug) {
    stderr.writeln('kitten_run: importing exported MLX functions');
  }
  final predDurFn = MlxExport.importFunction(predDurPath.path);
  final fullAlignedFn = MlxExport.importFunction(fullAlignedPath.path);
  try {
    for (var index = 0; index < warmup; index++) {
      final result = _runOnce(
        predDurFn,
        fullAlignedFn,
        inputIds,
        refS,
        harmonicDim: harmonicDim,
        upsampleScale: (meta['upsample_scale'] as num).toInt(),
        randomPath: randomPath,
      );
      result.audio.close();
      result.predDur.close();
    }

    final stopwatch = Stopwatch()..start();
    _RunResult? last;
    for (var index = 0; index < iters; index++) {
      last?.audio.close();
      last?.predDur.close();
      last = _runOnce(
        predDurFn,
        fullAlignedFn,
        inputIds,
        refS,
        harmonicDim: harmonicDim,
        upsampleScale: (meta['upsample_scale'] as num).toInt(),
        randomPath: randomPath,
      );
    }
    stopwatch.stop();
    if (last == null) {
      throw StateError('No KittenTTS inference iterations were executed.');
    }

    final audioFlat = last.audio.reshape([last.audio.size]);
    final preview = audioFlat
        .slice(start: [0], stop: [audioFlat.shape[0] < 16 ? audioFlat.shape[0] : 16])
        .astype(MlxDType.MLX_FLOAT32);
    final payload = <String, Object?>{
      'shape': last.audio.shape,
      'pred_dur_shape': last.predDur.shape,
      'pred_dur': List<int>.from(last.predDur.toList().cast<int>()),
      'sample_rate': sampleRate,
      'per_iter_ms': stopwatch.elapsedMicroseconds / 1000.0 / iters,
      'output_preview': List<double>.from(preview.toList().cast<double>()),
    };
    if (valuesPath != null && valuesPath.isNotEmpty) {
      mx.io.saveSafetensors(valuesPath, <String, MlxArray>{
        'audio': last.audio,
        'pred_dur': last.predDur,
      });
      payload['values_path'] = valuesPath;
    }
    if (fullValues) {
      final dense = audioFlat.astype(MlxDType.MLX_FLOAT32);
      try {
        payload['values'] = List<double>.from(dense.toList().cast<double>());
      } finally {
        dense.close();
      }
    }
    preview.close();
    audioFlat.close();
    stdout.writeln(jsonEncode(payload));
    exit(0);
  } catch (error, stackTrace) {
    stderr.writeln(error);
    stderr.writeln(stackTrace);
    exit(1);
  } finally {
    predDurFn.close();
    fullAlignedFn.close();
    inputIds.close();
    refS.close();
  }
}

final class _RunResult {
  const _RunResult(this.audio, this.predDur);

  final MlxArray audio;
  final MlxArray predDur;
}

_RunResult _runOnce(
  MlxImportedFunction predDurFn,
  MlxImportedFunction fullAlignedFn,
  MlxArray inputIds,
  MlxArray refS, {
  required int harmonicDim,
  required int upsampleScale,
  required String? randomPath,
}) {
  final debug = Platform.environment['KITTEN_DEBUG'] == '1';
  if (debug) {
    stderr.writeln('kitten_run: pred_dur start');
  }
  final predOutputs = predDurFn.call([inputIds, refS]);
  if (predOutputs.length != 1) {
    throw StateError('Expected 1 output from pred_dur, got ${predOutputs.length}.');
  }
  final predDur = predOutputs[0];
  final predDurList = predDur.toList().cast<int>();
  final alignment = _buildAlignment(predDurList, inputIds.shape[1]);
  final noiseLength =
      predDurList.fold<int>(0, (sum, value) => sum + value) * 2 * upsampleScale;
  if (debug) {
    stderr.writeln('kitten_run: pred_dur done alignment=${alignment.shape}');
  }
  late final MlxArray randIni;
  late final MlxArray noise;
  if (randomPath != null && randomPath.isNotEmpty) {
    final randoms = mx.io.loadSafetensors(randomPath).tensors;
    randIni = randoms['rand_ini']!;
    noise = randoms['noise']!;
  } else {
    MlxRuntime.seed(0);
    randIni = mx.random.normal([inputIds.shape[0], harmonicDim]);
    noise = mx.random.normal([inputIds.shape[0], noiseLength, harmonicDim]);
  }
  try {
    if (debug) {
      stderr.writeln('kitten_run: full_aligned start');
    }
    final outputs = fullAlignedFn.call([
      inputIds,
      refS,
      alignment,
      randIni,
      noise,
    ]);
    if (outputs.length != 1) {
      throw StateError('Expected 1 output from full_aligned, got ${outputs.length}.');
    }
    final audio = outputs[0];
    if (debug) {
      stderr.writeln('kitten_run: full_aligned done shape=${audio.shape}');
    }
    MlxRuntime.evalAll([audio, predDur]);
    final stream = MlxStream.defaultGpu();
    try {
      stream.synchronize();
    } finally {
      stream.close();
    }
    return _RunResult(audio, predDur);
  } finally {
    alignment.close();
    randIni.close();
    noise.close();
  }
}

MlxArray _buildAlignment(List<int> counts, int tokenCount) {
  final frames = counts.fold<int>(0, (sum, value) => sum + value);
  final values = List<double>.filled(tokenCount * frames, 0.0);
  var frame = 0;
  for (var token = 0; token < counts.length; token++) {
    final count = counts[token];
    for (var step = 0; step < count; step++) {
      values[(token * frames) + frame] = 1.0;
      frame++;
    }
  }
  return MlxArray.fromFloat32List(values, shape: [1, tokenCount, frames]);
}

MlxArray _intMatrixToArray(String jsonText) {
  final matrix = (jsonDecode(jsonText) as List<Object?>)
      .map(
        (row) => (row as List<Object?>).cast<num>().map((v) => v.toInt()).toList(),
      )
      .toList(growable: false);
  final rows = matrix.length;
  final cols = matrix.first.length;
  final flat = <int>[
    for (final row in matrix) ...row,
  ];
  return MlxArray.fromInt32List(flat, shape: [rows, cols]);
}

MlxArray _floatMatrixToArray(String jsonText) {
  final matrix = (jsonDecode(jsonText) as List<Object?>)
      .map(
        (row) => (row as List<Object?>)
            .cast<num>()
            .map((v) => v.toDouble())
            .toList(),
      )
      .toList(growable: false);
  final rows = matrix.length;
  final cols = matrix.first.length;
  final flat = <double>[
    for (final row in matrix) ...row,
  ];
  return MlxArray.fromFloat32List(flat, shape: [rows, cols]);
}
