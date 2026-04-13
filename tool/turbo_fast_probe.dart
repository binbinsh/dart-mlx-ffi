import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';
import 'package:dart_mlx_ffi/src/models/paddle_ocr_vl/paddle_ocr_vl.dart';

String _snapshotPath() {
  final home = Platform.environment['HOME']!;
  return '$home/.cache/huggingface/hub/models--mlx-community--PaddleOCR-VL-1.5-8bit/snapshots/37d4c85284434b6e6fd4c03f8b719b1aefaa013c';
}

List<int> _loadPromptIds() {
  final idsArr = mx.io.load('/tmp/paddle_v15_ref/input_ids.npy');
  try {
    MlxRuntime.evalAll([idsArr]);
    return idsArr
        .toList()
        .cast<num>()
        .map((n) => n.toInt())
        .toList(growable: false);
  } finally {
    idsArr.close();
  }
}

String _hiddenPath() =>
    Platform.environment['TQ_HIDDEN_PATH'] ?? '/tmp/tq_cur_visual_cast.npy';

double _diff(MlxArray a, MlxArray b) {
  final da = a.astype(MlxDType.MLX_FLOAT32);
  final db = b.astype(MlxDType.MLX_FLOAT32);
  final d = mx.sum(mx.abs(da - db));
  MlxRuntime.evalAll([d]);
  final value = (d.toList().first as num).toDouble();
  da.close();
  db.close();
  d.close();
  return value;
}

typedef _OverridesSnapshot =
    ({
      String? kvQuantScheme,
      double? turboBits,
      int? turboStart,
      bool? turboDisableFusedKv,
      bool? turboDisableSingleQuant,
      bool? turboDisableFastScore,
      bool? turboDisableFastValue,
      bool? turboDisableFusedDecode,
    });

_OverridesSnapshot _captureOverrides() => (
  kvQuantScheme: PaddleOcrVlDebugOverrides.kvQuantScheme,
  turboBits: PaddleOcrVlDebugOverrides.turboBits,
  turboStart: PaddleOcrVlDebugOverrides.turboStart,
  turboDisableFusedKv: PaddleOcrVlDebugOverrides.turboDisableFusedKv,
  turboDisableSingleQuant: PaddleOcrVlDebugOverrides.turboDisableSingleQuant,
  turboDisableFastScore: PaddleOcrVlDebugOverrides.turboDisableFastScore,
  turboDisableFastValue: PaddleOcrVlDebugOverrides.turboDisableFastValue,
  turboDisableFusedDecode: PaddleOcrVlDebugOverrides.turboDisableFusedDecode,
);

void _restoreOverrides(_OverridesSnapshot s) {
  PaddleOcrVlDebugOverrides.kvQuantScheme = s.kvQuantScheme;
  PaddleOcrVlDebugOverrides.turboBits = s.turboBits;
  PaddleOcrVlDebugOverrides.turboStart = s.turboStart;
  PaddleOcrVlDebugOverrides.turboDisableFusedKv = s.turboDisableFusedKv;
  PaddleOcrVlDebugOverrides.turboDisableSingleQuant = s.turboDisableSingleQuant;
  PaddleOcrVlDebugOverrides.turboDisableFastScore = s.turboDisableFastScore;
  PaddleOcrVlDebugOverrides.turboDisableFastValue = s.turboDisableFastValue;
  PaddleOcrVlDebugOverrides.turboDisableFusedDecode = s.turboDisableFusedDecode;
}

void _setTurboBase() {
  PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
  PaddleOcrVlDebugOverrides.turboBits = 3.5;
  PaddleOcrVlDebugOverrides.turboStart = 0;
  PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
  PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
  PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
}

void main() {
  final layerIndex = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta = jsonDecode(
        File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync(),
      )
      as Map<String, Object?>;
  final runner = PaddleOcrVlRunner.load(_snapshotPath());
  final promptIds = _loadPromptIds();
  final hidden = mx.io.load(_hiddenPath());
  final snapshot = _captureOverrides();
  try {
    final qk = runner.debugSecondDecodeRopedQkFromVisionFeatures(
      promptIds,
      hidden,
      gridHeight: (meta['grid_h'] as num).toInt(),
      gridWidth: (meta['grid_w'] as num).toInt(),
      layerIndex: layerIndex,
    );
    final proj = runner.debugSecondDecodeProjectedQkvFromVisionFeatures(
      promptIds,
      hidden,
      gridHeight: (meta['grid_h'] as num).toInt(),
      gridWidth: (meta['grid_w'] as num).toInt(),
      layerIndex: layerIndex,
    );
    try {
      final numKvHeads = runner.config.numKeyValueHeads;
      final headDim = runner.config.headDim;
      final v = proj.v
          .reshape([1, 1, numKvHeads, headDim])
          .transposeAxes([0, 2, 1, 3]);
      try {
        _setTurboBase();

        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        final keyFast = runner.debugTurboQuantMseQuantize(qk.k, bits: 3, seed: 0);
        final valFast = runner.debugTurboQuantMseQuantize(v, bits: 4, seed: 1);
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
        final keySlow = runner.debugTurboQuantMseQuantize(qk.k, bits: 3, seed: 0);
        final valSlow = runner.debugTurboQuantMseQuantize(v, bits: 4, seed: 1);
        try {
          print('layer=$layerIndex firstToken=${qk.firstToken}');
          print('singleKeyNormDiff=${_diff(keyFast.norms, keySlow.norms)}');
          print('singleKeyIdxDiff=${_diff(keyFast.indices, keySlow.indices)}');
          print('singleValNormDiff=${_diff(valFast.norms, valSlow.norms)}');
          print('singleValIdxDiff=${_diff(valFast.indices, valSlow.indices)}');
        } finally {
          keyFast.norms.close();
          keyFast.indices.close();
          valFast.norms.close();
          valFast.indices.close();
          keySlow.norms.close();
          keySlow.indices.close();
          valSlow.norms.close();
          valSlow.indices.close();
        }

        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        final fused = runner.debugTurboQuantFusedKv(qk.k, v);
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
        final keyRef = runner.debugTurboQuantMseQuantize(qk.k, bits: 3, seed: 0);
        final valRef = runner.debugTurboQuantMseQuantize(v, bits: 4, seed: 1);
        try {
          print('fusedKeyNormDiff=${_diff(fused.keyNorms, keyRef.norms)}');
          print('fusedKeyIdxDiff=${_diff(fused.keyIndices, keyRef.indices)}');
          print('fusedValNormDiff=${_diff(fused.valueNorms, valRef.norms)}');
          print('fusedValIdxDiff=${_diff(fused.valueIndices, valRef.indices)}');
        } finally {
          fused.keyNorms.close();
          fused.keyIndices.close();
          fused.valueNorms.close();
          fused.valueIndices.close();
          keyRef.norms.close();
          keyRef.indices.close();
          valRef.norms.close();
          valRef.indices.close();
        }
      } finally {
        v.close();
      }
    } finally {
      qk.q.close();
      qk.k.close();
      proj.q.close();
      proj.k.close();
      proj.v.close();
    }
  } finally {
    _restoreOverrides(snapshot);
    hidden.close();
    runner.close();
  }
}
