import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

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
    return idsArr.toList().cast<num>().map((n) => n.toInt()).toList(
      growable: false,
    );
  } finally {
    idsArr.close();
  }
}

String _hiddenPath() =>
    Platform.environment['TQ_HIDDEN_PATH'] ?? '/tmp/tq_cur_visual_cast.npy';

String _sample(MlxArray array, [int count = 12]) {
  final cast = array.astype(MlxDType.MLX_FLOAT32);
  MlxRuntime.evalAll([cast]);
  final values = cast.toList().take(count).map((v) {
    final d = (v as num).toDouble();
    return d.toStringAsFixed(4);
  }).join(',');
  cast.close();
  return values;
}

String _mismatchPositions(MlxArray a, MlxArray b, [int limit = 16]) {
  final aa = a.astype(MlxDType.MLX_FLOAT32);
  final bb = b.astype(MlxDType.MLX_FLOAT32);
  MlxRuntime.evalAll([aa, bb]);
  final av = aa.toList();
  final bv = bb.toList();
  final parts = <String>[];
  for (var i = 0; i < av.length && parts.length < limit; i++) {
    final da = (av[i] as num).toDouble();
    final db = (bv[i] as num).toDouble();
    if (da != db) {
      parts.add('$i:${da.toInt()}!=${db.toInt()}');
    }
  }
  aa.close();
  bb.close();
  return parts.join(',');
}

String _repeatPairDiffs(MlxArray fast, MlxArray slow) {
  if (fast.shape.length != 5 || slow.shape.length != 5) return 'n/a';
  final repeats = fast.shape[2];
  final tokens = fast.shape[4];
  final fastVals = fast.astype(MlxDType.MLX_FLOAT32);
  final slowVals = slow.astype(MlxDType.MLX_FLOAT32);
  try {
    final parts = <String>[];
    for (var r = 0; r < repeats; r++) {
      final row = <String>[];
      final fastSlice = fastVals.slice(
        start: [0, 0, r, 0, 0],
        stop: [1, 1, r + 1, 1, tokens],
      );
      try {
        for (var s = 0; s < repeats; s++) {
          final slowSlice = slowVals.slice(
            start: [0, 0, s, 0, 0],
            stop: [1, 1, s + 1, 1, tokens],
          );
          try {
            row.add(_sumAbsDiff(fastSlice, slowSlice).toStringAsFixed(1));
          } finally {
            slowSlice.close();
          }
        }
      } finally {
        fastSlice.close();
      }
      parts.add(row.join('/'));
    }
    return parts.join(' | ');
  } finally {
    fastVals.close();
    slowVals.close();
  }
}

double _sumAbsDiff(MlxArray a, MlxArray b) {
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

double _maxAbsDiff(MlxArray a, MlxArray b) {
  final da = a.astype(MlxDType.MLX_FLOAT32);
  final db = b.astype(MlxDType.MLX_FLOAT32);
  final diff = mx.abs(da - db);
  MlxRuntime.evalAll([diff]);
  var value = 0.0;
  for (final raw in diff.toList()) {
    final cur = (raw as num).toDouble().abs();
    if (cur > value) value = cur;
  }
  da.close();
  db.close();
  diff.close();
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

void _setCase(String compare) {
  PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
  PaddleOcrVlDebugOverrides.turboBits = 3.5;
  PaddleOcrVlDebugOverrides.turboStart = 0;
  switch (compare) {
    case 'all_fast':
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = false;
      return;
    case 'fused_decode_only':
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = false;
      return;
    case 'no_fused_decode':
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      return;
    case 'all_slow':
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      return;
    default:
      throw ArgumentError('Unknown TQ_COMPARE=$compare');
  }
}

void main() {
  final kind = Platform.environment['TQ_KIND'] ?? 'scores';
  final compare = Platform.environment['TQ_COMPARE'] ?? 'no_fused_decode';
  final compareB = Platform.environment['TQ_COMPARE_B'] ?? 'all_slow';
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
    _setCase(compare);
    if (kind == 'scores') {
      final scores = runner.debugSecondDecodeScoresFromVisionFeatures(
        promptIds,
        hidden,
        gridHeight: (meta['grid_h'] as num).toInt(),
        gridWidth: (meta['grid_w'] as num).toInt(),
        layerIndex: layerIndex,
      );
      try {
        print('kind=scores compare=$compare layer=$layerIndex firstToken=${scores.firstToken}');
        print('fastShape=${scores.fastScores.shape} slowShape=${scores.slowScores.shape}');
        print('sumAbsDiff=${_sumAbsDiff(scores.fastScores, scores.slowScores)}');
        print('maxAbsDiff=${_maxAbsDiff(scores.fastScores, scores.slowScores)}');
        final fastSwapped = scores.fastScores.transposeAxes([0, 1, 3, 2, 4]);
        try {
          print('swap23SumAbsDiff=${_sumAbsDiff(fastSwapped, scores.slowScores)}');
          print('swap23MaxAbsDiff=${_maxAbsDiff(fastSwapped, scores.slowScores)}');
        } finally {
          fastSwapped.close();
        }
        print('fastSample=${_sample(scores.fastScores)}');
        print('slowSample=${_sample(scores.slowScores)}');
        print('repeatPairDiffs=${_repeatPairDiffs(scores.fastScores, scores.slowScores)}');
      } finally {
        scores.fastScores.close();
        scores.slowScores.close();
      }
      return;
    }
    if (kind == 'attention') {
      final attn = runner.debugSecondDecodeAttentionOutputsFromVisionFeatures(
        promptIds,
        hidden,
        gridHeight: (meta['grid_h'] as num).toInt(),
        gridWidth: (meta['grid_w'] as num).toInt(),
        layerIndex: layerIndex,
      );
      try {
        final splitReshaped = attn.splitOutput.reshape(attn.directOutput.shape);
        print('kind=attention compare=$compare layer=$layerIndex firstToken=${attn.firstToken}');
        print('directShape=${attn.directOutput.shape} splitShape=${attn.splitOutput.shape}');
        print('sumAbsDiff=${_sumAbsDiff(attn.directOutput, splitReshaped)}');
        print('maxAbsDiff=${_maxAbsDiff(attn.directOutput, splitReshaped)}');
        print('directSample=${_sample(attn.directOutput)}');
        print('splitSample=${_sample(splitReshaped)}');
        splitReshaped.close();
      } finally {
        attn.directOutput.close();
        attn.splitOutput.close();
      }
      return;
    }
    if (kind == 'attention_vs_slow') {
      final hiddenA = mx.io.load(_hiddenPath());
      final hiddenB = mx.io.load(_hiddenPath());
      try {
        _setCase(compare);
        final attnA = runner.debugSecondDecodeAttentionOutputsFromVisionFeatures(
          promptIds,
          hiddenA,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          _setCase(compareB);
          final attnB = runner.debugSecondDecodeAttentionOutputsFromVisionFeatures(
            promptIds,
            hiddenB,
            gridHeight: (meta['grid_h'] as num).toInt(),
            gridWidth: (meta['grid_w'] as num).toInt(),
            layerIndex: layerIndex,
          );
          try {
            print('kind=attention_vs_slow compare=$compare compareB=$compareB layer=$layerIndex');
            print('aShape=${attnA.directOutput.shape} bShape=${attnB.directOutput.shape}');
            print('sumAbsDiff=${_sumAbsDiff(attnA.directOutput, attnB.directOutput)}');
            print('maxAbsDiff=${_maxAbsDiff(attnA.directOutput, attnB.directOutput)}');
            print('aSample=${_sample(attnA.directOutput)}');
            print('bSample=${_sample(attnB.directOutput)}');
          } finally {
            attnB.directOutput.close();
            attnB.splitOutput.close();
          }
        } finally {
          attnA.directOutput.close();
          attnA.splitOutput.close();
        }
      } finally {
        hiddenA.close();
        hiddenB.close();
      }
      return;
    }
    if (kind == 'attention_vs_py') {
      final pyPath =
          Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_py_turbo_attn0.npy';
      final py = mx.io.load(pyPath);
      try {
        final attn = runner.debugSecondDecodeAttentionOutputsFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=attention_vs_py compare=$compare layer=$layerIndex');
          print('directShape=${attn.directOutput.shape} pyShape=${py.shape}');
          print('sumAbsDiff=${_sumAbsDiff(attn.directOutput, py)}');
          print('maxAbsDiff=${_maxAbsDiff(attn.directOutput, py)}');
          print('directSample=${_sample(attn.directOutput)}');
          print('pySample=${_sample(py)}');
        } finally {
          attn.directOutput.close();
          attn.splitOutput.close();
        }
      } finally {
        py.close();
      }
      return;
    }
    if (kind == 'real_attention_vs_py') {
      final pyPath =
          Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_py_turbo_attn0.npy';
      final py = mx.io.load(pyPath);
      try {
        final out = runner.debugSecondDecodeAttentionOutputFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=real_attention_vs_py compare=$compare layer=$layerIndex firstToken=${out.firstToken}');
          print('outShape=${out.attentionOutput.shape} pyShape=${py.shape}');
          print('sumAbsDiff=${_sumAbsDiff(out.attentionOutput, py)}');
          print('maxAbsDiff=${_maxAbsDiff(out.attentionOutput, py)}');
          print('outSample=${_sample(out.attentionOutput)}');
          print('pySample=${_sample(py)}');
        } finally {
          out.attentionOutput.close();
        }
      } finally {
        py.close();
      }
      return;
    }
    if (kind == 'rope_vs_py') {
      final pyQ =
          mx.io.load(Platform.environment['TQ_PY_Q'] ?? '/tmp/tq_py_turbo_step_qrope.npy');
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_step_krope.npy');
      try {
        final rope = runner.debugSecondDecodeRopedQkFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=rope_vs_py compare=$compare layer=$layerIndex firstToken=${rope.firstToken}');
          print('qDiff=${_sumAbsDiff(rope.q, pyQ)}');
          print('kDiff=${_sumAbsDiff(rope.k, pyK)}');
          print('qMaxDiff=${_maxAbsDiff(rope.q, pyQ)}');
          print('kMaxDiff=${_maxAbsDiff(rope.k, pyK)}');
        } finally {
          rope.q.close();
          rope.k.close();
        }
      } finally {
        pyQ.close();
        pyK.close();
      }
      return;
    }
    if (kind == 'rope_apply_vs_py') {
      final pyQProj =
          mx.io.load(Platform.environment['TQ_PY_QPROJ'] ?? '/tmp/tq_py_turbo_step_qproj.npy');
      final pyKProj =
          mx.io.load(Platform.environment['TQ_PY_KPROJ'] ?? '/tmp/tq_py_turbo_step_kproj.npy');
      final pyQ =
          mx.io.load(Platform.environment['TQ_PY_Q'] ?? '/tmp/tq_py_turbo_step_qrope.npy');
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_step_krope.npy');
      try {
        final inputs = runner.debugInputsEmbedsAndPositionIdsFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
        );
        try {
          final q = pyQProj.reshape([1, 1, 16, 128]).transposeAxes([0, 2, 1, 3]);
          final k = pyKProj.reshape([1, 1, 2, 128]).transposeAxes([0, 2, 1, 3]);
          try {
            final rope = runner.debugApplyMropeAtTextOffset(
              q,
              k,
              inputs.nextTextPosition,
            );
            try {
              print('kind=rope_apply_vs_py offset=${inputs.nextTextPosition}');
              print('qDiff=${_sumAbsDiff(rope.q, pyQ)}');
              print('kDiff=${_sumAbsDiff(rope.k, pyK)}');
              print('qMaxDiff=${_maxAbsDiff(rope.q, pyQ)}');
              print('kMaxDiff=${_maxAbsDiff(rope.k, pyK)}');
            } finally {
              rope.q.close();
              rope.k.close();
            }
          } finally {
            q.close();
            k.close();
          }
        } finally {
          inputs.inputsEmbeds.close();
          inputs.positionIds.close();
        }
      } finally {
        pyQProj.close();
        pyKProj.close();
        pyQ.close();
        pyK.close();
      }
      return;
    }
    if (kind == 'actual_kv_vs_py') {
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_actual_step_keys.npy');
      final pyV =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_actual_step_values.npy');
      try {
        final rope = runner.debugSecondDecodeRopedQkFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          final qkv = runner.debugSecondDecodeProjectedQkvFromVisionFeatures(
            promptIds,
            hidden,
            gridHeight: (meta['grid_h'] as num).toInt(),
            gridWidth: (meta['grid_w'] as num).toInt(),
            layerIndex: layerIndex,
          );
          try {
            final localV = qkv.v.reshape([1, 1, 2, 128]).transposeAxes([0, 2, 1, 3]);
            try {
              print('kind=actual_kv_vs_py compare=$compare layer=$layerIndex');
              print('kDiff=${_sumAbsDiff(rope.k, pyK)}');
              print('vDiff=${_sumAbsDiff(localV, pyV)}');
              print('kMaxDiff=${_maxAbsDiff(rope.k, pyK)}');
              print('vMaxDiff=${_maxAbsDiff(localV, pyV)}');
            } finally {
              localV.close();
            }
          } finally {
            qkv.q.close();
            qkv.k.close();
            qkv.v.close();
          }
        } finally {
          rope.q.close();
          rope.k.close();
        }
      } finally {
        pyK.close();
        pyV.close();
      }
      return;
    }
    if (kind == 'qrot_vs_py') {
      final pyQ =
          mx.io.load(Platform.environment['TQ_PY_Q'] ?? '/tmp/tq_py_turbo_step_qrope.npy');
      final pyQRot =
          mx.io.load(Platform.environment['TQ_PY_QROT'] ?? '/tmp/tq_py_wrapper_qrot.npy');
      try {
        final inputs = runner.debugInputsEmbedsAndPositionIdsFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
        );
        try {
          final qproj = mx.io.load('/tmp/tq_py_turbo_step_qproj.npy');
          final kproj = mx.io.load('/tmp/tq_py_turbo_step_kproj.npy');
          try {
            final q = qproj.reshape([1, 1, 16, 128]).transposeAxes([0, 2, 1, 3]);
            final k = kproj.reshape([1, 1, 2, 128]).transposeAxes([0, 2, 1, 3]);
            try {
              final rope = runner.debugApplyMropeAtTextOffset(q, k, inputs.nextTextPosition);
              try {
                final grouped = rope.q.reshape([1, 2, 8, 1, 128]);
                final scaleArr = MlxArray.full([], 1.0 / math.sqrt(128.0), dtype: grouped.dtype);
                final scaled = grouped * scaleArr;
                scaleArr.close();
                final qrot = runner.debugTurboQuantPrepareQueries(
                  scaled,
                  dim: 128,
                  bits: 3,
                  seed: 0,
                );
                try {
                  print('kind=qrot_vs_py offset=${inputs.nextTextPosition}');
                  print('qropeDiff=${_sumAbsDiff(rope.q, pyQ)}');
                  print('qrotDiff=${_sumAbsDiff(qrot, pyQRot)}');
                  print('qrotMaxDiff=${_maxAbsDiff(qrot, pyQRot)}');
                } finally {
                  qrot.close();
                }
                scaled.close();
              } finally {
                rope.q.close();
                rope.k.close();
              }
            } finally {
              q.close();
              k.close();
            }
          } finally {
            qproj.close();
            kproj.close();
          }
        } finally {
          inputs.inputsEmbeds.close();
          inputs.positionIds.close();
        }
      } finally {
        pyQ.close();
        pyQRot.close();
      }
      return;
    }
    if (kind == 'fused_raw_vs_py') {
      final pyRaw =
          mx.io.load(Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_py_wrapper_fused_raw.npy');
      final pyQRot =
          mx.io.load(Platform.environment['TQ_PY_QROT'] ?? '/tmp/tq_py_wrapper_qrot.npy');
      final pyKn =
          mx.io.load(Platform.environment['TQ_PY_KN'] ?? '/tmp/tq_py_wrapper_l0_second_kn.npy');
      final pyKi =
          mx.io.load(Platform.environment['TQ_PY_KI'] ?? '/tmp/tq_py_wrapper_l0_second_ki.npy');
      final pyVn =
          mx.io.load(Platform.environment['TQ_PY_VN'] ?? '/tmp/tq_py_wrapper_l0_second_vn.npy');
      final pyVi =
          mx.io.load(Platform.environment['TQ_PY_VI'] ?? '/tmp/tq_py_wrapper_l0_second_vi.npy');
      try {
        final qRotFlat = pyQRot.reshape([16, 128]);
        try {
          final local = runner.debugTurboQuantFusedDecodeRaw(
            qRotFlat,
            pyKn,
            pyKi,
            pyVn,
            pyVi,
            repeatCount: 8,
            dim: 128,
            keyBits: 3,
            valueBits: 4,
          );
          try {
            print('kind=fused_raw_vs_py');
            print('sumAbsDiff=${_sumAbsDiff(local, pyRaw)}');
            print('maxAbsDiff=${_maxAbsDiff(local, pyRaw)}');
            print('localSample=${_sample(local, 16)}');
            print('pySample=${_sample(pyRaw, 16)}');
          } finally {
            local.close();
          }
        } finally {
          qRotFlat.close();
        }
      } finally {
        pyRaw.close();
        pyQRot.close();
        pyKn.close();
        pyKi.close();
        pyVn.close();
        pyVi.close();
      }
      return;
    }
    if (kind == 'fused_composed_vs_py') {
      final pyPre =
          mx.io.load(Platform.environment['TQ_PY_PRE'] ?? '/tmp/tq_py_wrapper_direct_attn_preproj.npy');
      final pyQRot =
          mx.io.load(Platform.environment['TQ_PY_QROT'] ?? '/tmp/tq_py_wrapper_qrot.npy');
      final pyKn =
          mx.io.load(Platform.environment['TQ_PY_KN'] ?? '/tmp/tq_py_wrapper_l0_second_kn.npy');
      final pyKi =
          mx.io.load(Platform.environment['TQ_PY_KI'] ?? '/tmp/tq_py_wrapper_l0_second_ki.npy');
      final pyVn =
          mx.io.load(Platform.environment['TQ_PY_VN'] ?? '/tmp/tq_py_wrapper_l0_second_vn.npy');
      final pyVi =
          mx.io.load(Platform.environment['TQ_PY_VI'] ?? '/tmp/tq_py_wrapper_l0_second_vi.npy');
      try {
        final qRotFlat = pyQRot.reshape([16, 128]);
        try {
          final raw = runner.debugTurboQuantFusedDecodeRaw(
            qRotFlat,
            pyKn,
            pyKi,
            pyVn,
            pyVi,
            repeatCount: 8,
            dim: 128,
            keyBits: 3,
            valueBits: 4,
          );
          try {
            final grouped = raw.reshape([1, 2, 8, 128]);
            final inv = runner.debugTurboQuantInverseRotate(
              grouped,
              dim: 128,
              bits: 4,
              seed: 1,
            );
            try {
              final reshaped = inv.reshape([1, 16, 1, 128]);
              try {
                print('kind=fused_composed_vs_py');
                print('sumAbsDiff=${_sumAbsDiff(reshaped, pyPre)}');
                print('maxAbsDiff=${_maxAbsDiff(reshaped, pyPre)}');
                print('localSample=${_sample(reshaped, 16)}');
                print('pySample=${_sample(pyPre, 16)}');
              } finally {
                reshaped.close();
              }
            } finally {
              inv.close();
            }
          } finally {
            raw.close();
          }
        } finally {
          qRotFlat.close();
        }
      } finally {
        pyPre.close();
        pyQRot.close();
        pyKn.close();
        pyKi.close();
        pyVn.close();
        pyVi.close();
      }
      return;
    }
    if (kind == 'actual_qrot_vs_py') {
      final pyQRot =
          mx.io.load(Platform.environment['TQ_PY_QROT'] ?? '/tmp/tq_py_wrapper_qrot.npy');
      try {
        final rope = runner.debugSecondDecodeRopedQkFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          final grouped = rope.q.reshape([1, 2, 8, 1, 128]);
          final scaleArr = MlxArray.full([], 1.0 / math.sqrt(128.0), dtype: grouped.dtype);
          final scaled = grouped * scaleArr;
          scaleArr.close();
          try {
            final qrot = runner.debugTurboQuantPrepareQueries(
              scaled,
              dim: 128,
              bits: 3,
              seed: 0,
            );
            try {
              print('kind=actual_qrot_vs_py');
              print('qrotDiff=${_sumAbsDiff(qrot, pyQRot)}');
              print('qrotMaxDiff=${_maxAbsDiff(qrot, pyQRot)}');
              print('localSample=${_sample(qrot, 16)}');
              print('pySample=${_sample(pyQRot, 16)}');
            } finally {
              qrot.close();
            }
          } finally {
            scaled.close();
          }
        } finally {
          rope.q.close();
          rope.k.close();
        }
      } finally {
        pyQRot.close();
      }
      return;
    }
    if (kind == 'local_fused_vs_direct') {
      final rope = runner.debugSecondDecodeRopedQkFromVisionFeatures(
        promptIds,
        hidden,
        gridHeight: (meta['grid_h'] as num).toInt(),
        gridWidth: (meta['grid_w'] as num).toInt(),
        layerIndex: layerIndex,
      );
      try {
        final cache = runner.debugSecondDecodeLayerCacheStateFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          final grouped = rope.q.reshape([1, 2, 8, 1, 128]);
          final scaleArr = MlxArray.full([], 1.0 / math.sqrt(128.0), dtype: grouped.dtype);
          final scaled = grouped * scaleArr;
          scaleArr.close();
          try {
            final qrot = runner.debugTurboQuantPrepareQueries(
              scaled,
              dim: 128,
              bits: 3,
              seed: 0,
            );
            try {
              final raw = runner.debugTurboQuantFusedDecodeRaw(
                qrot.reshape([16, 128]),
                cache.keyNorms,
                cache.keyIndices,
                cache.valueNorms,
                cache.valueIndices,
                repeatCount: 8,
                dim: 128,
                keyBits: 3,
                valueBits: 4,
              );
              try {
                final inv = runner.debugTurboQuantInverseRotate(
                  raw.reshape([1, 2, 8, 128]),
                  dim: 128,
                  bits: 4,
                  seed: 1,
                );
                try {
                  final fused = inv.reshape([1, 16, 1, 128]);
                  try {
                    final attn = runner.debugSecondDecodeAttentionOutputsFromVisionFeatures(
                      promptIds,
                      hidden,
                      gridHeight: (meta['grid_h'] as num).toInt(),
                      gridWidth: (meta['grid_w'] as num).toInt(),
                      layerIndex: layerIndex,
                    );
                    try {
                      print('kind=local_fused_vs_direct compare=$compare');
                      print('sumAbsDiff=${_sumAbsDiff(fused, attn.directOutput)}');
                      print('maxAbsDiff=${_maxAbsDiff(fused, attn.directOutput)}');
                      print('fusedSample=${_sample(fused, 16)}');
                      print('directSample=${_sample(attn.directOutput, 16)}');
                    } finally {
                      attn.directOutput.close();
                      attn.splitOutput.close();
                    }
                  } finally {
                    fused.close();
                  }
                } finally {
                  inv.close();
                }
              } finally {
                raw.close();
              }
            } finally {
              qrot.close();
            }
          } finally {
            scaled.close();
          }
        } finally {
          cache.keyNorms.close();
          cache.keyIndices.close();
          cache.valueNorms.close();
          cache.valueIndices.close();
        }
      } finally {
        rope.q.close();
        rope.k.close();
      }
      return;
    }
    if (kind == 'direct_fused_call_vs_direct') {
      final rope = runner.debugSecondDecodeRopedQkFromVisionFeatures(
        promptIds,
        hidden,
        gridHeight: (meta['grid_h'] as num).toInt(),
        gridWidth: (meta['grid_w'] as num).toInt(),
        layerIndex: layerIndex,
      );
      try {
        final cache = runner.debugSecondDecodeLayerCacheStateFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          final fused = runner.debugTurboQuantFusedDecodeDirect(
            rope.q,
            cache.keyNorms,
            cache.keyIndices,
            cache.valueNorms,
            cache.valueIndices,
            numKvHeads: 2,
            headDim: 128,
            keyBits: 3,
            valueBits: 4,
            scale: 1.0 / math.sqrt(128.0),
          );
          if (fused == null) {
            print('kind=direct_fused_call_vs_direct fused=null');
            return;
          }
          try {
            final attn = runner.debugSecondDecodeAttentionOutputsFromVisionFeatures(
              promptIds,
              hidden,
              gridHeight: (meta['grid_h'] as num).toInt(),
              gridWidth: (meta['grid_w'] as num).toInt(),
              layerIndex: layerIndex,
            );
                    try {
                      print('kind=direct_fused_call_vs_direct compare=$compare');
                      print('lastPath=${runner.debugTurboLastAttentionPath()}');
                      print('sumAbsDiff=${_sumAbsDiff(fused, attn.directOutput)}');
              print('maxAbsDiff=${_maxAbsDiff(fused, attn.directOutput)}');
              print('fusedSample=${_sample(fused, 16)}');
              print('directSample=${_sample(attn.directOutput, 16)}');
            } finally {
              attn.directOutput.close();
              attn.splitOutput.close();
            }
          } finally {
            fused.close();
          }
        } finally {
          cache.keyNorms.close();
          cache.keyIndices.close();
          cache.valueNorms.close();
          cache.valueIndices.close();
        }
      } finally {
        rope.q.close();
        rope.k.close();
      }
      return;
    }
    if (kind == 'same_run_fused_vs_direct') {
      final out = runner.debugSecondDecodeAttentionDirectAndFusedFromVisionFeatures(
        promptIds,
        hidden,
        gridHeight: (meta['grid_h'] as num).toInt(),
        gridWidth: (meta['grid_w'] as num).toInt(),
        layerIndex: layerIndex,
      );
      try {
        print('kind=same_run_fused_vs_direct compare=$compare firstToken=${out.firstToken}');
        print('lastPath=${runner.debugTurboLastAttentionPath()}');
        print('sumAbsDiff=${_sumAbsDiff(out.fusedOutput, out.directOutput)}');
        print('maxAbsDiff=${_maxAbsDiff(out.fusedOutput, out.directOutput)}');
        print('fusedSample=${_sample(out.fusedOutput, 16)}');
        print('directSample=${_sample(out.directOutput, 16)}');
      } finally {
        out.directOutput.close();
        out.fusedOutput.close();
      }
      return;
    }
    if (kind == 'qkv_vs_py') {
      final pyQ =
          mx.io.load(Platform.environment['TQ_PY_Q'] ?? '/tmp/tq_py_turbo_step_qproj.npy');
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_step_kproj.npy');
      final pyV =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_turbo_step_vproj.npy');
      try {
        final qkv = runner.debugSecondDecodeProjectedQkvFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=qkv_vs_py compare=$compare layer=$layerIndex firstToken=${qkv.firstToken}');
          print('qDiff=${_sumAbsDiff(qkv.q, pyQ)}');
          print('kDiff=${_sumAbsDiff(qkv.k, pyK)}');
          print('vDiff=${_sumAbsDiff(qkv.v, pyV)}');
          print('qMaxDiff=${_maxAbsDiff(qkv.q, pyQ)}');
          print('kMaxDiff=${_maxAbsDiff(qkv.k, pyK)}');
          print('vMaxDiff=${_maxAbsDiff(qkv.v, pyV)}');
        } finally {
          qkv.q.close();
          qkv.k.close();
          qkv.v.close();
        }
      } finally {
        pyQ.close();
        pyK.close();
        pyV.close();
      }
      return;
    }
    if (kind == 'second_norm1_vs_py') {
      final py =
          mx.io.load(Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_py_turbo_step_norm1.npy');
      try {
        final norm1 = runner.debugSecondDecodeNorm1FromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=second_norm1_vs_py compare=$compare layer=$layerIndex firstToken=${norm1.firstToken}');
          print('sumAbsDiff=${_sumAbsDiff(norm1.norm1, py)}');
          print('maxAbsDiff=${_maxAbsDiff(norm1.norm1, py)}');
          print('localSample=${_sample(norm1.norm1)}');
          print('pySample=${_sample(py)}');
        } finally {
          norm1.norm1.close();
        }
      } finally {
        py.close();
      }
      return;
    }
    if (kind == 'fast_norm1_vs_py') {
      final py =
          mx.io.load(Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_py_turbo_step_norm1.npy');
      final local = runner.debugEmbedIds([806]);
      final weight = runner.debugLmInputNormWeight(layerIndex);
      try {
        final norm = mx.fast.rmsNorm(
          local,
          weight: weight,
          eps: runner.config.rmsNormEps,
        );
        try {
          print('kind=fast_norm1_vs_py');
          print('sumAbsDiff=${_sumAbsDiff(norm, py)}');
          print('maxAbsDiff=${_maxAbsDiff(norm, py)}');
          print('localSample=${_sample(norm)}');
          print('pySample=${_sample(py)}');
        } finally {
          norm.close();
        }
      } finally {
        local.close();
        weight.close();
        py.close();
      }
      return;
    }
    if (kind == 'embed_vs_py') {
      final py =
          mx.io.load(Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_py_turbo_step_embed.npy');
      try {
        final local = runner.debugEmbedIds([806]);
        final w = runner.debugLmInputNormWeight(0);
        try {
          print('kind=embed_vs_py');
          print('localDtype=${local.dtype} pyDtype=${py.dtype} weightDtype=${w.dtype}');
          print('sumAbsDiff=${_sumAbsDiff(local, py)}');
          print('maxAbsDiff=${_maxAbsDiff(local, py)}');
          print('localSample=${_sample(local)}');
          print('pySample=${_sample(py)}');
        } finally {
          local.close();
          w.close();
        }
      } finally {
        py.close();
      }
      return;
    }
    if (kind == 'qkv_apply_vs_py') {
      final pyNorm =
          mx.io.load(Platform.environment['TQ_PY_NORM'] ?? '/tmp/tq_cur_l0_norm1.npy');
      final pyQ =
          mx.io.load(Platform.environment['TQ_PY_Q'] ?? '/tmp/tq_py_turbo_step_qproj.npy');
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_step_kproj.npy');
      final pyV =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_turbo_step_vproj.npy');
      try {
        final qkv = runner.debugLmProjectedQkvApply(pyNorm, layerIndex);
        try {
          print('kind=qkv_apply_vs_py layer=$layerIndex');
          print('qDiff=${_sumAbsDiff(qkv.q, pyQ)}');
          print('kDiff=${_sumAbsDiff(qkv.k, pyK)}');
          print('vDiff=${_sumAbsDiff(qkv.v, pyV)}');
          print('qMaxDiff=${_maxAbsDiff(qkv.q, pyQ)}');
          print('kMaxDiff=${_maxAbsDiff(qkv.k, pyK)}');
          print('vMaxDiff=${_maxAbsDiff(qkv.v, pyV)}');
        } finally {
          qkv.q.close();
          qkv.k.close();
          qkv.v.close();
        }
      } finally {
        pyNorm.close();
        pyQ.close();
        pyK.close();
        pyV.close();
      }
      return;
    }
    if (kind == 'layerout_vs_py') {
      final pyPath =
          Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_py_turbo_layer0_out.npy';
      final py = mx.io.load(pyPath);
      try {
        final out = runner.debugSecondDecodeLayerOutputFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=layerout_vs_py compare=$compare layer=$layerIndex firstToken=${out.firstToken}');
          print('outShape=${out.layerOutput.shape} pyShape=${py.shape}');
          print('sumAbsDiff=${_sumAbsDiff(out.layerOutput, py)}');
          print('maxAbsDiff=${_maxAbsDiff(out.layerOutput, py)}');
          print('outSample=${_sample(out.layerOutput)}');
          print('pySample=${_sample(py)}');
        } finally {
          out.layerOutput.close();
        }
      } finally {
        py.close();
      }
      return;
    }
    if (kind == 'prefill_cache_vs_py') {
      final prefix =
          Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_py_turbo_l0_prefill';
      final pyKn = mx.io.load('${prefix}_kn.npy');
      final pyKi = mx.io.load('${prefix}_ki.npy');
      final pyVn = mx.io.load('${prefix}_vn.npy');
      final pyVi = mx.io.load('${prefix}_vi.npy');
      try {
        final result = runner.debugPrefillLayerCacheStateFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=prefill_cache_vs_py compare=$compare layer=$layerIndex');
          print('keyNormDiff=${_sumAbsDiff(result.keyNorms, pyKn)}');
          print('keyIdxDiff=${_sumAbsDiff(result.keyIndices, pyKi)}');
          print('valueNormDiff=${_sumAbsDiff(result.valueNorms, pyVn)}');
          print('valueIdxDiff=${_sumAbsDiff(result.valueIndices, pyVi)}');
        } finally {
          result.keyNorms.close();
          result.keyIndices.close();
          result.valueNorms.close();
          result.valueIndices.close();
        }
      } finally {
        pyKn.close();
        pyKi.close();
        pyVn.close();
        pyVi.close();
      }
      return;
    }
    if (kind == 'second_cache_vs_py') {
      final prefix =
          Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_py_turbo_l0_second';
      final pyKn = mx.io.load('${prefix}_kn.npy');
      final pyKi = mx.io.load('${prefix}_ki.npy');
      final pyVn = mx.io.load('${prefix}_vn.npy');
      final pyVi = mx.io.load('${prefix}_vi.npy');
      try {
        final result = runner.debugSecondDecodeLayerCacheStateFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=second_cache_vs_py compare=$compare layer=$layerIndex firstToken=${result.firstToken}');
          print('keyNormDiff=${_sumAbsDiff(result.keyNorms, pyKn)}');
          print('keyIdxDiff=${_sumAbsDiff(result.keyIndices, pyKi)}');
          print('valueNormDiff=${_sumAbsDiff(result.valueNorms, pyVn)}');
          print('valueIdxDiff=${_sumAbsDiff(result.valueIndices, pyVi)}');
          final lKn = result.keyNorms.slice(start: [0, 0, result.keyNorms.shape[2] - 1], stop: [1, result.keyNorms.shape[1], result.keyNorms.shape[2]]);
          final lKi = result.keyIndices.slice(start: [0, 0, result.keyIndices.shape[2] - 1, 0], stop: [1, result.keyIndices.shape[1], result.keyIndices.shape[2], result.keyIndices.shape[3]]);
          final lVn = result.valueNorms.slice(start: [0, 0, result.valueNorms.shape[2] - 1], stop: [1, result.valueNorms.shape[1], result.valueNorms.shape[2]]);
          final lVi = result.valueIndices.slice(start: [0, 0, result.valueIndices.shape[2] - 1, 0], stop: [1, result.valueIndices.shape[1], result.valueIndices.shape[2], result.valueIndices.shape[3]]);
          final pKn = pyKn.slice(start: [0, 0, pyKn.shape[2] - 1], stop: [1, pyKn.shape[1], pyKn.shape[2]]);
          final pKi = pyKi.slice(start: [0, 0, pyKi.shape[2] - 1, 0], stop: [1, pyKi.shape[1], pyKi.shape[2], pyKi.shape[3]]);
          final pVn = pyVn.slice(start: [0, 0, pyVn.shape[2] - 1], stop: [1, pyVn.shape[1], pyVn.shape[2]]);
          final pVi = pyVi.slice(start: [0, 0, pyVi.shape[2] - 1, 0], stop: [1, pyVi.shape[1], pyVi.shape[2], pyVi.shape[3]]);
          try {
            print('lastKeyNormDiff=${_sumAbsDiff(lKn, pKn)}');
            print('lastKeyIdxDiff=${_sumAbsDiff(lKi, pKi)}');
            print('lastValueNormDiff=${_sumAbsDiff(lVn, pVn)}');
            print('lastValueIdxDiff=${_sumAbsDiff(lVi, pVi)}');
            if ((Platform.environment['TQ_PRINT_LAST_KEY'] ?? '') == '1') {
              final localUnpacked = runner.debugTurboQuantUnpackIndices(
                lKi,
                bits: 3,
                length: 128,
              );
              final pyUnpacked = runner.debugTurboQuantUnpackIndices(
                pKi,
                bits: 3,
                length: 128,
              );
              try {
                print('lastKeyPackedLocal=${_sample(lKi, 12)}');
                print('lastKeyPackedPy=${_sample(pKi, 12)}');
                print('lastKeyUnpackedLocal=${_sample(localUnpacked, 32)}');
                print('lastKeyUnpackedPy=${_sample(pyUnpacked, 32)}');
                print('lastKeyMismatchPos=${_mismatchPositions(localUnpacked, pyUnpacked, 24)}');
              } finally {
                localUnpacked.close();
                pyUnpacked.close();
              }
            }
            final preKn = result.keyNorms.slice(start: [0, 0, 0], stop: [1, result.keyNorms.shape[1], result.keyNorms.shape[2] - 1]);
            final preKi = result.keyIndices.slice(start: [0, 0, 0, 0], stop: [1, result.keyIndices.shape[1], result.keyIndices.shape[2] - 1, result.keyIndices.shape[3]]);
            final preVn = result.valueNorms.slice(start: [0, 0, 0], stop: [1, result.valueNorms.shape[1], result.valueNorms.shape[2] - 1]);
            final preVi = result.valueIndices.slice(start: [0, 0, 0, 0], stop: [1, result.valueIndices.shape[1], result.valueIndices.shape[2] - 1, result.valueIndices.shape[3]]);
            final pyPreKn = pyKn.slice(start: [0, 0, 0], stop: [1, pyKn.shape[1], pyKn.shape[2] - 1]);
            final pyPreKi = pyKi.slice(start: [0, 0, 0, 0], stop: [1, pyKi.shape[1], pyKi.shape[2] - 1, pyKi.shape[3]]);
            final pyPreVn = pyVn.slice(start: [0, 0, 0], stop: [1, pyVn.shape[1], pyVn.shape[2] - 1]);
            final pyPreVi = pyVi.slice(start: [0, 0, 0, 0], stop: [1, pyVi.shape[1], pyVi.shape[2] - 1, pyVi.shape[3]]);
            try {
              print('prefillPartKeyNormDiff=${_sumAbsDiff(preKn, pyPreKn)}');
              print('prefillPartKeyIdxDiff=${_sumAbsDiff(preKi, pyPreKi)}');
              print('prefillPartValueNormDiff=${_sumAbsDiff(preVn, pyPreVn)}');
              print('prefillPartValueIdxDiff=${_sumAbsDiff(preVi, pyPreVi)}');
            } finally {
              preKn.close();
              preKi.close();
              preVn.close();
              preVi.close();
              pyPreKn.close();
              pyPreKi.close();
              pyPreVn.close();
              pyPreVi.close();
            }
          } finally {
            lKn.close();
            lKi.close();
            lVn.close();
            lVi.close();
            pKn.close();
            pKi.close();
            pVn.close();
            pVi.close();
          }
        } finally {
          result.keyNorms.close();
          result.keyIndices.close();
          result.valueNorms.close();
          result.valueIndices.close();
        }
      } finally {
        pyKn.close();
        pyKi.close();
        pyVn.close();
        pyVi.close();
      }
      return;
    }
    if (kind == 'second_token_quant_vs_py') {
      final disableSingle = (Platform.environment['TQ_DISABLE_SINGLE'] ?? '') == '1';
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_step_krope.npy');
      final pyVFlat =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_turbo_step_vproj.npy');
      final pyKnAll =
          mx.io.load(Platform.environment['TQ_PY_KN'] ?? '/tmp/tq_py_turbo_l0_second_kn.npy');
      final pyKiAll =
          mx.io.load(Platform.environment['TQ_PY_KI'] ?? '/tmp/tq_py_turbo_l0_second_ki.npy');
      final pyVnAll =
          mx.io.load(Platform.environment['TQ_PY_VN'] ?? '/tmp/tq_py_turbo_l0_second_vn.npy');
      final pyViAll =
          mx.io.load(Platform.environment['TQ_PY_VI'] ?? '/tmp/tq_py_turbo_l0_second_vi.npy');
      try {
        final pyV = pyVFlat.reshape([1, 1, 2, 128]).transposeAxes([0, 2, 1, 3]);
        try {
          final oldDisableSingle = PaddleOcrVlDebugOverrides.turboDisableSingleQuant;
          PaddleOcrVlDebugOverrides.turboDisableSingleQuant = disableSingle;
          final keyOut = runner.debugTurboQuantMseQuantize(pyK, bits: 3, seed: 0);
          final valueOut = runner.debugTurboQuantMseQuantize(pyV, bits: 4, seed: 1);
          PaddleOcrVlDebugOverrides.turboDisableSingleQuant = oldDisableSingle;
          try {
            final pyKn = pyKnAll.slice(start: [0, 0, pyKnAll.shape[2] - 1], stop: [1, pyKnAll.shape[1], pyKnAll.shape[2]]);
            final pyKi = pyKiAll.slice(start: [0, 0, pyKiAll.shape[2] - 1, 0], stop: [1, pyKiAll.shape[1], pyKiAll.shape[2], pyKiAll.shape[3]]);
            final pyVn = pyVnAll.slice(start: [0, 0, pyVnAll.shape[2] - 1], stop: [1, pyVnAll.shape[1], pyVnAll.shape[2]]);
            final pyVi = pyViAll.slice(start: [0, 0, pyViAll.shape[2] - 1, 0], stop: [1, pyViAll.shape[1], pyViAll.shape[2], pyViAll.shape[3]]);
            try {
              print('kind=second_token_quant_vs_py disableSingle=$disableSingle');
              print('keyNormDiff=${_sumAbsDiff(keyOut.norms, pyKn)}');
              print('keyIdxDiff=${_sumAbsDiff(keyOut.indices, pyKi)}');
              print('valueNormDiff=${_sumAbsDiff(valueOut.norms, pyVn)}');
              print('valueIdxDiff=${_sumAbsDiff(valueOut.indices, pyVi)}');
            } finally {
              pyKn.close();
              pyKi.close();
              pyVn.close();
              pyVi.close();
            }
          } finally {
            keyOut.norms.close();
            keyOut.indices.close();
            valueOut.norms.close();
            valueOut.indices.close();
          }
        } finally {
          pyV.close();
        }
      } finally {
        pyK.close();
        pyVFlat.close();
        pyKnAll.close();
        pyKiAll.close();
        pyVnAll.close();
        pyViAll.close();
      }
      return;
    }
    if (kind == 'fused_kv_vs_py') {
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_step_krope.npy');
      final pyVFlat =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_turbo_step_vproj.npy');
      final pyKnAll =
          mx.io.load(Platform.environment['TQ_PY_KN'] ?? '/tmp/tq_py_turbo_l0_second_kn.npy');
      final pyKiAll =
          mx.io.load(Platform.environment['TQ_PY_KI'] ?? '/tmp/tq_py_turbo_l0_second_ki.npy');
      final pyVnAll =
          mx.io.load(Platform.environment['TQ_PY_VN'] ?? '/tmp/tq_py_turbo_l0_second_vn.npy');
      final pyViAll =
          mx.io.load(Platform.environment['TQ_PY_VI'] ?? '/tmp/tq_py_turbo_l0_second_vi.npy');
      try {
        final pyV = pyVFlat.reshape([1, 1, 2, 128]).transposeAxes([0, 2, 1, 3]);
        try {
          final out = runner.debugTurboQuantFusedKv(pyK, pyV);
          try {
            final pyKn = pyKnAll.slice(start: [0, 0, pyKnAll.shape[2] - 1], stop: [1, pyKnAll.shape[1], pyKnAll.shape[2]]);
            final pyKi = pyKiAll.slice(start: [0, 0, pyKiAll.shape[2] - 1, 0], stop: [1, pyKiAll.shape[1], pyKiAll.shape[2], pyKiAll.shape[3]]);
            final pyVn = pyVnAll.slice(start: [0, 0, pyVnAll.shape[2] - 1], stop: [1, pyVnAll.shape[1], pyVnAll.shape[2]]);
            final pyVi = pyViAll.slice(start: [0, 0, pyViAll.shape[2] - 1, 0], stop: [1, pyViAll.shape[1], pyViAll.shape[2], pyViAll.shape[3]]);
            try {
              print('kind=fused_kv_vs_py');
              print('keyNormDiff=${_sumAbsDiff(out.keyNorms, pyKn)}');
              print('keyIdxDiff=${_sumAbsDiff(out.keyIndices, pyKi)}');
              print('valueNormDiff=${_sumAbsDiff(out.valueNorms, pyVn)}');
              print('valueIdxDiff=${_sumAbsDiff(out.valueIndices, pyVi)}');
            } finally {
              pyKn.close();
              pyKi.close();
              pyVn.close();
              pyVi.close();
            }
          } finally {
            out.keyNorms.close();
            out.keyIndices.close();
            out.valueNorms.close();
            out.valueIndices.close();
          }
        } finally {
          pyV.close();
        }
      } finally {
        pyK.close();
        pyVFlat.close();
        pyKnAll.close();
        pyKiAll.close();
        pyVnAll.close();
        pyViAll.close();
      }
      return;
    }
    if (kind == 'second_token_prepare_vs_py') {
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_step_krope.npy');
      final pyVFlat =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_turbo_step_vproj.npy');
      final pyKNorm =
          mx.io.load(Platform.environment['TQ_PY_KNORM'] ?? '/tmp/tq_py_turbo_k_norms.npy');
      final pyKUnit =
          mx.io.load(Platform.environment['TQ_PY_KUNIT'] ?? '/tmp/tq_py_turbo_k_unit.npy');
      final pyKRot =
          mx.io.load(Platform.environment['TQ_PY_KROT'] ?? '/tmp/tq_py_turbo_k_rot.npy');
      final pyVNorm =
          mx.io.load(Platform.environment['TQ_PY_VNORM'] ?? '/tmp/tq_py_turbo_v_norms.npy');
      final pyVUnit =
          mx.io.load(Platform.environment['TQ_PY_VUNIT'] ?? '/tmp/tq_py_turbo_v_unit.npy');
      final pyVRot =
          mx.io.load(Platform.environment['TQ_PY_VROT'] ?? '/tmp/tq_py_turbo_v_rot.npy');
      try {
        final pyV = pyVFlat.reshape([1, 1, 2, 128]).transposeAxes([0, 2, 1, 3]);
        try {
          final kPrep = runner.debugTurboQuantMsePrepare(pyK, bits: 3, seed: 0);
          final vPrep = runner.debugTurboQuantMsePrepare(pyV, bits: 4, seed: 1);
          try {
            print('kind=second_token_prepare_vs_py');
            print('kNormDiff=${_sumAbsDiff(kPrep.norms, pyKNorm)}');
            print('kUnitDiff=${_sumAbsDiff(kPrep.unit, pyKUnit)}');
            print('kRotDiff=${_sumAbsDiff(kPrep.rotated, pyKRot)}');
            print('vNormDiff=${_sumAbsDiff(vPrep.norms, pyVNorm)}');
            print('vUnitDiff=${_sumAbsDiff(vPrep.unit, pyVUnit)}');
            print('vRotDiff=${_sumAbsDiff(vPrep.rotated, pyVRot)}');
          } finally {
            kPrep.norms.close();
            kPrep.unit.close();
            kPrep.rotated.close();
            vPrep.norms.close();
            vPrep.unit.close();
            vPrep.rotated.close();
          }
        } finally {
          pyV.close();
        }
      } finally {
        pyK.close();
        pyVFlat.close();
        pyKNorm.close();
        pyKUnit.close();
        pyKRot.close();
        pyVNorm.close();
        pyVUnit.close();
        pyVRot.close();
      }
      return;
    }
    if (kind == 'rotation_vs_py') {
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_turbo_k_rotation.npy');
      final pyV =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_turbo_v_rotation.npy');
      try {
        final localK = runner.debugTurboQuantRotation(128, bits: 3, seed: 0);
        final localV = runner.debugTurboQuantRotation(128, bits: 4, seed: 1);
        try {
          print('kind=rotation_vs_py');
          print('kDiff=${_sumAbsDiff(localK, pyK)}');
          print('vDiff=${_sumAbsDiff(localV, pyV)}');
          print('kMaxDiff=${_maxAbsDiff(localK, pyK)}');
          print('vMaxDiff=${_maxAbsDiff(localV, pyV)}');
        } finally {
          localK.close();
          localV.close();
        }
      } finally {
        pyK.close();
        pyV.close();
      }
      return;
    }
    if (kind == 'codebook_vs_py') {
      final pyKC =
          mx.io.load(Platform.environment['TQ_PY_KC'] ?? '/tmp/tq_py_turbo_k_codebook.npy');
      final pyKM =
          mx.io.load(Platform.environment['TQ_PY_KM'] ?? '/tmp/tq_py_turbo_k_midpoints.npy');
      final pyVC =
          mx.io.load(Platform.environment['TQ_PY_VC'] ?? '/tmp/tq_py_turbo_v_codebook.npy');
      final pyVM =
          mx.io.load(Platform.environment['TQ_PY_VM'] ?? '/tmp/tq_py_turbo_v_midpoints.npy');
      try {
        final k = runner.debugTurboQuantCodebook(128, bits: 3, seed: 0);
        final v = runner.debugTurboQuantCodebook(128, bits: 4, seed: 1);
        try {
          print('kind=codebook_vs_py');
          print('kCodebookDiff=${_sumAbsDiff(k.codebook, pyKC)}');
          print('kMidpointsDiff=${_sumAbsDiff(k.midpoints, pyKM)}');
          print('vCodebookDiff=${_sumAbsDiff(v.codebook, pyVC)}');
          print('vMidpointsDiff=${_sumAbsDiff(v.midpoints, pyVM)}');
        } finally {
          k.codebook.close();
          k.midpoints.close();
          v.codebook.close();
          v.midpoints.close();
        }
      } finally {
        pyKC.close();
        pyKM.close();
        pyVC.close();
        pyVM.close();
      }
      return;
    }
    if (kind == 'inverse_vs_py') {
      final pyRot =
          mx.io.load(Platform.environment['TQ_PY_ROT'] ?? '/tmp/tq_py_turbo_v_rot.npy');
      final pyUnit =
          mx.io.load(Platform.environment['TQ_PY_UNIT'] ?? '/tmp/tq_py_turbo_v_unit.npy');
      try {
        final local = runner.debugTurboQuantInverseRotate(
          pyRot,
          dim: 128,
          bits: 4,
          seed: 1,
        );
        try {
          print('kind=inverse_vs_py');
          print('sumAbsDiff=${_sumAbsDiff(local, pyUnit)}');
          print('maxAbsDiff=${_maxAbsDiff(local, pyUnit)}');
          print('localSample=${_sample(local, 16)}');
          print('pySample=${_sample(pyUnit, 16)}');
        } finally {
          local.close();
        }
      } finally {
        pyRot.close();
        pyUnit.close();
      }
      return;
    }
    if (kind == 'dense_prefill_vs_py') {
      final pyK =
          mx.io.load(Platform.environment['TQ_PY_K'] ?? '/tmp/tq_py_dense_l0_prefill_k_cur.npy');
      final pyV =
          mx.io.load(Platform.environment['TQ_PY_V'] ?? '/tmp/tq_py_dense_l0_prefill_v_cur.npy');
      try {
        final result = runner.debugPrefillDenseLayerCacheStateFromVisionFeatures(
          promptIds,
          hidden,
          gridHeight: (meta['grid_h'] as num).toInt(),
          gridWidth: (meta['grid_w'] as num).toInt(),
          layerIndex: layerIndex,
        );
        try {
          print('kind=dense_prefill_vs_py layer=$layerIndex');
          print('keyDiff=${_sumAbsDiff(result.keys, pyK)}');
          print('valueDiff=${_sumAbsDiff(result.values, pyV)}');
        } finally {
          result.keys.close();
          result.values.close();
        }
      } finally {
        pyK.close();
        pyV.close();
      }
      return;
    }
    if (kind == 'inputnorm_vs_py') {
      final py =
          mx.io.load(Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_cur_l0_inputnorm.npy');
      try {
        final local = runner.debugLmInputNormWeight(layerIndex);
        try {
          print('kind=inputnorm_vs_py layer=$layerIndex');
          print('sumAbsDiff=${_sumAbsDiff(local, py)}');
          print('maxAbsDiff=${_maxAbsDiff(local, py)}');
        } finally {
          local.close();
        }
      } finally {
        py.close();
      }
      return;
    }
    if (kind == 'norm1_vs_py') {
      final py =
          mx.io.load(Platform.environment['TQ_PY_PATH'] ?? '/tmp/tq_cur_l0_norm1.npy');
      final useManual = (Platform.environment['TQ_USE_MANUAL'] ?? '') == '1';
      try {
        final local = useManual
            ? runner.debugLmNorm1ManualFromVisionFeatures(
                promptIds,
                hidden,
                gridHeight: (meta['grid_h'] as num).toInt(),
                gridWidth: (meta['grid_w'] as num).toInt(),
                layerIndex: layerIndex,
              )
            : runner.debugLmNorm1FromVisionFeatures(
                promptIds,
                hidden,
                gridHeight: (meta['grid_h'] as num).toInt(),
                gridWidth: (meta['grid_w'] as num).toInt(),
                layerIndex: layerIndex,
              );
        try {
          print('kind=norm1_vs_py manual=$useManual layer=$layerIndex');
          print('sumAbsDiff=${_sumAbsDiff(local, py)}');
          print('maxAbsDiff=${_maxAbsDiff(local, py)}');
          print('localSample=${_sample(local)}');
          print('pySample=${_sample(py)}');
        } finally {
          local.close();
        }
      } finally {
        py.close();
      }
      return;
    }
    throw ArgumentError('Unknown TQ_KIND=$kind');
  } finally {
    _restoreOverrides(snapshot);
    hidden.close();
    runner.close();
  }
}
