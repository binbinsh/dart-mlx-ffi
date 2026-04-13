import 'dart:io';
import 'dart:convert';

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
    Platform.environment['TQ_HIDDEN_PATH'] ??
    '/tmp/paddle_v15_ref/vision_projected.npy';

void Function(String message)? _stageLogger() {
  final value = Platform.environment['TQ_STAGE_LOG'];
  if (value == null) return null;
  if (value != '1' && value.toLowerCase() != 'true') return null;
  return (message) => stdout.writeln(message);
}

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

double _diffCompatible(MlxArray a, MlxArray b) {
  if (_sameShape(a.shape, b.shape)) {
    return _diff(a, b);
  }
  if (a.size != b.size) {
    throw StateError(
      'shape mismatch: a=${a.shape.join("x")} b=${b.shape.join("x")}',
    );
  }
  final reshaped = b.reshape(a.shape);
  try {
    return _diff(a, reshaped);
  } finally {
    reshaped.close();
  }
}

bool _sameShape(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

typedef _OverridesSnapshot = ({
  String? kvQuantScheme,
  double? turboBits,
  int? turboStart,
  bool? turboDensePrefill,
  bool? uniformQuantizedPrefill,
  bool? turboDisableFusedKv,
  bool? turboDisableFusedDecode,
      bool? turboDisableSingleQuant,
      bool? turboDisableFastScore,
      bool? turboDisableFastValue,
      int? turboCompactBudget,
      int? turboCompactKeepRecent,
      int? turboCompactInterval,
      int? turboCompactKeepPrefix,
      int? turboCompactHysteresis,
    });

_OverridesSnapshot _captureOverrides() => (
  kvQuantScheme: PaddleOcrVlDebugOverrides.kvQuantScheme,
  turboBits: PaddleOcrVlDebugOverrides.turboBits,
  turboStart: PaddleOcrVlDebugOverrides.turboStart,
  turboDensePrefill: PaddleOcrVlDebugOverrides.turboDensePrefill,
  uniformQuantizedPrefill: PaddleOcrVlDebugOverrides.uniformQuantizedPrefill,
  turboDisableFusedKv: PaddleOcrVlDebugOverrides.turboDisableFusedKv,
  turboDisableFusedDecode: PaddleOcrVlDebugOverrides.turboDisableFusedDecode,
  turboDisableSingleQuant: PaddleOcrVlDebugOverrides.turboDisableSingleQuant,
  turboDisableFastScore: PaddleOcrVlDebugOverrides.turboDisableFastScore,
  turboDisableFastValue: PaddleOcrVlDebugOverrides.turboDisableFastValue,
  turboCompactBudget: PaddleOcrVlDebugOverrides.turboCompactBudget,
  turboCompactKeepRecent: PaddleOcrVlDebugOverrides.turboCompactKeepRecent,
  turboCompactInterval: PaddleOcrVlDebugOverrides.turboCompactInterval,
  turboCompactKeepPrefix: PaddleOcrVlDebugOverrides.turboCompactKeepPrefix,
  turboCompactHysteresis: PaddleOcrVlDebugOverrides.turboCompactHysteresis,
);

void _restoreOverrides(_OverridesSnapshot s) {
  PaddleOcrVlDebugOverrides.kvQuantScheme = s.kvQuantScheme;
  PaddleOcrVlDebugOverrides.turboBits = s.turboBits;
  PaddleOcrVlDebugOverrides.turboStart = s.turboStart;
  PaddleOcrVlDebugOverrides.turboDensePrefill = s.turboDensePrefill;
  PaddleOcrVlDebugOverrides.uniformQuantizedPrefill = s.uniformQuantizedPrefill;
  PaddleOcrVlDebugOverrides.turboDisableFusedKv = s.turboDisableFusedKv;
  PaddleOcrVlDebugOverrides.turboDisableFusedDecode = s.turboDisableFusedDecode;
  PaddleOcrVlDebugOverrides.turboDisableSingleQuant = s.turboDisableSingleQuant;
  PaddleOcrVlDebugOverrides.turboDisableFastScore = s.turboDisableFastScore;
  PaddleOcrVlDebugOverrides.turboDisableFastValue = s.turboDisableFastValue;
  PaddleOcrVlDebugOverrides.turboCompactBudget = s.turboCompactBudget;
  PaddleOcrVlDebugOverrides.turboCompactKeepRecent = s.turboCompactKeepRecent;
  PaddleOcrVlDebugOverrides.turboCompactInterval = s.turboCompactInterval;
  PaddleOcrVlDebugOverrides.turboCompactKeepPrefix = s.turboCompactKeepPrefix;
  PaddleOcrVlDebugOverrides.turboCompactHysteresis = s.turboCompactHysteresis;
}

void _runPrefix(PaddleOcrVlRunner runner, List<int> ids, MlxArray image) {
  final maxNewTokens =
      int.tryParse(Platform.environment['TQ_MAX_NEW_TOKENS'] ?? '') ?? 32;
  final result = runner.generateFromImageDetailed(
    ids,
    image,
    maxNewTokens: maxNewTokens,
  );
  final generated = result.fullTokenIds.sublist(result.expandedPromptLength);
  print(generated.join(','));
}

void _runPrefixFromHidden(PaddleOcrVlRunner runner, List<int> ids) {
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final maxNewTokens =
      int.tryParse(Platform.environment['TQ_MAX_NEW_TOKENS'] ?? '') ?? 32;
  try {
    final result = runner.generateFromVisionFeaturesDetailed(
      ids,
      hidden,
      gridHeight: (meta['grid_h'] as num).toInt(),
      gridWidth: (meta['grid_w'] as num).toInt(),
      maxNewTokens: maxNewTokens,
      onStage: _stageLogger(),
    );
    final generated = result.fullTokenIds.sublist(result.expandedPromptLength);
    print(generated.join(','));
  } finally {
    hidden.close();
  }
}

String _prefixFromHiddenString(PaddleOcrVlRunner runner, List<int> ids) {
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final maxNewTokens =
      int.tryParse(Platform.environment['TQ_MAX_NEW_TOKENS'] ?? '') ?? 32;
  try {
    final result = runner.generateFromVisionFeaturesDetailed(
      ids,
      hidden,
      gridHeight: (meta['grid_h'] as num).toInt(),
      gridWidth: (meta['grid_w'] as num).toInt(),
      maxNewTokens: maxNewTokens,
      onStage: _stageLogger(),
    );
    final generated = result.fullTokenIds.sublist(result.expandedPromptLength);
    return generated.join(',');
  } finally {
    hidden.close();
  }
}

void _runTurboAblationFromHidden(PaddleOcrVlRunner runner, List<int> ids) {
  final snapshot = _captureOverrides();
  try {
    final cases = <String, void Function()>{
      'baseline_uniform': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'uniform';
        PaddleOcrVlDebugOverrides.uniformQuantizedPrefill = null;
      },
      'turbo_all_fast': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = false;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
      },
      'turbo_no_fused_decode': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
      },
      'turbo_no_fast_score': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
      },
      'turbo_no_fast_value': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      },
      'turbo_no_single_quant': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
      },
      'turbo_no_fused_kv': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
      },
      'turbo_no_fused_decode_no_fast_sv_keep_single_fusedkv': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      },
      'turbo_no_fused_decode_no_fast_sv_no_single': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      },
      'turbo_no_fused_decode_no_fast_sv_no_fusedkv': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      },
      'turbo_fused_decode_only': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = false;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      },
      'turbo_fast_score_only': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      },
      'turbo_all_slow': () {
        PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
        PaddleOcrVlDebugOverrides.turboBits = 3.5;
        PaddleOcrVlDebugOverrides.turboStart = 0;
        PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
        PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
        PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
        PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
        PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
      },
    };
    for (final entry in cases.entries) {
      _restoreOverrides(snapshot);
      entry.value();
      final prefix = _prefixFromHiddenString(runner, ids);
      print('${entry.key}: $prefix');
    }
  } finally {
    _restoreOverrides(snapshot);
  }
}

void _applyTurboFast() {
  PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
  PaddleOcrVlDebugOverrides.turboBits = 3.5;
  PaddleOcrVlDebugOverrides.turboStart = 0;
  PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
  PaddleOcrVlDebugOverrides.turboDisableFusedDecode = false;
  PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
  PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
  PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
}

void _applyTurboSlow() {
  PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
  PaddleOcrVlDebugOverrides.turboBits = 3.5;
  PaddleOcrVlDebugOverrides.turboStart = 0;
  PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
  PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
  PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
  PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
  PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
}

void _runTurboFastSlowSecondLayerOutDiff(
  PaddleOcrVlRunner runner,
  List<int> ids,
) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final snapshot = _captureOverrides();
  final compareMode =
      Platform.environment['TQ_COMPARE_MODE'] ?? 'all_fast_vs_all_slow';
  try {
    if (compareMode == 'no_fused_decode_vs_slow') {
      PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
      PaddleOcrVlDebugOverrides.turboBits = 3.5;
      PaddleOcrVlDebugOverrides.turboStart = 0;
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
    } else if (compareMode == 'single_only_vs_slow') {
      PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
      PaddleOcrVlDebugOverrides.turboBits = 3.5;
      PaddleOcrVlDebugOverrides.turboStart = 0;
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
    } else if (compareMode == 'fusedkv_only_vs_slow') {
      PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
      PaddleOcrVlDebugOverrides.turboBits = 3.5;
      PaddleOcrVlDebugOverrides.turboStart = 0;
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
    } else {
      _applyTurboFast();
    }
    final hiddenFast = mx.io.load(_hiddenPath());
    final fast = runner.debugSecondDecodeLayerOutputFromVisionFeatures(
      ids,
      hiddenFast,
      gridHeight: (meta['grid_h'] as num).toInt(),
      gridWidth: (meta['grid_w'] as num).toInt(),
      layerIndex: layer,
    );
    hiddenFast.close();
    try {
      _applyTurboSlow();
      final hiddenSlow = mx.io.load(_hiddenPath());
      final slow = runner.debugSecondDecodeLayerOutputFromVisionFeatures(
        ids,
        hiddenSlow,
        gridHeight: (meta['grid_h'] as num).toInt(),
        gridWidth: (meta['grid_w'] as num).toInt(),
        layerIndex: layer,
      );
      hiddenSlow.close();
      try {
        print('fastFirst=${fast.firstToken} slowFirst=${slow.firstToken}');
        print(
          'fastShape=${fast.layerOutput.shape} slowShape=${slow.layerOutput.shape}',
        );
        print(
          'fastSlowLayerOutDiff=${_diffCompatible(fast.layerOutput, slow.layerOutput)}',
        );
      } finally {
        slow.layerOutput.close();
      }
    } finally {
      fast.layerOutput.close();
    }
  } finally {
    _restoreOverrides(snapshot);
  }
}

void _runTurboFastSlowSecondLayerCacheDiff(
  PaddleOcrVlRunner runner,
  List<int> ids,
) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final snapshot = _captureOverrides();
  final compareMode =
      Platform.environment['TQ_COMPARE_MODE'] ?? 'all_fast_vs_all_slow';
  try {
    if (compareMode == 'no_fused_decode_vs_slow') {
      PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
      PaddleOcrVlDebugOverrides.turboBits = 3.5;
      PaddleOcrVlDebugOverrides.turboStart = 0;
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = false;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = false;
    } else if (compareMode == 'single_only_vs_slow') {
      PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
      PaddleOcrVlDebugOverrides.turboBits = 3.5;
      PaddleOcrVlDebugOverrides.turboStart = 0;
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = true;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = false;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
    } else if (compareMode == 'fusedkv_only_vs_slow') {
      PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
      PaddleOcrVlDebugOverrides.turboBits = 3.5;
      PaddleOcrVlDebugOverrides.turboStart = 0;
      PaddleOcrVlDebugOverrides.turboDisableFusedKv = false;
      PaddleOcrVlDebugOverrides.turboDisableFusedDecode = true;
      PaddleOcrVlDebugOverrides.turboDisableSingleQuant = true;
      PaddleOcrVlDebugOverrides.turboDisableFastScore = true;
      PaddleOcrVlDebugOverrides.turboDisableFastValue = true;
    } else {
      _applyTurboFast();
    }
    final hiddenFast = mx.io.load(_hiddenPath());
    final fast = runner.debugSecondDecodeLayerCacheStateFromVisionFeatures(
      ids,
      hiddenFast,
      gridHeight: (meta['grid_h'] as num).toInt(),
      gridWidth: (meta['grid_w'] as num).toInt(),
      layerIndex: layer,
    );
    hiddenFast.close();
    try {
      _applyTurboSlow();
      final hiddenSlow = mx.io.load(_hiddenPath());
      final slow = runner.debugSecondDecodeLayerCacheStateFromVisionFeatures(
        ids,
        hiddenSlow,
        gridHeight: (meta['grid_h'] as num).toInt(),
        gridWidth: (meta['grid_w'] as num).toInt(),
        layerIndex: layer,
      );
      hiddenSlow.close();
      try {
        print('fastFirst=${fast.firstToken} slowFirst=${slow.firstToken}');
        print('keyNormDiff=${_diffCompatible(fast.keyNorms, slow.keyNorms)}');
        print(
          'keyIdxDiff=${_diffCompatible(fast.keyIndices, slow.keyIndices)}',
        );
        print(
          'valueNormDiff=${_diffCompatible(fast.valueNorms, slow.valueNorms)}',
        );
        print(
          'valueIdxDiff=${_diffCompatible(fast.valueIndices, slow.valueIndices)}',
        );
      } finally {
        slow.keyNorms.close();
        slow.keyIndices.close();
        slow.valueNorms.close();
        slow.valueIndices.close();
      }
    } finally {
      fast.keyNorms.close();
      fast.keyIndices.close();
      fast.valueNorms.close();
      fast.valueIndices.close();
    }
  } finally {
    _restoreOverrides(snapshot);
  }
}

void _runLayerScan(PaddleOcrVlRunner runner, List<int> ids, MlxArray image) {
  final start = int.tryParse(Platform.environment['TQ_LAYER_START'] ?? '') ?? 0;
  final end =
      int.tryParse(Platform.environment['TQ_LAYER_END'] ?? '') ??
      runner.config.numHiddenLayers;
  for (var layer = start; layer < end; layer++) {
    final result = runner.debugSecondDecodeAttentionOutputsFromImage(
      ids,
      image,
      layerIndex: layer,
    );
    try {
      final diff = _diffCompatible(result.directOutput, result.splitOutput);
      print(
        'layer=$layer firstToken=${result.firstToken} '
        'directShape=${result.directOutput.shape} '
        'splitShape=${result.splitOutput.shape} '
        'directSplitDiff=$diff',
      );
    } finally {
      result.directOutput.close();
      result.splitOutput.close();
    }
  }
}

void _runLayerScanFromHidden(PaddleOcrVlRunner runner, List<int> ids) {
  final start = int.tryParse(Platform.environment['TQ_LAYER_START'] ?? '') ?? 0;
  final end =
      int.tryParse(Platform.environment['TQ_LAYER_END'] ?? '') ??
      runner.config.numHiddenLayers;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  for (var layer = start; layer < end; layer++) {
    final hidden = mx.io.load(_hiddenPath());
    final result = runner.debugSecondDecodeAttentionOutputsFromVisionFeatures(
      ids,
      hidden,
      gridHeight: (meta['grid_h'] as num).toInt(),
      gridWidth: (meta['grid_w'] as num).toInt(),
      layerIndex: layer,
    );
    try {
      final diff = _diffCompatible(result.directOutput, result.splitOutput);
      print(
        'layer=$layer firstToken=${result.firstToken} '
        'directShape=${result.directOutput.shape} '
        'splitShape=${result.splitOutput.shape} '
        'directSplitDiff=$diff',
      );
    } finally {
      hidden.close();
      result.directOutput.close();
      result.splitOutput.close();
    }
  }
}

void _runLayerCacheDiff(
  PaddleOcrVlRunner runner,
  List<int> ids,
  MlxArray image,
) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final prefix =
      Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_py_turbo_l0_second';
  final result = runner.debugSecondDecodeLayerCacheStateFromImage(
    ids,
    image,
    layerIndex: layer,
  );
  final pyKeyNorms = mx.io.load('${prefix}_kn.npy');
  final pyKeyIndices = mx.io.load('${prefix}_ki.npy');
  final pyValueNorms = mx.io.load('${prefix}_vn.npy');
  final pyValueIndices = mx.io.load('${prefix}_vi.npy');
  try {
    print('layer=$layer firstToken=${result.firstToken}');
    print('keyNormDiff=${_diffCompatible(result.keyNorms, pyKeyNorms)}');
    print('keyIdxDiff=${_diffCompatible(result.keyIndices, pyKeyIndices)}');
    print('valueNormDiff=${_diffCompatible(result.valueNorms, pyValueNorms)}');
    print(
      'valueIdxDiff=${_diffCompatible(result.valueIndices, pyValueIndices)}',
    );
  } finally {
    pyKeyNorms.close();
    pyKeyIndices.close();
    pyValueNorms.close();
    pyValueIndices.close();
    result.keyNorms.close();
    result.keyIndices.close();
    result.valueNorms.close();
    result.valueIndices.close();
  }
}

void _runLayerCacheDiffFromHidden(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final prefix =
      Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_py_turbo_l0_second';
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyKeyNorms = mx.io.load('${prefix}_kn.npy');
  final pyKeyIndices = mx.io.load('${prefix}_ki.npy');
  final pyValueNorms = mx.io.load('${prefix}_vn.npy');
  final pyValueIndices = mx.io.load('${prefix}_vi.npy');
  final result = runner.debugSecondDecodeLayerCacheStateFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('layer=$layer firstToken=${result.firstToken}');
    print('keyNormDiff=${_diffCompatible(result.keyNorms, pyKeyNorms)}');
    print('keyIdxDiff=${_diffCompatible(result.keyIndices, pyKeyIndices)}');
    print('valueNormDiff=${_diffCompatible(result.valueNorms, pyValueNorms)}');
    print(
      'valueIdxDiff=${_diffCompatible(result.valueIndices, pyValueIndices)}',
    );
  } finally {
    hidden.close();
    pyKeyNorms.close();
    pyKeyIndices.close();
    pyValueNorms.close();
    pyValueIndices.close();
    result.keyNorms.close();
    result.keyIndices.close();
    result.valueNorms.close();
    result.valueIndices.close();
  }
}

void _runPrefillLayerCacheDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final prefix =
      Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_py_turbo_l0_prefill';
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyKeyNorms = mx.io.load('${prefix}_kn.npy');
  final pyKeyIndices = mx.io.load('${prefix}_ki.npy');
  final pyValueNorms = mx.io.load('${prefix}_vn.npy');
  final pyValueIndices = mx.io.load('${prefix}_vi.npy');
  final result = runner.debugPrefillLayerCacheStateFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('layer=$layer');
    print('keyNormDiff=${_diffCompatible(result.keyNorms, pyKeyNorms)}');
    print('keyIdxDiff=${_diffCompatible(result.keyIndices, pyKeyIndices)}');
    print('valueNormDiff=${_diffCompatible(result.valueNorms, pyValueNorms)}');
    print(
      'valueIdxDiff=${_diffCompatible(result.valueIndices, pyValueIndices)}',
    );
  } finally {
    hidden.close();
    pyKeyNorms.close();
    pyKeyIndices.close();
    pyValueNorms.close();
    pyValueIndices.close();
    result.keyNorms.close();
    result.keyIndices.close();
    result.valueNorms.close();
    result.valueIndices.close();
  }
}

void _runPrefillQuantizeDiff(PaddleOcrVlRunner runner) {
  final keyTensor = mx.io.load('/tmp/tq_py_dense_l0_prefill_k.npy');
  final valueTensor = mx.io.load('/tmp/tq_py_dense_l0_prefill_v.npy');
  final pyKeyNorms = mx.io.load('/tmp/tq_py_turbo_l0_prefill_kn.npy');
  final pyKeyIndices = mx.io.load('/tmp/tq_py_turbo_l0_prefill_ki.npy');
  final pyValueNorms = mx.io.load('/tmp/tq_py_turbo_l0_prefill_vn.npy');
  final pyValueIndices = mx.io.load('/tmp/tq_py_turbo_l0_prefill_vi.npy');
  final keyOut = runner.debugTurboQuantMseQuantize(keyTensor, bits: 3, seed: 0);
  final valueOut = runner.debugTurboQuantMseQuantize(
    valueTensor,
    bits: 4,
    seed: 1,
  );
  try {
    print('prefillKeyNormDiff=${_diffCompatible(keyOut.norms, pyKeyNorms)}');
    print('prefillKeyIdxDiff=${_diffCompatible(keyOut.indices, pyKeyIndices)}');
    print(
      'prefillValueNormDiff=${_diffCompatible(valueOut.norms, pyValueNorms)}',
    );
    print(
      'prefillValueIdxDiff=${_diffCompatible(valueOut.indices, pyValueIndices)}',
    );
  } finally {
    keyTensor.close();
    valueTensor.close();
    pyKeyNorms.close();
    pyKeyIndices.close();
    pyValueNorms.close();
    pyValueIndices.close();
    keyOut.norms.close();
    keyOut.indices.close();
    valueOut.norms.close();
    valueOut.indices.close();
  }
}

void _runDensePrefillCacheDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyKeys = mx.io.load('/tmp/tq_py_dense_l0_prefill_k.npy');
  final pyValues = mx.io.load('/tmp/tq_py_dense_l0_prefill_v.npy');
  final result = runner.debugPrefillDenseLayerCacheStateFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('layer=$layer');
    print('denseKeyDiff=${_diffCompatible(result.keys, pyKeys)}');
    print('denseValueDiff=${_diffCompatible(result.values, pyValues)}');
  } finally {
    hidden.close();
    pyKeys.close();
    pyValues.close();
    result.keys.close();
    result.values.close();
  }
}

void _runDensePrefillCacheDiffFromImage(
  PaddleOcrVlRunner runner,
  List<int> ids,
) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyKeys = mx.io.load('/tmp/tq_py_dense_l0_prefill_k.npy');
  final pyValues = mx.io.load('/tmp/tq_py_dense_l0_prefill_v.npy');
  final result = runner.debugPrefillDenseLayerCacheStateFromImage(
    ids,
    image,
    layerIndex: layer,
  );
  try {
    print('layer=$layer');
    print('denseKeyDiff=${_diffCompatible(result.keys, pyKeys)}');
    print('denseValueDiff=${_diffCompatible(result.values, pyValues)}');
  } finally {
    image.close();
    pyKeys.close();
    pyValues.close();
    result.keys.close();
    result.values.close();
  }
}

void _runInputsDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyEmb = mx.io.load('/tmp/tq_cur_inputs_embeds.npy');
  final pyPos = mx.io.load('/tmp/paddle_v15_ref/position_ids.npy');
  final result = runner.debugInputsEmbedsAndPositionIdsFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
  );
  try {
    print('embedDiff=${_diffCompatible(result.inputsEmbeds, pyEmb)}');
    print('positionDiff=${_diffCompatible(result.positionIds, pyPos)}');
  } finally {
    hidden.close();
    pyEmb.close();
    pyPos.close();
    result.inputsEmbeds.close();
    result.positionIds.close();
  }
}

void _runInputsPartitionDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyEmb = mx.io.load('/tmp/tq_cur_inputs_embeds.npy');
  final result = runner.debugInputsEmbedsAndPositionIdsFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
  );
  try {
    final imageMask = MlxArray.fromInt32List(
      ids.map((id) => id == runner.config.imageTokenId ? 1 : 0).toList(),
      shape: [ids.length],
    ).astype(MlxDType.MLX_BOOL);
    final textMask = mx.logicalNot(imageMask);
    final imageMask3d = imageMask.reshape([1, ids.length, 1]);
    final textMask3d = textMask.reshape([1, ids.length, 1]);
    final zero = MlxArray.zeros([1, ids.length, runner.config.hiddenSize]);
    final localImage = mx.where(imageMask3d, result.inputsEmbeds, zero);
    final pyImage = mx.where(imageMask3d, pyEmb, zero);
    final localText = mx.where(textMask3d, result.inputsEmbeds, zero);
    final pyText = mx.where(textMask3d, pyEmb, zero);
    print('imageEmbedDiff=${_diffCompatible(localImage, pyImage)}');
    print('textEmbedDiff=${_diffCompatible(localText, pyText)}');
    final localImageSlice = result.inputsEmbeds
        .slice(start: [0, 5, 0], stop: [1, 599, runner.config.hiddenSize])
        .reshape([594, runner.config.hiddenSize]);
    final pyImageSlice = pyEmb
        .slice(start: [0, 5, 0], stop: [1, 599, runner.config.hiddenSize])
        .reshape([594, runner.config.hiddenSize]);
    print('localImageVsHidden=${_diffCompatible(localImageSlice, hidden)}');
    print('pyImageVsHidden=${_diffCompatible(pyImageSlice, hidden)}');
    localImageSlice.close();
    pyImageSlice.close();
    zero.close();
    imageMask.close();
    textMask.close();
    imageMask3d.close();
    textMask3d.close();
    localImage.close();
    pyImage.close();
    localText.close();
    pyText.close();
  } finally {
    hidden.close();
    pyEmb.close();
    result.inputsEmbeds.close();
    result.positionIds.close();
  }
}

void _runPrefillQkvDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final prefix = Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_l0';
  final pyQ = mx.io.load('${prefix}_q.npy');
  final pyK = mx.io.load('${prefix}_k.npy');
  final pyV = mx.io.load('${prefix}_v.npy');
  final result = runner.debugLmProjectedQkvFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('localQShape=${result.q.shape} pyQShape=${pyQ.shape}');
    print('localKShape=${result.k.shape} pyKShape=${pyK.shape}');
    print('localVShape=${result.v.shape} pyVShape=${pyV.shape}');
    print('qDiff=${_diffCompatible(result.q, pyQ)}');
    print('kDiff=${_diffCompatible(result.k, pyK)}');
    print('vDiff=${_diffCompatible(result.v, pyV)}');
  } finally {
    hidden.close();
    pyQ.close();
    pyK.close();
    pyV.close();
    result.q.close();
    result.k.close();
    result.v.close();
  }
}

void _runPrefillNorm1Diff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final prefix = Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_l0';
  final pyNorm1 = mx.io.load('${prefix}_norm1.npy');
  final norm1 = runner.debugLmNorm1FromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('localNorm1Shape=${norm1.shape} pyNorm1Shape=${pyNorm1.shape}');
    print('norm1Diff=${_diffCompatible(norm1, pyNorm1)}');
  } finally {
    hidden.close();
    pyNorm1.close();
    norm1.close();
  }
}

void _runPrefillLayer0AttentionDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final prefix = Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_l0';
  final pyAttn = mx.io.load('${prefix}_attn.npy');
  final attn = runner.debugLmAttentionOutputFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('localAttnShape=${attn.shape} pyAttnShape=${pyAttn.shape}');
    print('attnDiff=${_diffCompatible(attn, pyAttn)}');
  } finally {
    hidden.close();
    pyAttn.close();
    attn.close();
  }
}

void _runPrefillLayer0OutDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final prefix = Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_l0';
  final pyOut = mx.io.load('${prefix}_out.npy');
  final out = runner.debugLmLayerOutputFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('localOutShape=${out.shape} pyOutShape=${pyOut.shape}');
    print('layer0OutDiff=${_diffCompatible(out, pyOut)}');
  } finally {
    hidden.close();
    pyOut.close();
    out.close();
  }
}

void _runSecondNorm1Diff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final prefix =
      Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_second_l0';
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyNorm1 = mx.io.load('${prefix}_norm1.npy');
  final out = runner.debugSecondDecodePostNormFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('firstToken=${out.firstToken}');
    print('localNorm1Shape=${out.norm2.shape} pyNorm1Shape=${pyNorm1.shape}');
    print('norm1Diff=${_diffCompatible(out.norm2, pyNorm1)}');
  } finally {
    hidden.close();
    pyNorm1.close();
    out.norm2.close();
  }
}

void _runSecondQkvDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final prefix =
      Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_second_l0';
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyQ = mx.io.load('${prefix}_q.npy');
  final pyK = mx.io.load('${prefix}_k.npy');
  final pyV = mx.io.load('${prefix}_v.npy');
  final out = runner.debugSecondDecodeProjectedQkvFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('firstToken=${out.firstToken}');
    print('localQShape=${out.q.shape} pyQShape=${pyQ.shape}');
    print('localKShape=${out.k.shape} pyKShape=${pyK.shape}');
    print('localVShape=${out.v.shape} pyVShape=${pyV.shape}');
    print('qDiff=${_diffCompatible(out.q, pyQ)}');
    print('kDiff=${_diffCompatible(out.k, pyK)}');
    print('vDiff=${_diffCompatible(out.v, pyV)}');
  } finally {
    hidden.close();
    pyQ.close();
    pyK.close();
    pyV.close();
    out.q.close();
    out.k.close();
    out.v.close();
  }
}

void _runSecondScoresDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final out = runner.debugSecondDecodeScoresFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('firstToken=${out.firstToken}');
    print(
      'fastScoresShape=${out.fastScores.shape} slowScoresShape=${out.slowScores.shape}',
    );
    print('scoresDiff=${_diffCompatible(out.fastScores, out.slowScores)}');
  } finally {
    hidden.close();
    out.fastScores.close();
    out.slowScores.close();
  }
}

void _runSecondAttnDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final prefix =
      Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_second_l0';
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyAttn = mx.io.load('${prefix}_attn.npy');
  final out = runner.debugSecondDecodeAttentionOutputFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('firstToken=${out.firstToken}');
    print(
      'localAttnShape=${out.attentionOutput.shape} pyAttnShape=${pyAttn.shape}',
    );
    print('attnDiff=${_diffCompatible(out.attentionOutput, pyAttn)}');
  } finally {
    hidden.close();
    pyAttn.close();
    out.attentionOutput.close();
  }
}

void _runSecondOutDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final layer = int.tryParse(Platform.environment['TQ_LAYER_INDEX'] ?? '') ?? 0;
  final prefix =
      Platform.environment['TQ_PY_PREFIX'] ?? '/tmp/tq_cur_second_l0';
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyOut = mx.io.load('${prefix}_out.npy');
  final out = runner.debugSecondDecodeLayerOutputFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
    layerIndex: layer,
  );
  try {
    print('firstToken=${out.firstToken}');
    print('localOutShape=${out.layerOutput.shape} pyOutShape=${pyOut.shape}');
    print('layerOutDiff=${_diffCompatible(out.layerOutput, pyOut)}');
  } finally {
    hidden.close();
    pyOut.close();
    out.layerOutput.close();
  }
}

void _runSecondFinalNormDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyNorm = mx.io.load('/tmp/tq_cur_second_final_norm.npy');
  final out = runner.debugSecondDecodeFinalNormFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
  );
  try {
    print('firstToken=${out.firstToken}');
    print(
      'localFinalNormShape=${out.finalNorm.shape} pyFinalNormShape=${pyNorm.shape}',
    );
    print('finalNormDiff=${_diffCompatible(out.finalNorm, pyNorm)}');
  } finally {
    hidden.close();
    pyNorm.close();
    out.finalNorm.close();
  }
}

void _runSecondLogitsDiff(PaddleOcrVlRunner runner, List<int> ids) {
  final meta =
      jsonDecode(File('/tmp/paddle_v15_ref/metadata.json').readAsStringSync())
          as Map<String, Object?>;
  final hidden = mx.io.load(_hiddenPath());
  final pyLogits = mx.io.load('/tmp/tq_cur_second_logits.npy');
  final normOut = runner.debugSecondDecodeFinalNormFromVisionFeatures(
    ids,
    hidden,
    gridHeight: (meta['grid_h'] as num).toInt(),
    gridWidth: (meta['grid_w'] as num).toInt(),
  );
  try {
    final logits = runner.debugLmHeadApply(normOut.finalNorm);
    try {
      print('firstToken=${normOut.firstToken}');
      print('localLogitsShape=${logits.shape} pyLogitsShape=${pyLogits.shape}');
      print('logitsDiff=${_diffCompatible(logits, pyLogits)}');
    } finally {
      logits.close();
    }
  } finally {
    hidden.close();
    pyLogits.close();
    normOut.finalNorm.close();
  }
}

void _runVisualImageDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyHidden = mx.io.load('/tmp/tq_cur_visual_cast.npy');
  try {
    final hidden = runner.encodeImageFeatures(image);
    try {
      print('visualDiff=${_diffCompatible(hidden, pyHidden)}');
    } finally {
      hidden.close();
    }
  } finally {
    image.close();
    pyHidden.close();
  }
}

void _runVisionEmbeddingsImageDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyEmb = mx.io.load('/tmp/tq_cast_vision_embeddings.npy');
  try {
    final hidden = runner.encodeImageEmbeddingsOnly(image);
    try {
      print('visionEmbeddingsDiff=${_diffCompatible(hidden, pyEmb)}');
    } finally {
      hidden.close();
    }
  } finally {
    image.close();
    pyEmb.close();
  }
}

void _runVisionPatchImageDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyPatch = mx.io.load('/tmp/tq_cast_patch_only.npy');
  try {
    final patch = runner.encodeImagePatchOnly(image);
    try {
      print('visionPatchDiff=${_diffCompatible(patch, pyPatch)}');
    } finally {
      patch.close();
    }
  } finally {
    image.close();
    pyPatch.close();
  }
}

void _runVisionPosImageDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyPos = mx.io.load('/tmp/tq_cast_pos_only.npy');
  try {
    final emb = runner.encodeImageEmbeddingsOnly(image);
    final patch = runner.encodeImagePatchOnly(image);
    try {
      final pos = emb - patch;
      try {
        print('visionPosDiff=${_diffCompatible(pos, pyPos)}');
      } finally {
        pos.close();
      }
    } finally {
      emb.close();
      patch.close();
    }
  } finally {
    image.close();
    pyPos.close();
  }
}

void _runVisionPosUsedImageDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyPos = mx.io.load('/tmp/tq_cast_pos_only.npy');
  try {
    final pos = runner.encodeVisionPositionEmbeddingUsed(image);
    try {
      print('visionPosUsedDiff=${_diffCompatible(pos, pyPos)}');
    } finally {
      pos.close();
    }
  } finally {
    image.close();
    pyPos.close();
  }
}

void _runVisionRotaryImageDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyRot = mx.io.load('/tmp/tq_cast_vision_rot.npy');
  try {
    final rot = runner.encodeVisionRotaryEmbedding(image);
    try {
      print('visionRotaryDiff=${_diffCompatible(rot, pyRot)}');
    } finally {
      rot.close();
    }
  } finally {
    image.close();
    pyRot.close();
  }
}

void _runVisionPosTableDiff(PaddleOcrVlRunner runner) {
  final pyTable = mx.io.load('/tmp/tq_cast_pos_table.npy');
  try {
    final table = runner.debugVisionPositionTable();
    try {
      print('visionPosTableDiff=${_diffCompatible(table, pyTable)}');
    } finally {
      table.close();
    }
  } finally {
    pyTable.close();
  }
}

void _runVisionLayer0ImageDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyHidden = mx.io.load('/tmp/tq_cast_vision_after_layer0.npy');
  try {
    final emb = runner.encodeImageEmbeddingsOnly(image);
    final rotary = runner.encodeVisionRotaryEmbedding(image);
    try {
      final hidden = runner.debugVisionBlockFromHidden(emb, 0, rotary);
      try {
        print('visionLayer0Diff=${_diffCompatible(hidden, pyHidden)}');
      } finally {
        hidden.close();
      }
    } finally {
      rotary.close();
      emb.close();
    }
  } finally {
    image.close();
    pyHidden.close();
  }
}

void _runVisionBlock0PartsDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyNorm1 = mx.io.load('/tmp/tq_cast_vb0_norm1.npy');
  final pyQkv = mx.io.load('/tmp/tq_cast_vb0_qkv.npy');
  final pyAttn = mx.io.load('/tmp/tq_cast_vb0_attn.npy');
  final pyOut = mx.io.load('/tmp/tq_cast_vb0_out.npy');
  try {
    final norm1 = runner.debugVisionLayerNorm1Output(image, 0);
    final qkv = runner.debugVisionQkvOutput(image, 0);
    final attn = runner.debugVisionAttentionOutput(image, 0);
    final emb = runner.encodeImageEmbeddingsOnly(image);
    final rotary = runner.encodeVisionRotaryEmbedding(image);
    final out = runner.debugVisionBlockFromHidden(emb, 0, rotary);
    try {
      print('visionNorm1Diff=${_diffCompatible(norm1, pyNorm1)}');
      print('visionQkvDiff=${_diffCompatible(qkv, pyQkv)}');
      print('visionAttnDiff=${_diffCompatible(attn, pyAttn)}');
      print('visionBlock0OutDiff=${_diffCompatible(out, pyOut)}');
    } finally {
      emb.close();
      rotary.close();
      norm1.close();
      qkv.close();
      attn.close();
      out.close();
    }
  } finally {
    image.close();
    pyNorm1.close();
    pyQkv.close();
    pyAttn.close();
    pyOut.close();
  }
}

void _runVisionAttnFromPyQkvDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyQkv = mx.io.load('/tmp/tq_cast_vb0_qkv.npy');
  final pyAttn = mx.io.load('/tmp/tq_cast_vb0_attn.npy');
  final weight = runner.debugVisionPatchWeight();
  try {
    final rotary = runner.encodeVisionRotaryEmbedding(image);
    try {
      final qkv = pyQkv.astype(weight.dtype);
      final attn = runner.debugVisionAttentionFromQkv(qkv, rotary, 0);
      try {
        print('visionAttnFromPyQkvDiff=${_diffCompatible(attn, pyAttn)}');
      } finally {
        qkv.close();
        attn.close();
      }
    } finally {
      rotary.close();
    }
  } finally {
    weight.close();
    image.close();
    pyQkv.close();
    pyAttn.close();
  }
}

void _runVisionRopeFromPyQkvDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyQkv = mx.io.load('/tmp/tq_cast_vb0_qkv.npy');
  final pyQRot = mx.io.load('/tmp/tq_cast_vb0_qrot.npy');
  final pyKRot = mx.io.load('/tmp/tq_cast_vb0_krot.npy');
  final weight = runner.debugVisionPatchWeight();
  try {
    final rotary = runner.encodeVisionRotaryEmbedding(image);
    try {
      final qkv = pyQkv.astype(weight.dtype);
      final out = runner.debugVisionRopedQkFromQkv(qkv, rotary);
      try {
        print('visionQRotDiff=${_diffCompatible(out.q, pyQRot)}');
        print('visionKRotDiff=${_diffCompatible(out.k, pyKRot)}');
      } finally {
        qkv.close();
        out.q.close();
        out.k.close();
      }
    } finally {
      rotary.close();
    }
  } finally {
    weight.close();
    image.close();
    pyQkv.close();
    pyQRot.close();
    pyKRot.close();
  }
}

void _runVisionBlock0TailDiff(PaddleOcrVlRunner runner) {
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  final pyH1 = mx.io.load('/tmp/tq_cast_vb0_h1_direct.npy');
  final pyNorm2 = mx.io.load('/tmp/tq_cast_vb0_norm2.npy');
  final pyMlp = mx.io.load('/tmp/tq_cast_vb0_mlp.npy');
  try {
    final emb = runner.encodeImageEmbeddingsOnly(image);
    final attn = runner.debugVisionAttentionOutput(image, 0);
    try {
      final h1 = emb + attn;
      try {
        print('visionH1Diff=${_diffCompatible(h1, pyH1)}');
        final norm2 = runner.debugApplyVisionLayerNorm2(h1, 0);
        try {
          final mlp = runner.debugVisionMlpApply(norm2, 0);
          try {
            print('visionNorm2Diff=${_diffCompatible(norm2, pyNorm2)}');
            print('visionMlpDiff=${_diffCompatible(mlp, pyMlp)}');
          } finally {
            mlp.close();
          }
        } finally {
          norm2.close();
        }
      } finally {
        h1.close();
      }
    } finally {
      emb.close();
      attn.close();
    }
  } finally {
    image.close();
    pyH1.close();
    pyNorm2.close();
    pyMlp.close();
  }
}

void _runVisionBlock0ApplyDiff(PaddleOcrVlRunner runner) {
  final pyH1 = mx.io.load('/tmp/tq_cast_vb0_h1_direct.npy');
  final pyNorm2 = mx.io.load('/tmp/tq_cast_vb0_norm2.npy');
  final pyMlp = mx.io.load('/tmp/tq_cast_vb0_mlp.npy');
  final weight = runner.debugVisionPatchWeight();
  try {
    final pyH1Cast = pyH1.astype(weight.dtype);
    final pyNorm2Cast = pyNorm2.astype(weight.dtype);
    final norm2 = runner.debugApplyVisionLayerNorm2(pyH1Cast, 0);
    try {
      final mlp = runner.debugVisionMlpApply(pyNorm2Cast, 0);
      try {
        print('visionNorm2ApplyDiff=${_diffCompatible(norm2, pyNorm2)}');
        print('visionMlpApplyDiff=${_diffCompatible(mlp, pyMlp)}');
      } finally {
        mlp.close();
      }
    } finally {
      pyH1Cast.close();
      pyNorm2Cast.close();
      norm2.close();
    }
  } finally {
    weight.close();
    pyH1.close();
    pyNorm2.close();
    pyMlp.close();
  }
}

void _runVisionLn2WeightDiff(PaddleOcrVlRunner runner) {
  final pyW = mx.io.load('/tmp/tq_cast_vb0_ln2_w.npy');
  final pyB = mx.io.load('/tmp/tq_cast_vb0_ln2_b.npy');
  final w = runner.debugVisionLayerNorm2Weight(0);
  final b = runner.debugVisionLayerNorm2Bias(0);
  try {
    print('visionLn2WeightDiff=${_diffCompatible(w, pyW)}');
    print('visionLn2BiasDiff=${_diffCompatible(b, pyB)}');
  } finally {
    w.close();
    b.close();
    pyW.close();
    pyB.close();
  }
}

void main(List<String> args) {
  final mode = args.isEmpty ? 'prefix' : args.first;
  final runner = PaddleOcrVlRunner.load(_snapshotPath());
  final ids = _loadPromptIds();
  final image = mx.io.load('/tmp/paddle_v15_ref/image_nhwc.npy');
  try {
    if (mode == 'prefix') {
      _runPrefix(runner, ids, image);
      return;
    }
    if (mode == 'prefix-from-hidden') {
      _runPrefixFromHidden(runner, ids);
      return;
    }
    if (mode == 'turbo-ablate-from-hidden') {
      _runTurboAblationFromHidden(runner, ids);
      return;
    }
    if (mode == 'turbo-fast-slow-second-layer-out-diff') {
      _runTurboFastSlowSecondLayerOutDiff(runner, ids);
      return;
    }
    if (mode == 'turbo-fast-slow-second-layer-cache-diff') {
      _runTurboFastSlowSecondLayerCacheDiff(runner, ids);
      return;
    }
    if (mode == 'layer-scan') {
      _runLayerScan(runner, ids, image);
      return;
    }
    if (mode == 'layer-scan-from-hidden') {
      _runLayerScanFromHidden(runner, ids);
      return;
    }
    if (mode == 'layer-cache-diff') {
      _runLayerCacheDiff(runner, ids, image);
      return;
    }
    if (mode == 'layer-cache-diff-from-hidden') {
      _runLayerCacheDiffFromHidden(runner, ids);
      return;
    }
    if (mode == 'prefill-layer-cache-diff') {
      _runPrefillLayerCacheDiff(runner, ids);
      return;
    }
    if (mode == 'prefill-quantize-diff') {
      _runPrefillQuantizeDiff(runner);
      return;
    }
    if (mode == 'dense-prefill-cache-diff') {
      _runDensePrefillCacheDiff(runner, ids);
      return;
    }
    if (mode == 'dense-prefill-cache-image-diff') {
      _runDensePrefillCacheDiffFromImage(runner, ids);
      return;
    }
    if (mode == 'inputs-diff') {
      _runInputsDiff(runner, ids);
      return;
    }
    if (mode == 'inputs-partition-diff') {
      _runInputsPartitionDiff(runner, ids);
      return;
    }
    if (mode == 'prefill-qkv-diff') {
      _runPrefillQkvDiff(runner, ids);
      return;
    }
    if (mode == 'prefill-norm1-diff') {
      _runPrefillNorm1Diff(runner, ids);
      return;
    }
    if (mode == 'prefill-layer0-attn-diff') {
      _runPrefillLayer0AttentionDiff(runner, ids);
      return;
    }
    if (mode == 'prefill-layer0-out-diff') {
      _runPrefillLayer0OutDiff(runner, ids);
      return;
    }
    if (mode == 'second-norm1-diff') {
      _runSecondNorm1Diff(runner, ids);
      return;
    }
    if (mode == 'second-qkv-diff') {
      _runSecondQkvDiff(runner, ids);
      return;
    }
    if (mode == 'second-scores-diff') {
      _runSecondScoresDiff(runner, ids);
      return;
    }
    if (mode == 'second-attn-diff') {
      _runSecondAttnDiff(runner, ids);
      return;
    }
    if (mode == 'second-out-diff') {
      _runSecondOutDiff(runner, ids);
      return;
    }
    if (mode == 'second-final-norm-diff') {
      _runSecondFinalNormDiff(runner, ids);
      return;
    }
    if (mode == 'second-logits-diff') {
      _runSecondLogitsDiff(runner, ids);
      return;
    }
    if (mode == 'visual-image-diff') {
      _runVisualImageDiff(runner);
      return;
    }
    if (mode == 'vision-embeddings-image-diff') {
      _runVisionEmbeddingsImageDiff(runner);
      return;
    }
    if (mode == 'vision-patch-image-diff') {
      _runVisionPatchImageDiff(runner);
      return;
    }
    if (mode == 'vision-pos-image-diff') {
      _runVisionPosImageDiff(runner);
      return;
    }
    if (mode == 'vision-pos-used-image-diff') {
      _runVisionPosUsedImageDiff(runner);
      return;
    }
    if (mode == 'vision-rotary-image-diff') {
      _runVisionRotaryImageDiff(runner);
      return;
    }
    if (mode == 'vision-pos-table-diff') {
      _runVisionPosTableDiff(runner);
      return;
    }
    if (mode == 'vision-layer0-image-diff') {
      _runVisionLayer0ImageDiff(runner);
      return;
    }
    if (mode == 'vision-block0-parts-diff') {
      _runVisionBlock0PartsDiff(runner);
      return;
    }
    if (mode == 'vision-attn-from-pyqkv-diff') {
      _runVisionAttnFromPyQkvDiff(runner);
      return;
    }
    if (mode == 'vision-rope-from-pyqkv-diff') {
      _runVisionRopeFromPyQkvDiff(runner);
      return;
    }
    if (mode == 'vision-block0-tail-diff') {
      _runVisionBlock0TailDiff(runner);
      return;
    }
    if (mode == 'vision-block0-apply-diff') {
      _runVisionBlock0ApplyDiff(runner);
      return;
    }
    if (mode == 'vision-ln2-weight-diff') {
      _runVisionLn2WeightDiff(runner);
      return;
    }
    stderr.writeln('unknown mode: $mode');
    exitCode = 64;
  } finally {
    image.close();
    runner.close();
  }
}
