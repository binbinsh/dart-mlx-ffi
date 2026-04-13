import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';
import 'package:dart_mlx_ffi/src/models/paddle_ocr_vl/paddle_ocr_vl.dart';

String _snapshotPath() {
  final home = Platform.environment['HOME']!;
  return '$home/.cache/huggingface/hub/'
      'models--mlx-community--PaddleOCR-VL-1.5-8bit/'
      'snapshots/37d4c85284434b6e6fd4c03f8b719b1aefaa013c';
}

String _refDir() =>
    Platform.environment['POCR_REF_DIR'] ?? '/tmp/paddle_v15_ref';

List<int> _loadPromptIds() {
  final idsArr = mx.io.load('${_refDir()}/input_ids.npy');
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

bool _truthyEnv(String name) {
  final raw = Platform.environment[name];
  if (raw == null) return false;
  final normalized = raw.toLowerCase();
  return normalized == '1' || normalized == 'true' || normalized == 'yes';
}

void _configureDebugOverrides() {
  PaddleOcrVlDebugOverrides.reset();
  final kvScheme = Platform.environment['POCR_KV_SCHEME'];
  if (!_truthyEnv('POCR_EMULATE_IOS') && kvScheme == null) return;
  PaddleOcrVlDebugOverrides.kvQuantScheme = kvScheme ?? 'turboquant';
  PaddleOcrVlDebugOverrides.decoderLayerEval = false;
  PaddleOcrVlDebugOverrides.aggressiveCacheClear = false;
  PaddleOcrVlDebugOverrides.decodeCacheEvalInterval = 8;
  PaddleOcrVlDebugOverrides.forceDecodeLmHeadFloat32 = true;
  PaddleOcrVlDebugOverrides.clearAfterCloseLogits = true;
  PaddleOcrVlDebugOverrides.decoderHiddenDetach = true;
  PaddleOcrVlDebugOverrides.decodeLogitsDetach = false;
  PaddleOcrVlDebugOverrides.disableFastSingleTokenMrope = true;
  PaddleOcrVlDebugOverrides.turboDequantKvToQueryDType = true;
}

void main() {
  _configureDebugOverrides();
  final runner = PaddleOcrVlRunner.load(_snapshotPath());
  final image = mx.io.load('${_refDir()}/image_nhwc.npy');
  try {
    final rows = runner.debugTokenMarginsFromImage(
      _loadPromptIds(),
      image,
      maxNewTokens:
          int.tryParse(Platform.environment['POCR_GAP_MAX_TOKENS'] ?? '') ?? 32,
    );
    stdout.writeln(const JsonEncoder.withIndent('  ').convert(rows));
  } finally {
    image.close();
    runner.close();
    PaddleOcrVlDebugOverrides.reset();
  }
}
