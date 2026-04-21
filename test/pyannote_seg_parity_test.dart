// ignore_for_file: avoid_print
@TestOn('mac-os')
library;

import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';
import 'package:test/test.dart';

const _bundleDir = 'models/pyannote_seg';
const _fxDir = 'test/data/pyannote_seg';

MlxArray _loadRef(String name) => mx.io.load('$_fxDir/$name.npy');

/// Return `(maxAbsDiff, meanAbsDiff)` for two MLX float32 arrays.
(double, double) _diffStats(MlxArray a, MlxArray b) {
  final af = a.astype(MlxDType.MLX_FLOAT32);
  final bf = b.astype(MlxDType.MLX_FLOAT32);
  final diff = mx.subtract(af, bf);
  final absDiff = mx.abs(diff);
  final meanVal = mx.mean(absDiff);
  final total = absDiff.shape.fold<int>(1, (acc, v) => acc * v);
  final flat = absDiff.reshape(<int>[total]);
  final maxIdx = mx.argmax(flat);
  MlxRuntime.evalAll([meanVal, maxIdx, flat]);
  final idx = (maxIdx.toList().first as num).toInt();
  final maxSlice = flat.slice(start: <int>[idx], stop: <int>[idx + 1]);
  MlxRuntime.evalAll([maxSlice]);
  final maxDiff = (maxSlice.toList().first as num).toDouble();
  final meanDiff = (meanVal.toList().first as num).toDouble();
  for (final t in <MlxArray>[
    maxSlice,
    meanVal,
    maxIdx,
    flat,
    absDiff,
    diff,
    bf,
    af,
  ]) {
    t.close();
  }
  return (maxDiff, meanDiff);
}

void _assertClose(
  MlxArray actual,
  MlxArray expected,
  String label, {
  double atol = 1e-3,
  double mtol = 1e-4,
}) {
  expect(
    actual.shape,
    expected.shape,
    reason: '$label shape mismatch ${actual.shape} vs ${expected.shape}',
  );
  final (maxDiff, meanDiff) = _diffStats(actual, expected);
  print('  $label  maxDiff=$maxDiff meanDiff=$meanDiff');
  expect(maxDiff, lessThan(atol),
      reason: '$label maxDiff=$maxDiff exceeds $atol');
  expect(meanDiff, lessThan(mtol),
      reason: '$label meanDiff=$meanDiff exceeds $mtol');
}

void main() {
  test('pyannote-seg-3.0 MLX parity vs PyTorch reference', () async {
    expect(File('$_bundleDir/weights.safetensors').existsSync(), isTrue,
        reason: 'bundle missing at $_bundleDir');

    final bundle = await loadPyannoteSegBundle(_bundleDir);
    addTearDown(bundle.close);
    final runtime = PyannoteSegRuntime.fromBundle(bundle);
    addTearDown(runtime.close);

    // 1. Load reference waveform.
    final wavArray = _loadRef('reference_waveform');
    MlxRuntime.evalAll([wavArray]);
    final wav = wavArray.toFloat32List();
    wavArray.close();
    expect(wav.length, bundle.manifest.windowSamples);

    // 2. Full staged forward.
    final trace = runtime.forwardStaged(wav);
    addTearDown(trace.close);

    // 3. Load references.
    final refSinc = _loadRef('reference_sincnet');
    final refLstm = _loadRef('reference_lstm');
    final refLogits = _loadRef('reference_logits');
    final refLogP = _loadRef('reference_log_probs');
    final refPower = _loadRef('reference_powerset');
    addTearDown(() {
      for (final t in <MlxArray>[
        refSinc,
        refLstm,
        refLogits,
        refLogP,
        refPower,
      ]) {
        t.close();
      }
    });

    // 4. Layer-by-layer parity against PyTorch float32 reference. Measured
    //    headroom on 2024 Apple Silicon:
    //      sincnet   maxDiff ~1.3e-5 / meanDiff ~7e-7
    //      lstm      maxDiff ~4e-4  / meanDiff ~3e-7
    //      logits    maxDiff ~7e-5  / meanDiff ~1e-5
    //      log_probs maxDiff ~1e-4  / meanDiff ~1e-5
    //      powerset  maxDiff ~1.5e-6/ meanDiff ~1.3e-7
    _assertClose(trace.sincnet, refSinc, 'sincnet',
        atol: 5e-4, mtol: 5e-6);
    _assertClose(trace.lstm, refLstm, 'lstm',
        atol: 2e-3, mtol: 5e-6);
    _assertClose(trace.logits, refLogits, 'logits',
        atol: 1e-3, mtol: 1e-4);
    _assertClose(trace.logProbs, refLogP, 'log_probs',
        atol: 1e-3, mtol: 1e-4);
    _assertClose(trace.powerset, refPower, 'powerset',
        atol: 1e-4, mtol: 5e-6);
  });
}
