// ignore_for_file: avoid_print
@TestOn('mac-os')
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';
import 'package:test/test.dart';

const _bundleDir = 'models/ecapa_tdnn';
const _fxDir = 'test/data/speaker_embedding';

MlxArray _loadRef(String name) => mx.io.load('$_fxDir/$name.npy');

/// Max-abs and mean-abs difference between two MLX float32 arrays. The arrays
/// are left untouched; all temporaries are closed before returning.
(double, double) _diffStats(MlxArray a, MlxArray b) {
  final af = a.astype(MlxDType.MLX_FLOAT32);
  final bf = b.astype(MlxDType.MLX_FLOAT32);
  final diff = mx.subtract(af, bf);
  final absDiff = mx.abs(diff);
  final meanVal = mx.mean(absDiff);
  final total = absDiff.shape.fold<int>(1, (acc, v) => acc * v);
  final flat = absDiff.reshape([total]);
  final maxIdx = mx.argmax(flat);
  MlxRuntime.evalAll([meanVal, maxIdx, flat]);
  final idx = (maxIdx.toList().first as num).toInt();
  final maxSlice = flat.slice(start: [idx], stop: [idx + 1]);
  MlxRuntime.evalAll([maxSlice]);
  final maxDiff = (maxSlice.toList().first as num).toDouble();
  final meanDiff = (meanVal.toList().first as num).toDouble();
  for (final t in [maxSlice, meanVal, maxIdx, flat, absDiff, diff, bf, af]) {
    t.close();
  }
  return (maxDiff, meanDiff);
}

/// Assert that [actual] matches [expected] within [atol] after an optional
/// axis permutation applied to [actual] so it lines up with the channel-first
/// reference shape.
void _assertCloseWithPerm(
  MlxArray actual,
  MlxArray expected,
  String label,
  List<int>? perm, {
  double atol = 5e-3,
  double mtol = 5e-4,
}) {
  MlxArray target = actual;
  MlxArray? permuted;
  if (perm != null) {
    permuted = actual.transposeAxes(perm);
    target = permuted;
  }
  try {
    expect(
      target.shape,
      expected.shape,
      reason: '$label shape mismatch ${target.shape} vs ${expected.shape}',
    );
    final (maxDiff, meanDiff) = _diffStats(target, expected);
    print('  $label  maxDiff=$maxDiff meanDiff=$meanDiff');
    expect(maxDiff, lessThan(atol),
        reason: '$label maxDiff=$maxDiff exceeds $atol');
    expect(meanDiff, lessThan(mtol),
        reason: '$label meanDiff=$meanDiff exceeds $mtol');
  } finally {
    permuted?.close();
  }
}

double _cosine(List<double> a, List<double> b) {
  var dot = 0.0;
  var aSq = 0.0;
  var bSq = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aSq += a[i] * a[i];
    bSq += b[i] * b[i];
  }
  return dot / (math.sqrt(aSq) * math.sqrt(bSq));
}

void main() {
  test('ECAPA-TDNN parity vs SpeechBrain reference', () async {
    final bundle = await loadEcapaBundle(_bundleDir);
    addTearDown(bundle.close);
    final runtime = EcapaRuntime(bundle);

    // 1. Load the reference waveform and run the full forward with hooks.
    final wavArray = _loadRef('reference_waveform');
    MlxRuntime.evalAll([wavArray]);
    final wav = wavArray.toFloat32List();
    wavArray.close();

    final trace = runtime.forwardWithFixtureHooks(wav);
    addTearDown(trace.close);

    // ---- Frontend parity --------------------------------------------------
    final refFbankRaw = _loadRef('reference_fbank_raw'); // (T, 80)
    final refFbankNorm = _loadRef('reference_fbank'); // (T, 80)
    addTearDown(() {
      refFbankRaw.close();
      refFbankNorm.close();
    });
    _assertCloseWithPerm(
      trace.fbankRaw,
      refFbankRaw,
      'fbank_raw',
      null,
      atol: 2e-2,
      mtol: 2e-3,
    );
    _assertCloseWithPerm(
      trace.fbankNorm,
      refFbankNorm,
      'fbank_norm',
      null,
      atol: 2e-2,
      mtol: 2e-3,
    );

    // ---- Backbone blocks (NTC -> NCL via transpose [0,2,1]) --------------
    final refBlock0 = _loadRef('reference_block_0');
    final refBlock1 = _loadRef('reference_block_1');
    final refBlock2 = _loadRef('reference_block_2');
    final refBlock3 = _loadRef('reference_block_3');
    final refPreMfa = _loadRef('reference_pre_mfa_concat');
    final refMfa = _loadRef('reference_mfa');
    final refAsp = _loadRef('reference_asp');
    final refAspBn = _loadRef('reference_asp_bn');
    final refFc = _loadRef('reference_fc');
    final refEmb = _loadRef('reference_embedding');
    addTearDown(() {
      for (final t in [
        refBlock0,
        refBlock1,
        refBlock2,
        refBlock3,
        refPreMfa,
        refMfa,
        refAsp,
        refAspBn,
        refFc,
        refEmb,
      ]) {
        t.close();
      }
    });

    _assertCloseWithPerm(trace.block0, refBlock0, 'block_0', [0, 2, 1]);
    _assertCloseWithPerm(trace.block1, refBlock1, 'block_1', [0, 2, 1]);
    _assertCloseWithPerm(trace.block2, refBlock2, 'block_2', [0, 2, 1]);
    _assertCloseWithPerm(trace.block3, refBlock3, 'block_3', [0, 2, 1]);
    _assertCloseWithPerm(
        trace.preMfaConcat, refPreMfa, 'pre_mfa_concat', [0, 2, 1]);
    _assertCloseWithPerm(trace.mfa, refMfa, 'mfa', [0, 2, 1]);
    _assertCloseWithPerm(trace.asp, refAsp, 'asp', [0, 2, 1]);
    _assertCloseWithPerm(trace.aspBn, refAspBn, 'asp_bn', [0, 2, 1]);
    _assertCloseWithPerm(trace.fc, refFc, 'fc', [0, 2, 1]);

    // ---- Final embedding --------------------------------------------------
    final embList = (trace.embedding.astype(MlxDType.MLX_FLOAT32).toList())
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    final refEmbList = (refEmb.astype(MlxDType.MLX_FLOAT32).toList())
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    expect(embList.length, refEmbList.length);
    final cosine = _cosine(embList, refEmbList);
    print('  embedding cosine=$cosine');
    expect(cosine, greaterThan(0.9999),
        reason: 'embedding cosine=$cosine vs SpeechBrain reference');

    // Quick sanity: the file fixtures exist on disk.
    expect(File('$_bundleDir/weights.safetensors').existsSync(), isTrue);
  });
}
