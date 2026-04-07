/// Dart CLI test for Qwen3.5-0.8B VLM — debug intermediate values.
///
/// Compares Dart vision encoder intermediate outputs with Python reference
/// at each stage: patch_embed, patch_plus_pos, after_blocks, vision_output.
///
/// Usage:
///   dart run tool/qwen35_vlm_debug.dart
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/src/models/qwen3_5/qwen3_5.dart';

void main() {
  const metadataPath = '/tmp/qwen35_vlm_test_input.json';
  const pixelsPath = '/tmp/qwen35_vlm_test_pixels.bin';

  final metadata =
      jsonDecode(File(metadataPath).readAsStringSync()) as Map<String, Object?>;
  final modelPath = metadata['model_path'] as String;
  final gridH = (metadata['grid_h'] as num).toInt();
  final gridW = (metadata['grid_w'] as num).toInt();
  final nPatches = (metadata['n_patches'] as num).toInt();
  final patchVecSize = (metadata['patch_vec_size'] as num).toInt();

  stdout.writeln('Grid: ${gridH}x$gridW, Patches: $nPatches x $patchVecSize');

  // Load pixel data
  final pixelFile = File(pixelsPath).readAsBytesSync();
  final pixelBytes = ByteData.sublistView(pixelFile);
  final pixelData = Float32List(nPatches * patchVecSize);
  for (var i = 0; i < pixelData.length; i++) {
    pixelData[i] = pixelBytes.getFloat32(8 + i * 4, Endian.little);
  }

  // Load model
  stdout.writeln('Loading model...');
  final runner = Qwen3_5Runner.load(modelPath);
  final vCfg = runner.config.visionConfig!;
  stdout.writeln('Model loaded. Vision depth=${vCfg.depth}');

  // Prompt and reference data
  final promptIds = (metadata['prompt_ids'] as List)
      .cast<num>()
      .map((v) => v.toInt())
      .toList();
  final eosTokenId = (metadata['eos_token_id'] as num).toInt();
  final referenceFirst50 = (metadata['reference_first_50_generated'] as List)
      .cast<num>()
      .map((v) => v.toInt())
      .toList();

  // Python reference files for intermediate comparison
  const pythonRefs = {
    'patch_embed': '/tmp/python_patch_embed.bin',
    'after_blocks': '/tmp/python_after_blocks.bin',
    'vision_output': '/tmp/python_vision_output.bin',
  };

  // Create patch tensor
  final patchArray = MlxArray.fromFloat32List(
    pixelData,
    shape: [nPatches, patchVecSize],
  );
  final patchCast = patchArray.astype(runner.config.computeDType);
  patchArray.close();

  // =====================================================================
  // Run generateFromImage with intermediate dumps
  // =====================================================================
  stdout.writeln('\n=== Running generateFromImage with intermediate dumps ===');
  final watch = Stopwatch()..start();

  try {
    final allTokenIds = runner.generateFromImage(
      promptIds,
      patchCast,
      gridH,
      gridW,
      maxNewTokens: 60,
      eosTokenId: eosTokenId,
      onStage: (msg) => stdout.writeln('  [stage] $msg'),
      onDumpIntermediate: (stage, value) {
        stdout.writeln('\n--- Intermediate: $stage ---');
        stdout.writeln('  shape: ${value.shape}, dtype: ${value.dtype}');

        final values = _toFloat64List(value);
        _printStats(stage, values);

        // Show first 8 values
        if (values.length >= 8) {
          stdout.writeln('  row 0[:8]: ${values.sublist(0, 8)}');
        }

        // Compare with Python reference
        final refPath = pythonRefs[stage];
        if (refPath != null) {
          _compareBin(refPath, values, stage);
        }
      },
    );
    watch.stop();

    // Token comparison
    final generatedIds = allTokenIds.sublist(promptIds.length);
    stdout.writeln(
      '\n=== Token Generation Results ==='
      '\nGenerated ${generatedIds.length} tokens in '
      '${watch.elapsedMilliseconds}ms',
    );

    final first50 = generatedIds.length > 50
        ? generatedIds.sublist(0, 50)
        : generatedIds;
    stdout.writeln('Dart first ${first50.length}: $first50');
    stdout.writeln('Python first 50: $referenceFirst50');

    var matchCount = 0;
    final compareLen = math.min(first50.length, referenceFirst50.length);
    for (var i = 0; i < compareLen; i++) {
      if (first50[i] == referenceFirst50[i]) {
        matchCount++;
      } else {
        stdout.writeln(
          '  FIRST MISMATCH at index $i: '
          'Dart=${first50[i]} vs Python=${referenceFirst50[i]}',
        );
        break;
      }
    }
    stdout.writeln('Token match: $matchCount/$compareLen');
  } catch (e, st) {
    watch.stop();
    stdout.writeln('ERROR: $e');
    stdout.writeln(st);
  } finally {
    patchCast.close();
  }

  // =====================================================================
  // Also compare position embedding and rotary (pure Dart, no lib access)
  // =====================================================================
  stdout.writeln('\n=== Position Embedding (replicated) ===');
  final posEmbedWeight = runner.tensors['vision_tower.pos_embed.weight']!;
  final posValues = _interpolateVisionPosEmbed(
    posEmbedWeight,
    gridH,
    gridW,
    vCfg,
  );
  _printStats('posEmbed', posValues);
  _compareBin('/tmp/python_pos_embed.bin', posValues, 'pos_embed');

  stdout.writeln('\n=== Vision Rotary PosEmb (replicated) ===');
  final rotValues = _buildVisionRotaryPosEmb(gridH, gridW, vCfg);
  _compareBin('/tmp/python_rotary_pos_emb.bin', rotValues, 'rotary_pos_emb');

  runner.close();
  stdout.writeln('\n=== Done ===');
}

// =========================================================================
// Replicate position embedding interpolation (pure Dart math)
// =========================================================================
List<double> _interpolateVisionPosEmbed(
  MlxArray posEmbedWeight,
  int gridH,
  int gridW,
  Qwen3_5VisionConfig vCfg,
) {
  final hiddenSize = vCfg.hiddenSize;
  final m = vCfg.spatialMergeSize;
  final base = posEmbedWeight.toFloat32List();
  final baseGrid = math.sqrt(base.length / hiddenSize).round();

  final rowCoords = List<double>.generate(
    gridH,
    (y) => gridH == 1 ? 0.0 : y * (baseGrid - 1) / (gridH - 1),
  );
  final colCoords = List<double>.generate(
    gridW,
    (x) => gridW == 1 ? 0.0 : x * (baseGrid - 1) / (gridW - 1),
  );

  final rowFloor = List<int>.filled(gridH, 0);
  final rowCeil = List<int>.filled(gridH, 0);
  final rowFrac = List<double>.filled(gridH, 0);
  for (var y = 0; y < gridH; y++) {
    final ry = rowCoords[y];
    rowFloor[y] = ry.floor().clamp(0, baseGrid - 1);
    rowCeil[y] = (rowFloor[y] + 1).clamp(0, baseGrid - 1);
    rowFrac[y] = ry - rowFloor[y];
  }

  final colFloor = List<int>.filled(gridW, 0);
  final colCeil = List<int>.filled(gridW, 0);
  final colFrac = List<double>.filled(gridW, 0);
  for (var x = 0; x < gridW; x++) {
    final rx = colCoords[x];
    colFloor[x] = rx.floor().clamp(0, baseGrid - 1);
    colCeil[x] = (colFloor[x] + 1).clamp(0, baseGrid - 1);
    colFrac[x] = rx - colFloor[x];
  }

  final rowMajor = Float64List(gridH * gridW * hiddenSize);
  for (var y = 0; y < gridH; y++) {
    final rf = rowFrac[y];
    final oneMinusRf = 1.0 - rf;
    for (var x = 0; x < gridW; x++) {
      final cf = colFrac[x];
      final oneMinusCf = 1.0 - cf;
      final tl = (rowFloor[y] * baseGrid + colFloor[x]) * hiddenSize;
      final tr = (rowFloor[y] * baseGrid + colCeil[x]) * hiddenSize;
      final bl = (rowCeil[y] * baseGrid + colFloor[x]) * hiddenSize;
      final br = (rowCeil[y] * baseGrid + colCeil[x]) * hiddenSize;
      final target = (y * gridW + x) * hiddenSize;
      for (var c = 0; c < hiddenSize; c++) {
        rowMajor[target + c] =
            (oneMinusRf * oneMinusCf * base[tl + c]) +
            (oneMinusRf * cf * base[tr + c]) +
            (rf * oneMinusCf * base[bl + c]) +
            (rf * cf * base[br + c]);
      }
    }
  }

  final mergedH = gridH ~/ m;
  final mergedW = gridW ~/ m;
  final result = List<double>.filled(gridH * gridW * hiddenSize, 0.0);
  var dstIdx = 0;
  for (var mh = 0; mh < mergedH; mh++) {
    for (var mw = 0; mw < mergedW; mw++) {
      for (var sh = 0; sh < m; sh++) {
        for (var sw = 0; sw < m; sw++) {
          final row = mh * m + sh;
          final col = mw * m + sw;
          final srcOff = (row * gridW + col) * hiddenSize;
          final dstOff = dstIdx * hiddenSize;
          for (var c = 0; c < hiddenSize; c++) {
            result[dstOff + c] = rowMajor[srcOff + c];
          }
          dstIdx++;
        }
      }
    }
  }

  return result;
}

// =========================================================================
// Replicate rotary pos emb (pure Dart math)
// =========================================================================
List<double> _buildVisionRotaryPosEmb(
  int gridH,
  int gridW,
  Qwen3_5VisionConfig vCfg,
) {
  final seqLen = gridH * gridW;
  final m = vCfg.spatialMergeSize;
  final rotaryDim = vCfg.headDim ~/ 2;
  final invFreqCount = rotaryDim ~/ 2;
  final mergedH = gridH ~/ m;
  final mergedW = gridW ~/ m;

  final invFreq = Float64List(invFreqCount);
  for (var i = 0; i < invFreqCount; i++) {
    final exponent = (2 * i) / rotaryDim;
    invFreq[i] = 1.0 / math.pow(10000.0, exponent);
  }

  final freqs = List<double>.filled(seqLen * rotaryDim, 0.0);
  var idx = 0;
  for (var mh = 0; mh < mergedH; mh++) {
    for (var mw = 0; mw < mergedW; mw++) {
      for (var sh = 0; sh < m; sh++) {
        for (var sw = 0; sw < m; sw++) {
          final row = mh * m + sh;
          final col = mw * m + sw;
          final target = idx * rotaryDim;
          for (var i = 0; i < invFreqCount; i++) {
            freqs[target + i] = row * invFreq[i];
            freqs[target + invFreqCount + i] = col * invFreq[i];
          }
          idx++;
        }
      }
    }
  }

  return freqs;
}

// =========================================================================
// Helpers
// =========================================================================

List<double> _toFloat64List(MlxArray arr) {
  final f32 = arr.astype(MlxDType.MLX_FLOAT32);
  final flat = f32.reshape([f32.size]).toList().cast<double>();
  f32.close();
  return flat;
}

void _printStats(String label, List<double> values) {
  var sum = 0.0;
  var absMax = 0.0;
  for (final v in values) {
    sum += v;
    final a = v.abs();
    if (a > absMax) absMax = a;
  }
  final mean = sum / values.length;
  stdout.writeln('  [$label] mean=$mean, sum=$sum, abs_max=$absMax');
}

void _compareBin(String path, List<double> dartValues, String label) {
  final file = File(path);
  if (!file.existsSync()) {
    stdout.writeln('  (no Python reference at $path)');
    return;
  }
  final bytes = ByteData.sublistView(file.readAsBytesSync());
  final n = bytes.getUint32(0, Endian.little);
  final d = bytes.getUint32(4, Endian.little);
  final totalPython = n * d;
  final totalDart = dartValues.length;

  if (totalPython != totalDart) {
    stdout.writeln('  SIZE MISMATCH: Python=$totalPython, Dart=$totalDart');
    return;
  }

  var maxAbsDiff = 0.0;
  var sumDiff = 0.0;
  var maxAbsDiffIdx = 0;
  var firstBigDiffIdx = -1;
  for (var i = 0; i < totalPython; i++) {
    final pyVal = bytes.getFloat32(8 + i * 4, Endian.little);
    final dartVal = dartValues[i];
    final diff = (pyVal - dartVal).abs();
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
      maxAbsDiffIdx = i;
    }
    sumDiff += diff;
    if (diff > 0.01 && firstBigDiffIdx < 0) {
      firstBigDiffIdx = i;
    }
  }
  final meanDiff = sumDiff / totalPython;

  stdout.writeln(
    '  [$label] max_abs_diff=$maxAbsDiff (at idx $maxAbsDiffIdx), '
    'mean_diff=$meanDiff',
  );

  if (firstBigDiffIdx >= 0) {
    final pyVal = bytes.getFloat32(8 + firstBigDiffIdx * 4, Endian.little);
    final row = firstBigDiffIdx ~/ d;
    final col = firstBigDiffIdx % d;
    stdout.writeln(
      '  First >0.01 diff at idx=$firstBigDiffIdx (row=$row, col=$col): '
      'Python=$pyVal, Dart=${dartValues[firstBigDiffIdx]}, '
      'diff=${(pyVal - dartValues[firstBigDiffIdx]).abs()}',
    );
  } else {
    stdout.writeln('  All values within 0.01 tolerance ✓');
  }

  // Also report percentage of values with >0.1 diff
  var bigDiffCount = 0;
  for (var i = 0; i < totalPython; i++) {
    final pyVal = bytes.getFloat32(8 + i * 4, Endian.little);
    final diff = (pyVal - dartValues[i]).abs();
    if (diff > 0.1) bigDiffCount++;
  }
  if (bigDiffCount > 0) {
    stdout.writeln(
      '  Values with >0.1 diff: $bigDiffCount / $totalPython '
      '(${(bigDiffCount * 100.0 / totalPython).toStringAsFixed(2)}%)',
    );
  }
}
