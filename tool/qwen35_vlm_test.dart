/// Dart CLI test for Qwen3.5-0.8B VLM vision pipeline.
///
/// Loads pre-processed image patches (from Python dump) and model weights,
/// runs `generateFromImage()`, and compares output with Python reference.
///
/// Usage:
///   dart run tool/qwen35_vlm_test.dart
///
/// Prerequisites:
///   - Run `/tmp/dump_vlm_test_input.py` to create test data
///   - Model snapshot at the expected path
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/src/models/qwen3_5/qwen3_5.dart';

void main() {
  // 1. Load test metadata
  const metadataPath = '/tmp/qwen35_vlm_test_input.json';
  const pixelsPath = '/tmp/qwen35_vlm_test_pixels.bin';

  if (!File(metadataPath).existsSync()) {
    stderr.writeln('Missing $metadataPath — run /tmp/dump_vlm_test_input.py');
    exitCode = 1;
    return;
  }

  final metadata =
      jsonDecode(File(metadataPath).readAsStringSync()) as Map<String, Object?>;
  final modelPath = metadata['model_path'] as String;
  final gridH = (metadata['grid_h'] as num).toInt();
  final gridW = (metadata['grid_w'] as num).toInt();
  final nPatches = (metadata['n_patches'] as num).toInt();
  final patchVecSize = (metadata['patch_vec_size'] as num).toInt();
  final promptIds = (metadata['prompt_ids'] as List)
      .cast<num>()
      .map((v) => v.toInt())
      .toList();
  final eosTokenId = (metadata['eos_token_id'] as num).toInt();
  final referenceFirst50 = (metadata['reference_first_50_generated'] as List)
      .cast<num>()
      .map((v) => v.toInt())
      .toList();
  final expectedPixelSum = (metadata['pixel_sum'] as num).toDouble();
  final expectedFirst8 = (metadata['pixel_first_8'] as List)
      .cast<num>()
      .map((v) => v.toDouble())
      .toList();

  print('=== Qwen3.5-0.8B VLM Dart Test ===');
  print('Model: $modelPath');
  print('Grid: ${gridH}x$gridW');
  print('Patches: $nPatches x $patchVecSize');
  print('Prompt tokens: ${promptIds.length}');

  // 2. Load pixel data from binary dump
  print('\nLoading pixel data...');
  final pixelFile = File(pixelsPath).readAsBytesSync();
  final pixelBytes = ByteData.sublistView(pixelFile);
  final headerN = pixelBytes.getUint32(0, Endian.little);
  final headerV = pixelBytes.getUint32(4, Endian.little);
  assert(headerN == nPatches, 'n_patches mismatch: $headerN vs $nPatches');
  assert(
    headerV == patchVecSize,
    'patch_vec_size mismatch: $headerV vs $patchVecSize',
  );

  final pixelData = Float32List(nPatches * patchVecSize);
  for (var i = 0; i < pixelData.length; i++) {
    pixelData[i] = pixelBytes.getFloat32(8 + i * 4, Endian.little);
  }

  // Verify pixel data integrity
  var pixelSum = 0.0;
  for (var i = 0; i < pixelData.length; i++) {
    pixelSum += pixelData[i];
  }
  final pixelSumDiff = (pixelSum - expectedPixelSum).abs();
  print(
    '  Pixel sum: $pixelSum (expected: $expectedPixelSum, diff: $pixelSumDiff)',
  );
  print('  First 8: ${pixelData.sublist(0, 8).toList()}');
  print('  Expected: $expectedFirst8');

  if (pixelSumDiff > 10.0) {
    stderr.writeln('ERROR: Pixel data mismatch!');
    exitCode = 1;
    return;
  }
  print('  Pixel data OK');

  // 3. Load model
  print('\nLoading model...');
  final loadWatch = Stopwatch()..start();
  final runner = Qwen3_5Runner.load(modelPath);
  loadWatch.stop();
  print('  Model loaded in ${loadWatch.elapsedMilliseconds}ms');
  print('  Has vision: ${runner.hasVision}');
  print(
    '  Vision config: depth=${runner.config.visionConfig?.depth}, '
    'hiddenSize=${runner.config.visionConfig?.hiddenSize}',
  );

  if (!runner.hasVision) {
    stderr.writeln('ERROR: No vision weights loaded!');
    runner.close();
    exitCode = 1;
    return;
  }

  // 4. Create MLX tensor from pixel data
  print('\nCreating patch tensor...');
  final patchArray = MlxArray.fromFloat32List(
    pixelData,
    shape: [nPatches, patchVecSize],
  );
  final patchCast = patchArray.astype(runner.config.computeDType);
  patchArray.close();
  print('  Patch tensor shape: ${patchCast.shape}, dtype: ${patchCast.dtype}');

  // 5. Run VLM generation
  print('\nRunning generateFromImage...');
  final genWatch = Stopwatch()..start();
  try {
    final allTokenIds = runner.generateFromImage(
      promptIds,
      patchCast,
      gridH,
      gridW,
      maxNewTokens: 512,
      eosTokenId: eosTokenId,
      onStage: (msg) => print('  [stage] $msg'),
      onToken: (tokenId) {
        // Print progress dots
        stdout.write('.');
      },
    );
    genWatch.stop();
    print('');

    // 6. Extract generated tokens
    final generatedIds = allTokenIds.sublist(promptIds.length);
    final totalMs = genWatch.elapsedMilliseconds;
    final tokPerSec = generatedIds.isEmpty
        ? 0.0
        : generatedIds.length * 1000.0 / totalMs;

    print('\n=== Results ===');
    print(
      'Total tokens: ${allTokenIds.length} '
      '(prompt: ${promptIds.length}, generated: ${generatedIds.length})',
    );
    print('Total time: ${totalMs}ms');
    print('Speed: ${tokPerSec.toStringAsFixed(1)} tok/s');

    print('\nGenerated token IDs (first 50):');
    final first50 = generatedIds.length > 50
        ? generatedIds.sublist(0, 50)
        : generatedIds;
    print('  $first50');

    print('\nPython reference (first 50):');
    print('  $referenceFirst50');

    // Compare first 50 tokens
    var matchCount = 0;
    final compareLen = first50.length < referenceFirst50.length
        ? first50.length
        : referenceFirst50.length;
    for (var i = 0; i < compareLen; i++) {
      if (first50[i] == referenceFirst50[i]) {
        matchCount++;
      } else {
        print(
          '  First mismatch at index $i: '
          'Dart=${first50[i]} vs Python=${referenceFirst50[i]}',
        );
        break;
      }
    }
    print(
      '\nToken match: $matchCount/$compareLen '
      '(${(matchCount * 100.0 / compareLen).toStringAsFixed(1)}%)',
    );

    if (generatedIds.isNotEmpty) {
      print('\nGenerated token IDs (last 10):');
      final last10 = generatedIds.length > 10
          ? generatedIds.sublist(generatedIds.length - 10)
          : generatedIds;
      print('  $last10');
    }

    // Decode using a minimal approach (just print raw IDs for now)
    print('\nAll generated token IDs:');
    print('  $generatedIds');
  } catch (e, stackTrace) {
    genWatch.stop();
    print('\nERROR during generation: $e');
    print(stackTrace);
    exitCode = 1;
  } finally {
    patchCast.close();
    runner.close();
  }
}
