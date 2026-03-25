import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:test/test.dart';

void _expectClose(Float32List actual, Float32List expected, String name) {
  expect(actual.length, expected.length, reason: name);
  for (var index = 0; index < actual.length; index++) {
    expect(actual[index], closeTo(expected[index], 3e-2), reason: name);
  }
}

void main() {
  test('qwen3-asr private ane benchmark stays within latency budgets', () async {
    final result = await Process.run('uv', [
      'run',
      'python',
      'private_ane/models/qwen3_asr/python/private_bench.py',
      '--json',
      '--iters',
      '10',
    ]);

    expect(result.exitCode, 0, reason: result.stderr.toString());
    final report =
        jsonDecode(result.stdout as String) as Map<String, Object?>;
    final models = List<Map<String, Object?>>.from(
      (report['models'] as List).cast<Map>(),
    );
    expect(models, hasLength(2));
    expect(models[0]['name'], 'qwen3-asr-1.7b-audio-mlp-l32');
    expect(models[1]['name'], 'qwen3-asr-1.7b-text-ffn-l32');
    for (final model in models) {
      expect(model['ok'], true, reason: model['name'] as String);
      expect(
        (model['per_iter_ms'] as num) < (model['max_ms'] as num),
        isTrue,
        reason: model['name'] as String,
      );
    }
  });

  test('qwen3-asr private ane dart path matches cpu references', () async {
    if (!mx.anePrivate.isEnabled()) {
      expect(
        () => mx.anePrivate.modelFromMil('program(1.3) {}'),
        throwsA(isA<MlxException>()),
      );
      return;
    }

    final probe = mx.anePrivate.probe();
    if (!probe.frameworkLoaded || !probe.supportsBasicEval) {
      return;
    }

    final tempDir = await Directory.systemTemp.createTemp('qwen3_asr_private_');
    addTearDown(() => tempDir.delete(recursive: true));

    final gen = await Process.run('uv', [
      'run',
      'python',
      'private_ane/models/qwen3_asr/export/make_private_blocks.py',
      '--out-dir',
      tempDir.path,
    ]);
    expect(gen.exitCode, 0, reason: '${gen.stderr}\n${gen.stdout}');

    final metadata = jsonDecode(
      await File('${tempDir.path}/metadata.json').readAsString(),
    ) as Map<String, Object?>;
    final models = List<Map<String, Object?>>.from(
      (metadata['models'] as List).cast<Map>(),
    );
    expect(models, hasLength(2));

    for (final spec in models) {
      final name = spec['name']! as String;
      final inputBytes = (spec['input_bytes']! as num).toInt();
      final outputBytes = (spec['output_bytes']! as num).toInt();
      final weightOffset = (spec['weight_offset']! as num).toInt();
      final milText = await File(spec['model_mil']! as String).readAsString();
      final weightBlob = await File(spec['weight_bin']! as String).readAsBytes();
      final input = mx.anePrivate.decodeRawFloat32Bytes(
        await File(spec['input_f32']! as String).readAsBytes(),
      );
      final expected = mx.anePrivate.decodeRawFloat32Bytes(
        await File(spec['expected_f32']! as String).readAsBytes(),
      );

      final model = mx.anePrivate.modelFromMilWithOffsets(
        milText,
        weights: [
          (
            path: '@model_path/weights/weight.bin',
            data: Uint8List.fromList(weightBlob),
            offset: weightOffset,
          ),
        ],
      );

      try {
        model.compile();
        model.load();

        final session = model.createSession(
          inputByteSizes: [inputBytes],
          outputByteSizes: [outputBytes],
        );
        try {
          final outputs = session.runRawFloat32([input]);
          expect(outputs, hasLength(1), reason: name);
          _expectClose(outputs.first, expected, name);
        } finally {
          session.close();
        }
      } finally {
        model.close();
      }
    }
  });
}
