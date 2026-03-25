import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:test/test.dart';

Float32List _makeInput(int count) {
  return Float32List.fromList(
    List<double>.generate(count, (index) => (index % 97) / 97.0),
  );
}

void _expectIdentity(Float32List actual, Float32List expected, String name) {
  expect(actual.length, expected.length, reason: name);
  for (var index = 0; index < actual.length; index++) {
    expect(actual[index], closeTo(expected[index], 5e-4), reason: name);
  }
}

void main() {
  test(
    'private ane runs two mlprogram conv models under latency budget',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'private_ane/shared/benchmark/ane_private_mlprogram_bench.py',
        '--json',
        '--iters',
        '20',
        '--max-ms',
        '1.0',
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report =
          jsonDecode(result.stdout as String) as Map<String, Object?>;
      final models = List<Map<String, Object?>>.from(
        (report['models'] as List).cast<Map>(),
      );
      expect(models, hasLength(2));
      expect(models[0]['name'], 'conv-256x64');
      expect(models[1]['name'], 'conv-512x64');
      expect(models[0]['ok'], true);
      expect(models[1]['ok'], true);
      expect((models[0]['per_iter_ms'] as num) < 1.0, isTrue);
      expect((models[1]['per_iter_ms'] as num) < 1.0, isTrue);
    },
  );

  test('private ane dart api runs two mlprogram conv models', () async {
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

    final tempDir = await Directory.systemTemp.createTemp(
      'ane_private_mlprogram_models_',
    );
    addTearDown(() => tempDir.delete(recursive: true));

    final gen = await Process.run('uv', [
      'run',
      'python',
      'private_ane/shared/benchmark/ane_private_make_models.py',
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
      final channels = (spec['channels']! as num).toInt();
      final spatial = (spec['spatial']! as num).toInt();
      final inputBytes = (spec['input_bytes']! as num).toInt();
      final outputBytes = (spec['output_bytes']! as num).toInt();
      final weightOffset = (spec['weight_offset']! as num).toInt();
      final milText = await File(spec['model_mil']! as String).readAsString();
      final weightBlob = await File(spec['weight_bin']! as String).readAsBytes();
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
          final input = _makeInput(channels * spatial);
          final outputs = session.runRawFloat32([input]);
          expect(outputs, hasLength(1), reason: name);
          _expectIdentity(outputs.first, input, name);
        } finally {
          session.close();
        }
      } finally {
        model.close();
      }
    }
  });
}
