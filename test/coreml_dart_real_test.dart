import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:test/test.dart';

int _argmax(List<double> values) {
  var bestIndex = 0;
  var bestValue = values.first;
  for (var index = 1; index < values.length; index++) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }
  return bestIndex;
}

void main() {
  test(
    'dart coreml wrapper runs two real models with ane compute units',
    () async {
      final dir = await Directory.systemTemp.createTemp('coreml_ane_models_');
      addTearDown(() => dir.delete(recursive: true));

      final make = await Process.run('uv', [
        'run',
        'python',
        'tool/coreml_ane_make_models.py',
        '--out-dir',
        dir.path,
      ]);
      expect(make.exitCode, 0, reason: make.stderr.toString());

      final metadata =
          jsonDecode(await File('${dir.path}/metadata.json').readAsString())
              as Map<String, Object?>;
      final models = List<Map<String, Object?>>.from(
        (metadata['models'] as List).cast<Map>(),
      );
      expect(models, hasLength(2));

      for (final model in models) {
        final coreml = MlxCoreMlModel.loadSingleIo(
          path: model['path'] as String,
          inputName: model['input_name'] as String,
          outputName: model['output_name'] as String,
          inputShape: List<int>.from(model['input_shape'] as List),
          outputCount: model['output_count'] as int,
        );
        addTearDown(coreml.close);

        final samples = List<Map<String, Object?>>.from(
          (model['samples'] as List).cast<Map>(),
        );
        for (final sample in samples) {
          final input = Float32List.fromList(
            List<double>.from(
              sample['input5'] as List,
            ).map((e) => e.toDouble()).toList(),
          );
          final expected = List<double>.from(
            sample['logits'] as List,
          ).map((e) => e.toDouble()).toList();
          final output = coreml.predict(input);
          expect(
            output.length,
            expected.length,
            reason: model['name'] as String,
          );
          expect(
            _argmax(output),
            sample['label'] as int,
            reason: model['name'] as String,
          );
          for (var index = 0; index < output.length; index++) {
            expect(
              output[index],
              closeTo(expected[index], 0.5),
              reason: model['name'] as String,
            );
          }
          expect(coreml.lastPredictTimeNs(), greaterThanOrEqualTo(0));
        }
      }
    },
  );
}
