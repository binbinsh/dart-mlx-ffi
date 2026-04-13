@TestOn('mac-os')
library;

import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/models.dart';

void main() {
  // -----------------------------------------------------------------------
  // ModelSpec
  // -----------------------------------------------------------------------

  group('ModelSpec', () {
    test('stores all constructor fields', () {
      final spec = ModelSpec(
        id: 'test_model',
        family: 'TestFamily',
        modalities: [ModelModality.textGeneration, ModelModality.embedding],
        description: 'A test model',
        version: '1.0',
        requiredFiles: ['config.json', 'weights.safetensors'],
        optionalFiles: ['tokenizer.json'],
        requiredTags: ['mlx', 'text-generation'],
        sizeHint: 1024 * 1024,
        metadata: {'key': 'value'},
      );

      expect(spec.id, 'test_model');
      expect(spec.family, 'TestFamily');
      expect(spec.modalities, hasLength(2));
      expect(spec.description, 'A test model');
      expect(spec.version, '1.0');
      expect(spec.requiredFiles, ['config.json', 'weights.safetensors']);
      expect(spec.optionalFiles, ['tokenizer.json']);
      expect(spec.requiredTags, ['mlx', 'text-generation']);
      expect(spec.sizeHint, 1024 * 1024);
      expect(spec.metadata, {'key': 'value'});
    });

    test('has sensible defaults', () {
      final spec = ModelSpec(
        id: 'minimal',
        family: 'Minimal',
        modalities: [ModelModality.textGeneration],
      );

      expect(spec.description, '');
      expect(spec.version, isNull);
      expect(spec.requiredFiles, ['config.json']);
      expect(spec.optionalFiles, isEmpty);
      expect(spec.requiredTags, isEmpty);
      expect(spec.sizeHint, isNull);
      expect(spec.metadata, isEmpty);
    });

    test('toJson round-trips the essential fields', () {
      final spec = ModelSpec(
        id: 'rt',
        family: 'RoundTrip',
        modalities: [ModelModality.speechToText],
        description: 'desc',
        version: '2.0',
        requiredFiles: ['config.json'],
        requiredTags: ['mlx'],
        sizeHint: 42,
        metadata: {'extra': true},
      );

      final json = spec.toJson();
      expect(json['id'], 'rt');
      expect(json['family'], 'RoundTrip');
      expect(json['modalities'], ['speechToText']);
      expect(json['description'], 'desc');
      expect(json['version'], '2.0');
      expect(json['sizeHint'], 42);
    });

    test('toJson omits empty optional fields', () {
      final spec = ModelSpec(
        id: 'minimal',
        family: 'Min',
        modalities: [ModelModality.textGeneration],
      );
      final json = spec.toJson();
      expect(json.containsKey('description'), isFalse);
      expect(json.containsKey('version'), isFalse);
      expect(json.containsKey('optionalFiles'), isFalse);
      expect(json.containsKey('requiredTags'), isFalse);
      expect(json.containsKey('sizeHint'), isFalse);
      expect(json.containsKey('metadata'), isFalse);
    });

    test('toString includes id and family', () {
      final spec = ModelSpec(
        id: 'foo',
        family: 'Bar',
        modalities: [ModelModality.textGeneration],
      );
      expect(spec.toString(), contains('foo'));
      expect(spec.toString(), contains('Bar'));
    });
  });

  // -----------------------------------------------------------------------
  // SnapshotValidator
  // -----------------------------------------------------------------------

  group('SnapshotValidator', () {
    late Directory tmpDir;

    setUp(() {
      tmpDir = Directory.systemTemp.createTempSync('snapshot_test_');
    });

    tearDown(() {
      tmpDir.deleteSync(recursive: true);
    });

    test('passes for a valid snapshot', () {
      // Create minimal valid snapshot.
      File(
        '${tmpDir.path}/config.json',
      ).writeAsStringSync(jsonEncode({'model_type': 'test'}));
      File('${tmpDir.path}/model.safetensors').writeAsStringSync('fake');

      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      final result = const SnapshotValidator().validate(tmpDir.path, spec);
      expect(result.isValid, isTrue);
      expect(result.summary, 'All checks passed.');
    });

    test('fails when required file is missing', () {
      File('${tmpDir.path}/model.safetensors').writeAsStringSync('fake');
      // No config.json.

      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      final result = const SnapshotValidator().validate(tmpDir.path, spec);
      expect(result.isValid, isFalse);
      expect(result.summary, contains('config.json'));
    });

    test('fails when no safetensors file exists', () {
      File(
        '${tmpDir.path}/config.json',
      ).writeAsStringSync(jsonEncode({'model_type': 'test'}));

      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      final result = const SnapshotValidator().validate(tmpDir.path, spec);
      expect(result.isValid, isFalse);
      expect(result.summary, contains('safetensors'));
    });

    test('fails for invalid JSON in config.json', () {
      File('${tmpDir.path}/config.json').writeAsStringSync('not json{{{');
      File('${tmpDir.path}/model.safetensors').writeAsStringSync('fake');

      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      final result = const SnapshotValidator().validate(tmpDir.path, spec);
      expect(result.isValid, isFalse);
      expect(result.summary, contains('config_json_valid'));
    });

    test('validates quantization metadata', () {
      File('${tmpDir.path}/config.json').writeAsStringSync(
        jsonEncode({
          'model_type': 'test',
          'quantization': {'bits': 4, 'group_size': 64},
        }),
      );
      File('${tmpDir.path}/model.safetensors').writeAsStringSync('fake');

      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      final result = const SnapshotValidator().validate(tmpDir.path, spec);
      expect(result.isValid, isTrue);
    });

    test('fails for bad quantization metadata', () {
      File('${tmpDir.path}/config.json').writeAsStringSync(
        jsonEncode({
          'model_type': 'test',
          'quantization': {'bits': 4}, // missing group_size
        }),
      );
      File('${tmpDir.path}/model.safetensors').writeAsStringSync('fake');

      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      final result = const SnapshotValidator().validate(tmpDir.path, spec);
      expect(result.isValid, isFalse);
      expect(result.summary, contains('quantization_fields'));
    });

    test('detectQuantScheme returns none for no quantization', () {
      File(
        '${tmpDir.path}/config.json',
      ).writeAsStringSync(jsonEncode({'model_type': 'test'}));
      expect(
        const SnapshotValidator().detectQuantScheme(tmpDir.path),
        QuantScheme.none,
      );
    });

    test('detectQuantScheme returns mlxAffine for affine mode', () {
      File('${tmpDir.path}/config.json').writeAsStringSync(
        jsonEncode({
          'quantization': {'mode': 'affine', 'bits': 4, 'group_size': 64},
        }),
      );
      expect(
        const SnapshotValidator().detectQuantScheme(tmpDir.path),
        QuantScheme.mlxAffine,
      );
    });

    test('detectQuantScheme returns custom for non-affine mode', () {
      File('${tmpDir.path}/config.json').writeAsStringSync(
        jsonEncode({
          'quantization': {'mode': 'gptq', 'bits': 4, 'group_size': 64},
        }),
      );
      expect(
        const SnapshotValidator().detectQuantScheme(tmpDir.path),
        QuantScheme.custom,
      );
    });
  });

  // -----------------------------------------------------------------------
  // SnapshotLocator
  // -----------------------------------------------------------------------

  group('SnapshotLocator', () {
    late Directory tmpDir;

    setUp(() {
      tmpDir = Directory.systemTemp.createTempSync('locator_test_');
    });

    tearDown(() {
      tmpDir.deleteSync(recursive: true);
    });

    test('finds a matching snapshot', () {
      // Create a valid snapshot directory.
      final snapDir = Directory('${tmpDir.path}/my_model')..createSync();
      File(
        '${snapDir.path}/config.json',
      ).writeAsStringSync(jsonEncode({'model_type': 'test'}));
      File('${snapDir.path}/model.safetensors').writeAsStringSync('fake');

      final locator = SnapshotLocator(searchPaths: [tmpDir.path]);
      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );

      final result = locator.locate(spec);
      expect(result, isNotNull);
      expect(result, contains('my_model'));
    });

    test('returns null when no snapshot matches', () {
      final locator = SnapshotLocator(searchPaths: [tmpDir.path]);
      final spec = ModelSpec(
        id: 'nonexistent',
        family: 'No',
        modalities: [ModelModality.textGeneration],
      );
      expect(locator.locate(spec), isNull);
    });

    test('locateAll finds multiple matches', () {
      for (final name in ['snap_a', 'snap_b']) {
        final d = Directory('${tmpDir.path}/$name')..createSync();
        File(
          '${d.path}/config.json',
        ).writeAsStringSync(jsonEncode({'model_type': 'test'}));
        File('${d.path}/model.safetensors').writeAsStringSync('fake');
      }

      final locator = SnapshotLocator(searchPaths: [tmpDir.path]);
      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      expect(locator.locateAll(spec), hasLength(2));
    });

    test('isMatch validates a specific path', () {
      final snapDir = Directory('${tmpDir.path}/direct')..createSync();
      File(
        '${snapDir.path}/config.json',
      ).writeAsStringSync(jsonEncode({'model_type': 'test'}));
      File('${snapDir.path}/model.safetensors').writeAsStringSync('fake');

      final locator = SnapshotLocator();
      final spec = ModelSpec(
        id: 'test',
        family: 'Test',
        modalities: [ModelModality.textGeneration],
      );
      expect(locator.isMatch(snapDir.path, spec), isTrue);
    });
  });
}
