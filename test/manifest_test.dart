@TestOn('mac-os')
library;

import 'dart:convert';

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/models.dart';

void main() {
  group('ModelManifest', () {
    test('builtIn contains all 6 model families', () {
      final manifest = ModelManifest.builtIn();
      expect(manifest.models.length, 6);

      final ids = manifest.models.map((m) => m.id).toSet();
      expect(
        ids,
        containsAll([
          'qwen2_5',
          'qwen3_5',
          'paddle_ocr_vl',
          'qwen3_asr',
          'kitten_tts',
          'silero_vad',
        ]),
      );
    });

    test('lookup by id works', () {
      final manifest = ModelManifest.builtIn();
      final spec = manifest['qwen3_5'];
      expect(spec, isNotNull);
      expect(spec!.family, 'Qwen3.5');
    });

    test('lookup by id returns null for unknown', () {
      final manifest = ModelManifest.builtIn();
      expect(manifest['nonexistent'], isNull);
    });

    test('byModality filters correctly', () {
      final manifest = ModelManifest.builtIn();

      final textGen = manifest.byModality(ModelModality.textGeneration);
      expect(textGen.length, greaterThanOrEqualTo(2)); // qwen2_5, qwen3_5

      final stt = manifest.byModality(ModelModality.speechToText);
      expect(stt.length, 1);
      expect(stt.first.id, 'qwen3_asr');

      final tts = manifest.byModality(ModelModality.textToSpeech);
      expect(tts.length, 1);
      expect(tts.first.id, 'kitten_tts');

      final vad = manifest.byModality(ModelModality.voiceActivityDetection);
      expect(vad.length, 1);
      expect(vad.first.id, 'silero_vad');
    });

    test('JSON round-trip preserves all specs', () {
      final original = ModelManifest.builtIn();
      final jsonStr = original.toJsonString();
      final decoded = jsonDecode(jsonStr) as Map<String, Object?>;
      final restored = ModelManifest.fromJson(decoded);

      expect(restored.models.length, original.models.length);
      for (var i = 0; i < original.models.length; i++) {
        expect(restored.models[i].id, original.models[i].id);
        expect(restored.models[i].family, original.models[i].family);
        expect(
          restored.models[i].modalities.map((m) => m.name).toList(),
          original.models[i].modalities.map((m) => m.name).toList(),
        );
        expect(
          restored.models[i].requiredFiles,
          original.models[i].requiredFiles,
        );
      }
    });

    test('toJson contains version field', () {
      final manifest = ModelManifest.builtIn();
      final json = manifest.toJson();
      expect(json['version'], 1);
      expect(json['models'], isList);
    });

    test('fromJson with minimal spec', () {
      final json = {
        'version': 1,
        'models': [
          {
            'id': 'test',
            'family': 'Test',
            'modalities': ['textGeneration'],
          },
        ],
      };
      final manifest = ModelManifest.fromJson(json);
      expect(manifest.models.length, 1);
      expect(manifest.models.first.id, 'test');
      expect(manifest.models.first.requiredFiles, ['config.json']);
    });
  });
}
