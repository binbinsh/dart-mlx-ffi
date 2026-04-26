@TestOn('mac-os')
library;

import 'dart:convert';

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

void main() {
  group('ModelManifest', () {
    test('builtIn contains all runtime model families', () {
      final manifest = ModelManifest.builtIn();
      expect(manifest.models.length, 17);

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
          'qwen3_vl',
          'gemma4',
          'function_gemma',
          'embedding_gemma',
          'qwen3_5_27b_dwq',
          'translategemma_27b_it',
          'nemotron3_nano_30b',
          'glm4_7_flash',
          'minicpm_o_4_5',
          'gemma_sea_lion_v4_4b_vl',
          'ming_omni_tts_0_5b',
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
      expect(
        stt.map((model) => model.id),
        containsAll(['qwen3_asr', 'minicpm_o_4_5']),
      );

      final tts = manifest.byModality(ModelModality.textToSpeech);
      expect(
        tts.map((model) => model.id),
        containsAll(['kitten_tts', 'ming_omni_tts_0_5b']),
      );

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

    test('builtIn models default to staging until full matrix passes', () {
      final manifest = ModelManifest.builtIn();
      expect(manifest.bySupportLevel(SupportLevel.staging), hasLength(17));
      expect(manifest.productionModels, isEmpty);
    });

    test('builtIn artifacts use Hugging Face sources without placeholders', () {
      final manifest = ModelManifest.builtIn();
      for (final spec in manifest.models) {
        expect(spec.platformArtifacts, isNotEmpty, reason: spec.id);
        for (final artifact in spec.platformArtifacts.values) {
          expect(artifact.path, startsWith('hf://'), reason: spec.id);
          expect(artifact.sourceUri, artifact.path, reason: spec.id);
          expect(artifact.metadata['source'], 'huggingface', reason: spec.id);
          expect(artifact.path, isNot(contains('/absolute/path')));
          expect(artifact.path, isNot(contains('C:/models')));
          expect(artifact.path, isNot(contains('/models/')));
        }
      }
    });

    test(
      'builtIn artifacts cover production target platforms or record blockers',
      () {
        const requiredPlatforms = {
          'ios',
          'macos',
          'windows',
          'linux',
          'android',
        };
        final manifest = ModelManifest.builtIn();
        for (final spec in manifest.models) {
          final covered = <String>{};
          for (final artifact in spec.platformArtifacts.values) {
            covered.addAll(artifact.targetPlatforms);
          }
          final missing = requiredPlatforms.difference(covered);
          if (missing.isEmpty) {
            continue;
          }
          final migration = spec.metadata['runtimeMigration'];
          expect(migration, isA<Map>(), reason: spec.id);
          final migrationMap = (migration as Map).cast<String, Object?>();
          expect(migrationMap['status'], 'partial', reason: spec.id);
          final blockers = (migrationMap['blockedPlatforms'] as Map)
              .cast<String, Object?>();
          expect(blockers.keys, containsAll(missing), reason: spec.id);
        }
      },
    );

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
      expect(manifest.models.first.supportLevel, SupportLevel.staging);
    });

    test('applies runtime promotion patch', () {
      final manifest = ModelManifest.builtIn().withRuntimeValidation({
        'version': 1,
        'models': [
          {
            'id': 'qwen3_5',
            'supportLevel': 'production',
            'validationStatus': {
              'macos': {
                'platform': 'macos',
                'engine': 'coreml',
                'correctnessPassed': true,
                'speedPassed': true,
                'peakMemoryPassed': true,
                'deviceProfilePassed': true,
                'peakMemoryRatio': 1.05,
              },
            },
          },
        ],
      });

      final promoted = manifest['qwen3_5']!;
      expect(promoted.supportLevel, SupportLevel.production);
      expect(promoted.validationStatus['macos']?.engine, RuntimeEngine.coreml);
      expect(promoted.validationStatus['macos']?.passed, isTrue);
      expect(manifest.productionModels.map((model) => model.id), ['qwen3_5']);
    });
  });
}
