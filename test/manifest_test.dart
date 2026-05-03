@TestOn('mac-os')
library;

import 'dart:convert';

import 'package:test/test.dart';

import 'package:dart_inference/models.dart';

void main() {
  group('ModelManifest', () {
    test('builtIn contains all runtime model families', () {
      final manifest = ModelManifest.builtIn();
      expect(manifest.models.length, 15);

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
          'qwen3_6_27b',
          'translategemma_4b_it',
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
      expect(manifest.bySupportLevel(SupportLevel.staging), hasLength(15));
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

    test('builtIn artifacts pin upgraded same-model runtime sources', () {
      final manifest = ModelManifest.builtIn();

      final qwenAsr = manifest['qwen3_asr']!;
      expect(
        qwenAsr.platformArtifacts[RuntimeEngine.mlx]!.metadata['sourceModel'],
        'Qwen/Qwen3-ASR-1.7B',
      );
      expect(
        qwenAsr.platformArtifacts[RuntimeEngine.coreml]!.path,
        contains('Qwen3-ASR-1.7B-CoreML-INT8'),
      );
      expect(
        qwenAsr.platformArtifacts[RuntimeEngine.coreml]!.targetPlatforms,
        containsAll(['ios', 'macos']),
      );
      expect(
        qwenAsr
            .platformArtifacts[RuntimeEngine.coreml]!
            .metadata['runtimeScope'],
        'model-level-coreml-stateful',
      );
      expect(
        qwenAsr.platformArtifacts[RuntimeEngine.onnx]!.path,
        contains('qwen3-asr-1.7b-onnx'),
      );
      expect(
        qwenAsr.platformArtifacts[RuntimeEngine.onnx]!.targetPlatforms,
        containsAll(['linux', 'android']),
      );
      expect(
        qwenAsr.platformArtifacts[RuntimeEngine.onnx]!.targetPlatforms,
        isNot(contains('windows')),
      );
      expect(qwenAsr.platformArtifacts[RuntimeEngine.onnx]!.accelerators, [
        Accelerator.npu,
        Accelerator.gpu,
        Accelerator.cpu,
      ]);
      expect(
        qwenAsr.platformArtifacts[RuntimeEngine.onnx]!.metadata['runtimeScope'],
        'model-level-asr-components',
      );
      expect(
        qwenAsr.platformArtifacts.containsKey(RuntimeEngine.litert),
        isFalse,
      );

      final kitten = manifest['kitten_tts']!;
      expect(
        kitten.platformArtifacts[RuntimeEngine.mlx]!.metadata['sourceModel'],
        'KittenML/kitten-tts-mini-0.8',
      );
      expect(
        kitten.platformArtifacts[RuntimeEngine.onnx]!.path,
        contains('KittenTTS-Mini-v0.8-ONNX'),
      );

      final gemma = manifest['gemma4']!;
      expect(
        gemma.platformArtifacts[RuntimeEngine.mlx]!.metadata['sourceModel'],
        'google/gemma-4-E4B-it',
      );
      expect(
        gemma.platformArtifacts[RuntimeEngine.onnx]!.path,
        contains('huggingworld/gemma-4-E4B-it-ONNX'),
      );
    });

    test(
      'builtIn artifacts cover production target platforms or record blockers',
      () {
        const requiredPlatforms = {'ios', 'macos', 'android'};
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
                'identityPassed': true,
                'correctnessPassed': true,
                'speedPassed': true,
                'peakMemoryPassed': true,
                'deviceProfilePassed': true,
                'endToEndRatio': 1.01,
                'peakMemoryRatio': 1.05,
                'iterationCount': 5,
                'warmupCount': 1,
                'latencyMs': {'sampleCount': 5, 'mean': 10.0},
                'runConfig': {'iters': 5},
                'inputSignature': {'digest': 'abc'},
              },
            },
          },
        ],
      });

      final promoted = manifest['qwen3_5']!;
      expect(promoted.supportLevel, SupportLevel.production);
      expect(promoted.validationStatus['macos']?.engine, RuntimeEngine.coreml);
      expect(promoted.validationStatus['macos']?.passed, isTrue);
      expect(promoted.validationStatus['macos']?.identityPassed, isTrue);
      expect(promoted.validationStatus['macos']?.endToEndRatio, 1.01);
      expect(promoted.validationStatus['macos']?.iterationCount, 5);
      expect(promoted.validationStatus['macos']?.latencyMs['sampleCount'], 5);
      expect(promoted.validationStatus['macos']?.runConfig['iters'], 5);
      expect(
        promoted.validationStatus['macos']?.inputSignature['digest'],
        'abc',
      );
      expect(manifest.productionModels.map((model) => model.id), ['qwen3_5']);
    });

    test('applies platform artifact patch', () {
      final manifest = ModelManifest.builtIn().withRuntimeValidation({
        'version': 1,
        'models': [
          {
            'id': 'paddle_ocr_vl',
            'platformArtifacts': {
              'coreml': {
                'engine': 'coreml',
                'path':
                    'benchmark/artifacts/converted/paddle_ocr_vl/coreml/pipeline.json',
                'sourceUri': 'converted://paddle_ocr_vl/coreml',
                'format': 'coreml-pipeline',
                'targetPlatforms': ['ios', 'macos'],
                'accelerators': ['ane', 'gpu', 'cpu'],
                'metadata': {'source': 'runtime_matrix'},
              },
            },
          },
        ],
      });

      final spec = manifest['paddle_ocr_vl']!;
      final coreml = spec.platformArtifacts[RuntimeEngine.coreml];
      expect(coreml, isNotNull);
      expect(coreml?.path, contains('paddle_ocr_vl/coreml/pipeline.json'));
      expect(coreml?.format, 'coreml-pipeline');
      expect(coreml?.targetPlatforms, containsAll(['ios', 'macos']));
      expect(coreml?.metadata['source'], 'runtime_matrix');
    });
  });
}
