@TestOn('mac-os')
library;

import 'dart:ffi';
import 'dart:io';

import 'package:dart_inference/models.dart';
import 'package:test/test.dart';

void main() {
  group('ds4 integration', () {
    test('describes DeepSeek V4 Flash as a bundled ds4.c FFI model', () {
      expect(deepSeekV4FlashDs4Spec.id, ds4ModelId);
      expect(deepSeekV4FlashDs4Spec.modalities, [ModelModality.textGeneration]);
      expect(
        deepSeekV4FlashDs4Spec.metadata['runtimeScope'],
        'bundled-ds4-ffi',
      );
      expect(deepSeekV4FlashDs4Spec.requiredTags, contains('ffi'));
      expect(Ds4ModelArtifacts.q2.path, 'hf://$ds4HuggingFaceRepo/$ds4Q2File');
      expect(Ds4ModelArtifacts.q2.format, 'ds4-gguf');
      expect(Ds4ModelArtifacts.q2.metadata['runtimeEngine'], 'ds4');
      expect(
        Ds4ModelArtifacts.q2.metadata,
        isNot(containsPair('serverModel', anything)),
      );
      expect(Ds4ModelArtifacts.main(Ds4Quant.q4).metadata['quant'], 'q4');
      expect(Ds4ModelArtifacts.mtp.metadata['component'], 'mtp');
    });

    test('keeps FFI chat requests independent from HTTP payloads', () {
      const request = Ds4ChatRequest(
        messages: [
          <String, Object?>{'role': 'user', 'content': 'hello'},
        ],
        maxTokens: 12,
        temperature: 0.4,
        thinking: Ds4Thinking.max,
      );
      expect(request.messages.single['content'], 'hello');
      expect(request.maxTokens, 12);
      expect(request.temperature, 0.4);
      expect(request.thinking, Ds4Thinking.max);
    });

    test('configures DS4 FFI library builds with Metal source overrides', () {
      const config = Ds4FfiConfig(
        sourceDir: '/opt/ds4',
        modelPath: '/models/ds4flash.gguf',
        quant: Ds4Quant.q4,
        contextLength: 200000,
      );
      expect(config.sourceDir, '/opt/ds4');
      expect(config.modelPath, '/models/ds4flash.gguf');
      expect(config.quant, Ds4Quant.q4);
      expect(config.contextLength, 200000);

      final env = ds4MetalSourceEnvironment('/opt/ds4');
      expect(env['DS4_METAL_FLASH_ATTN_SOURCE'], endsWith('flash_attn.metal'));
      expect(env['DS4_METAL_SET_ROWS_SOURCE'], endsWith('set_rows.metal'));
    });

    test('discovers bundled ds4 submodule source candidates', () async {
      final candidates = await ds4BundledSourceCandidates();
      expect(
        candidates.any((path) => path.endsWith('third_party/ds4')),
        isTrue,
      );

      final sourceDir = await resolveDs4SourceDir();
      if (sourceDir != null) {
        expect(File('$sourceDir/ds4.c').existsSync(), isTrue);
        expect(File('$sourceDir/ds4.h').existsSync(), isTrue);
      }
    });

    test('builds bundled ds4 dynamic library for FFI loading', () async {
      final sourceDir = await resolveDs4SourceDir();
      expect(sourceDir, isNotNull);

      final libraryPath = await buildDs4DynamicLibrary(sourceDir!);
      expect(File(libraryPath).existsSync(), isTrue);

      final library = DynamicLibrary.open(libraryPath);
      expect(
        library
            .lookup<NativeFunction<Void Function()>>('ds4_engine_open')
            .address,
        isNot(0),
      );
      expect(
        library
            .lookup<NativeFunction<Void Function()>>('ds4_session_create')
            .address,
        isNot(0),
      );
      expect(
        library
            .lookup<NativeFunction<Void Function()>>('ds4_session_sample')
            .address,
        isNot(0),
      );
    });
  });
}
