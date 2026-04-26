import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_inference/runtime.dart';

void main() {
  group('Zig native runtime', () {
    test('reports Zig backend metadata', () {
      final info = NativeRuntimeBackend.info();
      expect(info['native_backend'], 'zig');
      expect(info['zig_version'], '0.16.0');
      expect(info['abi'], 'dart_inference_runtime_v1');
      final mlx = info['mlx_backend'] as Map<Object?, Object?>;
      expect(mlx['owner'], 'zig');
      expect(mlx['api'], 'mlx-c');
      expect(mlx['enabled'], isFalse);
    });

    test('keeps MLX ownership in Zig instead of the C++ adapter', () {
      const spec = ModelSpec(
        id: 'zig_mlx',
        family: 'Zig MLX',
        modalities: [ModelModality.embedding],
        platformArtifacts: {
          RuntimeEngine.mlx: RuntimeArtifact(
            engine: RuntimeEngine.mlx,
            path: 'zig://mlx-test',
          ),
        },
      );
      const bundle = ModelBundle(
        spec: spec,
        rootPath: '',
        artifact: RuntimeArtifact(
          engine: RuntimeEngine.mlx,
          path: 'zig://mlx-test',
        ),
      );
      final runtime = NativeModelRuntime(RuntimeEngine.mlx);
      expect(
        () => runtime.load(bundle, const RuntimeOptions()),
        throwsA(
          isA<StateError>().having(
            (error) => error.message,
            'message',
            contains('Zig-owned MLX backend is not implemented yet'),
          ),
        ),
      );
    });

    test('runs explicit echo mode through the model runtime ABI', () {
      const spec = ModelSpec(
        id: 'zig_echo',
        family: 'Zig echo',
        modalities: [ModelModality.embedding],
        platformArtifacts: {
          RuntimeEngine.onnx: RuntimeArtifact(
            engine: RuntimeEngine.onnx,
            path: 'zig://echo',
          ),
        },
      );
      const bundle = ModelBundle(
        spec: spec,
        rootPath: '',
        artifact: RuntimeArtifact(
          engine: RuntimeEngine.onnx,
          path: 'zig://echo',
        ),
      );
      final runtime = NativeModelRuntime(RuntimeEngine.onnx);
      final session = runtime.load(
        bundle,
        const RuntimeOptions(
          backendOptions: {'zigRuntimeMode': 'echo'},
          diagnostics: true,
        ),
      );
      try {
        final outputs = session.run(
          ModelInputs({
            'x': RuntimeTensor.float32([3], Float32List.fromList([1, 2, 3])),
          }),
        );
        final tensor = outputs.values['x'] as RuntimeTensor?;
        expect(tensor, isA<RuntimeTensor>());
        expect(tensor!.shape, [3]);
        expect(tensor.asFloat32List(), [1, 2, 3]);
        expect(outputs.diagnostics['native_backend'], 'zig');
        expect(outputs.diagnostics['mode'], 'echo');
      } finally {
        session.close();
      }
    });

    test(
      'accepts Zig-owned native tensor buffers without input scratch copy',
      () {
        const spec = ModelSpec(
          id: 'zig_echo_native_input',
          family: 'Zig echo native input',
          modalities: [ModelModality.embedding],
          platformArtifacts: {
            RuntimeEngine.onnx: RuntimeArtifact(
              engine: RuntimeEngine.onnx,
              path: 'zig://echo',
            ),
          },
        );
        const bundle = ModelBundle(
          spec: spec,
          rootPath: '',
          artifact: RuntimeArtifact(
            engine: RuntimeEngine.onnx,
            path: 'zig://echo',
          ),
        );
        final runtime = NativeModelRuntime(RuntimeEngine.onnx);
        final session = runtime.load(
          bundle,
          const RuntimeOptions(backendOptions: {'zigRuntimeMode': 'echo'}),
        );
        final input = NativeTensorBuffer.float32([3]);
        try {
          input.asFloat32List().setAll(0, [4, 5, 6]);
          final tensor = input.tensor;
          input.asFloat32List()[1] = 7;

          final outputs = session.run(ModelInputs({'x': tensor}));
          try {
            final output = outputs.values['x'] as RuntimeTensor;
            expect(output.asFloat32List(), [4, 7, 6]);
            expect(outputs.diagnostics, isEmpty);
          } finally {
            outputs.close();
          }
        } finally {
          input.close();
          session.close();
        }
      },
    );

    test('rejects closed native input buffers', () {
      const spec = ModelSpec(
        id: 'zig_echo_closed_native_input',
        family: 'Zig echo closed native input',
        modalities: [ModelModality.embedding],
        platformArtifacts: {
          RuntimeEngine.onnx: RuntimeArtifact(
            engine: RuntimeEngine.onnx,
            path: 'zig://echo',
          ),
        },
      );
      const bundle = ModelBundle(
        spec: spec,
        rootPath: '',
        artifact: RuntimeArtifact(
          engine: RuntimeEngine.onnx,
          path: 'zig://echo',
        ),
      );
      final runtime = NativeModelRuntime(RuntimeEngine.onnx);
      final session = runtime.load(
        bundle,
        const RuntimeOptions(backendOptions: {'zigRuntimeMode': 'echo'}),
      );
      final input = NativeTensorBuffer.float32([1]);
      final tensor = input.tensor;
      input.close();
      try {
        expect(
          () => session.run(ModelInputs({'x': tensor})),
          throwsA(isA<StateError>()),
        );
      } finally {
        session.close();
      }
    });
  });
}
