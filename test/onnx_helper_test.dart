import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_inference/runtime.dart';

void main() {
  test('creates zero-copy int64 tensor view', () {
    final values = Int64List.fromList([1, 2, 3, 4]);
    final tensor = int64Tensor(values, const [2, 2]);

    expect(tensor.dtype, RuntimeTensorDataType.int64);
    expect(tensor.shape, [2, 2]);
    expect(tensor.asInt64List(), [1, 2, 3, 4]);
  });

  test('views float32 tensor bytes with offset', () {
    final raw = Float32List.fromList([0.25, 0.5, 0.75]);
    final tensor = RuntimeTensor(
      dtype: RuntimeTensorDataType.float32,
      shape: const [3],
      bytes: Uint8List.view(raw.buffer),
    );

    expect(float32View(tensor), [0.25, 0.5, 0.75]);
  });

  test('discovers ONNX runtime preload libraries from explicit dirs', () async {
    final dir = await Directory.systemTemp.createTemp('onnx-preload-test-');
    try {
      final cudart = File('${dir.path}/libcudart.so.12');
      await cudart.writeAsBytes(const []);

      final libraries = discoverOnnxRuntimePreloadLibraries(
        libraryDirectories: [dir.path],
      );

      expect(libraries, [cudart.absolute.path]);
      expect(
        encodeOnnxRuntimePreloadLibraries(libraries),
        cudart.absolute.path,
      );
    } finally {
      await dir.delete(recursive: true);
    }
  });

  test('discovers CUDA preload libraries from runtime env file', () async {
    final root = await Directory.systemTemp.createTemp('onnx-env-test-');
    try {
      final ortLibDir = Directory('${root.path}/runtime/onnxruntime/lib')
        ..createSync(recursive: true);
      final cudaLibDir = Directory('${root.path}/runtime/cuda/lib')
        ..createSync(recursive: true);
      final ort = File('${ortLibDir.path}/libonnxruntime.so.1.25.0')
        ..writeAsBytesSync(const []);
      final cudart = File('${cudaLibDir.path}/libcudart.so.12')
        ..writeAsBytesSync(const []);
      final envFile = File('${root.path}/.dart_inference_runtime_env.json')
        ..writeAsStringSync(
          '{"DART_INFERENCE_ORT_RUNTIME_LIBRARY":"${ort.path}"}',
        );

      final runtimeEnv = DartInferenceRuntimeEnv.load(
        runtimeEnvFile: envFile.path,
      );
      expect(
        runtimeEnv.onnxPreloadLibraryDirectories(),
        contains(cudaLibDir.absolute.path),
      );

      final libraries = discoverDefaultOnnxRuntimePreloadLibraries(
        runtimeEnvFile: envFile.path,
      );

      expect(libraries, contains(cudart.absolute.path));
    } finally {
      await root.delete(recursive: true);
    }
  });
}
