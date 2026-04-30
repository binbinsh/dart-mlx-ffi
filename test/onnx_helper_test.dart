import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_inference/runtime.dart';

void main() {
  test('creates native-backed int64 tensor', () {
    final values = Int64List.fromList([1, 2, 3, 4]);
    final tensor = int64Tensor(values, const [2, 2]);
    values[0] = 99;

    expect(tensor.dtype, RuntimeTensorDataType.int64);
    expect(tensor.shape, [2, 2]);
    expect(tensor.asInt64List(), [1, 2, 3, 4]);
    expect(tensor.nativeData, isNotNull);
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

      final libraries = discoverDefaultOnnxRuntimePreloadLibraries(
        runtimeEnvFile: envFile.path,
      );

      expect(libraries, contains(cudart.absolute.path));
    } finally {
      await root.delete(recursive: true);
    }
  });

  test('discovers CUDA preload libraries from provider Python venv', () async {
    final root = await Directory.systemTemp.createTemp('onnx-venv-test-');
    try {
      final cudaLibDir = Directory(
        '${root.path}/src/ttsbackends/providers/sarashina2-tts/src/'
        '.venv/lib/python3.12/site-packages/nvidia/cublas/lib',
      )..createSync(recursive: true);
      final cublasLt = File('${cudaLibDir.path}/libcublasLt.so.12')
        ..writeAsBytesSync(const []);

      final libraries = discoverOnnxRuntimePreloadLibraries(
        runtimeEnvSearchRoots: [root.path],
        libraryNames: const ['libcublasLt.so.12'],
      );

      expect(libraries, [cublasLt.absolute.path]);
    } finally {
      await root.delete(recursive: true);
    }
  });

  test('filters preload libraries by execution provider', () async {
    final root = await Directory.systemTemp.createTemp('onnx-provider-libs-');
    try {
      final sitePackages = '${root.path}/.venv/lib/python3.12/site-packages';
      final cudaDir = Directory('$sitePackages/nvidia/cuda_runtime/lib')
        ..createSync(recursive: true);
      final trtDir = Directory('$sitePackages/tensorrt_libs')
        ..createSync(recursive: true);
      final cudart = File('${cudaDir.path}/libcudart.so.12')
        ..writeAsBytesSync(const []);
      final nvinfer = File('${trtDir.path}/libnvinfer.so.10')
        ..writeAsBytesSync(const []);

      final cudaLibraries = discoverOnnxRuntimePreloadLibraries(
        runtimeEnvSearchRoots: [root.path],
        libraryNames: onnxRuntimePreloadLibraryNamesForProvider('cuda'),
      );
      final trtLibraries = discoverOnnxRuntimePreloadLibraries(
        runtimeEnvSearchRoots: [root.path],
        libraryNames: onnxRuntimePreloadLibraryNamesForProvider('tensorrt'),
      );

      expect(cudaLibraries, contains(cudart.absolute.path));
      expect(cudaLibraries, isNot(contains(nvinfer.absolute.path)));
      expect(trtLibraries, contains(cudart.absolute.path));
      expect(trtLibraries, contains(nvinfer.absolute.path));
    } finally {
      await root.delete(recursive: true);
    }
  });
}
