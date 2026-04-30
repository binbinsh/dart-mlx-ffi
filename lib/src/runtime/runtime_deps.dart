import 'dart:io';

import 'runtime_library_dirs.dart';

/// Native runtime dependency discovery for provider-specific libraries.
final class RuntimeDependencyAudit {
  RuntimeDependencyAudit._({
    required this.provider,
    required this.cudaRequested,
    required this.tensorrtRequested,
    required this.searchedDirs,
    required this.cuda,
    required this.tensorrt10,
    required this.tensorrt9,
    required this.tensorRtCompatibilityError,
  });

  factory RuntimeDependencyAudit.inspect({
    String? root,
    required String provider,
    Map<String, String>? environment,
    Iterable<String> extraSearchDirs = const [],
    bool includeSystemDirs = true,
  }) {
    final env = environment ?? Platform.environment;
    final searchDirs = _existingDirs(
      _searchDirs(
        root: root,
        environment: env,
        extraSearchDirs: extraSearchDirs,
        includeSystemDirs: includeSystemDirs,
      ),
    );
    final cudaRequested =
        isCudaProvider(provider) || isTensorRtProvider(provider);
    return RuntimeDependencyAudit._(
      provider: provider,
      cudaRequested: cudaRequested,
      tensorrtRequested: isTensorRtProvider(provider),
      searchedDirs: searchDirs,
      cuda: RuntimeLibraryGroup.find(searchDirs, cudaLibraries),
      tensorrt10: RuntimeLibraryGroup.find(searchDirs, tensorRt10Libraries),
      tensorrt9: RuntimeLibraryGroup.find(searchDirs, tensorRt9Libraries),
      tensorRtCompatibilityError: _tensorRtCompatibilityError(searchDirs),
    );
  }

  static const cudaLibraries = [
    'libcudart.so.12',
    'libcublas.so.12',
    'libcublasLt.so.12',
    'libcurand.so.10',
    'libcufft.so.11',
    'libcudnn.so.9',
  ];

  static const tensorRt10Libraries = [
    'libnvinfer.so.10',
    'libnvinfer_plugin.so.10',
    'libnvonnxparser.so.10',
  ];

  static const tensorRt9Libraries = [
    'libnvinfer.so.9',
    'libnvinfer_plugin.so.9',
    'libnvonnxparser.so.9',
  ];

  static List<String> preloadLibrariesForProvider(String provider) {
    if (isTensorRtProvider(provider)) {
      return const [...cudaLibraries, ...tensorRt10Libraries];
    }
    if (isCudaProvider(provider)) {
      return cudaLibraries;
    }
    return const [];
  }

  final String provider;
  final bool cudaRequested;
  final bool tensorrtRequested;
  final List<String> searchedDirs;
  final RuntimeLibraryGroup cuda;
  final RuntimeLibraryGroup tensorrt10;
  final RuntimeLibraryGroup tensorrt9;
  final String? tensorRtCompatibilityError;

  bool get cudaReady => !cudaRequested || cuda.ready;

  bool get tensorrtReady =>
      !tensorrtRequested ||
      (tensorrt10.ready && tensorRtCompatibilityError == null);

  bool get runtimeReady => cudaReady && tensorrtReady;

  String? get skipReason {
    if (runtimeReady) {
      return null;
    }
    final messages = <String>[];
    if (!cudaReady) {
      messages.add('CUDA runtime dependencies are missing.');
      if (cuda.missing.isNotEmpty) {
        messages.add('CUDA missing: ${cuda.missing.join(', ')}.');
      }
    }
    if (!tensorrtReady) {
      if (tensorRtCompatibilityError != null) {
        messages.add('TensorRT runtime dependencies are incompatible.');
      } else {
        messages.add('TensorRT runtime dependencies are missing.');
      }
      if (tensorrt10.missing.isNotEmpty && !tensorrt10.ready) {
        messages.add('TensorRT 10 missing: ${tensorrt10.missing.join(', ')}.');
      }
      if (!tensorrt10.ready && tensorrt9.ready) {
        messages.add(
          'TensorRT 9 libraries were found, but the CUDA 12 ONNX Runtime '
          'TensorRT provider requires TensorRT 10.',
        );
      }
      if (tensorRtCompatibilityError != null) {
        messages.add(tensorRtCompatibilityError!);
      }
    }
    return messages.join(' ');
  }

  Map<String, Object?> toJson() => {
    'provider': provider,
    'cudaRequested': cudaRequested,
    'tensorrtRequested': tensorrtRequested,
    'searchedDirs': searchedDirs,
    'cuda': cuda.toJson(),
    'tensorrt10': tensorrt10.toJson(),
    'tensorrt9': tensorrt9.toJson(),
    'tensorRtCompatibilityError': tensorRtCompatibilityError,
    'cudaReady': cudaReady,
    'tensorrtReady': tensorrtReady,
    'runtimeReady': runtimeReady,
  };

  static bool isCudaProvider(String provider) {
    final value = provider.trim().toLowerCase();
    return value == 'cuda' || value == 'cudaexecutionprovider';
  }

  static bool isTensorRtProvider(String provider) {
    final value = provider.trim().toLowerCase();
    return value == 'trt' ||
        value == 'tensorrt' ||
        value == 'tensorrtexecutionprovider';
  }

  static List<String> splitPathEnv(String? value) {
    if (value == null || value.isEmpty) {
      return const [];
    }
    return [
      for (final item in value.split(Platform.isWindows ? ';' : ':'))
        if (item.trim().isNotEmpty) item.trim(),
    ];
  }

  static List<String> _searchDirs({
    required String? root,
    required Map<String, String> environment,
    required Iterable<String> extraSearchDirs,
    required bool includeSystemDirs,
  }) {
    final roots = cleanRuntimeLibraryPaths([
      if (root != null && root.isNotEmpty) root,
    ]);
    final values = <String>[
      ...runtimeLibraryDirectories(roots),
      ...pythonNvidiaLibraryDirectories(roots),
      ...splitPathEnv(environment['LD_LIBRARY_PATH']),
      ...extraSearchDirs,
      if (includeSystemDirs) ...const [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda/lib64',
        '/usr/local/tensorrt/lib',
        '/opt/tensorrt/lib',
      ],
    ];
    final seen = <String>{};
    return [
      for (final path in values)
        if (path.trim().isNotEmpty && seen.add(path.trim())) path.trim(),
    ];
  }

  static List<String> _existingDirs(List<String> dirs) => [
    for (final dir in dirs)
      if (Directory(dir).existsSync()) Directory(dir).absolute.path,
  ];

  static String? _tensorRtCompatibilityError(List<String> searchDirs) {
    for (final dir in searchDirs) {
      final builderResource = File(
        '$dir/libnvinfer_builder_resource.so.10.0.1',
      );
      final nvinfer = File('$dir/libnvinfer.so.10');
      if (builderResource.existsSync() && nvinfer.existsSync()) {
        return 'TensorRT 10.0.1 Python wheel runtime is not compatible with '
            'this ORT TensorRT path on this host; install TensorRT 10.9+ '
            'runtime libraries.';
      }
    }
    return null;
  }
}

final class RuntimeLibraryGroup {
  const RuntimeLibraryGroup({
    required this.required,
    required this.found,
    required this.missing,
  });

  factory RuntimeLibraryGroup.find(List<String> dirs, List<String> names) {
    final found = <String, String>{};
    for (final name in names) {
      for (final dir in dirs) {
        final candidate = File('$dir/$name');
        if (candidate.existsSync()) {
          found[name] = candidate.absolute.path;
          break;
        }
      }
    }
    return RuntimeLibraryGroup(
      required: names,
      found: found,
      missing: [
        for (final name in names)
          if (!found.containsKey(name)) name,
      ],
    );
  }

  final List<String> required;
  final Map<String, String> found;
  final List<String> missing;

  bool get ready => missing.isEmpty;

  Map<String, Object?> toJson() => {
    'required': required,
    'found': found,
    'missing': missing,
    'ready': ready,
  };
}
