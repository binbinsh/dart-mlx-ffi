import 'dart:io';

/// Native runtime dependency discovery for provider-specific libraries.
final class RuntimeDependencyAudit {
  RuntimeDependencyAudit._({
    required this.provider,
    required this.tensorrtRequested,
    required this.searchedDirs,
    required this.tensorrt10,
    required this.tensorrt9,
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
    return RuntimeDependencyAudit._(
      provider: provider,
      tensorrtRequested: isTensorRtProvider(provider),
      searchedDirs: searchDirs,
      tensorrt10: RuntimeLibraryGroup.find(searchDirs, tensorRt10Libraries),
      tensorrt9: RuntimeLibraryGroup.find(searchDirs, tensorRt9Libraries),
    );
  }

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

  final String provider;
  final bool tensorrtRequested;
  final List<String> searchedDirs;
  final RuntimeLibraryGroup tensorrt10;
  final RuntimeLibraryGroup tensorrt9;

  bool get tensorrtReady =>
      !tensorrtRequested || tensorrt10.ready || tensorrt9.ready;

  String? get skipReason {
    if (tensorrtReady) {
      return null;
    }
    if (tensorrt10.missing.isEmpty && tensorrt9.missing.isEmpty) {
      return 'TensorRT runtime dependencies are missing.';
    }
    return [
      'TensorRT runtime dependencies are missing.',
      if (tensorrt10.missing.isNotEmpty)
        'TensorRT 10 missing: ${tensorrt10.missing.join(', ')}.',
      if (tensorrt9.missing.isNotEmpty)
        'TensorRT 9 missing: ${tensorrt9.missing.join(', ')}.',
    ].join(' ');
  }

  Map<String, Object?> toJson() => {
    'provider': provider,
    'tensorrtRequested': tensorrtRequested,
    'searchedDirs': searchedDirs,
    'tensorrt10': tensorrt10.toJson(),
    'tensorrt9': tensorrt9.toJson(),
    'tensorrtReady': tensorrtReady,
  };

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
    final values = <String>[
      if (root != null && root.isNotEmpty) ...[
        '$root/artifacts/runtime/tensorrt/lib',
        '$root/artifacts/runtime/cuda/lib',
        '$root/artifacts/runtime/onnxruntime/lib',
      ],
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
