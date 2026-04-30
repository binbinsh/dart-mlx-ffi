import 'dart:io';

import 'catalog.dart';

final class TtsBackendAssetAudit {
  const TtsBackendAssetAudit({
    required this.root,
    required this.providersRoot,
    required this.providers,
  });

  final String root;
  final String providersRoot;
  final Map<String, TtsBackendProviderAssetAudit> providers;

  static TtsBackendAssetAudit audit(String root) {
    final providersRoot = Directory('$root/src/ttsbackends/providers');
    return TtsBackendAssetAudit(
      root: root,
      providersRoot: providersRoot.path,
      providers: {
        for (final capability in TtsBackendCatalog.all)
          capability.provider: TtsBackendProviderAssetAudit.audit(
            providersRoot: providersRoot,
            capability: capability,
          ),
      },
    );
  }

  Map<String, Object?> toJson() => {
    'root': root,
    'providersRoot': providersRoot,
    'providers': {
      for (final entry in providers.entries) entry.key: entry.value.toJson(),
    },
  };
}

final class TtsBackendProviderAssetAudit {
  const TtsBackendProviderAssetAudit({
    required this.provider,
    required this.providerDir,
    required this.exists,
    required this.completeLocalOnnxAssets,
    required this.missingLocalOnnxAssets,
    required this.onnxFiles,
    required this.blockingModelFiles,
    required this.canRunPureDartNow,
    this.onnxTargets = const [],
    this.sourceAssets = const [],
  });

  final String provider;
  final String? providerDir;
  final bool exists;
  final bool completeLocalOnnxAssets;
  final List<String> missingLocalOnnxAssets;
  final List<String> onnxFiles;
  final List<String> blockingModelFiles;
  final bool canRunPureDartNow;
  final List<Map<String, Object?>> onnxTargets;
  final List<Map<String, Object?>> sourceAssets;

  static TtsBackendProviderAssetAudit audit({
    required Directory providersRoot,
    required TtsBackendCapability capability,
  }) {
    final dir = _providerDirectory(providersRoot, capability.provider);
    if (dir == null || !dir.existsSync()) {
      return TtsBackendProviderAssetAudit(
        provider: capability.provider,
        providerDir: null,
        exists: false,
        completeLocalOnnxAssets: false,
        missingLocalOnnxAssets: _expectedOnnxPaths(capability),
        onnxFiles: const [],
        blockingModelFiles: const [],
        canRunPureDartNow: false,
        onnxTargets: [
          for (final target in capability.onnxTargets)
            _missingTargetStatus(target),
        ],
        sourceAssets: [
          for (final source in capability.sourceAssets)
            _missingSourceStatus(source),
        ],
      );
    }

    final modelFiles = <String>[];
    final onnxFiles = <String>[];
    final blockingFiles = <String>[];
    for (final entity in dir.listSync(recursive: true).whereType<File>()) {
      final relative = entity.path.substring(dir.path.length + 1);
      if (!relative.startsWith('models/')) {
        continue;
      }
      modelFiles.add(relative);
      final lower = relative.toLowerCase();
      if (lower.endsWith('.onnx')) {
        onnxFiles.add(relative);
      } else if (_isBlockingModelFile(lower)) {
        blockingFiles.add(relative);
      }
    }
    modelFiles.sort();
    onnxFiles.sort();
    blockingFiles.sort();

    final expectedOnnx = _expectedOnnxPaths(capability);
    final missingOnnxAssets = <String>[
      for (final asset in expectedOnnx)
        if (!_assetExists(dir, modelFiles, asset)) asset,
    ];
    final completeLocalOnnxAssets =
        expectedOnnx.isNotEmpty && missingOnnxAssets.isEmpty;

    return TtsBackendProviderAssetAudit(
      provider: capability.provider,
      providerDir: dir.path,
      exists: true,
      completeLocalOnnxAssets: completeLocalOnnxAssets,
      missingLocalOnnxAssets: missingOnnxAssets,
      onnxFiles: onnxFiles,
      blockingModelFiles: blockingFiles,
      canRunPureDartNow:
          capability.isLocalDartOnnxReady && completeLocalOnnxAssets,
      onnxTargets: [
        for (final target in capability.onnxTargets) _targetStatus(dir, target),
      ],
      sourceAssets: [
        for (final source in capability.sourceAssets)
          _sourceStatus(dir, modelFiles, source),
      ],
    );
  }

  List<String> get missingRequiredOnnxTargets => [
    for (final target in onnxTargets)
      if (target['requiredForSynthesis'] == true && target['exists'] != true)
        target['path']!.toString(),
  ];

  List<String> get missingRequiredSourceAssets => [
    for (final source in sourceAssets)
      if (source['requiredForExport'] == true && source['exists'] != true)
        source['locator']!.toString(),
  ];

  Map<String, Object?> toJson() => {
    'providerDir': providerDir,
    'exists': exists,
    'completeLocalOnnxAssets': completeLocalOnnxAssets,
    'missingLocalOnnxAssets': missingLocalOnnxAssets,
    'onnxFiles': onnxFiles,
    'blockingModelFiles': blockingModelFiles,
    'canRunPureDartNow': canRunPureDartNow,
    if (onnxTargets.isNotEmpty) 'onnxTargets': onnxTargets,
    if (sourceAssets.isNotEmpty) 'sourceAssets': sourceAssets,
    if (missingRequiredOnnxTargets.isNotEmpty)
      'missingRequiredOnnxTargets': missingRequiredOnnxTargets,
    if (missingRequiredSourceAssets.isNotEmpty)
      'missingRequiredSourceAssets': missingRequiredSourceAssets,
  };
}

List<String> _expectedOnnxPaths(TtsBackendCapability capability) {
  if (capability.onnxTargets.isNotEmpty) {
    return [
      for (final target in capability.onnxTargets)
        if (target.requiredForSynthesis) target.path,
    ];
  }
  return capability.localOnnxAssets;
}

bool _assetExists(
  Directory providerDir,
  List<String> modelFiles,
  String asset,
) {
  if (File('${providerDir.path}/$asset').existsSync()) {
    return true;
  }
  final basename = _basename(asset);
  return modelFiles.any((path) => _basename(path) == basename);
}

Map<String, Object?> _targetStatus(
  Directory providerDir,
  TtsBackendOnnxTarget target,
) {
  final file = File('${providerDir.path}/${target.path}');
  final exists = file.existsSync();
  return {
    ...target.toJson(),
    'exists': exists,
    if (exists) 'sizeBytes': file.lengthSync(),
  };
}

Map<String, Object?> _missingTargetStatus(TtsBackendOnnxTarget target) => {
  ...target.toJson(),
  'exists': false,
};

Map<String, Object?> _sourceStatus(
  Directory providerDir,
  List<String> modelFiles,
  TtsBackendSourceAsset source,
) {
  final relative = _findSourcePath(providerDir, modelFiles, source);
  final file = relative == null ? null : File('${providerDir.path}/$relative');
  final exists = file?.existsSync() ?? false;
  return {
    ...source.toJson(),
    'locator': source.locator,
    'resolvedPath': ?relative,
    'exists': exists,
    if (exists) 'sizeBytes': file!.lengthSync(),
  };
}

Map<String, Object?> _missingSourceStatus(TtsBackendSourceAsset source) => {
  ...source.toJson(),
  'locator': source.locator,
  'exists': false,
};

String? _findSourcePath(
  Directory providerDir,
  List<String> modelFiles,
  TtsBackendSourceAsset source,
) {
  final path = source.path;
  if (path != null && File('${providerDir.path}/$path').existsSync()) {
    return path;
  }
  for (final candidate in source.paths) {
    if (File('${providerDir.path}/$candidate').existsSync()) {
      return candidate;
    }
  }
  final basename = source.basename;
  if (basename == null) {
    return null;
  }
  for (final file in modelFiles) {
    if (_basename(file) == basename) {
      return file;
    }
  }
  return null;
}

Directory? _providerDirectory(Directory providersRoot, String provider) {
  const aliases = {
    'glm-tts': ['glm-tts', 'glm_tts'],
    'elevenlabs3': ['elevenlabs', 'elevenlabs3'],
    'sonic3': ['sonic', 'sonic3'],
    'neutts-air': ['neutts-air', 'neutts_air'],
  };
  for (final name in [provider, ...?aliases[provider]]) {
    final dir = Directory('${providersRoot.path}/$name');
    if (dir.existsSync()) {
      return dir;
    }
  }
  return null;
}

bool _isBlockingModelFile(String lowerPath) =>
    lowerPath.endsWith('.pt') ||
    lowerPath.endsWith('.pth') ||
    lowerPath.endsWith('.safetensors') ||
    lowerPath.endsWith('.gguf') ||
    lowerPath.endsWith('.zip') ||
    lowerPath.endsWith('.bin');

String _basename(String path) {
  final slash = path.lastIndexOf('/');
  return slash < 0 ? path : path.substring(slash + 1);
}
