import 'dart:convert';
import 'dart:io';

import 'package:dart_inference/models.dart';

void main(List<String> args) {
  final parsed = _Args(args);
  final pretty = parsed.flag('pretty');
  final root = _discoverProjectRoot(parsed.option('root'));
  final payload = {
    ...TtsBackendCatalog.toJson(),
    if (root != null) 'assetAudit': _auditProviderAssets(root),
  };
  final encoder = pretty
      ? const JsonEncoder.withIndent('  ')
      : const JsonEncoder();
  stdout.writeln(encoder.convert(payload));
}

Map<String, Object?> _auditProviderAssets(String root) {
  final providersRoot = Directory('$root/src/ttsbackends/providers');
  final audits = <String, Object?>{};
  for (final capability in TtsBackendCatalog.all) {
    final dir = _providerDirectory(providersRoot, capability.provider);
    if (dir == null || !dir.existsSync()) {
      audits[capability.provider] = {
        'providerDir': null,
        'exists': false,
        'completeLocalOnnxAssets': false,
        'onnxFiles': const <String>[],
        'blockingModelFiles': const <String>[],
      };
      continue;
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
      } else if (lower.endsWith('.pt') ||
          lower.endsWith('.pth') ||
          lower.endsWith('.safetensors') ||
          lower.endsWith('.bin')) {
        blockingFiles.add(relative);
      }
    }
    modelFiles.sort();
    onnxFiles.sort();
    blockingFiles.sort();
    final missingOnnxAssets = <String>[
      for (final asset in capability.localOnnxAssets)
        if (!_containsBasename(modelFiles, asset) &&
            !File('${dir.path}/$asset').existsSync())
          asset,
    ];
    final completeLocalOnnxAssets =
        capability.localOnnxAssets.isNotEmpty && missingOnnxAssets.isEmpty;
    audits[capability.provider] = {
      'providerDir': dir.path,
      'exists': true,
      'completeLocalOnnxAssets': completeLocalOnnxAssets,
      'missingLocalOnnxAssets': missingOnnxAssets,
      'onnxFiles': onnxFiles,
      'blockingModelFiles': blockingFiles,
      'canRunPureDartNow':
          capability.isLocalDartOnnxReady && completeLocalOnnxAssets,
    };
  }
  return {
    'root': root,
    'providersRoot': providersRoot.path,
    'providers': audits,
  };
}

Directory? _providerDirectory(Directory providersRoot, String provider) {
  final aliases = {
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

bool _containsBasename(List<String> paths, String basename) {
  return paths.any((path) => path.split('/').last == basename);
}

String? _discoverProjectRoot(String? explicitRoot) {
  final envRoot = Platform.environment['UNIFRONTEND_ROOT'];
  for (final value in [explicitRoot, envRoot]) {
    if (value == null || value.isEmpty) {
      continue;
    }
    final root = Directory(value).absolute;
    if (_looksLikeUniFrontendRoot(root)) {
      return root.path;
    }
  }
  return null;
}

bool _looksLikeUniFrontendRoot(Directory directory) {
  return File('${directory.path}/src/ttsbackends/registry.toml').existsSync() &&
      Directory('${directory.path}/src/unifrontend').existsSync();
}

final class _Args {
  _Args(this.args);

  final List<String> args;

  bool flag(String name) => args.contains('--$name');

  String? option(String name) {
    final prefix = '--$name=';
    for (var i = 0; i < args.length; i++) {
      final value = args[i];
      if (value.startsWith(prefix)) {
        return value.substring(prefix.length);
      }
      if (value == '--$name' && i + 1 < args.length) {
        return args[i + 1];
      }
    }
    return null;
  }
}
