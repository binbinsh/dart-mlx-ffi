import 'dart:io';

List<String> cleanRuntimeLibraryPaths(Iterable<String> values) {
  final out = <String>{};
  for (final value in values) {
    final trimmed = value.trim();
    if (trimmed.isNotEmpty) {
      out.add(trimmed);
    }
  }
  return out.toList(growable: false);
}

List<String> runtimeLibraryDirectories(Iterable<String> roots) {
  final dirs = <String>{};
  for (final root in cleanRuntimeLibraryPaths(roots)) {
    for (final relative in const [
      'artifacts/runtime/onnxruntime/lib',
      'artifacts/runtime/cuda/lib',
      'artifacts/runtime/tensorrt/lib',
    ]) {
      addExistingRuntimeLibraryDir(dirs, '$root/$relative');
    }
  }
  return dirs.toList(growable: false);
}

List<String> pythonNvidiaLibraryDirectories(Iterable<String> roots) {
  final dirs = <String>{};
  for (final root in cleanRuntimeLibraryPaths(roots)) {
    for (final venv in _venvCandidates(root)) {
      _addNvidiaLibDirsFromVenv(dirs, venv);
    }
  }
  return dirs.toList(growable: false);
}

void addExistingRuntimeLibraryDir(Set<String> out, String path) {
  final dir = Directory(path);
  if (dir.existsSync()) {
    out.add(dir.absolute.path);
  }
}

Iterable<String> _venvCandidates(String rawRoot) sync* {
  final root = rawRoot.trim();
  if (root.isEmpty) {
    return;
  }
  final entityType = FileSystemEntity.typeSync(root);
  final base = entityType == FileSystemEntityType.file
      ? File(root).absolute.parent.path
      : Directory(root).absolute.path;
  yield '$base/.venv';
  yield '$base/src/.venv';

  const modelsMarker = '/models/';
  final modelsIndex = base.indexOf(modelsMarker);
  if (modelsIndex > 0) {
    final providerRoot = base.substring(0, modelsIndex);
    yield '$providerRoot/src/.venv';
  }

  final providers = Directory('$base/src/ttsbackends/providers');
  if (providers.existsSync()) {
    for (final entry in providers.listSync(followLinks: false)) {
      if (entry is Directory) {
        yield '${entry.absolute.path}/src/.venv';
      }
    }
  }
}

void _addNvidiaLibDirsFromVenv(Set<String> out, String venv) {
  final lib = Directory('$venv/lib');
  if (!lib.existsSync()) {
    return;
  }
  for (final entry in lib.listSync(followLinks: false)) {
    if (entry is! Directory) {
      continue;
    }
    final name = _basename(entry.path);
    if (!name.startsWith('python')) {
      continue;
    }
    final sitePackages = '${entry.path}/site-packages';
    addExistingRuntimeLibraryDir(out, '$sitePackages/tensorrt_libs');
    final nvidia = '$sitePackages/nvidia';
    for (final package in const [
      'cuda_runtime',
      'cublas',
      'curand',
      'cufft',
      'cudnn',
    ]) {
      addExistingRuntimeLibraryDir(out, '$nvidia/$package/lib');
    }
  }
}

String _basename(String path) {
  final normalized = path.replaceAll('\\', '/');
  final index = normalized.lastIndexOf('/');
  return index < 0 ? normalized : normalized.substring(index + 1);
}
