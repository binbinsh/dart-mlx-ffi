library;

import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

Map<String, MlxArray> loadTensorMap(String snapshotPath) {
  final dir = Directory(snapshotPath);
  final files =
      dir
          .listSync()
          .whereType<File>()
          .where((file) => file.path.endsWith('.safetensors'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));
  if (files.isEmpty) {
    throw StateError('No safetensors files found under $snapshotPath.');
  }
  final merged = <String, MlxArray>{};
  for (final file in files) {
    final loaded = mx.io.loadSafetensors(file.path);
    for (final entry in loaded.tensors.entries) {
      if (merged.containsKey(entry.key)) {
        throw StateError('Duplicate tensor key ${entry.key} in ${file.path}.');
      }
      merged[entry.key] = entry.value;
    }
  }
  return merged;
}
