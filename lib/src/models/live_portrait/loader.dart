/// Snapshot loader for LivePortrait weights.
///
/// Reads `manifest.json` produced by
/// `tool/convert_live_portrait_weights.py`, validates the schema kind
/// against [kLivePortraitSchemaVersion], and resolves per-module paths
/// on disk. The eight modules are listed in [LivePortraitModule].
///
/// **Today** weights ship as ONNX files (consumed by ORT through the
/// CoreML EP). The Phase 2 conversion to MLX safetensors will replace
/// the bytes on disk; the manifest schema and this loader stay
/// unchanged because [LivePortraitWeightPaths] only stores filenames.
///
/// **Validation**: [open] performs an existence check across every
/// module and reports all missing paths in a single
/// [LivePortraitIncompleteSnapshot] so the download UI can resume the
/// fetch in one pass.
///
/// **sha256 verification** is deferred. The manifest carries `sha256`
/// per module (see [LivePortraitModuleMeta.sha256]); a future
/// download/validation flow will compare against on-disk bytes when
/// the cmdspace-app side wires the first-launch verification UI.
/// Skipped here to avoid pulling the `crypto` package into the
/// inference runtime for a path nothing currently exercises.
library;

import 'dart:convert';
import 'dart:io';

import 'config.dart';

/// Identifier for the eight sub-models we load piecewise.
enum LivePortraitModule {
  appearance,
  motion,
  warp,
  decoder,
  stitch,
  hubert,
  lmdm,
  faceDetector,
}

extension on LivePortraitModule {
  String relativePath(LivePortraitWeightPaths w) => switch (this) {
    LivePortraitModule.appearance => w.appearance,
    LivePortraitModule.motion => w.motion,
    LivePortraitModule.warp => w.warp,
    LivePortraitModule.decoder => w.decoder,
    LivePortraitModule.stitch => w.stitch,
    LivePortraitModule.hubert => w.hubert,
    LivePortraitModule.lmdm => w.lmdm,
    LivePortraitModule.faceDetector => w.faceDetector,
  };

  /// Manifest key under `modules[].key` in the produced JSON. Used to
  /// look up sha256/bytes metadata.
  String get manifestKey => switch (this) {
    LivePortraitModule.appearance => 'appearance',
    LivePortraitModule.motion => 'motion',
    LivePortraitModule.warp => 'warp',
    LivePortraitModule.decoder => 'decoder',
    LivePortraitModule.stitch => 'stitch',
    LivePortraitModule.hubert => 'hubert',
    LivePortraitModule.lmdm => 'lmdm',
    LivePortraitModule.faceDetector => 'face_detector',
  };
}

/// Per-module metadata pulled from the `modules` array in `manifest.json`.
final class LivePortraitModuleMeta {
  const LivePortraitModuleMeta({
    required this.key,
    required this.filename,
    required this.subdir,
    required this.bytes,
    required this.sha256,
  });

  factory LivePortraitModuleMeta.fromJson(Map<String, Object?> json) =>
      LivePortraitModuleMeta(
        key: json['key'] as String,
        filename: json['filename'] as String,
        subdir: json['subdir'] as String,
        bytes: (json['bytes'] as num).toInt(),
        sha256: json['sha256'] as String,
      );

  final String key;
  final String filename;
  final String subdir;
  final int bytes;

  /// Hex sha256 of the on-disk file at conversion time. Verification
  /// against actual bytes is deferred — see library comment.
  final String sha256;
}

/// Thrown by [LivePortraitSnapshot.open] when one or more module files
/// are missing from disk. Aggregates every missing module so callers
/// (e.g. a download UI) can resume the entire fetch in one pass
/// instead of failing once per module.
final class LivePortraitIncompleteSnapshot implements Exception {
  LivePortraitIncompleteSnapshot(this.snapshotDir, this.missing);

  final String snapshotDir;
  final List<({LivePortraitModule module, String absolutePath})> missing;

  @override
  String toString() {
    final lines = missing
        .map((m) => '  - ${m.module.name}: ${m.absolutePath}')
        .join('\n');
    return 'LivePortraitIncompleteSnapshot: $snapshotDir is missing '
        '${missing.length} module(s):\n$lines';
  }
}

final class LivePortraitSnapshot {
  LivePortraitSnapshot._({
    required this.snapshotDir,
    required this.config,
    required Map<String, LivePortraitModuleMeta> moduleMeta,
  }) : _moduleMeta = moduleMeta;

  /// Read `<snapshotDir>/manifest.json`, validate the schema version,
  /// and check that every module file exists on disk. Throws
  /// [LivePortraitIncompleteSnapshot] aggregating every missing module
  /// so callers can re-run the download script with a known module
  /// list.
  factory LivePortraitSnapshot.open(String snapshotDir) {
    final manifestPath = '$snapshotDir/manifest.json';
    final file = File(manifestPath);
    if (!file.existsSync()) {
      throw FileSystemException(
        'live_portrait manifest not found',
        manifestPath,
      );
    }
    final raw = jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
    final config = LivePortraitConfig.fromJson(raw);
    final moduleMeta = <String, LivePortraitModuleMeta>{};
    final modulesRaw = raw['modules'];
    if (modulesRaw is List) {
      for (final entry in modulesRaw) {
        if (entry is Map<String, Object?>) {
          final meta = LivePortraitModuleMeta.fromJson(entry);
          moduleMeta[meta.key] = meta;
        }
      }
    }
    final snapshot = LivePortraitSnapshot._(
      snapshotDir: snapshotDir,
      config: config,
      moduleMeta: moduleMeta,
    );
    snapshot.validate();
    return snapshot;
  }

  final String snapshotDir;
  final LivePortraitConfig config;
  final Map<String, LivePortraitModuleMeta> _moduleMeta;

  /// Resolve the on-disk path for [module]. Does **not** check
  /// existence — call [validate] for that. Cheap; safe to call inside
  /// hot paths.
  String pathFor(LivePortraitModule module) {
    final rel = module.relativePath(config.weights);
    return '$snapshotDir/$rel';
  }

  /// Manifest metadata (filename, bytes, sha256) for [module], or null
  /// if the manifest predates the `modules[]` array (older snapshots).
  LivePortraitModuleMeta? metaFor(LivePortraitModule module) =>
      _moduleMeta[module.manifestKey];

  /// Existence check across all eight modules. Throws
  /// [LivePortraitIncompleteSnapshot] aggregating every missing file.
  /// Called by [open]; safe to re-invoke (cheap, only stats files).
  void validate() {
    final missing = <({LivePortraitModule module, String absolutePath})>[];
    for (final module in LivePortraitModule.values) {
      final abs = pathFor(module);
      if (!File(abs).existsSync()) {
        missing.add((module: module, absolutePath: abs));
      }
    }
    if (missing.isNotEmpty) {
      throw LivePortraitIncompleteSnapshot(snapshotDir, missing);
    }
  }
}
