/// Runtime artifact resolution helpers.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;

import '../models/shared/runtime_metadata.dart';

/// Resolves remote runtime artifacts into local paths before native loading.
abstract interface class RuntimeArtifactResolver {
  /// Return a locally cached artifact without network access.
  RuntimeArtifact resolveCached(RuntimeArtifact artifact);

  /// Resolve an artifact, downloading it when necessary.
  Future<RuntimeArtifact> resolve(RuntimeArtifact artifact);
}

/// Parsed Hugging Face artifact reference.
final class HuggingFaceArtifactRef {
  const HuggingFaceArtifactRef({
    required this.repo,
    required this.path,
    this.revision = 'main',
  });

  final String repo;
  final String path;
  final String revision;

  String get sourceUri => 'hf://$repo/$path';

  static HuggingFaceArtifactRef? maybeParse(RuntimeArtifact artifact) {
    final metadata = artifact.metadata;
    final repo = metadata['repo'] as String?;
    final path = metadata['artifact'] as String?;
    final revision = metadata['revision'] as String? ?? 'main';
    if (repo != null && path != null) {
      return HuggingFaceArtifactRef(
        repo: repo,
        path: path.isEmpty ? '.' : path,
        revision: revision,
      );
    }

    final uri = Uri.tryParse(artifact.sourceUri ?? artifact.path);
    if (uri == null || uri.scheme != 'hf' || uri.host.isEmpty) {
      return null;
    }
    final segments = uri.pathSegments.where((s) => s.isNotEmpty).toList();
    if (segments.isEmpty) {
      return HuggingFaceArtifactRef(
        repo: uri.host,
        path: '.',
        revision: revision,
      );
    }
    final repoFromUri = '${uri.host}/${segments.first}';
    final artifactPath = segments.length == 1
        ? '.'
        : segments.skip(1).join('/');
    return HuggingFaceArtifactRef(
      repo: repoFromUri,
      path: artifactPath,
      revision: revision,
    );
  }
}

/// Downloads and caches `hf://` runtime artifacts.
///
/// This is intentionally small and dependency-light. It uses the Hugging Face
/// repository tree endpoint for directory artifacts such as `.mlmodelc` and
/// `.mlpackage`, and direct `resolve` URLs for single-file ONNX/LiteRT assets.
final class HuggingFaceArtifactCache implements RuntimeArtifactResolver {
  HuggingFaceArtifactCache({
    String? cacheRoot,
    String endpoint = 'https://huggingface.co',
    String? token,
    this.refresh = false,
    this.timeout = const Duration(seconds: 30),
  }) : cacheRoot = cacheRoot ?? _defaultCacheRoot(),
       endpoint = Uri.parse(endpoint),
       token =
           token ??
           Platform.environment['HF_TOKEN'] ??
           Platform.environment['HUGGINGFACE_HUB_TOKEN'];

  final String cacheRoot;
  final Uri endpoint;
  final String? token;
  final bool refresh;
  final Duration timeout;

  @override
  RuntimeArtifact resolveCached(RuntimeArtifact artifact) {
    final ref = HuggingFaceArtifactRef.maybeParse(artifact);
    if (ref == null) return artifact;
    final localPath = _localArtifactPath(ref);
    if (_hasCachedArtifact(localPath)) {
      return _resolvedArtifact(artifact, ref, localPath);
    }
    throw StateError(
      'Hugging Face artifact ${ref.sourceUri} is not cached at $localPath. '
      'Use RuntimeRegistry.loadAsync() or call HuggingFaceArtifactCache.resolve().',
    );
  }

  @override
  Future<RuntimeArtifact> resolve(RuntimeArtifact artifact) async {
    final ref = HuggingFaceArtifactRef.maybeParse(artifact);
    if (ref == null) return artifact;
    final localPath = _localArtifactPath(ref);
    if (!refresh && _hasCachedArtifact(localPath)) {
      return _resolvedArtifact(artifact, ref, localPath);
    }
    if (_isDirectoryArtifact(ref.path)) {
      await _downloadTree(ref);
    } else {
      await _downloadFile(ref, ref.path);
    }
    if (!_hasCachedArtifact(localPath)) {
      throw StateError(
        'Downloaded ${ref.sourceUri}, but no local artifact was created at '
        '$localPath.',
      );
    }
    return _resolvedArtifact(artifact, ref, localPath);
  }

  Future<void> _downloadTree(HuggingFaceArtifactRef ref) async {
    final files = await _listFiles(ref);
    if (files.isEmpty) {
      throw StateError(
        'No files found for Hugging Face artifact ${ref.sourceUri}.',
      );
    }
    for (final filePath in files) {
      await _downloadFile(ref, filePath);
    }
  }

  Future<List<String>> _listFiles(HuggingFaceArtifactRef ref) async {
    final files = <String>[];
    Uri? next = _treeUri(ref);
    while (next != null) {
      next = await _withResponse<Uri?>(next, (response) async {
        if (response.statusCode != HttpStatus.ok) {
          throw StateError(
            'Hugging Face tree request failed for ${ref.sourceUri}: '
            'HTTP ${response.statusCode} ${await _readError(response)}',
          );
        }
        final body = await utf8.decodeStream(response);
        final decoded = jsonDecode(body);
        if (decoded is! List<Object?>) {
          throw StateError(
            'Unexpected Hugging Face tree response for ${ref.sourceUri}.',
          );
        }
        for (final entry in decoded) {
          if (entry is! Map) continue;
          if (entry['type'] != 'file') continue;
          final path = entry['path'];
          if (path is String) files.add(path);
        }
        final link = _nextLink(response.headers.value('link'));
        return link == null ? null : endpoint.resolve(link);
      });
    }
    files.sort();
    return files;
  }

  Future<void> _downloadFile(
    HuggingFaceArtifactRef ref,
    String filePath,
  ) async {
    final destination = File(
      p.joinAll([_snapshotRoot(ref), ...filePath.split('/')]),
    );
    await destination.parent.create(recursive: true);
    final tmp = File('${destination.path}.incomplete');
    try {
      await _withResponse(_fileUri(ref, filePath), (response) async {
        if (response.statusCode != HttpStatus.ok) {
          throw StateError(
            'Hugging Face download failed for hf://${ref.repo}/$filePath: '
            'HTTP ${response.statusCode} ${await _readError(response)}',
          );
        }
        final sink = tmp.openWrite();
        try {
          await response.pipe(sink);
        } finally {
          await sink.close();
        }
      });
      if (await destination.exists()) {
        await destination.delete();
      }
      await tmp.rename(destination.path);
    } catch (_) {
      if (await tmp.exists()) {
        await tmp.delete();
      }
      rethrow;
    }
  }

  Future<T> _withResponse<T>(
    Uri uri,
    Future<T> Function(HttpClientResponse response) handle,
  ) async {
    final client = HttpClient()..connectionTimeout = timeout;
    try {
      var current = uri;
      for (var redirects = 0; redirects <= 8; redirects += 1) {
        final request = await client.getUrl(current).timeout(timeout);
        request.followRedirects = false;
        request.headers.set(
          HttpHeaders.userAgentHeader,
          'dart-inference-runtime',
        );
        request.headers.set(HttpHeaders.acceptHeader, '*/*');
        if (_shouldAuthenticate(current)) {
          request.headers.set(HttpHeaders.authorizationHeader, 'Bearer $token');
        }
        final response = await request.close().timeout(timeout);
        if (_isRedirect(response.statusCode)) {
          final location = response.headers.value(HttpHeaders.locationHeader);
          if (location == null || location.isEmpty) {
            return await handle(response);
          }
          await response.drain<void>();
          current = current.resolve(location);
          continue;
        }
        return await handle(response);
      }
      throw StateError('Too many redirects while downloading $uri.');
    } finally {
      client.close(force: false);
    }
  }

  bool _shouldAuthenticate(Uri uri) {
    return token != null &&
        token!.isNotEmpty &&
        uri.scheme == endpoint.scheme &&
        uri.host == endpoint.host &&
        uri.port == endpoint.port;
  }

  Uri _treeUri(HuggingFaceArtifactRef ref) {
    final pathSegments = <String>[
      ..._baseSegments,
      'api',
      'models',
      ...ref.repo.split('/'),
      'tree',
      ref.revision,
      if (ref.path != '.') ...ref.path.split('/'),
    ];
    return endpoint.replace(
      pathSegments: pathSegments,
      queryParameters: const {'recursive': 'true'},
    );
  }

  Uri _fileUri(HuggingFaceArtifactRef ref, String filePath) {
    return endpoint.replace(
      pathSegments: [
        ..._baseSegments,
        ...ref.repo.split('/'),
        'resolve',
        ref.revision,
        ...filePath.split('/'),
      ],
      queryParameters: const {'download': 'true'},
    );
  }

  List<String> get _baseSegments =>
      endpoint.pathSegments.where((s) => s.isNotEmpty).toList();

  String _snapshotRoot(HuggingFaceArtifactRef ref) {
    return p.join(
      cacheRoot,
      'models--${ref.repo.replaceAll('/', '--')}',
      'snapshots',
      _safePathSegment(ref.revision),
    );
  }

  String _localArtifactPath(HuggingFaceArtifactRef ref) {
    if (ref.path == '.') return _snapshotRoot(ref);
    return p.joinAll([_snapshotRoot(ref), ...ref.path.split('/')]);
  }

  RuntimeArtifact _resolvedArtifact(
    RuntimeArtifact artifact,
    HuggingFaceArtifactRef ref,
    String path,
  ) {
    return artifact.copyWith(
      path: path,
      sourceUri: artifact.sourceUri ?? ref.sourceUri,
      metadata: {
        ...artifact.metadata,
        'resolvedSourceUri': ref.sourceUri,
        'resolvedRevision': ref.revision,
        'localCacheRoot': cacheRoot,
      },
    );
  }
}

bool _hasCachedArtifact(String path) {
  if (File(path).existsSync()) return true;
  final directory = Directory(path);
  if (!directory.existsSync()) return false;
  return directory.listSync(followLinks: false).isNotEmpty;
}

bool _isDirectoryArtifact(String path) {
  if (path == '.' || path.endsWith('/')) return true;
  final lower = path.toLowerCase();
  if (lower.endsWith('.mlmodelc') || lower.endsWith('.mlpackage')) {
    return true;
  }
  return !p.basename(path).contains('.');
}

bool _isRedirect(int statusCode) {
  return statusCode == HttpStatus.movedPermanently ||
      statusCode == HttpStatus.found ||
      statusCode == HttpStatus.seeOther ||
      statusCode == HttpStatus.temporaryRedirect ||
      statusCode == HttpStatus.permanentRedirect;
}

String _safePathSegment(String value) {
  return value.replaceAll(RegExp(r'[^A-Za-z0-9._-]'), '_');
}

String? _nextLink(String? header) {
  if (header == null || header.isEmpty) return null;
  for (final part in header.split(',')) {
    final pieces = part.split(';').map((s) => s.trim()).toList();
    if (pieces.isEmpty || !pieces.any((p) => p == 'rel="next"')) {
      continue;
    }
    final target = pieces.first;
    if (target.startsWith('<') && target.endsWith('>')) {
      return target.substring(1, target.length - 1);
    }
  }
  return null;
}

Future<String> _readError(HttpClientResponse response) async {
  final body = await utf8.decodeStream(response).catchError((Object _) => '');
  if (body.isEmpty) return '';
  return body.length <= 300 ? body : body.substring(0, 300);
}

String _defaultCacheRoot() {
  final env = Platform.environment;
  if (env['DART_INFERENCE_HF_CACHE'] case final value? when value.isNotEmpty) {
    return value;
  }
  if (Platform.isWindows) {
    final base = env['LOCALAPPDATA'] ?? env['APPDATA'];
    if (base != null && base.isNotEmpty) {
      return p.join(base, 'dart_inference', 'huggingface');
    }
  }
  final home = env['HOME'];
  if (home != null && home.isNotEmpty) {
    return p.join(home, '.cache', 'dart_inference', 'huggingface');
  }
  return p.join(Directory.systemTemp.path, 'dart_inference', 'huggingface');
}
