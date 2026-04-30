/// Runtime artifact resolution helpers.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';

import 'native_ffi.dart' as dz;
import 'package:ffi/ffi.dart';

import '../models/shared/runtime_metadata.dart';
import 'native_bindings.dart' as native;

const _hfFieldSep = '\x1f';

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
    final parsed = _hfRef(artifact);
    if (parsed == null) return null;
    return HuggingFaceArtifactRef(
      repo: parsed.repo,
      path: parsed.path,
      revision: parsed.revision,
    );
  }
}

final class _HfRef {
  const _HfRef({
    required this.repo,
    required this.path,
    required this.revision,
  });

  final String repo;
  final String path;
  final String revision;
}

_HfRef? _hfRef(RuntimeArtifact artifact) {
  final metadata = artifact.metadata;
  final repoMetadata = metadata['repo'];
  final artifactMetadata = metadata['artifact'];
  final hasMetadata = repoMetadata is String && artifactMetadata is String;
  final sourceUri = dz.NativeUtf8CString.utf8(artifact.sourceUri ?? '');
  final artifactPath = dz.NativeUtf8CString.utf8(artifact.path);
  final repo = dz.NativeUtf8CString.utf8(hasMetadata ? repoMetadata : '');
  final path = dz.NativeUtf8CString.utf8(hasMetadata ? artifactMetadata : '');
  final revision = dz.NativeUtf8CString.utf8(
    metadata['revision'] is String ? '${metadata['revision']}' : '',
  );
  ffi.Pointer<ffi.Char> result = ffi.nullptr;
  try {
    result = native.hfRef(
      sourceUri.pointer,
      artifactPath.pointer,
      repo.pointer,
      path.pointer,
      revision.pointer,
    );
    if (result == ffi.nullptr) return null;
    final fields = result.cast<Utf8>().toDartString().split(_hfFieldSep);
    if (fields.length != 3) {
      return null;
    }
    final repoValue = fields[0];
    final pathValue = fields[1];
    final revisionValue = fields[2];
    if (repoValue.isEmpty || pathValue.isEmpty || revisionValue.isEmpty) {
      return null;
    }
    return _HfRef(repo: repoValue, path: pathValue, revision: revisionValue);
  } finally {
    if (result != ffi.nullptr) native.freeStr(result);
    sourceUri.close();
    artifactPath.close();
    repo.close();
    path.close();
    revision.close();
  }
}

String _cachePath(
  String cacheRoot,
  String repo,
  String revision,
  String artifactPath,
) {
  final root = dz.NativeUtf8CString.utf8(cacheRoot);
  final repoPtr = dz.NativeUtf8CString.utf8(repo);
  final revisionPtr = dz.NativeUtf8CString.utf8(revision);
  final artifactPtr = dz.NativeUtf8CString.utf8(artifactPath);
  ffi.Pointer<ffi.Char> result = ffi.nullptr;
  try {
    result = native.hfCachePath(
      root.pointer,
      repoPtr.pointer,
      revisionPtr.pointer,
      artifactPtr.pointer,
    );
    if (result == ffi.nullptr) {
      throw StateError('Failed to resolve Hugging Face cache path.');
    }
    final value = result.cast<Utf8>().toDartString();
    if (value.isEmpty) {
      throw StateError('Failed to resolve Hugging Face cache path.');
    }
    return value;
  } finally {
    if (result != ffi.nullptr) native.freeStr(result);
    root.close();
    repoPtr.close();
    revisionPtr.close();
    artifactPtr.close();
  }
}

String _cacheRoot() {
  final result = native.hfCacheRoot();
  if (result == ffi.nullptr) {
    throw StateError('Failed to resolve Hugging Face cache root.');
  }
  try {
    final value = result.cast<Utf8>().toDartString();
    if (value.isEmpty) {
      throw StateError('Failed to resolve Hugging Face cache root.');
    }
    return value;
  } finally {
    native.freeStr(result);
  }
}

String? _authToken() {
  final result = native.hfToken();
  if (result == ffi.nullptr) {
    return null;
  }
  try {
    final value = result.cast<Utf8>().toDartString();
    return value.isEmpty ? null : value;
  } finally {
    native.freeStr(result);
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
  }) : cacheRoot = cacheRoot ?? _cacheRoot(),
       endpoint = Uri.parse(endpoint),
       token = token ?? _authToken();

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
      _cachePath(cacheRoot, ref.repo, ref.revision, filePath),
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

  String _localArtifactPath(HuggingFaceArtifactRef ref) =>
      _cachePath(cacheRoot, ref.repo, ref.revision, ref.path);

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
  final value = dz.NativeUtf8CString.utf8(path);
  try {
    return native.hfDirArtifact(value.pointer) != 0;
  } finally {
    value.close();
  }
}

bool _isRedirect(int statusCode) {
  return statusCode == HttpStatus.movedPermanently ||
      statusCode == HttpStatus.found ||
      statusCode == HttpStatus.seeOther ||
      statusCode == HttpStatus.temporaryRedirect ||
      statusCode == HttpStatus.permanentRedirect;
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
