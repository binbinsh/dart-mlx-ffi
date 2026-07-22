import 'dart:convert';
import 'dart:math' as math;
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

import 'onnx_server_preflight.dart';
import 'package:gridheap_model_contracts/model_spec.dart';
import 'package:gridheap_model_contracts/runtime_metadata.dart';

Future<void> main(List<String> args) async {
  final parsed = _Args(args);
  final modelPath = parsed.option('model', required: true)!;
  final defaultSessionId = parsed.option('session-id') ?? 'default';
  final protocol = (parsed.option('protocol') ?? 'jsonl').trim().toLowerCase();
  final provider = parsed.option('provider');
  final preferCpu = parsed.flag('prefer-cpu');
  final requireProvider = parsed.flag('require-provider');
  final trtFp16 = parsed.flag('trt-fp16');
  final backendOptionsExtra = _backendOptionsFromArgs(parsed, provider);
  final runtimeRoot = onnxRuntimeRoot(
    explicitRuntimeRoot: parsed.option('runtime-root'),
    explicitRoot: parsed.option('root'),
  );
  final dependencySearchDirs = onnxDependencySearchDirs(
    cudaLibraryDirs: parsed.values('cuda-library-dir'),
    nativeLibraryDirs: parsed.values('native-library-dir'),
    libraryDirs: parsed.values('library-dir'),
  );
  final deviceId = parsed.option('device-id') == null
      ? null
      : int.tryParse(parsed.option('device-id')!);
  final numThreads = parsed.option('num-threads') == null
      ? null
      : int.tryParse(parsed.option('num-threads')!);

  late final _LoadedSession loaded;
  try {
    loaded = _loadSession(
      sessionId: defaultSessionId,
      modelPath: modelPath,
      provider: provider,
      preferCpu: preferCpu,
      requireProvider: requireProvider,
      trtFp16: trtFp16,
      deviceId: deviceId,
      numThreads: numThreads,
      backendOptionsExtra: backendOptionsExtra,
      runtimeRoot: runtimeRoot,
      dependencySearchDirs: dependencySearchDirs,
    );
  } catch (error, stack) {
    final payload = onnxServerFatalPayload(error, stack);
    if (_isBinaryProtocol(protocol)) {
      await _writeBinaryFrame(payload);
    } else {
      stdout.writeln(jsonEncode(payload));
    }
    if (error is OnnxServerPreflightException) {
      exitCode = 78;
    }
    return;
  }

  final sessions = <String, _LoadedSession>{defaultSessionId: loaded};
  final readyPayload = {
    'type': 'ready',
    'runtime': 'dart_inference',
    'protocol': _isBinaryProtocol(protocol) ? 'binary' : 'jsonl',
    'session_id': defaultSessionId,
    'input_names': loaded.inputNames,
    'output_names': loaded.outputNames,
    'diagnostics': loaded.diagnostics,
  };
  if (_isBinaryProtocol(protocol)) {
    await _serveBinary(
      sessions: sessions,
      defaultSessionId: defaultSessionId,
      readyPayload: readyPayload,
      defaultProvider: provider,
      defaultPreferCpu: preferCpu,
      defaultRequireProvider: requireProvider,
      defaultTrtFp16: trtFp16,
      defaultDeviceId: deviceId,
      defaultNumThreads: numThreads,
      defaultBackendOptionsExtra: backendOptionsExtra,
      defaultRuntimeRoot: runtimeRoot,
      defaultDependencySearchDirs: dependencySearchDirs,
    );
  } else {
    await _serveJsonl(
      sessions: sessions,
      defaultSessionId: defaultSessionId,
      readyPayload: readyPayload,
      defaultProvider: provider,
      defaultPreferCpu: preferCpu,
      defaultRequireProvider: requireProvider,
      defaultTrtFp16: trtFp16,
      defaultDeviceId: deviceId,
      defaultNumThreads: numThreads,
      defaultBackendOptionsExtra: backendOptionsExtra,
      defaultRuntimeRoot: runtimeRoot,
      defaultDependencySearchDirs: dependencySearchDirs,
    );
  }
  for (final item in sessions.values) {
    item.session.close();
  }
}

bool _isBinaryProtocol(String protocol) =>
    protocol == 'binary' || protocol == 'bin' || protocol == 'frame';

final class _LoadedSession {
  _LoadedSession({
    required this.sessionId,
    required this.session,
    required this.diagnostics,
    required this.inputNames,
    required this.outputNames,
  });

  final String sessionId;
  final ModelSession session;
  final Map<String, Object?> diagnostics;
  final List<String> inputNames;
  final List<String> outputNames;
}

_LoadedSession _loadSession({
  required String sessionId,
  required String modelPath,
  required String? provider,
  required bool preferCpu,
  required bool requireProvider,
  required bool trtFp16,
  required int? deviceId,
  required int? numThreads,
  required Map<String, Object?> backendOptionsExtra,
  required String? runtimeRoot,
  required List<String> dependencySearchDirs,
}) {
  preflightOnnxProvider(
    provider: provider,
    requireProvider: requireProvider,
    runtimeRoot: runtimeRoot,
    dependencySearchDirs: dependencySearchDirs,
  );
  final spec = ModelSpec(
    id: 'unifrontend_onnx_bridge_$sessionId',
    family: 'unifrontend_onnx_bridge',
    modalities: const [ModelModality.textGeneration],
    platformArtifacts: {
      RuntimeEngine.onnx: RuntimeArtifact(
        engine: RuntimeEngine.onnx,
        path: modelPath,
        targetPlatforms: [RuntimePlatformCurrent.current().name],
      ),
    },
  );
  final backendOptions = <String, Object?>{
    if (provider != null && provider.isNotEmpty) 'provider': provider,
    if (deviceId != null && deviceId >= 0) 'deviceId': deviceId,
    if (requireProvider) 'requireProvider': true,
    if (trtFp16) 'trtFp16': true,
    ...backendOptionsExtra,
  };
  final options = RuntimeOptions(
    engine: RuntimeEngine.onnx,
    prefer: preferCpu
        ? const [Accelerator.cpu]
        : const [Accelerator.gpu, Accelerator.cpu],
    diagnostics: true,
    numThreads: numThreads,
    backendOptions: backendOptions,
  );
  final session = RuntimeRegistry.native().load(spec, options: options);
  final diagnostics = session.diagnostics;
  final inputNames =
      (diagnostics['input_names'] as List<dynamic>? ?? const <dynamic>[])
          .map((value) => value.toString())
          .toList(growable: false);
  final outputNames =
      (diagnostics['output_names'] as List<dynamic>? ?? const <dynamic>[])
          .map((value) => value.toString())
          .toList(growable: false);
  return _LoadedSession(
    sessionId: sessionId,
    session: session,
    diagnostics: diagnostics,
    inputNames: inputNames,
    outputNames: outputNames,
  );
}

_LoadedSession _loadSessionFromPayload({
  required Map<String, dynamic> payload,
  required String? defaultProvider,
  required bool defaultPreferCpu,
  required bool defaultRequireProvider,
  required bool defaultTrtFp16,
  required int? defaultDeviceId,
  required int? defaultNumThreads,
  required Map<String, Object?> defaultBackendOptionsExtra,
  required String? defaultRuntimeRoot,
  required List<String> defaultDependencySearchDirs,
}) {
  final sessionId = (payload['session_id'] ?? '').toString().trim();
  final modelPath = (payload['model'] ?? '').toString().trim();
  if (sessionId.isEmpty) {
    throw FormatException('load_model requires session_id');
  }
  if (modelPath.isEmpty) {
    throw FormatException('load_model requires model');
  }
  final provider = (payload['provider'] ?? defaultProvider)?.toString();
  final preferCpu = _boolPayload(payload['prefer_cpu'], defaultPreferCpu);
  final requireProvider = _boolPayload(
    payload['require_provider'],
    defaultRequireProvider,
  );
  final trtFp16 = _boolPayload(payload['trt_fp16'], defaultTrtFp16);
  final deviceId = _intPayload(payload['device_id'], defaultDeviceId);
  final numThreads = _intPayload(payload['num_threads'], defaultNumThreads);
  final backendOptionsExtra = {
    ...defaultBackendOptionsExtra,
    ..._backendOptionsFromPayload(payload),
  };
  final runtimeRoot = onnxRuntimeRootFromPayload(payload, defaultRuntimeRoot);
  final dependencySearchDirs = [
    ...defaultDependencySearchDirs,
    ...onnxDependencySearchDirsFromPayload(payload),
  ];
  return _loadSession(
    sessionId: sessionId,
    modelPath: modelPath,
    provider: provider,
    preferCpu: preferCpu,
    requireProvider: requireProvider,
    trtFp16: trtFp16,
    deviceId: deviceId,
    numThreads: numThreads,
    backendOptionsExtra: backendOptionsExtra,
    runtimeRoot: runtimeRoot,
    dependencySearchDirs: dependencySearchDirs,
  );
}

Map<String, Object?> _loadedPayload(
  _LoadedSession loaded, {
  required Object? requestId,
}) => {
  'type': 'loaded',
  'id': requestId,
  'session_id': loaded.sessionId,
  'input_names': loaded.inputNames,
  'output_names': loaded.outputNames,
  'diagnostics': loaded.diagnostics,
};

bool _boolPayload(Object? value, bool fallback) {
  if (value == null) {
    return fallback;
  }
  if (value is bool) {
    return value;
  }
  final normalized = value.toString().trim().toLowerCase();
  if (normalized.isEmpty) {
    return fallback;
  }
  return const {'1', 'true', 'yes', 'on'}.contains(normalized);
}

int? _intPayload(Object? value, int? fallback) {
  if (value == null) {
    return fallback;
  }
  return int.tryParse(value.toString()) ?? fallback;
}

Map<String, Object?> _backendOptionsFromArgs(_Args parsed, String? provider) {
  final preloadLibraries = discoverDefaultOnnxRuntimePreloadLibraries(
    explicitLibraries: parsed.values('preload-library'),
    libraryDirectories: [
      ...parsed.values('cuda-library-dir'),
      ...parsed.values('native-library-dir'),
    ],
    libraryNames: provider == null
        ? null
        : onnxRuntimePreloadLibraryNamesForProvider(provider),
  );
  final trtCacheDir = parsed.option('trt-cache-dir');
  return {
    if (trtCacheDir != null && trtCacheDir.isNotEmpty)
      'trtCacheDir': trtCacheDir,
    if (parsed.option('cuda-memory-limit-mb') != null)
      'cudaMemoryLimitMb': int.tryParse(parsed.option('cuda-memory-limit-mb')!),
    if (parsed.option('trt-workspace-mb') != null)
      'trtWorkspaceMemoryLimitMb': int.tryParse(
        parsed.option('trt-workspace-mb')!,
      ),
    if (parsed.option('trt-min-subgraph-size') != null)
      'trtMinSubgraphSize': int.tryParse(
        parsed.option('trt-min-subgraph-size')!,
      ),
    if (preloadLibraries.isNotEmpty)
      'preloadLibraries': encodeOnnxRuntimePreloadLibraries(preloadLibraries),
  };
}

Map<String, Object?> _backendOptionsFromPayload(Map<String, dynamic> payload) {
  final preload = payload['preload_libraries'] ?? payload['preloadLibraries'];
  return {
    if (payload['trt_cache_dir'] != null)
      'trtCacheDir': payload['trt_cache_dir'].toString(),
    if (payload['trtCacheDir'] != null)
      'trtCacheDir': payload['trtCacheDir'].toString(),
    if (_intPayload(payload['cuda_memory_limit_mb'], null) != null)
      'cudaMemoryLimitMb': _intPayload(payload['cuda_memory_limit_mb'], null),
    if (_intPayload(payload['cudaMemoryLimitMb'], null) != null)
      'cudaMemoryLimitMb': _intPayload(payload['cudaMemoryLimitMb'], null),
    if (_intPayload(payload['trt_workspace_mb'], null) != null)
      'trtWorkspaceMemoryLimitMb': _intPayload(
        payload['trt_workspace_mb'],
        null,
      ),
    if (_intPayload(payload['trtWorkspaceMemoryLimitMb'], null) != null)
      'trtWorkspaceMemoryLimitMb': _intPayload(
        payload['trtWorkspaceMemoryLimitMb'],
        null,
      ),
    if (_intPayload(payload['trt_min_subgraph_size'], null) != null)
      'trtMinSubgraphSize': _intPayload(payload['trt_min_subgraph_size'], null),
    if (_intPayload(payload['trtMinSubgraphSize'], null) != null)
      'trtMinSubgraphSize': _intPayload(payload['trtMinSubgraphSize'], null),
    if (_preloadLibrariesFromPayload(preload).isNotEmpty)
      'preloadLibraries': encodeOnnxRuntimePreloadLibraries(
        _preloadLibrariesFromPayload(preload),
      ),
  };
}

List<String> _preloadLibrariesFromPayload(Object? value) {
  if (value is String) {
    return value
        .split(RegExp(r'[:,;\n\r]+'))
        .map((part) => part.trim())
        .where((part) => part.isNotEmpty)
        .toList(growable: false);
  }
  if (value is List) {
    return value
        .map((item) => item.toString().trim())
        .where((part) => part.isNotEmpty)
        .toList(growable: false);
  }
  return const [];
}

Future<void> _serveJsonl({
  required Map<String, _LoadedSession> sessions,
  required String defaultSessionId,
  required Map<String, Object?> readyPayload,
  required String? defaultProvider,
  required bool defaultPreferCpu,
  required bool defaultRequireProvider,
  required bool defaultTrtFp16,
  required int? defaultDeviceId,
  required int? defaultNumThreads,
  required Map<String, Object?> defaultBackendOptionsExtra,
  required String? defaultRuntimeRoot,
  required List<String> defaultDependencySearchDirs,
}) async {
  stdout.writeln(jsonEncode(readyPayload));
  await for (final line
      in stdin.transform(utf8.decoder).transform(const LineSplitter())) {
    if (line.trim().isEmpty) {
      continue;
    }
    final dynamic payload;
    try {
      payload = jsonDecode(line);
    } catch (error) {
      stdout.writeln(
        jsonEncode({
          'type': 'error',
          'error': 'invalid_json',
          'detail': '$error',
        }),
      );
      continue;
    }
    if (payload is! Map<String, dynamic>) {
      stdout.writeln(
        jsonEncode({
          'type': 'error',
          'error': 'invalid_payload',
          'detail': 'Payload must be an object.',
        }),
      );
      continue;
    }
    final requestId = payload['id'];
    if ((payload['cmd'] ?? '') == 'close') {
      stdout.writeln(jsonEncode({'type': 'closed', 'id': requestId}));
      return;
    }
    try {
      if ((payload['cmd'] ?? '') == 'load_model') {
        final loaded = _loadSessionFromPayload(
          payload: payload,
          defaultProvider: defaultProvider,
          defaultPreferCpu: defaultPreferCpu,
          defaultRequireProvider: defaultRequireProvider,
          defaultTrtFp16: defaultTrtFp16,
          defaultDeviceId: defaultDeviceId,
          defaultNumThreads: defaultNumThreads,
          defaultBackendOptionsExtra: defaultBackendOptionsExtra,
          defaultRuntimeRoot: defaultRuntimeRoot,
          defaultDependencySearchDirs: defaultDependencySearchDirs,
        );
        final existing = sessions[loaded.sessionId];
        if (existing != null) {
          existing.session.close();
        }
        sessions[loaded.sessionId] = loaded;
        stdout.writeln(
          jsonEncode(_loadedPayload(loaded, requestId: requestId)),
        );
        continue;
      }
      final inputMap = _decodeInputs(payload['inputs']);
      final result = _executeRequest(
        sessions: sessions,
        defaultSessionId: defaultSessionId,
        payload: payload,
        inputs: inputMap,
      );
      final encoded = _encodeOutputs(result.outputs);
      stdout.writeln(
        jsonEncode({
          'type': 'result',
          'id': requestId,
          'outputs': encoded,
          'diagnostics': result.diagnostics,
        }),
      );
    } catch (error, stack) {
      stdout.writeln(
        jsonEncode(onnxServerErrorPayload(requestId, error, stack)),
      );
    }
  }
}

Future<void> _serveBinary({
  required Map<String, _LoadedSession> sessions,
  required String defaultSessionId,
  required Map<String, Object?> readyPayload,
  required String? defaultProvider,
  required bool defaultPreferCpu,
  required bool defaultRequireProvider,
  required bool defaultTrtFp16,
  required int? defaultDeviceId,
  required int? defaultNumThreads,
  required Map<String, Object?> defaultBackendOptionsExtra,
  required String? defaultRuntimeRoot,
  required List<String> defaultDependencySearchDirs,
}) async {
  await _writeBinaryFrame(readyPayload);
  final parser = _BinaryFrameParser();
  await for (final chunk in stdin) {
    final frames = parser.push(chunk);
    for (final frame in frames) {
      Map<String, dynamic> payload;
      try {
        final decoded = jsonDecode(utf8.decode(frame.headerBytes));
        if (decoded is! Map) {
          throw FormatException('binary frame header must be an object');
        }
        payload = Map<String, dynamic>.from(
          decoded.map((k, v) => MapEntry(k.toString(), v)),
        );
      } catch (error, stack) {
        await _writeBinaryFrame({
          'type': 'error',
          'error': 'invalid_binary_header',
          'detail': '$error',
          'stack': '$stack',
        });
        continue;
      }

      final requestId = payload['id'];
      if ((payload['cmd'] ?? '') == 'close') {
        await _writeBinaryFrame({'type': 'closed', 'id': requestId});
        return;
      }
      try {
        if ((payload['cmd'] ?? '') == 'load_model') {
          final loaded = _loadSessionFromPayload(
            payload: payload,
            defaultProvider: defaultProvider,
            defaultPreferCpu: defaultPreferCpu,
            defaultRequireProvider: defaultRequireProvider,
            defaultTrtFp16: defaultTrtFp16,
            defaultDeviceId: defaultDeviceId,
            defaultNumThreads: defaultNumThreads,
            defaultBackendOptionsExtra: defaultBackendOptionsExtra,
            defaultRuntimeRoot: defaultRuntimeRoot,
            defaultDependencySearchDirs: defaultDependencySearchDirs,
          );
          final existing = sessions[loaded.sessionId];
          if (existing != null) {
            existing.session.close();
          }
          sessions[loaded.sessionId] = loaded;
          await _writeBinaryFrame(_loadedPayload(loaded, requestId: requestId));
          continue;
        }
        final inputMap = _decodeBinaryInputs(
          payload['inputs'],
          frame.bodyBytes,
        );
        final result = _executeRequest(
          sessions: sessions,
          defaultSessionId: defaultSessionId,
          payload: payload,
          inputs: inputMap,
        );
        final requestedOutputNames = _decodeOutputNames(
          payload['output_names'],
          result.fallbackOutputNames,
        );
        final encoded = _encodeBinaryOutputs(
          result.outputs,
          requestedOutputNames,
        );
        await _writeBinaryFrame({
          'type': 'result',
          'id': requestId,
          'outputs': encoded.specs,
          'diagnostics': result.diagnostics,
        }, bodyChunks: encoded.chunks);
      } catch (error, stack) {
        await _writeBinaryFrame(
          onnxServerErrorPayload(requestId, error, stack),
        );
      }
    }
  }
}

List<String> _decodeOutputNames(Object? raw, List<String> fallback) {
  if (raw is! List) {
    return fallback;
  }
  final values = raw
      .map((value) => value.toString())
      .where((value) => value.isNotEmpty)
      .toList(growable: false);
  return values.isEmpty ? fallback : values;
}

Map<String, Object?> _decodeInputs(Object? raw) {
  if (raw is! Map) {
    throw FormatException('inputs must be a map');
  }
  final out = <String, Object?>{};
  for (final entry in raw.entries) {
    final key = entry.key.toString();
    final value = entry.value;
    if (value is! Map) {
      throw FormatException('input $key must be an object');
    }
    final object = Map<String, Object?>.from(
      value.map((k, v) => MapEntry(k.toString(), v)),
    );
    out[key] = _decodeTensor(object, key);
  }
  return out;
}

RuntimeTensor _decodeTensor(Map<String, Object?> spec, String key) {
  final dtype = (spec['dtype'] ?? '').toString().trim().toLowerCase();
  final shapeRaw = spec['shape'];
  final base64Value = spec['base64'];
  if (shapeRaw is! List) {
    throw FormatException('input $key missing shape');
  }
  if (base64Value is! String || base64Value.isEmpty) {
    throw FormatException('input $key missing base64');
  }
  final shape = shapeRaw.map((value) => int.parse(value.toString())).toList();
  final bytes = Uint8List.fromList(base64Decode(base64Value));
  return _tensorFromWire(dtype: dtype, shape: shape, bytes: bytes, key: key);
}

Map<String, Object?> _encodeOutputs(Map<String, Object?> values) {
  final out = <String, Object?>{};
  for (final entry in values.entries) {
    if (entry.value is! RuntimeTensor) {
      continue;
    }
    final tensor = entry.value as RuntimeTensor;
    out[entry.key] = {
      'dtype': tensor.dtype.name,
      'shape': tensor.shape,
      'base64': base64Encode(tensor.bytes),
    };
  }
  return out;
}

final class _BinaryFrame {
  _BinaryFrame({required this.headerBytes, required this.bodyBytes});

  final Uint8List headerBytes;
  final Uint8List bodyBytes;
}

final class _BinaryFrameParser {
  final Uint8List _prefix = Uint8List(12);
  int _prefixOffset = 0;
  Uint8List? _header;
  int _headerOffset = 0;
  Uint8List? _body;
  int _bodyOffset = 0;

  List<_BinaryFrame> push(List<int> chunk) {
    final frames = <_BinaryFrame>[];
    var offset = 0;
    while (true) {
      if (offset >= chunk.length) {
        break;
      }
      if (_prefixOffset < _prefix.length) {
        final copied = _copyChunk(
          src: chunk,
          srcOffset: offset,
          dst: _prefix,
          dstOffset: _prefixOffset,
          maxLen: _prefix.length - _prefixOffset,
        );
        offset += copied;
        _prefixOffset += copied;
        if (_prefixOffset < _prefix.length) {
          break;
        }
        if (_prefix[0] != 0x44 ||
            _prefix[1] != 0x4d ||
            _prefix[2] != 0x46 ||
            _prefix[3] != 0x31) {
          throw FormatException('invalid binary frame magic');
        }
        final prefixData = ByteData.sublistView(_prefix);
        final headerLen = prefixData.getUint32(4, Endian.little);
        final bodyLen = prefixData.getUint32(8, Endian.little);
        _header = Uint8List(headerLen);
        _headerOffset = 0;
        _body = Uint8List(bodyLen);
        _bodyOffset = 0;
      }
      if (_header == null || _body == null) {
        throw StateError('binary frame parser internal state is invalid');
      }
      if (_headerOffset < _header!.length) {
        final copied = _copyChunk(
          src: chunk,
          srcOffset: offset,
          dst: _header!,
          dstOffset: _headerOffset,
          maxLen: _header!.length - _headerOffset,
        );
        offset += copied;
        _headerOffset += copied;
        if (_headerOffset < _header!.length) {
          break;
        }
      }
      if (_bodyOffset < _body!.length) {
        final copied = _copyChunk(
          src: chunk,
          srcOffset: offset,
          dst: _body!,
          dstOffset: _bodyOffset,
          maxLen: _body!.length - _bodyOffset,
        );
        offset += copied;
        _bodyOffset += copied;
        if (_bodyOffset < _body!.length) {
          break;
        }
      }
      frames.add(_BinaryFrame(headerBytes: _header!, bodyBytes: _body!));
      _reset();
    }
    return frames;
  }

  void _reset() {
    _prefixOffset = 0;
    _header = null;
    _headerOffset = 0;
    _body = null;
    _bodyOffset = 0;
  }

  int _copyChunk({
    required List<int> src,
    required int srcOffset,
    required Uint8List dst,
    required int dstOffset,
    required int maxLen,
  }) {
    final available = src.length - srcOffset;
    if (available <= 0 || maxLen <= 0) {
      return 0;
    }
    final copied = math.min(available, maxLen);
    dst.setRange(dstOffset, dstOffset + copied, src, srcOffset);
    return copied;
  }
}

Future<void> _writeBinaryFrame(
  Map<String, Object?> header, {
  List<Uint8List> bodyChunks = const <Uint8List>[],
}) async {
  var bodyLen = 0;
  for (final chunk in bodyChunks) {
    bodyLen += chunk.length;
  }
  final headerBytes = Uint8List.fromList(utf8.encode(jsonEncode(header)));
  final prefix = Uint8List(12);
  prefix.setAll(0, const [0x44, 0x4d, 0x46, 0x31]); // "DMF1"
  final byteData = ByteData.sublistView(prefix);
  byteData.setUint32(4, headerBytes.length, Endian.little);
  byteData.setUint32(8, bodyLen, Endian.little);
  stdout.add(prefix);
  stdout.add(headerBytes);
  for (final chunk in bodyChunks) {
    if (chunk.isNotEmpty) {
      stdout.add(chunk);
    }
  }
  await stdout.flush();
}

_RunResult _executeRequest({
  required Map<String, _LoadedSession> sessions,
  required String defaultSessionId,
  required Map<String, dynamic> payload,
  required Map<String, Object?> inputs,
}) {
  final sessionId = (payload['session_id'] ?? defaultSessionId).toString();
  final loaded = sessions[sessionId];
  if (loaded == null) {
    throw StateError('Unknown ONNX session_id: $sessionId');
  }
  final requestedOutputNames = _decodeOutputNames(
    payload['output_names'],
    loaded.outputNames,
  );
  final chunkedMode = (payload['mode'] ?? '').toString() == 'chunked';
  if (!chunkedMode) {
    final outputs = loaded.session.run(ModelInputs(inputs));
    return _RunResult(outputs.values, outputs.diagnostics, loaded.outputNames);
  }
  final fixedBatchRaw = payload['fixed_batch'];
  final fixedBatch = int.tryParse('${fixedBatchRaw ?? ''}') ?? 0;
  if (fixedBatch <= 0) {
    throw FormatException('chunked mode requires positive fixed_batch');
  }
  return _runChunked(
    session: loaded.session,
    inputs: inputs,
    outputNames: requestedOutputNames,
    fixedBatch: fixedBatch,
    fallbackOutputNames: loaded.outputNames,
  );
}

Map<String, Object?> _decodeBinaryInputs(Object? raw, Uint8List bodyBytes) {
  if (raw is! List) {
    throw FormatException('binary inputs must be a list');
  }
  final out = <String, Object?>{};
  var offset = 0;
  for (final item in raw) {
    if (item is! Map) {
      throw FormatException('binary input spec must be an object');
    }
    final spec = Map<String, Object?>.from(
      item.map((k, v) => MapEntry(k.toString(), v)),
    );
    final name = (spec['name'] ?? '').toString();
    if (name.isEmpty) {
      throw FormatException('binary input spec missing name');
    }
    final dtype = (spec['dtype'] ?? '').toString();
    final shapeRaw = spec['shape'];
    if (shapeRaw is! List) {
      throw FormatException('binary input $name missing shape');
    }
    final shape = shapeRaw.map((value) => int.parse(value.toString())).toList();
    final nbytes = int.tryParse('${spec['nbytes'] ?? ''}') ?? -1;
    if (nbytes < 0) {
      throw FormatException('binary input $name missing nbytes');
    }
    if (offset + nbytes > bodyBytes.length) {
      throw RangeError(
        'binary input $name exceeds frame body size '
        '(offset=$offset nbytes=$nbytes total=${bodyBytes.length})',
      );
    }
    final tensorBytes = Uint8List.sublistView(
      bodyBytes,
      offset,
      offset + nbytes,
    );
    out[name] = _tensorFromWire(
      dtype: dtype,
      shape: shape,
      bytes: tensorBytes,
      key: name,
    );
    offset += nbytes;
  }
  if (offset != bodyBytes.length) {
    throw FormatException(
      'binary input frame has trailing bytes: '
      'used=$offset total=${bodyBytes.length}',
    );
  }
  return out;
}

final class _BinaryOutputs {
  _BinaryOutputs(this.specs, this.chunks);

  final List<Map<String, Object?>> specs;
  final List<Uint8List> chunks;
}

_BinaryOutputs _encodeBinaryOutputs(
  Map<String, Object?> values,
  List<String> outputNames,
) {
  final specs = <Map<String, Object?>>[];
  final chunks = <Uint8List>[];
  for (final name in outputNames) {
    final value = values[name];
    if (value is! RuntimeTensor) {
      continue;
    }
    final bytes = value.bytes;
    specs.add({
      'name': name,
      'dtype': value.dtype.name,
      'shape': value.shape,
      'nbytes': bytes.length,
    });
    chunks.add(bytes);
  }
  return _BinaryOutputs(specs, chunks);
}

RuntimeTensor _tensorFromWire({
  required String dtype,
  required List<int> shape,
  required Uint8List bytes,
  required String key,
}) {
  final normalized = dtype.trim().toLowerCase();
  switch (normalized) {
    case 'float32':
      return RuntimeTensor(
        dtype: RuntimeTensorDataType.float32,
        shape: shape,
        bytes: bytes,
      );
    case 'float64':
      return RuntimeTensor(
        dtype: RuntimeTensorDataType.float64,
        shape: shape,
        bytes: bytes,
      );
    case 'float16':
      return RuntimeTensor(
        dtype: RuntimeTensorDataType.float16,
        shape: shape,
        bytes: bytes,
      );
    case 'int64':
      return RuntimeTensor(
        dtype: RuntimeTensorDataType.int64,
        shape: shape,
        bytes: bytes,
      );
    case 'int32':
      return RuntimeTensor(
        dtype: RuntimeTensorDataType.int32,
        shape: shape,
        bytes: bytes,
      );
    case 'uint8':
      return RuntimeTensor(
        dtype: RuntimeTensorDataType.uint8,
        shape: shape,
        bytes: bytes,
      );
    case 'bool':
    case 'boolean':
      return RuntimeTensor(
        dtype: RuntimeTensorDataType.boolean,
        shape: shape,
        bytes: bytes,
      );
    default:
      throw FormatException('input $key has unsupported dtype: $dtype');
  }
}

final class _RunResult {
  _RunResult(this.outputs, this.diagnostics, this.fallbackOutputNames);

  final Map<String, Object?> outputs;
  final Map<String, Object?> diagnostics;
  final List<String> fallbackOutputNames;
}

_RunResult _runChunked({
  required ModelSession session,
  required Map<String, Object?> inputs,
  required List<String> outputNames,
  required int fixedBatch,
  required List<String> fallbackOutputNames,
}) {
  RuntimeTensor? firstTensor;
  for (final value in inputs.values) {
    if (value is RuntimeTensor) {
      firstTensor = value;
      break;
    }
  }
  if (firstTensor == null) {
    throw StateError('chunked run requires at least one tensor input');
  }
  if (firstTensor.shape.isEmpty) {
    throw StateError('chunked run requires batched tensor inputs');
  }
  final totalRows = firstTensor.shape[0];
  if (totalRows <= 0) {
    throw StateError('chunked run requires positive batch size');
  }

  final mergedBuffers = <String, Uint8List>{};
  final writeOffsets = <String, int>{};
  final rowBytesByName = <String, int>{};
  final dtypes = <String, RuntimeTensorDataType>{};
  final tails = <String, List<int>>{};
  Map<String, Object?>? lastDiagnostics;

  for (var start = 0; start < totalRows; start += fixedBatch) {
    final end = math.min(totalRows, start + fixedBatch);
    final chunkRows = end - start;
    final chunkInputs = <String, Object?>{};
    for (final entry in inputs.entries) {
      final value = entry.value;
      if (value is! RuntimeTensor) {
        chunkInputs[entry.key] = value;
        continue;
      }
      var tensor = _sliceRows(value, start, end);
      if (chunkRows < fixedBatch) {
        tensor = _padRows(tensor, fixedBatch);
      }
      chunkInputs[entry.key] = tensor;
    }

    final outputs = session.run(ModelInputs(chunkInputs));
    lastDiagnostics = outputs.diagnostics;
    for (final name in outputNames) {
      final dynamic value = outputs.values[name];
      if (value is! RuntimeTensor) {
        continue;
      }
      final rowBytes = _rowBytes(value);
      final keepBytes = chunkRows * rowBytes;
      final existingRowBytes = rowBytesByName[name];
      if (existingRowBytes != null && existingRowBytes != rowBytes) {
        throw StateError(
          'chunked output row bytes changed for $name: '
          '$existingRowBytes -> $rowBytes',
        );
      }
      rowBytesByName[name] ??= rowBytes;
      final buffer = mergedBuffers.putIfAbsent(
        name,
        () => Uint8List(totalRows * rowBytes),
      );
      final writeOffset = writeOffsets[name] ?? 0;
      if (writeOffset + keepBytes > buffer.length) {
        throw RangeError(
          'chunked output overflow for $name: '
          'offset=$writeOffset keep=$keepBytes total=${buffer.length}',
        );
      }
      buffer.setRange(writeOffset, writeOffset + keepBytes, value.bytes, 0);
      writeOffsets[name] = writeOffset + keepBytes;
      dtypes[name] ??= value.dtype;
      tails[name] ??= value.shape.skip(1).toList(growable: false);
    }
  }

  final merged = <String, Object?>{};
  for (final entry in mergedBuffers.entries) {
    final name = entry.key;
    final dtype = dtypes[name];
    if (dtype == null) {
      continue;
    }
    final tail = tails[name] ?? const <int>[];
    merged[name] = RuntimeTensor(
      dtype: dtype,
      shape: <int>[totalRows, ...tail],
      bytes: entry.value,
    );
  }
  return _RunResult(
    merged,
    lastDiagnostics ?? const <String, Object?>{},
    fallbackOutputNames,
  );
}

RuntimeTensor _sliceRows(RuntimeTensor tensor, int start, int end) {
  final rows = tensor.shape[0];
  if (start < 0 || end > rows || start >= end) {
    throw RangeError('invalid row slice: [$start, $end) for rows=$rows');
  }
  final rowBytes = _rowBytes(tensor);
  final begin = start * rowBytes;
  final finish = end * rowBytes;
  return RuntimeTensor(
    dtype: tensor.dtype,
    shape: <int>[end - start, ...tensor.shape.skip(1)],
    bytes: Uint8List.sublistView(tensor.bytes, begin, finish),
  );
}

RuntimeTensor _padRows(RuntimeTensor tensor, int targetRows) {
  final rows = tensor.shape[0];
  if (rows >= targetRows) {
    return tensor;
  }
  if (rows <= 0) {
    throw StateError('cannot pad empty tensor');
  }
  final rowBytes = _rowBytes(tensor);
  final out = Uint8List(targetRows * rowBytes);
  out.setAll(0, tensor.bytes);
  final lastStart = (rows - 1) * rowBytes;
  final lastRow = Uint8List.sublistView(
    tensor.bytes,
    lastStart,
    lastStart + rowBytes,
  );
  for (var row = rows; row < targetRows; row++) {
    out.setRange(row * rowBytes, (row + 1) * rowBytes, lastRow);
  }
  return RuntimeTensor(
    dtype: tensor.dtype,
    shape: <int>[targetRows, ...tensor.shape.skip(1)],
    bytes: out,
  );
}

int _rowBytes(RuntimeTensor tensor) {
  if (tensor.shape.isEmpty) {
    throw StateError('scalar tensor has no row stride');
  }
  final elementBytes = _dtypeBytes(tensor.dtype);
  var elemsPerRow = 1;
  for (var i = 1; i < tensor.shape.length; i++) {
    elemsPerRow *= tensor.shape[i];
  }
  return elemsPerRow * elementBytes;
}

int _dtypeBytes(RuntimeTensorDataType dtype) {
  switch (dtype) {
    case RuntimeTensorDataType.float64:
    case RuntimeTensorDataType.int64:
      return 8;
    case RuntimeTensorDataType.float32:
    case RuntimeTensorDataType.int32:
      return 4;
    case RuntimeTensorDataType.float16:
      return 2;
    case RuntimeTensorDataType.uint8:
    case RuntimeTensorDataType.boolean:
      return 1;
  }
}

final class _Args {
  _Args(List<String> args) : this._(_parseAll(args));

  _Args._((Map<String, String?>, Map<String, List<String>>) parsed)
    : _values = parsed.$1,
      _allValues = parsed.$2;

  final Map<String, String?> _values;
  final Map<String, List<String>> _allValues;

  bool flag(String name) => _values.containsKey(name);

  String? option(String name, {bool required = false}) {
    final value = _values[name];
    if (required && (value == null || value.isEmpty)) {
      throw ArgumentError('Missing --$name');
    }
    return value;
  }

  List<String> values(String name) => _allValues[name] ?? const [];

  static (Map<String, String?>, Map<String, List<String>>) _parseAll(
    List<String> args,
  ) {
    final values = <String, String?>{};
    final allValues = <String, List<String>>{};
    for (var i = 0; i < args.length; i++) {
      final arg = args[i];
      if (!arg.startsWith('--')) {
        throw ArgumentError('Unexpected positional argument: $arg');
      }
      final name = arg.substring(2);
      if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
        final value = args[++i];
        values[name] = value;
        allValues.putIfAbsent(name, () => []).add(value);
      } else {
        values[name] = null;
        allValues.putIfAbsent(name, () => []);
      }
    }
    return (values, allValues);
  }
}
