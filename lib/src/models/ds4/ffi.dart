library;

import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;

import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as p;

import '../../runtime/artifact_resolver.dart';
import 'ds4.dart';
import 'source.dart';

enum Ds4NativeBackend { metal, cpu }

final class Ds4FfiConfig {
  const Ds4FfiConfig({
    this.libraryPath = '',
    this.sourceDir = '',
    this.modelPath = '',
    this.quant = Ds4Quant.q2,
    this.resolveModelArtifact = true,
    this.mtpPath = '',
    this.enableMtp = false,
    this.contextLength = 100000,
    this.backend = Ds4NativeBackend.metal,
    this.threads = 0,
    this.warmWeights = false,
    this.quality = false,
    this.mtpDraftTokens = 0,
    this.mtpMargin = 0,
    this.token,
    this.artifactResolver,
  });

  final String libraryPath;
  final String sourceDir;
  final String modelPath;
  final Ds4Quant quant;
  final bool resolveModelArtifact;
  final String mtpPath;
  final bool enableMtp;
  final int contextLength;
  final Ds4NativeBackend backend;
  final int threads;
  final bool warmWeights;
  final bool quality;
  final int mtpDraftTokens;
  final double mtpMargin;
  final String? token;
  final RuntimeArtifactResolver? artifactResolver;
}

final class Ds4FfiBackend implements Ds4ChatBackend {
  Ds4FfiBackend(this.config);

  final Ds4FfiConfig config;
  _Ds4Native? _native;
  Pointer<Void>? _engine;
  Pointer<Void>? _session;
  bool _generating = false;
  final _Libc _libc = _Libc();

  @override
  Stream<Ds4ChatChunk> stream(Ds4ChatRequest request) async* {
    if (_generating) {
      throw StateError('Ds4FfiBackend does not support concurrent streams.');
    }
    _generating = true;
    try {
      await _ensureReady();
      yield* _streamReady(request);
    } finally {
      _generating = false;
    }
  }

  @override
  Future<void> close() async {
    final native = _native;
    final session = _session;
    final engine = _engine;
    _session = null;
    _engine = null;
    if (native != null && session != null && session.address != 0) {
      native.sessionFree(session);
    }
    if (native != null && engine != null && engine.address != 0) {
      native.engineClose(engine);
    }
  }

  Future<void> _ensureReady() async {
    if (_engine != null && _session != null) {
      return;
    }
    final sourceDir = await resolveDs4SourceDir(
      explicitSourceDir: config.sourceDir,
    );
    final libraryPath = await _resolveLibraryPath(sourceDir);
    if (sourceDir != null && sourceDir.isNotEmpty) {
      _setMetalSourceEnvironment(sourceDir);
    }
    final native = _Ds4Native(DynamicLibrary.open(libraryPath));
    final modelPath = await _resolveModelPath();
    final mtpPath = await _resolveMtpPath();

    final modelPtr = modelPath.toNativeUtf8();
    final mtpPtr = mtpPath == null ? nullptr : mtpPath.toNativeUtf8();
    final options = calloc<_Ds4EngineOptions>();
    final engineOut = calloc<Pointer<Void>>();
    final sessionOut = calloc<Pointer<Void>>();
    try {
      options.ref
        ..modelPath = modelPtr
        ..mtpPath = mtpPtr
        ..backend = config.backend == Ds4NativeBackend.cpu ? 1 : 0
        ..threads = config.threads
        ..mtpDraftTokens = config.mtpDraftTokens
        ..mtpMargin = config.mtpMargin
        ..warmWeights = config.warmWeights
        ..quality = config.quality;

      final openResult = native.engineOpen(engineOut, options);
      if (openResult != 0 || engineOut.value.address == 0) {
        throw StateError('ds4_engine_open failed with code $openResult.');
      }
      final createResult = native.sessionCreate(
        sessionOut,
        engineOut.value,
        math.max(1, config.contextLength),
      );
      if (createResult != 0 || sessionOut.value.address == 0) {
        native.engineClose(engineOut.value);
        throw StateError('ds4_session_create failed with code $createResult.');
      }
      _native = native;
      _engine = engineOut.value;
      _session = sessionOut.value;
    } finally {
      malloc.free(modelPtr);
      if (mtpPtr.address != 0) {
        malloc.free(mtpPtr);
      }
      calloc.free(options);
      calloc.free(engineOut);
      calloc.free(sessionOut);
    }
  }

  Stream<Ds4ChatChunk> _streamReady(Ds4ChatRequest request) async* {
    final native = _native!;
    final engine = _engine!;
    final session = _session!;
    final prompt = _buildPrompt(native, engine, request);
    final err = calloc<Char>(256);
    final rng = calloc<Uint64>();
    final accepted = calloc<Int32>(17);
    try {
      final syncResult = native.sessionSync(session, prompt, err, 256);
      if (syncResult != 0) {
        throw StateError('ds4_session_sync failed: ${_errorString(err)}');
      }

      final decoder = _StreamingUtf8Decoder();
      final splitter = _ThinkSplitter(
        reasoning: _thinkingEnabled(native, request.thinking),
      );
      rng.value = request.seed ?? _seed();
      var remaining = _maxGenerationTokens(native, session, request.maxTokens);
      while (remaining > 0) {
        final token = native.sessionSample(
          session,
          request.temperature,
          request.topK ?? 0,
          request.topP ?? 1.0,
          request.minP ?? 0.0,
          rng,
        );
        if (token == native.tokenEos(engine)) {
          break;
        }
        final tokenCount = _evalToken(
          native,
          engine,
          session,
          token,
          remaining,
          accepted,
          err,
        );
        if (tokenCount < 0) {
          throw StateError('ds4 decode failed: ${_errorString(err)}');
        }
        for (var i = 0; i < tokenCount && remaining > 0; i += 1) {
          final acceptedToken = accepted[i];
          if (acceptedToken == native.tokenEos(engine)) {
            remaining = 0;
            break;
          }
          final text = decoder.add(_tokenBytes(native, engine, acceptedToken));
          for (final chunk in splitter.add(text)) {
            yield chunk;
          }
          remaining -= 1;
        }
      }
      for (final chunk in splitter.add(decoder.close(), finish: true)) {
        yield chunk;
      }
      yield const Ds4ChatChunk(kind: Ds4ChatChunkKind.done);
    } finally {
      native.tokensFree(prompt);
      calloc.free(prompt);
      calloc.free(err);
      calloc.free(rng);
      calloc.free(accepted);
    }
  }

  Pointer<_Ds4Tokens> _buildPrompt(
    _Ds4Native native,
    Pointer<Void> engine,
    Ds4ChatRequest request,
  ) {
    final tokens = calloc<_Ds4Tokens>();
    native.chatBegin(engine, tokens);
    final thinkMode = _effectiveThinkMode(native, request.thinking);
    if (thinkMode == _ds4ThinkMax) {
      native.chatAppendMaxEffortPrefix(engine, tokens);
    }
    for (final message in request.messages) {
      _withNativeUtf8(_role(message), (role) {
        _withNativeUtf8(_content(message), (content) {
          native.chatAppendMessage(engine, tokens, role, content);
        });
      });
    }
    native.chatAppendAssistantPrefix(engine, tokens, thinkMode);
    return tokens;
  }

  int _evalToken(
    _Ds4Native native,
    Pointer<Void> engine,
    Pointer<Void> session,
    int token,
    int remaining,
    Pointer<Int32> accepted,
    Pointer<Char> err,
  ) {
    if (native.engineMtpDraftTokens(engine) > 1) {
      return native.sessionEvalSpeculativeArgmax(
        session,
        token,
        remaining,
        native.tokenEos(engine),
        accepted,
        17,
        err,
        256,
      );
    }
    final result = native.sessionEval(session, token, err, 256);
    if (result != 0) {
      return -1;
    }
    accepted[0] = token;
    return 1;
  }

  int _maxGenerationTokens(
    _Ds4Native native,
    Pointer<Void> session,
    int requested,
  ) {
    final room = native.sessionCtx(session) - native.sessionPos(session);
    if (room <= 1) {
      return 0;
    }
    return math.min(math.max(0, requested), room - 1);
  }

  int _effectiveThinkMode(_Ds4Native native, Ds4Thinking thinking) {
    return native.thinkModeForContext(
      _thinkMode(thinking),
      config.contextLength,
    );
  }

  bool _thinkingEnabled(_Ds4Native native, Ds4Thinking thinking) {
    return native.thinkModeEnabled(_effectiveThinkMode(native, thinking));
  }

  List<int> _tokenBytes(_Ds4Native native, Pointer<Void> engine, int token) {
    final len = calloc<Size>();
    try {
      final ptr = native.tokenText(engine, token, len);
      if (ptr.address == 0 || len.value == 0) {
        return const <int>[];
      }
      final bytes = ptr.cast<Uint8>().asTypedList(len.value).toList();
      _libc.free(ptr.cast<Void>());
      return bytes;
    } finally {
      calloc.free(len);
    }
  }

  Future<String> _resolveLibraryPath(String? sourceDir) async {
    final explicit = config.libraryPath.trim();
    if (explicit.isNotEmpty) {
      return _normalizePath(explicit);
    }
    if (sourceDir != null && sourceDir.isNotEmpty) {
      return buildDs4DynamicLibrary(sourceDir);
    }
    if (Platform.isMacOS) {
      return 'libds4.dylib';
    }
    if (Platform.isLinux || Platform.isAndroid) {
      return 'libds4.so';
    }
    if (Platform.isWindows) {
      return 'ds4.dll';
    }
    throw UnsupportedError('ds4 FFI is not supported on this platform.');
  }

  Future<String> _resolveModelPath() async {
    final explicit = config.modelPath.trim();
    if (explicit.isNotEmpty) {
      return _normalizePath(explicit);
    }
    if (!config.resolveModelArtifact) {
      return 'ds4flash.gguf';
    }
    final resolver =
        config.artifactResolver ??
        HuggingFaceArtifactCache(
          token: config.token?.trim().isEmpty == true ? null : config.token,
          timeout: const Duration(minutes: 2),
        );
    final resolved = await resolver.resolve(
      Ds4ModelArtifacts.main(config.quant),
    );
    return _normalizePath(resolved.path);
  }

  Future<String?> _resolveMtpPath() async {
    final explicit = config.mtpPath.trim();
    if (explicit.isNotEmpty) {
      return _normalizePath(explicit);
    }
    if (!config.enableMtp) {
      return null;
    }
    final resolver =
        config.artifactResolver ??
        HuggingFaceArtifactCache(
          token: config.token?.trim().isEmpty == true ? null : config.token,
          timeout: const Duration(minutes: 2),
        );
    final resolved = await resolver.resolve(Ds4ModelArtifacts.mtp);
    return _normalizePath(resolved.path);
  }

  void _setMetalSourceEnvironment(String sourceDir) {
    for (final entry in ds4MetalSourceEnvironment(sourceDir).entries) {
      if (Platform.environment.containsKey(entry.key)) {
        continue;
      }
      _libc.setEnv(entry.key, entry.value);
    }
  }
}

String _role(Map<String, Object?> message) {
  final role = message['role']?.toString().trim();
  return role == null || role.isEmpty ? 'user' : role;
}

String _content(Map<String, Object?> message) {
  final content = message['content'];
  if (content is String) {
    return content;
  }
  if (content is List) {
    return content
        .map(_contentPart)
        .where((part) => part.isNotEmpty)
        .join('\n');
  }
  return content?.toString() ?? '';
}

String _contentPart(Object? part) {
  if (part is String) {
    return part;
  }
  if (part is Map) {
    final text = part['text'];
    if (text is String) {
      return text;
    }
  }
  return '';
}

int _thinkMode(Ds4Thinking thinking) {
  return switch (thinking) {
    Ds4Thinking.disabled => _ds4ThinkNone,
    Ds4Thinking.max => _ds4ThinkMax,
    Ds4Thinking.high => _ds4ThinkHigh,
  };
}

int _seed() {
  return DateTime.now().microsecondsSinceEpoch ^
      (math.Random().nextInt(1 << 32) << 16);
}

String _errorString(Pointer<Char> err) {
  return err.cast<Utf8>().toDartString();
}

String _normalizePath(String path) {
  final normalized = p.normalize(path);
  return p.basename(normalized) == '.' ? p.dirname(normalized) : normalized;
}

void _withNativeUtf8(String value, void Function(Pointer<Utf8>) run) {
  final ptr = value.toNativeUtf8();
  try {
    run(ptr);
  } finally {
    malloc.free(ptr);
  }
}

const int _ds4ThinkNone = 0;
const int _ds4ThinkHigh = 1;
const int _ds4ThinkMax = 2;

final class _ThinkSplitter {
  _ThinkSplitter({required bool reasoning}) : _inReasoning = reasoning;

  bool _inReasoning;
  String _pending = '';

  List<Ds4ChatChunk> add(String text, {bool finish = false}) {
    if (text.isEmpty && _pending.isEmpty) {
      return const <Ds4ChatChunk>[];
    }
    final buffer = StringBuffer();
    final chunks = <Ds4ChatChunk>[];
    final input = '$_pending$text';
    _pending = '';
    var i = 0;
    Ds4ChatChunkKind kind() {
      return _inReasoning
          ? Ds4ChatChunkKind.reasoning
          : Ds4ChatChunkKind.content;
    }

    void flush() {
      if (buffer.isEmpty) {
        return;
      }
      chunks.add(Ds4ChatChunk(kind: kind(), text: buffer.toString()));
      buffer.clear();
    }

    while (i < input.length) {
      final tail = input.substring(i);
      if (tail.startsWith('<think>')) {
        flush();
        _inReasoning = true;
        i += '<think>'.length;
        continue;
      }
      if (tail.startsWith('</think>')) {
        flush();
        _inReasoning = false;
        i += '</think>'.length;
        continue;
      }
      if (!finish && input.codeUnitAt(i) == 60 && _isPartialThinkTag(tail)) {
        _pending = tail;
        break;
      }
      buffer.writeCharCode(input.codeUnitAt(i));
      i += 1;
    }
    flush();
    return chunks;
  }
}

bool _isPartialThinkTag(String value) {
  return '<think>'.startsWith(value) || '</think>'.startsWith(value);
}

final class _StreamingUtf8Decoder {
  final _sink = _StringCollector();
  late final ByteConversionSink _input = utf8.decoder.startChunkedConversion(
    _sink,
  );

  String add(List<int> bytes) {
    if (bytes.isNotEmpty) {
      _input.add(bytes);
    }
    return _sink.drain();
  }

  String close() {
    _input.close();
    return _sink.drain();
  }
}

final class _StringCollector extends StringConversionSinkBase {
  final _buffer = StringBuffer();

  @override
  void add(String str) {
    _buffer.write(str);
  }

  @override
  void addSlice(String str, int start, int end, bool isLast) {
    _buffer.write(str.substring(start, end));
    if (isLast) {
      close();
    }
  }

  @override
  void close() {}

  String drain() {
    final value = _buffer.toString();
    _buffer.clear();
    return value;
  }
}

final class _Ds4EngineOptions extends Struct {
  external Pointer<Utf8> modelPath;
  external Pointer<Utf8> mtpPath;
  @Int32()
  external int backend;
  @Int32()
  external int threads;
  @Int32()
  external int mtpDraftTokens;
  @Float()
  external double mtpMargin;
  @Bool()
  external bool warmWeights;
  @Bool()
  external bool quality;
}

final class _Ds4Tokens extends Struct {
  external Pointer<Int32> values;
  @Int32()
  external int len;
  @Int32()
  external int cap;
}

final class _Ds4Native {
  _Ds4Native(DynamicLibrary library)
    : engineOpen = library
          .lookupFunction<
            Int32 Function(Pointer<Pointer<Void>>, Pointer<_Ds4EngineOptions>),
            int Function(Pointer<Pointer<Void>>, Pointer<_Ds4EngineOptions>)
          >('ds4_engine_open'),
      engineClose = library
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('ds4_engine_close'),
      thinkModeEnabled = library
          .lookupFunction<Bool Function(Int32), bool Function(int)>(
            'ds4_think_mode_enabled',
          ),
      thinkModeForContext = library
          .lookupFunction<Int32 Function(Int32, Int32), int Function(int, int)>(
            'ds4_think_mode_for_context',
          ),
      tokensFree = library
          .lookupFunction<
            Void Function(Pointer<_Ds4Tokens>),
            void Function(Pointer<_Ds4Tokens>)
          >('ds4_tokens_free'),
      chatBegin = library
          .lookupFunction<
            Void Function(Pointer<Void>, Pointer<_Ds4Tokens>),
            void Function(Pointer<Void>, Pointer<_Ds4Tokens>)
          >('ds4_chat_begin'),
      chatAppendMaxEffortPrefix = library
          .lookupFunction<
            Void Function(Pointer<Void>, Pointer<_Ds4Tokens>),
            void Function(Pointer<Void>, Pointer<_Ds4Tokens>)
          >('ds4_chat_append_max_effort_prefix'),
      chatAppendMessage = library
          .lookupFunction<
            Void Function(
              Pointer<Void>,
              Pointer<_Ds4Tokens>,
              Pointer<Utf8>,
              Pointer<Utf8>,
            ),
            void Function(
              Pointer<Void>,
              Pointer<_Ds4Tokens>,
              Pointer<Utf8>,
              Pointer<Utf8>,
            )
          >('ds4_chat_append_message'),
      chatAppendAssistantPrefix = library
          .lookupFunction<
            Void Function(Pointer<Void>, Pointer<_Ds4Tokens>, Int32),
            void Function(Pointer<Void>, Pointer<_Ds4Tokens>, int)
          >('ds4_chat_append_assistant_prefix'),
      tokenText = library
          .lookupFunction<
            Pointer<Utf8> Function(Pointer<Void>, Int32, Pointer<Size>),
            Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Size>)
          >('ds4_token_text'),
      tokenEos = library
          .lookupFunction<
            Int32 Function(Pointer<Void>),
            int Function(Pointer<Void>)
          >('ds4_token_eos'),
      sessionCreate = library
          .lookupFunction<
            Int32 Function(Pointer<Pointer<Void>>, Pointer<Void>, Int32),
            int Function(Pointer<Pointer<Void>>, Pointer<Void>, int)
          >('ds4_session_create'),
      sessionFree = library
          .lookupFunction<
            Void Function(Pointer<Void>),
            void Function(Pointer<Void>)
          >('ds4_session_free'),
      sessionSync = library
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              Pointer<_Ds4Tokens>,
              Pointer<Char>,
              Size,
            ),
            int Function(Pointer<Void>, Pointer<_Ds4Tokens>, Pointer<Char>, int)
          >('ds4_session_sync'),
      sessionSample = library
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              Float,
              Int32,
              Float,
              Float,
              Pointer<Uint64>,
            ),
            int Function(
              Pointer<Void>,
              double,
              int,
              double,
              double,
              Pointer<Uint64>,
            )
          >('ds4_session_sample'),
      sessionEval = library
          .lookupFunction<
            Int32 Function(Pointer<Void>, Int32, Pointer<Char>, Size),
            int Function(Pointer<Void>, int, Pointer<Char>, int)
          >('ds4_session_eval'),
      sessionEvalSpeculativeArgmax = library
          .lookupFunction<
            Int32 Function(
              Pointer<Void>,
              Int32,
              Int32,
              Int32,
              Pointer<Int32>,
              Int32,
              Pointer<Char>,
              Size,
            ),
            int Function(
              Pointer<Void>,
              int,
              int,
              int,
              Pointer<Int32>,
              int,
              Pointer<Char>,
              int,
            )
          >('ds4_session_eval_speculative_argmax'),
      sessionPos = library
          .lookupFunction<
            Int32 Function(Pointer<Void>),
            int Function(Pointer<Void>)
          >('ds4_session_pos'),
      sessionCtx = library
          .lookupFunction<
            Int32 Function(Pointer<Void>),
            int Function(Pointer<Void>)
          >('ds4_session_ctx'),
      engineMtpDraftTokens = library
          .lookupFunction<
            Int32 Function(Pointer<Void>),
            int Function(Pointer<Void>)
          >('ds4_engine_mtp_draft_tokens');

  final int Function(Pointer<Pointer<Void>>, Pointer<_Ds4EngineOptions>)
  engineOpen;
  final void Function(Pointer<Void>) engineClose;
  final bool Function(int) thinkModeEnabled;
  final int Function(int, int) thinkModeForContext;
  final void Function(Pointer<_Ds4Tokens>) tokensFree;
  final void Function(Pointer<Void>, Pointer<_Ds4Tokens>) chatBegin;
  final void Function(Pointer<Void>, Pointer<_Ds4Tokens>)
  chatAppendMaxEffortPrefix;
  final void Function(
    Pointer<Void>,
    Pointer<_Ds4Tokens>,
    Pointer<Utf8>,
    Pointer<Utf8>,
  )
  chatAppendMessage;
  final void Function(Pointer<Void>, Pointer<_Ds4Tokens>, int)
  chatAppendAssistantPrefix;
  final Pointer<Utf8> Function(Pointer<Void>, int, Pointer<Size>) tokenText;
  final int Function(Pointer<Void>) tokenEos;
  final int Function(Pointer<Pointer<Void>>, Pointer<Void>, int) sessionCreate;
  final void Function(Pointer<Void>) sessionFree;
  final int Function(Pointer<Void>, Pointer<_Ds4Tokens>, Pointer<Char>, int)
  sessionSync;
  final int Function(
    Pointer<Void>,
    double,
    int,
    double,
    double,
    Pointer<Uint64>,
  )
  sessionSample;
  final int Function(Pointer<Void>, int, Pointer<Char>, int) sessionEval;
  final int Function(
    Pointer<Void>,
    int,
    int,
    int,
    Pointer<Int32>,
    int,
    Pointer<Char>,
    int,
  )
  sessionEvalSpeculativeArgmax;
  final int Function(Pointer<Void>) sessionPos;
  final int Function(Pointer<Void>) sessionCtx;
  final int Function(Pointer<Void>) engineMtpDraftTokens;
}

final class _Libc {
  _Libc() {
    final library = DynamicLibrary.process();
    _free = library
        .lookupFunction<
          Void Function(Pointer<Void>),
          void Function(Pointer<Void>)
        >('free');
    _setenv = library
        .lookupFunction<
          Int32 Function(Pointer<Utf8>, Pointer<Utf8>, Int32),
          int Function(Pointer<Utf8>, Pointer<Utf8>, int)
        >('setenv');
  }

  late final void Function(Pointer<Void>) _free;
  late final int Function(Pointer<Utf8>, Pointer<Utf8>, int) _setenv;

  void free(Pointer<Void> ptr) {
    if (ptr.address != 0) {
      _free(ptr);
    }
  }

  void setEnv(String name, String value) {
    _withNativeUtf8(name, (namePtr) {
      _withNativeUtf8(value, (valuePtr) {
        _setenv(namePtr, valuePtr, 0);
      });
    });
  }
}
