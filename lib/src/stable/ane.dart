part of '../stable_api.dart';

bool _aneRuntimeEnabledOverride() {
  final raw = Platform.environment['DART_MLX_ENABLE_PRIVATE_ANE'];
  if (raw == null || raw.isEmpty) {
    return true;
  }
  switch (raw.toLowerCase()) {
    case '0':
    case 'false':
    case 'no':
    case 'off':
      return false;
    default:
      return true;
  }
}

void _ensureAneRuntimeEnabled() {
  if (!_aneRuntimeEnabledOverride()) {
    throw const MlxException(
      'Private ANE runtime is disabled by DART_MLX_ENABLE_PRIVATE_ANE=0.',
    );
  }
}

String? _copyAnePrivateErrorOrNull() {
  final copy = shim.dart_mlx_ane_private_last_error_copy();
  if (copy == ffi.nullptr) {
    return null;
  }
  try {
    return copy.cast<Utf8>().toDartString();
  } finally {
    shim.dart_mlx_string_free_copy(copy);
  }
}

void _clearAnePrivateError() {
  shim.dart_mlx_ane_private_clear_error();
}

Never _throwAnePrivateError(String operation, [int? code]) {
  final message =
      _copyAnePrivateErrorOrNull() ?? 'Private ANE $operation failed.';
  throw MlxException(message, code: code);
}

ffi.Pointer<ffi.Void> _checkAnePrivateHandle(
  String operation,
  ffi.Pointer<ffi.Void> handle,
) {
  if (handle == ffi.nullptr) {
    _throwAnePrivateError(operation);
  }
  return handle;
}

void _checkAnePrivateStatus(String operation, int status) {
  if (status != 0) {
    _throwAnePrivateError(operation, status);
  }
}

String? _copyAnePrivateOwnedStringOrNull(ffi.Pointer<ffi.Char> value) {
  if (value == ffi.nullptr) {
    final error = _copyAnePrivateErrorOrNull();
    if (error != null) {
      throw MlxException(error);
    }
    return null;
  }
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    shim.dart_mlx_string_free_copy(value);
  }
}

/// Snapshot of the private Apple Neural Engine bridge capabilities.
final class MlxAnePrivateInfo {
  const MlxAnePrivateInfo._(this.values);

  final Map<String, Object?> values;

  bool get compiled => values['compiled'] == true;

  bool get enabled => values['enabled'] == true;

  bool get frameworkLoaded => values['framework_loaded'] == true;

  String? get frameworkPath => values['framework_path'] as String?;

  bool get supportsBasicEval => values['supports_basic_eval'] == true;

  bool get supportsRealtimeEval => values['supports_realtime_eval'] == true;

  bool get supportsChaining => values['supports_chaining'] == true;

  bool get supportsPerfStats => values['supports_perf_stats'] == true;

  Iterable<String> get keys => values.keys;

  bool containsKey(String key) => values.containsKey(key);

  Object? operator [](String key) => values[key];
}

/// Snapshot of private ANE chaining/shared-event preparation capability.
final class MlxAnePrivateChainingInfo {
  const MlxAnePrivateChainingInfo._(this.values);

  final Map<String, Object?> values;

  int get stage => (values['stage'] as num?)?.toInt() ?? 0;
  bool get prepared => values['prepared'] == true;
  String get error => (values['error'] as String?) ?? '';
  bool get enqueueSetsSucceeded => values['enqueue_sets_succeeded'] == true;
  bool get buffersReadySucceeded => values['buffers_ready_succeeded'] == true;

  Iterable<String> get keys => values.keys;
  bool containsKey(String key) => values.containsKey(key);
  Object? operator [](String key) => values[key];
}

/// Managed wrapper for a private ANE in-memory model.
final class MlxAnePrivateModel {
  MlxAnePrivateModel._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  factory MlxAnePrivateModel.fromMil(
    String milText, {
    List<MlxAneWeight> weights = const [],
  }) {
    return MlxAnePrivateModel.fromMilWithOffsets(
      milText,
      weights: [
        for (final weight in weights)
          (path: weight.path, data: weight.data, offset: 0),
      ],
    );
  }

  factory MlxAnePrivateModel.fromMilWithOffsets(
    String milText, {
    List<MlxAneWeightWithOffset> weights = const [],
  }) {
    if (milText.isEmpty) {
      throw ArgumentError.value(milText, 'milText', 'Must not be empty.');
    }

    return _withCString(milText, (nativeMil) {
      final weightPaths = weights.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Pointer<ffi.Char>>(weights.length);
      final weightData = weights.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Pointer<ffi.Uint8>>(weights.length);
      final weightLens = weights.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Size>(weights.length);
      final weightOffsets = weights.isEmpty
          ? ffi.nullptr
          : calloc<ffi.Size>(weights.length);
      final allocatedPaths = <ffi.Pointer<ffi.Char>>[];
      final allocatedBuffers = <ffi.Pointer<ffi.Uint8>>[];

      try {
        for (var index = 0; index < weights.length; index++) {
          final path = weights[index].path.toNativeUtf8().cast<ffi.Char>();
          allocatedPaths.add(path);
          weightPaths[index] = path;

          final bytes = weights[index].data;
          final buffer = calloc<ffi.Uint8>(bytes.length);
          allocatedBuffers.add(buffer);
          buffer.asTypedList(bytes.length).setAll(0, bytes);
          weightData[index] = buffer;
          weightLens[index] = bytes.length;
          weightOffsets[index] = weights[index].offset;
        }

        _clearAnePrivateError();
        final handle = _checkAnePrivateHandle(
          'model creation',
          shim.dart_mlx_ane_private_model_new_mil_ex(
            nativeMil,
            weightPaths,
            weightData,
            weightLens,
            weightOffsets,
            weights.length,
          ),
        );
        return MlxAnePrivateModel._(handle);
      } finally {
        for (final path in allocatedPaths) {
          calloc.free(path);
        }
        for (final buffer in allocatedBuffers) {
          calloc.free(buffer);
        }
        if (weightPaths != ffi.nullptr) {
          calloc.free(weightPaths);
        }
        if (weightData != ffi.nullptr) {
          calloc.free(weightData);
        }
        if (weightLens != ffi.nullptr) {
          calloc.free(weightLens);
        }
        if (weightOffsets != ffi.nullptr) {
          calloc.free(weightOffsets);
        }
      }
    });
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_ane_private_model_free);

  final shim.DartMlxAnePrivateModelHandle _handle;
  bool _closed = false;

  bool get isClosed => _closed;

  bool get isLoaded {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_ane_private_model_is_loaded(_handle);
    if (result < 0) {
      _throwAnePrivateError('isLoaded');
    }
    return result == 1;
  }

  bool get compiledModelExists {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_ane_private_model_compiled_exists(_handle);
    if (result < 0) {
      _throwAnePrivateError('compiledModelExists');
    }
    return result == 1;
  }

  String? get hexIdentifier {
    _ensureOpen();
    _clearAnePrivateError();
    return _copyAnePrivateOwnedStringOrNull(
      shim.dart_mlx_ane_private_model_hex_identifier_copy(_handle),
    );
  }

  void compile() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'compile',
      shim.dart_mlx_ane_private_model_compile(_handle),
    );
  }

  void load() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'load',
      shim.dart_mlx_ane_private_model_load(_handle),
    );
  }

  void unload() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'unload',
      shim.dart_mlx_ane_private_model_unload(_handle),
    );
  }

  MlxAnePrivateSession createSession({
    required List<int> inputByteSizes,
    required List<int> outputByteSizes,
  }) {
    _ensureOpen();
    return MlxAnePrivateSession._create(
      this,
      inputByteSizes: inputByteSizes,
      outputByteSizes: outputByteSizes,
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_ane_private_model_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxAnePrivateModel has been closed.');
    }
  }
}

/// Byte-oriented private ANE execution session backed by IOSurface buffers.
final class MlxAnePrivateSession {
  MlxAnePrivateSession._(
    this._handle,
    this.inputByteSizes,
    this.outputByteSizes,
  ) : _inputScratch = List<ffi.Pointer<ffi.Uint8>?>.filled(
        inputByteSizes.length,
        null,
      ),
      _outputScratch = List<ffi.Pointer<ffi.Uint8>?>.filled(
        outputByteSizes.length,
        null,
      ) {
    _finalizer.attach(this, _handle, detach: this);
  }

  factory MlxAnePrivateSession._create(
    MlxAnePrivateModel model, {
    required List<int> inputByteSizes,
    required List<int> outputByteSizes,
  }) {
    if (inputByteSizes.any((value) => value < 0)) {
      throw ArgumentError.value(
        inputByteSizes,
        'inputByteSizes',
        'All sizes must be non-negative.',
      );
    }
    if (outputByteSizes.any((value) => value < 0)) {
      throw ArgumentError.value(
        outputByteSizes,
        'outputByteSizes',
        'All sizes must be non-negative.',
      );
    }

    final nativeInputs = inputByteSizes.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Size>(inputByteSizes.length);
    final nativeOutputs = outputByteSizes.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Size>(outputByteSizes.length);
    try {
      for (var index = 0; index < inputByteSizes.length; index++) {
        nativeInputs[index] = inputByteSizes[index];
      }
      for (var index = 0; index < outputByteSizes.length; index++) {
        nativeOutputs[index] = outputByteSizes[index];
      }

      _clearAnePrivateError();
      final handle = _checkAnePrivateHandle(
        'session creation',
        shim.dart_mlx_ane_private_session_new(
          model._handle,
          nativeInputs,
          inputByteSizes.length,
          nativeOutputs,
          outputByteSizes.length,
        ),
      );
      return MlxAnePrivateSession._(
        handle,
        List<int>.unmodifiable(inputByteSizes),
        List<int>.unmodifiable(outputByteSizes),
      );
    } finally {
      if (nativeInputs != ffi.nullptr) {
        calloc.free(nativeInputs);
      }
      if (nativeOutputs != ffi.nullptr) {
        calloc.free(nativeOutputs);
      }
    }
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_ane_private_session_free);

  final shim.DartMlxAnePrivateSessionHandle _handle;
  final List<int> inputByteSizes;
  final List<int> outputByteSizes;
  final List<ffi.Pointer<ffi.Uint8>?> _inputScratch;
  final List<ffi.Pointer<ffi.Uint8>?> _outputScratch;
  bool _closed = false;

  bool get isClosed => _closed;

  bool get isRealtimeLoaded {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_ane_private_session_realtime_is_loaded(
      _handle,
    );
    if (result < 0) {
      _throwAnePrivateError('isRealtimeLoaded');
    }
    return result == 1;
  }

  void writeInputBytes(int index, Uint8List bytes) {
    _ensureOpen();
    _checkInputIndex(index);
    if (bytes.length != inputByteSizes[index]) {
      throw ArgumentError.value(
        bytes.length,
        'bytes.length',
        'Expected ${inputByteSizes[index]} bytes for input $index.',
      );
    }
    final nativeBytes = _ensureScratch(_inputScratch, index, bytes.length);
    nativeBytes.asTypedList(bytes.length).setAll(0, bytes);
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'writeInputBytes',
      shim.dart_mlx_ane_private_session_write_input_bytes(
        _handle,
        index,
        nativeBytes,
        bytes.length,
      ),
    );
  }

  void writeInputFloat32(int index, Float32List values) {
    writeInputBytes(index, _encodeFp16Bytes(values));
  }

  void writeInputRawFloat32(int index, Float32List values) {
    _ensureOpen();
    _checkInputIndex(index);
    if (values.lengthInBytes != inputByteSizes[index]) {
      throw ArgumentError.value(
        values.lengthInBytes,
        'values.lengthInBytes',
        'Expected ${inputByteSizes[index]} bytes for input $index.',
      );
    }
    final nativeBytes = _ensureScratch(
      _inputScratch,
      index,
      values.lengthInBytes,
    );
    nativeBytes.cast<ffi.Float>().asTypedList(values.length).setAll(0, values);
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'writeInputRawFloat32',
      shim.dart_mlx_ane_private_session_write_input_bytes(
        _handle,
        index,
        nativeBytes,
        values.lengthInBytes,
      ),
    );
  }

  void writeInputPackedArrayFloat32(
    int index,
    MlxArray input, {
    required int seqLen,
    required int dim,
    required int lane,
  }) {
    _ensureOpen();
    _checkInputIndex(index);
    input._ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'writeInputPackedArrayFloat32',
      shim.dart_mlx_ane_private_session_write_input_array_packed_f32(
        _handle,
        index,
        input._handle,
        seqLen,
        dim,
        lane,
      ),
    );
  }

  Uint8List readOutputBytes(int index) {
    _ensureOpen();
    _checkOutputIndex(index);
    final len = outputByteSizes[index];
    final out = _ensureScratch(_outputScratch, index, len);
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'readOutputBytes',
      shim.dart_mlx_ane_private_session_read_output_bytes(
        _handle,
        index,
        out,
        len,
      ),
    );
    return Uint8List.fromList(out.asTypedList(len));
  }

  Float32List readOutputFloat32(int index) {
    return _decodeFp16Bytes(readOutputBytes(index));
  }

  Float32List readOutputRawFloat32(int index) {
    return Float32List.fromList(readOutputRawFloat32View(index));
  }

  Float32List readOutputRawFloat32View(int index) {
    _ensureOpen();
    _checkOutputIndex(index);
    final len = outputByteSizes[index];
    if (len % Float32List.bytesPerElement != 0) {
      throw ArgumentError.value(
        len,
        'outputByteSizes[$index]',
        'Expected a multiple of ${Float32List.bytesPerElement} bytes.',
      );
    }
    final out = _ensureScratch(_outputScratch, index, len);
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'readOutputRawFloat32',
      shim.dart_mlx_ane_private_session_read_output_bytes(
        _handle,
        index,
        out,
        len,
      ),
    );
    return out.cast<ffi.Float>().asTypedList(
      len ~/ Float32List.bytesPerElement,
    );
  }

  void evaluate() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'evaluate',
      shim.dart_mlx_ane_private_session_evaluate(_handle),
    );
  }

  void prepareRealtime() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'prepareRealtime',
      shim.dart_mlx_ane_private_session_prepare_realtime(_handle),
    );
  }

  void evaluateRealtime() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'evaluateRealtime',
      shim.dart_mlx_ane_private_session_evaluate_realtime(_handle),
    );
  }

  void teardownRealtime() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'teardownRealtime',
      shim.dart_mlx_ane_private_session_teardown_realtime(_handle),
    );
  }

  List<Uint8List> run(List<Uint8List> inputs) {
    if (inputs.length != inputByteSizes.length) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected ${inputByteSizes.length} inputs.',
      );
    }
    for (var index = 0; index < inputs.length; index++) {
      writeInputBytes(index, inputs[index]);
    }
    evaluate();
    return List<Uint8List>.generate(outputByteSizes.length, readOutputBytes);
  }

  List<Float32List> runFloat32(List<Float32List> inputs) {
    if (inputs.length != inputByteSizes.length) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected ${inputByteSizes.length} inputs.',
      );
    }
    for (var index = 0; index < inputs.length; index++) {
      writeInputFloat32(index, inputs[index]);
    }
    evaluate();
    return List<Float32List>.generate(
      outputByteSizes.length,
      readOutputFloat32,
    );
  }

  List<Float32List> runRawFloat32(List<Float32List> inputs) {
    if (inputs.length != inputByteSizes.length) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected ${inputByteSizes.length} inputs.',
      );
    }
    for (var index = 0; index < inputs.length; index++) {
      writeInputRawFloat32(index, inputs[index]);
    }
    evaluate();
    return List<Float32List>.generate(
      outputByteSizes.length,
      readOutputRawFloat32,
    );
  }

  List<Uint8List> runRealtime(List<Uint8List> inputs) {
    if (inputs.length != inputByteSizes.length) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected ${inputByteSizes.length} inputs.',
      );
    }
    for (var index = 0; index < inputs.length; index++) {
      writeInputBytes(index, inputs[index]);
    }
    evaluateRealtime();
    return List<Uint8List>.generate(outputByteSizes.length, readOutputBytes);
  }

  List<Float32List> runFloat32Realtime(List<Float32List> inputs) {
    if (inputs.length != inputByteSizes.length) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected ${inputByteSizes.length} inputs.',
      );
    }
    for (var index = 0; index < inputs.length; index++) {
      writeInputFloat32(index, inputs[index]);
    }
    evaluateRealtime();
    return List<Float32List>.generate(
      outputByteSizes.length,
      readOutputFloat32,
    );
  }

  List<Float32List> runRawFloat32Realtime(List<Float32List> inputs) {
    if (inputs.length != inputByteSizes.length) {
      throw ArgumentError.value(
        inputs.length,
        'inputs.length',
        'Expected ${inputByteSizes.length} inputs.',
      );
    }
    for (var index = 0; index < inputs.length; index++) {
      writeInputRawFloat32(index, inputs[index]);
    }
    evaluateRealtime();
    return List<Float32List>.generate(
      outputByteSizes.length,
      readOutputRawFloat32,
    );
  }

  MlxAnePrivateChainingInfo probeChaining({
    bool validateRequest = true,
    bool useSharedSignalEvent = false,
    bool attemptPrepare = true,
    bool callEnqueueSets = false,
    bool callBuffersReady = false,
  }) {
    _ensureOpen();
    _clearAnePrivateError();
    final json = _copyOwnedString(
      shim.dart_mlx_ane_private_session_probe_chaining_json_copy(
        _handle,
        validateRequest,
        useSharedSignalEvent,
        attemptPrepare,
        callEnqueueSets,
        callBuffersReady,
      ),
    );
    return MlxAnePrivateChainingInfo._(
      Map<String, Object?>.from(jsonDecode(json) as Map),
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    for (final pointer in _inputScratch) {
      if (pointer != null) {
        calloc.free(pointer);
      }
    }
    for (final pointer in _outputScratch) {
      if (pointer != null) {
        calloc.free(pointer);
      }
    }
    shim.dart_mlx_ane_private_session_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxAnePrivateSession has been closed.');
    }
  }

  void _checkInputIndex(int index) {
    if (index < 0 || index >= inputByteSizes.length) {
      throw RangeError.index(index, inputByteSizes, 'index');
    }
  }

  void _checkOutputIndex(int index) {
    if (index < 0 || index >= outputByteSizes.length) {
      throw RangeError.index(index, outputByteSizes, 'index');
    }
  }

  ffi.Pointer<ffi.Uint8> _ensureScratch(
    List<ffi.Pointer<ffi.Uint8>?> slots,
    int index,
    int len,
  ) {
    final existing = slots[index];
    if (existing != null) {
      return existing;
    }
    final pointer = calloc<ffi.Uint8>(len);
    slots[index] = pointer;
    return pointer;
  }
}
