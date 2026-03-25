part of '../stable_api.dart';

typedef MlxAneFeedbackEdge = ({int inputIndex, int outputIndex});

/// Result returned by a multi-step private ANE loop step.
final class MlxAnePrivateLoopResult<T> {
  const MlxAnePrivateLoopResult({
    required this.outputs,
    required this.chainAdvance,
  });

  final List<T> outputs;
  final MlxAnePrivateChainAdvanceResult chainAdvance;
}

/// Generic multi-step loop runner with explicit output-to-input feedback.
final class MlxAnePrivateLoopRunner {
  MlxAnePrivateLoopRunner._({
    required MlxAnePrivateRunner runner,
    required List<MlxAneFeedbackEdge> feedback,
    required bool autoAdvanceChain,
  }) : _runner = runner,
       _feedback = List<MlxAneFeedbackEdge>.unmodifiable(feedback),
       _autoAdvanceChain = autoAdvanceChain {
    final inputCount = _runner._session.inputByteSizes.length;
    final outputCount = _runner._session.outputByteSizes.length;
    for (final edge in _feedback) {
      if (edge.inputIndex < 0 || edge.inputIndex >= inputCount) {
        throw RangeError.index(
          edge.inputIndex,
          _runner._session.inputByteSizes,
        );
      }
      if (edge.outputIndex < 0 || edge.outputIndex >= outputCount) {
        throw RangeError.index(
          edge.outputIndex,
          _runner._session.outputByteSizes,
        );
      }
      if (_runner._session.inputByteSizes[edge.inputIndex] !=
          _runner._session.outputByteSizes[edge.outputIndex]) {
        throw ArgumentError(
          'Feedback edge ${edge.outputIndex} -> ${edge.inputIndex} '
          'requires matching byte sizes.',
        );
      }
    }
    _stateBytes = List<Uint8List?>.filled(inputCount, null);
  }

  final MlxAnePrivateRunner _runner;
  final List<MlxAneFeedbackEdge> _feedback;
  final bool _autoAdvanceChain;
  late final List<Uint8List?> _stateBytes;
  bool _closed = false;

  bool get isClosed => _closed;

  List<MlxAneFeedbackEdge> get feedback => _feedback;

  bool get autoAdvanceChain => _autoAdvanceChain;

  void seedInputBytes(int index, Uint8List bytes) {
    _ensureOpen();
    _checkInputIndex(index);
    final expected = _runner._session.inputByteSizes[index];
    if (bytes.length != expected) {
      throw ArgumentError.value(
        bytes.length,
        'bytes.length',
        'Expected $expected bytes for input $index.',
      );
    }
    _stateBytes[index] = Uint8List.fromList(bytes);
  }

  void seedInputFloat32(int index, Float32List values) {
    seedInputBytes(index, MlxAnePrivate.encodeFp16Bytes(values));
  }

  Uint8List? inputStateBytes(int index) {
    _ensureOpen();
    _checkInputIndex(index);
    final value = _stateBytes[index];
    return value == null ? null : Uint8List.fromList(value);
  }

  Float32List? inputStateFloat32(int index) {
    final value = inputStateBytes(index);
    return value == null ? null : MlxAnePrivate.decodeFp16Bytes(value);
  }

  void clearState() {
    _ensureOpen();
    for (var index = 0; index < _stateBytes.length; index++) {
      _stateBytes[index] = null;
    }
  }

  MlxAnePrivateLoopResult<Uint8List> stepBytes({
    Map<int, Uint8List> inputs = const {},
  }) {
    _ensureOpen();
    final materialized = _materializeByteInputs(inputs);
    final outputs = _runner.runBytes(materialized);
    _applyFeedback(outputs);
    final chainAdvance = _maybeAdvanceChain();
    return MlxAnePrivateLoopResult(
      outputs: outputs,
      chainAdvance: chainAdvance,
    );
  }

  MlxAnePrivateLoopResult<Float32List> stepFloat32({
    Map<int, Float32List> inputs = const {},
  }) {
    _ensureOpen();
    final byteInputs = <int, Uint8List>{};
    for (final entry in inputs.entries) {
      byteInputs[entry.key] = MlxAnePrivate.encodeFp16Bytes(entry.value);
    }
    final result = stepBytes(inputs: byteInputs);
    return MlxAnePrivateLoopResult(
      outputs: result.outputs.map(MlxAnePrivate.decodeFp16Bytes).toList(),
      chainAdvance: result.chainAdvance,
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _runner.close();
  }

  List<Uint8List> _materializeByteInputs(Map<int, Uint8List> overrides) {
    final inputCount = _runner._session.inputByteSizes.length;
    final materialized = List<Uint8List?>.filled(inputCount, null);
    for (final entry in overrides.entries) {
      _checkInputIndex(entry.key);
      final expected = _runner._session.inputByteSizes[entry.key];
      if (entry.value.length != expected) {
        throw ArgumentError.value(
          entry.value.length,
          'inputs[${entry.key}].length',
          'Expected $expected bytes.',
        );
      }
      materialized[entry.key] = Uint8List.fromList(entry.value);
    }
    for (var index = 0; index < inputCount; index++) {
      materialized[index] ??= _stateBytes[index];
      final value = materialized[index];
      if (value == null) {
        throw StateError(
          'Input $index has no provided value or feedback state.',
        );
      }
    }
    return materialized.cast<Uint8List>();
  }

  void _applyFeedback(List<Uint8List> outputs) {
    for (final edge in _feedback) {
      _stateBytes[edge.inputIndex] = Uint8List.fromList(
        outputs[edge.outputIndex],
      );
    }
  }

  MlxAnePrivateChainAdvanceResult _maybeAdvanceChain() {
    if (!_autoAdvanceChain) {
      return const MlxAnePrivateChainAdvanceResult(
        hasChain: false,
        hasBuffersReady: false,
        hasEnqueueSets: false,
        buffersReadyAttempted: false,
        buffersReadySucceeded: false,
        enqueueSetsAttempted: false,
        enqueueSetsSucceeded: false,
      );
    }
    return _runner.advanceChain();
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxAnePrivateLoopRunner has been closed.');
    }
  }

  void _checkInputIndex(int index) {
    if (index < 0 || index >= _runner._session.inputByteSizes.length) {
      throw RangeError.index(index, _runner._session.inputByteSizes, 'index');
    }
  }
}

extension MlxAnePrivateModelLoopExt on MlxAnePrivateModel {
  /// Creates a multi-step loop runner with explicit feedback edges.
  MlxAnePrivateLoopRunner createLoopRunner({
    required List<int> inputByteSizes,
    required List<int> outputByteSizes,
    required List<MlxAneFeedbackEdge> feedback,
    MlxAnePrivateExecution execution = MlxAnePrivateExecution.standard,
    bool enableChain = false,
    bool validateChainRequest = true,
    bool useSharedSignalEvent = false,
    bool attemptChainPrepare = true,
    bool closeModelOnClose = false,
    bool autoAdvanceChain = true,
  }) {
    final runner = createRunner(
      inputByteSizes: inputByteSizes,
      outputByteSizes: outputByteSizes,
      execution: execution,
      enableChain: enableChain,
      validateChainRequest: validateChainRequest,
      useSharedSignalEvent: useSharedSignalEvent,
      attemptChainPrepare: attemptChainPrepare,
      closeModelOnClose: closeModelOnClose,
    );
    return MlxAnePrivateLoopRunner._(
      runner: runner,
      feedback: feedback,
      autoAdvanceChain: autoAdvanceChain,
    );
  }
}
