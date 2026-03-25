part of '../stable_api.dart';

/// Result returned by a decode-oriented private ANE step.
final class MlxAnePrivateDecodeStepResult<T> {
  const MlxAnePrivateDecodeStepResult({
    required this.token,
    required this.outputs,
    required this.chainAdvance,
  });

  final T token;
  final List<T> outputs;
  final MlxAnePrivateChainAdvanceResult chainAdvance;
}

/// Decode-oriented wrapper around [MlxAnePrivateLoopRunner].
final class MlxAnePrivateDecodeRunner {
  MlxAnePrivateDecodeRunner._({
    required MlxAnePrivateLoopRunner loop,
    required this.tokenInputIndex,
    required this.tokenOutputIndex,
  }) : _loop = loop {
    final inputCount = _loop._runner._session.inputByteSizes.length;
    final outputCount = _loop._runner._session.outputByteSizes.length;
    if (tokenInputIndex < 0 || tokenInputIndex >= inputCount) {
      throw RangeError.index(
        tokenInputIndex,
        _loop._runner._session.inputByteSizes,
      );
    }
    if (tokenOutputIndex < 0 || tokenOutputIndex >= outputCount) {
      throw RangeError.index(
        tokenOutputIndex,
        _loop._runner._session.outputByteSizes,
      );
    }
  }

  final MlxAnePrivateLoopRunner _loop;
  final int tokenInputIndex;
  final int tokenOutputIndex;

  bool get isClosed => _loop.isClosed;

  void prepare() {
    _ensureOpen();
    _loop._runner.prepare();
  }

  void seedStateBytes(int index, Uint8List bytes) =>
      _loop.seedInputBytes(index, bytes);

  void seedStateFloat32(int index, Float32List values) =>
      _loop.seedInputFloat32(index, values);

  Uint8List? stateBytes(int index) => _loop.inputStateBytes(index);

  Float32List? stateFloat32(int index) => _loop.inputStateFloat32(index);

  void clearState() => _loop.clearState();

  MlxAnePrivateDecodeStepResult<Uint8List> stepBytes(
    Uint8List token, {
    Map<int, Uint8List> inputs = const {},
  }) {
    _ensureOpen();
    final mergedInputs = <int, Uint8List>{...inputs, tokenInputIndex: token};
    final result = _loop.stepBytes(inputs: mergedInputs);
    return MlxAnePrivateDecodeStepResult(
      token: result.outputs[tokenOutputIndex],
      outputs: result.outputs,
      chainAdvance: result.chainAdvance,
    );
  }

  MlxAnePrivateDecodeStepResult<Float32List> stepFloat32(
    Float32List token, {
    Map<int, Float32List> inputs = const {},
  }) {
    _ensureOpen();
    final mergedInputs = <int, Float32List>{...inputs, tokenInputIndex: token};
    final result = _loop.stepFloat32(inputs: mergedInputs);
    return MlxAnePrivateDecodeStepResult(
      token: result.outputs[tokenOutputIndex],
      outputs: result.outputs,
      chainAdvance: result.chainAdvance,
    );
  }

  List<Uint8List> generateBytes(
    Uint8List seedToken,
    int steps, {
    Map<int, Uint8List> inputs = const {},
  }) {
    _ensureOpen();
    if (steps < 0) {
      throw ArgumentError.value(steps, 'steps', 'Must be non-negative.');
    }
    final outputs = <Uint8List>[];
    var token = Uint8List.fromList(seedToken);
    for (var index = 0; index < steps; index++) {
      final result = stepBytes(token, inputs: inputs);
      outputs.add(Uint8List.fromList(result.token));
      token = result.token;
    }
    return outputs;
  }

  List<Float32List> generateFloat32(
    Float32List seedToken,
    int steps, {
    Map<int, Float32List> inputs = const {},
  }) {
    _ensureOpen();
    if (steps < 0) {
      throw ArgumentError.value(steps, 'steps', 'Must be non-negative.');
    }
    final outputs = <Float32List>[];
    var token = Float32List.fromList(seedToken);
    for (var index = 0; index < steps; index++) {
      final result = stepFloat32(token, inputs: inputs);
      outputs.add(Float32List.fromList(result.token));
      token = result.token;
    }
    return outputs;
  }

  void close() => _loop.close();

  void _ensureOpen() {
    if (isClosed) {
      throw StateError('MlxAnePrivateDecodeRunner has been closed.');
    }
  }
}

extension MlxAnePrivateModelDecodeExt on MlxAnePrivateModel {
  /// Creates a decode-oriented runner with a distinguished token input/output pair.
  MlxAnePrivateDecodeRunner createDecodeRunner({
    required List<int> inputByteSizes,
    required List<int> outputByteSizes,
    required int tokenInputIndex,
    required int tokenOutputIndex,
    List<MlxAneFeedbackEdge> feedback = const [],
    MlxAnePrivateExecution execution = MlxAnePrivateExecution.standard,
    bool enableChain = false,
    bool validateChainRequest = true,
    bool useSharedSignalEvent = false,
    bool attemptChainPrepare = true,
    bool closeModelOnClose = false,
    bool autoAdvanceChain = true,
  }) {
    final loop = createLoopRunner(
      inputByteSizes: inputByteSizes,
      outputByteSizes: outputByteSizes,
      feedback: feedback,
      execution: execution,
      enableChain: enableChain,
      validateChainRequest: validateChainRequest,
      useSharedSignalEvent: useSharedSignalEvent,
      attemptChainPrepare: attemptChainPrepare,
      closeModelOnClose: closeModelOnClose,
      autoAdvanceChain: autoAdvanceChain,
    );
    return MlxAnePrivateDecodeRunner._(
      loop: loop,
      tokenInputIndex: tokenInputIndex,
      tokenOutputIndex: tokenOutputIndex,
    );
  }
}
