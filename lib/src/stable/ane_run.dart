part of '../stable_api.dart';

/// Execution path used by [MlxAnePrivateRunner].
enum MlxAnePrivateExecution { standard, realtime }

/// Summary returned by [MlxAnePrivateRunner.advanceChain].
final class MlxAnePrivateChainAdvanceResult {
  const MlxAnePrivateChainAdvanceResult({
    required this.hasChain,
    required this.hasBuffersReady,
    required this.hasEnqueueSets,
    required this.buffersReadyAttempted,
    required this.buffersReadySucceeded,
    required this.enqueueSetsAttempted,
    required this.enqueueSetsSucceeded,
  });

  final bool hasChain;
  final bool hasBuffersReady;
  final bool hasEnqueueSets;
  final bool buffersReadyAttempted;
  final bool buffersReadySucceeded;
  final bool enqueueSetsAttempted;
  final bool enqueueSetsSucceeded;
}

/// High-level long-lived private ANE runner built on a model/session/chain stack.
final class MlxAnePrivateRunner {
  MlxAnePrivateRunner._({
    required MlxAnePrivateSession session,
    required MlxAnePrivateExecution execution,
    required bool enableChain,
    required bool validateChainRequest,
    required bool useSharedSignalEvent,
    required bool attemptChainPrepare,
    required bool closeModelOnClose,
    required MlxAnePrivateModel model,
  }) : _session = session,
       _execution = execution,
       _enableChain = enableChain,
       _validateChainRequest = validateChainRequest,
       _useSharedSignalEvent = useSharedSignalEvent,
       _attemptChainPrepare = attemptChainPrepare,
       _closeModelOnClose = closeModelOnClose,
       _model = model;

  final MlxAnePrivateSession _session;
  final MlxAnePrivateExecution _execution;
  final bool _enableChain;
  final bool _validateChainRequest;
  final bool _useSharedSignalEvent;
  final bool _attemptChainPrepare;
  final bool _closeModelOnClose;
  final MlxAnePrivateModel _model;
  MlxAnePrivateChain? _chain;
  bool _closed = false;
  bool _prepared = false;

  bool get isClosed => _closed;

  bool get isPrepared => _prepared;

  bool get usesRealtime => _execution == MlxAnePrivateExecution.realtime;

  bool get hasChain => _chain != null;

  bool get isRealtimePrepared =>
      usesRealtime ? _session.isRealtimeLoaded : false;

  bool get isChainPrepared => _chain?.isPrepared ?? false;

  void prepare() {
    _ensureOpen();
    if (_prepared) {
      return;
    }
    if (usesRealtime) {
      _session.prepareRealtime();
    }
    if (_enableChain) {
      _chain = _session.createChain(
        validateRequest: _validateChainRequest,
        useSharedSignalEvent: _useSharedSignalEvent,
        attemptPrepare: _attemptChainPrepare,
      );
    }
    _prepared = true;
  }

  List<Uint8List> runBytes(List<Uint8List> inputs) {
    _ensureReady();
    return switch (_execution) {
      MlxAnePrivateExecution.standard => _session.run(inputs),
      MlxAnePrivateExecution.realtime => _session.runRealtime(inputs),
    };
  }

  List<Float32List> runFloat32(List<Float32List> inputs) {
    _ensureReady();
    return switch (_execution) {
      MlxAnePrivateExecution.standard => _session.runFloat32(inputs),
      MlxAnePrivateExecution.realtime => _session.runFloat32Realtime(inputs),
    };
  }

  MlxAnePrivateChainAdvanceResult advanceChain() {
    _ensureReady();
    final chain = _chain;
    if (chain == null) {
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

    var buffersReadyAttempted = false;
    var buffersReadySucceeded = false;
    var enqueueSetsAttempted = false;
    var enqueueSetsSucceeded = false;

    if (chain.hasBuffersReady) {
      buffersReadyAttempted = true;
      chain.buffersReady();
      buffersReadySucceeded = true;
    }
    if (chain.hasEnqueueSets) {
      enqueueSetsAttempted = true;
      chain.enqueueSets();
      enqueueSetsSucceeded = true;
    }

    return MlxAnePrivateChainAdvanceResult(
      hasChain: true,
      hasBuffersReady: chain.hasBuffersReady,
      hasEnqueueSets: chain.hasEnqueueSets,
      buffersReadyAttempted: buffersReadyAttempted,
      buffersReadySucceeded: buffersReadySucceeded,
      enqueueSetsAttempted: enqueueSetsAttempted,
      enqueueSetsSucceeded: enqueueSetsSucceeded,
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _chain?.close();
    if (usesRealtime && _session.isRealtimeLoaded) {
      try {
        _session.teardownRealtime();
      } on MlxException {
        // Best-effort teardown during close.
      }
    }
    _session.close();
    if (_closeModelOnClose) {
      _model.close();
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxAnePrivateRunner has been closed.');
    }
  }

  void _ensureReady() {
    _ensureOpen();
    if (!_prepared) {
      prepare();
    }
  }
}

extension MlxAnePrivateModelRunnerExt on MlxAnePrivateModel {
  /// Creates a long-lived private ANE runner for repeated steps.
  MlxAnePrivateRunner createRunner({
    required List<int> inputByteSizes,
    required List<int> outputByteSizes,
    MlxAnePrivateExecution execution = MlxAnePrivateExecution.standard,
    bool enableChain = false,
    bool validateChainRequest = true,
    bool useSharedSignalEvent = false,
    bool attemptChainPrepare = true,
    bool closeModelOnClose = false,
  }) {
    _ensureOpen();
    final session = createSession(
      inputByteSizes: inputByteSizes,
      outputByteSizes: outputByteSizes,
    );
    return MlxAnePrivateRunner._(
      session: session,
      execution: execution,
      enableChain: enableChain,
      validateChainRequest: validateChainRequest,
      useSharedSignalEvent: useSharedSignalEvent,
      attemptChainPrepare: attemptChainPrepare,
      closeModelOnClose: closeModelOnClose,
      model: this,
    );
  }
}
