part of '../stable_api.dart';

/// Stateful private ANE chaining handle.
final class MlxAnePrivateChain {
  MlxAnePrivateChain._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_ane_private_chain_free);

  final shim.DartMlxAnePrivateChainHandle _handle;
  bool _closed = false;

  bool get isClosed => _closed;

  bool get isPrepared {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_ane_private_chain_is_prepared(_handle);
    if (result < 0) {
      _throwAnePrivateError('chain.isPrepared');
    }
    return result == 1;
  }

  bool get hasEnqueueSets {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_ane_private_chain_has_enqueue_sets(_handle);
    if (result < 0) {
      _throwAnePrivateError('chain.hasEnqueueSets');
    }
    return result == 1;
  }

  bool get hasBuffersReady {
    _ensureOpen();
    _clearAnePrivateError();
    final result = shim.dart_mlx_ane_private_chain_has_buffers_ready(_handle);
    if (result < 0) {
      _throwAnePrivateError('chain.hasBuffersReady');
    }
    return result == 1;
  }

  void enqueueSets() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'chain.enqueueSets',
      shim.dart_mlx_ane_private_chain_enqueue_sets(_handle),
    );
  }

  void buffersReady() {
    _ensureOpen();
    _clearAnePrivateError();
    _checkAnePrivateStatus(
      'chain.buffersReady',
      shim.dart_mlx_ane_private_chain_buffers_ready(_handle),
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_ane_private_chain_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxAnePrivateChain has been closed.');
    }
  }
}

extension MlxAnePrivateSessionChainExt on MlxAnePrivateSession {
  /// Builds a reusable private ANE chaining handle from this session.
  MlxAnePrivateChain createChain({
    bool validateRequest = true,
    bool useSharedSignalEvent = false,
    bool attemptPrepare = true,
  }) {
    _ensureOpen();
    _clearAnePrivateError();
    final handle = _checkAnePrivateHandle(
      'chain creation',
      shim.dart_mlx_ane_private_chain_new(
        _handle,
        validateRequest,
        useSharedSignalEvent,
        attemptPrepare,
      ),
    );
    return MlxAnePrivateChain._(handle);
  }
}
