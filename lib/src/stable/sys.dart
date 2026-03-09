part of '../stable_api.dart';

/// Snapshot of backend-specific device metadata.
final class MlxDeviceInfo {
  const MlxDeviceInfo._(this.values);

  final Map<String, Object> values;

  Iterable<String> get keys => values.keys;

  bool containsKey(String key) => values.containsKey(key);

  Object? operator [](String key) => values[key];
}

/// Managed wrapper for an MLX stream.
final class MlxStream {
  MlxStream._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  // dart format off
  factory MlxStream.create() { _clearError(); final handle = hooks.debugStreamNewOverride?.call() ?? shim.dart_mlx_stream_new(); return MlxStream._(_checkHandle('dart_mlx_stream_new', handle)); }
  // dart format on

  factory MlxStream.forDevice(MlxDevice device) {
    device._ensureOpen();
    _clearError();
    return MlxStream._(
      _checkHandle(
        'dart_mlx_stream_new_device',
        shim.dart_mlx_stream_new_device(device._handle),
      ),
    );
  }

  factory MlxStream.defaultFor(MlxDevice device) {
    device._ensureOpen();
    _clearError();
    return MlxStream._(
      _checkHandle(
        'dart_mlx_get_default_stream',
        shim.dart_mlx_get_default_stream(device._handle),
      ),
    );
  }

  factory MlxStream.defaultCpu() {
    _clearError();
    return MlxStream._(
      _checkHandle(
        'dart_mlx_default_cpu_stream',
        shim.dart_mlx_default_cpu_stream(),
      ),
    );
  }

  factory MlxStream.defaultGpu() {
    _clearError();
    return MlxStream._(
      _checkHandle(
        'dart_mlx_default_gpu_stream',
        shim.dart_mlx_default_gpu_stream(),
      ),
    );
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_stream_free);

  final shim.DartMlxStreamHandle _handle;
  bool _closed = false;

  // dart format off
  int get index { _ensureOpen(); final result = hooks.debugStreamGetIndexOverride?.call(_handle) ?? shim.dart_mlx_stream_get_index(_handle); if (result < 0) _throwError('dart_mlx_stream_get_index'); return result; }
  // dart format on

  MlxDevice get device {
    _ensureOpen();
    _clearError();
    return MlxDevice._(
      _checkHandle(
        'dart_mlx_stream_get_device',
        shim.dart_mlx_stream_get_device(_handle),
      ),
    );
  }

  bool equals(MlxStream other) {
    _ensureOpen();
    other._ensureOpen();
    return shim.dart_mlx_stream_equal(_handle, other._handle);
  }

  void synchronize() {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_stream_synchronize',
      shim.dart_mlx_stream_synchronize(_handle),
    );
  }

  void setAsDefault() {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_set_default_stream',
      shim.dart_mlx_set_default_stream(_handle),
    );
  }

  @override
  String toString() {
    _ensureOpen();
    _clearError();
    return _copyOwnedString(shim.dart_mlx_stream_tostring_copy(_handle));
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_stream_free(_handle);
  }

  // dart format off
  void _ensureOpen() { if (_closed) throw StateError('MlxStream has been closed.'); }
  // dart format on
}

/// High-level distributed group handle.
final class MlxDistributedGroup {
  MlxDistributedGroup._(this._handle) {
    _finalizer.attach(this, _handle, detach: this);
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_distributed_group_free);

  final shim.DartMlxGroupHandle _handle;
  bool _closed = false;

  int get rank {
    _ensureOpen();
    return shim.dart_mlx_distributed_group_rank(_handle);
  }

  int get size {
    _ensureOpen();
    return shim.dart_mlx_distributed_group_size(_handle);
  }

  // dart format off
  MlxDistributedGroup split(int color, int key) { _ensureOpen(); _clearError(); final handle = hooks.debugDistributedGroupSplitOverride?.call(_handle, color, key) ?? shim.dart_mlx_distributed_group_split(_handle, color, key); return MlxDistributedGroup._(_checkHandle('dart_mlx_distributed_group_split', handle)); }
  // dart format on

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _finalizer.detach(this);
    shim.dart_mlx_distributed_group_free(_handle);
  }

  // dart format off
  void _ensureOpen() { if (_closed) throw StateError('MlxDistributedGroup has been closed.'); }
  // dart format on
}

/// Distributed collectives.
abstract final class MlxDistributed {
  static bool isAvailable() => shim.dart_mlx_distributed_is_available();

  static MlxDistributedGroup init({bool strict = true}) {
    _clearError();
    return MlxDistributedGroup._(
      _checkHandle(
        'dart_mlx_distributed_init',
        shim.dart_mlx_distributed_init(strict),
      ),
    );
  }

  static MlxArray allGather(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => _callUnary(
    'dart_mlx_distributed_all_gather',
    input,
    group,
    stream,
    shim.dart_mlx_distributed_all_gather,
  );

  static MlxArray allSum(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => _callUnary(
    'dart_mlx_distributed_all_sum',
    input,
    group,
    stream,
    shim.dart_mlx_distributed_all_sum,
  );

  static MlxArray allMax(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => _callUnary(
    'dart_mlx_distributed_all_max',
    input,
    group,
    stream,
    shim.dart_mlx_distributed_all_max,
  );

  static MlxArray allMin(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => _callUnary(
    'dart_mlx_distributed_all_min',
    input,
    group,
    stream,
    shim.dart_mlx_distributed_all_min,
  );

  static MlxArray sumScatter(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => _callUnary(
    'dart_mlx_distributed_sum_scatter',
    input,
    group,
    stream,
    shim.dart_mlx_distributed_sum_scatter,
  );

  // dart format off
  static MlxArray send(MlxArray input, int dst, {MlxDistributedGroup? group, MlxStream? stream}) { _clearError(); final handle = hooks.debugDistributedSendOverride?.call(input._handle, dst, group?._handle ?? ffi.nullptr, stream?._handle ?? ffi.nullptr) ?? shim.dart_mlx_distributed_send(input._handle, dst, group?._handle ?? ffi.nullptr, stream?._handle ?? ffi.nullptr); return MlxArray._(_checkHandle('dart_mlx_distributed_send', handle)); }
  // dart format on

  // dart format off
  static MlxArray recvLike(MlxArray like, int src, {MlxDistributedGroup? group, MlxStream? stream}) { _clearError(); final handle = hooks.debugDistributedRecvLikeOverride?.call(like._handle, src, group?._handle ?? ffi.nullptr, stream?._handle ?? ffi.nullptr) ?? shim.dart_mlx_distributed_recv_like(like._handle, src, group?._handle ?? ffi.nullptr, stream?._handle ?? ffi.nullptr); return MlxArray._(_checkHandle('dart_mlx_distributed_recv_like', handle)); }
  // dart format on

  // dart format off
  static MlxArray recv(List<int> shape, MlxDType dtype, int src, {MlxDistributedGroup? group, MlxStream? stream}) => _withInts(shape, (shapePtr, shapeLen) { _clearError(); final handle = hooks.debugDistributedRecvOverride?.call(shapePtr, shapeLen, dtype.value, src, group?._handle ?? ffi.nullptr, stream?._handle ?? ffi.nullptr) ?? shim.dart_mlx_distributed_recv(shapePtr, shapeLen, dtype.value, src, group?._handle ?? ffi.nullptr, stream?._handle ?? ffi.nullptr); return MlxArray._(_checkHandle('dart_mlx_distributed_recv', handle)); });
  // dart format on

  static MlxArray _callUnary(
    String op,
    MlxArray input,
    MlxDistributedGroup? group,
    MlxStream? stream,
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    )
    callback,
  ) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        op,
        callback(
          input._handle,
          group?._handle ?? ffi.nullptr,
          stream?._handle ?? ffi.nullptr,
        ),
      ),
    );
  }
}

final class MlxStreamModule {
  const MlxStreamModule._();

  MlxStream create() => MlxStream.create();

  MlxStream forDevice(MlxDevice device) => MlxStream.forDevice(device);

  MlxStream defaultFor(MlxDevice device) => MlxStream.defaultFor(device);

  MlxStream defaultCpu() => MlxStream.defaultCpu();

  MlxStream defaultGpu() => MlxStream.defaultGpu();
}

final class MlxDistributedModule {
  const MlxDistributedModule._();

  bool isAvailable() => MlxDistributed.isAvailable();

  MlxDistributedGroup init({bool strict = true}) =>
      MlxDistributed.init(strict: strict);

  MlxArray allGather(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => MlxDistributed.allGather(input, group: group, stream: stream);

  MlxArray allSum(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => MlxDistributed.allSum(input, group: group, stream: stream);

  MlxArray allMax(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => MlxDistributed.allMax(input, group: group, stream: stream);

  MlxArray allMin(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => MlxDistributed.allMin(input, group: group, stream: stream);

  MlxArray sumScatter(
    MlxArray input, {
    MlxDistributedGroup? group,
    MlxStream? stream,
  }) => MlxDistributed.sumScatter(input, group: group, stream: stream);

  // dart format off
  MlxArray send(MlxArray input, int dst, {MlxDistributedGroup? group, MlxStream? stream}) => MlxDistributed.send(input, dst, group: group, stream: stream);
  MlxArray recvLike(MlxArray like, int src, {MlxDistributedGroup? group, MlxStream? stream}) => MlxDistributed.recvLike(like, src, group: group, stream: stream);
  MlxArray recv(List<int> shape, MlxDType dtype, int src, {MlxDistributedGroup? group, MlxStream? stream}) => MlxDistributed.recv(shape, dtype, src, group: group, stream: stream);
  // dart format on
}
