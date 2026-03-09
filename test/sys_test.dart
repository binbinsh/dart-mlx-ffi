// ignore_for_file: unused_import

@TestOn('mac-os')
library;

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/raw.dart' as raw;
import 'package:dart_mlx_ffi/src/internal_hooks.dart' as hooks;
import 'package:dart_mlx_ffi/src/shim_bindings.dart' as shim;

void main() {
  setUp(hooks.resetDebugHooks);
  tearDown(hooks.resetDebugHooks);

  test('throws state errors after stream close', () {
    final device = MlxDevice.defaultDevice();
    final stream = MlxStream.defaultFor(device);
    final other = MlxStream.defaultCpu();

    try {
      stream.close();
      stream.close();

      expect(_captureStateError(() => stream.index), isNotNull);
      expect(_captureStateError(() => stream.device), isNotNull);
      expect(_captureStateError(() => stream.equals(other)), isNotNull);
      expect(_captureStateError(stream.synchronize), isNotNull);
      expect(_captureStateError(stream.setAsDefault), isNotNull);
      expect(_captureStateError(stream.toString), isNotNull);
    } finally {
      other.close();
      device.close();
    }
  });

  test('covers sys debug-hooked stream branches', () {
    hooks.debugStreamNewOverride = () => shim.dart_mlx_default_cpu_stream();
    final created = MlxStream.create();
    try {
      expect(created.toString(), isNotEmpty);
    } finally {
      created.close();
    }

    final failingIndexStream = MlxStream.defaultCpu();
    hooks.debugStreamGetIndexOverride = (_) => -1;
    try {
      expect(() => failingIndexStream.index, throwsA(isA<MlxException>()));
    } finally {
      failingIndexStream.close();
      hooks.debugStreamGetIndexOverride = null;
    }
  });

  test('covers sys debug-hooked distributed split branch', () {
    if (mx.distributed.isAvailable()) {
      final group = mx.distributed.init(strict: false);
      hooks.debugDistributedGroupSplitOverride = (groupHandle, color, key) =>
          shim.dart_mlx_distributed_init(false);
      try {
        final split = group.split(7, 9);
        try {
          expect(split.size, greaterThanOrEqualTo(1));
        } finally {
          split.close();
        }
      } finally {
        group.close();
        hooks.debugDistributedGroupSplitOverride = null;
      }
    }
  });

  test('covers sys debug-hooked distributed recv branches', () {
    final payload = MlxArray.fromFloat32List([1, 2], shape: [2]);
    hooks.debugDistributedSendOverride = _fakeDistributedArrayHandle;
    hooks.debugDistributedRecvLikeOverride = _fakeDistributedArrayHandle;
    hooks.debugDistributedRecvOverride = _fakeDistributedRecvHandle;
    try {
      final sent = MlxDistributed.send(payload, 0);
      final recvLike = MlxDistributed.recvLike(payload, 0);
      final recv = MlxDistributed.recv([2], MlxDType.MLX_FLOAT32, 0);
      final wrappedRecvLike = mx.distributed.recvLike(payload, 0);
      final wrappedRecv = mx.distributed.recv([2], MlxDType.MLX_FLOAT32, 0);

      expect(sent.shape, [2]);
      expect(recvLike.shape, [2]);
      expect(recv.shape, [2]);
      expect(wrappedRecvLike.shape, [2]);
      expect(wrappedRecv.shape, [2]);

      sent.close();
      recvLike.close();
      recv.close();
      wrappedRecvLike.close();
      wrappedRecv.close();
    } finally {
      payload.close();
    }
  });

  test('supports device info and stream helpers', () {
    final device = MlxDevice.defaultDevice();
    final stream = MlxStream.defaultFor(device);
    final cpu = MlxStream.defaultCpu();
    final custom = MlxStream.forDevice(device);
    MlxStream? gpu;

    try {
      final info = device.info;
      expect(info.values, isNotEmpty);
      expect(info.keys, isNotEmpty);
      expect(stream.index, greaterThanOrEqualTo(0));
      expect(stream.toString(), isNotEmpty);
      expect(stream.equals(stream), isTrue);
      final streamDevice = stream.device;
      try {
        expect(
          streamDevice.type,
          anyOf(raw.mlx_device_type_.MLX_CPU, raw.mlx_device_type_.MLX_GPU),
        );
      } finally {
        streamDevice.close();
      }
      stream.synchronize();
      stream.setAsDefault();
      custom.setAsDefault();
      final current = MlxStream.defaultFor(device);
      try {
        expect(current.equals(custom), isTrue);
      } finally {
        current.close();
        stream.setAsDefault();
      }
      MlxDevice.setDefault(device);

      expect(cpu.toString(), isNotEmpty);
      try {
        gpu = MlxStream.defaultGpu();
        expect(gpu.toString(), isNotEmpty);
      } on MlxException {
        gpu = null;
      }
    } finally {
      gpu?.close();
      custom.close();
      cpu.close();
      stream.close();
      device.close();
    }
  });

  test('supports distributed availability and group wrappers', () {
    final available = mx.distributed.isAvailable();
    expect(available, anyOf(isTrue, isFalse));
    if (!available) {
      return;
    }
    final group = mx.distributed.init(strict: false);
    try {
      expect(group.rank, greaterThanOrEqualTo(0));
      expect(group.size, greaterThanOrEqualTo(1));
      MlxDistributedGroup? split;
      try {
        split = group.split(0, group.rank);
        expect(split.rank, greaterThanOrEqualTo(0));
        expect(split.size, greaterThanOrEqualTo(1));
      } on MlxException {
        // Some backends require stronger runtime setup for subgroup creation.
      } finally {
        split?.close();
      }

      final payload = MlxArray.fromFloat32List([1, 2], shape: [2]);
      final stream = MlxStream.defaultCpu();
      try {
        void expectArrayLike(MlxArray value) {
          expect(value.shape, isNotEmpty);
          value.close();
        }

        void expectCollective(MlxArray Function() callback) {
          try {
            expectArrayLike(callback());
          } on MlxException {
            // Some environments expose distributed but still need launcher/runtime wiring.
          }
        }

        expectCollective(
          () => mx.distributed.allGather(payload, group: group, stream: stream),
        );
        expectCollective(
          () => mx.distributed.allSum(payload, group: group, stream: stream),
        );
        expectCollective(
          () => mx.distributed.allMax(payload, group: group, stream: stream),
        );
        expectCollective(
          () => mx.distributed.allMin(payload, group: group, stream: stream),
        );
        expectCollective(
          () =>
              mx.distributed.sumScatter(payload, group: group, stream: stream),
        );
        expectCollective(
          () => mx.distributed.send(
            payload,
            group.rank,
            group: group,
            stream: stream,
          ),
        );
        expectCollective(
          () => mx.distributed.recvLike(
            payload,
            group.rank,
            group: group,
            stream: stream,
          ),
        );
        expectCollective(
          () => mx.distributed.recv(
            payload.shape,
            payload.dtype,
            group.rank,
            group: group,
            stream: stream,
          ),
        );
      } finally {
        stream.close();
        payload.close();
      }
    } finally {
      group.close();
    }
  });

  test('throws state errors after distributed group close', () {
    if (!mx.distributed.isAvailable()) {
      return;
    }

    final group = mx.distributed.init(strict: false);
    group.close();
    group.close();

    expect(_captureStateError(() => group.rank), isNotNull);
    expect(_captureStateError(() => group.size), isNotNull);
    expect(_captureStateError(() => group.split(0, 0)), isNotNull);
  });
}

StateError? _captureStateError(Object? Function() callback) {
  try {
    callback();
  } on StateError catch (error) {
    return error;
  }
  return null;
}

ffi.Pointer<ffi.Void> _fakeDistributedArrayHandle(
  ffi.Pointer<ffi.Void> inputHandle,
  int peer,
  ffi.Pointer<ffi.Void> groupHandle,
  ffi.Pointer<ffi.Void> streamHandle,
) {
  final data = calloc<ffi.Float>(2);
  final shape = calloc<ffi.Int>(1);
  try {
    data[0] = 3;
    data[1] = 4;
    shape[0] = 2;
    return shim.dart_mlx_array_from_float32(data.cast(), shape, 1);
  } finally {
    calloc.free(shape);
  }
}

ffi.Pointer<ffi.Void> _fakeDistributedRecvHandle(
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  int dtypeValue,
  int peer,
  ffi.Pointer<ffi.Void> groupHandle,
  ffi.Pointer<ffi.Void> streamHandle,
) {
  final size = List<int>.generate(
    shapeLen,
    (index) => shape[index],
  ).fold<int>(1, (product, value) => product * value);
  final data = calloc<ffi.Float>(size);
  for (var index = 0; index < size; index++) {
    data[index] = index + 1;
  }
  return shim.dart_mlx_array_from_float32(data.cast(), shape, shapeLen);
}
