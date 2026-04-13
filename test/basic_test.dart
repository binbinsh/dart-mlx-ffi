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

void main() {
  test('exposes MLX version', () {
    final version = MlxVersion.current();
    expect(version, isNotEmpty);
    expect(version, contains('.'));
  });

  test('exposes allocator runtime stats', () {
    final active = MlxMemory.activeBytes();
    final cache = MlxMemory.cacheBytes();
    final cacheCount = MlxMemory.cacheCount();
    final peak = MlxMemory.peakBytes();
    final memoryLimit = MlxMemory.memoryLimitBytes();
    final cacheLimit = MlxMemory.cacheLimitBytes();
    final wiredLimit = MlxMemory.wiredLimitBytes();
    final resourceCount = MlxMemory.resourceCount();
    final resourceLimit = MlxMemory.resourceLimit();
    final commitCount = MlxMemory.commandBufferCommitCount();
    final pendingOutputCount = MlxMemory.pendingOutputCount();
    final temporaryCount = MlxMemory.temporaryCount();
    final bufferOpCount = MlxMemory.bufferOpCount();
    final bufferSizeBytes = MlxMemory.bufferSizeBytes();
    final streamCount = MlxMemory.streamCount();
    final setDataCount = MlxMemory.setDataCount();
    final sharedBufferCopyCount = MlxMemory.sharedBufferCopyCount();
    final allocationRequestCount = MlxMemory.allocationRequestCount();
    final cacheReuseHitCount = MlxMemory.cacheReuseHitCount();
    final newAllocationCount = MlxMemory.newAllocationCount();
    final heapAllocationCount = MlxMemory.heapAllocationCount();
    final deviceAllocationCount = MlxMemory.deviceAllocationCount();
    final metalNormAllocationCount = MlxMemory.metalNormAllocationCount();
    final metalNormSharedCopyCount = MlxMemory.metalNormSharedCopyCount();
    final metalMatmulAllocationCount = MlxMemory.metalMatmulAllocationCount();
    final metalMatmulSharedCopyCount =
        MlxMemory.metalMatmulSharedCopyCount();
    final metalQuantizedAllocationCount =
        MlxMemory.metalQuantizedAllocationCount();
    final metalQuantizedSharedCopyCount =
        MlxMemory.metalQuantizedSharedCopyCount();
    final metalSdpaAllocationCount = MlxMemory.metalSdpaAllocationCount();
    final metalSdpaSharedCopyCount = MlxMemory.metalSdpaSharedCopyCount();
    final donationRejectNotUniqueCount =
        MlxMemory.donationRejectNotUniqueCount();
    final donationRejectDescNotUniqueCount =
        MlxMemory.donationRejectDescNotUniqueCount();
    final donationRejectDataNotUniqueCount =
        MlxMemory.donationRejectDataNotUniqueCount();
    final donationRejectItemsizeCount = MlxMemory.donationRejectItemsizeCount();
    final donationRejectOversizeCount = MlxMemory.donationRejectOversizeCount();
    final donationRejectLayoutCount = MlxMemory.donationRejectLayoutCount();
    final commonCopyRejectDescNotUniqueCount =
        MlxMemory.commonCopyRejectDescNotUniqueCount();
    final commonCopyRejectDataNotUniqueCount =
        MlxMemory.commonCopyRejectDataNotUniqueCount();
    final commonBinaryRejectDescNotUniqueCount =
        MlxMemory.commonBinaryRejectDescNotUniqueCount();
    final commonBinaryRejectDataNotUniqueCount =
        MlxMemory.commonBinaryRejectDataNotUniqueCount();
    final commonUnaryRejectDescNotUniqueCount =
        MlxMemory.commonUnaryRejectDescNotUniqueCount();
    final commonUnaryRejectDataNotUniqueCount =
        MlxMemory.commonUnaryRejectDataNotUniqueCount();
    final commonBinaryDataNotUniqueScalarVectorCount =
        MlxMemory.commonBinaryDataNotUniqueScalarVectorCount();
    final commonBinaryDataNotUniqueVectorScalarCount =
        MlxMemory.commonBinaryDataNotUniqueVectorScalarCount();
    final commonBinaryDataNotUniqueVectorVectorCount =
        MlxMemory.commonBinaryDataNotUniqueVectorVectorCount();
    final commonBinaryDataNotUniqueGeneralCount =
        MlxMemory.commonBinaryDataNotUniqueGeneralCount();
    final commonBinaryAddDataNotUniqueVectorVectorCount =
        MlxMemory.commonBinaryAddDataNotUniqueVectorVectorCount();
    final commonBinaryAddDataNotUniqueGeneralCount =
        MlxMemory.commonBinaryAddDataNotUniqueGeneralCount();
    final commonBinaryMultiplyDataNotUniqueVectorVectorCount =
        MlxMemory.commonBinaryMultiplyDataNotUniqueVectorVectorCount();
    final commonBinaryMultiplyDataNotUniqueGeneralCount =
        MlxMemory.commonBinaryMultiplyDataNotUniqueGeneralCount();
    final stats = MlxMemory.allocatorStats();

    expect(active, greaterThanOrEqualTo(0));
    expect(cache, greaterThanOrEqualTo(0));
    expect(cacheCount, greaterThanOrEqualTo(0));
    expect(peak, greaterThanOrEqualTo(0));
    expect(memoryLimit, greaterThanOrEqualTo(0));
    expect(cacheLimit, greaterThanOrEqualTo(0));
    expect(wiredLimit, greaterThanOrEqualTo(0));
    expect(resourceCount, greaterThanOrEqualTo(0));
    expect(resourceLimit, greaterThanOrEqualTo(0));
    expect(commitCount, greaterThanOrEqualTo(0));
    expect(pendingOutputCount, greaterThanOrEqualTo(0));
    expect(temporaryCount, greaterThanOrEqualTo(0));
    expect(bufferOpCount, greaterThanOrEqualTo(0));
    expect(bufferSizeBytes, greaterThanOrEqualTo(0));
    expect(streamCount, greaterThanOrEqualTo(0));
    expect(setDataCount, greaterThanOrEqualTo(0));
    expect(sharedBufferCopyCount, greaterThanOrEqualTo(0));
    expect(allocationRequestCount, greaterThanOrEqualTo(0));
    expect(cacheReuseHitCount, greaterThanOrEqualTo(0));
    expect(newAllocationCount, greaterThanOrEqualTo(0));
    expect(heapAllocationCount, greaterThanOrEqualTo(0));
    expect(deviceAllocationCount, greaterThanOrEqualTo(0));
    expect(metalNormAllocationCount, greaterThanOrEqualTo(0));
    expect(metalNormSharedCopyCount, greaterThanOrEqualTo(0));
    expect(metalMatmulAllocationCount, greaterThanOrEqualTo(0));
    expect(metalMatmulSharedCopyCount, greaterThanOrEqualTo(0));
    expect(metalQuantizedAllocationCount, greaterThanOrEqualTo(0));
    expect(metalQuantizedSharedCopyCount, greaterThanOrEqualTo(0));
    expect(metalSdpaAllocationCount, greaterThanOrEqualTo(0));
    expect(metalSdpaSharedCopyCount, greaterThanOrEqualTo(0));
    expect(donationRejectNotUniqueCount, greaterThanOrEqualTo(0));
    expect(donationRejectDescNotUniqueCount, greaterThanOrEqualTo(0));
    expect(donationRejectDataNotUniqueCount, greaterThanOrEqualTo(0));
    expect(donationRejectItemsizeCount, greaterThanOrEqualTo(0));
    expect(donationRejectOversizeCount, greaterThanOrEqualTo(0));
    expect(donationRejectLayoutCount, greaterThanOrEqualTo(0));
    expect(commonCopyRejectDescNotUniqueCount, greaterThanOrEqualTo(0));
    expect(commonCopyRejectDataNotUniqueCount, greaterThanOrEqualTo(0));
    expect(commonBinaryRejectDescNotUniqueCount, greaterThanOrEqualTo(0));
    expect(commonBinaryRejectDataNotUniqueCount, greaterThanOrEqualTo(0));
    expect(commonUnaryRejectDescNotUniqueCount, greaterThanOrEqualTo(0));
    expect(commonUnaryRejectDataNotUniqueCount, greaterThanOrEqualTo(0));
    expect(commonBinaryDataNotUniqueScalarVectorCount, greaterThanOrEqualTo(0));
    expect(commonBinaryDataNotUniqueVectorScalarCount, greaterThanOrEqualTo(0));
    expect(commonBinaryDataNotUniqueVectorVectorCount, greaterThanOrEqualTo(0));
    expect(commonBinaryDataNotUniqueGeneralCount, greaterThanOrEqualTo(0));
    expect(
      commonBinaryAddDataNotUniqueVectorVectorCount,
      greaterThanOrEqualTo(0),
    );
    expect(commonBinaryAddDataNotUniqueGeneralCount, greaterThanOrEqualTo(0));
    expect(
      commonBinaryMultiplyDataNotUniqueVectorVectorCount,
      greaterThanOrEqualTo(0),
    );
    expect(
      commonBinaryMultiplyDataNotUniqueGeneralCount,
      greaterThanOrEqualTo(0),
    );
    expect(stats.activeBytes, active);
    expect(stats.cacheBytes, cache);
    expect(stats.cacheCount, cacheCount);
    expect(stats.peakBytes, peak);
    expect(stats.memoryLimitBytes, memoryLimit);
    expect(stats.cacheLimitBytes, cacheLimit);
    expect(stats.wiredLimitBytes, wiredLimit);
    expect(stats.resourceCount, resourceCount);
    expect(stats.resourceLimit, resourceLimit);
    expect(stats.commandBufferCommitCount, commitCount);
    expect(stats.pendingOutputCount, pendingOutputCount);
    expect(stats.temporaryCount, temporaryCount);
    expect(stats.bufferOpCount, bufferOpCount);
    expect(stats.bufferSizeBytes, bufferSizeBytes);
    expect(stats.streamCount, streamCount);
    expect(stats.setDataCount, setDataCount);
    expect(stats.sharedBufferCopyCount, sharedBufferCopyCount);
    expect(stats.allocationRequestCount, allocationRequestCount);
    expect(stats.cacheReuseHitCount, cacheReuseHitCount);
    expect(stats.newAllocationCount, newAllocationCount);
    expect(stats.heapAllocationCount, heapAllocationCount);
    expect(stats.deviceAllocationCount, deviceAllocationCount);
    expect(stats.metalNormAllocationCount, metalNormAllocationCount);
    expect(stats.metalNormSharedCopyCount, metalNormSharedCopyCount);
    expect(stats.metalMatmulAllocationCount, metalMatmulAllocationCount);
    expect(stats.metalMatmulSharedCopyCount, metalMatmulSharedCopyCount);
    expect(
      stats.metalQuantizedAllocationCount,
      metalQuantizedAllocationCount,
    );
    expect(
      stats.metalQuantizedSharedCopyCount,
      metalQuantizedSharedCopyCount,
    );
    expect(stats.metalSdpaAllocationCount, metalSdpaAllocationCount);
    expect(stats.metalSdpaSharedCopyCount, metalSdpaSharedCopyCount);
    expect(stats.donationRejectNotUniqueCount, donationRejectNotUniqueCount);
    expect(
      stats.donationRejectDescNotUniqueCount,
      donationRejectDescNotUniqueCount,
    );
    expect(
      stats.donationRejectDataNotUniqueCount,
      donationRejectDataNotUniqueCount,
    );
    expect(stats.donationRejectItemsizeCount, donationRejectItemsizeCount);
    expect(stats.donationRejectOversizeCount, donationRejectOversizeCount);
    expect(stats.donationRejectLayoutCount, donationRejectLayoutCount);
    expect(
      stats.commonCopyRejectDescNotUniqueCount,
      commonCopyRejectDescNotUniqueCount,
    );
    expect(
      stats.commonCopyRejectDataNotUniqueCount,
      commonCopyRejectDataNotUniqueCount,
    );
    expect(
      stats.commonBinaryRejectDescNotUniqueCount,
      commonBinaryRejectDescNotUniqueCount,
    );
    expect(
      stats.commonBinaryRejectDataNotUniqueCount,
      commonBinaryRejectDataNotUniqueCount,
    );
    expect(
      stats.commonUnaryRejectDescNotUniqueCount,
      commonUnaryRejectDescNotUniqueCount,
    );
    expect(
      stats.commonUnaryRejectDataNotUniqueCount,
      commonUnaryRejectDataNotUniqueCount,
    );
    expect(
      stats.commonBinaryDataNotUniqueScalarVectorCount,
      commonBinaryDataNotUniqueScalarVectorCount,
    );
    expect(
      stats.commonBinaryDataNotUniqueVectorScalarCount,
      commonBinaryDataNotUniqueVectorScalarCount,
    );
    expect(
      stats.commonBinaryDataNotUniqueVectorVectorCount,
      commonBinaryDataNotUniqueVectorVectorCount,
    );
    expect(
      stats.commonBinaryDataNotUniqueGeneralCount,
      commonBinaryDataNotUniqueGeneralCount,
    );
    expect(
      stats.commonBinaryAddDataNotUniqueVectorVectorCount,
      commonBinaryAddDataNotUniqueVectorVectorCount,
    );
    expect(
      stats.commonBinaryAddDataNotUniqueGeneralCount,
      commonBinaryAddDataNotUniqueGeneralCount,
    );
    expect(
      stats.commonBinaryMultiplyDataNotUniqueVectorVectorCount,
      commonBinaryMultiplyDataNotUniqueVectorVectorCount,
    );
    expect(
      stats.commonBinaryMultiplyDataNotUniqueGeneralCount,
      commonBinaryMultiplyDataNotUniqueGeneralCount,
    );
  });

  test('tracks fresh buffer assignments and shared-buffer reuse', () {
    final beforeSetData = MlxMemory.setDataCount();
    final beforeSharedCopy = MlxMemory.sharedBufferCopyCount();
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final reshaped = a.reshape([4, 1]);

    try {
      a.eval();
      reshaped.eval();
      expect(MlxMemory.setDataCount(), greaterThan(beforeSetData));
      expect(MlxMemory.sharedBufferCopyCount(), greaterThan(beforeSharedCopy));
    } finally {
      reshaped.close();
      a.close();
    }
  });

  test('formats MlxException', () {
    expect(const MlxException('boom').toString(), 'MlxException: boom');
    expect(
      const MlxException('boom', code: 7).toString(),
      'MlxException(code: 7, message: boom)',
    );
  });

  test('decodes native error messages through the internal hook', () {
    MlxVersion.current();
    final message = 'native boom'.toNativeUtf8().cast<ffi.Char>();
    try {
      hooks.debugDispatchError!(message);
      expect(
        () => MlxVersion.current(),
        isNot(throwsException),
      );
    } finally {
      calloc.free(message);
      hooks.resetDebugHooks();
    }
  });

  test('surfaces string-copy failures as MlxException', () {
    hooks.debugVersionCopyOverride = () => ffi.nullptr.cast();
    try {
      expect(() => MlxVersion.current(), throwsA(isA<MlxException>()));
    } finally {
      hooks.resetDebugHooks();
    }
  });

  test('exposes default device information', () {
    final device = MlxDevice.defaultDevice();
    final other = MlxDevice.defaultDevice();
    try {
      expect(device.toString(), isNotEmpty);
      expect(device.index, greaterThanOrEqualTo(0));
      expect(
        device.type,
        anyOf(raw.mlx_device_type_.MLX_CPU, raw.mlx_device_type_.MLX_GPU),
      );
      expect(device.equals(other), isTrue);
      expect(device.isClosed, isFalse);
      device.close();
      expect(device.isClosed, isTrue);
      device.close();
      expect(() => device.toString(), throwsStateError);
      expect(() => device.index, throwsStateError);
      expect(() => device.type, throwsStateError);
    } finally {
      other.close();
      device.close();
    }
  });

  test('creates arrays across supported high-level dtypes', () {
    final bools = MlxArray.fromBoolList([true, false, true, false], shape: [2, 2]);
    final ints = MlxArray.fromInt32List([1, 2, 3, 4], shape: [2, 2]);
    final ints64 = MlxArray.fromInt64List([1, 2, 3, 4], shape: [2, 2]);
    final uints64 = MlxArray.fromUint64List([1, 2, 3, 4], shape: [2, 2]);
    final floats32 = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final floats64 = MlxArray.fromFloat64List([1, 2, 3, 4], shape: [2, 2]);
    final zeros = MlxArray.zeros([2, 2]);
    final ones = MlxArray.ones([2, 2]);
    final full = MlxArray.full([2, 2], 7);
    final range = MlxArray.arange(0, 4, 1);

    try {
      expect(bools.dtype, MlxDType.MLX_BOOL);
      expect(ints.dtype, MlxDType.MLX_INT32);
      expect(ints64.dtype, MlxDType.MLX_INT64);
      expect(uints64.dtype, MlxDType.MLX_UINT64);
      expect(floats32.dtype, MlxDType.MLX_FLOAT32);
      expect(floats64.dtype, MlxDType.MLX_FLOAT64);

      floats32.eval();
      expect(bools.shape, <int>[2, 2]);
      expect(ints.ndim, 2);
      expect(floats32.size, 4);

      expect(bools.toList(), <Object>[true, false, true, false]);
      expect(ints.toList(), <Object>[1, 2, 3, 4]);
      expect(ints64.toList(), <Object>[1, 2, 3, 4]);
      expect(uints64.toList(), <Object>[1, 2, 3, 4]);
      expect(floats32.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(floats64.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(zeros.toList(), <Object>[0.0, 0.0, 0.0, 0.0]);
      expect(ones.toList(), <Object>[1.0, 1.0, 1.0, 1.0]);
      expect(full.toList(), <Object>[7.0, 7.0, 7.0, 7.0]);
      expect(range.toList(), <Object>[0.0, 1.0, 2.0, 3.0]);
      expect(floats32.toString(), contains('array'));
    } finally {
      range.close();
      full.close();
      ones.close();
      zeros.close();
      floats64.close();
      floats32.close();
      uints64.close();
      ints64.close();
      ints.close();
      bools.close();
    }
  });

  test('supports reshape and transpose', () {
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final reshaped = a.reshape([4, 1]);
    final transposed = a.transpose();
    final transposedViaGetter = a.T;

    try {
      expect(reshaped.shape, <int>[4, 1]);
      expect(reshaped.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(transposed.shape, <int>[2, 2]);
      expect(transposed.toList(), <Object>[1.0, 3.0, 2.0, 4.0]);
      expect(transposedViaGetter.toList(), <Object>[1.0, 3.0, 2.0, 4.0]);
    } finally {
      transposedViaGetter.close();
      transposed.close();
      reshaped.close();
      a.close();
    }
  });

  test('supports tanh, variance, and addmm helpers', () {
    final input = MlxArray.fromFloat32List([1, -1, 2, -2], shape: [2, 2]);
    final bias = MlxArray.fromFloat32List([1, 1, 1, 1], shape: [2, 2]);
    final lhs = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final rhs = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);

    final tanhOut = input.tanh();
    final varAll = input.variance();
    final varAxis = input.variance(axis: 1, keepDims: true);
    final addmmOut = bias.addmm(lhs, rhs);

    try {
      expect(tanhOut.shape, <int>[2, 2]);
      expect(
        tanhOut.toList().cast<double>(),
        everyElement(inInclusiveRange(-1.0, 1.0)),
      );
      expect(varAll.shape, isEmpty);
      expect(varAll.toList().single, isA<double>());
      expect(varAxis.shape, <int>[2, 1]);
      expect(addmmOut.toList(), <Object>[20.0, 23.0, 44.0, 51.0]);
    } finally {
      addmmOut.close();
      varAxis.close();
      varAll.close();
      tanhOut.close();
      rhs.close();
      lhs.close();
      bias.close();
      input.close();
    }
  });

}
