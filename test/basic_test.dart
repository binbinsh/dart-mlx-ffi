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
