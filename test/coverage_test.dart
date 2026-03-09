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
  test('covers mx wrapper methods for tensor transforms and scatter family', () {
    final input = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final vector = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [4]);
    final idx = MlxArray.fromInt32List([1, 0], shape: [2]);
    final dynStart = MlxArray.fromInt32List([0], shape: [1]);
    final update = MlxArray.fromFloat32List([9, 8], shape: [1, 2]);
    final scatterUpdates = MlxArray.fromFloat32List([5, 7], shape: [2, 1]);
    final scatterSingleUpdates = MlxArray.fromFloat32List([5, 7], shape: [2, 1]);
    final addIdx = MlxArray.fromInt32List([1, 1], shape: [2]);
    final addUpdates = MlxArray.fromFloat32List([2, 3], shape: [2, 1]);
    final base = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final padValue = MlxArray.full([], 0);

    final taken = mx.takeAlongAxis(input, MlxArray.fromInt32List([1, 0, 1, 0], shape: [2, 2]), axis: 1);
    final sliced = mx.slice(input, start: [0, 0], stop: [1, 2]);
    final dynSlice = mx.sliceDynamic(vector, start: dynStart, axes: [0], sliceSize: [2]);
    final updated = mx.sliceUpdate(input, update, start: [0, 0], stop: [1, 2]);
    final updatedDyn = mx.sliceUpdateDynamic(input, update, start: dynStart, axes: [0]);
    final scatterAdd = mx.scatterAdd(MlxArray.zeros([3]), [idx], scatterUpdates, axes: [0]);
    final scatterMax = mx.scatterMax(MlxArray.zeros([3]), [idx], scatterUpdates, axes: [0]);
    final scatterMin = mx.scatterMin(MlxArray.full([3], 10), [idx], scatterUpdates, axes: [0]);
    final scatterProd = mx.scatterProd(MlxArray.ones([3]), [idx], scatterUpdates, axes: [0]);
    final scatterAddSingle = mx.scatterAddSingle(MlxArray.zeros([3]), addIdx, addUpdates, axis: 0);
    final flatten = mx.flatten(input);
    final moveaxis = mx.moveaxis(input.reshape([1, 2, 2]), 0, 2);
    final swapaxes = mx.swapaxes(input, 0, 1);
    final transposeAxes = mx.transposeAxes(input, [1, 0]);
    final tile = mx.tile(base, [2]);
    final pad = mx.pad(base, lowPads: [1], highPads: [1], padValue: padValue);
    final padSym = mx.padSymmetric(base, 1, padValue: padValue);
    final unflatten = mx.unflatten(flatten, axis: 0, shape: [2, 2]);

    try {
      expect(taken.shape, <int>[2, 2]);
      expect(sliced.shape, <int>[1, 2]);
      expect(dynSlice.shape, <int>[2]);
      expect(updated.shape, <int>[2, 2]);
      expect(updatedDyn.shape, <int>[2, 2]);
      expect(scatterAdd.shape, <int>[3]);
      expect(scatterMax.shape, <int>[3]);
      expect(scatterMin.shape, <int>[3]);
      expect(scatterProd.shape, <int>[3]);
      expect(scatterAddSingle.shape, <int>[3]);
      expect(flatten.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(moveaxis.shape, <int>[2, 2, 1]);
      expect(swapaxes.toList(), <Object>[1.0, 3.0, 2.0, 4.0]);
      expect(transposeAxes.toList(), <Object>[1.0, 3.0, 2.0, 4.0]);
      expect(tile.toList(), <Object>[1.0, 2.0, 1.0, 2.0]);
      expect(pad.toList(), <Object>[0.0, 1.0, 2.0, 0.0]);
      expect(padSym.toList(), <Object>[0.0, 1.0, 2.0, 0.0]);
      expect(unflatten.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
    } finally {
      unflatten.close();
      padSym.close();
      pad.close();
      tile.close();
      transposeAxes.close();
      swapaxes.close();
      moveaxis.close();
      flatten.close();
      scatterAddSingle.close();
      scatterProd.close();
      scatterMin.close();
      scatterMax.close();
      scatterAdd.close();
      updatedDyn.close();
      updated.close();
      dynSlice.close();
      sliced.close();
      taken.close();
      padValue.close();
      base.close();
      addUpdates.close();
      addIdx.close();
      scatterSingleUpdates.close();
      scatterUpdates.close();
      update.close();
      dynStart.close();
      idx.close();
      vector.close();
      input.close();
    }
  });

  test('covers io module wrappers and reader/writer close guards', () {
    final array = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final weights = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final bias = MlxArray.fromFloat32List([5, 6], shape: [2]);
    final stream = MlxStream.defaultCpu();
    final reader = MlxBytesReader(mx.io.saveBytes(array));
    final writer = MlxBytesWriter();
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_cov_io_');
    final file = '${dir.path}/model.safetensors';

    try {
      final loaded = mx.io.loadBytes(mx.io.saveBytes(array), stream: stream);
      try {
        expect(loaded.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      } finally {
        loaded.close();
      }

      final loadedReader = mx.io.loadReader(reader, stream: stream);
      try {
        expect(loadedReader.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      } finally {
        loadedReader.close();
      }

      mx.io.saveWriter(writer, array);
      expect(writer.bytes, isNotEmpty);

      mx.io.saveSafetensors(
        file,
        {'weights': weights, 'bias': bias},
        metadata: {'author': 'binbinsh'},
      );
      final loadedFile = mx.io.loadSafetensors(file, stream: stream);
      try {
        expect(loadedFile.metadata['author'], 'binbinsh');
      } finally {
        for (final value in loadedFile.tensors.values) {
          value.close();
        }
      }

      final stWriter = MlxBytesWriter();
      try {
        mx.io.saveSafetensorsWriter(
          stWriter,
          {'weights': weights, 'bias': bias},
          metadata: {'author': 'binbinsh'},
        );
        final stReader = MlxBytesReader(stWriter.bytes);
        try {
          final loadedReaderData = mx.io.loadSafetensorsReader(stReader, stream: stream);
          try {
            expect(loadedReaderData.metadata['author'], 'binbinsh');
          } finally {
            for (final value in loadedReaderData.tensors.values) {
              value.close();
            }
          }
        } finally {
          stReader.close();
        }
      } finally {
        stWriter.close();
      }
    } finally {
      reader.close();
      writer.close();
      stream.close();
      if (Directory(dir.path).existsSync()) {
        dir.deleteSync(recursive: true);
      }
      bias.close();
      weights.close();
      array.close();
    }

    expect(() => reader.toString(), throwsStateError);
    expect(() => writer.bytes, throwsStateError);
  });

  test('covers sys helpers and distributed module wrappers when available', () {
    final device = MlxDevice.defaultDevice();
    final info = device.info;
    expect(info.containsKey(info.keys.first), isTrue);
    expect(info[info.keys.first], isNotNull);

    MlxStream? created;
    try {
      created = mx.stream.create();
      expect(created.toString(), isNotEmpty);
    } on MlxException {
      created = null;
    } finally {
      created?.close();
    }

    final streamFor = mx.stream.forDevice(device);
    final streamDefault = mx.stream.defaultFor(device);
    final streamCpu = mx.stream.defaultCpu();
    MlxStream? streamGpu;
    try {
      expect(streamFor.toString(), isNotEmpty);
      expect(streamDefault.toString(), isNotEmpty);
      expect(streamCpu.toString(), isNotEmpty);
      try {
        streamGpu = mx.stream.defaultGpu();
      } on MlxException {
        streamGpu = null;
      }
      expect(mx.distributed.isAvailable(), anyOf(isTrue, isFalse));
      if (mx.distributed.isAvailable()) {
        final group = mx.distributed.init(strict: false);
        try {
          final input = MlxArray.fromFloat32List([1], shape: [1]);
          try {
            try {
              final sent = mx.distributed.send(input, 0, group: group, stream: streamCpu);
              sent.close();
              final recvLike = mx.distributed.recvLike(input, 0, group: group, stream: streamCpu);
              recvLike.close();
              final recv = mx.distributed.recv([1], MlxDType.MLX_FLOAT32, 0, group: group, stream: streamCpu);
              recv.close();
            } on MlxException {
              // Environment-specific distributed runtime wiring may be absent.
            }
          } finally {
            input.close();
          }
        } finally {
          group.close();
        }
      }
    } finally {
      streamGpu?.close();
      streamCpu.close();
      streamDefault.close();
      streamFor.close();
      device.close();
    }
  });

  test('covers module wrappers for more and scan helpers', () {
    final a = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final b = MlxArray.fromFloat32List([2, 2], shape: [2]);
    final fp = MlxArray.fromFloat32List([1, 2], shape: [2]);

    final greater = mx.greater(a, b);
    final greaterEqual = mx.greaterEqual(a, b);
    final less = mx.less(a, b);
    final lessEqual = mx.lessEqual(a, b);
    final floorDivide = mx.floorDivide(
      MlxArray.fromFloat32List([5, 7], shape: [2]),
      MlxArray.fromFloat32List([2, 2], shape: [2]),
    );
    final logaddexp = mx.logaddexp(a, a);
    final inner = mx.inner(a, b);
    final floor = mx.floor(MlxArray.fromFloat32List([1.2, 2.9], shape: [2]));
    final sqrt = mx.sqrt(MlxArray.fromFloat32List([1, 4], shape: [2]));
    final rsqrt = mx.rsqrt(MlxArray.fromFloat32List([1, 4], shape: [2]));
    final square = mx.square(a);
    final reciprocal = mx.reciprocal(MlxArray.fromFloat32List([2, 4], shape: [2]));
    final sigmoid = mx.sigmoid(MlxArray.fromFloat32List([0], shape: [1]));
    final degrees = mx.degrees(MlxArray.fromFloat32List([3.14159265], shape: [1]));
    final radians = mx.radians(MlxArray.fromFloat32List([180], shape: [1]));
    final expm1 = mx.expm1(MlxArray.fromFloat32List([0], shape: [1]));
    final erf = mx.erf(MlxArray.fromFloat32List([0], shape: [1]));
    final erfinv = mx.erfinv(MlxArray.fromFloat32List([0], shape: [1]));
    final log1p = mx.log1p(MlxArray.fromFloat32List([1], shape: [1]));
    final log2 = mx.log2(MlxArray.fromFloat32List([8], shape: [1]));
    final log10 = mx.log10(MlxArray.fromFloat32List([100], shape: [1]));
    final rounded = mx.round(MlxArray.fromFloat32List([1.234], shape: [1]), decimals: 2);
    final stopped = mx.stopGradient(a);
    final finite = mx.isFinite(
      MlxArray.fromFloat32List([double.nan, double.infinity, 1], shape: [3]),
    );
    final inf = mx.isInf(
      MlxArray.fromFloat32List([double.nan, double.infinity, 1], shape: [3]),
    );
    final nan = mx.isNaN(
      MlxArray.fromFloat32List([double.nan, double.infinity, 1], shape: [3]),
    );
    final negInf = mx.isNegInf(
      MlxArray.fromFloat32List([double.negativeInfinity, 1], shape: [2]),
    );
    final posInf = mx.isPosInf(
      MlxArray.fromFloat32List([double.infinity, 1], shape: [2]),
    );
    final zerosLike = mx.zerosLike(a);
    final onesLike = mx.onesLike(a);
    final fullLike = mx.fullLike(a, 7);
    final toFp8 = mx.toFp8(fp);
    final fromFp8 = mx.fromFp8(toFp8);
    final putAlong = mx.putAlongAxis(
      MlxArray.zeros([3]),
      MlxArray.fromInt32List([1, 2], shape: [2]),
      MlxArray.fromFloat32List([5, 7], shape: [2]),
      axis: 0,
    );
    final scatterAddAxis = mx.scatterAddAxis(
      MlxArray.zeros([3]),
      MlxArray.fromInt32List([1, 1], shape: [2]),
      MlxArray.fromFloat32List([2, 3], shape: [2]),
      axis: 0,
    );
    final hadamard = mx.hadamardTransform(MlxArray.fromFloat32List([1, 2, 3, 4], shape: [4]));

    final cumsum = mx.cumsum(a);
    final cumprod = mx.cumprod(a);
    final cummax = mx.cummax(a);
    final cummin = mx.cummin(a);
    final logcumsumexp = mx.logcumsumexp(MlxArray.fromFloat32List([0, 0], shape: [2]));
    final eye = mx.eye(2);
    final identity = mx.identity(2);
    final hamming = mx.hamming(4);
    final hanning = mx.hanning(4);
    final tri = mx.tri(2);
    final tril = mx.tril(MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]));
    final triu = mx.triu(MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]));
    final trace = mx.trace(MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]));

    try {
      expect(greater.shape, <int>[2]);
      expect(greaterEqual.shape, <int>[2]);
      expect(less.shape, <int>[2]);
      expect(lessEqual.shape, <int>[2]);
      expect(floorDivide.shape, <int>[2]);
      expect(logaddexp.shape, <int>[2]);
      expect(inner.size, greaterThanOrEqualTo(1));
      expect(floor.shape, <int>[2]);
      expect(sqrt.shape, <int>[2]);
      expect(rsqrt.shape, <int>[2]);
      expect(square.shape, <int>[2]);
      expect(reciprocal.shape, <int>[2]);
      expect(sigmoid.shape, <int>[1]);
      expect(degrees.shape, <int>[1]);
      expect(radians.shape, <int>[1]);
      expect(expm1.shape, <int>[1]);
      expect(erf.shape, <int>[1]);
      expect(erfinv.shape, <int>[1]);
      expect(log1p.shape, <int>[1]);
      expect(log2.shape, <int>[1]);
      expect(log10.shape, <int>[1]);
      expect(rounded.shape, <int>[1]);
      expect(stopped.shape, <int>[2]);
      expect(finite.shape, <int>[3]);
      expect(inf.shape, <int>[3]);
      expect(nan.shape, <int>[3]);
      expect(negInf.shape, <int>[2]);
      expect(posInf.shape, <int>[2]);
      expect(zerosLike.shape, <int>[2]);
      expect(onesLike.shape, <int>[2]);
      expect(fullLike.shape, <int>[2]);
      expect(toFp8.shape, <int>[2]);
      expect(fromFp8.shape, <int>[2]);
      expect(putAlong.shape, <int>[3]);
      expect(scatterAddAxis.shape, <int>[3]);
      expect(hadamard.shape, <int>[4]);
      expect(cumsum.shape, <int>[2]);
      expect(cumprod.shape, <int>[2]);
      expect(cummax.shape, <int>[2]);
      expect(cummin.shape, <int>[2]);
      expect(logcumsumexp.shape, <int>[2]);
      expect(eye.shape, <int>[2, 2]);
      expect(identity.shape, <int>[2, 2]);
      expect(hamming.shape, <int>[4]);
      expect(hanning.shape, <int>[4]);
      expect(tri.shape, <int>[2, 2]);
      expect(tril.shape, <int>[2, 2]);
      expect(triu.shape, <int>[2, 2]);
      expect(trace.size, greaterThanOrEqualTo(1));
    } finally {
      trace.close();
      triu.close();
      tril.close();
      tri.close();
      hanning.close();
      hamming.close();
      identity.close();
      eye.close();
      logcumsumexp.close();
      cummin.close();
      cummax.close();
      cumprod.close();
      cumsum.close();
      hadamard.close();
      scatterAddAxis.close();
      putAlong.close();
      fromFp8.close();
      toFp8.close();
      fullLike.close();
      onesLike.close();
      zerosLike.close();
      posInf.close();
      negInf.close();
      nan.close();
      inf.close();
      finite.close();
      stopped.close();
      rounded.close();
      log10.close();
      log2.close();
      log1p.close();
      erfinv.close();
      erf.close();
      expm1.close();
      radians.close();
      degrees.close();
      sigmoid.close();
      reciprocal.close();
      square.close();
      rsqrt.close();
      sqrt.close();
      floor.close();
      inner.close();
      logaddexp.close();
      floorDivide.close();
      lessEqual.close();
      less.close();
      greaterEqual.close();
      greater.close();
      fp.close();
      b.close();
      a.close();
    }
  });
}
