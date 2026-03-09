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
  test('runs high-level arithmetic operations', () {
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
    final sum = MlxOps.add(a, b);
    final sumOp = a + b;
    final diff = MlxOps.subtract(b, a);
    final diffOp = b - a;
    final productElementwise = MlxOps.multiply(a, b);
    final productElementwiseOp = a * b;
    final quotient = MlxOps.divide(b, a);
    final quotientOp = b / a;
    final product = MlxOps.matmul(a, b);
    final productMethod = a.matmul(b);

    try {
      expect(sum.toList(), <Object>[6.0, 8.0, 10.0, 12.0]);
      expect(sumOp.toList(), <Object>[6.0, 8.0, 10.0, 12.0]);
      expect(diff.toList(), <Object>[4.0, 4.0, 4.0, 4.0]);
      expect(diffOp.toList(), <Object>[4.0, 4.0, 4.0, 4.0]);
      expect(productElementwise.toList(), <Object>[5.0, 12.0, 21.0, 32.0]);
      expect(productElementwiseOp.toList(), <Object>[5.0, 12.0, 21.0, 32.0]);
      final quotientValues = quotient.toList();
      final quotientValuesOp = quotientOp.toList();
      expect(quotientValues[0], 5.0);
      expect(quotientValues[1], 3.0);
      expect(quotientValues[2] as double, closeTo(7.0 / 3.0, 1e-5));
      expect(quotientValues[3], 2.0);
      expect(quotientValuesOp[0], 5.0);
      expect(quotientValuesOp[1], 3.0);
      expect(product.toList(), <Object>[19.0, 22.0, 43.0, 50.0]);
      expect(productMethod.toList(), <Object>[19.0, 22.0, 43.0, 50.0]);
    } finally {
      productMethod.close();
      product.close();
      quotientOp.close();
      quotient.close();
      productElementwiseOp.close();
      productElementwise.close();
      diffOp.close();
      diff.close();
      sumOp.close();
      sum.close();
      b.close();
      a.close();
    }
  });

  test('runs unary, comparison, selection, and reduction operations', () {
    final a = MlxArray.fromFloat32List([1, -2, 3, -4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([1, 2, 0, -4], shape: [2, 2]);
    final expInput = MlxArray.fromFloat32List([0, 1], shape: [2]);
    final logInput = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final trigInput = MlxArray.fromFloat32List([0], shape: [1]);
    final reduceInput = MlxArray.fromFloat32List([0, 0], shape: [2]);
    final topkInput = MlxArray.fromFloat32List([3, 1, 2], shape: [3]);
    final condition = MlxArray.fromBoolList([true, false, true, false], shape: [2, 2]);
    final abs = MlxOps.abs(a);
    final absMethod = a.abs();
    final neg = MlxOps.negative(a);
    final negNamedMethod = a.negative();
    final negMethod = -a;
    final exp = MlxOps.exp(expInput);
    final expMethod = expInput.exp();
    final log = MlxOps.log(logInput);
    final logMethod = logInput.log();
    final sin = MlxOps.sin(trigInput);
    final sinMethod = trigInput.sin();
    final cos = MlxOps.cos(trigInput);
    final cosMethod = trigInput.cos();
    final equal = MlxOps.equal(a, b);
    final equalMethod = a.equal(b);
    final where = MlxOps.where(condition, a, b);
    final sumAll = MlxOps.sum(abs);
    final sumMethod = abs.sum();
    final sumAxis = MlxOps.sum(abs, axis: 1, keepDims: true);
    final meanAll = MlxOps.mean(abs);
    final meanMethod = abs.mean();
    final meanAxis = MlxOps.mean(abs, axis: 0);
    final logSumExp = MlxOps.logSumExp(reduceInput);
    final logSumExpMethod = reduceInput.logSumExp();
    final softmax = MlxOps.softmax(reduceInput);
    final softmaxMethod = reduceInput.softmax();
    final topK = MlxOps.topK(topkInput, 2);
    final topKMethod = topkInput.topK(2);

    try {
      expect(abs.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(absMethod.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(neg.toList(), <Object>[-1.0, 2.0, -3.0, 4.0]);
      expect(negNamedMethod.toList(), <Object>[-1.0, 2.0, -3.0, 4.0]);
      expect(negMethod.toList(), <Object>[-1.0, 2.0, -3.0, 4.0]);
      expect((exp.toList()[0] as double), closeTo(1.0, 1e-5));
      expect((exp.toList()[1] as double), closeTo(2.7182817, 1e-4));
      expect((expMethod.toList()[1] as double), closeTo(2.7182817, 1e-4));
      expect((log.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((log.toList()[1] as double), closeTo(0.693147, 1e-4));
      expect((logMethod.toList()[1] as double), closeTo(0.693147, 1e-4));
      expect((sin.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((sinMethod.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((cos.toList()[0] as double), closeTo(1.0, 1e-5));
      expect((cosMethod.toList()[0] as double), closeTo(1.0, 1e-5));
      expect(equal.toList(), <Object>[true, false, false, true]);
      expect(equalMethod.toList(), <Object>[true, false, false, true]);
      expect(where.toList(), <Object>[1.0, 2.0, 3.0, -4.0]);
      expect(sumAll.toList(), <Object>[10.0]);
      expect(sumMethod.toList(), <Object>[10.0]);
      expect(sumAxis.shape, <int>[2, 1]);
      expect(sumAxis.toList(), <Object>[3.0, 7.0]);
      expect(meanAll.toList(), <Object>[2.5]);
      expect(meanMethod.toList(), <Object>[2.5]);
      expect(meanAxis.toList(), <Object>[2.0, 3.0]);
      expect((logSumExp.toList()[0] as double), closeTo(0.693147, 1e-4));
      expect((logSumExpMethod.toList()[0] as double), closeTo(0.693147, 1e-4));
      expect(softmax.toList(), <Object>[0.5, 0.5]);
      expect(softmaxMethod.toList(), <Object>[0.5, 0.5]);
      expect(topK.toList(), <Object>[2.0, 3.0]);
      expect(topKMethod.toList(), <Object>[2.0, 3.0]);
      final castBack = expInput.astype(MlxDType.MLX_FLOAT16).astype(MlxDType.MLX_FLOAT32);
      expect((castBack.toList()[0] as double), closeTo(0.0, 1e-5));
      expect((castBack.toList()[1] as double), closeTo(1.0, 1e-3));
      castBack.close();
    } finally {
      topKMethod.close();
      topK.close();
      softmaxMethod.close();
      softmax.close();
      logSumExpMethod.close();
      logSumExp.close();
      meanAxis.close();
      meanMethod.close();
      meanAll.close();
      sumAxis.close();
      sumMethod.close();
      sumAll.close();
      where.close();
      equalMethod.close();
      equal.close();
      cosMethod.close();
      cos.close();
      sinMethod.close();
      sin.close();
      logMethod.close();
      log.close();
      expMethod.close();
      exp.close();
      condition.close();
      topkInput.close();
      reduceInput.close();
      trigInput.close();
      logInput.close();
      expInput.close();
      negNamedMethod.close();
      negMethod.close();
      neg.close();
      absMethod.close();
      abs.close();
      b.close();
      a.close();
    }
  });

  test('supports tensor composition and indexing helpers', () {
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
    final concatenated = MlxOps.concatenate([a, b], axis: 0);
    final stacked = MlxOps.stack([a, b], axis: 0);
    final broadcasted = MlxArray.fromFloat32List([1, 2], shape: [2]).broadcastTo([2, 2]);
    final expanded = MlxArray.fromFloat32List([1, 2], shape: [2]).expandDims(0);
    final squeezed = expanded.squeeze();
    final clipped = MlxArray.fromFloat32List([-1, 0.5, 3], shape: [3]).clip(min: 0, max: 1);
    final minned = a.minimum(b);
    final maxed = a.maximum(b);
    final argmax = a.argmax();
    final argminAxis = a.argmin(axis: 1, keepDims: true);
    final sorted = MlxArray.fromFloat32List([3, 1, 2], shape: [3]).sort();
    final argsorted = MlxArray.fromFloat32List([3, 1, 2], shape: [3]).argsort();

    try {
      expect(concatenated.shape, <int>[4, 2]);
      expect(concatenated.toList(), <Object>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
      expect(stacked.shape, <int>[2, 2, 2]);
      expect(stacked.toList(), <Object>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
      expect(broadcasted.toList(), <Object>[1.0, 2.0, 1.0, 2.0]);
      expect(expanded.shape, <int>[1, 2]);
      expect(squeezed.shape, <int>[2]);
      expect(clipped.toList(), <Object>[0.0, 0.5, 1.0]);
      expect(minned.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      expect(maxed.toList(), <Object>[5.0, 6.0, 7.0, 8.0]);
      expect(argmax.toList(), <Object>[3]);
      expect(argminAxis.shape, <int>[2, 1]);
      expect(argminAxis.toList(), <Object>[0, 0]);
      expect(sorted.toList(), <Object>[1.0, 2.0, 3.0]);
      expect(argsorted.toList(), <Object>[1, 2, 0]);
    } finally {
      argsorted.close();
      sorted.close();
      argminAxis.close();
      argmax.close();
      maxed.close();
      minned.close();
      clipped.close();
      squeezed.close();
      expanded.close();
      broadcasted.close();
      stacked.close();
      concatenated.close();
      b.close();
      a.close();
    }
  });

  test('supports quantize, dequantize, and quantized matmul', () {
    final weightValues = List<double>.generate(
      64,
      (index) => ((index % 8) - 4) / 4,
    );
    final inputValues = List<double>.generate(32, (index) => (index % 7) / 7);
    final weights = MlxArray.fromFloat32List(weightValues, shape: [2, 32]);
    final input = MlxArray.fromFloat32List(inputValues, shape: [1, 32]);

    try {
      final quantized = mx.quant.quantize(
        weights,
        groupSize: 32,
        bits: 8,
        mode: 'affine',
      );
      try {
        expect(quantized.weights.dtype, MlxDType.MLX_UINT32);
        expect(quantized.scales.shape, <int>[2, 1]);
        expect(quantized.biases, isNotNull);

        final restored = mx.quant.dequantize(
          quantized,
          groupSize: 32,
          bits: 8,
          mode: 'affine',
          dtype: MlxDType.MLX_FLOAT32,
        );
        final reference = input.matmul(weights.transpose());
        final qmm = mx.quant.matmul(
          input,
          quantized,
          transpose: true,
          groupSize: 32,
          bits: 8,
          mode: 'affine',
        );
        try {
          expect(restored.shape, weights.shape);
          final restoredValues = List<double>.from(restored.toList().cast<double>());
          for (var index = 0; index < restoredValues.length; index++) {
            expect(restoredValues[index], closeTo(weightValues[index], 1e-2));
          }
          final refValues = List<double>.from(reference.toList().cast<double>());
          final qmmValues = List<double>.from(qmm.toList().cast<double>());
          expect(qmmValues, hasLength(refValues.length));
          for (var index = 0; index < qmmValues.length; index++) {
            expect(qmmValues[index], closeTo(refValues[index], 1e-1));
          }
          expect(
            () => mx.quant.dequantize(
              MlxQuantizedMatrix(quantized.weights, quantized.scales),
            ),
            throwsArgumentError,
          );
        } finally {
          qmm.close();
          reference.close();
          restored.close();
        }
      } finally {
        quantized.close();
      }
    } finally {
      input.close();
      weights.close();
    }
  });

  test('supports take, gather, slice, einsum, and tensordot', () {
    final vector = MlxArray.fromFloat32List([10, 20, 30, 40], shape: [4]);
    final takeIndices = MlxArray.fromInt32List([2, 0], shape: [2]);
    final matrix = MlxArray.fromFloat32List([1, 2, 3, 4, 5, 6], shape: [2, 3]);
    final takeAxisIndices = MlxArray.fromInt32List([2, 0], shape: [2]);
    final alongAxisIndices = MlxArray.fromInt32List([2, 0, 1, 1], shape: [2, 2]);
    final rowIndices = MlxArray.fromInt32List([1, 0], shape: [2]);
    final colIndices = MlxArray.fromInt32List([2, 1], shape: [2]);
    final left = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final right = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);

    final taken = vector.take(takeIndices);
    final takenAxis = mx.take(matrix, takeAxisIndices, axis: 1);
    final takenAlongAxis = matrix.takeAlongAxis(alongAxisIndices, axis: 1);
    final sliced = vector.slice(start: [1], stop: [4], strides: [2]);
    final gatheredSingle = mx.gatherSingle(
      matrix,
      rowIndices,
      axis: 0,
      sliceSizes: [1, 3],
    );
    final gathered = mx.gather(
      matrix,
      [rowIndices, colIndices],
      axes: [0, 1],
      sliceSizes: [1, 1],
    );
    final einsum = mx.einsum('ij,jk->ik', [left, right]);
    final tensordotAxis = mx.tensordot(left, right, axis: 1);
    final tensordotAxes = left.tensordot(right, axesA: [1], axesB: [0]);

    try {
      expect(taken.toList(), <Object>[30.0, 10.0]);
      expect(takenAxis.toList(), <Object>[3.0, 1.0, 6.0, 4.0]);
      expect(takenAlongAxis.toList(), <Object>[3.0, 1.0, 5.0, 5.0]);
      expect(sliced.toList(), <Object>[20.0, 40.0]);
      expect(gatheredSingle.toList(), <Object>[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
      expect(gathered.toList(), <Object>[6.0, 2.0]);
      expect(einsum.toList(), <Object>[19.0, 22.0, 43.0, 50.0]);
      expect(tensordotAxis.toList(), <Object>[19.0, 22.0, 43.0, 50.0]);
      expect(tensordotAxes.toList(), <Object>[19.0, 22.0, 43.0, 50.0]);
    } finally {
      tensordotAxes.close();
      tensordotAxis.close();
      einsum.close();
      gathered.close();
      gatheredSingle.close();
      sliced.close();
      takenAlongAxis.close();
      takenAxis.close();
      taken.close();
      right.close();
      left.close();
      colIndices.close();
      rowIndices.close();
      alongAxisIndices.close();
      takeAxisIndices.close();
      matrix.close();
      takeIndices.close();
      vector.close();
    }
  });

  test('supports qqmm and gatherQmm', () {
    final qWeightValues = List<double>.generate(
      32,
      (index) => ((index % 8) - 4) / 8,
    );
    final qInputValues = List<double>.generate(16, (index) => (index % 7) / 14);
    final qqmmWeights = MlxArray.fromFloat32List(qWeightValues, shape: [2, 16]);
    final qqmmInput = MlxArray.fromFloat32List(qInputValues, shape: [1, 16]);
    final weightValues = List<double>.generate(
      64,
      (index) => ((index % 8) - 4) / 4,
    );
    final inputValues = List<double>.generate(32, (index) => (index % 7) / 7);
    final weights = MlxArray.fromFloat32List(weightValues, shape: [2, 32]);
    final input = MlxArray.fromFloat32List(inputValues, shape: [1, 32]);

    try {
      final quantized = mx.quant.quantize(
        weights,
        groupSize: 32,
        bits: 8,
        mode: 'affine',
      );
      try {
        final qqmmRef = qqmmInput.matmul(qqmmWeights.transpose());
        final qqmm = mx.quant.qqmm(
          qqmmInput,
          qqmmWeights,
          mode: 'nvfp4',
        );
        final qmm = mx.quant.matmul(
          input,
          quantized,
          groupSize: 32,
          bits: 8,
          mode: 'affine',
        );
        final gathered = mx.quant.gatherQmm(
          input,
          quantized,
          groupSize: 32,
          bits: 8,
          mode: 'affine',
        );
        try {
          expect(qqmm.shape, qqmmRef.shape);
          expect(qqmm.dtype, qqmmRef.dtype);
          final qmmValues = List<double>.from(qmm.toList().cast<double>());
          final gatheredValues = List<double>.from(gathered.toList().cast<double>());
          for (var index = 0; index < qmmValues.length; index++) {
            expect(gatheredValues[index], closeTo(qmmValues[index], 1e-5));
          }
        } finally {
          qqmmRef.close();
          gathered.close();
          qmm.close();
          qqmm.close();
        }
      } finally {
        quantized.close();
      }
    } finally {
      input.close();
      weights.close();
      qqmmInput.close();
      qqmmWeights.close();
    }
  });

  test('supports random key splitting', () {
    final key = MlxRandom.key(42);
    final split = MlxRandom.split(key);

    try {
      expect(key.shape, isNotEmpty);
      expect(split.first.shape, key.shape);
      expect(split.second.shape, key.shape);
      expect(split.first.toString(), contains('array'));
      expect(split.second.toString(), contains('array'));
      expect(key.toList(), isNotEmpty);
    } finally {
      split.second.close();
      split.first.close();
      key.close();
    }
  });

  test('samples random arrays', () {
    final uniform = MlxRandom.uniform([2, 3]);
    final normal = MlxRandom.normal([2, 3]);
    final probability = MlxArray.zeros([4]);
    final logits = MlxArray.fromFloat32List([0, 50], shape: [2]);
    final permutationInput = MlxArray.fromInt32List([1, 2, 3, 4], shape: [4]);
    final bernoulli = MlxRandom.bernoulli(probability);
    final categorical = MlxRandom.categorical(logits);
    final categoricalMany = MlxRandom.categorical(logits, numSamples: 3);
    final categoricalShaped = MlxRandom.categorical(logits, shape: [2]);
    final permutation = MlxRandom.permutation(permutationInput);
    final permutationArange = MlxRandom.permutationArange(4);

    try {
      expect(uniform.shape, <int>[2, 3]);
      expect(normal.shape, <int>[2, 3]);
      expect(uniform.dtype, MlxDType.MLX_FLOAT32);
      expect(normal.dtype, MlxDType.MLX_FLOAT32);
      expect(uniform.size, 6);
      expect(normal.size, 6);
      expect(bernoulli.toList(), <Object>[false, false, false, false]);
      expect(categorical.toList(), everyElement(1));
      expect(categoricalMany.toList(), everyElement(1));
      expect(categoricalShaped.shape, <int>[2]);
      expect(categoricalShaped.toList(), everyElement(1));
      expect(
        () => MlxRandom.categorical(logits, shape: [2], numSamples: 2),
        throwsArgumentError,
      );

      final permutationValues = List<int>.from(permutation.toList());
      final permutationArangeValues = List<int>.from(permutationArange.toList());
      permutationValues.sort();
      permutationArangeValues.sort();
      expect(permutationValues, <int>[1, 2, 3, 4]);
      expect(permutationArangeValues, <int>[0, 1, 2, 3]);
    } finally {
      permutationArange.close();
      permutation.close();
      categoricalShaped.close();
      categoricalMany.close();
      categorical.close();
      bernoulli.close();
      permutationInput.close();
      logits.close();
      probability.close();
      normal.close();
      uniform.close();
    }
  });

}
