/// Unit tests for [paddleOcrVlScatterImageEmbeddings] — the pure-tensor half
/// of `_buildMultimodalEmbedding` extracted in support of the hybrid
/// CoreML+MLX OCR refactor (issue #1). Covers the contiguous-run
/// `sliceUpdate` path and the sparse `equal -> cumsum -> where` path with
/// small synthetic tensors so we exercise the scatter without loading any
/// model weights.
library;

import 'dart:typed_data';

import 'package:dart_inference/mlx.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/coreml_scatter.dart';
import 'package:dart_inference/src/models/paddle_ocr_vl/paddle_ocr_vl.dart';
import 'package:test/test.dart';

const int _hiddenSize = 4;
const int _imageTokenId = 100295;

/// Builds a `[1, totalLen, _hiddenSize]` text-embedding fixture where row `i`
/// is `[i*0.01, i*0.01+1, i*0.01+2, i*0.01+3]` so each row is uniquely
/// distinguishable from image-feature rows (which we keep negative).
MlxArray _makeTextEmbed(int totalLen) {
  final values = Float32List(totalLen * _hiddenSize);
  for (var i = 0; i < totalLen; i++) {
    for (var j = 0; j < _hiddenSize; j++) {
      values[i * _hiddenSize + j] = i * 0.01 + j.toDouble();
    }
  }
  return MlxArray.fromFloat32List(values, shape: [1, totalLen, _hiddenSize]);
}

/// Builds a `[numFeatures, _hiddenSize]` image-hidden fixture where row `i`
/// is `[-i-1, -i-1-0.1, -i-1-0.2, -i-1-0.3]` — clearly disjoint from the
/// text-embed values above.
MlxArray _makeImageHidden(int numFeatures) {
  final values = Float32List(numFeatures * _hiddenSize);
  for (var i = 0; i < numFeatures; i++) {
    for (var j = 0; j < _hiddenSize; j++) {
      values[i * _hiddenSize + j] = -(i + 1).toDouble() - j * 0.1;
    }
  }
  return MlxArray.fromFloat32List(values, shape: [numFeatures, _hiddenSize]);
}

/// Hand-computes the expected scatter result row-by-row.
List<double> _expectedScatter({
  required int totalLen,
  required List<int> imagePositions,
}) {
  final out = List<double>.filled(totalLen * _hiddenSize, 0);
  // Map position -> feature index (positions are visited in order).
  final posToFeature = <int, int>{
    for (var i = 0; i < imagePositions.length; i++) imagePositions[i]: i,
  };
  for (var i = 0; i < totalLen; i++) {
    if (posToFeature.containsKey(i)) {
      final f = posToFeature[i]!;
      for (var j = 0; j < _hiddenSize; j++) {
        out[i * _hiddenSize + j] = -(f + 1).toDouble() - j * 0.1;
      }
    } else {
      for (var j = 0; j < _hiddenSize; j++) {
        out[i * _hiddenSize + j] = i * 0.01 + j.toDouble();
      }
    }
  }
  return out;
}

void _expectFlatClose(MlxArray got, List<double> expected) {
  expect(got.shape, equals([1, expected.length ~/ _hiddenSize, _hiddenSize]));
  final actual = got.toFloat32List();
  expect(actual.length, equals(expected.length));
  for (var i = 0; i < expected.length; i++) {
    expect(
      actual[i],
      closeTo(expected[i], 1e-5),
      reason: 'mismatch at flat index $i',
    );
  }
}

void main() {
  group('paddleOcrVlScatterImageEmbeddings', () {
    test('contiguous run: image tokens occupy a single span', () {
      // Prompt layout (totalLen=8): [t, t, IMG, IMG, IMG, t, t, t]
      const totalLen = 8;
      final tokenIds = <int>[
        10,
        11,
        _imageTokenId,
        _imageTokenId,
        _imageTokenId,
        20,
        21,
        22,
      ];
      final imagePositions = [2, 3, 4];
      final textEmbed = _makeTextEmbed(totalLen);
      final imageHidden = _makeImageHidden(imagePositions.length);
      addTearDown(textEmbed.close);
      addTearDown(imageHidden.close);

      final result = paddleOcrVlScatterImageEmbeddings(
        textEmbed: textEmbed,
        imageHidden: imageHidden,
        imagePositions: imagePositions,
        tokenIds: tokenIds,
        imageTokenId: _imageTokenId,
        hiddenSize: _hiddenSize,
      );
      addTearDown(result.close);

      _expectFlatClose(
        result,
        _expectedScatter(totalLen: totalLen, imagePositions: imagePositions),
      );
    });

    test('contiguous run: single image token at position 0', () {
      // Edge of contiguous case: count=1 always satisfies (last-first+1)==1.
      const totalLen = 4;
      final tokenIds = <int>[_imageTokenId, 7, 8, 9];
      final imagePositions = [0];
      final textEmbed = _makeTextEmbed(totalLen);
      final imageHidden = _makeImageHidden(1);
      addTearDown(textEmbed.close);
      addTearDown(imageHidden.close);

      final result = paddleOcrVlScatterImageEmbeddings(
        textEmbed: textEmbed,
        imageHidden: imageHidden,
        imagePositions: imagePositions,
        tokenIds: tokenIds,
        imageTokenId: _imageTokenId,
        hiddenSize: _hiddenSize,
      );
      addTearDown(result.close);

      _expectFlatClose(
        result,
        _expectedScatter(totalLen: totalLen, imagePositions: imagePositions),
      );
    });

    test('sparse mask: image tokens scattered across the prompt', () {
      // Prompt layout (totalLen=8): [IMG, t, t, IMG, t, t, IMG, t]
      // Non-contiguous positions exercise the equal/cumsum/where path.
      const totalLen = 8;
      final tokenIds = <int>[
        _imageTokenId,
        30,
        31,
        _imageTokenId,
        32,
        33,
        _imageTokenId,
        34,
      ];
      final imagePositions = [0, 3, 6];
      final textEmbed = _makeTextEmbed(totalLen);
      final imageHidden = _makeImageHidden(imagePositions.length);
      addTearDown(textEmbed.close);
      addTearDown(imageHidden.close);

      final result = paddleOcrVlScatterImageEmbeddings(
        textEmbed: textEmbed,
        imageHidden: imageHidden,
        imagePositions: imagePositions,
        tokenIds: tokenIds,
        imageTokenId: _imageTokenId,
        hiddenSize: _hiddenSize,
      );
      addTearDown(result.close);

      _expectFlatClose(
        result,
        _expectedScatter(totalLen: totalLen, imagePositions: imagePositions),
      );
    });

    test('sparse mask: two adjacent runs (still non-contiguous overall)', () {
      // [IMG, IMG, t, t, IMG, IMG, t, t] — first/last span 6 with 4 positions,
      // so contiguous check fails and we go through the sparse path.
      const totalLen = 8;
      final tokenIds = <int>[
        _imageTokenId,
        _imageTokenId,
        40,
        41,
        _imageTokenId,
        _imageTokenId,
        42,
        43,
      ];
      final imagePositions = [0, 1, 4, 5];
      final textEmbed = _makeTextEmbed(totalLen);
      final imageHidden = _makeImageHidden(imagePositions.length);
      addTearDown(textEmbed.close);
      addTearDown(imageHidden.close);

      final result = paddleOcrVlScatterImageEmbeddings(
        textEmbed: textEmbed,
        imageHidden: imageHidden,
        imagePositions: imagePositions,
        tokenIds: tokenIds,
        imageTokenId: _imageTokenId,
        hiddenSize: _hiddenSize,
      );
      addTearDown(result.close);

      _expectFlatClose(
        result,
        _expectedScatter(totalLen: totalLen, imagePositions: imagePositions),
      );
    });

    test('imageHidden with extra rows is sliced to imagePositions.length', () {
      // Sparse path slices imageHidden[:numPositions]; supply an extra row
      // and confirm it is ignored (output uses the first 3 rows only).
      const totalLen = 6;
      final tokenIds = <int>[
        _imageTokenId,
        50,
        _imageTokenId,
        51,
        _imageTokenId,
        52,
      ];
      final imagePositions = [0, 2, 4];
      final textEmbed = _makeTextEmbed(totalLen);
      final imageHidden = _makeImageHidden(imagePositions.length + 2);
      addTearDown(textEmbed.close);
      addTearDown(imageHidden.close);

      final result = paddleOcrVlScatterImageEmbeddings(
        textEmbed: textEmbed,
        imageHidden: imageHidden,
        imagePositions: imagePositions,
        tokenIds: tokenIds,
        imageTokenId: _imageTokenId,
        hiddenSize: _hiddenSize,
      );
      addTearDown(result.close);

      _expectFlatClose(
        result,
        _expectedScatter(totalLen: totalLen, imagePositions: imagePositions),
      );
    });

    test('empty imagePositions throws ArgumentError (caller-handled fast path)', () {
      // The helper documents that the zero-image case must be handled by the
      // caller; passing an empty list is a programmer error.
      final textEmbed = _makeTextEmbed(4);
      final imageHidden = _makeImageHidden(1);
      addTearDown(textEmbed.close);
      addTearDown(imageHidden.close);

      expect(
        () => paddleOcrVlScatterImageEmbeddings(
          textEmbed: textEmbed,
          imageHidden: imageHidden,
          imagePositions: const <int>[],
          tokenIds: const <int>[1, 2, 3, 4],
          imageTokenId: _imageTokenId,
          hiddenSize: _hiddenSize,
        ),
        throwsArgumentError,
      );
    });

    test('result is independent of textEmbed (caller still owns textEmbed)', () {
      // After scattering, mutating the returned tensor's lifetime must not
      // affect textEmbed: the helper documents that textEmbed is left intact.
      const totalLen = 5;
      final tokenIds = <int>[60, _imageTokenId, _imageTokenId, 61, 62];
      final imagePositions = [1, 2];
      final textEmbed = _makeTextEmbed(totalLen);
      final imageHidden = _makeImageHidden(imagePositions.length);
      addTearDown(textEmbed.close);
      addTearDown(imageHidden.close);

      final result = paddleOcrVlScatterImageEmbeddings(
        textEmbed: textEmbed,
        imageHidden: imageHidden,
        imagePositions: imagePositions,
        tokenIds: tokenIds,
        imageTokenId: _imageTokenId,
        hiddenSize: _hiddenSize,
      );
      // Close the result first; textEmbed must still be readable.
      result.close();
      final originalRow0 = textEmbed
          .toFloat32List()
          .sublist(0, _hiddenSize);
      expect(originalRow0[0], closeTo(0.0, 1e-6));
      expect(originalRow0[1], closeTo(1.0, 1e-6));
      expect(originalRow0[2], closeTo(2.0, 1e-6));
      expect(originalRow0[3], closeTo(3.0, 1e-6));
    });
  });

  group('paddleOcrVlScatterImageEmbeddingsFloat32', () {
    // Float32-only sibling used by the CoreML hybrid runner (commit #6).
    // Mirrors the row-by-row contract of the MLX helper without dragging in
    // any MLX tensors.

    Float32List makeTextEmbedF32(int totalLen) {
      final values = Float32List(totalLen * _hiddenSize);
      for (var i = 0; i < totalLen; i++) {
        for (var j = 0; j < _hiddenSize; j++) {
          values[i * _hiddenSize + j] = i * 0.01 + j.toDouble();
        }
      }
      return values;
    }

    Float32List makeImageHiddenF32(int numFeatures) {
      final values = Float32List(numFeatures * _hiddenSize);
      for (var i = 0; i < numFeatures; i++) {
        for (var j = 0; j < _hiddenSize; j++) {
          values[i * _hiddenSize + j] = -(i + 1).toDouble() - j * 0.1;
        }
      }
      return values;
    }

    test('contiguous run matches expected scatter', () {
      const totalLen = 8;
      final imagePositions = [2, 3, 4];
      final result = paddleOcrVlScatterImageEmbeddingsFloat32(
        textEmbed: makeTextEmbedF32(totalLen),
        imageHidden: makeImageHiddenF32(imagePositions.length),
        imagePositions: imagePositions,
        promptLen: totalLen,
        hiddenSize: _hiddenSize,
      );
      final expected = _expectedScatter(
        totalLen: totalLen,
        imagePositions: imagePositions,
      );
      expect(result.length, equals(expected.length));
      for (var i = 0; i < expected.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-5),
            reason: 'mismatch at flat index $i');
      }
    });

    test('sparse positions match expected scatter', () {
      const totalLen = 8;
      final imagePositions = [0, 3, 6];
      final result = paddleOcrVlScatterImageEmbeddingsFloat32(
        textEmbed: makeTextEmbedF32(totalLen),
        imageHidden: makeImageHiddenF32(imagePositions.length),
        imagePositions: imagePositions,
        promptLen: totalLen,
        hiddenSize: _hiddenSize,
      );
      final expected = _expectedScatter(
        totalLen: totalLen,
        imagePositions: imagePositions,
      );
      for (var i = 0; i < expected.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-5));
      }
    });

    test('extra image rows beyond imagePositions.length are ignored', () {
      const totalLen = 6;
      final imagePositions = [0, 2, 4];
      final result = paddleOcrVlScatterImageEmbeddingsFloat32(
        textEmbed: makeTextEmbedF32(totalLen),
        imageHidden: makeImageHiddenF32(imagePositions.length + 2),
        imagePositions: imagePositions,
        promptLen: totalLen,
        hiddenSize: _hiddenSize,
      );
      final expected = _expectedScatter(
        totalLen: totalLen,
        imagePositions: imagePositions,
      );
      for (var i = 0; i < expected.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-5));
      }
    });

    test('empty imagePositions throws ArgumentError', () {
      expect(
        () => paddleOcrVlScatterImageEmbeddingsFloat32(
          textEmbed: makeTextEmbedF32(4),
          imageHidden: makeImageHiddenF32(1),
          imagePositions: const <int>[],
          promptLen: 4,
          hiddenSize: _hiddenSize,
        ),
        throwsArgumentError,
      );
    });

    test('out-of-range position throws ArgumentError', () {
      expect(
        () => paddleOcrVlScatterImageEmbeddingsFloat32(
          textEmbed: makeTextEmbedF32(4),
          imageHidden: makeImageHiddenF32(1),
          imagePositions: const <int>[7],
          promptLen: 4,
          hiddenSize: _hiddenSize,
        ),
        throwsArgumentError,
      );
    });

    test('undersized imageHidden throws ArgumentError', () {
      expect(
        () => paddleOcrVlScatterImageEmbeddingsFloat32(
          textEmbed: makeTextEmbedF32(4),
          imageHidden: Float32List(_hiddenSize - 1),
          imagePositions: const <int>[1],
          promptLen: 4,
          hiddenSize: _hiddenSize,
        ),
        throwsArgumentError,
      );
    });
  });
}
