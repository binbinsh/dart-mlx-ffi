import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';

import '../benchmark/runtime/input_json.dart';
import 'package:dart_mlx_ffi/runtime.dart';

void main() {
  group('runtime input JSON', () {
    late Directory dir;

    setUp(() {
      dir = Directory.systemTemp.createTempSync('dmf_runtime_input_');
    });

    tearDown(() {
      dir.deleteSync(recursive: true);
    });

    test('reads legacy flat input map', () {
      final path = _writeJson(dir, {
        'input': {
          'dtype': 'float32',
          'shape': [1, 2],
          'values': [1.0, 2.0],
        },
      });

      final inputs = readRuntimeInputsJson(path);
      final tensor = inputs['input'] as RuntimeTensor;
      expect(tensor.dtype, RuntimeTensorDataType.float32);
      expect(tensor.shape, [1, 2]);
      expect(tensor.asFloat32List().toList(), [1.0, 2.0]);
    });

    test('reads nested values from inputs object', () {
      final path = _writeJson(dir, {
        'metadata': {'fixture': 'nested'},
        'inputs': {
          'input_ids': {
            'dtype': 'int64',
            'shape': [1, 2, 2],
            'values': [
              [
                [1, 2],
                [3, 4],
              ],
            ],
          },
        },
      });

      final tensor = readRuntimeInputsJson(path)['input_ids'] as RuntimeTensor;
      expect(tensor.dtype, RuntimeTensorDataType.int64);
      expect(tensor.shape, [1, 2, 2]);
      expect(tensor.asInt64List().toList(), [1, 2, 3, 4]);
    });

    test('reads raw base64 tensor bytes', () {
      final bytes = _bytes(Int32List.fromList([7, 8, 9]));
      final path = _writeJson(dir, {
        'inputs': {
          'positions': {
            'dtype': 'int32',
            'shape': [3],
            'base64': base64Encode(bytes),
          },
        },
      });

      final tensor = readRuntimeInputsJson(path)['positions'] as RuntimeTensor;
      expect(tensor.asInt32List().toList(), [7, 8, 9]);
    });

    test('reads raw tensor bytes from file relative to json', () {
      File(
        '${dir.path}/image.bin',
      ).writeAsBytesSync(Uint8List.fromList([1, 2, 3]));
      final path = _writeJson(dir, {
        'inputs': {
          'image': {
            'dtype': 'uint8',
            'shape': [1, 3],
            'file': 'image.bin',
          },
        },
      });

      final tensor = readRuntimeInputsJson(path)['image'] as RuntimeTensor;
      expect(tensor.shape, [1, 3]);
      expect(tensor.asUint8List(), [1, 2, 3]);
    });

    test('encodes float16 values', () {
      final path = _writeJson(dir, {
        'input': {
          'dtype': 'float16',
          'shape': [2],
          'values': [1.0, -2.0],
        },
      });

      final tensor = readRuntimeInputsJson(path)['input'] as RuntimeTensor;
      final halves = tensor.bytes.buffer.asUint16List(
        tensor.bytes.offsetInBytes,
        tensor.bytes.lengthInBytes ~/ 2,
      );
      expect(tensor.dtype, RuntimeTensorDataType.float16);
      expect(halves.toList(), [0x3c00, 0xc000]);
    });

    test('encodes bool values', () {
      final path = _writeJson(dir, {
        'use_cache_branch': {
          'dtype': 'bool',
          'shape': [3],
          'values': [false, true, 1],
        },
      });

      final tensor =
          readRuntimeInputsJson(path)['use_cache_branch'] as RuntimeTensor;
      expect(tensor.dtype, RuntimeTensorDataType.boolean);
      expect(tensor.asUint8List(), [0, 1, 1]);
    });

    test('rejects shape byte-length mismatch', () {
      final path = _writeJson(dir, {
        'input': {
          'dtype': 'float32',
          'shape': [3],
          'values': [1.0, 2.0],
        },
      });

      expect(() => readRuntimeInputsJson(path), throwsFormatException);
    });
  });
}

String _writeJson(Directory dir, Map<String, Object?> json) {
  final file = File('${dir.path}/input.json');
  file.writeAsStringSync(jsonEncode(json));
  return file.path;
}

Uint8List _bytes(TypedData data) {
  return Uint8List.fromList(
    data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
  );
}
