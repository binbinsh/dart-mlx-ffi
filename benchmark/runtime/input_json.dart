import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/runtime.dart';

Map<String, Object?> readRuntimeInputsJson(String path) {
  final file = File(path);
  final json = jsonDecode(file.readAsStringSync());
  if (json is! Map) {
    throw FormatException('Input JSON must be an object.');
  }
  final map = Map<String, Object?>.from(json);
  final rawInputs = map['inputs'];
  final inputs = rawInputs is Map ? Map<String, Object?>.from(rawInputs) : map;
  final baseDir = file.parent.path;
  final result = <String, Object?>{};
  for (final entry in inputs.entries) {
    final spec = entry.value;
    if (spec is! Map) {
      throw FormatException('Input ${entry.key} must be an object.');
    }
    final tensor = runtimeTensorFromJson(
      Map<String, Object?>.from(spec),
      baseDir: baseDir,
      inputName: entry.key,
    );
    result[entry.key] = tensor;
  }
  return result;
}

RuntimeTensor runtimeTensorFromJson(
  Map<String, Object?> json, {
  String? baseDir,
  String inputName = 'input',
}) {
  final dtype = json['dtype'];
  if (dtype is! String) {
    throw FormatException('Input $inputName must define string dtype.');
  }
  final shape = _shape(json['shape']);
  final bytes = _tensorBytes(
    json,
    dtype,
    baseDir: baseDir,
    inputName: inputName,
  );
  final inferredShape = shape ?? [_elementCount(dtype, bytes.lengthInBytes)];
  _validateByteLength(inputName, dtype, inferredShape, bytes.lengthInBytes);
  return RuntimeTensor(
    dtype: _runtimeDtype(dtype),
    shape: inferredShape,
    bytes: bytes,
  );
}

Uint8List _tensorBytes(
  Map<String, Object?> json,
  String dtype, {
  String? baseDir,
  required String inputName,
}) {
  if (json.containsKey('values')) {
    final values = json['values'];
    if (values is! List) {
      throw FormatException('Input $inputName values must be a list.');
    }
    return _valuesToBytes(dtype, _flatten(values));
  }
  if (json.containsKey('base64')) {
    final encoded = json['base64'];
    if (encoded is! String) {
      throw FormatException('Input $inputName base64 must be a string.');
    }
    return base64Decode(encoded);
  }
  final file = json['file'] ?? json['path'];
  if (file is String) {
    final path = _resolvePath(file, baseDir);
    if (json['encoding'] == 'base64') {
      return base64Decode(File(path).readAsStringSync().trim());
    }
    return File(path).readAsBytesSync();
  }
  throw FormatException(
    'Input $inputName must define values, base64, file, or path.',
  );
}

Uint8List _valuesToBytes(String dtype, List<Object?> values) {
  return switch (dtype) {
    'float32' => _copyBytes(
      Float32List.fromList(
        values.map((value) => _number(value).toDouble()).toList(),
      ),
    ),
    'float64' => _copyBytes(
      Float64List.fromList(
        values.map((value) => _number(value).toDouble()).toList(),
      ),
    ),
    'float16' => _copyBytes(
      Uint16List.fromList(
        values.map((value) => _floatToHalf(_number(value).toDouble())).toList(),
      ),
    ),
    'int32' => _copyBytes(
      Int32List.fromList(
        values.map((value) => _number(value).toInt()).toList(),
      ),
    ),
    'int64' => _copyBytes(
      Int64List.fromList(
        values.map((value) => _number(value).toInt()).toList(),
      ),
    ),
    'uint8' => Uint8List.fromList(
      values.map((value) => _number(value).toInt()).toList(),
    ),
    'bool' => Uint8List.fromList(values.map(_boolByte).toList()),
    _ => throw FormatException('Unsupported input dtype: $dtype'),
  };
}

List<Object?> _flatten(List<Object?> values) {
  final result = <Object?>[];
  void visit(Object? value) {
    if (value is List) {
      for (final item in value) {
        visit(item);
      }
    } else {
      result.add(value);
    }
  }

  visit(values);
  return result;
}

num _number(Object? value) {
  if (value is num) return value;
  throw FormatException('Tensor values must be numbers.');
}

List<int>? _shape(Object? raw) {
  if (raw == null) return null;
  if (raw is! List) {
    throw FormatException('Tensor shape must be a list.');
  }
  return raw.map((value) {
    if (value is! num) {
      throw FormatException('Tensor shape entries must be numbers.');
    }
    return value.toInt();
  }).toList();
}

RuntimeTensorDataType _runtimeDtype(String dtype) => switch (dtype) {
  'float32' => RuntimeTensorDataType.float32,
  'float64' => RuntimeTensorDataType.float64,
  'float16' => RuntimeTensorDataType.float16,
  'int32' => RuntimeTensorDataType.int32,
  'int64' => RuntimeTensorDataType.int64,
  'uint8' => RuntimeTensorDataType.uint8,
  'bool' => RuntimeTensorDataType.boolean,
  _ => throw FormatException('Unsupported input dtype: $dtype'),
};

void _validateByteLength(
  String inputName,
  String dtype,
  List<int> shape,
  int byteLength,
) {
  final expected =
      shape.fold<int>(1, (value, dim) => value * dim) * _byteWidth(dtype);
  if (expected != byteLength) {
    throw FormatException(
      'Input $inputName byte length mismatch: shape=$shape dtype=$dtype '
      'expects $expected bytes but got $byteLength.',
    );
  }
}

int _elementCount(String dtype, int byteLength) {
  final width = _byteWidth(dtype);
  if (byteLength % width != 0) {
    throw FormatException(
      'Tensor byte length $byteLength is not divisible by dtype width $width.',
    );
  }
  return byteLength ~/ width;
}

int _byteWidth(String dtype) => switch (dtype) {
  'float32' || 'int32' => 4,
  'float64' || 'int64' => 8,
  'float16' => 2,
  'uint8' || 'bool' => 1,
  _ => throw FormatException('Unsupported input dtype: $dtype'),
};

int _boolByte(Object? value) {
  if (value is bool) return value ? 1 : 0;
  return _number(value).toInt() == 0 ? 0 : 1;
}

String _resolvePath(String value, String? baseDir) {
  final path = File(value);
  if (path.isAbsolute || baseDir == null) return value;
  return File('$baseDir/$value').path;
}

Uint8List _copyBytes(TypedData data) {
  return Uint8List.fromList(
    data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
  );
}

int _floatToHalf(double value) {
  final bytes = ByteData(4)..setFloat32(0, value, Endian.host);
  final bits = bytes.getUint32(0, Endian.host);
  final sign = (bits >> 16) & 0x8000;
  final exponent = ((bits >> 23) & 0xff) - 127 + 15;
  final mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    final shifted = (mantissa | 0x800000) >> (1 - exponent);
    return sign | ((shifted + 0x1000) >> 13);
  }
  if (exponent >= 0x1f) {
    if (mantissa == 0) return sign | 0x7c00;
    return sign | 0x7c00 | (mantissa >> 13) | 1;
  }
  return sign | (exponent << 10) | ((mantissa + 0x1000) >> 13);
}
