library;

import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

final class NpyArray {
  NpyArray({required this.shape, required this.values});

  final List<int> shape;
  final Float32List values;

  MlxArray toMlxArray() => MlxArray.fromFloat32List(values, shape: shape);
}

Map<String, NpyArray> loadNpzFloat32(String path) {
  final bytes = File(path).readAsBytesSync();
  final archive = ZipDecoder().decodeBytes(bytes);
  final out = <String, NpyArray>{};
  for (final file in archive.files) {
    if (!file.isFile) {
      continue;
    }
    final name = file.name.endsWith('.npy')
        ? file.name.substring(0, file.name.length - 4)
        : file.name;
    final fileBytes = file.content;
    out[name] = parseNpyFloat32(fileBytes);
  }
  return out;
}

NpyArray parseNpyFloat32(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  if (bytes.length < 10 ||
      bytes[0] != 0x93 ||
      String.fromCharCodes(bytes.sublist(1, 6)) != 'NUMPY') {
    throw FormatException('Invalid .npy header.');
  }

  final major = bytes[6];
  final minor = bytes[7];
  if (major != 1 && major != 2 && major != 3) {
    throw UnsupportedError('Unsupported .npy version $major.$minor.');
  }

  final headerLenOffset = 8;
  final headerLen = switch (major) {
    1 => data.getUint16(headerLenOffset, Endian.little),
    _ => data.getUint32(headerLenOffset, Endian.little),
  };
  final headerOffset = major == 1 ? 10 : 12;
  final headerEnd = headerOffset + headerLen;
  final header = String.fromCharCodes(bytes.sublist(headerOffset, headerEnd));

  final descr = _extract(header, RegExp(r"'descr':\s*'([^']+)'"));
  if (descr != '<f4' && descr != '|f4') {
    throw UnsupportedError(
      'Only little-endian float32 npy is supported, got $descr.',
    );
  }
  final fortran = _extract(header, RegExp(r"'fortran_order':\s*(True|False)"));
  if (fortran != 'False') {
    throw UnsupportedError('Fortran-order npy arrays are not supported.');
  }
  final shapeText = _extract(header, RegExp(r"'shape':\s*\(([^)]*)\)"));
  final shape = shapeText
      .split(',')
      .map((part) => part.trim())
      .where((part) => part.isNotEmpty)
      .map(int.parse)
      .toList(growable: false);
  if (shape.isEmpty) {
    throw FormatException('Empty npy shape.');
  }

  final count = shape.fold<int>(1, (acc, dim) => acc * dim);
  final payload = bytes.sublist(headerEnd);
  if (payload.length < count * 4) {
    throw FormatException('Truncated npy payload.');
  }
  final values = Float32List(count);
  final payloadView = ByteData.sublistView(
    Uint8List.sublistView(bytes, headerEnd),
  );
  for (var i = 0; i < count; i++) {
    values[i] = payloadView.getFloat32(i * 4, Endian.little);
  }
  return NpyArray(shape: shape, values: values);
}

String _extract(String input, RegExp pattern) {
  final match = pattern.firstMatch(input);
  if (match == null || match.groupCount < 1) {
    throw FormatException('Unable to parse npy header: $input');
  }
  return match.group(1)!;
}
