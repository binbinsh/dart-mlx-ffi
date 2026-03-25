import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

String _arg(List<String> args, String name) {
  final prefix = '$name=';
  for (final arg in args) {
    if (arg.startsWith(prefix)) {
      return arg.substring(prefix.length);
    }
  }
  throw ArgumentError('Missing $name');
}

Future<void> main(List<String> args) async {
  final samplesPath = _arg(args, '--samples');
  final outputData = _arg(args, '--out-data');
  final outputMeta = _arg(args, '--out-meta');
  final bundlePath = _arg(args, '--bundle');

  final sampleBytes = await File(samplesPath).readAsBytes();
  final samples = Float32List.sublistView(sampleBytes);
  final manifest = await readParakeetTdtManifest(bundlePath);
  final frontend = ParakeetTdtMelFrontend(manifest);
  final mel = frontend.compute(samples).astype(MlxDType.MLX_FLOAT32);
  await File(outputData).writeAsBytes(
    mel.toFloat32List().buffer.asUint8List(),
    flush: true,
  );
  await File(outputMeta).writeAsString(
    jsonEncode(<String, Object?>{
      'shape': mel.shape,
      'bundle': bundlePath,
      'samples': samplesPath,
    }),
  );
  mel.close();
}
