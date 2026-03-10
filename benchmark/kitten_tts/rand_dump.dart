import 'dart:convert';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main() {
  MlxRuntime.seed(0);
  final randIni = mx.random.normal([1, 9]).astype(MlxDType.MLX_FLOAT32);
  final noise = mx.random.normal([1, 20, 9]).astype(MlxDType.MLX_FLOAT32);
  try {
    final payload = <String, Object?>{
      'rand_ini': List<double>.from(randIni.reshape([9]).toList().cast<double>()),
      'noise': List<double>.from(
        noise.reshape([20 * 9]).toList().cast<double>(),
      ),
    };
    print(jsonEncode(payload));
  } finally {
    noise.close();
    randIni.close();
  }
}
