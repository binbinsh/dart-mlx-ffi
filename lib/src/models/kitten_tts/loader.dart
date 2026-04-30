library;

import 'package:dart_inference/mlx.dart';

import 'args.dart';
import 'npz.dart';

Map<String, MlxArray> loadKittenTensors(String snapshotPath) =>
    mx.io.loadSafetensors('$snapshotPath/model.safetensors').tensors;

Map<String, NpyArray> loadKittenVoices(
  String snapshotPath,
  ModelConfig config,
) => loadNpzFloat32('$snapshotPath/${config.voicesPath}');
