import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

typedef _RealSample = ({int label, List<double> input5, List<double> logits});
typedef _RealSpec = ({
  String name,
  int outputChannels,
  List<double> weights,
  List<_RealSample> samples,
});

const _irisInputBytes = 10;

const _iris3Weights = <double>[
  -2.4592809677124023,
  2.380864381790161,
  -4.534583568572998,
  -4.283433437347412,
  -0.3940414488315582,
  1.8275331258773804,
  -0.463300496339798,
  -1.8468471765518188,
  -1.7169692516326904,
  5.267071723937988,
  0.6317527890205383,
  -1.917563796043396,
  6.3814287185668945,
  6.00039005279541,
  -4.8730363845825195,
];

const _irisBinaryWeights = <double>[
  -0.895805299282074,
  1.4704875946044922,
  -2.066481351852417,
  -1.8808748722076416,
  -2.147366523742676,
  0.8958030939102173,
  -1.470487117767334,
  2.0664796829223633,
  1.8808743953704834,
  2.1473684310913086,
];

const _iris3Samples = <_RealSample>[
  (
    label: 0,
    input5: [
      -0.9006831645965576,
      1.032057285308838,
      -1.3412725925445557,
      -1.3129769563674927,
      1.0,
    ],
    logits: [15.984342002859592, 7.8743573148955335, -23.858690481618893],
  ),
  (
    label: 1,
    input5: [
      1.4015071392059326,
      0.33784863352775574,
      0.5352959036827087,
      0.26469850540161133,
      1.0,
    ],
    logits: [-6.597531942408246, 6.228758084325478, 0.3687702827411634],
  ),
];

const _irisBinarySamples = <_RealSample>[
  (
    label: 0,
    input5: [
      -0.9006831645965576,
      1.032057285308838,
      -1.3412725925445557,
      -1.3129769563674927,
      1.0,
    ],
    logits: [5.41835782830826, -5.41835057792872],
  ),
  (
    label: 1,
    input5: [
      1.4015071392059326,
      0.33784863352775574,
      0.5352959036827087,
      0.26469850540161133,
      1.0,
    ],
    logits: [-4.51008559177248, 4.510083549785989],
  ),
];

const _specs = <_RealSpec>[
  (
    name: 'iris-3class',
    outputChannels: 3,
    weights: _iris3Weights,
    samples: _iris3Samples,
  ),
  (
    name: 'iris-binary',
    outputChannels: 2,
    weights: _irisBinaryWeights,
    samples: _irisBinarySamples,
  ),
];

String _convMil(int outputChannels) =>
    '''
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 5, 1, 1]> x) {
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, 5, 1, 1]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        tensor<fp16, [$outputChannels, 5, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [$outputChannels, 5, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, $outputChannels, 1, 1]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, $outputChannels, 1, 1]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    } -> (y);
}
''';

void _writeLe32(List<int> bytes, int offset, int value) {
  bytes[offset] = value & 0xFF;
  bytes[offset + 1] = (value >> 8) & 0xFF;
  bytes[offset + 2] = (value >> 16) & 0xFF;
  bytes[offset + 3] = (value >> 24) & 0xFF;
}

Uint8List _weightBlob(List<double> weights) {
  final payloadBytes = weights.length * 2;
  final data = Uint8List(128 + payloadBytes);
  data[0] = 1;
  data[4] = 2;
  data[64] = 0xEF;
  data[65] = 0xBE;
  data[66] = 0xAD;
  data[67] = 0xDE;
  data[68] = 1;
  _writeLe32(data, 72, payloadBytes);
  _writeLe32(data, 80, 128);
  final payload = mx.anePrivate.encodeFp16Bytes(
    Float32List.fromList(weights.map((value) => value.toDouble()).toList()),
  );
  data.setRange(128, 128 + payload.length, payload);
  return data;
}

void main() {
  final warmup =
      int.tryParse(Platform.environment['ANE_REAL_BENCH_WARMUP'] ?? '') ?? 2;
  final iters =
      int.tryParse(Platform.environment['ANE_REAL_BENCH_ITERS'] ?? '') ?? 10;
  final maxMs = double.tryParse(
    Platform.environment['ANE_REAL_BENCH_MAX_MS'] ?? '',
  );

  final probe = mx.anePrivate.probe();
  stdout.writeln('enabled=${probe.enabled}');
  stdout.writeln('frameworkLoaded=${probe.frameworkLoaded}');
  stdout.writeln('supportsBasicEval=${probe.supportsBasicEval}');
  if (!probe.enabled || !probe.frameworkLoaded || !probe.supportsBasicEval) {
    stdout.writeln('private ANE unavailable; exiting');
    return;
  }

  for (final spec in _specs) {
    stdout.writeln('model=${spec.name}');
    final model = mx.anePrivate.modelFromMil(
      _convMil(spec.outputChannels),
      weights: [
        (
          path: '@model_path/weights/weight.bin',
          data: _weightBlob(spec.weights),
        ),
      ],
    );
    try {
      model.compile();
      model.load();
      final session = model.createSession(
        inputByteSizes: const [_irisInputBytes],
        outputByteSizes: [spec.outputChannels * 2],
      );
      try {
        for (var i = 0; i < warmup; i++) {
          session.runFloat32([Float32List.fromList(spec.samples.first.input5)]);
        }
        final stopwatch = Stopwatch()..start();
        for (var i = 0; i < iters; i++) {
          session.runFloat32([
            Float32List.fromList(spec.samples[i % spec.samples.length].input5),
          ]);
        }
        stopwatch.stop();
        final perIterMs = stopwatch.elapsedMicroseconds / 1000.0 / iters;
        stdout.writeln('perIterMs=${perIterMs.toStringAsFixed(4)}');
        if (maxMs != null && perIterMs > maxMs) {
          throw StateError(
            '${spec.name} exceeded max ms/iter: $perIterMs > $maxMs',
          );
        }
      } finally {
        session.close();
      }
    } on MlxException catch (error) {
      stdout.writeln('aneError=$error');
    } finally {
      model.close();
    }
  }
}
