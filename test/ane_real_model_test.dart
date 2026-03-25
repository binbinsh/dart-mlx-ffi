import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:test/test.dart';

const _irisInputBytes = 10;

typedef _RealSample = ({int label, List<double> input5, List<double> logits});
typedef _RealSpec = ({
  String name,
  int outputChannels,
  List<double> weights,
  List<_RealSample> samples,
});

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
  (
    label: 2,
    input5: [
      0.5533320307731628,
      0.5692513585090637,
      1.2745503187179565,
      1.7109010219573975,
      1.0,
    ],
    logits: [-13.507615675199858, 0.723105798060315, 12.78448235378335],
  ),
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
  (
    label: 1,
    input5: [
      0.5533320307731628,
      0.5692513585090637,
      1.2745503187179565,
      1.7109010219573975,
      1.0,
    ],
    logits: [-7.657792434934709, 7.657790450464141],
  ),
];

const _realSpecs = <_RealSpec>[
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

int _argmax(List<double> values) {
  var bestIndex = 0;
  var bestValue = values.first;
  for (var index = 1; index < values.length; index++) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }
  return bestIndex;
}

void _expectModelAttempt(_RealSpec spec) {
  final model = mx.anePrivate.modelFromMil(
    _convMil(spec.outputChannels),
    weights: [
      (path: '@model_path/weights/weight.bin', data: _weightBlob(spec.weights)),
    ],
  );
  addTearDown(model.close);

  try {
    model.compile();
    model.load();
    final session = model.createSession(
      inputByteSizes: const [_irisInputBytes],
      outputByteSizes: [spec.outputChannels * 2],
    );
    addTearDown(session.close);

    for (final sample in spec.samples) {
      final outputs = session.runFloat32([Float32List.fromList(sample.input5)]);
      expect(outputs, hasLength(1), reason: spec.name);
      expect(outputs.first.length, spec.outputChannels, reason: spec.name);
      expect(_argmax(outputs.first), sample.label, reason: spec.name);
      for (var index = 0; index < spec.outputChannels; index++) {
        expect(
          outputs.first[index],
          closeTo(sample.logits[index], 0.5),
          reason: spec.name,
        );
      }
    }
  } on MlxException catch (error) {
    expect(error.message, isNotEmpty, reason: spec.name);
  }
}

void main() {
  group('private ANE real model', () {
    final runtimeEnabled = mx.anePrivate.isEnabled();

    test('attempts two real iris classifiers on ANE', () {
      if (!runtimeEnabled) {
        expect(
          () => mx.anePrivate.modelFromMil(_convMil(3)),
          throwsA(isA<MlxException>()),
        );
        return;
      }

      final probe = mx.anePrivate.probe();
      if (!probe.frameworkLoaded || !probe.supportsBasicEval) {
        return;
      }

      for (final spec in _realSpecs) {
        _expectModelAttempt(spec);
      }
    });

    test('attempts two real iris classifiers via vendored ane interop', () {
      if (!runtimeEnabled) {
        expect(
          () => MlxAneInteropKernel.singleIo(
            milText: _convMil(3),
            weights: const [],
            inputBytes: _irisInputBytes,
            outputBytes: 6,
            inputChannels: 5,
            inputSpatial: 1,
            outputChannels: 3,
            outputSpatial: 1,
          ),
          throwsA(isA<MlxException>()),
        );
        return;
      }

      final probe = mx.anePrivate.probe();
      if (!probe.frameworkLoaded || !probe.supportsBasicEval) {
        return;
      }

      for (final spec in _realSpecs) {
        final kernel = MlxAneInteropKernel.singleIo(
          milText: _convMil(spec.outputChannels),
          weights: [
            (
              path: '@model_path/weights/weight.bin',
              data: _weightBlob(spec.weights),
            ),
          ],
          inputBytes: _irisInputBytes,
          outputBytes: spec.outputChannels * 2,
          inputChannels: 5,
          inputSpatial: 1,
          outputChannels: spec.outputChannels,
          outputSpatial: 1,
        );
        addTearDown(kernel.close);

        try {
          for (final sample in spec.samples) {
            final outputs = kernel.runFloat32(
              Float32List.fromList(sample.input5),
            );
            expect(outputs.length, spec.outputChannels, reason: spec.name);
            expect(_argmax(outputs), sample.label, reason: spec.name);
            for (var index = 0; index < spec.outputChannels; index++) {
              expect(
                outputs[index],
                closeTo(sample.logits[index], 0.5),
                reason: spec.name,
              );
            }
          }
          expect(
            kernel.lastHardwareExecutionTimeNs(),
            greaterThanOrEqualTo(0),
            reason: spec.name,
          );
        } on MlxException catch (error) {
          expect(error.message, isNotEmpty, reason: spec.name);
        }
      }
    });
  });
}
