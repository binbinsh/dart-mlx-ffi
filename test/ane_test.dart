import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:test/test.dart';

const _identityChannels = 4;
const _identitySpatial = 8;
const _identityBytes = _identityChannels * _identitySpatial * 2;

String _identityConvMil(int channels, int spatial) =>
    '''
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, $channels, 1, $spatial]> x) {
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        tensor<fp16, [$channels, $channels, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [$channels, $channels, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, $channels, 1, $spatial]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string("conv")];
    } -> (y);
}
''';

void _writeLe32(List<int> bytes, int offset, int value) {
  bytes[offset] = value & 0xFF;
  bytes[offset + 1] = (value >> 8) & 0xFF;
  bytes[offset + 2] = (value >> 16) & 0xFF;
  bytes[offset + 3] = (value >> 24) & 0xFF;
}

Uint8List _identityWeightBlob(int channels) {
  final payloadElements = channels * channels;
  final payloadBytes = payloadElements * 2;
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
  for (var index = 0; index < channels; index++) {
    final byteOffset = 128 + (index * channels + index) * 2;
    data[byteOffset] = 0x00;
    data[byteOffset + 1] = 0x3C;
  }
  return data;
}

void main() {
  group('private ANE bridge', () {
    final runtimeEnabled = mx.anePrivate.isEnabled();

    test('probe is internally consistent', () {
      final compiled = mx.anePrivate.isCompiled();
      final enabled = mx.anePrivate.isEnabled();
      final probe = mx.anePrivate.probe();

      expect(probe.compiled, compiled);
      expect(probe.enabled, enabled);
      expect(
        probe.frameworkPath,
        '/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine',
      );
      expect(probe.containsKey('supports_basic_eval'), isTrue);
      expect(probe.containsKey('supports_realtime_eval'), isTrue);
      expect(probe.containsKey('supports_chaining'), isTrue);
      expect(probe.containsKey('supports_perf_stats'), isTrue);
    });

    test('empty mil is rejected before touching native code', () {
      if (!runtimeEnabled) {
        expect(
          () => mx.anePrivate.modelFromMil(''),
          throwsA(isA<MlxException>()),
        );
        return;
      }
      expect(
        () => mx.anePrivate.modelFromMil(''),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('invalid mil reports a native error', () {
      expect(() {
        final model = mx.anePrivate.modelFromMil('this is not valid MIL');
        try {
          model.compile();
        } finally {
          model.close();
        }
      }, throwsA(isA<MlxException>()));
    });

    test('fp16 encode/decode helpers roundtrip basic values', () {
      if (!runtimeEnabled) {
        expect(
          () => mx.anePrivate.encodeFp16Bytes(Float32List.fromList([1])),
          throwsA(isA<MlxException>()),
        );
        return;
      }
      final input = Float32List.fromList([0, 1, -2, 3.5, 8, 16]);
      final encoded = mx.anePrivate.encodeFp16Bytes(input);
      final decoded = mx.anePrivate.decodeFp16Bytes(encoded);

      expect(encoded.length, input.length * 2);
      expect(decoded.length, input.length);
      for (var index = 0; index < input.length; index++) {
        expect(decoded[index], closeTo(input[index], 1e-3));
      }
    });

    test(
      'model lifecycle compiles or skips cleanly when runtime is unavailable',
      () {
        if (!runtimeEnabled) {
          expect(
            () => mx.anePrivate.modelFromMil(_identityConvMil(1, 1)),
            throwsA(isA<MlxException>()),
          );
          return;
        }
        final probe = mx.anePrivate.probe();
        if (!probe.frameworkLoaded) {
          return;
        }

        final model = mx.anePrivate.modelFromMil(
          _identityConvMil(_identityChannels, _identitySpatial),
          weights: [
            (
              path: '@model_path/weights/weight.bin',
              data: _identityWeightBlob(_identityChannels),
            ),
          ],
        );
        addTearDown(model.close);

        expect(model.isLoaded, isFalse);
        expect(model.compiledModelExists, anyOf(isFalse, isTrue));

        try {
          model.compile();
        } on MlxException catch (error) {
          expect(error.message, isNotEmpty);
          return;
        }
        expect(model.compiledModelExists, anyOf(isFalse, isTrue));

        model.load();
        expect(model.isLoaded, isTrue);
        expect(model.hexIdentifier, anyOf(isNull, isNotEmpty));

        final session = model.createSession(
          inputByteSizes: const [_identityBytes],
          outputByteSizes: const [_identityBytes],
        );
        addTearDown(session.close);

        final chaining = session.probeChaining(
          callEnqueueSets: true,
          callBuffersReady: true,
          useSharedSignalEvent: true,
        );
        expect(chaining.containsKey('stage'), isTrue);
        expect(chaining.containsKey('prepared'), isTrue);
        expect(chaining.containsKey('error'), isTrue);
        expect(chaining.containsKey('called_enqueue_sets'), isTrue);
        expect(chaining.containsKey('called_buffers_ready'), isTrue);

        try {
          final chain = session.createChain(
            useSharedSignalEvent: true,
            attemptPrepare: true,
          );
          addTearDown(chain.close);
          expect(chain.hasEnqueueSets, anyOf(isFalse, isTrue));
          expect(chain.hasBuffersReady, anyOf(isFalse, isTrue));
          expect(chain.isPrepared, anyOf(isFalse, isTrue));

          if (chain.hasBuffersReady) {
            try {
              chain.buffersReady();
            } on MlxException catch (error) {
              expect(error.message, isNotEmpty);
            }
          }
          if (chain.hasEnqueueSets) {
            try {
              chain.enqueueSets();
            } on MlxException catch (error) {
              expect(error.message, isNotEmpty);
            }
          }
        } on MlxException catch (error) {
          expect(error.message, isNotEmpty);
        }

        final input = Uint8List.fromList(
          List<int>.generate(_identityBytes, (index) => index & 0xFF),
        );
        final runner = model.createRunner(
          inputByteSizes: const [_identityBytes],
          outputByteSizes: const [_identityBytes],
          enableChain: true,
          useSharedSignalEvent: true,
        );
        addTearDown(runner.close);

        try {
          runner.prepare();
          final runnerOutputs = runner.runBytes([input]);
          expect(runnerOutputs, hasLength(1));
          expect(runnerOutputs.first, input);

          final runnerFp32Outputs = runner.runFloat32([
            Float32List.fromList(
              List<double>.generate(
                _identityChannels * _identitySpatial,
                (index) => index.toDouble(),
              ),
            ),
          ]);
          expect(runnerFp32Outputs, hasLength(1));

          final advance = runner.advanceChain();
          expect(advance.hasChain, isTrue);
        } on MlxException catch (error) {
          expect(error.message, isNotEmpty);
        }

        final loop = model.createLoopRunner(
          inputByteSizes: const [_identityBytes],
          outputByteSizes: const [_identityBytes],
          feedback: const [(inputIndex: 0, outputIndex: 0)],
          enableChain: true,
          useSharedSignalEvent: true,
        );
        addTearDown(loop.close);

        try {
          final first = loop.stepBytes(inputs: {0: input});
          expect(first.outputs, hasLength(1));
          expect(first.outputs.first, input);

          final second = loop.stepBytes();
          expect(second.outputs, hasLength(1));
          expect(second.outputs.first, input);

          final floatSeed = Float32List.fromList(
            List<double>.generate(
              _identityChannels * _identitySpatial,
              (index) => (index + 1).toDouble(),
            ),
          );
          final floatStep = loop.stepFloat32(inputs: {0: floatSeed});
          expect(floatStep.outputs, hasLength(1));
          expect(floatStep.outputs.first.length, floatSeed.length);
          for (var index = 0; index < floatSeed.length; index++) {
            expect(
              floatStep.outputs.first[index],
              closeTo(floatSeed[index], 1e-3),
            );
          }
        } on MlxException catch (error) {
          expect(error.message, isNotEmpty);
        }

        final decode = model.createDecodeRunner(
          inputByteSizes: const [_identityBytes],
          outputByteSizes: const [_identityBytes],
          tokenInputIndex: 0,
          tokenOutputIndex: 0,
          enableChain: true,
          useSharedSignalEvent: true,
        );
        addTearDown(decode.close);

        try {
          decode.prepare();
          final decodeStep = decode.stepBytes(input);
          expect(decodeStep.token, input);

          final generated = decode.generateBytes(input, 3);
          expect(generated, hasLength(3));
          for (final token in generated) {
            expect(token, input);
          }

          final floatSeed = Float32List.fromList(
            List<double>.generate(
              _identityChannels * _identitySpatial,
              (index) => (index + 2).toDouble(),
            ),
          );
          final floatStep = decode.stepFloat32(floatSeed);
          expect(floatStep.token.length, floatSeed.length);
          for (var index = 0; index < floatSeed.length; index++) {
            expect(floatStep.token[index], closeTo(floatSeed[index], 1e-3));
          }

          final floatGenerated = decode.generateFloat32(floatSeed, 2);
          expect(floatGenerated, hasLength(2));
        } on MlxException catch (error) {
          expect(error.message, isNotEmpty);
        }

        try {
          final outputs = session.run([input]);
          expect(outputs, hasLength(1));
          expect(outputs.first, input);

          final fp32Input = Float32List.fromList(
            List<double>.generate(
              _identityChannels * _identitySpatial,
              (index) => index.toDouble(),
            ),
          );
          final fp32Outputs = session.runFloat32([fp32Input]);
          expect(fp32Outputs, hasLength(1));
          expect(fp32Outputs.first.length, fp32Input.length);
          for (var index = 0; index < fp32Input.length; index++) {
            expect(fp32Outputs.first[index], closeTo(fp32Input[index], 1e-3));
          }

          if (probe.supportsRealtimeEval) {
            try {
              session.prepareRealtime();
              expect(session.isRealtimeLoaded, isTrue);

              final realtimeOutputs = session.runRealtime([input]);
              expect(realtimeOutputs, hasLength(1));
              expect(realtimeOutputs.first, input);

              final realtimeFp32Outputs = session.runFloat32Realtime([
                fp32Input,
              ]);
              expect(realtimeFp32Outputs, hasLength(1));
              expect(realtimeFp32Outputs.first.length, fp32Input.length);
              for (var index = 0; index < fp32Input.length; index++) {
                expect(
                  realtimeFp32Outputs.first[index],
                  closeTo(fp32Input[index], 1e-3),
                );
              }

              session.teardownRealtime();
              expect(session.isRealtimeLoaded, isFalse);
            } on MlxException catch (error) {
              expect(error.message, isNotEmpty);
            }
          }
        } on MlxException catch (error) {
          expect(error.message, isNotEmpty);
          return;
        }

        model.unload();
        expect(model.isLoaded, isFalse);
      },
    );
  });
}
