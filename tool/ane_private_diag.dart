import 'dart:io';
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

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
  final probe = mx.anePrivate.probe();
  stdout.writeln('compiled=${probe.compiled}');
  stdout.writeln('enabled=${probe.enabled}');
  stdout.writeln('frameworkLoaded=${probe.frameworkLoaded}');
  stdout.writeln('supportsBasicEval=${probe.supportsBasicEval}');
  stdout.writeln('supportsRealtimeEval=${probe.supportsRealtimeEval}');
  stdout.writeln('supportsChaining=${probe.supportsChaining}');
  stdout.writeln('supportsPerfStats=${probe.supportsPerfStats}');

  if (!probe.frameworkLoaded || !probe.compiled || !probe.enabled) {
    stdout.writeln('private ANE runtime unavailable; exiting after probe');
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

  try {
    model.compile();
    model.load();
    stdout.writeln('modelLoaded=${model.isLoaded}');
    stdout.writeln('compiledModelExists=${model.compiledModelExists}');
    stdout.writeln('hexIdentifier=${model.hexIdentifier}');

    final session = model.createSession(
      inputByteSizes: const [_identityBytes],
      outputByteSizes: const [_identityBytes],
    );
    try {
      final input = Uint8List.fromList(
        List<int>.generate(_identityBytes, (index) => index & 0xFF),
      );
      final outputs = session.run([input]);
      stdout.writeln(
        'sessionRoundTrip=${outputs.length == 1 && _equalBytes(outputs.first, input)}',
      );

      final chaining = session.probeChaining(
        callEnqueueSets: true,
        callBuffersReady: true,
        useSharedSignalEvent: true,
      );
      stdout.writeln('chainingStage=${chaining.stage}');
      stdout.writeln('chainingPrepared=${chaining.prepared}');
      stdout.writeln('chainingError=${chaining.error}');

      final runner = model.createRunner(
        inputByteSizes: const [_identityBytes],
        outputByteSizes: const [_identityBytes],
        enableChain: true,
        useSharedSignalEvent: true,
      );
      try {
        runner.prepare();
        final runnerOutputs = runner.runBytes([input]);
        stdout.writeln(
          'runnerRoundTrip=${runnerOutputs.length == 1 && _equalBytes(runnerOutputs.first, input)}',
        );
      } finally {
        runner.close();
      }

      final decode = model.createDecodeRunner(
        inputByteSizes: const [_identityBytes],
        outputByteSizes: const [_identityBytes],
        tokenInputIndex: 0,
        tokenOutputIndex: 0,
        enableChain: true,
        useSharedSignalEvent: true,
      );
      try {
        decode.prepare();
        final generated = decode.generateBytes(input, 2);
        stdout.writeln('decodeSteps=${generated.length}');
      } finally {
        decode.close();
      }
    } finally {
      session.close();
    }
  } on MlxException catch (error) {
    stdout.writeln('private ANE error: $error');
  } finally {
    model.close();
  }
}

bool _equalBytes(Uint8List a, Uint8List b) {
  if (a.length != b.length) {
    return false;
  }
  for (var index = 0; index < a.length; index++) {
    if (a[index] != b[index]) {
      return false;
    }
  }
  return true;
}
