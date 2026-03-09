import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main() {
  stdout.writeln('dart_mlx_ffi distributed smoke');
  stdout.writeln('version: ${MlxVersion.current()}');

  final available = mx.distributed.isAvailable();
  stdout.writeln('available: $available');
  if (!available) {
    stdout.writeln(
      'skip: MLX distributed backend is not available in this runtime.',
    );
    return;
  }

  final group = mx.distributed.init(strict: false);
  final stream = MlxStream.defaultCpu();
  final payload = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [4]);

  try {
    stdout.writeln('group: rank=${group.rank} size=${group.size}');

    _reportCollective(
      'allGather',
      () => mx.distributed.allGather(payload, group: group, stream: stream),
    );
    _reportCollective(
      'allSum',
      () => mx.distributed.allSum(payload, group: group, stream: stream),
    );
    _reportCollective(
      'allMax',
      () => mx.distributed.allMax(payload, group: group, stream: stream),
    );
    _reportCollective(
      'allMin',
      () => mx.distributed.allMin(payload, group: group, stream: stream),
    );
    _reportCollective(
      'sumScatter',
      () => mx.distributed.sumScatter(payload, group: group, stream: stream),
    );

    stdout.writeln(
      'note: send/recv paths are skipped here because they require',
    );
    stdout.writeln(
      'a launcher-configured multi-rank environment to be meaningful.',
    );
  } finally {
    payload.close();
    stream.close();
    group.close();
  }
}

void _reportCollective(String name, MlxArray Function() callback) {
  try {
    final result = callback();
    try {
      stdout.writeln('$name: ok shape=${result.shape}');
    } finally {
      result.close();
    }
  } on MlxException catch (error) {
    stdout.writeln('$name: runtime-error ${error.message}');
  }
}
