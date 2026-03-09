// ignore_for_file: unused_import

@TestOn('mac-os')

library;

import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/raw.dart' as raw;
import 'package:dart_mlx_ffi/src/internal_hooks.dart' as hooks;

void main() {
  test('supports stream-aware file IO and byte IO', () {
    final array = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final stream = MlxStream.defaultCpu();
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_io_stream_');
    final file = '${dir.path}/array.npy';

    try {
      mx.io.save(file, array);
      final loaded = mx.io.load(file, stream: stream);
      final bytes = mx.io.saveBytes(array);
      final loadedBytes = mx.io.loadBytes(bytes, stream: stream);
      try {
        expect(loaded.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
        expect(loadedBytes.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
      } finally {
        loadedBytes.close();
        loaded.close();
      }
    } finally {
      dir.deleteSync(recursive: true);
      stream.close();
      array.close();
    }
  });

  test('supports stream-aware safetensors bytes IO', () {
    final weights = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final bias = MlxArray.fromFloat32List([5, 6], shape: [2]);
    final stream = MlxStream.defaultCpu();

    try {
      final bytes = mx.io.saveSafetensorsBytes(
        {'weights': weights, 'bias': bias},
        metadata: {'author': 'binbinsh'},
      );
      final loaded = mx.io.loadSafetensorsBytes(bytes, stream: stream);
      try {
        expect(loaded.tensors.keys.toSet(), {'weights', 'bias'});
        expect(loaded.metadata['author'], 'binbinsh');
        expect(
          loaded.tensors['weights']!.toList(),
          <Object>[1.0, 2.0, 3.0, 4.0],
        );
        expect(loaded.tensors['bias']!.toList(), <Object>[5.0, 6.0]);
      } finally {
        for (final value in loaded.tensors.values) {
          value.close();
        }
      }
    } finally {
      stream.close();
      bias.close();
      weights.close();
    }
  });

  test('supports reusable bytes reader and writer objects', () {
    final array = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final writer = MlxBytesWriter();

    try {
      mx.io.saveWriter(writer, array);
      final bytes = writer.bytes;
      expect(bytes, isNotEmpty);
      writer.rewind();

      final reader = MlxBytesReader(bytes);
      try {
        final first = mx.io.loadReader(reader);
        try {
          expect(first.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
        } finally {
          first.close();
        }
        reader.rewind();
        final second = mx.io.loadReader(reader);
        try {
          expect(second.toList(), <Object>[1.0, 2.0, 3.0, 4.0]);
        } finally {
          second.close();
        }
        expect(reader.toString(), isNotEmpty);
      } finally {
        reader.close();
      }
      expect(writer.toString(), isNotEmpty);
    } finally {
      writer.close();
      array.close();
    }
  });
}
