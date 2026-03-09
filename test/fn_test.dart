// ignore_for_file: unused_import

@TestOn('mac-os')

library;

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/raw.dart' as raw;
import 'package:dart_mlx_ffi/src/internal_hooks.dart' as hooks;

void main() {
  test('supports Dart-backed MlxFunction callbacks', () {
    final fn = MlxFunction.fromCallback((args) {
      final x = args[0];
      final y = args[1];
      return [x + y, x * y];
    });
    final x = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final y = MlxArray.fromFloat32List([3, 4], shape: [2]);

    try {
      final outputs = fn([x, y]);
      try {
        expect(outputs, hasLength(2));
        expect(outputs[0].toList(), <Object>[4.0, 6.0]);
        expect(outputs[1].toList(), <Object>[3.0, 8.0]);
      } finally {
        for (final output in outputs) {
          output.close();
        }
      }
    } finally {
      y.close();
      x.close();
      fn.close();
    }
  });

  test('supports Dart-backed MlxKwFunction callbacks', () {
    final fn = MlxKwFunction.fromCallback((args, kwargs) {
      final x = args[0];
      final bias = kwargs['bias']!;
      return [x + bias];
    });
    final x = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final bias = MlxArray.fromFloat32List([3, 4], shape: [2]);

    try {
      final outputs = fn([x], kwargs: {'bias': bias});
      try {
        expect(outputs, hasLength(1));
        expect(outputs[0].toList(), <Object>[4.0, 6.0]);
      } finally {
        for (final output in outputs) {
          output.close();
        }
      }
    } finally {
      bias.close();
      x.close();
      fn.close();
    }
  });

  test('supports compile and checkpoint for MlxFunction', () {
    final fn = MlxFunction.fromCallback((args) {
      final x = args[0];
      return [x * x];
    });
    final compiled = fn.compile();
    final compiledViaStatic = MlxCompile.compile(fn);
    final checkpointed = fn.checkpoint();
    final x = MlxArray.fromFloat32List([3], shape: [1]);

    try {
      MlxCompile.enable();
      MlxCompile.setMode(MlxCompileMode.MLX_COMPILE_MODE_ENABLED);
      final compiledOut = compiled([x]);
      final compiledStaticOut = compiledViaStatic([x]);
      final checkpointedOut = checkpointed([x]);
      try {
        expect(compiledOut[0].toList(), <Object>[9.0]);
        expect(compiledStaticOut[0].toList(), <Object>[9.0]);
        expect(checkpointedOut[0].toList(), <Object>[9.0]);
      } finally {
        for (final output in compiledOut) {
          output.close();
        }
        for (final output in compiledStaticOut) {
          output.close();
        }
        for (final output in checkpointedOut) {
          output.close();
        }
      }
      MlxCompile.clearCache();
      MlxCompile.disable();
    } finally {
      x.close();
      checkpointed.close();
      compiledViaStatic.close();
      compiled.close();
      fn.close();
    }
  });

  test('compiled and checkpointed functions survive parent close order', () {
    final fn = MlxFunction.fromCallback((args) => [args[0] * args[0]]);
    final compiled = fn.compile();
    final checkpointed = fn.checkpoint();
    final x = MlxArray.full([], 5);

    try {
      fn.close();
      final compiledOut = compiled([x]);
      final checkpointedOut = checkpointed([x]);
      try {
        expect(compiledOut[0].toList(), <Object>[25.0]);
        expect(checkpointedOut[0].toList(), <Object>[25.0]);
      } finally {
        for (final output in compiledOut) {
          output.close();
        }
        for (final output in checkpointedOut) {
          output.close();
        }
      }
    } finally {
      x.close();
      checkpointed.close();
      compiled.close();
    }
  });

  test('supports jvp, vjp, and valueAndGrad', () {
    final fn = MlxFunction.fromCallback((args) {
      final x = args[0];
      return [x * x];
    });
    final x = MlxArray.full([], 3);
    final tangent = MlxArray.full([], 1);
    final cotangent = MlxArray.full([], 1);

    try {
      final jvp = MlxTransforms.jvp(fn, [x], [tangent]);
      final vjp = MlxTransforms.vjp(fn, [x], [cotangent]);
      final vag = MlxTransforms.valueAndGrad(fn, [x]);
      try {
        expect(jvp.outputs, hasLength(1));
        expect(jvp.tangents, hasLength(1));
        expect(jvp.outputs[0].toList(), <Object>[9.0]);
        expect(jvp.tangents[0].toList(), <Object>[6.0]);

        expect(vjp.outputs, hasLength(1));
        expect(vjp.cotangents, hasLength(1));
        expect(vjp.outputs[0].toList(), <Object>[9.0]);
        expect(vjp.cotangents[0].toList(), <Object>[6.0]);

        expect(vag.values, hasLength(1));
        expect(vag.gradients, hasLength(1));
        expect(vag.values[0].toList(), <Object>[9.0]);
        expect(vag.gradients[0].toList(), <Object>[6.0]);
      } finally {
        for (final value in jvp.outputs) {
          value.close();
        }
        for (final value in jvp.tangents) {
          value.close();
        }
        for (final value in vjp.outputs) {
          value.close();
        }
        for (final value in vjp.cotangents) {
          value.close();
        }
        for (final value in vag.values) {
          value.close();
        }
        for (final value in vag.gradients) {
          value.close();
        }
      }
    } finally {
      cotangent.close();
      tangent.close();
      x.close();
      fn.close();
    }
  });

  test('supports reusable valueAndGrad function objects', () {
    final fn = MlxFunction.fromCallback((args) => [args[0] * args[0]]);
    final vag = fn.valueAndGrad(argnums: [0]);
    final x1 = MlxArray.full([], 2);
    final x2 = MlxArray.full([], 4);

    try {
      fn.close();
      final first = vag([x1]);
      final second = vag([x2]);
      try {
        expect(first.values[0].toList(), <Object>[4.0]);
        expect(first.gradients[0].toList(), <Object>[4.0]);
        expect(second.values[0].toList(), <Object>[16.0]);
        expect(second.gradients[0].toList(), <Object>[8.0]);
      } finally {
        for (final value in first.values) {
          value.close();
        }
        for (final value in first.gradients) {
          value.close();
        }
        for (final value in second.values) {
          value.close();
        }
        for (final value in second.gradients) {
          value.close();
        }
      }
    } finally {
      x2.close();
      x1.close();
      vag.close();
    }
  });

  test('supports custom_vjp overrides', () {
    final fn = MlxFunction.fromCallback((args) => [args[0] * args[0]]);
    final custom = MlxCustomVjp.fromCallback((primals, cotangents, outputs) {
      return [MlxArray.full([], 100)];
    });
    final wrapped = fn.customVjp(custom);
    final x = MlxArray.full([], 3);
    final cotangent = MlxArray.full([], 1);

    try {
      fn.close();
      custom.close();
      final result = MlxTransforms.vjp(wrapped, [x], [cotangent]);
      try {
        expect(result.outputs[0].toList(), <Object>[9.0]);
        expect(result.cotangents[0].toList(), <Object>[100.0]);
      } finally {
        for (final value in result.outputs) {
          value.close();
        }
        for (final value in result.cotangents) {
          value.close();
        }
      }
    } finally {
      cotangent.close();
      x.close();
      wrapped.close();
    }
  });

  test('supports custom_function with custom JVP', () {
    final fn = MlxFunction.fromCallback((args) => [args[0] * args[0]]);
    final customJvp = MlxCustomJvp.fromCallback((primals, tangents, argnums) {
      final six = MlxArray.full([], 6);
      final result = tangents[0] * six;
      six.close();
      return [result];
    });
    final wrapped = fn.customFunction(jvp: customJvp);
    final x = MlxArray.full([], 3);
    final tangent = MlxArray.full([], 1);

    try {
      fn.close();
      customJvp.close();
      final result = MlxTransforms.jvp(wrapped, [x], [tangent]);
      try {
        expect(result.outputs[0].toList(), <Object>[9.0]);
        expect(result.tangents[0].toList(), <Object>[6.0]);
      } finally {
        for (final value in result.outputs) {
          value.close();
        }
        for (final value in result.tangents) {
          value.close();
        }
      }
    } finally {
      tangent.close();
      x.close();
      wrapped.close();
    }
  });

  test('supports export and imported function roundtrip', () {
    final fn = MlxFunction.fromCallback((args) {
      final x = args[0];
      return [x * x + x];
    });
    final sample = MlxArray.full([], 3);
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_export_');
    final file = '${dir.path}/fn.mlx';

    try {
      MlxExport.exportFunction(file, fn, [sample]);
      final imported = MlxExport.importFunction(file);
      try {
        final outputs = imported([sample]);
        try {
          expect(outputs, hasLength(1));
          expect(outputs[0].toList(), <Object>[12.0]);
        } finally {
          for (final output in outputs) {
            output.close();
          }
        }
      } finally {
        imported.close();
      }
    } finally {
      dir.deleteSync(recursive: true);
      sample.close();
      fn.close();
    }
  });

  test('supports incremental exporter roundtrip', () {
    final fn = MlxFunction.fromCallback((args) => [args[0] * args[0]]);
    final sampleA = MlxArray.full([], 2);
    final sampleB = MlxArray.fromFloat32List([3, 4], shape: [2]);
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_xfunc_');
    final file = '${dir.path}/fn.mlx';

    try {
      final exporter = MlxExport.exporter(file, fn);
      try {
        exporter.addSample([sampleA]);
        exporter.addSample([sampleB]);
      } finally {
        exporter.close();
      }
      expect(() => exporter.addSample([sampleA]), throwsStateError);

      final imported = MlxExport.importFunction(file);
      try {
        final outputs = imported([sampleB]);
        try {
          expect(outputs, hasLength(1));
          expect(outputs[0].toList(), <Object>[9.0, 16.0]);
        } finally {
          for (final output in outputs) {
            output.close();
          }
        }
      } finally {
        imported.close();
      }
      expect(() => imported([sampleB]), throwsStateError);
    } finally {
      dir.deleteSync(recursive: true);
      sampleB.close();
      sampleA.close();
      fn.close();
    }
  });

  test('supports kwargs export and imported function kwargs calls', () {
    final fn = MlxKwFunction.fromCallback((args, kwargs) {
      final x = args[0];
      final bias = kwargs['bias']!;
      return [x + bias];
    });
    final sample = MlxArray.full([], 3);
    final bias = MlxArray.full([], 4);
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_kw_export_');
    final file = '${dir.path}/fn.mlx';

    try {
      MlxExport.exportKwFunction(file, fn, [sample], kwargs: {'bias': bias});
      final imported = MlxExport.importFunction(file);
      try {
        final outputs = imported([sample], kwargs: {'bias': bias});
        try {
          expect(outputs, hasLength(1));
          expect(outputs[0].toList(), <Object>[7.0]);
        } finally {
          for (final output in outputs) {
            output.close();
          }
        }
      } finally {
        imported.close();
      }
    } finally {
      dir.deleteSync(recursive: true);
      bias.close();
      sample.close();
      fn.close();
    }
  });

  test('supports safetensors save/load roundtrip', () {
    final weights = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final bias = MlxArray.fromFloat32List([5, 6], shape: [2]);
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_st_');
    final file = '${dir.path}/model.safetensors';

    try {
      mx.io.saveSafetensors(
        file,
        {'weights': weights, 'bias': bias},
        metadata: {'author': 'binbinsh', 'format': 'test'},
      );

      final loaded = mx.io.loadSafetensors(file);
      try {
        expect(loaded.tensors.keys.toSet(), {'weights', 'bias'});
        expect(loaded.metadata['author'], 'binbinsh');
        expect(loaded.metadata['format'], 'test');
        expect(
          loaded.tensors['weights']!.toList(),
          <Object>[1.0, 2.0, 3.0, 4.0],
        );
        expect(loaded.tensors['bias']!.toList(), <Object>[5.0, 6.0]);
      } finally {
        for (final tensor in loaded.tensors.values) {
          tensor.close();
        }
      }
    } finally {
      bias.close();
      weights.close();
      if (Directory(dir.path).existsSync()) {
        dir.deleteSync(recursive: true);
      }
    }
  });

  test('surfaces native MLX errors as MlxException', () {
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([1, 2, 3], shape: [3, 1]);
    final nonKey = MlxArray.fromFloat32List([1, 2], shape: [2]);

    try {
      expect(() => MlxOps.matmul(a, b), throwsA(isA<MlxException>()));
      expect(() => MlxRandom.split(nonKey), throwsA(isA<MlxException>()));
    } finally {
      nonKey.close();
      b.close();
      a.close();
    }
  });

  test('surfaces Dart callback failures as MlxException', () {
    final fn = MlxFunction.fromCallback((_) {
      throw StateError('callback boom');
    });
    final x = MlxArray.fromFloat32List([1], shape: [1]);
    try {
      expect(() => fn([x]), throwsA(isA<MlxException>()));
    } finally {
      x.close();
      fn.close();
    }
  });

  test('surfaces array-creation failures as MlxException', () {
    hooks.debugArrayFromBoolOverride = (_, _, _) => ffi.nullptr.cast();
    hooks.debugArrayFromInt32Override = (_, _, _) => ffi.nullptr.cast();
    hooks.debugArrayFromFloat32Override = (_, _, _) => ffi.nullptr.cast();
    hooks.debugArrayFromFloat64Override = (_, _, _) => ffi.nullptr.cast();
    hooks.debugArrayFromInt64Override = (_, _, _) => ffi.nullptr.cast();
    hooks.debugArrayFromUint64Override = (_, _, _) => ffi.nullptr.cast();

    try {
      expect(
        () => MlxArray.fromBoolList([true], shape: [1]),
        throwsA(isA<MlxException>()),
      );
      expect(
        () => MlxArray.fromInt32List([1], shape: [1]),
        throwsA(isA<MlxException>()),
      );
      expect(
        () => MlxArray.fromFloat32List([1], shape: [1]),
        throwsA(isA<MlxException>()),
      );
      expect(
        () => MlxArray.fromFloat64List([1], shape: [1]),
        throwsA(isA<MlxException>()),
      );
      expect(
        () => MlxArray.fromInt64List([1], shape: [1]),
        throwsA(isA<MlxException>()),
      );
      expect(
        () => MlxArray.fromUint64List([1], shape: [1]),
        throwsA(isA<MlxException>()),
      );
    } finally {
      hooks.resetDebugHooks();
    }
  });

  test('close is idempotent and prevents further use', () {
    final array = MlxArray.fromFloat32List([1, 2], shape: [2]);
    final function = MlxFunction.fromCallback((args) => [args[0]]);
    final vag = function.valueAndGrad();

    expect(array.isClosed, isFalse);
    array.close();
    expect(array.isClosed, isTrue);
    array.close();

    expect(() => array.eval(), throwsStateError);
    expect(() => array.toList(), throwsStateError);
    expect(() => array.toString(), throwsStateError);

    function.close();
    final input = MlxArray.fromFloat32List([1], shape: [1]);
    try {
      expect(() => function([input]), throwsStateError);
    } finally {
      input.close();
    }

    vag.close();
    final scalar = MlxArray.full([], 1);
    expect(
      () => vag([scalar]),
      throwsStateError,
    );
    scalar.close();
  });
}
