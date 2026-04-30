/// Stable, documented Dart API for common MLX workflows.
library;

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'internal_hooks.dart' as hooks;
import 'raw/raw.dart' as raw;
import 'shim_bindings.dart' as shim;

part 'stable/core.dart';
part 'stable/array.dart';
part 'stable/dev.dart';
part 'stable/sys.dart';
part 'stable/fun.dart';
part 'stable/ops.dart';
part 'stable/more.dart';
part 'stable/scan.dart';
part 'stable/misc.dart';
part 'stable/conv.dart';
part 'stable/extra.dart';
part 'stable/tensor.dart';
part 'stable/xform.dart';
part 'stable/mod.dart';
part 'stable/io.dart';
part 'stable/io_mem.dart';
part 'stable/math.dart';
part 'stable/quant.dart';
part 'stable/fast.dart';
part 'stable/export.dart';
part 'stable/rt.dart';
part 'stable/fast_k.dart';
part 'stable/util.dart';

typedef MlxDType = raw.mlx_dtype_;
typedef MlxCompileMode = raw.mlx_compile_mode_;

/// Result pair returned by [MlxRandom.split].
typedef MlxRandomSplit = ({MlxArray first, MlxArray second});

/// Result pair returned by [MlxLinalg.qr].
typedef MlxQrResult = ({MlxArray q, MlxArray r});

/// Result pair returned by [MlxLinalg.eig].
typedef MlxEigResult = ({MlxArray values, MlxArray vectors});

/// Result returned by [MlxLinalg.svd].
typedef MlxSvdResult = ({MlxArray? u, MlxArray s, MlxArray? vt});

/// Result returned by [MlxLinalg.lu].
typedef MlxLuResult = ({MlxArray rowPivots, MlxArray l, MlxArray u});

/// Result returned by [MlxLinalg.luFactor].
typedef MlxLuFactorResult = ({MlxArray lu, MlxArray pivots});

/// Result returned by [MlxMisc.divmod].
typedef MlxDivModResult = ({MlxArray quotient, MlxArray remainder});

/// Result returned by [MlxIo.loadSafetensors].
typedef MlxSafetensorsData = ({
  Map<String, MlxArray> tensors,
  Map<String, String> metadata,
});

/// Dart callback shape used by [MlxFunction].
typedef MlxCallback = List<MlxArray> Function(List<MlxArray> args);

/// Dart callback shape used by [MlxKwFunction].
typedef MlxKwCallback =
    List<MlxArray> Function(List<MlxArray> args, Map<String, MlxArray> kwargs);

/// Dart callback shape used by [MlxCustomVjp].
typedef MlxCustomVjpCallback =
    List<MlxArray> Function(
      List<MlxArray> primals,
      List<MlxArray> cotangents,
      List<MlxArray> outputs,
    );

/// Dart callback shape used by [MlxCustomJvp].
typedef MlxCustomJvpCallback =
    List<MlxArray> Function(
      List<MlxArray> primals,
      List<MlxArray> tangents,
      List<int> argnums,
    );

/// Result returned by [MlxTransforms.jvp].
typedef MlxJvpResult = ({List<MlxArray> outputs, List<MlxArray> tangents});

/// Result returned by [MlxTransforms.vjp].
typedef MlxVjpResult = ({List<MlxArray> outputs, List<MlxArray> cotangents});

/// Result returned by [MlxTransforms.valueAndGrad].
typedef MlxValueAndGradResult = ({
  List<MlxArray> values,
  List<MlxArray> gradients,
});

/// Python-like module-style entrypoint for high-level MLX operations.
const mx = MlxModule._();
