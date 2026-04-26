@ffi.DefaultAsset('package:dart_inference/rt_bindings.dart')
library;

import 'dart:ffi' as ffi;

final class TensorAbi extends ffi.Struct {
  @ffi.Int32()
  external int dtype;

  @ffi.Int32()
  external int rank;

  external ffi.Pointer<ffi.Int64> shape;

  @ffi.IntPtr()
  external int byteLength;

  external ffi.Pointer<ffi.Void> data;
}

final class NamedTensorAbi extends ffi.Struct {
  external ffi.Pointer<ffi.Char> name;

  external TensorAbi tensor;
}

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Int32,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_open')
external ffi.Pointer<ffi.Void> open(
  int engine,
  ffi.Pointer<ffi.Char> modelPath,
  ffi.Pointer<ffi.Char> optionsJson,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_close')
external void close(ffi.Pointer<ffi.Void> session);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<NamedTensorAbi>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<NamedTensorAbi>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_run')
external int run(
  ffi.Pointer<ffi.Void> session,
  ffi.Pointer<NamedTensorAbi> inputs,
  int inputCount,
  ffi.Pointer<ffi.Pointer<NamedTensorAbi>> outputs,
  ffi.Pointer<ffi.IntPtr> outputCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<NamedTensorAbi>, ffi.IntPtr)>(
  symbol: 'dinf_free_tensors',
)
external void freeTensors(ffi.Pointer<NamedTensorAbi> tensors, int count);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Char>)>(symbol: 'dinf_free_str')
external void freeStr(ffi.Pointer<ffi.Char> value);

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(symbol: 'dinf_info_json')
external ffi.Pointer<ffi.Char> infoJson();

@ffi.Native<ffi.Int32 Function()>(symbol: 'dinf_platform_id')
external int platformId();

@ffi.Native<ffi.Int32 Function(ffi.Int32)>(symbol: 'dinf_accel_mask')
external int accelMask(int engine);

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dinf_resolve_json',
)
external ffi.Pointer<ffi.Char> resolveJson(ffi.Pointer<ffi.Char> requestJson);

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dinf_fallback_json',
)
external ffi.Pointer<ffi.Char> fallbackJson(ffi.Pointer<ffi.Char> requestJson);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Char>)
>(symbol: 'dinf_artifact_path')
external ffi.Pointer<ffi.Char> artifactPath(
  ffi.Pointer<ffi.Char> rootPath,
  ffi.Pointer<ffi.Char> artifactPath,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(symbol: 'dinf_mem_json')
external ffi.Pointer<ffi.Char> memJson();

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
  )
>(symbol: 'dinf_ort_libs_json')
external ffi.Pointer<ffi.Char> ortLibsJson(
  ffi.Pointer<ffi.Char> runtimeEnvFile,
  ffi.Pointer<ffi.Char> searchRoots,
  ffi.Pointer<ffi.Char> explicitLibraries,
  ffi.Pointer<ffi.Char> libraryDirs,
  ffi.Pointer<ffi.Char> libraryNames,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dinf_coreml_layout_json',
)
external ffi.Pointer<ffi.Char> coremlLayoutJson(ffi.Pointer<ffi.Char> rootPath);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
  )
>(symbol: 'dinf_hf_ref_json')
external ffi.Pointer<ffi.Char> hfRefJson(
  ffi.Pointer<ffi.Char> sourceUri,
  ffi.Pointer<ffi.Char> artifactPath,
  ffi.Pointer<ffi.Char> repo,
  ffi.Pointer<ffi.Char> artifact,
  ffi.Pointer<ffi.Char> revision,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(symbol: 'dinf_hf_cache_root')
external ffi.Pointer<ffi.Char> hfCacheRoot();

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(symbol: 'dinf_hf_token')
external ffi.Pointer<ffi.Char> hfToken();

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
  )
>(symbol: 'dinf_hf_cache_path')
external ffi.Pointer<ffi.Char> hfCachePath(
  ffi.Pointer<ffi.Char> cacheRoot,
  ffi.Pointer<ffi.Char> repo,
  ffi.Pointer<ffi.Char> revision,
  ffi.Pointer<ffi.Char> artifactPath,
);

@ffi.Native<ffi.Int32 Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dinf_hf_dir_artifact',
)
external int hfDirArtifact(ffi.Pointer<ffi.Char> artifactPath);

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'dinf_diag_json',
)
external ffi.Pointer<ffi.Char> diagJson(ffi.Pointer<ffi.Void> session);

@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.IntPtr)>(symbol: 'dinf_alloc')
external ffi.Pointer<ffi.Void> alloc(int byteLength);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Int32,
    ffi.Pointer<ffi.Int64>,
    ffi.Int32,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_alloc_tensor')
external ffi.Pointer<ffi.Void> allocTensor(
  int dtype,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  ffi.Pointer<ffi.IntPtr> byteLength,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_free_buf')
external void freeBuf(ffi.Pointer<ffi.Void> value);
