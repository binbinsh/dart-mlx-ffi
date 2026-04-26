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

final class ResolveArtifactAbi extends ffi.Struct {
  @ffi.Int32()
  external int engine;

  external ffi.Pointer<ffi.Char> path;

  external ffi.Pointer<ffi.Char> format;

  external ffi.Pointer<ffi.Char> targetPlatforms;
}

final class ResolveResultAbi extends ffi.Struct {
  @ffi.Int32()
  external int engine;

  @ffi.Int32()
  external int accelMask;

  @ffi.Int32()
  external int fallbackEngine;
}

final class InfoAbi extends ffi.Struct {
  external ffi.Pointer<ffi.Char> nativeBackend;

  external ffi.Pointer<ffi.Char> zigVersion;

  external ffi.Pointer<ffi.Char> asyncModel;

  external ffi.Pointer<ffi.Char> abi;

  external ffi.Pointer<ffi.Char> mlxOwner;

  external ffi.Pointer<ffi.Char> mlxApi;

  @ffi.Int32()
  external int mlxLinked;

  @ffi.Int32()
  external int mlxEnabled;

  external ffi.Pointer<ffi.Char> mlxArtifacts;
}

final class MemAbi extends ffi.Struct {
  external ffi.Pointer<ffi.Char> nativeBackend;

  @ffi.Uint64()
  external int peakMemoryBytes;

  @ffi.Uint64()
  external int vmHwm;

  @ffi.Uint64()
  external int vmRss;

  @ffi.Uint64()
  external int physFootprint;

  @ffi.Uint64()
  external int residentSize;

  @ffi.Uint64()
  external int virtualSize;

  @ffi.Uint64()
  external int peakWorkingSet;

  @ffi.Uint64()
  external int workingSet;

  @ffi.Uint64()
  external int androidPeakPss;

  @ffi.Uint64()
  external int androidPss;

  @ffi.Uint64()
  external int androidRss;

  @ffi.Uint64()
  external int androidNativeHeapPss;

  @ffi.Uint64()
  external int androidJavaHeapPss;

  @ffi.Uint64()
  external int androidNativeHeapPrivateDirty;

  @ffi.Uint64()
  external int androidJavaHeapPrivateDirty;
}

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Int32,
    ffi.Pointer<ffi.Char>,
    ffi.Int32,
    ffi.Int32,
    ffi.Int32,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_open_opts')
external ffi.Pointer<ffi.Void> openOpts(
  int engine,
  ffi.Pointer<ffi.Char> modelPath,
  int preferMask,
  int diagnostics,
  int numThreads,
  ffi.Pointer<ffi.Char> metadataJson,
  ffi.Pointer<ffi.Char> backendJson,
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

@ffi.Native<ffi.Int32 Function(ffi.Pointer<InfoAbi>)>(symbol: 'dinf_info')
external int info(ffi.Pointer<InfoAbi> out);

@ffi.Native<ffi.Int32 Function()>(symbol: 'dinf_platform_id')
external int platformId();

@ffi.Native<ffi.Int32 Function(ffi.Int32)>(symbol: 'dinf_accel_mask')
external int accelMask(int engine);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Int32,
    ffi.Int32,
    ffi.Int32,
    ffi.Int32,
    ffi.Pointer<ResolveArtifactAbi>,
    ffi.IntPtr,
    ffi.Pointer<ResolveResultAbi>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_resolve')
external int resolve(
  ffi.Pointer<ffi.Char> modelId,
  int platform,
  int requestedEngine,
  int allowFallback,
  int preferMask,
  ffi.Pointer<ResolveArtifactAbi> artifacts,
  int artifactCount,
  ffi.Pointer<ResolveResultAbi> result,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Int32,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ResolveArtifactAbi>,
    ffi.IntPtr,
  )
>(symbol: 'dinf_fallback')
external int fallback(
  int platform,
  ffi.Pointer<ffi.Int32> registeredEngines,
  int registeredCount,
  ffi.Pointer<ResolveArtifactAbi> artifacts,
  int artifactCount,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>, ffi.Pointer<ffi.Char>)
>(symbol: 'dinf_artifact_path')
external ffi.Pointer<ffi.Char> artifactPath(
  ffi.Pointer<ffi.Char> rootPath,
  ffi.Pointer<ffi.Char> artifactPath,
);

@ffi.Native<ffi.Int32 Function(ffi.Pointer<MemAbi>)>(symbol: 'dinf_mem')
external int mem(ffi.Pointer<MemAbi> out);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
  )
>(symbol: 'dinf_ort_libs')
external ffi.Pointer<ffi.Char> ortLibs(
  ffi.Pointer<ffi.Char> runtimeEnvFile,
  ffi.Pointer<ffi.Char> searchRoots,
  ffi.Pointer<ffi.Char> explicitLibraries,
  ffi.Pointer<ffi.Char> libraryDirs,
  ffi.Pointer<ffi.Char> libraryNames,
);

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dinf_coreml_layout',
)
external ffi.Pointer<ffi.Char> coremlLayout(ffi.Pointer<ffi.Char> rootPath);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
  )
>(symbol: 'dinf_hf_ref')
external ffi.Pointer<ffi.Char> hfRef(
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
