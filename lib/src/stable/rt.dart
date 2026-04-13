part of '../stable_api.dart';

abstract final class MlxRuntime {
  /// Seeds the global random generator.
  static void seed(int seed) {
    _clearError();
    _checkStatus('mlx_random_seed', raw.mlx_random_seed(seed));
  }

  /// Batch-evaluates a list of arrays.
  static void evalAll(List<MlxArray> arrays) {
    final handles = calloc<ffi.Pointer<ffi.Void>>(arrays.length);
    try {
      for (var index = 0; index < arrays.length; index++) {
        handles[index] = arrays[index]._handle;
      }
      _clearError();
      _checkStatus('dart_mlx_eval_many', shim.dart_mlx_eval_many(handles, arrays.length));
    } finally {
      calloc.free(handles);
    }
  }

  /// Schedules asynchronous evaluation for a list of arrays.
  static void asyncEvalAll(List<MlxArray> arrays) {
    final handles = calloc<ffi.Pointer<ffi.Void>>(arrays.length);
    try {
      for (var index = 0; index < arrays.length; index++) {
        handles[index] = arrays[index]._handle;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_async_eval_many',
        shim.dart_mlx_async_eval_many(handles, arrays.length),
      );
    } finally {
      calloc.free(handles);
    }
  }
}

/// High-level memory statistics and tuning.
abstract final class MlxMemory {
  /// Metal/CUDA allocator runtime stats when available.
  static ({
    int activeBytes,
    int cacheBytes,
    int cacheCount,
    int peakBytes,
    int memoryLimitBytes,
    int cacheLimitBytes,
    int wiredLimitBytes,
    int resourceCount,
    int resourceLimit,
    int commandBufferCommitCount,
    int pendingOutputCount,
    int temporaryCount,
    int bufferOpCount,
    int bufferSizeBytes,
    int streamCount,
    int setDataCount,
    int sharedBufferCopyCount,
    int allocationRequestCount,
    int cacheReuseHitCount,
    int newAllocationCount,
    int heapAllocationCount,
    int deviceAllocationCount,
    int commonBinaryAllocationCount,
    int commonBinarySharedCopyCount,
    int commonUnaryAllocationCount,
    int commonUnarySharedCopyCount,
    int commonCopyAllocationCount,
    int commonCopySharedCopyCount,
    int commonCopyScalarAllocationCount,
    int commonCopyScalarSharedCopyCount,
    int commonCopyVectorAllocationCount,
    int commonCopyVectorSharedCopyCount,
    int commonCopyGeneralAllocationCount,
    int commonCopyGeneralSharedCopyCount,
    int commonCopyGeneralGeneralAllocationCount,
    int commonCopyGeneralGeneralSharedCopyCount,
    int commonCopyGpriAllocationCount,
    int commonCopyGpriSharedCopyCount,
    int commonCopyGpriAstypeAllocationCount,
    int commonCopyGpriAstypeSharedCopyCount,
    int commonCopyGpriContiguousAllocationCount,
    int commonCopyGpriContiguousSharedCopyCount,
    int commonCopyGpriFullAllocationCount,
    int commonCopyGpriFullSharedCopyCount,
    int commonCopyGpriSliceUpdateAllocationCount,
    int commonCopyGpriSliceUpdateSharedCopyCount,
    int commonCopyGpriDynamicSliceUpdateAllocationCount,
    int commonCopyGpriDynamicSliceUpdateSharedCopyCount,
    int commonCopyIdxAllocationCount,
    int commonCopyIdxSharedCopyCount,
    int commonCopyRopeAllocationCount,
    int commonCopyRopeSharedCopyCount,
    int commonCopyMatmulAllocationCount,
    int commonCopyMatmulSharedCopyCount,
    int commonCopyHadamardAllocationCount,
    int commonCopyHadamardSharedCopyCount,
    int commonTernaryAllocationCount,
    int commonTernarySharedCopyCount,
    int gpuPrimitiveAllocationCount,
    int gpuPrimitiveSharedCopyCount,
    int gpuContiguousCopyCount,
    int quantizedContiguousXCount,
    int quantizedContiguousWCount,
    int quantizedContiguousScalesCount,
    int quantizedContiguousBiasesCount,
    int quantizedContiguousIndicesCount,
    int metalNormAllocationCount,
    int metalNormSharedCopyCount,
    int metalMatmulAllocationCount,
    int metalMatmulSharedCopyCount,
    int metalQuantizedAllocationCount,
    int metalQuantizedSharedCopyCount,
    int metalSdpaAllocationCount,
    int metalSdpaSharedCopyCount,
    int metalReduceAllocationCount,
    int metalReduceSharedCopyCount,
    int metalIndexingAllocationCount,
    int metalIndexingSharedCopyCount,
    int metalIndexConcatAllocationCount,
    int metalIndexConcatSharedCopyCount,
    int metalIndexGatherAllocationCount,
    int metalIndexGatherSharedCopyCount,
    int metalIndexGatherAxisAllocationCount,
    int metalIndexGatherAxisSharedCopyCount,
    int metalIndexDynamicOffsetAllocationCount,
    int metalIndexDynamicOffsetSharedCopyCount,
    int metalCopyAllocationCount,
    int metalCopySharedCopyCount,
    int metalDirectCopyAllocationCount,
    int metalDirectCopySharedCopyCount,
    int metalRopeCopyAllocationCount,
    int metalRopeCopySharedCopyCount,
    int metalScanCopyAllocationCount,
    int metalScanCopySharedCopyCount,
    int metalPrimitiveCopyAllocationCount,
    int metalPrimitiveCopySharedCopyCount,
    int metalReshapeCopyCount,
    int metalReshapeSharedCount,
    int donationRejectNotUniqueCount,
    int donationRejectDescNotUniqueCount,
    int donationRejectDataNotUniqueCount,
    int donationRejectItemsizeCount,
    int donationRejectOversizeCount,
    int donationRejectLayoutCount,
    int commonCopyRejectDescNotUniqueCount,
    int commonCopyRejectDataNotUniqueCount,
    int commonBinaryRejectDescNotUniqueCount,
    int commonBinaryRejectDataNotUniqueCount,
    int commonUnaryRejectDescNotUniqueCount,
    int commonUnaryRejectDataNotUniqueCount,
    int commonBinaryDataNotUniqueScalarVectorCount,
    int commonBinaryDataNotUniqueVectorScalarCount,
    int commonBinaryDataNotUniqueVectorVectorCount,
    int commonBinaryDataNotUniqueGeneralCount,
    int commonBinaryAddDataNotUniqueVectorVectorCount,
    int commonBinaryAddDataNotUniqueGeneralCount,
    int commonBinaryMultiplyDataNotUniqueVectorVectorCount,
    int commonBinaryMultiplyDataNotUniqueGeneralCount,
  })
  allocatorStats() => (
    activeBytes: activeBytes(),
    cacheBytes: cacheBytes(),
    cacheCount: cacheCount(),
    peakBytes: peakBytes(),
    memoryLimitBytes: memoryLimitBytes(),
    cacheLimitBytes: cacheLimitBytes(),
    wiredLimitBytes: wiredLimitBytes(),
    resourceCount: resourceCount(),
    resourceLimit: resourceLimit(),
    commandBufferCommitCount: commandBufferCommitCount(),
    pendingOutputCount: pendingOutputCount(),
    temporaryCount: temporaryCount(),
    bufferOpCount: bufferOpCount(),
    bufferSizeBytes: bufferSizeBytes(),
    streamCount: streamCount(),
    setDataCount: setDataCount(),
    sharedBufferCopyCount: sharedBufferCopyCount(),
    allocationRequestCount: allocationRequestCount(),
    cacheReuseHitCount: cacheReuseHitCount(),
    newAllocationCount: newAllocationCount(),
    heapAllocationCount: heapAllocationCount(),
    deviceAllocationCount: deviceAllocationCount(),
    commonBinaryAllocationCount: commonBinaryAllocationCount(),
    commonBinarySharedCopyCount: commonBinarySharedCopyCount(),
    commonUnaryAllocationCount: commonUnaryAllocationCount(),
    commonUnarySharedCopyCount: commonUnarySharedCopyCount(),
    commonCopyAllocationCount: commonCopyAllocationCount(),
    commonCopySharedCopyCount: commonCopySharedCopyCount(),
    commonCopyScalarAllocationCount: commonCopyScalarAllocationCount(),
    commonCopyScalarSharedCopyCount: commonCopyScalarSharedCopyCount(),
    commonCopyVectorAllocationCount: commonCopyVectorAllocationCount(),
    commonCopyVectorSharedCopyCount: commonCopyVectorSharedCopyCount(),
    commonCopyGeneralAllocationCount: commonCopyGeneralAllocationCount(),
    commonCopyGeneralSharedCopyCount: commonCopyGeneralSharedCopyCount(),
    commonCopyGeneralGeneralAllocationCount:
        commonCopyGeneralGeneralAllocationCount(),
    commonCopyGeneralGeneralSharedCopyCount:
        commonCopyGeneralGeneralSharedCopyCount(),
    commonCopyGpriAllocationCount: commonCopyGpriAllocationCount(),
    commonCopyGpriSharedCopyCount: commonCopyGpriSharedCopyCount(),
    commonCopyGpriAstypeAllocationCount: commonCopyGpriAstypeAllocationCount(),
    commonCopyGpriAstypeSharedCopyCount: commonCopyGpriAstypeSharedCopyCount(),
    commonCopyGpriContiguousAllocationCount:
        commonCopyGpriContiguousAllocationCount(),
    commonCopyGpriContiguousSharedCopyCount:
        commonCopyGpriContiguousSharedCopyCount(),
    commonCopyGpriFullAllocationCount: commonCopyGpriFullAllocationCount(),
    commonCopyGpriFullSharedCopyCount: commonCopyGpriFullSharedCopyCount(),
    commonCopyGpriSliceUpdateAllocationCount:
        commonCopyGpriSliceUpdateAllocationCount(),
    commonCopyGpriSliceUpdateSharedCopyCount:
        commonCopyGpriSliceUpdateSharedCopyCount(),
    commonCopyGpriDynamicSliceUpdateAllocationCount:
        commonCopyGpriDynamicSliceUpdateAllocationCount(),
    commonCopyGpriDynamicSliceUpdateSharedCopyCount:
        commonCopyGpriDynamicSliceUpdateSharedCopyCount(),
    commonCopyIdxAllocationCount: commonCopyIdxAllocationCount(),
    commonCopyIdxSharedCopyCount: commonCopyIdxSharedCopyCount(),
    commonCopyRopeAllocationCount: commonCopyRopeAllocationCount(),
    commonCopyRopeSharedCopyCount: commonCopyRopeSharedCopyCount(),
    commonCopyMatmulAllocationCount: commonCopyMatmulAllocationCount(),
    commonCopyMatmulSharedCopyCount: commonCopyMatmulSharedCopyCount(),
    commonCopyHadamardAllocationCount: commonCopyHadamardAllocationCount(),
    commonCopyHadamardSharedCopyCount: commonCopyHadamardSharedCopyCount(),
    commonTernaryAllocationCount: commonTernaryAllocationCount(),
    commonTernarySharedCopyCount: commonTernarySharedCopyCount(),
    gpuPrimitiveAllocationCount: gpuPrimitiveAllocationCount(),
    gpuPrimitiveSharedCopyCount: gpuPrimitiveSharedCopyCount(),
    gpuContiguousCopyCount: gpuContiguousCopyCount(),
    quantizedContiguousXCount: quantizedContiguousXCount(),
    quantizedContiguousWCount: quantizedContiguousWCount(),
    quantizedContiguousScalesCount: quantizedContiguousScalesCount(),
    quantizedContiguousBiasesCount: quantizedContiguousBiasesCount(),
    quantizedContiguousIndicesCount: quantizedContiguousIndicesCount(),
    metalNormAllocationCount: metalNormAllocationCount(),
    metalNormSharedCopyCount: metalNormSharedCopyCount(),
    metalMatmulAllocationCount: metalMatmulAllocationCount(),
    metalMatmulSharedCopyCount: metalMatmulSharedCopyCount(),
    metalQuantizedAllocationCount: metalQuantizedAllocationCount(),
    metalQuantizedSharedCopyCount: metalQuantizedSharedCopyCount(),
    metalSdpaAllocationCount: metalSdpaAllocationCount(),
    metalSdpaSharedCopyCount: metalSdpaSharedCopyCount(),
    metalReduceAllocationCount: metalReduceAllocationCount(),
    metalReduceSharedCopyCount: metalReduceSharedCopyCount(),
    metalIndexingAllocationCount: metalIndexingAllocationCount(),
    metalIndexingSharedCopyCount: metalIndexingSharedCopyCount(),
    metalIndexConcatAllocationCount: metalIndexConcatAllocationCount(),
    metalIndexConcatSharedCopyCount: metalIndexConcatSharedCopyCount(),
    metalIndexGatherAllocationCount: metalIndexGatherAllocationCount(),
    metalIndexGatherSharedCopyCount: metalIndexGatherSharedCopyCount(),
    metalIndexGatherAxisAllocationCount: metalIndexGatherAxisAllocationCount(),
    metalIndexGatherAxisSharedCopyCount: metalIndexGatherAxisSharedCopyCount(),
    metalIndexDynamicOffsetAllocationCount:
        metalIndexDynamicOffsetAllocationCount(),
    metalIndexDynamicOffsetSharedCopyCount:
        metalIndexDynamicOffsetSharedCopyCount(),
    metalCopyAllocationCount: metalCopyAllocationCount(),
    metalCopySharedCopyCount: metalCopySharedCopyCount(),
    metalDirectCopyAllocationCount: metalDirectCopyAllocationCount(),
    metalDirectCopySharedCopyCount: metalDirectCopySharedCopyCount(),
    metalRopeCopyAllocationCount: metalRopeCopyAllocationCount(),
    metalRopeCopySharedCopyCount: metalRopeCopySharedCopyCount(),
    metalScanCopyAllocationCount: metalScanCopyAllocationCount(),
    metalScanCopySharedCopyCount: metalScanCopySharedCopyCount(),
    metalPrimitiveCopyAllocationCount: metalPrimitiveCopyAllocationCount(),
    metalPrimitiveCopySharedCopyCount: metalPrimitiveCopySharedCopyCount(),
    metalReshapeCopyCount: metalReshapeCopyCount(),
    metalReshapeSharedCount: metalReshapeSharedCount(),
    donationRejectNotUniqueCount: donationRejectNotUniqueCount(),
    donationRejectDescNotUniqueCount: donationRejectDescNotUniqueCount(),
    donationRejectDataNotUniqueCount: donationRejectDataNotUniqueCount(),
    donationRejectItemsizeCount: donationRejectItemsizeCount(),
    donationRejectOversizeCount: donationRejectOversizeCount(),
    donationRejectLayoutCount: donationRejectLayoutCount(),
    commonCopyRejectDescNotUniqueCount: commonCopyRejectDescNotUniqueCount(),
    commonCopyRejectDataNotUniqueCount: commonCopyRejectDataNotUniqueCount(),
    commonBinaryRejectDescNotUniqueCount:
        commonBinaryRejectDescNotUniqueCount(),
    commonBinaryRejectDataNotUniqueCount:
        commonBinaryRejectDataNotUniqueCount(),
    commonUnaryRejectDescNotUniqueCount:
        commonUnaryRejectDescNotUniqueCount(),
    commonUnaryRejectDataNotUniqueCount:
        commonUnaryRejectDataNotUniqueCount(),
    commonBinaryDataNotUniqueScalarVectorCount:
        commonBinaryDataNotUniqueScalarVectorCount(),
    commonBinaryDataNotUniqueVectorScalarCount:
        commonBinaryDataNotUniqueVectorScalarCount(),
    commonBinaryDataNotUniqueVectorVectorCount:
        commonBinaryDataNotUniqueVectorVectorCount(),
    commonBinaryDataNotUniqueGeneralCount:
        commonBinaryDataNotUniqueGeneralCount(),
    commonBinaryAddDataNotUniqueVectorVectorCount:
        commonBinaryAddDataNotUniqueVectorVectorCount(),
    commonBinaryAddDataNotUniqueGeneralCount:
        commonBinaryAddDataNotUniqueGeneralCount(),
    commonBinaryMultiplyDataNotUniqueVectorVectorCount:
        commonBinaryMultiplyDataNotUniqueVectorVectorCount(),
    commonBinaryMultiplyDataNotUniqueGeneralCount:
        commonBinaryMultiplyDataNotUniqueGeneralCount(),
  );

  /// Currently active memory in bytes.
  static int activeBytes() => _readSizeValue('mlx_get_active_memory', raw.mlx_get_active_memory);

  /// Cached memory in bytes.
  static int cacheBytes() => _readSizeValue('mlx_get_cache_memory', raw.mlx_get_cache_memory);

  /// Number of cached buffers retained by the allocator.
  static int cacheCount() =>
      _readSizeValue('mlx_get_cache_count', raw.mlx_get_cache_count);

  /// Configured cache limit in bytes.
  static int cacheLimitBytes() =>
      _readSizeValue('mlx_get_cache_limit', raw.mlx_get_cache_limit);

  /// Peak memory in bytes.
  static int peakBytes() => _readSizeValue('mlx_get_peak_memory', raw.mlx_get_peak_memory);

  /// Configured memory limit in bytes.
  static int memoryLimitBytes() =>
      _readSizeValue('mlx_get_memory_limit', raw.mlx_get_memory_limit);

  /// Configured wired limit in bytes.
  static int wiredLimitBytes() =>
      _readSizeValue('mlx_get_wired_limit', raw.mlx_get_wired_limit);

  /// Current allocator-managed resource count.
  static int resourceCount() =>
      _readSizeValue('mlx_get_resource_count', raw.mlx_get_resource_count);

  /// Allocator resource limit.
  static int resourceLimit() =>
      _readSizeValue('mlx_get_resource_limit', raw.mlx_get_resource_limit);

  /// Number of committed command buffers since startup.
  static int commandBufferCommitCount() => _readSizeValue(
    'mlx_get_command_buffer_commit_count',
    raw.mlx_get_command_buffer_commit_count,
  );

  /// Number of pending output->fence entries across streams.
  static int pendingOutputCount() =>
      _readSizeValue('mlx_get_pending_output_count', raw.mlx_get_pending_output_count);

  /// Number of temporary arrays retained across streams.
  static int temporaryCount() =>
      _readSizeValue('mlx_get_temporary_count', raw.mlx_get_temporary_count);

  /// Current aggregate buffer op count across streams.
  static int bufferOpCount() =>
      _readSizeValue('mlx_get_buffer_op_count', raw.mlx_get_buffer_op_count);

  /// Current aggregate buffer size in bytes across streams.
  static int bufferSizeBytes() =>
      _readSizeValue('mlx_get_buffer_size_bytes', raw.mlx_get_buffer_size_bytes);

  /// Number of active streams.
  static int streamCount() =>
      _readSizeValue('mlx_get_stream_count', raw.mlx_get_stream_count);

  /// Number of array::set_data(...) calls since startup.
  static int setDataCount() =>
      _readSizeValue('mlx_get_set_data_count', raw.mlx_get_set_data_count);

  /// Number of array::copy_shared_buffer(...) calls since startup.
  static int sharedBufferCopyCount() => _readSizeValue(
    'mlx_get_shared_buffer_copy_count',
    raw.mlx_get_shared_buffer_copy_count,
  );

  /// Number of allocator allocation requests since startup.
  static int allocationRequestCount() => _readSizeValue(
    'mlx_get_allocation_request_count',
    raw.mlx_get_allocation_request_count,
  );

  /// Number of allocator cache reuse hits since startup.
  static int cacheReuseHitCount() => _readSizeValue(
    'mlx_get_cache_reuse_hit_count',
    raw.mlx_get_cache_reuse_hit_count,
  );

  /// Number of fresh allocator allocations since startup.
  static int newAllocationCount() => _readSizeValue(
    'mlx_get_new_allocation_count',
    raw.mlx_get_new_allocation_count,
  );

  /// Number of heap-backed fresh allocations since startup.
  static int heapAllocationCount() => _readSizeValue(
    'mlx_get_heap_allocation_count',
    raw.mlx_get_heap_allocation_count,
  );

  /// Number of standalone device-buffer fresh allocations since startup.
  static int deviceAllocationCount() => _readSizeValue(
    'mlx_get_device_allocation_count',
    raw.mlx_get_device_allocation_count,
  );

  static int commonBinaryAllocationCount() => _readSizeValue(
    'mlx_get_common_binary_allocation_count',
    raw.mlx_get_common_binary_allocation_count,
  );

  static int commonBinarySharedCopyCount() => _readSizeValue(
    'mlx_get_common_binary_shared_copy_count',
    raw.mlx_get_common_binary_shared_copy_count,
  );

  static int commonUnaryAllocationCount() => _readSizeValue(
    'mlx_get_common_unary_allocation_count',
    raw.mlx_get_common_unary_allocation_count,
  );

  static int commonUnarySharedCopyCount() => _readSizeValue(
    'mlx_get_common_unary_shared_copy_count',
    raw.mlx_get_common_unary_shared_copy_count,
  );

  static int commonCopyAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_allocation_count',
    raw.mlx_get_common_copy_allocation_count,
  );

  static int commonCopySharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_shared_copy_count',
    raw.mlx_get_common_copy_shared_copy_count,
  );

  static int commonCopyScalarAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_scalar_allocation_count',
    raw.mlx_get_common_copy_scalar_allocation_count,
  );

  static int commonCopyScalarSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_scalar_shared_copy_count',
    raw.mlx_get_common_copy_scalar_shared_copy_count,
  );

  static int commonCopyVectorAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_vector_allocation_count',
    raw.mlx_get_common_copy_vector_allocation_count,
  );

  static int commonCopyVectorSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_vector_shared_copy_count',
    raw.mlx_get_common_copy_vector_shared_copy_count,
  );

  static int commonCopyGeneralAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_general_allocation_count',
    raw.mlx_get_common_copy_general_allocation_count,
  );

  static int commonCopyGeneralSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_general_shared_copy_count',
    raw.mlx_get_common_copy_general_shared_copy_count,
  );

  static int commonCopyGeneralGeneralAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_general_general_allocation_count',
    raw.mlx_get_common_copy_general_general_allocation_count,
  );

  static int commonCopyGeneralGeneralSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_general_general_shared_copy_count',
    raw.mlx_get_common_copy_general_general_shared_copy_count,
  );

  static int commonCopyGpriAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_allocation_count',
    raw.mlx_get_common_copy_gpri_allocation_count,
  );

  static int commonCopyGpriSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_shared_copy_count',
    raw.mlx_get_common_copy_gpri_shared_copy_count,
  );

  static int commonCopyGpriAstypeAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_astype_allocation_count',
    raw.mlx_get_common_copy_gpri_astype_allocation_count,
  );

  static int commonCopyGpriAstypeSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_astype_shared_copy_count',
    raw.mlx_get_common_copy_gpri_astype_shared_copy_count,
  );

  static int commonCopyGpriContiguousAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_contiguous_allocation_count',
    raw.mlx_get_common_copy_gpri_contiguous_allocation_count,
  );

  static int commonCopyGpriContiguousSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_contiguous_shared_copy_count',
    raw.mlx_get_common_copy_gpri_contiguous_shared_copy_count,
  );

  static int commonCopyGpriFullAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_full_allocation_count',
    raw.mlx_get_common_copy_gpri_full_allocation_count,
  );

  static int commonCopyGpriFullSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_full_shared_copy_count',
    raw.mlx_get_common_copy_gpri_full_shared_copy_count,
  );

  static int commonCopyGpriSliceUpdateAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_slice_update_allocation_count',
    raw.mlx_get_common_copy_gpri_slice_update_allocation_count,
  );

  static int commonCopyGpriSliceUpdateSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_slice_update_shared_copy_count',
    raw.mlx_get_common_copy_gpri_slice_update_shared_copy_count,
  );

  static int commonCopyGpriDynamicSliceUpdateAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_dynamic_slice_update_allocation_count',
    raw.mlx_get_common_copy_gpri_dynamic_slice_update_allocation_count,
  );

  static int commonCopyGpriDynamicSliceUpdateSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_gpri_dynamic_slice_update_shared_copy_count',
    raw.mlx_get_common_copy_gpri_dynamic_slice_update_shared_copy_count,
  );

  static void resetGpuPrimitiveTraceBudgets() {
    _clearError();
    _checkStatus(
      'mlx_reset_gpu_primitive_trace_budgets',
      raw.mlx_reset_gpu_primitive_trace_budgets(),
    );
  }

  static int commonCopyIdxAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_idx_allocation_count',
    raw.mlx_get_common_copy_idx_allocation_count,
  );

  static int commonCopyIdxSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_idx_shared_copy_count',
    raw.mlx_get_common_copy_idx_shared_copy_count,
  );

  static int commonCopyRopeAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_rope_allocation_count',
    raw.mlx_get_common_copy_rope_allocation_count,
  );

  static int commonCopyRopeSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_rope_shared_copy_count',
    raw.mlx_get_common_copy_rope_shared_copy_count,
  );

  static int commonCopyMatmulAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_matmul_allocation_count',
    raw.mlx_get_common_copy_matmul_allocation_count,
  );

  static int commonCopyMatmulSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_matmul_shared_copy_count',
    raw.mlx_get_common_copy_matmul_shared_copy_count,
  );

  static int commonCopyHadamardAllocationCount() => _readSizeValue(
    'mlx_get_common_copy_hadamard_allocation_count',
    raw.mlx_get_common_copy_hadamard_allocation_count,
  );

  static int commonCopyHadamardSharedCopyCount() => _readSizeValue(
    'mlx_get_common_copy_hadamard_shared_copy_count',
    raw.mlx_get_common_copy_hadamard_shared_copy_count,
  );

  static int commonTernaryAllocationCount() => _readSizeValue(
    'mlx_get_common_ternary_allocation_count',
    raw.mlx_get_common_ternary_allocation_count,
  );

  static int commonTernarySharedCopyCount() => _readSizeValue(
    'mlx_get_common_ternary_shared_copy_count',
    raw.mlx_get_common_ternary_shared_copy_count,
  );

  static int gpuPrimitiveAllocationCount() => _readSizeValue(
    'mlx_get_gpu_primitive_allocation_count',
    raw.mlx_get_gpu_primitive_allocation_count,
  );

  static int gpuPrimitiveSharedCopyCount() => _readSizeValue(
    'mlx_get_gpu_primitive_shared_copy_count',
    raw.mlx_get_gpu_primitive_shared_copy_count,
  );

  static int gpuContiguousCopyCount() => _readSizeValue(
    'mlx_get_gpu_contiguous_copy_count',
    raw.mlx_get_gpu_contiguous_copy_count,
  );

  static int quantizedContiguousXCount() => _readSizeValue(
    'mlx_get_quantized_contiguous_x_count',
    raw.mlx_get_quantized_contiguous_x_count,
  );

  static int quantizedContiguousWCount() => _readSizeValue(
    'mlx_get_quantized_contiguous_w_count',
    raw.mlx_get_quantized_contiguous_w_count,
  );

  static int quantizedContiguousScalesCount() => _readSizeValue(
    'mlx_get_quantized_contiguous_scales_count',
    raw.mlx_get_quantized_contiguous_scales_count,
  );

  static int quantizedContiguousBiasesCount() => _readSizeValue(
    'mlx_get_quantized_contiguous_biases_count',
    raw.mlx_get_quantized_contiguous_biases_count,
  );

  static int quantizedContiguousIndicesCount() => _readSizeValue(
    'mlx_get_quantized_contiguous_indices_count',
    raw.mlx_get_quantized_contiguous_indices_count,
  );

  static int metalNormAllocationCount() => _readSizeValue(
    'mlx_get_metal_norm_allocation_count',
    raw.mlx_get_metal_norm_allocation_count,
  );

  static int metalNormSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_norm_shared_copy_count',
    raw.mlx_get_metal_norm_shared_copy_count,
  );

  static int metalMatmulAllocationCount() => _readSizeValue(
    'mlx_get_metal_matmul_allocation_count',
    raw.mlx_get_metal_matmul_allocation_count,
  );

  static int metalMatmulSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_matmul_shared_copy_count',
    raw.mlx_get_metal_matmul_shared_copy_count,
  );

  static int metalQuantizedAllocationCount() => _readSizeValue(
    'mlx_get_metal_quantized_allocation_count',
    raw.mlx_get_metal_quantized_allocation_count,
  );

  static int metalQuantizedSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_quantized_shared_copy_count',
    raw.mlx_get_metal_quantized_shared_copy_count,
  );

  static int metalSdpaAllocationCount() => _readSizeValue(
    'mlx_get_metal_sdpa_allocation_count',
    raw.mlx_get_metal_sdpa_allocation_count,
  );

  static int metalSdpaSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_sdpa_shared_copy_count',
    raw.mlx_get_metal_sdpa_shared_copy_count,
  );

  static int metalReduceAllocationCount() => _readSizeValue(
    'mlx_get_metal_reduce_allocation_count',
    raw.mlx_get_metal_reduce_allocation_count,
  );

  static int metalReduceSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_reduce_shared_copy_count',
    raw.mlx_get_metal_reduce_shared_copy_count,
  );

  static int metalIndexingAllocationCount() => _readSizeValue(
    'mlx_get_metal_indexing_allocation_count',
    raw.mlx_get_metal_indexing_allocation_count,
  );

  static int metalIndexingSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_indexing_shared_copy_count',
    raw.mlx_get_metal_indexing_shared_copy_count,
  );

  static int metalIndexConcatAllocationCount() => _readSizeValue(
    'mlx_get_metal_index_concat_allocation_count',
    raw.mlx_get_metal_index_concat_allocation_count,
  );

  static int metalIndexConcatSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_index_concat_shared_copy_count',
    raw.mlx_get_metal_index_concat_shared_copy_count,
  );

  static int metalIndexGatherAllocationCount() => _readSizeValue(
    'mlx_get_metal_index_gather_allocation_count',
    raw.mlx_get_metal_index_gather_allocation_count,
  );

  static int metalIndexGatherSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_index_gather_shared_copy_count',
    raw.mlx_get_metal_index_gather_shared_copy_count,
  );

  static int metalIndexGatherAxisAllocationCount() => _readSizeValue(
    'mlx_get_metal_index_gather_axis_allocation_count',
    raw.mlx_get_metal_index_gather_axis_allocation_count,
  );

  static int metalIndexGatherAxisSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_index_gather_axis_shared_copy_count',
    raw.mlx_get_metal_index_gather_axis_shared_copy_count,
  );

  static int metalIndexDynamicOffsetAllocationCount() => _readSizeValue(
    'mlx_get_metal_index_dynamic_offset_allocation_count',
    raw.mlx_get_metal_index_dynamic_offset_allocation_count,
  );

  static int metalIndexDynamicOffsetSharedCopyCount() => _readSizeValue(
    'mlx_get_metal_index_dynamic_offset_shared_copy_count',
    raw.mlx_get_metal_index_dynamic_offset_shared_copy_count,
  );

  static int metalCopyAllocationCount() => _readSizeValue(
    'mlx_get_metal_copy_allocation_count',
    raw.mlx_get_metal_copy_allocation_count,
  );

  static int metalCopySharedCopyCount() => _readSizeValue(
    'mlx_get_metal_copy_shared_copy_count',
    raw.mlx_get_metal_copy_shared_copy_count,
  );

  static int metalDirectCopyAllocationCount() => _readSizeValue(
    'mlx_get_metal_direct_copy_allocation_count',
    raw.mlx_get_metal_direct_copy_allocation_count,
  );

  static int metalDirectCopySharedCopyCount() => _readSizeValue(
    'mlx_get_metal_direct_copy_shared_copy_count',
    raw.mlx_get_metal_direct_copy_shared_copy_count,
  );

  static int metalRopeCopyAllocationCount() => _readSizeValue(
    'mlx_get_metal_rope_copy_allocation_count',
    raw.mlx_get_metal_rope_copy_allocation_count,
  );

  static int metalRopeCopySharedCopyCount() => _readSizeValue(
    'mlx_get_metal_rope_copy_shared_copy_count',
    raw.mlx_get_metal_rope_copy_shared_copy_count,
  );

  static int metalScanCopyAllocationCount() => _readSizeValue(
    'mlx_get_metal_scan_copy_allocation_count',
    raw.mlx_get_metal_scan_copy_allocation_count,
  );

  static int metalScanCopySharedCopyCount() => _readSizeValue(
    'mlx_get_metal_scan_copy_shared_copy_count',
    raw.mlx_get_metal_scan_copy_shared_copy_count,
  );

  static int metalPrimitiveCopyAllocationCount() => _readSizeValue(
    'mlx_get_metal_primitive_copy_allocation_count',
    raw.mlx_get_metal_primitive_copy_allocation_count,
  );

  static int metalPrimitiveCopySharedCopyCount() => _readSizeValue(
    'mlx_get_metal_primitive_copy_shared_copy_count',
    raw.mlx_get_metal_primitive_copy_shared_copy_count,
  );

  static int metalReshapeCopyCount() => _readSizeValue(
    'mlx_get_metal_reshape_copy_count',
    raw.mlx_get_metal_reshape_copy_count,
  );

  static int metalReshapeSharedCount() => _readSizeValue(
    'mlx_get_metal_reshape_shared_count',
    raw.mlx_get_metal_reshape_shared_count,
  );

  static int donationRejectNotUniqueCount() => _readSizeValue(
    'mlx_get_donation_reject_not_unique_count',
    raw.mlx_get_donation_reject_not_unique_count,
  );

  static int donationRejectDescNotUniqueCount() => _readSizeValue(
    'mlx_get_donation_reject_desc_not_unique_count',
    raw.mlx_get_donation_reject_desc_not_unique_count,
  );

  static int donationRejectDataNotUniqueCount() => _readSizeValue(
    'mlx_get_donation_reject_data_not_unique_count',
    raw.mlx_get_donation_reject_data_not_unique_count,
  );

  static int donationRejectItemsizeCount() => _readSizeValue(
    'mlx_get_donation_reject_itemsize_count',
    raw.mlx_get_donation_reject_itemsize_count,
  );

  static int donationRejectOversizeCount() => _readSizeValue(
    'mlx_get_donation_reject_oversize_count',
    raw.mlx_get_donation_reject_oversize_count,
  );

  static int donationRejectLayoutCount() => _readSizeValue(
    'mlx_get_donation_reject_layout_count',
    raw.mlx_get_donation_reject_layout_count,
  );

  static int commonCopyRejectDescNotUniqueCount() => _readSizeValue(
    'mlx_get_common_copy_reject_desc_not_unique_count',
    raw.mlx_get_common_copy_reject_desc_not_unique_count,
  );

  static int commonCopyRejectDataNotUniqueCount() => _readSizeValue(
    'mlx_get_common_copy_reject_data_not_unique_count',
    raw.mlx_get_common_copy_reject_data_not_unique_count,
  );

  static int commonBinaryRejectDescNotUniqueCount() => _readSizeValue(
    'mlx_get_common_binary_reject_desc_not_unique_count',
    raw.mlx_get_common_binary_reject_desc_not_unique_count,
  );

  static int commonBinaryRejectDataNotUniqueCount() => _readSizeValue(
    'mlx_get_common_binary_reject_data_not_unique_count',
    raw.mlx_get_common_binary_reject_data_not_unique_count,
  );

  static int commonUnaryRejectDescNotUniqueCount() => _readSizeValue(
    'mlx_get_common_unary_reject_desc_not_unique_count',
    raw.mlx_get_common_unary_reject_desc_not_unique_count,
  );

  static int commonUnaryRejectDataNotUniqueCount() => _readSizeValue(
    'mlx_get_common_unary_reject_data_not_unique_count',
    raw.mlx_get_common_unary_reject_data_not_unique_count,
  );

  static int commonBinaryDataNotUniqueScalarVectorCount() => _readSizeValue(
    'mlx_get_common_binary_data_not_unique_scalar_vector_count',
    raw.mlx_get_common_binary_data_not_unique_scalar_vector_count,
  );

  static int commonBinaryDataNotUniqueVectorScalarCount() => _readSizeValue(
    'mlx_get_common_binary_data_not_unique_vector_scalar_count',
    raw.mlx_get_common_binary_data_not_unique_vector_scalar_count,
  );

  static int commonBinaryDataNotUniqueVectorVectorCount() => _readSizeValue(
    'mlx_get_common_binary_data_not_unique_vector_vector_count',
    raw.mlx_get_common_binary_data_not_unique_vector_vector_count,
  );

  static int commonBinaryDataNotUniqueGeneralCount() => _readSizeValue(
    'mlx_get_common_binary_data_not_unique_general_count',
    raw.mlx_get_common_binary_data_not_unique_general_count,
  );

  static int commonBinaryAddDataNotUniqueVectorVectorCount() => _readSizeValue(
    'mlx_get_common_binary_add_data_not_unique_vector_vector_count',
    raw.mlx_get_common_binary_add_data_not_unique_vector_vector_count,
  );

  static int commonBinaryAddDataNotUniqueGeneralCount() => _readSizeValue(
    'mlx_get_common_binary_add_data_not_unique_general_count',
    raw.mlx_get_common_binary_add_data_not_unique_general_count,
  );

  static int commonBinaryMultiplyDataNotUniqueVectorVectorCount() =>
      _readSizeValue(
        'mlx_get_common_binary_multiply_data_not_unique_vector_vector_count',
        raw.mlx_get_common_binary_multiply_data_not_unique_vector_vector_count,
      );

  static int commonBinaryMultiplyDataNotUniqueGeneralCount() => _readSizeValue(
    'mlx_get_common_binary_multiply_data_not_unique_general_count',
    raw.mlx_get_common_binary_multiply_data_not_unique_general_count,
  );

  /// Sets the cache limit and returns the applied value.
  static int setCacheLimitBytes(int bytes) =>
      _writeSizeValue('mlx_set_cache_limit', raw.mlx_set_cache_limit, bytes);

  /// Sets the memory limit and returns the applied value.
  static int setMemoryLimitBytes(int bytes) =>
      _writeSizeValue('mlx_set_memory_limit', raw.mlx_set_memory_limit, bytes);

  /// Sets the wired memory limit and returns the applied value.
  static int setWiredLimitBytes(int bytes) =>
      _writeSizeValue('mlx_set_wired_limit', raw.mlx_set_wired_limit, bytes);

  /// Clears cached allocator memory.
  static void clearCache() {
    _clearError();
    _checkStatus('mlx_clear_cache', raw.mlx_clear_cache());
  }

  /// Resets the peak memory counter.
  static void resetPeak() {
    _clearError();
    _checkStatus('mlx_reset_peak_memory', raw.mlx_reset_peak_memory());
  }
}

/// High-level Metal runtime helpers.
abstract final class MlxMetal {
  /// Whether the Metal backend is available in the current runtime.
  static bool isAvailable() {
    final result = calloc<ffi.Bool>();
    try {
      _clearError();
      _checkStatus('mlx_metal_is_available', raw.mlx_metal_is_available(result));
      return result.value;
    } finally {
      calloc.free(result);
    }
  }

  /// Starts a Metal capture to the given output path.
  static void startCapture(String path) {
    _withNativePath(path, (nativePath) {
      _clearError();
      _checkStatus('mlx_metal_start_capture', raw.mlx_metal_start_capture(nativePath));
    });
  }

  /// Stops the active Metal capture.
  static void stopCapture() {
    _clearError();
    _checkStatus('mlx_metal_stop_capture', raw.mlx_metal_stop_capture());
  }
}

/// Module-style memory/runtime namespace.
final class MlxMemoryModule {
  const MlxMemoryModule._();

  /// Aggregated allocator/runtime stats.
  ({
    int activeBytes,
    int cacheBytes,
    int cacheCount,
    int peakBytes,
    int memoryLimitBytes,
    int cacheLimitBytes,
    int wiredLimitBytes,
    int resourceCount,
    int resourceLimit,
    int commandBufferCommitCount,
    int pendingOutputCount,
    int temporaryCount,
    int bufferOpCount,
    int bufferSizeBytes,
    int streamCount,
    int setDataCount,
    int sharedBufferCopyCount,
    int allocationRequestCount,
    int cacheReuseHitCount,
    int newAllocationCount,
    int heapAllocationCount,
    int deviceAllocationCount,
    int commonBinaryAllocationCount,
    int commonBinarySharedCopyCount,
    int commonUnaryAllocationCount,
    int commonUnarySharedCopyCount,
    int commonCopyAllocationCount,
    int commonCopySharedCopyCount,
    int commonCopyScalarAllocationCount,
    int commonCopyScalarSharedCopyCount,
    int commonCopyVectorAllocationCount,
    int commonCopyVectorSharedCopyCount,
    int commonCopyGeneralAllocationCount,
    int commonCopyGeneralSharedCopyCount,
    int commonCopyGeneralGeneralAllocationCount,
    int commonCopyGeneralGeneralSharedCopyCount,
    int commonCopyGpriAllocationCount,
    int commonCopyGpriSharedCopyCount,
    int commonCopyGpriAstypeAllocationCount,
    int commonCopyGpriAstypeSharedCopyCount,
    int commonCopyGpriContiguousAllocationCount,
    int commonCopyGpriContiguousSharedCopyCount,
    int commonCopyGpriFullAllocationCount,
    int commonCopyGpriFullSharedCopyCount,
    int commonCopyGpriSliceUpdateAllocationCount,
    int commonCopyGpriSliceUpdateSharedCopyCount,
    int commonCopyGpriDynamicSliceUpdateAllocationCount,
    int commonCopyGpriDynamicSliceUpdateSharedCopyCount,
    int commonCopyIdxAllocationCount,
    int commonCopyIdxSharedCopyCount,
    int commonCopyRopeAllocationCount,
    int commonCopyRopeSharedCopyCount,
    int commonCopyMatmulAllocationCount,
    int commonCopyMatmulSharedCopyCount,
    int commonCopyHadamardAllocationCount,
    int commonCopyHadamardSharedCopyCount,
    int commonTernaryAllocationCount,
    int commonTernarySharedCopyCount,
    int gpuPrimitiveAllocationCount,
    int gpuPrimitiveSharedCopyCount,
    int gpuContiguousCopyCount,
    int quantizedContiguousXCount,
    int quantizedContiguousWCount,
    int quantizedContiguousScalesCount,
    int quantizedContiguousBiasesCount,
    int quantizedContiguousIndicesCount,
    int metalNormAllocationCount,
    int metalNormSharedCopyCount,
    int metalMatmulAllocationCount,
    int metalMatmulSharedCopyCount,
    int metalQuantizedAllocationCount,
    int metalQuantizedSharedCopyCount,
    int metalSdpaAllocationCount,
    int metalSdpaSharedCopyCount,
    int metalReduceAllocationCount,
    int metalReduceSharedCopyCount,
    int metalIndexingAllocationCount,
    int metalIndexingSharedCopyCount,
    int metalIndexConcatAllocationCount,
    int metalIndexConcatSharedCopyCount,
    int metalIndexGatherAllocationCount,
    int metalIndexGatherSharedCopyCount,
    int metalIndexGatherAxisAllocationCount,
    int metalIndexGatherAxisSharedCopyCount,
    int metalIndexDynamicOffsetAllocationCount,
    int metalIndexDynamicOffsetSharedCopyCount,
    int metalCopyAllocationCount,
    int metalCopySharedCopyCount,
    int metalDirectCopyAllocationCount,
    int metalDirectCopySharedCopyCount,
    int metalRopeCopyAllocationCount,
    int metalRopeCopySharedCopyCount,
    int metalScanCopyAllocationCount,
    int metalScanCopySharedCopyCount,
    int metalPrimitiveCopyAllocationCount,
    int metalPrimitiveCopySharedCopyCount,
    int metalReshapeCopyCount,
    int metalReshapeSharedCount,
    int donationRejectNotUniqueCount,
    int donationRejectDescNotUniqueCount,
    int donationRejectDataNotUniqueCount,
    int donationRejectItemsizeCount,
    int donationRejectOversizeCount,
    int donationRejectLayoutCount,
    int commonCopyRejectDescNotUniqueCount,
    int commonCopyRejectDataNotUniqueCount,
    int commonBinaryRejectDescNotUniqueCount,
    int commonBinaryRejectDataNotUniqueCount,
    int commonUnaryRejectDescNotUniqueCount,
    int commonUnaryRejectDataNotUniqueCount,
    int commonBinaryDataNotUniqueScalarVectorCount,
    int commonBinaryDataNotUniqueVectorScalarCount,
    int commonBinaryDataNotUniqueVectorVectorCount,
    int commonBinaryDataNotUniqueGeneralCount,
    int commonBinaryAddDataNotUniqueVectorVectorCount,
    int commonBinaryAddDataNotUniqueGeneralCount,
    int commonBinaryMultiplyDataNotUniqueVectorVectorCount,
    int commonBinaryMultiplyDataNotUniqueGeneralCount,
  })
  allocatorStats() => MlxMemory.allocatorStats();

  /// Currently active memory in bytes.
  int activeBytes() => MlxMemory.activeBytes();

  /// Cached memory in bytes.
  int cacheBytes() => MlxMemory.cacheBytes();

  /// Number of cached buffers retained by the allocator.
  int cacheCount() => MlxMemory.cacheCount();

  /// Peak memory in bytes.
  int peakBytes() => MlxMemory.peakBytes();

  /// Memory limit in bytes.
  int memoryLimitBytes() => MlxMemory.memoryLimitBytes();

  /// Cache limit in bytes.
  int cacheLimitBytes() => MlxMemory.cacheLimitBytes();

  /// Wired limit in bytes.
  int wiredLimitBytes() => MlxMemory.wiredLimitBytes();

  /// Allocator-managed resource count.
  int resourceCount() => MlxMemory.resourceCount();

  /// Allocator resource limit.
  int resourceLimit() => MlxMemory.resourceLimit();

  /// Number of committed command buffers since startup.
  int commandBufferCommitCount() => MlxMemory.commandBufferCommitCount();

  /// Number of pending output->fence entries across streams.
  int pendingOutputCount() => MlxMemory.pendingOutputCount();

  /// Number of temporaries retained across streams.
  int temporaryCount() => MlxMemory.temporaryCount();

  /// Current aggregate buffer op count across streams.
  int bufferOpCount() => MlxMemory.bufferOpCount();

  /// Current aggregate buffer size in bytes across streams.
  int bufferSizeBytes() => MlxMemory.bufferSizeBytes();

  /// Number of active streams.
  int streamCount() => MlxMemory.streamCount();

  /// Number of array::set_data(...) calls since startup.
  int setDataCount() => MlxMemory.setDataCount();

  /// Number of array::copy_shared_buffer(...) calls since startup.
  int sharedBufferCopyCount() => MlxMemory.sharedBufferCopyCount();

  /// Number of allocator allocation requests since startup.
  int allocationRequestCount() => MlxMemory.allocationRequestCount();

  /// Number of allocator cache reuse hits since startup.
  int cacheReuseHitCount() => MlxMemory.cacheReuseHitCount();

  /// Number of fresh allocator allocations since startup.
  int newAllocationCount() => MlxMemory.newAllocationCount();

  /// Number of heap-backed fresh allocations since startup.
  int heapAllocationCount() => MlxMemory.heapAllocationCount();

  /// Number of standalone device-buffer fresh allocations since startup.
  int deviceAllocationCount() => MlxMemory.deviceAllocationCount();

  int commonBinaryAllocationCount() => MlxMemory.commonBinaryAllocationCount();

  int commonBinarySharedCopyCount() => MlxMemory.commonBinarySharedCopyCount();

  int commonUnaryAllocationCount() => MlxMemory.commonUnaryAllocationCount();

  int commonUnarySharedCopyCount() => MlxMemory.commonUnarySharedCopyCount();

  int commonCopyAllocationCount() => MlxMemory.commonCopyAllocationCount();

  int commonCopySharedCopyCount() => MlxMemory.commonCopySharedCopyCount();

  int commonCopyScalarAllocationCount() =>
      MlxMemory.commonCopyScalarAllocationCount();

  int commonCopyScalarSharedCopyCount() =>
      MlxMemory.commonCopyScalarSharedCopyCount();

  int commonCopyVectorAllocationCount() =>
      MlxMemory.commonCopyVectorAllocationCount();

  int commonCopyVectorSharedCopyCount() =>
      MlxMemory.commonCopyVectorSharedCopyCount();

  int commonCopyGeneralAllocationCount() =>
      MlxMemory.commonCopyGeneralAllocationCount();

  int commonCopyGeneralSharedCopyCount() =>
      MlxMemory.commonCopyGeneralSharedCopyCount();

  int commonCopyGeneralGeneralAllocationCount() =>
      MlxMemory.commonCopyGeneralGeneralAllocationCount();

  int commonCopyGeneralGeneralSharedCopyCount() =>
      MlxMemory.commonCopyGeneralGeneralSharedCopyCount();

  int commonCopyGpriAllocationCount() =>
      MlxMemory.commonCopyGpriAllocationCount();

  int commonCopyGpriSharedCopyCount() =>
      MlxMemory.commonCopyGpriSharedCopyCount();

  int commonCopyGpriAstypeAllocationCount() =>
      MlxMemory.commonCopyGpriAstypeAllocationCount();

  int commonCopyGpriAstypeSharedCopyCount() =>
      MlxMemory.commonCopyGpriAstypeSharedCopyCount();

  int commonCopyGpriContiguousAllocationCount() =>
      MlxMemory.commonCopyGpriContiguousAllocationCount();

  int commonCopyGpriContiguousSharedCopyCount() =>
      MlxMemory.commonCopyGpriContiguousSharedCopyCount();

  int commonCopyGpriFullAllocationCount() =>
      MlxMemory.commonCopyGpriFullAllocationCount();

  int commonCopyGpriFullSharedCopyCount() =>
      MlxMemory.commonCopyGpriFullSharedCopyCount();

  int commonCopyGpriSliceUpdateAllocationCount() =>
      MlxMemory.commonCopyGpriSliceUpdateAllocationCount();

  int commonCopyGpriSliceUpdateSharedCopyCount() =>
      MlxMemory.commonCopyGpriSliceUpdateSharedCopyCount();

  int commonCopyGpriDynamicSliceUpdateAllocationCount() =>
      MlxMemory.commonCopyGpriDynamicSliceUpdateAllocationCount();

  int commonCopyGpriDynamicSliceUpdateSharedCopyCount() =>
      MlxMemory.commonCopyGpriDynamicSliceUpdateSharedCopyCount();

  int commonCopyIdxAllocationCount() =>
      MlxMemory.commonCopyIdxAllocationCount();

  int commonCopyIdxSharedCopyCount() =>
      MlxMemory.commonCopyIdxSharedCopyCount();

  int commonCopyRopeAllocationCount() =>
      MlxMemory.commonCopyRopeAllocationCount();

  int commonCopyRopeSharedCopyCount() =>
      MlxMemory.commonCopyRopeSharedCopyCount();

  int commonCopyMatmulAllocationCount() =>
      MlxMemory.commonCopyMatmulAllocationCount();

  int commonCopyMatmulSharedCopyCount() =>
      MlxMemory.commonCopyMatmulSharedCopyCount();

  int commonCopyHadamardAllocationCount() =>
      MlxMemory.commonCopyHadamardAllocationCount();

  int commonCopyHadamardSharedCopyCount() =>
      MlxMemory.commonCopyHadamardSharedCopyCount();

  int commonTernaryAllocationCount() => MlxMemory.commonTernaryAllocationCount();

  int commonTernarySharedCopyCount() => MlxMemory.commonTernarySharedCopyCount();

  int gpuPrimitiveAllocationCount() => MlxMemory.gpuPrimitiveAllocationCount();

  int gpuPrimitiveSharedCopyCount() => MlxMemory.gpuPrimitiveSharedCopyCount();

  int gpuContiguousCopyCount() => MlxMemory.gpuContiguousCopyCount();

  int quantizedContiguousXCount() => MlxMemory.quantizedContiguousXCount();

  int quantizedContiguousWCount() => MlxMemory.quantizedContiguousWCount();

  int quantizedContiguousScalesCount() =>
      MlxMemory.quantizedContiguousScalesCount();

  int quantizedContiguousBiasesCount() =>
      MlxMemory.quantizedContiguousBiasesCount();

  int quantizedContiguousIndicesCount() =>
      MlxMemory.quantizedContiguousIndicesCount();

  int metalNormAllocationCount() => MlxMemory.metalNormAllocationCount();

  int metalNormSharedCopyCount() => MlxMemory.metalNormSharedCopyCount();

  int metalMatmulAllocationCount() => MlxMemory.metalMatmulAllocationCount();

  int metalMatmulSharedCopyCount() => MlxMemory.metalMatmulSharedCopyCount();

  int metalQuantizedAllocationCount() => MlxMemory.metalQuantizedAllocationCount();

  int metalQuantizedSharedCopyCount() => MlxMemory.metalQuantizedSharedCopyCount();

  int metalSdpaAllocationCount() => MlxMemory.metalSdpaAllocationCount();

  int metalSdpaSharedCopyCount() => MlxMemory.metalSdpaSharedCopyCount();

  int metalReduceAllocationCount() => MlxMemory.metalReduceAllocationCount();

  int metalReduceSharedCopyCount() => MlxMemory.metalReduceSharedCopyCount();

  int metalIndexingAllocationCount() =>
      MlxMemory.metalIndexingAllocationCount();

  int metalIndexingSharedCopyCount() =>
      MlxMemory.metalIndexingSharedCopyCount();

  int metalIndexConcatAllocationCount() =>
      MlxMemory.metalIndexConcatAllocationCount();

  int metalIndexConcatSharedCopyCount() =>
      MlxMemory.metalIndexConcatSharedCopyCount();

  int metalIndexGatherAllocationCount() =>
      MlxMemory.metalIndexGatherAllocationCount();

  int metalIndexGatherSharedCopyCount() =>
      MlxMemory.metalIndexGatherSharedCopyCount();

  int metalIndexGatherAxisAllocationCount() =>
      MlxMemory.metalIndexGatherAxisAllocationCount();

  int metalIndexGatherAxisSharedCopyCount() =>
      MlxMemory.metalIndexGatherAxisSharedCopyCount();

  int metalIndexDynamicOffsetAllocationCount() =>
      MlxMemory.metalIndexDynamicOffsetAllocationCount();

  int metalIndexDynamicOffsetSharedCopyCount() =>
      MlxMemory.metalIndexDynamicOffsetSharedCopyCount();

  int metalCopyAllocationCount() => MlxMemory.metalCopyAllocationCount();

  int metalCopySharedCopyCount() => MlxMemory.metalCopySharedCopyCount();

  int metalDirectCopyAllocationCount() =>
      MlxMemory.metalDirectCopyAllocationCount();

  int metalDirectCopySharedCopyCount() =>
      MlxMemory.metalDirectCopySharedCopyCount();

  int metalRopeCopyAllocationCount() => MlxMemory.metalRopeCopyAllocationCount();

  int metalRopeCopySharedCopyCount() => MlxMemory.metalRopeCopySharedCopyCount();

  int metalScanCopyAllocationCount() => MlxMemory.metalScanCopyAllocationCount();

  int metalScanCopySharedCopyCount() => MlxMemory.metalScanCopySharedCopyCount();

  int metalPrimitiveCopyAllocationCount() =>
      MlxMemory.metalPrimitiveCopyAllocationCount();

  int metalPrimitiveCopySharedCopyCount() =>
      MlxMemory.metalPrimitiveCopySharedCopyCount();

  int metalReshapeCopyCount() => MlxMemory.metalReshapeCopyCount();

  int metalReshapeSharedCount() => MlxMemory.metalReshapeSharedCount();

  int donationRejectNotUniqueCount() => MlxMemory.donationRejectNotUniqueCount();

  int donationRejectDescNotUniqueCount() =>
      MlxMemory.donationRejectDescNotUniqueCount();

  int donationRejectDataNotUniqueCount() =>
      MlxMemory.donationRejectDataNotUniqueCount();

  int donationRejectItemsizeCount() => MlxMemory.donationRejectItemsizeCount();

  int donationRejectOversizeCount() => MlxMemory.donationRejectOversizeCount();

  int donationRejectLayoutCount() => MlxMemory.donationRejectLayoutCount();

  int commonCopyRejectDescNotUniqueCount() =>
      MlxMemory.commonCopyRejectDescNotUniqueCount();

  int commonCopyRejectDataNotUniqueCount() =>
      MlxMemory.commonCopyRejectDataNotUniqueCount();

  int commonBinaryRejectDescNotUniqueCount() =>
      MlxMemory.commonBinaryRejectDescNotUniqueCount();

  int commonBinaryRejectDataNotUniqueCount() =>
      MlxMemory.commonBinaryRejectDataNotUniqueCount();

  int commonUnaryRejectDescNotUniqueCount() =>
      MlxMemory.commonUnaryRejectDescNotUniqueCount();

  int commonUnaryRejectDataNotUniqueCount() =>
      MlxMemory.commonUnaryRejectDataNotUniqueCount();

  int commonBinaryDataNotUniqueScalarVectorCount() =>
      MlxMemory.commonBinaryDataNotUniqueScalarVectorCount();

  int commonBinaryDataNotUniqueVectorScalarCount() =>
      MlxMemory.commonBinaryDataNotUniqueVectorScalarCount();

  int commonBinaryDataNotUniqueVectorVectorCount() =>
      MlxMemory.commonBinaryDataNotUniqueVectorVectorCount();

  int commonBinaryDataNotUniqueGeneralCount() =>
      MlxMemory.commonBinaryDataNotUniqueGeneralCount();

  int commonBinaryAddDataNotUniqueVectorVectorCount() =>
      MlxMemory.commonBinaryAddDataNotUniqueVectorVectorCount();

  int commonBinaryAddDataNotUniqueGeneralCount() =>
      MlxMemory.commonBinaryAddDataNotUniqueGeneralCount();

  int commonBinaryMultiplyDataNotUniqueVectorVectorCount() =>
      MlxMemory.commonBinaryMultiplyDataNotUniqueVectorVectorCount();

  int commonBinaryMultiplyDataNotUniqueGeneralCount() =>
      MlxMemory.commonBinaryMultiplyDataNotUniqueGeneralCount();

  /// Sets the cache limit and returns the applied value.
  int setCacheLimitBytes(int bytes) => MlxMemory.setCacheLimitBytes(bytes);

  /// Sets the memory limit and returns the applied value.
  int setMemoryLimitBytes(int bytes) => MlxMemory.setMemoryLimitBytes(bytes);

  /// Sets the wired memory limit and returns the applied value.
  int setWiredLimitBytes(int bytes) => MlxMemory.setWiredLimitBytes(bytes);

  /// Clears cached allocator memory.
  void clearCache() => MlxMemory.clearCache();

  /// Resets the peak memory counter.
  void resetPeak() => MlxMemory.resetPeak();
}

/// Module-style Metal namespace.
final class MlxMetalModule {
  const MlxMetalModule._();

  /// Whether the Metal backend is available.
  bool isAvailable() => MlxMetal.isAvailable();

  /// Starts a Metal capture.
  void startCapture(String path) => MlxMetal.startCapture(path);

  /// Stops the current Metal capture.
  void stopCapture() => MlxMetal.stopCapture();
}

/// MLX runtime version helpers.
abstract final class MlxVersion {
  /// Returns the MLX runtime version string.
  static String current() {
    _clearError();
    final copy = hooks.debugVersionCopyOverride?.call() ?? shim.dart_mlx_version_copy();
    return _copyOwnedString(copy);
  }
}
