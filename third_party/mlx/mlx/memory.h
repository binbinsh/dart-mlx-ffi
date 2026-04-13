// Copyright © 2025 Apple Inc.

#pragma once

#include <cstdlib>

#include "mlx/api.h"

namespace mlx::core {

/* Get the actively used memory in bytes.
 *
 * Note, this will not always match memory use reported by the system because
 * it does not include cached memory buffers.
 * */
MLX_API size_t get_active_memory();

/* Get the peak amount of used memory in bytes.
 *
 * The maximum memory used recorded from the beginning of the program
 * execution or since the last call to reset_peak_memory.
 * */
MLX_API size_t get_peak_memory();

/* Reset the peak memory to zero.
 * */
MLX_API void reset_peak_memory();

/* Get the cache size in bytes.
 *
 * The cache includes memory not currently used that has not been returned
 * to the system allocator.
 * */
MLX_API size_t get_cache_memory();

/* Get the current cache limit in bytes.
 *
 * Returns the current maximum cache pool size. */
MLX_API size_t get_cache_limit();

/* Get the number of cached buffers currently retained by the allocator. */
MLX_API size_t get_cache_count();

/* Set the memory limit.
 * The memory limit is a guideline for the maximum amount of memory to use
 * during graph evaluation. If the memory limit is exceeded and there is no
 * more RAM (including swap when available) allocations will result in an
 * exception.
 *
 * When Metal is available the memory limit defaults to 1.5 times the maximum
 * recommended working set size reported by the device.
 *
 * Returns the previous memory limit.
 * */
MLX_API size_t set_memory_limit(size_t limit);

/* Get the current memory limit. */
MLX_API size_t get_memory_limit();

/* Set the cache limit.
 * If using more than the given limit, free memory will be reclaimed
 * from the cache on the next allocation. To disable the cache,
 * set the limit to 0.
 *
 * The cache limit defaults to the memory limit.
 *
 * Returns the previous cache limit.
 * */
MLX_API size_t set_cache_limit(size_t limit);

/* Clear the memory cache. */
MLX_API void clear_cache();

/* Set the wired size limit.
 *
 * Note, this function is only useful when using the Metal backend with
 * macOS 15.0 or higher.
 *
 * The wired limit is the total size in bytes of memory that will be kept
 * resident. The default value is ``0``.
 *
 * Setting a wired limit larger than system wired limit is an error.
 *
 * Returns the previous wired limit.
 * */
MLX_API size_t set_wired_limit(size_t limit);

/* Get the current wired size limit in bytes. */
MLX_API size_t get_wired_limit();

/* Get the current number of allocator-managed resources.
 *
 * This is primarily useful on Metal for tracking MTLBuffer/heap pressure. */
MLX_API size_t get_resource_count();

/* Get the current allocator resource limit.
 *
 * This is primarily useful on Metal for tracking the iogpu resource limit. */
MLX_API size_t get_resource_limit();

/* Get the number of committed command buffers since startup. */
MLX_API size_t get_command_buffer_commit_count();

/* Get the number of pending output->fence entries across active streams. */
MLX_API size_t get_pending_output_count();

/* Get the number of temporary arrays retained across active streams. */
MLX_API size_t get_temporary_count();

/* Get the total current buffer op count across active streams. */
MLX_API size_t get_buffer_op_count();

/* Get the total current buffer size in bytes across active streams. */
MLX_API size_t get_buffer_size_bytes();

/* Get the number of active Metal/CUDA streams. */
MLX_API size_t get_stream_count();

/* Get the number of array::set_data(...) calls since process startup.
 *
 * This approximates the number of fresh backing-buffer assignments across the
 * runtime. */
MLX_API size_t get_set_data_count();

/* Get the number of array::copy_shared_buffer(...) calls since process startup.
 *
 * This approximates successful backing-buffer reuse/donation across the
 * runtime. */
MLX_API size_t get_shared_buffer_copy_count();

/* Get the number of allocator allocation requests since process startup. */
MLX_API size_t get_allocation_request_count();

/* Get the number of cache reuse hits since process startup. */
MLX_API size_t get_cache_reuse_hit_count();

/* Get the number of fresh allocator allocations since process startup. */
MLX_API size_t get_new_allocation_count();

/* Get the number of fresh heap-backed Metal allocations since startup. */
MLX_API size_t get_heap_allocation_count();

/* Get the number of fresh standalone device-buffer allocations since startup. */
MLX_API size_t get_device_allocation_count();

/* Get the number of output allocations attributed to common binary helpers. */
MLX_API size_t get_common_binary_allocation_count();

/* Get the number of shared-buffer reuses attributed to common binary helpers. */
MLX_API size_t get_common_binary_shared_copy_count();

/* Get the number of output allocations attributed to common unary helpers. */
MLX_API size_t get_common_unary_allocation_count();

/* Get the number of shared-buffer reuses attributed to common unary helpers. */
MLX_API size_t get_common_unary_shared_copy_count();

/* Get the number of output allocations attributed to common copy helpers. */
MLX_API size_t get_common_copy_allocation_count();

/* Get the number of shared-buffer reuses attributed to common copy helpers. */
MLX_API size_t get_common_copy_shared_copy_count();

/* Get common copy allocations/shared-copy broken down by CopyType. */
MLX_API size_t get_common_copy_scalar_allocation_count();
MLX_API size_t get_common_copy_scalar_shared_copy_count();
MLX_API size_t get_common_copy_vector_allocation_count();
MLX_API size_t get_common_copy_vector_shared_copy_count();
MLX_API size_t get_common_copy_general_allocation_count();
MLX_API size_t get_common_copy_general_shared_copy_count();
MLX_API size_t get_common_copy_general_general_allocation_count();
MLX_API size_t get_common_copy_general_general_shared_copy_count();
MLX_API size_t get_common_copy_gpri_allocation_count();
MLX_API size_t get_common_copy_gpri_shared_copy_count();
MLX_API size_t get_common_copy_gpri_astype_allocation_count();
MLX_API size_t get_common_copy_gpri_astype_shared_copy_count();
MLX_API size_t get_common_copy_gpri_contiguous_allocation_count();
MLX_API size_t get_common_copy_gpri_contiguous_shared_copy_count();
MLX_API size_t get_common_copy_gpri_full_allocation_count();
MLX_API size_t get_common_copy_gpri_full_shared_copy_count();
MLX_API size_t get_common_copy_gpri_slice_update_allocation_count();
MLX_API size_t get_common_copy_gpri_slice_update_shared_copy_count();
MLX_API size_t get_common_copy_gpri_dynamic_slice_update_allocation_count();
MLX_API size_t get_common_copy_gpri_dynamic_slice_update_shared_copy_count();
MLX_API void reset_gpu_primitive_trace_budgets();
MLX_API void reset_ops_trace_budgets();
MLX_API size_t get_common_copy_idx_allocation_count();
MLX_API size_t get_common_copy_idx_shared_copy_count();
MLX_API size_t get_common_copy_rope_allocation_count();
MLX_API size_t get_common_copy_rope_shared_copy_count();
MLX_API size_t get_common_copy_matmul_allocation_count();
MLX_API size_t get_common_copy_matmul_shared_copy_count();
MLX_API size_t get_common_copy_hadamard_allocation_count();
MLX_API size_t get_common_copy_hadamard_shared_copy_count();

/* Get the number of output allocations attributed to common ternary helpers. */
MLX_API size_t get_common_ternary_allocation_count();

/* Get the number of shared-buffer reuses attributed to common ternary helpers. */
MLX_API size_t get_common_ternary_shared_copy_count();

/* Get the number of output allocations attributed to GPU primitive helpers. */
MLX_API size_t get_gpu_primitive_allocation_count();

/* Get the number of shared-buffer reuses attributed to GPU primitive helpers. */
MLX_API size_t get_gpu_primitive_shared_copy_count();

/* Get high-level GPU copy helper counts. */
MLX_API size_t get_gpu_contiguous_copy_count();
MLX_API size_t get_quantized_contiguous_x_count();
MLX_API size_t get_quantized_contiguous_w_count();
MLX_API size_t get_quantized_contiguous_scales_count();
MLX_API size_t get_quantized_contiguous_biases_count();
MLX_API size_t get_quantized_contiguous_indices_count();
MLX_API size_t get_metal_reshape_copy_count();
MLX_API size_t get_metal_reshape_shared_count();

/* Get the number of output allocations attributed to Metal normalization ops. */
MLX_API size_t get_metal_norm_allocation_count();

/* Get the number of shared-buffer reuses attributed to Metal normalization ops. */
MLX_API size_t get_metal_norm_shared_copy_count();

/* Get the number of output allocations attributed to Metal matmul ops. */
MLX_API size_t get_metal_matmul_allocation_count();

/* Get the number of shared-buffer reuses attributed to Metal matmul ops. */
MLX_API size_t get_metal_matmul_shared_copy_count();

/* Get the number of output allocations attributed to Metal quantized ops. */
MLX_API size_t get_metal_quantized_allocation_count();

/* Get the number of shared-buffer reuses attributed to Metal quantized ops. */
MLX_API size_t get_metal_quantized_shared_copy_count();

/* Get the number of output allocations attributed to Metal SDPA ops. */
MLX_API size_t get_metal_sdpa_allocation_count();

/* Get the number of shared-buffer reuses attributed to Metal SDPA ops. */
MLX_API size_t get_metal_sdpa_shared_copy_count();

/* Get the number of output allocations attributed to Metal reduce-style ops. */
MLX_API size_t get_metal_reduce_allocation_count();

/* Get the number of shared-buffer reuses attributed to Metal reduce-style ops. */
MLX_API size_t get_metal_reduce_shared_copy_count();

/* Get the number of output allocations attributed to Metal indexing/slicing ops. */
MLX_API size_t get_metal_indexing_allocation_count();

/* Get the number of shared-buffer reuses attributed to Metal indexing/slicing ops. */
MLX_API size_t get_metal_indexing_shared_copy_count();

/* Get Metal indexing/slicing counts broken down by hot helper. */
MLX_API size_t get_metal_index_concat_allocation_count();
MLX_API size_t get_metal_index_concat_shared_copy_count();
MLX_API size_t get_metal_index_gather_allocation_count();
MLX_API size_t get_metal_index_gather_shared_copy_count();
MLX_API size_t get_metal_index_gather_axis_allocation_count();
MLX_API size_t get_metal_index_gather_axis_shared_copy_count();
MLX_API size_t get_metal_index_dynamic_offset_allocation_count();
MLX_API size_t get_metal_index_dynamic_offset_shared_copy_count();

/* Get the number of output allocations attributed to Metal copy/rope/meta ops. */
MLX_API size_t get_metal_copy_allocation_count();

/* Get the number of shared-buffer reuses attributed to Metal copy/rope/meta ops. */
MLX_API size_t get_metal_copy_shared_copy_count();
MLX_API size_t get_metal_direct_copy_allocation_count();
MLX_API size_t get_metal_direct_copy_shared_copy_count();
MLX_API size_t get_metal_rope_copy_allocation_count();
MLX_API size_t get_metal_rope_copy_shared_copy_count();
MLX_API size_t get_metal_scan_copy_allocation_count();
MLX_API size_t get_metal_scan_copy_shared_copy_count();
MLX_API size_t get_metal_primitive_copy_allocation_count();
MLX_API size_t get_metal_primitive_copy_shared_copy_count();

/* Get aggregated donation rejection counts. */
MLX_API size_t get_donation_reject_not_unique_count();
MLX_API size_t get_donation_reject_desc_not_unique_count();
MLX_API size_t get_donation_reject_data_not_unique_count();
MLX_API size_t get_donation_reject_itemsize_count();
MLX_API size_t get_donation_reject_oversize_count();
MLX_API size_t get_donation_reject_layout_count();

MLX_API size_t get_common_copy_reject_desc_not_unique_count();
MLX_API size_t get_common_copy_reject_data_not_unique_count();
MLX_API size_t get_common_binary_reject_desc_not_unique_count();
MLX_API size_t get_common_binary_reject_data_not_unique_count();
MLX_API size_t get_common_unary_reject_desc_not_unique_count();
MLX_API size_t get_common_unary_reject_data_not_unique_count();
MLX_API size_t get_common_binary_data_not_unique_scalar_vector_count();
MLX_API size_t get_common_binary_data_not_unique_vector_scalar_count();
MLX_API size_t get_common_binary_data_not_unique_vector_vector_count();
MLX_API size_t get_common_binary_data_not_unique_general_count();
MLX_API size_t get_common_binary_add_data_not_unique_vector_vector_count();
MLX_API size_t get_common_binary_add_data_not_unique_general_count();
MLX_API size_t get_common_binary_multiply_data_not_unique_vector_vector_count();
MLX_API size_t get_common_binary_multiply_data_not_unique_general_count();

/* Internal recording hooks for runtime allocation classification. */
MLX_API void record_common_binary_allocation();
MLX_API void record_common_binary_shared_copy();
MLX_API void record_common_unary_allocation();
MLX_API void record_common_unary_shared_copy();
MLX_API void record_common_copy_allocation();
MLX_API void record_common_copy_shared_copy();
MLX_API void record_common_copy_scalar_allocation();
MLX_API void record_common_copy_scalar_shared_copy();
MLX_API void record_common_copy_vector_allocation();
MLX_API void record_common_copy_vector_shared_copy();
MLX_API void record_common_copy_general_allocation();
MLX_API void record_common_copy_general_shared_copy();
MLX_API void record_common_copy_general_general_allocation();
MLX_API void record_common_copy_general_general_shared_copy();
MLX_API void record_current_copy_site_allocation();
MLX_API void record_current_copy_site_shared_copy();
MLX_API void record_common_ternary_allocation();
MLX_API void record_common_ternary_shared_copy();
MLX_API void record_gpu_primitive_allocation();
MLX_API void record_gpu_primitive_shared_copy();
MLX_API void record_gpu_contiguous_copy();
MLX_API void record_quantized_contiguous_x();
MLX_API void record_quantized_contiguous_w();
MLX_API void record_quantized_contiguous_scales();
MLX_API void record_quantized_contiguous_biases();
MLX_API void record_quantized_contiguous_indices();
MLX_API void record_metal_norm_allocation();
MLX_API void record_metal_norm_shared_copy();
MLX_API void record_metal_matmul_allocation();
MLX_API void record_metal_matmul_shared_copy();
MLX_API void record_metal_quantized_allocation();
MLX_API void record_metal_quantized_shared_copy();
MLX_API void record_metal_sdpa_allocation();
MLX_API void record_metal_sdpa_shared_copy();
MLX_API void record_metal_reduce_allocation();
MLX_API void record_metal_reduce_shared_copy();
MLX_API void record_metal_indexing_allocation();
MLX_API void record_metal_indexing_shared_copy();
MLX_API void record_metal_index_concat_allocation();
MLX_API void record_metal_index_concat_shared_copy();
MLX_API void record_metal_index_gather_allocation();
MLX_API void record_metal_index_gather_shared_copy();
MLX_API void record_metal_index_gather_axis_allocation();
MLX_API void record_metal_index_gather_axis_shared_copy();
MLX_API void record_metal_index_dynamic_offset_allocation();
MLX_API void record_metal_index_dynamic_offset_shared_copy();
MLX_API void record_metal_copy_allocation();
MLX_API void record_metal_copy_shared_copy();
MLX_API void record_metal_direct_copy_allocation();
MLX_API void record_metal_direct_copy_shared_copy();
MLX_API void record_metal_rope_copy_allocation();
MLX_API void record_metal_rope_copy_shared_copy();
MLX_API void record_metal_scan_copy_allocation();
MLX_API void record_metal_scan_copy_shared_copy();
MLX_API void record_metal_primitive_copy_allocation();
MLX_API void record_metal_primitive_copy_shared_copy();
MLX_API void record_metal_reshape_copy();
MLX_API void record_metal_reshape_shared();
MLX_API void record_donation_reject_not_unique();
MLX_API void record_donation_reject_desc_not_unique();
MLX_API void record_donation_reject_data_not_unique();
MLX_API void record_donation_reject_itemsize();
MLX_API void record_donation_reject_oversize();
MLX_API void record_donation_reject_layout();
MLX_API void record_common_copy_reject_desc_not_unique();
MLX_API void record_common_copy_reject_data_not_unique();
MLX_API void record_common_binary_reject_desc_not_unique();
MLX_API void record_common_binary_reject_data_not_unique();
MLX_API void record_common_unary_reject_desc_not_unique();
MLX_API void record_common_unary_reject_data_not_unique();
MLX_API void record_common_binary_data_not_unique_scalar_vector();
MLX_API void record_common_binary_data_not_unique_vector_scalar();
MLX_API void record_common_binary_data_not_unique_vector_vector();
MLX_API void record_common_binary_data_not_unique_general();
MLX_API void record_common_binary_add_data_not_unique_vector_vector();
MLX_API void record_common_binary_add_data_not_unique_general();
MLX_API void record_common_binary_multiply_data_not_unique_vector_vector();
MLX_API void record_common_binary_multiply_data_not_unique_general();

// Internal helper for per-op binary instrumentation.
MLX_API void set_current_binary_op_name(const char* op_name);
MLX_API const char* current_binary_op_name();
MLX_API void set_current_copy_site_name(const char* site_name);
MLX_API const char* current_copy_site_name();

} // namespace mlx::core
