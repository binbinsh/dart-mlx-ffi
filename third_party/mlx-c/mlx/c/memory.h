/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_MEMORY_H
#define MLX_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup memory Memory operations
 */
/**@{*/

int mlx_clear_cache(void);
int mlx_get_active_memory(size_t* res);
int mlx_get_cache_memory(size_t* res);
int mlx_get_cache_count(size_t* res);
int mlx_get_cache_limit(size_t* res);
int mlx_get_memory_limit(size_t* res);
int mlx_get_peak_memory(size_t* res);
int mlx_get_resource_count(size_t* res);
int mlx_get_resource_limit(size_t* res);
int mlx_get_wired_limit(size_t* res);
int mlx_get_command_buffer_commit_count(size_t* res);
int mlx_get_pending_output_count(size_t* res);
int mlx_get_temporary_count(size_t* res);
int mlx_get_buffer_op_count(size_t* res);
int mlx_get_buffer_size_bytes(size_t* res);
int mlx_get_stream_count(size_t* res);
int mlx_get_set_data_count(size_t* res);
int mlx_get_shared_buffer_copy_count(size_t* res);
int mlx_get_allocation_request_count(size_t* res);
int mlx_get_cache_reuse_hit_count(size_t* res);
int mlx_get_new_allocation_count(size_t* res);
int mlx_get_heap_allocation_count(size_t* res);
int mlx_get_device_allocation_count(size_t* res);
int mlx_get_common_binary_allocation_count(size_t* res);
int mlx_get_common_binary_shared_copy_count(size_t* res);
int mlx_get_common_unary_allocation_count(size_t* res);
int mlx_get_common_unary_shared_copy_count(size_t* res);
int mlx_get_common_copy_allocation_count(size_t* res);
int mlx_get_common_copy_shared_copy_count(size_t* res);
int mlx_get_common_copy_scalar_allocation_count(size_t* res);
int mlx_get_common_copy_scalar_shared_copy_count(size_t* res);
int mlx_get_common_copy_vector_allocation_count(size_t* res);
int mlx_get_common_copy_vector_shared_copy_count(size_t* res);
int mlx_get_common_copy_general_allocation_count(size_t* res);
int mlx_get_common_copy_general_shared_copy_count(size_t* res);
int mlx_get_common_copy_general_general_allocation_count(size_t* res);
int mlx_get_common_copy_general_general_shared_copy_count(size_t* res);
int mlx_get_common_ternary_allocation_count(size_t* res);
int mlx_get_common_ternary_shared_copy_count(size_t* res);
int mlx_get_gpu_primitive_allocation_count(size_t* res);
int mlx_get_gpu_primitive_shared_copy_count(size_t* res);
int mlx_get_gpu_contiguous_copy_count(size_t* res);
int mlx_get_quantized_contiguous_x_count(size_t* res);
int mlx_get_quantized_contiguous_w_count(size_t* res);
int mlx_get_quantized_contiguous_scales_count(size_t* res);
int mlx_get_quantized_contiguous_biases_count(size_t* res);
int mlx_get_quantized_contiguous_indices_count(size_t* res);
int mlx_get_metal_norm_allocation_count(size_t* res);
int mlx_get_metal_norm_shared_copy_count(size_t* res);
int mlx_get_metal_matmul_allocation_count(size_t* res);
int mlx_get_metal_matmul_shared_copy_count(size_t* res);
int mlx_get_metal_quantized_allocation_count(size_t* res);
int mlx_get_metal_quantized_shared_copy_count(size_t* res);
int mlx_get_metal_sdpa_allocation_count(size_t* res);
int mlx_get_metal_sdpa_shared_copy_count(size_t* res);
int mlx_get_metal_reduce_allocation_count(size_t* res);
int mlx_get_metal_reduce_shared_copy_count(size_t* res);
int mlx_get_metal_indexing_allocation_count(size_t* res);
int mlx_get_metal_indexing_shared_copy_count(size_t* res);
int mlx_get_metal_index_concat_allocation_count(size_t* res);
int mlx_get_metal_index_concat_shared_copy_count(size_t* res);
int mlx_get_metal_index_gather_allocation_count(size_t* res);
int mlx_get_metal_index_gather_shared_copy_count(size_t* res);
int mlx_get_metal_index_gather_axis_allocation_count(size_t* res);
int mlx_get_metal_index_gather_axis_shared_copy_count(size_t* res);
int mlx_get_metal_index_dynamic_offset_allocation_count(size_t* res);
int mlx_get_metal_index_dynamic_offset_shared_copy_count(size_t* res);
int mlx_get_metal_copy_allocation_count(size_t* res);
int mlx_get_metal_copy_shared_copy_count(size_t* res);
int mlx_get_metal_direct_copy_allocation_count(size_t* res);
int mlx_get_metal_direct_copy_shared_copy_count(size_t* res);
int mlx_get_metal_rope_copy_allocation_count(size_t* res);
int mlx_get_metal_rope_copy_shared_copy_count(size_t* res);
int mlx_get_metal_scan_copy_allocation_count(size_t* res);
int mlx_get_metal_scan_copy_shared_copy_count(size_t* res);
int mlx_get_metal_primitive_copy_allocation_count(size_t* res);
int mlx_get_metal_primitive_copy_shared_copy_count(size_t* res);
int mlx_get_metal_reshape_copy_count(size_t* res);
int mlx_get_metal_reshape_shared_count(size_t* res);
int mlx_get_common_copy_gpri_allocation_count(size_t* res);
int mlx_get_common_copy_gpri_shared_copy_count(size_t* res);
int mlx_get_common_copy_gpri_astype_allocation_count(size_t* res);
int mlx_get_common_copy_gpri_astype_shared_copy_count(size_t* res);
int mlx_get_common_copy_gpri_contiguous_allocation_count(size_t* res);
int mlx_get_common_copy_gpri_contiguous_shared_copy_count(size_t* res);
int mlx_get_common_copy_gpri_full_allocation_count(size_t* res);
int mlx_get_common_copy_gpri_full_shared_copy_count(size_t* res);
int mlx_get_common_copy_gpri_slice_update_allocation_count(size_t* res);
int mlx_get_common_copy_gpri_slice_update_shared_copy_count(size_t* res);
int mlx_get_common_copy_gpri_dynamic_slice_update_allocation_count(size_t* res);
int mlx_get_common_copy_gpri_dynamic_slice_update_shared_copy_count(size_t* res);
int mlx_reset_gpu_primitive_trace_budgets(void);
int mlx_get_common_copy_idx_allocation_count(size_t* res);
int mlx_get_common_copy_idx_shared_copy_count(size_t* res);
int mlx_get_common_copy_rope_allocation_count(size_t* res);
int mlx_get_common_copy_rope_shared_copy_count(size_t* res);
int mlx_get_common_copy_matmul_allocation_count(size_t* res);
int mlx_get_common_copy_matmul_shared_copy_count(size_t* res);
int mlx_get_common_copy_hadamard_allocation_count(size_t* res);
int mlx_get_common_copy_hadamard_shared_copy_count(size_t* res);
int mlx_get_donation_reject_not_unique_count(size_t* res);
int mlx_get_donation_reject_desc_not_unique_count(size_t* res);
int mlx_get_donation_reject_data_not_unique_count(size_t* res);
int mlx_get_donation_reject_itemsize_count(size_t* res);
int mlx_get_donation_reject_oversize_count(size_t* res);
int mlx_get_donation_reject_layout_count(size_t* res);
int mlx_get_common_copy_reject_desc_not_unique_count(size_t* res);
int mlx_get_common_copy_reject_data_not_unique_count(size_t* res);
int mlx_get_common_binary_reject_desc_not_unique_count(size_t* res);
int mlx_get_common_binary_reject_data_not_unique_count(size_t* res);
int mlx_get_common_unary_reject_desc_not_unique_count(size_t* res);
int mlx_get_common_unary_reject_data_not_unique_count(size_t* res);
int mlx_get_common_binary_data_not_unique_scalar_vector_count(size_t* res);
int mlx_get_common_binary_data_not_unique_vector_scalar_count(size_t* res);
int mlx_get_common_binary_data_not_unique_vector_vector_count(size_t* res);
int mlx_get_common_binary_data_not_unique_general_count(size_t* res);
int mlx_get_common_binary_add_data_not_unique_vector_vector_count(size_t* res);
int mlx_get_common_binary_add_data_not_unique_general_count(size_t* res);
int mlx_get_common_binary_multiply_data_not_unique_vector_vector_count(size_t* res);
int mlx_get_common_binary_multiply_data_not_unique_general_count(size_t* res);
int mlx_reset_peak_memory(void);
int mlx_set_cache_limit(size_t* res, size_t limit);
int mlx_set_memory_limit(size_t* res, size_t limit);
int mlx_set_wired_limit(size_t* res, size_t limit);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
