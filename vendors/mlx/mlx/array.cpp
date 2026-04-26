// Copyright © 2023-2024 Apple Inc.
#include <atomic>
#include <cstring>
#include <functional>
#include <unordered_map>

#include "mlx/array.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

namespace {

std::atomic<size_t> g_set_data_count{0};
std::atomic<size_t> g_shared_buffer_copy_count{0};
std::atomic<size_t> g_common_binary_allocation_count{0};
std::atomic<size_t> g_common_binary_shared_copy_count{0};
std::atomic<size_t> g_common_unary_allocation_count{0};
std::atomic<size_t> g_common_unary_shared_copy_count{0};
std::atomic<size_t> g_common_copy_allocation_count{0};
std::atomic<size_t> g_common_copy_shared_copy_count{0};
std::atomic<size_t> g_common_copy_scalar_allocation_count{0};
std::atomic<size_t> g_common_copy_scalar_shared_copy_count{0};
std::atomic<size_t> g_common_copy_vector_allocation_count{0};
std::atomic<size_t> g_common_copy_vector_shared_copy_count{0};
std::atomic<size_t> g_common_copy_general_allocation_count{0};
std::atomic<size_t> g_common_copy_general_shared_copy_count{0};
std::atomic<size_t> g_common_copy_general_general_allocation_count{0};
std::atomic<size_t> g_common_copy_general_general_shared_copy_count{0};
std::atomic<size_t> g_common_copy_gpri_allocation_count{0};
std::atomic<size_t> g_common_copy_gpri_shared_copy_count{0};
std::atomic<size_t> g_common_copy_gpri_astype_allocation_count{0};
std::atomic<size_t> g_common_copy_gpri_astype_shared_copy_count{0};
std::atomic<size_t> g_common_copy_gpri_contiguous_allocation_count{0};
std::atomic<size_t> g_common_copy_gpri_contiguous_shared_copy_count{0};
std::atomic<size_t> g_common_copy_gpri_full_allocation_count{0};
std::atomic<size_t> g_common_copy_gpri_full_shared_copy_count{0};
std::atomic<size_t> g_common_copy_gpri_slice_update_allocation_count{0};
std::atomic<size_t> g_common_copy_gpri_slice_update_shared_copy_count{0};
std::atomic<size_t> g_common_copy_gpri_dynamic_slice_update_allocation_count{0};
std::atomic<size_t> g_common_copy_gpri_dynamic_slice_update_shared_copy_count{0};
std::atomic<size_t> g_common_copy_idx_allocation_count{0};
std::atomic<size_t> g_common_copy_idx_shared_copy_count{0};
std::atomic<size_t> g_common_copy_rope_allocation_count{0};
std::atomic<size_t> g_common_copy_rope_shared_copy_count{0};
std::atomic<size_t> g_common_copy_matmul_allocation_count{0};
std::atomic<size_t> g_common_copy_matmul_shared_copy_count{0};
std::atomic<size_t> g_common_copy_hadamard_allocation_count{0};
std::atomic<size_t> g_common_copy_hadamard_shared_copy_count{0};
std::atomic<size_t> g_common_ternary_allocation_count{0};
std::atomic<size_t> g_common_ternary_shared_copy_count{0};
std::atomic<size_t> g_gpu_primitive_allocation_count{0};
std::atomic<size_t> g_gpu_primitive_shared_copy_count{0};
std::atomic<size_t> g_gpu_contiguous_copy_count{0};
std::atomic<size_t> g_quantized_contiguous_x_count{0};
std::atomic<size_t> g_quantized_contiguous_w_count{0};
std::atomic<size_t> g_quantized_contiguous_scales_count{0};
std::atomic<size_t> g_quantized_contiguous_biases_count{0};
std::atomic<size_t> g_quantized_contiguous_indices_count{0};
std::atomic<size_t> g_metal_norm_allocation_count{0};
std::atomic<size_t> g_metal_norm_shared_copy_count{0};
std::atomic<size_t> g_metal_matmul_allocation_count{0};
std::atomic<size_t> g_metal_matmul_shared_copy_count{0};
std::atomic<size_t> g_metal_quantized_allocation_count{0};
std::atomic<size_t> g_metal_quantized_shared_copy_count{0};
std::atomic<size_t> g_metal_sdpa_allocation_count{0};
std::atomic<size_t> g_metal_sdpa_shared_copy_count{0};
std::atomic<size_t> g_metal_reduce_allocation_count{0};
std::atomic<size_t> g_metal_reduce_shared_copy_count{0};
std::atomic<size_t> g_metal_indexing_allocation_count{0};
std::atomic<size_t> g_metal_indexing_shared_copy_count{0};
std::atomic<size_t> g_metal_index_concat_allocation_count{0};
std::atomic<size_t> g_metal_index_concat_shared_copy_count{0};
std::atomic<size_t> g_metal_index_gather_allocation_count{0};
std::atomic<size_t> g_metal_index_gather_shared_copy_count{0};
std::atomic<size_t> g_metal_index_gather_axis_allocation_count{0};
std::atomic<size_t> g_metal_index_gather_axis_shared_copy_count{0};
std::atomic<size_t> g_metal_index_dynamic_offset_allocation_count{0};
std::atomic<size_t> g_metal_index_dynamic_offset_shared_copy_count{0};
std::atomic<size_t> g_metal_copy_allocation_count{0};
std::atomic<size_t> g_metal_copy_shared_copy_count{0};
std::atomic<size_t> g_metal_direct_copy_allocation_count{0};
std::atomic<size_t> g_metal_direct_copy_shared_copy_count{0};
std::atomic<size_t> g_metal_rope_copy_allocation_count{0};
std::atomic<size_t> g_metal_rope_copy_shared_copy_count{0};
std::atomic<size_t> g_metal_scan_copy_allocation_count{0};
std::atomic<size_t> g_metal_scan_copy_shared_copy_count{0};
std::atomic<size_t> g_metal_primitive_copy_allocation_count{0};
std::atomic<size_t> g_metal_primitive_copy_shared_copy_count{0};
std::atomic<size_t> g_metal_reshape_copy_count{0};
std::atomic<size_t> g_metal_reshape_shared_count{0};
std::atomic<size_t> g_donation_reject_not_unique_count{0};
std::atomic<size_t> g_donation_reject_desc_not_unique_count{0};
std::atomic<size_t> g_donation_reject_data_not_unique_count{0};
std::atomic<size_t> g_donation_reject_itemsize_count{0};
std::atomic<size_t> g_donation_reject_oversize_count{0};
std::atomic<size_t> g_donation_reject_layout_count{0};
std::atomic<size_t> g_common_copy_reject_desc_not_unique_count{0};
std::atomic<size_t> g_common_copy_reject_data_not_unique_count{0};
std::atomic<size_t> g_common_binary_reject_desc_not_unique_count{0};
std::atomic<size_t> g_common_binary_reject_data_not_unique_count{0};
std::atomic<size_t> g_common_unary_reject_desc_not_unique_count{0};
std::atomic<size_t> g_common_unary_reject_data_not_unique_count{0};
std::atomic<size_t> g_common_binary_data_not_unique_scalar_vector_count{0};
std::atomic<size_t> g_common_binary_data_not_unique_vector_scalar_count{0};
std::atomic<size_t> g_common_binary_data_not_unique_vector_vector_count{0};
std::atomic<size_t> g_common_binary_data_not_unique_general_count{0};
std::atomic<size_t> g_common_binary_add_data_not_unique_vector_vector_count{0};
std::atomic<size_t> g_common_binary_add_data_not_unique_general_count{0};
std::atomic<size_t>
    g_common_binary_multiply_data_not_unique_vector_vector_count{0};
std::atomic<size_t> g_common_binary_multiply_data_not_unique_general_count{0};
thread_local const char* g_current_binary_op_name = nullptr;
thread_local const char* g_current_copy_site_name = nullptr;

} // namespace

size_t get_set_data_count() {
  return g_set_data_count.load(std::memory_order_relaxed);
}

size_t get_shared_buffer_copy_count() {
  return g_shared_buffer_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_binary_allocation_count() {
  return g_common_binary_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_binary_shared_copy_count() {
  return g_common_binary_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_unary_allocation_count() {
  return g_common_unary_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_unary_shared_copy_count() {
  return g_common_unary_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_allocation_count() {
  return g_common_copy_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_shared_copy_count() {
  return g_common_copy_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_scalar_allocation_count() {
  return g_common_copy_scalar_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_scalar_shared_copy_count() {
  return g_common_copy_scalar_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_vector_allocation_count() {
  return g_common_copy_vector_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_vector_shared_copy_count() {
  return g_common_copy_vector_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_general_allocation_count() {
  return g_common_copy_general_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_general_shared_copy_count() {
  return g_common_copy_general_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_general_general_allocation_count() {
  return g_common_copy_general_general_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_general_general_shared_copy_count() {
  return g_common_copy_general_general_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_allocation_count() {
  return g_common_copy_gpri_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_gpri_shared_copy_count() {
  return g_common_copy_gpri_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_gpri_astype_allocation_count() {
  return g_common_copy_gpri_astype_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_astype_shared_copy_count() {
  return g_common_copy_gpri_astype_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_contiguous_allocation_count() {
  return g_common_copy_gpri_contiguous_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_contiguous_shared_copy_count() {
  return g_common_copy_gpri_contiguous_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_full_allocation_count() {
  return g_common_copy_gpri_full_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_full_shared_copy_count() {
  return g_common_copy_gpri_full_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_slice_update_allocation_count() {
  return g_common_copy_gpri_slice_update_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_slice_update_shared_copy_count() {
  return g_common_copy_gpri_slice_update_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_dynamic_slice_update_allocation_count() {
  return g_common_copy_gpri_dynamic_slice_update_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_gpri_dynamic_slice_update_shared_copy_count() {
  return g_common_copy_gpri_dynamic_slice_update_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_idx_allocation_count() {
  return g_common_copy_idx_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_idx_shared_copy_count() {
  return g_common_copy_idx_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_rope_allocation_count() {
  return g_common_copy_rope_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_rope_shared_copy_count() {
  return g_common_copy_rope_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_matmul_allocation_count() {
  return g_common_copy_matmul_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_matmul_shared_copy_count() {
  return g_common_copy_matmul_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_hadamard_allocation_count() {
  return g_common_copy_hadamard_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_hadamard_shared_copy_count() {
  return g_common_copy_hadamard_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_common_ternary_allocation_count() {
  return g_common_ternary_allocation_count.load(std::memory_order_relaxed);
}

size_t get_common_ternary_shared_copy_count() {
  return g_common_ternary_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_gpu_primitive_allocation_count() {
  return g_gpu_primitive_allocation_count.load(std::memory_order_relaxed);
}

size_t get_gpu_primitive_shared_copy_count() {
  return g_gpu_primitive_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_gpu_contiguous_copy_count() {
  return g_gpu_contiguous_copy_count.load(std::memory_order_relaxed);
}

size_t get_quantized_contiguous_x_count() {
  return g_quantized_contiguous_x_count.load(std::memory_order_relaxed);
}

size_t get_quantized_contiguous_w_count() {
  return g_quantized_contiguous_w_count.load(std::memory_order_relaxed);
}

size_t get_quantized_contiguous_scales_count() {
  return g_quantized_contiguous_scales_count.load(std::memory_order_relaxed);
}

size_t get_quantized_contiguous_biases_count() {
  return g_quantized_contiguous_biases_count.load(std::memory_order_relaxed);
}

size_t get_quantized_contiguous_indices_count() {
  return g_quantized_contiguous_indices_count.load(std::memory_order_relaxed);
}

size_t get_metal_norm_allocation_count() {
  return g_metal_norm_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_norm_shared_copy_count() {
  return g_metal_norm_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_matmul_allocation_count() {
  return g_metal_matmul_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_matmul_shared_copy_count() {
  return g_metal_matmul_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_quantized_allocation_count() {
  return g_metal_quantized_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_quantized_shared_copy_count() {
  return g_metal_quantized_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_sdpa_allocation_count() {
  return g_metal_sdpa_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_sdpa_shared_copy_count() {
  return g_metal_sdpa_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_reduce_allocation_count() {
  return g_metal_reduce_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_reduce_shared_copy_count() {
  return g_metal_reduce_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_indexing_allocation_count() {
  return g_metal_indexing_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_indexing_shared_copy_count() {
  return g_metal_indexing_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_index_concat_allocation_count() {
  return g_metal_index_concat_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_index_concat_shared_copy_count() {
  return g_metal_index_concat_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_index_gather_allocation_count() {
  return g_metal_index_gather_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_index_gather_shared_copy_count() {
  return g_metal_index_gather_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_index_gather_axis_allocation_count() {
  return g_metal_index_gather_axis_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_metal_index_gather_axis_shared_copy_count() {
  return g_metal_index_gather_axis_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_metal_index_dynamic_offset_allocation_count() {
  return g_metal_index_dynamic_offset_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_metal_index_dynamic_offset_shared_copy_count() {
  return g_metal_index_dynamic_offset_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_metal_copy_allocation_count() {
  return g_metal_copy_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_copy_shared_copy_count() {
  return g_metal_copy_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_direct_copy_allocation_count() {
  return g_metal_direct_copy_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_direct_copy_shared_copy_count() {
  return g_metal_direct_copy_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_rope_copy_allocation_count() {
  return g_metal_rope_copy_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_rope_copy_shared_copy_count() {
  return g_metal_rope_copy_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_scan_copy_allocation_count() {
  return g_metal_scan_copy_allocation_count.load(std::memory_order_relaxed);
}

size_t get_metal_scan_copy_shared_copy_count() {
  return g_metal_scan_copy_shared_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_primitive_copy_allocation_count() {
  return g_metal_primitive_copy_allocation_count.load(
      std::memory_order_relaxed);
}

size_t get_metal_primitive_copy_shared_copy_count() {
  return g_metal_primitive_copy_shared_copy_count.load(
      std::memory_order_relaxed);
}

size_t get_metal_reshape_copy_count() {
  return g_metal_reshape_copy_count.load(std::memory_order_relaxed);
}

size_t get_metal_reshape_shared_count() {
  return g_metal_reshape_shared_count.load(std::memory_order_relaxed);
}

size_t get_donation_reject_not_unique_count() {
  return g_donation_reject_not_unique_count.load(std::memory_order_relaxed);
}

size_t get_donation_reject_itemsize_count() {
  return g_donation_reject_itemsize_count.load(std::memory_order_relaxed);
}

size_t get_donation_reject_desc_not_unique_count() {
  return g_donation_reject_desc_not_unique_count.load(std::memory_order_relaxed);
}

size_t get_donation_reject_data_not_unique_count() {
  return g_donation_reject_data_not_unique_count.load(std::memory_order_relaxed);
}

size_t get_donation_reject_oversize_count() {
  return g_donation_reject_oversize_count.load(std::memory_order_relaxed);
}

size_t get_donation_reject_layout_count() {
  return g_donation_reject_layout_count.load(std::memory_order_relaxed);
}

size_t get_common_copy_reject_desc_not_unique_count() {
  return g_common_copy_reject_desc_not_unique_count.load(
      std::memory_order_relaxed);
}

size_t get_common_copy_reject_data_not_unique_count() {
  return g_common_copy_reject_data_not_unique_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_reject_desc_not_unique_count() {
  return g_common_binary_reject_desc_not_unique_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_reject_data_not_unique_count() {
  return g_common_binary_reject_data_not_unique_count.load(
      std::memory_order_relaxed);
}

size_t get_common_unary_reject_desc_not_unique_count() {
  return g_common_unary_reject_desc_not_unique_count.load(
      std::memory_order_relaxed);
}

size_t get_common_unary_reject_data_not_unique_count() {
  return g_common_unary_reject_data_not_unique_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_data_not_unique_scalar_vector_count() {
  return g_common_binary_data_not_unique_scalar_vector_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_data_not_unique_vector_scalar_count() {
  return g_common_binary_data_not_unique_vector_scalar_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_data_not_unique_vector_vector_count() {
  return g_common_binary_data_not_unique_vector_vector_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_data_not_unique_general_count() {
  return g_common_binary_data_not_unique_general_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_add_data_not_unique_vector_vector_count() {
  return g_common_binary_add_data_not_unique_vector_vector_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_add_data_not_unique_general_count() {
  return g_common_binary_add_data_not_unique_general_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_multiply_data_not_unique_vector_vector_count() {
  return g_common_binary_multiply_data_not_unique_vector_vector_count.load(
      std::memory_order_relaxed);
}

size_t get_common_binary_multiply_data_not_unique_general_count() {
  return g_common_binary_multiply_data_not_unique_general_count.load(
      std::memory_order_relaxed);
}

void record_common_binary_allocation() {
  g_common_binary_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_binary_shared_copy() {
  g_common_binary_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_unary_allocation() {
  g_common_unary_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_unary_shared_copy() {
  g_common_unary_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_copy_allocation() {
  g_common_copy_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_copy_shared_copy() {
  g_common_copy_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_copy_scalar_allocation() {
  g_common_copy_scalar_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_copy_scalar_shared_copy() {
  g_common_copy_scalar_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_copy_vector_allocation() {
  g_common_copy_vector_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_copy_vector_shared_copy() {
  g_common_copy_vector_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_copy_general_allocation() {
  g_common_copy_general_allocation_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_copy_general_shared_copy() {
  g_common_copy_general_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_copy_general_general_allocation() {
  g_common_copy_general_general_allocation_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_copy_general_general_shared_copy() {
  g_common_copy_general_general_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_current_copy_site_allocation() {
  if (g_current_copy_site_name == nullptr) {
    return;
  }
  if (std::strcmp(g_current_copy_site_name, "gpri_astype") == 0) {
    g_common_copy_gpri_allocation_count.fetch_add(1, std::memory_order_relaxed);
    g_common_copy_gpri_astype_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "gpri_contiguous") == 0) {
    g_common_copy_idx_allocation_count.fetch_add(1, std::memory_order_relaxed);
    g_common_copy_gpri_contiguous_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "gpri_full") == 0) {
    g_common_copy_rope_allocation_count.fetch_add(1, std::memory_order_relaxed);
    g_common_copy_gpri_full_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (
      std::strcmp(g_current_copy_site_name, "gpri_slice_update") == 0) {
    g_common_copy_matmul_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
    g_common_copy_gpri_slice_update_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (
      std::strcmp(g_current_copy_site_name, "gpri_dynamic_slice_update") ==
      0) {
    g_common_copy_hadamard_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
    g_common_copy_gpri_dynamic_slice_update_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "idx") == 0) {
    g_common_copy_idx_allocation_count.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "rope") == 0) {
    g_common_copy_rope_allocation_count.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "matmul") == 0) {
    g_common_copy_matmul_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "hadamard") == 0) {
    g_common_copy_hadamard_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "norm") == 0) {
    g_common_copy_rope_allocation_count.fetch_add(1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "reduce") == 0) {
    g_common_copy_matmul_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "sdpa") == 0) {
    g_common_copy_hadamard_allocation_count.fetch_add(
        1, std::memory_order_relaxed);
  }
}

void record_current_copy_site_shared_copy() {
  if (g_current_copy_site_name == nullptr) {
    return;
  }
  if (std::strcmp(g_current_copy_site_name, "gpri_astype") == 0) {
    g_common_copy_gpri_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
    g_common_copy_gpri_astype_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "gpri_contiguous") == 0) {
    g_common_copy_idx_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
    g_common_copy_gpri_contiguous_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "gpri_full") == 0) {
    g_common_copy_rope_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
    g_common_copy_gpri_full_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (
      std::strcmp(g_current_copy_site_name, "gpri_slice_update") == 0) {
    g_common_copy_matmul_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
    g_common_copy_gpri_slice_update_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (
      std::strcmp(g_current_copy_site_name, "gpri_dynamic_slice_update") ==
      0) {
    g_common_copy_hadamard_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
    g_common_copy_gpri_dynamic_slice_update_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "idx") == 0) {
    g_common_copy_idx_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "rope") == 0) {
    g_common_copy_rope_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "matmul") == 0) {
    g_common_copy_matmul_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "hadamard") == 0) {
    g_common_copy_hadamard_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "norm") == 0) {
    g_common_copy_rope_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "reduce") == 0) {
    g_common_copy_matmul_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  } else if (std::strcmp(g_current_copy_site_name, "sdpa") == 0) {
    g_common_copy_hadamard_shared_copy_count.fetch_add(
        1, std::memory_order_relaxed);
  }
}

void record_common_ternary_allocation() {
  g_common_ternary_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_ternary_shared_copy() {
  g_common_ternary_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_gpu_primitive_allocation() {
  g_gpu_primitive_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_gpu_primitive_shared_copy() {
  g_gpu_primitive_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_gpu_contiguous_copy() {
  g_gpu_contiguous_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_quantized_contiguous_x() {
  g_quantized_contiguous_x_count.fetch_add(1, std::memory_order_relaxed);
}

void record_quantized_contiguous_w() {
  g_quantized_contiguous_w_count.fetch_add(1, std::memory_order_relaxed);
}

void record_quantized_contiguous_scales() {
  g_quantized_contiguous_scales_count.fetch_add(1, std::memory_order_relaxed);
}

void record_quantized_contiguous_biases() {
  g_quantized_contiguous_biases_count.fetch_add(1, std::memory_order_relaxed);
}

void record_quantized_contiguous_indices() {
  g_quantized_contiguous_indices_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_norm_allocation() {
  g_metal_norm_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_norm_shared_copy() {
  g_metal_norm_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_matmul_allocation() {
  g_metal_matmul_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_matmul_shared_copy() {
  g_metal_matmul_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_quantized_allocation() {
  g_metal_quantized_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_quantized_shared_copy() {
  g_metal_quantized_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_sdpa_allocation() {
  g_metal_sdpa_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_sdpa_shared_copy() {
  g_metal_sdpa_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_reduce_allocation() {
  g_metal_reduce_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_reduce_shared_copy() {
  g_metal_reduce_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_indexing_allocation() {
  g_metal_indexing_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_indexing_shared_copy() {
  g_metal_indexing_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_index_concat_allocation() {
  g_metal_index_concat_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_index_concat_shared_copy() {
  g_metal_index_concat_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_index_gather_allocation() {
  g_metal_index_gather_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_index_gather_shared_copy() {
  g_metal_index_gather_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_index_gather_axis_allocation() {
  g_metal_index_gather_axis_allocation_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_index_gather_axis_shared_copy() {
  g_metal_index_gather_axis_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_index_dynamic_offset_allocation() {
  g_metal_index_dynamic_offset_allocation_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_index_dynamic_offset_shared_copy() {
  g_metal_index_dynamic_offset_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_copy_allocation() {
  g_metal_copy_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_copy_shared_copy() {
  g_metal_copy_shared_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_direct_copy_allocation() {
  g_metal_direct_copy_allocation_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_direct_copy_shared_copy() {
  g_metal_direct_copy_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_rope_copy_allocation() {
  g_metal_rope_copy_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_rope_copy_shared_copy() {
  g_metal_rope_copy_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_scan_copy_allocation() {
  g_metal_scan_copy_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_scan_copy_shared_copy() {
  g_metal_scan_copy_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_primitive_copy_allocation() {
  g_metal_primitive_copy_allocation_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_primitive_copy_shared_copy() {
  g_metal_primitive_copy_shared_copy_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_metal_reshape_copy() {
  g_metal_reshape_copy_count.fetch_add(1, std::memory_order_relaxed);
}

void record_metal_reshape_shared() {
  g_metal_reshape_shared_count.fetch_add(1, std::memory_order_relaxed);
}

void record_donation_reject_not_unique() {
  g_donation_reject_not_unique_count.fetch_add(1, std::memory_order_relaxed);
}

void record_donation_reject_itemsize() {
  g_donation_reject_itemsize_count.fetch_add(1, std::memory_order_relaxed);
}

void record_donation_reject_desc_not_unique() {
  g_donation_reject_desc_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_donation_reject_data_not_unique() {
  g_donation_reject_data_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_donation_reject_oversize() {
  g_donation_reject_oversize_count.fetch_add(1, std::memory_order_relaxed);
}

void record_donation_reject_layout() {
  g_donation_reject_layout_count.fetch_add(1, std::memory_order_relaxed);
}

void record_common_copy_reject_desc_not_unique() {
  g_common_copy_reject_desc_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_copy_reject_data_not_unique() {
  g_common_copy_reject_data_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_reject_desc_not_unique() {
  g_common_binary_reject_desc_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_reject_data_not_unique() {
  g_common_binary_reject_data_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_unary_reject_desc_not_unique() {
  g_common_unary_reject_desc_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_unary_reject_data_not_unique() {
  g_common_unary_reject_data_not_unique_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_data_not_unique_scalar_vector() {
  g_common_binary_data_not_unique_scalar_vector_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_data_not_unique_vector_scalar() {
  g_common_binary_data_not_unique_vector_scalar_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_data_not_unique_vector_vector() {
  g_common_binary_data_not_unique_vector_vector_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_data_not_unique_general() {
  g_common_binary_data_not_unique_general_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_add_data_not_unique_vector_vector() {
  g_common_binary_add_data_not_unique_vector_vector_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_add_data_not_unique_general() {
  g_common_binary_add_data_not_unique_general_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_multiply_data_not_unique_vector_vector() {
  g_common_binary_multiply_data_not_unique_vector_vector_count.fetch_add(
      1, std::memory_order_relaxed);
}

void record_common_binary_multiply_data_not_unique_general() {
  g_common_binary_multiply_data_not_unique_general_count.fetch_add(
      1, std::memory_order_relaxed);
}

void set_current_binary_op_name(const char* op_name) {
  g_current_binary_op_name = op_name;
}

const char* current_binary_op_name() {
  return g_current_binary_op_name;
}

void set_current_copy_site_name(const char* site_name) {
  g_current_copy_site_name = site_name;
}

const char* current_copy_site_name() {
  return g_current_copy_site_name;
}

array::array(const std::complex<float>& val, Dtype dtype /* = complex64 */)
    : array_desc_(std::make_shared<ArrayDesc>(Shape{}, dtype)) {
  auto cval = static_cast<complex64_t>(val);
  init(&cval);
}

array::array(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              std::move(shape),
              dtype,
              std::move(primitive),
              std::move(inputs))) {
  if (has_primitive() && this->primitive().stream().device == Device::gpu) {
    for (auto& in : this->inputs()) {
      if (in.dtype() == float64) {
        throw std::invalid_argument("float64 is not supported on the GPU");
      }
    }
    if (this->dtype() == float64) {
      throw std::invalid_argument("float64 is not supported on the GPU");
    }
  }
}

std::vector<array> array::make_arrays(
    std::vector<Shape> shapes,
    const std::vector<Dtype>& dtypes,
    const std::shared_ptr<Primitive>& primitive,
    const std::vector<array>& inputs) {
  std::vector<array> outputs;
  for (size_t i = 0; i < shapes.size(); ++i) {
    outputs.emplace_back(std::move(shapes[i]), dtypes[i], primitive, inputs);
  }
  // For each node in |outputs|, its siblings are the other nodes.
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto siblings = outputs;
    siblings.erase(siblings.begin() + i);
    outputs[i].set_siblings(std::move(siblings), i);
  }
  return outputs;
}

array array::unsafe_weak_copy(const array& other) {
  auto cpy = array(other.shape(), other.dtype(), nullptr, {});
  cpy.set_data(
      other.buffer(),
      other.data_size(),
      other.strides(),
      other.flags(),
      [](auto) {});
  cpy.array_desc_->offset = other.array_desc_->offset;
  return cpy;
}

array::array(std::initializer_list<float> data)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              float32)) {
  init(data.begin());
}

array::array(std::initializer_list<int> data, Dtype dtype)
    : array_desc_(
          std::make_shared<ArrayDesc>(
              Shape{static_cast<ShapeElem>(data.size())},
              dtype)) {
  init(data.begin());
}

array::array(
    void* data,
    Shape shape,
    Dtype dtype,
    const std::function<void(void*)>& deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  auto buffer = allocator::make_buffer(data, nbytes());
  if (buffer.ptr() == nullptr) {
    set_data(allocator::malloc(nbytes()));
    auto ptr = static_cast<char*>(data);
    std::copy(ptr, ptr + nbytes(), this->data<char>());
    deleter(data);
  } else {
    auto wrapped_deleter = [deleter](allocator::Buffer buffer) {
      auto ptr = buffer.raw_ptr();
      allocator::release(buffer);
      return deleter(ptr);
    };
    set_data(buffer, std::move(wrapped_deleter));
  }
}

/* Build an array from a shared buffer */
array::array(allocator::Buffer data, Shape shape, Dtype dtype, Deleter deleter)
    : array_desc_(std::make_shared<ArrayDesc>(std::move(shape), dtype)) {
  set_data(data, deleter);
}

void array::detach() {
  array_desc_->primitive = nullptr;
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->primitive = nullptr;
  }
  for (auto& s : array_desc_->siblings) {
    s.array_desc_->inputs.clear();
    s.array_desc_->siblings.clear();
    s.array_desc_->position = 0;
  }
  array_desc_->inputs.clear();
  array_desc_->siblings.clear();
  array_desc_->position = 0;
}

bool array::is_available() const {
  if (status() == Status::available) {
    return true;
  } else if (
      status() == Status::evaluated &&
      (!event().valid() || event().is_signaled())) {
    detach_event();
    set_status(Status::available);
    return true;
  }
  return false;
}

void array::wait() {
  if (!is_available()) {
    if (event().valid()) {
      event().wait();
      detach_event();
    }
    set_status(Status::available);
  }
}

void array::eval() {
  // Ensure the array is ready to be read
  if (status() == Status::unscheduled) {
    mlx::core::eval({*this});
  } else {
    wait();
  }
}

bool array::is_tracer() const {
  return (array_desc_->is_tracer && detail::in_tracing()) ||
      detail::retain_graph();
}

void array::set_data(allocator::Buffer buffer, Deleter d) {
  g_set_data_count.fetch_add(1, std::memory_order_relaxed);
  array_desc_->data = std::make_shared<Data>(buffer, d);
  array_desc_->offset = 0;
  array_desc_->data_size = size();
  array_desc_->flags.contiguous = true;
  array_desc_->flags.row_contiguous = true;
  auto max_dim = std::max_element(shape().begin(), shape().end());
  array_desc_->flags.col_contiguous = size() <= 1 || size() == *max_dim;
}

void array::set_data(
    allocator::Buffer buffer,
    size_t data_size,
    Strides strides,
    Flags flags,
    Deleter d) {
  g_set_data_count.fetch_add(1, std::memory_order_relaxed);
  array_desc_->data = std::make_shared<Data>(buffer, d);
  array_desc_->offset = 0;
  array_desc_->data_size = data_size;
  array_desc_->strides = std::move(strides);
  array_desc_->flags = flags;
}

void array::copy_shared_buffer(
    const array& other,
    const Strides& strides,
    Flags flags,
    size_t data_size,
    int64_t offset /* = 0 */) {
  g_shared_buffer_copy_count.fetch_add(1, std::memory_order_relaxed);
  array_desc_->data = other.array_desc_->data;
  array_desc_->strides = strides;
  array_desc_->flags = flags;
  array_desc_->data_size = data_size;
  array_desc_->offset =
      sizeof(char) * itemsize() * offset + other.array_desc_->offset;
}

void array::copy_shared_buffer(const array& other) {
  copy_shared_buffer(other, other.strides(), other.flags(), other.data_size());
}

array::~array() {
  if (array_desc_ == nullptr) {
    return;
  }

  // Detached/detaching
  if (array_desc_->primitive == nullptr) {
    return;
  }

  // Break circular reference for non-detached arrays with siblings
  if (auto n = siblings().size(); n > 0) {
    bool do_detach = true;
    // If all siblings have siblings.size() references except
    // the one we are currently destroying (which has siblings.size() + 1)
    // then there are no more external references
    do_detach &= (array_desc_.use_count() == (n + 1));
    for (auto& s : siblings()) {
      do_detach &= (s.array_desc_.use_count() == n);
      if (!do_detach) {
        break;
      }
    }
    if (do_detach) {
      for (auto& s : siblings()) {
        for (auto& ss : s.siblings()) {
          // Set to null here to avoid descending into array destructor
          // for siblings
          ss.array_desc_ = nullptr;
        }
        s.array_desc_->siblings.clear();
      }
    }
  }
}

void array::ArrayDesc::init() {
  strides.resize(shape.size());
  size = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = size;
    size *= shape[i];
  }
  for (const auto& in : inputs) {
    is_tracer |= in.is_tracer();
  }
}

array::ArrayDesc::ArrayDesc(Shape shape, Dtype dtype)
    : shape(std::move(shape)), dtype(dtype), status(Status::available) {
  init();
}

array::ArrayDesc::ArrayDesc(
    Shape shape,
    Dtype dtype,
    std::shared_ptr<Primitive> primitive,
    std::vector<array> inputs)
    : shape(std::move(shape)),
      dtype(dtype),
      primitive(std::move(primitive)),
      status(Status::unscheduled),
      inputs(std::move(inputs)) {
  init();
}

array::ArrayDesc::~ArrayDesc() {
  // When an array description is destroyed it will delete a bunch of arrays
  // that may also destroy their corresponding descriptions and so on and so
  // forth.
  //
  // This calls recursively the destructor and can result in stack overflow, we
  // instead put them in a vector and destroy them one at a time resulting in a
  // max stack depth of 2.
  if (inputs.empty()) {
    return;
  }

  std::vector<std::shared_ptr<ArrayDesc>> for_deletion;

  auto append_deletable_inputs = [&for_deletion](ArrayDesc& ad) {
    std::unordered_map<std::uintptr_t, array> input_map;
    for (array& a : ad.inputs) {
      if (a.array_desc_) {
        input_map.insert({a.id(), a});
        for (auto& s : a.siblings()) {
          input_map.insert({s.id(), s});
        }
      }
    }
    ad.inputs.clear();
    for (auto& [_, a] : input_map) {
      bool is_deletable =
          (a.array_desc_.use_count() <= a.siblings().size() + 1);
      // An array with siblings is deletable only if all of its siblings
      // are deletable
      for (auto& s : a.siblings()) {
        if (!is_deletable) {
          break;
        }
        int is_input = (input_map.find(s.id()) != input_map.end());
        is_deletable &=
            s.array_desc_.use_count() <= a.siblings().size() + is_input;
      }
      if (is_deletable) {
        for_deletion.push_back(std::move(a.array_desc_));
      }
    }
  };

  append_deletable_inputs(*this);

  while (!for_deletion.empty()) {
    // top is going to be deleted at the end of the block *after* the arrays
    // with inputs have been moved into the vector
    auto top = std::move(for_deletion.back());
    for_deletion.pop_back();
    append_deletable_inputs(*top);

    // Clear out possible siblings to break circular references
    for (auto& s : top->siblings) {
      // Set to null here to avoid descending into top-level
      // array destructor for siblings
      s.array_desc_ = nullptr;
    }
    top->siblings.clear();
  }
}

array::ArrayIterator::ArrayIterator(const array& arr, int idx)
    : arr(arr), idx(idx) {
  if (arr.ndim() == 0) {
    throw std::invalid_argument("Cannot iterate over 0-d array.");
  }
}

array::ArrayIterator::reference array::ArrayIterator::operator*() const {
  auto start = Shape(arr.ndim(), 0);
  auto end = arr.shape();
  auto shape = arr.shape();
  shape.erase(shape.begin());
  start[0] = idx;
  end[0] = idx + 1;
  return reshape(slice(arr, start, end), shape);
};

} // namespace mlx::core
