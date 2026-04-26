// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <mutex>

#include "mlx/allocator.h"
#include "mlx/memory.h"

#ifdef __APPLE__
#include "mlx/backend/no_gpu/apple_memory.h"
#elif defined(__linux__)
#include "mlx/backend/no_gpu/linux_memory.h"
#else
size_t get_memory_size() {
  return 0;
}
#endif

namespace mlx::core {

namespace allocator {

class CommonAllocator : public Allocator {
  /** A general CPU allocator. */
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;

  size_t get_active_memory() const {
    return active_memory_;
  };
  size_t get_peak_memory() const {
    return peak_memory_;
  };
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  };
  size_t get_memory_limit() {
    return memory_limit_;
  }
  size_t set_memory_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(memory_limit_, limit);
    return limit;
  }

 private:
  size_t memory_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  std::mutex mutex_;
  CommonAllocator() : memory_limit_(0.8 * get_memory_size()) {
    if (memory_limit_ == 0) {
      memory_limit_ = 1UL << 33;
    }
  };

  friend CommonAllocator& common_allocator();
};

CommonAllocator& common_allocator() {
  static CommonAllocator allocator_;
  return allocator_;
}

Allocator& allocator() {
  return common_allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<size_t*>(ptr_) + 1;
}

Buffer CommonAllocator::malloc(size_t size) {
  void* ptr = std::malloc(size + sizeof(size_t));
  if (ptr != nullptr) {
    *static_cast<size_t*>(ptr) = size;
  }
  std::unique_lock lk(mutex_);
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{ptr};
}

void CommonAllocator::free(Buffer buffer) {
  auto sz = size(buffer);
  std::free(buffer.ptr());
  std::unique_lock lk(mutex_);
  active_memory_ -= sz;
}

size_t CommonAllocator::size(Buffer buffer) const {
  if (buffer.ptr() == nullptr) {
    return 0;
  }
  return *static_cast<size_t*>(buffer.ptr());
}

} // namespace allocator

size_t get_active_memory() {
  return allocator::common_allocator().get_active_memory();
}
size_t get_peak_memory() {
  return allocator::common_allocator().get_peak_memory();
}
void reset_peak_memory() {
  return allocator::common_allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return allocator::common_allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return allocator::common_allocator().get_memory_limit();
}

// No-ops for common allocator
size_t get_cache_memory() {
  return 0;
}
size_t get_cache_limit() {
  return 0;
}
size_t get_cache_count() {
  return 0;
}
size_t set_cache_limit(size_t) {
  return 0;
}
size_t set_wired_limit(size_t) {
  return 0;
}
size_t get_wired_limit() {
  return 0;
}
size_t get_resource_count() {
  return 0;
}
size_t get_resource_limit() {
  return 0;
}
size_t get_command_buffer_commit_count() {
  return 0;
}
size_t get_pending_output_count() {
  return 0;
}
size_t get_temporary_count() {
  return 0;
}
size_t get_buffer_op_count() {
  return 0;
}
size_t get_buffer_size_bytes() {
  return 0;
}
size_t get_stream_count() {
  return 0;
}
size_t get_allocation_request_count() {
  return 0;
}
size_t get_cache_reuse_hit_count() {
  return 0;
}
size_t get_new_allocation_count() {
  return 0;
}
size_t get_heap_allocation_count() {
  return 0;
}
size_t get_device_allocation_count() {
  return 0;
}
size_t get_common_binary_allocation_count() {
  return 0;
}
size_t get_common_binary_shared_copy_count() {
  return 0;
}
size_t get_common_unary_allocation_count() {
  return 0;
}
size_t get_common_unary_shared_copy_count() {
  return 0;
}
size_t get_common_copy_allocation_count() {
  return 0;
}
size_t get_common_copy_shared_copy_count() {
  return 0;
}
size_t get_common_ternary_allocation_count() {
  return 0;
}
size_t get_common_ternary_shared_copy_count() {
  return 0;
}
size_t get_gpu_primitive_allocation_count() {
  return 0;
}
size_t get_gpu_primitive_shared_copy_count() {
  return 0;
}
size_t get_metal_norm_allocation_count() {
  return 0;
}
size_t get_metal_norm_shared_copy_count() {
  return 0;
}
size_t get_metal_matmul_allocation_count() {
  return 0;
}
size_t get_metal_matmul_shared_copy_count() {
  return 0;
}
size_t get_metal_quantized_allocation_count() {
  return 0;
}
size_t get_metal_quantized_shared_copy_count() {
  return 0;
}
size_t get_metal_sdpa_allocation_count() {
  return 0;
}
size_t get_metal_sdpa_shared_copy_count() {
  return 0;
}
size_t get_metal_reduce_allocation_count() {
  return 0;
}
size_t get_metal_reduce_shared_copy_count() {
  return 0;
}
size_t get_metal_indexing_allocation_count() {
  return 0;
}
size_t get_metal_indexing_shared_copy_count() {
  return 0;
}
size_t get_metal_copy_allocation_count() {
  return 0;
}
size_t get_metal_copy_shared_copy_count() {
  return 0;
}
size_t get_donation_reject_not_unique_count() {
  return 0;
}
size_t get_donation_reject_desc_not_unique_count() {
  return 0;
}
size_t get_donation_reject_data_not_unique_count() {
  return 0;
}
size_t get_donation_reject_itemsize_count() {
  return 0;
}
size_t get_donation_reject_oversize_count() {
  return 0;
}
size_t get_donation_reject_layout_count() {
  return 0;
}
size_t get_common_copy_reject_desc_not_unique_count() {
  return 0;
}
size_t get_common_copy_reject_data_not_unique_count() {
  return 0;
}
size_t get_common_binary_reject_desc_not_unique_count() {
  return 0;
}
size_t get_common_binary_reject_data_not_unique_count() {
  return 0;
}
size_t get_common_unary_reject_desc_not_unique_count() {
  return 0;
}
size_t get_common_unary_reject_data_not_unique_count() {
  return 0;
}
size_t get_common_binary_data_not_unique_scalar_vector_count() {
  return 0;
}
size_t get_common_binary_data_not_unique_vector_scalar_count() {
  return 0;
}
size_t get_common_binary_data_not_unique_vector_vector_count() {
  return 0;
}
size_t get_common_binary_data_not_unique_general_count() {
  return 0;
}
size_t get_common_binary_add_data_not_unique_vector_vector_count() {
  return 0;
}
size_t get_common_binary_add_data_not_unique_general_count() {
  return 0;
}
size_t get_common_binary_multiply_data_not_unique_vector_vector_count() {
  return 0;
}
size_t get_common_binary_multiply_data_not_unique_general_count() {
  return 0;
}
void clear_cache() {}

} // namespace mlx::core
