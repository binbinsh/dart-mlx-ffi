// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/memory.h"

namespace mlx::core {

inline void record_common_unary_decision(DonationDecision decision) {
  switch (decision) {
    case DonationDecision::desc_not_unique:
      record_common_unary_reject_desc_not_unique();
      break;
    case DonationDecision::data_not_unique:
      record_common_unary_reject_data_not_unique();
      break;
    default:
      break;
  }
}

inline void set_unary_output_data(
    const array& in,
    array& out,
    std::function<allocator::Buffer(size_t)> mallocfn = allocator::malloc) {
  if (in.flags().contiguous) {
    auto decision = donation_decision(in, out);
    if (decision == DonationDecision::success) {
      record_common_unary_shared_copy();
      out.copy_shared_buffer(in);
    } else {
      record_donation_rejection(decision);
      record_common_unary_decision(decision);
      record_common_unary_allocation();
      out.set_data(
          mallocfn(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    record_common_unary_allocation();
    out.set_data(mallocfn(out.nbytes()));
  }
}

} // namespace mlx::core
