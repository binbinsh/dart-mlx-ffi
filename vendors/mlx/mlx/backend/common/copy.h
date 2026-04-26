// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/backend/common/utils.h"
#include "mlx/memory.h"

namespace mlx::core {

inline void record_common_copy_decision(DonationDecision decision) {
  switch (decision) {
    case DonationDecision::desc_not_unique:
      record_common_copy_reject_desc_not_unique();
      break;
    case DonationDecision::data_not_unique:
      record_common_copy_reject_data_not_unique();
      break;
    default:
      break;
  }
}

enum class CopyType {
  // Copy a raw scalar input into the full contiguous output
  Scalar,

  // Copy the raw input buffer contiguously into a raw output buffer of the same
  // size
  Vector,

  // Copy the full virtual input to the full contiguous output
  General,

  // Copy the full virtual input to the full virtual output. We assume the
  // input and output have the same shape.
  GeneralGeneral
};

struct ScopedCopySite {
  explicit ScopedCopySite(const char* site_name)
      : previous_(current_copy_site_name()) {
    set_current_copy_site_name(site_name);
  }

  ~ScopedCopySite() {
    set_current_copy_site_name(previous_);
  }

 private:
  const char* previous_;
};

inline void record_common_copy_allocation_by_type(CopyType ctype) {
  switch (ctype) {
    case CopyType::Scalar:
      record_common_copy_scalar_allocation();
      break;
    case CopyType::Vector:
      record_common_copy_vector_allocation();
      break;
    case CopyType::General:
      record_common_copy_general_allocation();
      break;
    case CopyType::GeneralGeneral:
      record_common_copy_general_general_allocation();
      break;
  }
}

inline void record_common_copy_shared_copy_by_type(CopyType ctype) {
  switch (ctype) {
    case CopyType::Scalar:
      record_common_copy_scalar_shared_copy();
      break;
    case CopyType::Vector:
      record_common_copy_vector_shared_copy();
      break;
    case CopyType::General:
      record_common_copy_general_shared_copy();
      break;
    case CopyType::GeneralGeneral:
      record_common_copy_general_general_shared_copy();
      break;
  }
}

inline bool set_copy_output_data(
    const array& in,
    array& out,
    CopyType ctype,
    std::function<allocator::Buffer(size_t)> mallocfn = allocator::malloc) {
  if (ctype == CopyType::Vector) {
    // If the input is donateable, we are doing a vector copy and the types
    // have the same size, then the input buffer can hold the output.
    auto decision = donation_decision(in, out);
    if (decision == DonationDecision::success) {
      record_common_copy_shared_copy();
      record_common_copy_shared_copy_by_type(ctype);
      record_current_copy_site_shared_copy();
      out.copy_shared_buffer(in);
      return true;
    } else {
      record_donation_rejection(decision);
      record_common_copy_decision(decision);
      record_common_copy_allocation();
      record_common_copy_allocation_by_type(ctype);
      record_current_copy_site_allocation();
      out.set_data(
          mallocfn(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
      return false;
    }
  } else {
    record_common_copy_allocation();
    record_common_copy_allocation_by_type(ctype);
    record_current_copy_site_allocation();
    out.set_data(mallocfn(out.nbytes()));
    return false;
  }
}

} // namespace mlx::core
