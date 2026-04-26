// Copyright © 2023 Apple Inc.

#pragma once
#include <cstring>
#include <atomic>
#include <iostream>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/memory.h"
#include "mlx/utils.h"

namespace mlx::core {

inline void record_common_binary_decision(DonationDecision decision) {
  switch (decision) {
    case DonationDecision::desc_not_unique:
      record_common_binary_reject_desc_not_unique();
      break;
    case DonationDecision::data_not_unique:
      record_common_binary_reject_data_not_unique();
      break;
    default:
      break;
  }
}

enum class BinaryOpType {
  ScalarScalar,
  ScalarVector,
  VectorScalar,
  VectorVector,
  General,
};

inline void record_common_binary_data_not_unique_by_type(BinaryOpType bopt) {
  auto op_name = current_binary_op_name();
  switch (bopt) {
    case BinaryOpType::ScalarScalar:
      break;
    case BinaryOpType::ScalarVector:
      record_common_binary_data_not_unique_scalar_vector();
      break;
    case BinaryOpType::VectorScalar:
      record_common_binary_data_not_unique_vector_scalar();
      break;
    case BinaryOpType::VectorVector:
      record_common_binary_data_not_unique_vector_vector();
      if (op_name && std::strcmp(op_name, "Add") == 0) {
        record_common_binary_add_data_not_unique_vector_vector();
      } else if (op_name && std::strcmp(op_name, "Multiply") == 0) {
        record_common_binary_multiply_data_not_unique_vector_vector();
      }
      break;
    case BinaryOpType::General:
      record_common_binary_data_not_unique_general();
      if (op_name && std::strcmp(op_name, "Add") == 0) {
        record_common_binary_add_data_not_unique_general();
      } else if (op_name && std::strcmp(op_name, "Multiply") == 0) {
        record_common_binary_multiply_data_not_unique_general();
      }
      break;
  }
}

inline void maybe_trace_binary_multiply_general(
    const array& a,
    const array& b,
    const array& out,
    DonationDecision a_decision,
    DonationDecision b_decision,
    BinaryOpType bopt) {
  auto op_name = current_binary_op_name();
  if (!op_name || std::strcmp(op_name, "Multiply") != 0 ||
      bopt != BinaryOpType::General) {
    return;
  }
  if (a_decision != DonationDecision::data_not_unique &&
      b_decision != DonationDecision::data_not_unique) {
    return;
  }
  bool likely_decoder_shape =
      (out.ndim() == 4 && out.shape(0) == 1 && out.shape(2) == 1) ||
      (out.ndim() == 3 && out.shape(0) == 1 && out.shape(1) == 1) ||
      (out.ndim() == 2 && out.shape(0) == 1);
  if (!likely_decoder_shape) {
    return;
  }
  static std::atomic<int> remaining{env::trace_binary_multiply_general_limit()};
  int prev = remaining.fetch_sub(1, std::memory_order_relaxed);
  if (prev <= 0) {
    return;
  }
  auto fmt_dims = [](const auto& vec) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << vec[i];
    }
    os << "]";
    return os.str();
  };
  std::cerr << "[mlx][multiply-general]"
            << " a_shape=" << fmt_dims(a.shape())
            << " a_strides=" << fmt_dims(a.strides())
            << " a_buf=" << a.buffer_size() << " a_row=" << a.flags().row_contiguous
            << " a_col=" << a.flags().col_contiguous
            << " b_shape=" << fmt_dims(b.shape())
            << " b_strides=" << fmt_dims(b.strides())
            << " b_buf=" << b.buffer_size() << " b_row=" << b.flags().row_contiguous
            << " b_col=" << b.flags().col_contiguous
            << " out_shape=" << fmt_dims(out.shape())
            << " out_strides=" << fmt_dims(out.strides())
            << " out_nbytes=" << out.nbytes()
            << " a_decision=" << static_cast<int>(a_decision)
            << " b_decision=" << static_cast<int>(b_decision)
            << std::endl;
}

inline BinaryOpType get_binary_op_type(const array& a, const array& b) {
  BinaryOpType bopt;
  if (a.data_size() == 1 && b.data_size() == 1) {
    bopt = BinaryOpType::ScalarScalar;
  } else if (a.data_size() == 1 && b.flags().contiguous) {
    bopt = BinaryOpType::ScalarVector;
  } else if (b.data_size() == 1 && a.flags().contiguous) {
    bopt = BinaryOpType::VectorScalar;
  } else if (
      (a.flags().row_contiguous && b.flags().row_contiguous) ||
      (a.flags().col_contiguous && b.flags().col_contiguous)) {
    bopt = BinaryOpType::VectorVector;
  } else {
    bopt = BinaryOpType::General;
  }
  return bopt;
}

inline void set_binary_op_output_data(
    const array& a,
    const array& b,
    array& out,
    BinaryOpType bopt,
    std::function<allocator::Buffer(size_t)> mallocfn = allocator::malloc) {
  auto b_decision = donation_decision(b, out);
  auto a_decision = donation_decision(a, out);
  bool b_donatable = b_decision == DonationDecision::success;
  bool a_donatable = a_decision == DonationDecision::success;
  auto donate_a = [&]() {
    record_common_binary_shared_copy();
    out.copy_shared_buffer(a);
  };
  auto donate_b = [&]() {
    record_common_binary_shared_copy();
    out.copy_shared_buffer(b);
  };
  auto set_data = [&](allocator::Buffer buffer, size_t dataSize, const Strides& strides, const array::Flags& flags) {
    record_common_binary_allocation();
    out.set_data(buffer, dataSize, strides, flags);
  };
  auto set_full = [&](allocator::Buffer buffer) {
    record_common_binary_allocation();
    out.set_data(buffer);
  };
  switch (bopt) {
    case BinaryOpType::ScalarScalar:
      set_data(mallocfn(out.itemsize()), 1, a.strides(), a.flags());
      break;
    case BinaryOpType::ScalarVector:
      if (b_donatable) {
        donate_b();
      } else {
        record_donation_rejection(b_decision);
        record_common_binary_decision(b_decision);
        if (b_decision == DonationDecision::data_not_unique) {
          record_common_binary_data_not_unique_by_type(bopt);
        }
        set_data(
            mallocfn(b.data_size() * out.itemsize()),
            b.data_size(),
            b.strides(),
            b.flags());
      }
      break;
    case BinaryOpType::VectorScalar:
      if (a_donatable) {
        donate_a();
      } else {
        record_donation_rejection(a_decision);
        record_common_binary_decision(a_decision);
        if (a_decision == DonationDecision::data_not_unique) {
          record_common_binary_data_not_unique_by_type(bopt);
        }
        set_data(
            mallocfn(a.data_size() * out.itemsize()),
            a.data_size(),
            a.strides(),
            a.flags());
      }
      break;
    case BinaryOpType::VectorVector:
      if (a_donatable) {
        donate_a();
      } else if (b_donatable) {
        donate_b();
      } else {
        record_donation_rejection(a_decision);
        record_common_binary_decision(a_decision);
        if (a_decision == DonationDecision::data_not_unique) {
          record_common_binary_data_not_unique_by_type(bopt);
        }
        record_donation_rejection(b_decision);
        record_common_binary_decision(b_decision);
        if (b_decision == DonationDecision::data_not_unique) {
          record_common_binary_data_not_unique_by_type(bopt);
        }
        set_data(
            mallocfn(a.data_size() * out.itemsize()),
            a.data_size(),
            a.strides(),
            a.flags());
      }
      break;
    case BinaryOpType::General:
      maybe_trace_binary_multiply_general(a, b, out, a_decision, b_decision, bopt);
      if (a_donatable && a.flags().row_contiguous && a.size() == out.size()) {
        donate_a();
      } else if (
          b_donatable && b.flags().row_contiguous && b.size() == out.size()) {
        donate_b();
      } else {
        if (a_donatable && !(a.flags().row_contiguous && a.size() == out.size())) {
          record_donation_reject_layout();
        } else {
          record_donation_rejection(a_decision);
          record_common_binary_decision(a_decision);
          if (a_decision == DonationDecision::data_not_unique) {
            record_common_binary_data_not_unique_by_type(bopt);
          }
        }
        if (b_donatable && !(b.flags().row_contiguous && b.size() == out.size())) {
          record_donation_reject_layout();
        } else {
          record_donation_rejection(b_decision);
          record_common_binary_decision(b_decision);
          if (b_decision == DonationDecision::data_not_unique) {
            record_common_binary_data_not_unique_by_type(bopt);
          }
        }
        set_full(mallocfn(out.nbytes()));
      }
      break;
  }
}

} // namespace mlx::core
