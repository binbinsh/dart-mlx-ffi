#include "bridge.h"
#include "ane_p.h"

#include "mlx/types/half_types.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace {

uint16_t fp16_bits_from_float(float value) {
  auto fp16 = static_cast<mlx::core::float16_t>(value);
  uint16_t bits = 0;
  static_assert(sizeof(fp16) == sizeof(bits));
  std::memcpy(&bits, &fp16, sizeof(bits));
  return bits;
}

float float_from_fp16_bits(uint16_t bits) {
  mlx::core::float16_t fp16{};
  static_assert(sizeof(fp16) == sizeof(bits));
  std::memcpy(&fp16, &bits, sizeof(bits));
  return static_cast<float>(fp16);
}

void set_error(const char* message) {
  dart_mlx_ane_private_set_error_message(message);
}

} // namespace

extern "C" uint8_t* dart_mlx_ane_private_encode_fp32_to_fp16_bytes_copy(
    const float* values,
    size_t count) {
  dart_mlx_ane_private_clear_error();
  if (count > 0 && values == nullptr) {
    set_error("FP32 input pointer is null.");
    return nullptr;
  }
  auto* bytes = static_cast<uint8_t*>(std::malloc(count * sizeof(uint16_t)));
  if (bytes == nullptr && count > 0) {
    set_error("Failed to allocate an FP16 output buffer.");
    return nullptr;
  }
  auto* out = reinterpret_cast<uint16_t*>(bytes);
  for (size_t index = 0; index < count; index++) {
    out[index] = fp16_bits_from_float(values[index]);
  }
  return bytes;
}

extern "C" float* dart_mlx_ane_private_decode_fp16_bytes_to_fp32_copy(
    const uint8_t* bytes,
    size_t byte_len,
    size_t* count_out) {
  dart_mlx_ane_private_clear_error();
  if (count_out == nullptr) {
    set_error("FP16 decode output-count pointer is null.");
    return nullptr;
  }
  if ((byte_len % sizeof(uint16_t)) != 0) {
    set_error("FP16 byte length must be divisible by 2.");
    return nullptr;
  }
  if (byte_len > 0 && bytes == nullptr) {
    set_error("FP16 input pointer is null.");
    return nullptr;
  }

  const auto count = byte_len / sizeof(uint16_t);
  auto* values = static_cast<float*>(std::malloc(count * sizeof(float)));
  if (values == nullptr && count > 0) {
    set_error("Failed to allocate an FP32 output buffer.");
    return nullptr;
  }

  const auto* in = reinterpret_cast<const uint16_t*>(bytes);
  for (size_t index = 0; index < count; index++) {
    values[index] = float_from_fp16_bits(in[index]);
  }
  *count_out = count;
  return values;
}
