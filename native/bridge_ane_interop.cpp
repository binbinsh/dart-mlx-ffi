#include "bridge.h"
#include "ane_p.h"

#include "ane_interop.h"

#include <cstdlib>
#include <string>

struct DartMlxAneInteropHandle {
  ANEHandle* value;
  size_t input_bytes;
  size_t output_bytes;
  int input_channels;
  int input_spatial;
  int output_channels;
  int output_spatial;
};

namespace {

void set_error(const std::string& message) {
  dart_mlx_ane_private_set_error_message(message.c_str());
}

} // namespace

extern "C" int dart_mlx_ane_interop_set_eval_path(const char* value) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (value == nullptr || value[0] == '\0') {
    unsetenv("ANE_EVAL_PATH");
    return 0;
  }
  if (setenv("ANE_EVAL_PATH", value, 1) != 0) {
    set_error("Failed to set ANEInterop eval path.");
    return -1;
  }
  return 0;
#else
  (void)value;
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
#endif
}

extern "C" char* dart_mlx_ane_interop_eval_path_copy() {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  const char* value = std::getenv("ANE_EVAL_PATH");
  if (value == nullptr || value[0] == '\0') {
    return nullptr;
  }
  return copy_c_string(value);
#else
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
#endif
}

extern "C" DartMlxAneInteropHandle* dart_mlx_ane_interop_new_single_io(
    const char* mil_text,
    const char* const* weight_paths,
    const uint8_t* const* weight_data,
    const size_t* weight_lens,
    size_t weight_count,
    size_t input_bytes,
    size_t output_bytes,
    int input_channels,
    int input_spatial,
    int output_channels,
    int output_spatial) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (mil_text == nullptr || mil_text[0] == '\0') {
    set_error("ANEInterop MIL text must be non-empty.");
    return nullptr;
  }
  if (input_bytes == 0 || output_bytes == 0) {
    set_error("ANEInterop byte sizes must be positive.");
    return nullptr;
  }
  if (input_channels <= 0 || input_spatial <= 0 || output_channels <= 0 ||
      output_spatial <= 0) {
    set_error("ANEInterop channel/spatial dimensions must be positive.");
    return nullptr;
  }

  const int n_inputs = 1;
  const int n_outputs = 1;
  auto in_bytes = input_bytes;
  auto out_bytes = output_bytes;
  auto* value = ane_interop_compile(
      reinterpret_cast<const uint8_t*>(mil_text),
      std::strlen(mil_text),
      const_cast<const char**>(weight_paths),
      const_cast<const uint8_t**>(weight_data),
      weight_lens,
      static_cast<int>(weight_count),
      n_inputs,
      &in_bytes,
      n_outputs,
      &out_bytes);
  if (value == nullptr) {
    set_error(
        "ANEInterop compile failed with code " +
        std::to_string(ane_interop_last_compile_error()) +
        ".");
    return nullptr;
  }

  auto* handle = new DartMlxAneInteropHandle{
      value,
      input_bytes,
      output_bytes,
      input_channels,
      input_spatial,
      output_channels,
      output_spatial,
  };
  return handle;
#else
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
#endif
}

extern "C" void dart_mlx_ane_interop_free(DartMlxAneInteropHandle* handle) {
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr) return;
  ane_interop_free(handle->value);
  delete handle;
#else
  (void)handle;
#endif
}

extern "C" int dart_mlx_ane_interop_eval(DartMlxAneInteropHandle* handle) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr || handle->value == nullptr) {
    set_error("ANEInterop handle is closed.");
    return -1;
  }
  if (!ane_interop_eval(handle->value)) {
    set_error("ANEInterop eval returned false.");
    return -1;
  }
  return 0;
#else
  (void)handle;
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
#endif
}

extern "C" int dart_mlx_ane_interop_write_input_f32(
    DartMlxAneInteropHandle* handle,
    const float* values,
    size_t count) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr || handle->value == nullptr) {
    set_error("ANEInterop handle is closed.");
    return -1;
  }
  const auto expected = static_cast<size_t>(handle->input_channels * handle->input_spatial);
  if (count != expected) {
    set_error("ANEInterop input element count mismatch.");
    return -1;
  }
  auto surface = ane_interop_get_input(handle->value, 0);
  auto owned_surface = ane_interop_copy_input(handle->value, 0);
  surface = owned_surface;
  if (!surface) {
    set_error("ANEInterop input surface is null.");
    return -1;
  }
  const bool ok = ane_interop_io_write_fp16(
      surface,
      values,
      handle->input_channels,
      handle->input_spatial);
  if (owned_surface) {
    CFRelease(owned_surface);
  }
  if (!ok) {
    set_error("ANEInterop input write failed.");
    return -1;
  }
  return 0;
#else
  (void)handle;
  (void)values;
  (void)count;
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
#endif
}

extern "C" int dart_mlx_ane_interop_write_input_raw_f32(
    DartMlxAneInteropHandle* handle,
    const float* values,
    size_t count) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr || handle->value == nullptr) {
    set_error("ANEInterop handle is closed.");
    return -1;
  }
  const auto expected =
      static_cast<size_t>(handle->input_channels * handle->input_spatial);
  if (count != expected) {
    set_error("ANEInterop raw input element count mismatch.");
    return -1;
  }
  auto surface = ane_interop_get_input(handle->value, 0);
  if (!surface) {
    set_error("ANEInterop input surface is null.");
    return -1;
  }
  if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) {
    set_error("ANEInterop input surface lock failed.");
    return -1;
  }
  auto* base = static_cast<float*>(IOSurfaceGetBaseAddress(surface));
  if (base == nullptr) {
    IOSurfaceUnlock(surface, 0, NULL);
    set_error("ANEInterop input base address is null.");
    return -1;
  }
  std::memcpy(base, values, count * sizeof(float));
  IOSurfaceUnlock(surface, 0, NULL);
  return 0;
#else
  (void)handle;
  (void)values;
  (void)count;
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
#endif
}

extern "C" float* dart_mlx_ane_interop_read_output_f32_copy(
    DartMlxAneInteropHandle* handle,
    size_t* count_out) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr || handle->value == nullptr || count_out == nullptr) {
    set_error("ANEInterop output read arguments are invalid.");
    return nullptr;
  }
  const auto count = static_cast<size_t>(handle->output_channels * handle->output_spatial);
  auto* out = static_cast<float*>(std::malloc(count * sizeof(float)));
  if (out == nullptr) {
    set_error("ANEInterop output allocation failed.");
    return nullptr;
  }
  auto surface = ane_interop_copy_output(handle->value, 0);
  if (!surface) {
    std::free(out);
    set_error("ANEInterop output surface is null.");
    return nullptr;
  }
  const bool ok = ane_interop_io_read_fp16(
      surface,
      0,
      out,
      handle->output_channels,
      handle->output_spatial);
  CFRelease(surface);
  if (!ok) {
    std::free(out);
    set_error("ANEInterop output read failed.");
    return nullptr;
  }
  *count_out = count;
  return out;
#else
  (void)handle;
  (void)count_out;
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
#endif
}

extern "C" float* dart_mlx_ane_interop_read_output_raw_f32_copy(
    DartMlxAneInteropHandle* handle,
    size_t* count_out) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr || handle->value == nullptr || count_out == nullptr) {
    set_error("ANEInterop raw output read arguments are invalid.");
    return nullptr;
  }
  const auto count =
      static_cast<size_t>(handle->output_channels * handle->output_spatial);
  auto* out = static_cast<float*>(std::malloc(count * sizeof(float)));
  if (out == nullptr) {
    set_error("ANEInterop raw output allocation failed.");
    return nullptr;
  }
  auto surface = ane_interop_get_output(handle->value, 0);
  if (!surface) {
    std::free(out);
    set_error("ANEInterop output surface is null.");
    return nullptr;
  }
  if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
    std::free(out);
    set_error("ANEInterop output surface lock failed.");
    return nullptr;
  }
  auto* base = static_cast<const float*>(IOSurfaceGetBaseAddress(surface));
  if (base == nullptr) {
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    std::free(out);
    set_error("ANEInterop output base address is null.");
    return nullptr;
  }
  std::memcpy(out, base, count * sizeof(float));
  IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
  *count_out = count;
  return out;
#else
  (void)handle;
  (void)count_out;
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
#endif
}

extern "C" int dart_mlx_ane_interop_read_output_raw_f32(
    DartMlxAneInteropHandle* handle,
    float* out,
    size_t count) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr || handle->value == nullptr || out == nullptr) {
    set_error("ANEInterop raw output read arguments are invalid.");
    return -1;
  }
  const auto expected =
      static_cast<size_t>(handle->output_channels * handle->output_spatial);
  if (count != expected) {
    set_error("ANEInterop raw output element count mismatch.");
    return -1;
  }
  auto surface = ane_interop_get_output(handle->value, 0);
  if (!surface) {
    set_error("ANEInterop output surface is null.");
    return -1;
  }
  if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
    set_error("ANEInterop output surface lock failed.");
    return -1;
  }
  auto* base = static_cast<const float*>(IOSurfaceGetBaseAddress(surface));
  if (base == nullptr) {
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    set_error("ANEInterop output base address is null.");
    return -1;
  }
  std::memcpy(out, base, count * sizeof(float));
  IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
  return 0;
#else
  (void)handle;
  (void)out;
  (void)count;
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
#endif
}

extern "C" int64_t dart_mlx_ane_interop_last_hw_execution_time_ns(
    DartMlxAneInteropHandle* handle) {
  dart_mlx_ane_private_clear_error();
#ifdef DART_MLX_ENABLE_PRIVATE_ANE_BRIDGE
  if (handle == nullptr || handle->value == nullptr) {
    set_error("ANEInterop handle is closed.");
    return -1;
  }
  return static_cast<int64_t>(ane_interop_last_hw_execution_time_ns(handle->value));
#else
  (void)handle;
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
#endif
}
