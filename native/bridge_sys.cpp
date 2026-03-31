#include "bridge.h"

#include <sstream>

namespace {

std::string json_escape(const char* value) {
  if (value == nullptr) {
    return "";
  }
  std::string out;
  for (const auto* p = value; *p != '\0'; ++p) {
    switch (*p) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += *p;
        break;
    }
  }
  return out;
}

mlx_distributed_group maybe_group(const DartMlxGroupHandle* handle) {
  return handle == nullptr ? mlx_distributed_group() : handle->value;
}

mlx_stream maybe_stream(const DartMlxStreamHandle* handle) {
  return handle == nullptr ? default_cpu_stream() : handle->value;
}

} // namespace

extern "C" char* dart_mlx_device_info_json_copy(
    const DartMlxDeviceHandle* handle) {
  auto info = mlx_device_info_new();
  if (mlx_device_info_get(&info, handle->value) != 0) {
    return nullptr;
  }
  auto keys = mlx_vector_string_new();
  if (mlx_device_info_get_keys(&keys, info) != 0) {
    mlx_device_info_free(info);
    return nullptr;
  }

  std::ostringstream json;
  json << "{";
  auto len = mlx_vector_string_size(keys);
  for (size_t i = 0; i < len; ++i) {
    char* key = nullptr;
    if (mlx_vector_string_get(&key, keys, i) != 0) {
      mlx_vector_string_free(keys);
      mlx_device_info_free(info);
      return nullptr;
    }
    bool is_string = false;
    if (mlx_device_info_is_string(&is_string, info, key) != 0) {
      mlx_vector_string_free(keys);
      mlx_device_info_free(info);
      return nullptr;
    }
    if (i > 0) {
      json << ",";
    }
    json << "\"" << json_escape(key) << "\":";
    if (is_string) {
      const char* value = nullptr;
      if (mlx_device_info_get_string(&value, info, key) != 0) {
        mlx_vector_string_free(keys);
        mlx_device_info_free(info);
        return nullptr;
      }
      json << "\"" << json_escape(value) << "\"";
    } else {
      size_t value = 0;
      if (mlx_device_info_get_size(&value, info, key) != 0) {
        mlx_vector_string_free(keys);
        mlx_device_info_free(info);
        return nullptr;
      }
      json << value;
    }
  }
  json << "}";
  mlx_vector_string_free(keys);
  mlx_device_info_free(info);
  return copy_c_string(json.str().c_str());
}

extern "C" int dart_mlx_set_default_device(const DartMlxDeviceHandle* handle) {
  return mlx_set_default_device(handle->value);
}

extern "C" DartMlxStreamHandle* dart_mlx_stream_new() {
  return wrap_stream(mlx_stream_new());
}

extern "C" DartMlxStreamHandle* dart_mlx_stream_new_device(
    const DartMlxDeviceHandle* device) {
  return wrap_stream(mlx_stream_new_device(device->value));
}

extern "C" DartMlxStreamHandle* dart_mlx_get_default_stream(
    const DartMlxDeviceHandle* device) {
  auto stream = mlx_stream_new();
  if (mlx_get_default_stream(&stream, device->value) != 0) {
    return nullptr;
  }
  return wrap_stream(stream);
}

extern "C" DartMlxStreamHandle* dart_mlx_default_cpu_stream() {
  return wrap_stream(mlx_default_cpu_stream_new());
}

extern "C" DartMlxStreamHandle* dart_mlx_default_gpu_stream() {
  return wrap_stream(mlx_default_gpu_stream_new());
}

extern "C" void dart_mlx_stream_free(DartMlxStreamHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_stream_free(handle->value);
  delete handle;
}

extern "C" char* dart_mlx_stream_tostring_copy(const DartMlxStreamHandle* handle) {
  auto value = mlx_string_new();
  if (mlx_stream_tostring(&value, handle->value) != 0) {
    return nullptr;
  }
  auto* copy = copy_c_string(mlx_string_data(value));
  mlx_string_free(value);
  return copy;
}

extern "C" int dart_mlx_stream_get_index(const DartMlxStreamHandle* handle) {
  int index = 0;
  if (mlx_stream_get_index(&index, handle->value) != 0) {
    return -1;
  }
  return index;
}

extern "C" DartMlxDeviceHandle* dart_mlx_stream_get_device(
    const DartMlxStreamHandle* handle) {
  auto device = mlx_device_new();
  if (mlx_stream_get_device(&device, handle->value) != 0) {
    return nullptr;
  }
  return wrap_device(device);
}

extern "C" bool dart_mlx_stream_equal(
    const DartMlxStreamHandle* lhs,
    const DartMlxStreamHandle* rhs) {
  return mlx_stream_equal(lhs->value, rhs->value);
}

extern "C" int dart_mlx_stream_synchronize(const DartMlxStreamHandle* handle) {
  return mlx_synchronize(handle->value);
}

extern "C" int dart_mlx_set_default_stream(const DartMlxStreamHandle* handle) {
  return mlx_set_default_stream(handle->value);
}

extern "C" bool dart_mlx_distributed_is_available() {
  return mlx_distributed_is_available(nullptr);
}

extern "C" DartMlxGroupHandle* dart_mlx_distributed_init(bool strict) {
  return wrap_group(mlx_distributed_init(strict, nullptr));
}

extern "C" void dart_mlx_distributed_group_free(DartMlxGroupHandle* handle) {
  delete handle;
}

extern "C" int dart_mlx_distributed_group_rank(const DartMlxGroupHandle* handle) {
  return mlx_distributed_group_rank(handle->value);
}

extern "C" int dart_mlx_distributed_group_size(const DartMlxGroupHandle* handle) {
  return mlx_distributed_group_size(handle->value);
}

extern "C" DartMlxGroupHandle* dart_mlx_distributed_group_split(
    const DartMlxGroupHandle* group,
    int color,
    int key) {
  return wrap_group(mlx_distributed_group_split(group->value, color, key));
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_all_gather(
    const DartMlxArrayHandle* input,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_all_gather(
          &out, input->value, maybe_group(group), maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_all_sum(
    const DartMlxArrayHandle* input,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_all_sum(
          &out, input->value, maybe_group(group), maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_all_max(
    const DartMlxArrayHandle* input,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_all_max(
          &out, input->value, maybe_group(group), maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_all_min(
    const DartMlxArrayHandle* input,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_all_min(
          &out, input->value, maybe_group(group), maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_sum_scatter(
    const DartMlxArrayHandle* input,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_sum_scatter(
          &out, input->value, maybe_group(group), maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_send(
    const DartMlxArrayHandle* input,
    int dst,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_send(
          &out, input->value, dst, maybe_group(group), maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_recv_like(
    const DartMlxArrayHandle* like,
    int src,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_recv_like(
          &out, like->value, src, maybe_group(group), maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_distributed_recv(
    const int* shape,
    size_t shape_len,
    int dtype,
    int src,
    const DartMlxGroupHandle* group,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_distributed_recv(
          &out,
          shape,
          shape_len,
          as_dtype(dtype),
          src,
          maybe_group(group),
          maybe_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
