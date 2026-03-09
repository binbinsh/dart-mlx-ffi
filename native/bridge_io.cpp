#include "bridge.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

struct BufferReader {
  std::vector<uint8_t> data;
  size_t pos;
  std::string label;
};

struct BufferWriter {
  std::vector<uint8_t> data;
  size_t pos;
  std::string label;
};

mlx_stream selected_stream(const DartMlxStreamHandle* stream) {
  return stream == nullptr ? default_cpu_stream() : stream->value;
}

bool io_is_open(void* desc) {
  return desc != nullptr;
}

bool reader_good(void* desc) {
  auto* reader = static_cast<BufferReader*>(desc);
  return reader->pos <= reader->data.size();
}

bool writer_good(void* desc) {
  return desc != nullptr;
}

size_t reader_tell(void* desc) {
  return static_cast<BufferReader*>(desc)->pos;
}

size_t writer_tell(void* desc) {
  return static_cast<BufferWriter*>(desc)->pos;
}

void reader_seek(void* desc, int64_t off, int whence) {
  auto* reader = static_cast<BufferReader*>(desc);
  int64_t base = 0;
  switch (whence) {
    case SEEK_CUR:
      base = static_cast<int64_t>(reader->pos);
      break;
    case SEEK_END:
      base = static_cast<int64_t>(reader->data.size());
      break;
    case SEEK_SET:
    default:
      break;
  }
  auto next = std::max<int64_t>(0, base + off);
  reader->pos = std::min<size_t>(static_cast<size_t>(next), reader->data.size());
}

void writer_seek(void* desc, int64_t off, int whence) {
  auto* writer = static_cast<BufferWriter*>(desc);
  int64_t base = 0;
  switch (whence) {
    case SEEK_CUR:
      base = static_cast<int64_t>(writer->pos);
      break;
    case SEEK_END:
      base = static_cast<int64_t>(writer->data.size());
      break;
    case SEEK_SET:
    default:
      break;
  }
  auto next = std::max<int64_t>(0, base + off);
  writer->pos = static_cast<size_t>(next);
  if (writer->pos > writer->data.size()) {
    writer->data.resize(writer->pos);
  }
}

void reader_read(void* desc, char* data, size_t n) {
  auto* reader = static_cast<BufferReader*>(desc);
  auto avail = std::min(n, reader->data.size() - reader->pos);
  std::memcpy(data, reader->data.data() + reader->pos, avail);
  if (avail < n) {
    std::memset(data + avail, 0, n - avail);
  }
  reader->pos += avail;
}

void reader_read_at_offset(void* desc, char* data, size_t n, size_t off) {
  auto* reader = static_cast<BufferReader*>(desc);
  auto avail = off < reader->data.size() ? std::min(n, reader->data.size() - off) : 0;
  if (avail > 0) {
    std::memcpy(data, reader->data.data() + off, avail);
  }
  if (avail < n) {
    std::memset(data + avail, 0, n - avail);
  }
}

void writer_read_unused(void*, char*, size_t) {}
void writer_read_at_offset_unused(void*, char*, size_t, size_t) {}
void reader_write_unused(void*, const char*, size_t) {}

void writer_write(void* desc, const char* data, size_t n) {
  auto* writer = static_cast<BufferWriter*>(desc);
  if (writer->pos + n > writer->data.size()) {
    writer->data.resize(writer->pos + n);
  }
  std::memcpy(writer->data.data() + writer->pos, data, n);
  writer->pos += n;
}

const char* reader_label(void* desc) {
  return static_cast<BufferReader*>(desc)->label.c_str();
}

const char* writer_label(void* desc) {
  return static_cast<BufferWriter*>(desc)->label.c_str();
}

void reader_free(void* desc) {
  delete static_cast<BufferReader*>(desc);
}

void writer_free(void* desc) {
  delete static_cast<BufferWriter*>(desc);
}

mlx_io_vtable reader_vtable() {
  return mlx_io_vtable{
      io_is_open,
      reader_good,
      reader_tell,
      reader_seek,
      reader_read,
      reader_read_at_offset,
      reader_write_unused,
      reader_label,
      reader_free,
  };
}

mlx_io_vtable writer_vtable() {
  return mlx_io_vtable{
      io_is_open,
      writer_good,
      writer_tell,
      writer_seek,
      writer_read_unused,
      writer_read_at_offset_unused,
      writer_write,
      writer_label,
      writer_free,
  };
}

int copy_writer_bytes(
    const BufferWriter* writer,
    uint8_t** bytes_out,
    size_t* len_out) {
  auto len = writer->data.size();
  auto* bytes = static_cast<uint8_t*>(std::malloc(len == 0 ? 1 : len));
  if (bytes == nullptr) {
    return 1;
  }
  if (len > 0) {
    std::memcpy(bytes, writer->data.data(), len);
  }
  *bytes_out = bytes;
  *len_out = len;
  return 0;
}

int export_safetensors_result(
    mlx_map_string_to_array arrays,
    mlx_map_string_to_string metadata,
    DartMlxArrayHandle*** arrays_out,
    char*** keys_out,
    size_t* arrays_len,
    char*** metadata_keys_out,
    char*** metadata_values_out,
    size_t* metadata_len) {
  size_t array_count = 0;
  {
    auto it = mlx_map_string_to_array_iterator_new(arrays);
    const char* key = nullptr;
    mlx_array value = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &value, it) == 0) {
      array_count++;
    }
    mlx_map_string_to_array_iterator_free(it);
  }

  auto** array_handles =
      static_cast<DartMlxArrayHandle**>(
          std::malloc(sizeof(DartMlxArrayHandle*) * array_count));
  auto** array_keys =
      static_cast<char**>(std::malloc(sizeof(char*) * array_count));
  if ((array_count > 0) &&
      (array_handles == nullptr || array_keys == nullptr)) {
    std::free(array_handles);
    std::free(array_keys);
    return 1;
  }

  {
    auto it = mlx_map_string_to_array_iterator_new(arrays);
    const char* key = nullptr;
    mlx_array value = mlx_array_new();
    size_t index = 0;
    while (mlx_map_string_to_array_iterator_next(&key, &value, it) == 0) {
      array_keys[index] = copy_c_string(key);
      array_handles[index] = wrap_array_copy(value);
      index++;
    }
    mlx_map_string_to_array_iterator_free(it);
  }

  size_t metadata_count = 0;
  {
    auto it = mlx_map_string_to_string_iterator_new(metadata);
    const char* key = nullptr;
    const char* value = nullptr;
    while (mlx_map_string_to_string_iterator_next(&key, &value, it) == 0) {
      metadata_count++;
    }
    mlx_map_string_to_string_iterator_free(it);
  }

  auto** meta_keys =
      static_cast<char**>(std::malloc(sizeof(char*) * metadata_count));
  auto** meta_values =
      static_cast<char**>(std::malloc(sizeof(char*) * metadata_count));
  if ((metadata_count > 0) && (meta_keys == nullptr || meta_values == nullptr)) {
    std::free(array_handles);
    std::free(array_keys);
    std::free(meta_keys);
    std::free(meta_values);
    return 1;
  }

  {
    auto it = mlx_map_string_to_string_iterator_new(metadata);
    const char* key = nullptr;
    const char* value = nullptr;
    size_t index = 0;
    while (mlx_map_string_to_string_iterator_next(&key, &value, it) == 0) {
      meta_keys[index] = copy_c_string(key);
      meta_values[index] = copy_c_string(value);
      index++;
    }
    mlx_map_string_to_string_iterator_free(it);
  }

  *arrays_out = array_handles;
  *keys_out = array_keys;
  *arrays_len = array_count;
  *metadata_keys_out = meta_keys;
  *metadata_values_out = meta_values;
  *metadata_len = metadata_count;
  return 0;
}

} // namespace

extern "C" DartMlxArrayHandle* dart_mlx_load(const char* file) {
  mlx_array out = mlx_array_new();
  if (mlx_load(&out, file, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_load_with_stream(
    const char* file,
    const DartMlxStreamHandle* stream) {
  mlx_array out = mlx_array_new();
  if (mlx_load(&out, file, selected_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_save(
    const char* file,
    const DartMlxArrayHandle* input) {
  return mlx_save(file, input->value);
}

extern "C" int dart_mlx_export_function(
    const char* file,
    const DartMlxClosureHandle* function,
    DartMlxArrayHandle** args,
    size_t args_len,
    bool shapeless) {
  auto values = build_array_vector(args, args_len);
  auto status = mlx_export_function(file, function->value, values, shapeless);
  mlx_vector_array_free(values);
  return status;
}

extern "C" int dart_mlx_export_kw_function(
    const char* file,
    const DartMlxKwHandle* function,
    DartMlxArrayHandle** args,
    size_t args_len,
    char** keys,
    DartMlxArrayHandle** values,
    size_t values_len,
    bool shapeless) {
  auto pos_args = build_array_vector(args, args_len);
  auto kwargs = mlx_map_string_to_array_new();
  for (size_t i = 0; i < values_len; ++i) {
    if (mlx_map_string_to_array_insert(kwargs, keys[i], values[i]->value) != 0) {
      mlx_vector_array_free(pos_args);
      mlx_map_string_to_array_free(kwargs);
      return 1;
    }
  }
  auto status =
      mlx_export_function_kwargs(file, function->value, pos_args, kwargs, shapeless);
  mlx_vector_array_free(pos_args);
  mlx_map_string_to_array_free(kwargs);
  return status;
}

extern "C" DartMlxExporterHandle* dart_mlx_function_exporter_new(
    const char* file,
    const DartMlxClosureHandle* function,
    bool shapeless) {
  return wrap_exporter(mlx_function_exporter_new(file, function->value, shapeless));
}

extern "C" void dart_mlx_function_exporter_free(DartMlxExporterHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_function_exporter_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_function_exporter_apply(
    const DartMlxExporterHandle* exporter,
    DartMlxArrayHandle** inputs,
    size_t input_len,
    char** keys,
    DartMlxArrayHandle** values,
    size_t values_len) {
  auto args = build_array_vector(inputs, input_len);
  auto kwargs = mlx_map_string_to_array_new();
  for (size_t i = 0; i < values_len; ++i) {
    if (mlx_map_string_to_array_insert(kwargs, keys[i], values[i]->value) != 0) {
      mlx_vector_array_free(args);
      mlx_map_string_to_array_free(kwargs);
      return 1;
    }
  }
  auto status = values_len == 0
                    ? mlx_function_exporter_apply(exporter->value, args)
                    : mlx_function_exporter_apply_kwargs(exporter->value, args, kwargs);
  mlx_vector_array_free(args);
  mlx_map_string_to_array_free(kwargs);
  return status;
}

extern "C" DartMlxImportedHandle* dart_mlx_imported_function_new(
    const char* file) {
  return wrap_imported(mlx_imported_function_new(file));
}

extern "C" void dart_mlx_imported_function_free(DartMlxImportedHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_imported_function_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_imported_function_apply(
    const DartMlxImportedHandle* function,
    DartMlxArrayHandle** inputs,
    size_t input_len,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto args = build_array_vector(inputs, input_len);
  auto results = mlx_vector_array_new();
  auto status = mlx_imported_function_apply(
      &results, function->value, args);
  mlx_vector_array_free(args);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(results, outputs_out, outputs_len_out);
  mlx_vector_array_free(results);
  return status;
}

extern "C" int dart_mlx_imported_function_apply_kwargs(
    const DartMlxImportedHandle* function,
    DartMlxArrayHandle** inputs,
    size_t input_len,
    char** keys,
    DartMlxArrayHandle** values,
    size_t values_len,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto args = build_array_vector(inputs, input_len);
  auto kwargs = mlx_map_string_to_array_new();
  for (size_t i = 0; i < values_len; ++i) {
    if (mlx_map_string_to_array_insert(kwargs, keys[i], values[i]->value) != 0) {
      mlx_vector_array_free(args);
      mlx_map_string_to_array_free(kwargs);
      return 1;
    }
  }
  auto results = mlx_vector_array_new();
  auto status =
      mlx_imported_function_apply_kwargs(&results, function->value, args, kwargs);
  mlx_vector_array_free(args);
  mlx_map_string_to_array_free(kwargs);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(results, outputs_out, outputs_len_out);
  mlx_vector_array_free(results);
  return status;
}

extern "C" DartMlxArrayHandle* dart_mlx_load_bytes(
    const uint8_t* bytes,
    size_t len,
    const DartMlxStreamHandle* stream) {
  auto reader = mlx_io_reader_new(
      new BufferReader{std::vector<uint8_t>(bytes, bytes + len), 0, "dart-bytes-reader"},
      reader_vtable());
  auto out = mlx_array_new();
  auto status = mlx_load_reader(&out, reader, selected_stream(stream));
  mlx_io_reader_free(reader);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_load_safetensors(
    const char* file,
    DartMlxArrayHandle*** arrays_out,
    char*** keys_out,
    size_t* arrays_len,
    char*** metadata_keys_out,
    char*** metadata_values_out,
    size_t* metadata_len) {
  mlx_map_string_to_array arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string metadata = mlx_map_string_to_string_new();

  if (mlx_load_safetensors(
          &arrays, &metadata, file, default_cpu_stream()) != 0) {
    mlx_map_string_to_array_free(arrays);
    mlx_map_string_to_string_free(metadata);
    return 1;
  }

  auto status = export_safetensors_result(
      arrays,
      metadata,
      arrays_out,
      keys_out,
      arrays_len,
      metadata_keys_out,
      metadata_values_out,
      metadata_len);
  mlx_map_string_to_array_free(arrays);
  mlx_map_string_to_string_free(metadata);
  return status;
}

extern "C" int dart_mlx_load_safetensors_with_stream(
    const char* file,
    const DartMlxStreamHandle* stream,
    DartMlxArrayHandle*** arrays_out,
    char*** keys_out,
    size_t* arrays_len,
    char*** metadata_keys_out,
    char*** metadata_values_out,
    size_t* metadata_len) {
  auto arrays = mlx_map_string_to_array_new();
  auto metadata = mlx_map_string_to_string_new();
  if (mlx_load_safetensors(
          &arrays, &metadata, file, selected_stream(stream)) != 0) {
    mlx_map_string_to_array_free(arrays);
    mlx_map_string_to_string_free(metadata);
    return 1;
  }
  auto status = export_safetensors_result(
      arrays,
      metadata,
      arrays_out,
      keys_out,
      arrays_len,
      metadata_keys_out,
      metadata_values_out,
      metadata_len);
  mlx_map_string_to_array_free(arrays);
  mlx_map_string_to_string_free(metadata);
  return status;
}

extern "C" int dart_mlx_load_safetensors_bytes(
    const uint8_t* bytes,
    size_t len,
    const DartMlxStreamHandle* stream,
    DartMlxArrayHandle*** arrays_out,
    char*** keys_out,
    size_t* arrays_len,
    char*** metadata_keys_out,
    char*** metadata_values_out,
    size_t* metadata_len) {
  auto reader = mlx_io_reader_new(
      new BufferReader{std::vector<uint8_t>(bytes, bytes + len), 0, "dart-bytes-reader"},
      reader_vtable());
  auto arrays = mlx_map_string_to_array_new();
  auto metadata = mlx_map_string_to_string_new();
  auto status = mlx_load_safetensors_reader(
      &arrays, &metadata, reader, selected_stream(stream));
  mlx_io_reader_free(reader);
  if (status != 0) {
    mlx_map_string_to_array_free(arrays);
    mlx_map_string_to_string_free(metadata);
    return status;
  }
  status = export_safetensors_result(
      arrays,
      metadata,
      arrays_out,
      keys_out,
      arrays_len,
      metadata_keys_out,
      metadata_values_out,
      metadata_len);
  mlx_map_string_to_array_free(arrays);
  mlx_map_string_to_string_free(metadata);
  return status;
}

extern "C" void dart_mlx_free_string_array(char** values, size_t len) {
  if (values == nullptr) {
    return;
  }
  for (size_t i = 0; i < len; ++i) {
    std::free(values[i]);
  }
  std::free(values);
}

extern "C" void dart_mlx_free_buffer(void* pointer) {
  std::free(pointer);
}

extern "C" int dart_mlx_save_bytes(
    const DartMlxArrayHandle* input,
    uint8_t** bytes_out,
    size_t* len_out) {
  auto* writer_desc = new BufferWriter{{}, 0, "dart-bytes-writer"};
  auto writer = mlx_io_writer_new(writer_desc, writer_vtable());
  auto status = mlx_save_writer(writer, input->value);
  if (status == 0) {
    status = copy_writer_bytes(writer_desc, bytes_out, len_out);
  }
  mlx_io_writer_free(writer);
  return status;
}

extern "C" int dart_mlx_save_safetensors(
    const char* file,
    DartMlxArrayHandle** arrays,
    char** keys,
    size_t arrays_len,
    char** metadata_keys,
    char** metadata_values,
    size_t metadata_len) {
  auto params = mlx_map_string_to_array_new();
  auto metadata = mlx_map_string_to_string_new();
  for (size_t i = 0; i < arrays_len; ++i) {
    if (mlx_map_string_to_array_insert(params, keys[i], arrays[i]->value) != 0) {
      mlx_map_string_to_array_free(params);
      mlx_map_string_to_string_free(metadata);
      return 1;
    }
  }
  for (size_t i = 0; i < metadata_len; ++i) {
    if (mlx_map_string_to_string_insert(
            metadata, metadata_keys[i], metadata_values[i]) != 0) {
      mlx_map_string_to_array_free(params);
      mlx_map_string_to_string_free(metadata);
      return 1;
    }
  }
  auto status = mlx_save_safetensors(file, params, metadata);
  mlx_map_string_to_array_free(params);
  mlx_map_string_to_string_free(metadata);
  return status;
}

extern "C" int dart_mlx_save_safetensors_bytes(
    DartMlxArrayHandle** arrays,
    char** keys,
    size_t arrays_len,
    char** metadata_keys,
    char** metadata_values,
    size_t metadata_len,
    uint8_t** bytes_out,
    size_t* len_out) {
  auto params = mlx_map_string_to_array_new();
  auto metadata = mlx_map_string_to_string_new();
  for (size_t i = 0; i < arrays_len; ++i) {
    if (mlx_map_string_to_array_insert(params, keys[i], arrays[i]->value) != 0) {
      mlx_map_string_to_array_free(params);
      mlx_map_string_to_string_free(metadata);
      return 1;
    }
  }
  for (size_t i = 0; i < metadata_len; ++i) {
    if (mlx_map_string_to_string_insert(
            metadata, metadata_keys[i], metadata_values[i]) != 0) {
      mlx_map_string_to_array_free(params);
      mlx_map_string_to_string_free(metadata);
      return 1;
    }
  }
  auto* writer_desc = new BufferWriter{{}, 0, "dart-bytes-writer"};
  auto writer = mlx_io_writer_new(writer_desc, writer_vtable());
  auto status = mlx_save_safetensors_writer(writer, params, metadata);
  if (status == 0) {
    status = copy_writer_bytes(writer_desc, bytes_out, len_out);
  }
  mlx_io_writer_free(writer);
  mlx_map_string_to_array_free(params);
  mlx_map_string_to_string_free(metadata);
  return status;
}
