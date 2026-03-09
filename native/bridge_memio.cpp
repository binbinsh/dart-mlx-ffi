#include "bridge.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

struct BufferReader {
  mlx_io_reader value;
  std::vector<uint8_t> data;
  size_t pos;
};

struct BufferWriter {
  mlx_io_writer value;
  std::vector<uint8_t> data;
  size_t pos;
};

struct DartMlxReaderHandle {
  BufferReader* reader;
};

struct DartMlxWriterHandle {
  BufferWriter* writer;
};

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
  return "dart-bytes-reader";
}

const char* writer_label(void* desc) {
  return "dart-bytes-writer";
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

mlx_stream selected_stream(const DartMlxStreamHandle* stream) {
  return stream == nullptr ? default_cpu_stream() : stream->value;
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

} // namespace

extern "C" DartMlxReaderHandle* dart_mlx_bytes_reader_new(
    const uint8_t* bytes,
    size_t len,
    const char*) {
  auto* reader = new BufferReader{
      mlx_io_reader(),
      std::vector<uint8_t>(bytes, bytes + len),
      0,
  };
  reader->value = mlx_io_reader_new(reader, reader_vtable());
  if (reader->value.ctx == nullptr) {
    delete reader;
    return nullptr;
  }
  auto* handle = new DartMlxReaderHandle{reader};
  return handle;
}

extern "C" void dart_mlx_io_reader_free(DartMlxReaderHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_io_reader_free(handle->reader->value);
  delete handle;
}

extern "C" char* dart_mlx_io_reader_tostring_copy(const DartMlxReaderHandle* handle) {
  auto value = mlx_string_new();
  if (mlx_io_reader_tostring(&value, handle->reader->value) != 0) {
    return nullptr;
  }
  auto* copy = copy_c_string(mlx_string_data(value));
  mlx_string_free(value);
  return copy;
}

extern "C" void dart_mlx_io_reader_rewind(DartMlxReaderHandle* handle) {
  handle->reader->pos = 0;
}

extern "C" DartMlxArrayHandle* dart_mlx_load_reader_handle(
    const DartMlxReaderHandle* reader,
    const DartMlxStreamHandle* stream) {
  auto out = mlx_array_new();
  if (mlx_load_reader(&out, reader->reader->value, selected_stream(stream)) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_load_safetensors_reader_handle(
    const DartMlxReaderHandle* reader,
    const DartMlxStreamHandle* stream,
    DartMlxArrayHandle*** arrays_out,
    char*** keys_out,
    size_t* arrays_len,
    char*** metadata_keys_out,
    char*** metadata_values_out,
    size_t* metadata_len) {
  auto arrays = mlx_map_string_to_array_new();
  auto metadata = mlx_map_string_to_string_new();
  auto status = mlx_load_safetensors_reader(
      &arrays,
      &metadata,
      reader->reader->value,
      selected_stream(stream));
  if (status != 0) {
    mlx_map_string_to_array_free(arrays);
    mlx_map_string_to_string_free(metadata);
    return status;
  }
  // Reuse bridge_io.cpp helper behavior inline.
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
  auto** array_handles = static_cast<DartMlxArrayHandle**>(
      std::malloc(sizeof(DartMlxArrayHandle*) * array_count));
  auto** array_keys =
      static_cast<char**>(std::malloc(sizeof(char*) * array_count));
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
  size_t meta_count = 0;
  {
    auto it = mlx_map_string_to_string_iterator_new(metadata);
    const char* key = nullptr;
    const char* value = nullptr;
    while (mlx_map_string_to_string_iterator_next(&key, &value, it) == 0) {
      meta_count++;
    }
    mlx_map_string_to_string_iterator_free(it);
  }
  auto** meta_keys = static_cast<char**>(std::malloc(sizeof(char*) * meta_count));
  auto** meta_values = static_cast<char**>(std::malloc(sizeof(char*) * meta_count));
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
  *metadata_len = meta_count;
  mlx_map_string_to_array_free(arrays);
  mlx_map_string_to_string_free(metadata);
  return 0;
}

extern "C" DartMlxWriterHandle* dart_mlx_bytes_writer_new(const char*) {
  auto* writer = new BufferWriter{mlx_io_writer(), {}, 0};
  writer->value = mlx_io_writer_new(writer, writer_vtable());
  if (writer->value.ctx == nullptr) {
    delete writer;
    return nullptr;
  }
  return new DartMlxWriterHandle{writer};
}

extern "C" void dart_mlx_io_writer_free(DartMlxWriterHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_io_writer_free(handle->writer->value);
  delete handle;
}

extern "C" char* dart_mlx_io_writer_tostring_copy(const DartMlxWriterHandle* handle) {
  auto value = mlx_string_new();
  if (mlx_io_writer_tostring(&value, handle->writer->value) != 0) {
    return nullptr;
  }
  auto* copy = copy_c_string(mlx_string_data(value));
  mlx_string_free(value);
  return copy;
}

extern "C" void dart_mlx_io_writer_rewind(DartMlxWriterHandle* handle) {
  handle->writer->pos = 0;
}

extern "C" int dart_mlx_io_writer_bytes_copy(
    const DartMlxWriterHandle* writer,
    uint8_t** bytes_out,
    size_t* len_out) {
  return copy_writer_bytes(writer->writer, bytes_out, len_out);
}

extern "C" int dart_mlx_save_writer_handle(
    const DartMlxWriterHandle* writer,
    const DartMlxArrayHandle* input) {
  return mlx_save_writer(writer->writer->value, input->value);
}

extern "C" int dart_mlx_save_safetensors_writer_handle(
    const DartMlxWriterHandle* writer,
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
  auto status =
      mlx_save_safetensors_writer(writer->writer->value, params, metadata);
  mlx_map_string_to_array_free(params);
  mlx_map_string_to_string_free(metadata);
  return status;
}
