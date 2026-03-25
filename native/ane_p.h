#pragma once

#include <IOSurface/IOSurface.h>

#include <cstddef>
#include <string>
#include <vector>

struct DartMlxAnePrivateWeight {
  std::string path;
  std::vector<uint8_t> data;
  size_t offset;
};

struct DartMlxAnePrivateModelHandle {
  void* model;
  void* eval_options;
  bool loaded;
  std::string mil_text;
  std::vector<DartMlxAnePrivateWeight> weights;
  std::string temp_dir;
};

struct DartMlxAnePrivateSessionHandle {
  void* model;
  void* eval_options;
  void* request;
  void* client;
  void* client_model;
  IOSurfaceRef* inputs;
  IOSurfaceRef* outputs;
  size_t* input_bytes;
  size_t* output_bytes;
  size_t input_count;
  size_t output_count;
  bool realtime_loaded;
};

struct DartMlxAnePrivateChainHandle {
  void* client;
  void* client_model;
  void* chaining_request;
  void* output_set_enqueue;
  void* input_buffers_ready;
  bool prepared;
  bool has_output_set_enqueue;
  bool has_input_buffers_ready;
};

extern "C" void dart_mlx_ane_private_clear_error();
extern "C" void dart_mlx_ane_private_set_error_message(const char* message);
