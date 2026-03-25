#include "bridge.h"
#include "ane_p.h"

#include <string>

namespace {

constexpr const char* kAneStubProbe =
    "{\"compiled\":false,"
    "\"enabled\":false,"
    "\"framework_path\":\"/System/Library/PrivateFrameworks/"
    "AppleNeuralEngine.framework/AppleNeuralEngine\","
    "\"framework_loaded\":false,"
    "\"has_ane_client_class\":false,"
    "\"has_in_memory_model_descriptor_class\":false,"
    "\"has_in_memory_model_class\":false,"
    "\"has_request_class\":false,"
    "\"has_io_surface_object_class\":false,"
    "\"has_chaining_request_class\":false,"
    "\"has_performance_stats_class\":false,"
    "\"has_descriptor_factory\":false,"
    "\"has_in_memory_factory\":false,"
    "\"has_io_surface_factory\":false,"
    "\"has_request_factory\":false,"
    "\"has_model_compile\":false,"
    "\"has_model_load\":false,"
    "\"has_model_unload\":false,"
    "\"has_model_evaluate\":false,"
    "\"has_model_purge_compiled\":false,"
    "\"has_model_compiled_exists\":false,"
    "\"has_model_hex_identifier\":false,"
    "\"has_shared_connection\":false,"
    "\"has_model_accessor\":false,"
    "\"has_perf_mask_setter\":false,"
    "\"has_client_prepare_chaining\":false,"
    "\"has_client_begin_realtime_task\":false,"
    "\"has_client_end_realtime_task\":false,"
    "\"has_client_load_realtime_model\":false,"
    "\"has_client_unload_realtime_model\":false,"
    "\"has_client_evaluate_realtime\":false,"
    "\"has_client_virtual_client\":false,"
    "\"has_chaining_request_factory\":false,"
    "\"has_request_shared_events\":false,"
    "\"has_request_completion_handler\":false,"
    "\"has_perf_driver_mask\":false,"
    "\"supports_basic_eval\":false,"
    "\"supports_realtime_eval\":false,"
    "\"supports_chaining\":false,"
    "\"supports_perf_stats\":false}";
thread_local std::string g_last_ane_error;

void set_error(const char* message) {
  g_last_ane_error = message;
}

} // namespace

extern "C" bool dart_mlx_ane_private_is_compiled() {
  return false;
}

extern "C" bool dart_mlx_ane_private_is_enabled() {
  return false;
}

extern "C" void dart_mlx_ane_private_set_error_message(const char* message) {
  g_last_ane_error = message == nullptr ? "" : message;
}

extern "C" void dart_mlx_ane_private_clear_error() {
  g_last_ane_error.clear();
}

extern "C" char* dart_mlx_ane_private_last_error_copy() {
  if (g_last_ane_error.empty()) {
    return nullptr;
  }
  return copy_c_string(g_last_ane_error.c_str());
}

extern "C" char* dart_mlx_ane_private_probe_json_copy() {
  dart_mlx_ane_private_clear_error();
  return copy_c_string(kAneStubProbe);
}

extern "C" DartMlxAnePrivateModelHandle* dart_mlx_ane_private_model_new_mil(
    const char*,
    const char* const*,
    const uint8_t* const*,
    const size_t*,
    size_t) {
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
}

extern "C" void dart_mlx_ane_private_model_free(DartMlxAnePrivateModelHandle*) {}

extern "C" int dart_mlx_ane_private_model_compile(DartMlxAnePrivateModelHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_model_load(DartMlxAnePrivateModelHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_model_unload(DartMlxAnePrivateModelHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_model_is_loaded(
    const DartMlxAnePrivateModelHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_model_compiled_exists(
    const DartMlxAnePrivateModelHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" char* dart_mlx_ane_private_model_hex_identifier_copy(
    const DartMlxAnePrivateModelHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
}

extern "C" DartMlxAnePrivateSessionHandle* dart_mlx_ane_private_session_new(
    const DartMlxAnePrivateModelHandle*,
    const size_t*,
    size_t,
    const size_t*,
    size_t) {
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
}

extern "C" void dart_mlx_ane_private_session_free(DartMlxAnePrivateSessionHandle*) {}

extern "C" int dart_mlx_ane_private_session_write_input_bytes(
    DartMlxAnePrivateSessionHandle*,
    size_t,
    const uint8_t*,
    size_t) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_session_write_input_array_packed_f32(
    DartMlxAnePrivateSessionHandle*,
    size_t,
    const DartMlxArrayHandle*,
    size_t,
    size_t,
    size_t) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" uint8_t* dart_mlx_ane_private_session_read_output_bytes_copy(
    DartMlxAnePrivateSessionHandle*,
    size_t,
    size_t*) {
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
}

extern "C" int dart_mlx_ane_private_session_read_output_bytes(
    DartMlxAnePrivateSessionHandle*,
    size_t,
    uint8_t*,
    size_t) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_session_evaluate(
    DartMlxAnePrivateSessionHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_session_prepare_realtime(
    DartMlxAnePrivateSessionHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_session_teardown_realtime(
    DartMlxAnePrivateSessionHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_session_evaluate_realtime(
    DartMlxAnePrivateSessionHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_session_realtime_is_loaded(
    const DartMlxAnePrivateSessionHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" char* dart_mlx_ane_private_session_probe_chaining_json_copy(
    const DartMlxAnePrivateSessionHandle*,
    bool,
    bool,
    bool,
    bool,
    bool) {
  return copy_c_string(
      "{\"has_chaining_request_class\":false,"
      "\"has_prepare_selector\":false,"
      "\"has_output_sets_class\":false,"
      "\"has_output_sets_factory\":false,"
      "\"has_output_set_enqueue_class\":false,"
      "\"has_output_set_enqueue_factory\":false,"
      "\"has_input_buffers_ready_class\":false,"
      "\"has_input_buffers_ready_factory\":false,"
      "\"has_buffer_factory\":false,"
      "\"has_shared_signal_event_class\":false,"
      "\"has_iosurface_shared_event_class\":false,"
      "\"has_enqueue_sets_selector\":false,"
      "\"has_buffers_ready_selector\":false,"
      "\"built_output_set\":false,"
      "\"built_request\":false,"
      "\"built_output_set_enqueue\":false,"
      "\"built_input_buffers_ready\":false,"
      "\"built_shared_signal_event\":false,"
      "\"request_validated\":false,"
      "\"request_valid\":false,"
      "\"input_buffers_ready_validated\":false,"
      "\"input_buffers_ready_valid\":false,"
      "\"prepared\":false,"
      "\"called_enqueue_sets\":false,"
      "\"enqueue_sets_succeeded\":false,"
      "\"called_buffers_ready\":false,"
      "\"buffers_ready_succeeded\":false,"
      "\"stage\":0,"
      "\"error\":\"\"}");
}

extern "C" DartMlxAnePrivateChainHandle* dart_mlx_ane_private_chain_new(
    const DartMlxAnePrivateSessionHandle*,
    bool,
    bool,
    bool) {
  set_error("Private ANE bridge was excluded at build time.");
  return nullptr;
}

extern "C" void dart_mlx_ane_private_chain_free(DartMlxAnePrivateChainHandle*) {}

extern "C" int dart_mlx_ane_private_chain_is_prepared(
    const DartMlxAnePrivateChainHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_chain_has_enqueue_sets(
    const DartMlxAnePrivateChainHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_chain_has_buffers_ready(
    const DartMlxAnePrivateChainHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_chain_enqueue_sets(
    DartMlxAnePrivateChainHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}

extern "C" int dart_mlx_ane_private_chain_buffers_ready(
    DartMlxAnePrivateChainHandle*) {
  set_error("Private ANE bridge was excluded at build time.");
  return -1;
}
