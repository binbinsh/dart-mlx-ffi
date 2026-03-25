#include "bridge.h"
#include "ane_p.h"

#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <sstream>
#include <string>

namespace {

constexpr const char* kAneFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
    "AppleNeuralEngine";
constexpr unsigned int kDefaultAneQos = 21;
thread_local std::string g_last_ane_error;

id as_id(void* value) {
  return reinterpret_cast<id>(value);
}

id cf_as_id(const void* value) {
  return reinterpret_cast<id>(const_cast<void*>(value));
}

void* as_void(id value) {
  return reinterpret_cast<void*>(value);
}

void clear_error() {
  g_last_ane_error.clear();
}

void set_error(std::string message) {
  g_last_ane_error = std::move(message);
}

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

void append_bool(
    std::ostringstream& json,
    bool& first,
    const char* key,
    bool value) {
  if (!first) {
    json << ",";
  }
  first = false;
  json << "\"" << key << "\":" << (value ? "true" : "false");
}

void append_string(
    std::ostringstream& json,
    bool& first,
    const char* key,
    const char* value) {
  if (!first) {
    json << ",";
  }
  first = false;
  json << "\"" << key << "\":\"" << json_escape(value) << "\"";
}

void* ane_framework_handle() {
  static void* handle = dlopen(kAneFrameworkPath, RTLD_NOW | RTLD_LOCAL);
  return handle;
}

bool framework_loaded() {
  return ane_framework_handle() != nullptr;
}

Class maybe_class(const char* name) {
  if (!framework_loaded()) {
    return nullptr;
  }
  return reinterpret_cast<Class>(objc_getClass(name));
}

bool has_class_method(Class cls, const char* selector_name) {
  if (cls == nullptr) {
    return false;
  }
  return class_getClassMethod(cls, sel_registerName(selector_name)) != nullptr;
}

bool has_instance_method(Class cls, const char* selector_name) {
  if (cls == nullptr) {
    return false;
  }
  return class_getInstanceMethod(cls, sel_registerName(selector_name)) != nullptr;
}

bool responds_to_selector(id value, const char* selector_name) {
  if (value == nil) {
    return false;
  }
  using Fn = bool (*)(id, SEL, SEL);
  return reinterpret_cast<Fn>(objc_msgSend)(
      value,
      sel_registerName("respondsToSelector:"),
      sel_registerName(selector_name));
}

std::string string_from_nsstring(id value) {
  if (value == nil || !responds_to_selector(value, "UTF8String")) {
    return "";
  }
  using Fn = const char* (*)(id, SEL);
  const auto* raw =
      reinterpret_cast<Fn>(objc_msgSend)(value, sel_registerName("UTF8String"));
  return raw == nullptr ? std::string{} : std::string(raw);
}

std::string description_from_object(id value) {
  if (value == nil) {
    return "";
  }
  if (responds_to_selector(value, "localizedDescription")) {
    using Fn = id (*)(id, SEL);
    return string_from_nsstring(reinterpret_cast<Fn>(objc_msgSend)(
        value,
        sel_registerName("localizedDescription")));
  }
  if (responds_to_selector(value, "description")) {
    using Fn = id (*)(id, SEL);
    return string_from_nsstring(reinterpret_cast<Fn>(objc_msgSend)(
        value,
        sel_registerName("description")));
  }
  return "";
}

id retain_object(id value) {
  return value == nil ? nil : cf_as_id(CFRetain(value));
}

void release_object(id value) {
  if (value != nil) {
    CFRelease(value);
  }
}

class ScopedAutoreleasePool {
 public:
  ScopedAutoreleasePool() {
    auto pool_class = reinterpret_cast<Class>(objc_getClass("NSAutoreleasePool"));
    if (pool_class == nullptr) {
      pool_ = nil;
      return;
    }
    using NewFn = id (*)(Class, SEL);
    pool_ = reinterpret_cast<NewFn>(objc_msgSend)(
        pool_class,
        sel_registerName("new"));
  }

  ~ScopedAutoreleasePool() {
    if (pool_ == nil) {
      return;
    }
    using DrainFn = void (*)(id, SEL);
    reinterpret_cast<DrainFn>(objc_msgSend)(pool_, sel_registerName("drain"));
  }

 private:
  id pool_;
};

CFDictionaryRef create_empty_dictionary() {
  return CFDictionaryCreate(
      kCFAllocatorDefault,
      nullptr,
      nullptr,
      0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
}

CFDictionaryRef create_weights_dictionary(
    const char* const* weight_paths,
    const uint8_t* const* weight_data,
    const size_t* weight_lens,
    size_t weight_count) {
  auto weights = CFDictionaryCreateMutable(
      kCFAllocatorDefault,
      0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  if (weights == nullptr) {
    set_error("Failed to allocate private ANE weights dictionary.");
    return nullptr;
  }

  for (size_t index = 0; index < weight_count; index++) {
    const auto* path = weight_paths[index];
    const auto* data = weight_data[index];
    const auto length = weight_lens[index];
    if (path == nullptr || path[0] == '\0') {
      CFRelease(weights);
      set_error("Private ANE weight paths must be non-empty.");
      return nullptr;
    }
    if (length > 0 && data == nullptr) {
      CFRelease(weights);
      set_error("Private ANE weight data cannot be null when length is non-zero.");
      return nullptr;
    }

    auto path_string =
        CFStringCreateWithCString(kCFAllocatorDefault, path, kCFStringEncodingUTF8);
    if (path_string == nullptr) {
      CFRelease(weights);
      set_error("Failed to convert a private ANE weight path to UTF-8.");
      return nullptr;
    }
    if (CFDictionaryContainsKey(weights, path_string)) {
      CFRelease(path_string);
      CFRelease(weights);
      set_error("Duplicate private ANE weight path: " + std::string(path));
      return nullptr;
    }

    auto blob = CFDataCreate(
        kCFAllocatorDefault,
        data,
        static_cast<CFIndex>(length));
    if (blob == nullptr) {
      CFRelease(path_string);
      CFRelease(weights);
      set_error("Failed to allocate a private ANE weight blob.");
      return nullptr;
    }

    int64_t zero = 0;
    auto offset =
        CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &zero);
    if (offset == nullptr) {
      CFRelease(blob);
      CFRelease(path_string);
      CFRelease(weights);
      set_error("Failed to allocate a private ANE weight offset.");
      return nullptr;
    }

    const void* entry_keys[] = {CFSTR("offset"), CFSTR("data")};
    const void* entry_values[] = {offset, blob};
    auto entry = CFDictionaryCreate(
        kCFAllocatorDefault,
        entry_keys,
        entry_values,
        2,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    CFRelease(offset);
    CFRelease(blob);
    if (entry == nullptr) {
      CFRelease(path_string);
      CFRelease(weights);
      set_error("Failed to allocate a private ANE weight entry.");
      return nullptr;
    }

    CFDictionarySetValue(weights, path_string, entry);
    CFRelease(entry);
    CFRelease(path_string);
  }

  return weights;
}

bool ensure_model_lifecycle_available() {
  if (!framework_loaded()) {
    set_error("AppleNeuralEngine private framework is unavailable.");
    return false;
  }
  const auto ane_desc = maybe_class("_ANEInMemoryModelDescriptor");
  const auto ane_in_memory = maybe_class("_ANEInMemoryModel");
  if (ane_desc == nullptr || ane_in_memory == nullptr) {
    set_error("Private ANE model classes are unavailable.");
    return false;
  }
  if (!has_class_method(ane_desc, "modelWithMILText:weights:optionsPlist:")) {
    set_error("Private ANE descriptor factory is unavailable.");
    return false;
  }
  if (!has_class_method(ane_in_memory, "inMemoryModelWithDescriptor:")) {
    set_error("Private ANE in-memory model factory is unavailable.");
    return false;
  }
  return true;
}

bool ensure_model_operation_available(
    DartMlxAnePrivateModelHandle* handle,
    const char* selector_name,
    const char* operation_name) {
  if (handle == nullptr || handle->model == nullptr) {
    set_error(std::string(operation_name) + " requires an open private ANE model.");
    return false;
  }
  if (!responds_to_selector(as_id(handle->model), selector_name)) {
    set_error(
        std::string("Private ANE selector is unavailable for ") + operation_name +
        ": " + selector_name);
    return false;
  }
  return true;
}

int call_model_operation(
    DartMlxAnePrivateModelHandle* handle,
    const char* selector_name,
    const char* operation_name) {
  clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (!ensure_model_operation_available(handle, selector_name, operation_name)) {
    return -1;
  }

  auto options = create_empty_dictionary();
  if (options == nullptr) {
    set_error(
        std::string("Failed to allocate options for private ANE ") +
        operation_name +
        ".");
    return -1;
  }

  id error = nil;
  using Fn = bool (*)(id, SEL, unsigned int, id, id*);
  const bool ok = reinterpret_cast<Fn>(objc_msgSend)(
      as_id(handle->model),
      sel_registerName(selector_name),
      kDefaultAneQos,
      cf_as_id(options),
      &error);
  CFRelease(options);
  if (!ok) {
    auto message = description_from_object(error);
    if (message.empty()) {
      message = std::string("Private ANE ") + operation_name + " failed.";
    }
    set_error(std::move(message));
    return -1;
  }
  return 0;
}

} // namespace

extern "C" bool dart_mlx_ane_private_is_compiled() {
  return true;
}

extern "C" bool dart_mlx_ane_private_is_enabled() {
  return true;
}

extern "C" void dart_mlx_ane_private_set_error_message(const char* message) {
  g_last_ane_error = message == nullptr ? "" : message;
}

extern "C" void dart_mlx_ane_private_clear_error() {
  clear_error();
}

extern "C" char* dart_mlx_ane_private_last_error_copy() {
  if (g_last_ane_error.empty()) {
    return nullptr;
  }
  return copy_c_string(g_last_ane_error.c_str());
}

extern "C" char* dart_mlx_ane_private_probe_json_copy() {
  clear_error();
  const bool loaded = framework_loaded();
  const auto ane_client = maybe_class("_ANEClient");
  const auto ane_desc = maybe_class("_ANEInMemoryModelDescriptor");
  const auto ane_in_memory = maybe_class("_ANEInMemoryModel");
  const auto ane_request = maybe_class("_ANERequest");
  const auto ane_io_surface = maybe_class("_ANEIOSurfaceObject");
  const auto ane_chaining_request = maybe_class("_ANEChainingRequest");
  const auto ane_perf_stats = maybe_class("_ANEPerformanceStats");

  const bool has_descriptor_factory =
      has_class_method(ane_desc, "modelWithMILText:weights:optionsPlist:");
  const bool has_in_memory_factory =
      has_class_method(ane_in_memory, "inMemoryModelWithDescriptor:");
  const bool has_io_surface_factory =
      has_class_method(ane_io_surface, "objectWithIOSurface:");
  const bool has_request_factory = has_class_method(
      ane_request,
      "requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:");
  const bool has_model_compile =
      has_instance_method(ane_in_memory, "compileWithQoS:options:error:");
  const bool has_model_load =
      has_instance_method(ane_in_memory, "loadWithQoS:options:error:");
  const bool has_model_unload =
      has_instance_method(ane_in_memory, "unloadWithQoS:error:");
  const bool has_model_evaluate =
      has_instance_method(ane_in_memory, "evaluateWithQoS:options:request:error:");
  const bool has_model_purge_compiled =
      has_instance_method(ane_in_memory, "purgeCompiledModel");
  const bool has_model_compiled_exists =
      has_instance_method(ane_in_memory, "compiledModelExists");
  const bool has_model_hex_identifier =
      has_instance_method(ane_in_memory, "hexStringIdentifier");
  const bool has_shared_connection =
      has_instance_method(ane_in_memory, "sharedConnection");
  const bool has_model_accessor = has_instance_method(ane_in_memory, "model");
  const bool has_perf_mask_setter =
      has_instance_method(ane_in_memory, "setPerfStatsMask:");
  const bool has_client_prepare_chaining = has_instance_method(
      ane_client,
      "prepareChainingWithModel:options:chainingReq:qos:error:");
  const bool has_client_begin_realtime_task =
      has_instance_method(ane_client, "beginRealTimeTask");
  const bool has_client_end_realtime_task =
      has_instance_method(ane_client, "endRealTimeTask");
  const bool has_client_load_realtime_model = has_instance_method(
      ane_client,
      "loadRealTimeModel:options:qos:error:");
  const bool has_client_unload_realtime_model = has_instance_method(
      ane_client,
      "unloadRealTimeModel:options:qos:error:");
  const bool has_client_evaluate_realtime = has_instance_method(
      ane_client,
      "evaluateRealTimeWithModel:options:request:error:");
  const bool has_client_virtual_client =
      has_instance_method(ane_client, "virtualClient");
  const bool has_chaining_request_factory = has_class_method(
      ane_chaining_request,
      "chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:"
      "procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:");
  const bool has_request_shared_events =
      has_instance_method(ane_request, "setSharedEvents:");
  const bool has_request_completion_handler =
      has_instance_method(ane_request, "setCompletionHandler:");
  const bool has_perf_driver_mask =
      has_class_method(ane_perf_stats, "driverMaskForANEFMask:");

  const bool supports_basic_eval =
      loaded && ane_desc != nullptr && ane_in_memory != nullptr &&
      ane_request != nullptr && ane_io_surface != nullptr &&
      has_descriptor_factory && has_in_memory_factory &&
      has_io_surface_factory && has_request_factory && has_model_compile &&
      has_model_load && has_model_unload && has_model_evaluate;

  const bool supports_realtime_eval =
      supports_basic_eval && ane_client != nullptr && has_shared_connection &&
      has_model_accessor && has_client_begin_realtime_task &&
      has_client_end_realtime_task && has_client_load_realtime_model &&
      has_client_unload_realtime_model && has_client_evaluate_realtime;

  const bool supports_chaining =
      supports_basic_eval && ane_client != nullptr &&
      ane_chaining_request != nullptr && has_client_prepare_chaining &&
      has_chaining_request_factory && has_request_shared_events;

  const bool supports_perf_stats =
      supports_basic_eval && ane_perf_stats != nullptr && has_perf_mask_setter &&
      has_perf_driver_mask;

  std::ostringstream json;
  bool first = true;
  json << "{";
  append_bool(json, first, "compiled", true);
  append_bool(json, first, "enabled", true);
  append_string(json, first, "framework_path", kAneFrameworkPath);
  append_bool(json, first, "framework_loaded", loaded);
  append_bool(json, first, "has_ane_client_class", ane_client != nullptr);
  append_bool(
      json,
      first,
      "has_in_memory_model_descriptor_class",
      ane_desc != nullptr);
  append_bool(json, first, "has_in_memory_model_class", ane_in_memory != nullptr);
  append_bool(json, first, "has_request_class", ane_request != nullptr);
  append_bool(
      json,
      first,
      "has_io_surface_object_class",
      ane_io_surface != nullptr);
  append_bool(
      json,
      first,
      "has_chaining_request_class",
      ane_chaining_request != nullptr);
  append_bool(json, first, "has_performance_stats_class", ane_perf_stats != nullptr);
  append_bool(json, first, "has_descriptor_factory", has_descriptor_factory);
  append_bool(json, first, "has_in_memory_factory", has_in_memory_factory);
  append_bool(json, first, "has_io_surface_factory", has_io_surface_factory);
  append_bool(json, first, "has_request_factory", has_request_factory);
  append_bool(json, first, "has_model_compile", has_model_compile);
  append_bool(json, first, "has_model_load", has_model_load);
  append_bool(json, first, "has_model_unload", has_model_unload);
  append_bool(json, first, "has_model_evaluate", has_model_evaluate);
  append_bool(json, first, "has_model_purge_compiled", has_model_purge_compiled);
  append_bool(
      json,
      first,
      "has_model_compiled_exists",
      has_model_compiled_exists);
  append_bool(
      json,
      first,
      "has_model_hex_identifier",
      has_model_hex_identifier);
  append_bool(json, first, "has_shared_connection", has_shared_connection);
  append_bool(json, first, "has_model_accessor", has_model_accessor);
  append_bool(json, first, "has_perf_mask_setter", has_perf_mask_setter);
  append_bool(
      json,
      first,
      "has_client_prepare_chaining",
      has_client_prepare_chaining);
  append_bool(
      json,
      first,
      "has_client_begin_realtime_task",
      has_client_begin_realtime_task);
  append_bool(
      json,
      first,
      "has_client_end_realtime_task",
      has_client_end_realtime_task);
  append_bool(
      json,
      first,
      "has_client_load_realtime_model",
      has_client_load_realtime_model);
  append_bool(
      json,
      first,
      "has_client_unload_realtime_model",
      has_client_unload_realtime_model);
  append_bool(
      json,
      first,
      "has_client_evaluate_realtime",
      has_client_evaluate_realtime);
  append_bool(
      json,
      first,
      "has_client_virtual_client",
      has_client_virtual_client);
  append_bool(
      json,
      first,
      "has_chaining_request_factory",
      has_chaining_request_factory);
  append_bool(
      json,
      first,
      "has_request_shared_events",
      has_request_shared_events);
  append_bool(
      json,
      first,
      "has_request_completion_handler",
      has_request_completion_handler);
  append_bool(json, first, "has_perf_driver_mask", has_perf_driver_mask);
  append_bool(json, first, "supports_basic_eval", supports_basic_eval);
  append_bool(json, first, "supports_realtime_eval", supports_realtime_eval);
  append_bool(json, first, "supports_chaining", supports_chaining);
  append_bool(json, first, "supports_perf_stats", supports_perf_stats);
  json << "}";
  return copy_c_string(json.str().c_str());
}
