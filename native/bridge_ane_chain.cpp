#include "bridge.h"
#include "ane_p.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <sstream>
#include <string>

namespace {

constexpr unsigned int kDefaultAneQos = 21;

id as_id(void* value) { return reinterpret_cast<id>(value); }
id cf_as_id(const void* value) { return reinterpret_cast<id>(const_cast<void*>(value)); }

bool responds_to_selector(id value, const char* selector_name) {
  if (value == nil) return false;
  using Fn = bool (*)(id, SEL, SEL);
  return reinterpret_cast<Fn>(objc_msgSend)(
      value,
      sel_registerName("respondsToSelector:"),
      sel_registerName(selector_name));
}

bool has_class_method(Class cls, const char* selector_name) {
  return cls != nullptr &&
      class_getClassMethod(cls, sel_registerName(selector_name)) != nullptr;
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
    if (pool_ == nil) return;
    using DrainFn = void (*)(id, SEL);
    reinterpret_cast<DrainFn>(objc_msgSend)(pool_, sel_registerName("drain"));
  }

 private:
  id pool_;
};

std::string string_from_nsstring(id value) {
  if (value == nil || !responds_to_selector(value, "UTF8String")) return "";
  using Fn = const char* (*)(id, SEL);
  const auto* raw =
      reinterpret_cast<Fn>(objc_msgSend)(value, sel_registerName("UTF8String"));
  return raw == nullptr ? std::string{} : std::string(raw);
}

std::string description_from_object(id value) {
  if (value == nil) return "";
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

std::string json_escape(const char* value) {
  if (value == nullptr) return "";
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
  if (!first) json << ",";
  first = false;
  json << "\"" << key << "\":" << (value ? "true" : "false");
}

void append_int(
    std::ostringstream& json,
    bool& first,
    const char* key,
    int value) {
  if (!first) json << ",";
  first = false;
  json << "\"" << key << "\":" << value;
}

void append_string(
    std::ostringstream& json,
    bool& first,
    const char* key,
    const std::string& value) {
  if (!first) json << ",";
  first = false;
  json << "\"" << key << "\":\"" << json_escape(value.c_str()) << "\"";
}

CFMutableArrayRef create_cf_array() {
  return CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
}

CFNumberRef create_cf_index(size_t value) {
  auto signed_value = static_cast<int64_t>(value);
  return CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &signed_value);
}

CFDictionaryRef create_empty_dictionary() {
  return CFDictionaryCreate(
      kCFAllocatorDefault,
      nullptr,
      nullptr,
      0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
}

IOSurfaceRef create_surface(size_t bytes) {
  int64_t width = static_cast<int64_t>(bytes);
  int64_t one = 1;
  int64_t pixel_format = 0;
  auto width_number =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &width);
  auto height_number =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &one);
  auto bpe_number =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &one);
  auto bpr_number =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &width);
  auto alloc_number =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &width);
  auto pixel_number =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &pixel_format);
  if (width_number == nullptr || height_number == nullptr || bpe_number == nullptr ||
      bpr_number == nullptr || alloc_number == nullptr || pixel_number == nullptr) {
    if (width_number != nullptr) CFRelease(width_number);
    if (height_number != nullptr) CFRelease(height_number);
    if (bpe_number != nullptr) CFRelease(bpe_number);
    if (bpr_number != nullptr) CFRelease(bpr_number);
    if (alloc_number != nullptr) CFRelease(alloc_number);
    if (pixel_number != nullptr) CFRelease(pixel_number);
    return nullptr;
  }
  const void* keys[] = {
      kIOSurfaceWidth,
      kIOSurfaceHeight,
      kIOSurfaceBytesPerElement,
      kIOSurfaceBytesPerRow,
      kIOSurfaceAllocSize,
      kIOSurfacePixelFormat,
  };
  const void* values[] = {
      width_number,
      height_number,
      bpe_number,
      bpr_number,
      alloc_number,
      pixel_number,
  };
  auto config = CFDictionaryCreate(
      kCFAllocatorDefault,
      keys,
      values,
      6,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFRelease(width_number);
  CFRelease(height_number);
  CFRelease(bpe_number);
  CFRelease(bpr_number);
  CFRelease(alloc_number);
  CFRelease(pixel_number);
  if (config == nullptr) return nullptr;
  auto surface = IOSurfaceCreate(config);
  CFRelease(config);
  return surface;
}

enum ChainingStage {
  unavailable = 0,
  output_sets_build_failed = 1,
  request_build_failed = 2,
  prepare_failed = 3,
  prepare_succeeded = 4,
  exception = 5,
  prepare_skipped = 6,
  output_set_enqueue_build_failed = 7,
  input_buffers_ready_build_failed = 8,
  input_buffers_ready_validate_failed = 10,
  input_buffers_ready_call_failed = 11,
  input_buffers_ready_call_succeeded = 12,
  enqueue_sets_call_failed = 13,
  enqueue_sets_call_succeeded = 14,
  signal_event_build_failed = 15,
};

} // namespace

extern "C" char* dart_mlx_ane_private_session_probe_chaining_json_copy(
    const DartMlxAnePrivateSessionHandle* handle,
    bool validate_request,
    bool use_shared_signal_event,
    bool attempt_prepare,
    bool call_enqueue_sets,
    bool call_buffers_ready) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  std::ostringstream json;
  bool first = true;

  auto client_cls = reinterpret_cast<Class>(objc_getClass("_ANEClient"));
  auto chaining_cls = reinterpret_cast<Class>(objc_getClass("_ANEChainingRequest"));
  auto buffer_cls = reinterpret_cast<Class>(objc_getClass("_ANEBuffer"));
  auto output_sets_cls = reinterpret_cast<Class>(objc_getClass("_ANEIOSurfaceOutputSets"));
  auto output_set_enqueue_cls = reinterpret_cast<Class>(objc_getClass("_ANEOutputSetEnqueue"));
  auto input_buffers_ready_cls = reinterpret_cast<Class>(objc_getClass("_ANEInputBuffersReady"));
  auto shared_signal_cls = reinterpret_cast<Class>(objc_getClass("_ANESharedSignalEvent"));
  auto io_shared_event_cls = reinterpret_cast<Class>(objc_getClass("IOSurfaceSharedEvent"));
  auto io_cls = reinterpret_cast<Class>(objc_getClass("_ANEIOSurfaceObject"));

  const bool has_prepare_selector =
      client_cls != nullptr &&
      class_getInstanceMethod(
          client_cls,
          sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:")) != nullptr;
  const bool has_chaining_factory = has_class_method(
      chaining_cls,
      "chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:"
      "procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:");
  const bool has_buffer_factory =
      has_class_method(buffer_cls, "bufferWithIOSurfaceObject:symbolIndex:source:");
  const bool has_output_sets_factory =
      has_class_method(output_sets_cls, "objectWithstatsSurRef:outputBuffer:");
  const bool has_output_set_enqueue_factory = has_class_method(
      output_set_enqueue_cls,
      "outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:");
  const bool has_input_buffers_ready_factory = has_class_method(
      input_buffers_ready_cls,
      "inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:");
  const bool has_shared_signal_factory = has_class_method(
      shared_signal_cls,
      "signalEventWithValue:symbolIndex:eventType:sharedEvent:");
  const bool has_enqueue_sets_selector =
      client_cls != nullptr &&
      class_getInstanceMethod(
          client_cls,
          sel_registerName("enqueueSetsWithModel:outputSet:options:qos:error:")) != nullptr;
  const bool has_buffers_ready_selector =
      client_cls != nullptr &&
      class_getInstanceMethod(
          client_cls,
          sel_registerName("buffersReadyWithModel:inputBuffers:options:qos:error:")) != nullptr;

  bool built_output_set = false;
  bool built_request = false;
  bool built_output_set_enqueue = false;
  bool built_input_buffers_ready = false;
  bool request_validated = false;
  bool request_valid = false;
  bool built_shared_signal = false;
  bool prepared = false;
  bool input_buffers_ready_validated = false;
  bool input_buffers_ready_valid = false;
  bool called_enqueue_sets = false;
  bool enqueue_sets_succeeded = false;
  bool called_buffers_ready = false;
  bool buffers_ready_succeeded = false;
  int stage = ChainingStage::unavailable;
  std::string error_message;

  if (handle != nullptr && handle->model != nullptr && handle->request != nullptr &&
      handle->output_count > 0 && has_prepare_selector && has_chaining_factory &&
      has_output_sets_factory && io_cls != nullptr) {
    using GetterFn = id (*)(id, SEL);
    auto client = reinterpret_cast<GetterFn>(objc_msgSend)(
        as_id(handle->model),
        sel_registerName("sharedConnection"));
    auto client_model = reinterpret_cast<GetterFn>(objc_msgSend)(
        as_id(handle->model),
        sel_registerName("model"));

    if (client != nil && client_model != nil) {
      auto wrapped_inputs = create_cf_array();
      auto wrapped_outputs = create_cf_array();
      auto loopback_inputs = create_cf_array();
      auto loopback_outputs = create_cf_array();
      auto output_buffer_values = create_cf_array();
      auto output_sets = create_cf_array();
      auto signal_events = create_cf_array();
      auto input_buffer_info_index = create_cf_array();
      auto input_free_value = create_cf_array();
      IOSurfaceRef stats_surface = create_surface(256);
      id shared_signal = nil;
      id chaining_request = nil;
      id output_set = nil;
      id output_set_enqueue = nil;
      id input_buffers_ready = nil;

      for (size_t index = 0; index < handle->input_count; index++) {
        using WrapFn = id (*)(Class, SEL, IOSurfaceRef);
        auto wrapped = reinterpret_cast<WrapFn>(objc_msgSend)(
            io_cls,
            sel_registerName("objectWithIOSurface:"),
            handle->inputs[index]);
        id value = wrapped;
        if (value != nil && has_buffer_factory) {
          auto number = create_cf_index(index);
          using BufferFn = id (*)(Class, SEL, id, id, long long);
          auto candidate = reinterpret_cast<BufferFn>(objc_msgSend)(
              buffer_cls,
              sel_registerName("bufferWithIOSurfaceObject:symbolIndex:source:"),
              value,
              cf_as_id(number),
              0LL);
          if (number != nullptr) CFRelease(number);
          if (candidate != nil) value = candidate;
        }
        if (value != nil) {
          CFArrayAppendValue(wrapped_inputs, value);
        }
        auto number = create_cf_index(index);
        if (number != nullptr) {
          CFArrayAppendValue(loopback_inputs, number);
          CFArrayAppendValue(input_buffer_info_index, number);
          CFRelease(number);
        }
        auto zero = create_cf_index(0);
        if (zero != nullptr) {
          CFArrayAppendValue(input_free_value, zero);
          CFRelease(zero);
        }
      }

      for (size_t index = 0; index < handle->output_count; index++) {
        using WrapFn = id (*)(Class, SEL, IOSurfaceRef);
        auto wrapped = reinterpret_cast<WrapFn>(objc_msgSend)(
            io_cls,
            sel_registerName("objectWithIOSurface:"),
            handle->outputs[index]);
        id value = wrapped;
        if (value != nil && has_buffer_factory) {
          auto number = create_cf_index(index);
          using BufferFn = id (*)(Class, SEL, id, id, long long);
          auto candidate = reinterpret_cast<BufferFn>(objc_msgSend)(
              buffer_cls,
              sel_registerName("bufferWithIOSurfaceObject:symbolIndex:source:"),
              value,
              cf_as_id(number),
              0LL);
          if (number != nullptr) CFRelease(number);
          if (candidate != nil) value = candidate;
        }
        if (value != nil) {
          CFArrayAppendValue(output_buffer_values, value);
        }
        auto number = create_cf_index(index);
        if (number != nullptr) {
          CFArrayAppendValue(loopback_outputs, number);
          CFRelease(number);
        }
      }

      using OutputSetFn = id (*)(Class, SEL, IOSurfaceRef, id);
      output_set = reinterpret_cast<OutputSetFn>(objc_msgSend)(
          output_sets_cls,
          sel_registerName("objectWithstatsSurRef:outputBuffer:"),
          stats_surface,
          cf_as_id(output_buffer_values));
      built_output_set = output_set != nil;
      if (!built_output_set) stage = ChainingStage::output_sets_build_failed;
      if (output_set != nil) CFArrayAppendValue(output_sets, output_set);

      if (use_shared_signal_event && shared_signal_cls != nullptr &&
          io_shared_event_cls != nullptr && has_shared_signal_factory) {
        using NewFn = id (*)(Class, SEL);
        auto io_shared_event = reinterpret_cast<NewFn>(objc_msgSend)(
            io_shared_event_cls,
            sel_registerName("new"));
        if (io_shared_event != nil) {
          using SignalFn = id (*)(Class, SEL, unsigned long long, unsigned int, long long, id);
          shared_signal = reinterpret_cast<SignalFn>(objc_msgSend)(
              shared_signal_cls,
              sel_registerName("signalEventWithValue:symbolIndex:eventType:sharedEvent:"),
              1ULL,
              0U,
              0LL,
              io_shared_event);
          built_shared_signal = shared_signal != nil;
          if (shared_signal != nil) CFArrayAppendValue(signal_events, shared_signal);
        }
        if (!built_shared_signal) stage = ChainingStage::signal_event_build_failed;
      }

      if (stage == ChainingStage::unavailable || stage == ChainingStage::signal_event_build_failed) {
        int64_t zero = 0;
        auto procedure_index =
            CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &zero);
        auto transaction =
            CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &zero);
        auto fw_delay =
            CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &zero);
        auto pool_id =
            CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &zero);
        using RequestFn = id (*)(Class, SEL, id, id, id, id, id, id, id, id, id);
        chaining_request = reinterpret_cast<RequestFn>(objc_msgSend)(
            chaining_cls,
            sel_registerName(
                "chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:"),
            cf_as_id(wrapped_inputs),
            cf_as_id(output_sets),
            cf_as_id(loopback_inputs),
            cf_as_id(loopback_outputs),
            cf_as_id(procedure_index),
            cf_as_id(signal_events),
            cf_as_id(transaction),
            cf_as_id(fw_delay),
            cf_as_id(pool_id));
        built_request = chaining_request != nil;
        if (procedure_index != nullptr) CFRelease(procedure_index);
        if (transaction != nullptr) CFRelease(transaction);
        if (fw_delay != nullptr) CFRelease(fw_delay);
        if (pool_id != nullptr) CFRelease(pool_id);
        if (!built_request && stage == ChainingStage::unavailable) {
          stage = ChainingStage::request_build_failed;
        }
      }

      if (built_request && validate_request && responds_to_selector(chaining_request, "validate")) {
        using ValidateFn = bool (*)(id, SEL);
        request_validated = true;
        request_valid = reinterpret_cast<ValidateFn>(objc_msgSend)(
            chaining_request,
            sel_registerName("validate"));
        if (!request_valid) stage = ChainingStage::request_build_failed;
      }

      if (built_request && attempt_prepare && stage != ChainingStage::request_build_failed &&
          stage != ChainingStage::output_sets_build_failed) {
        auto options = create_empty_dictionary();
        id prepare_error = nil;
        using PrepareFn = bool (*)(id, SEL, id, id, id, unsigned int, id*);
        prepared = reinterpret_cast<PrepareFn>(objc_msgSend)(
            client,
            sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:"),
            client_model,
            cf_as_id(options),
            chaining_request,
            kDefaultAneQos,
            &prepare_error);
        if (options != nullptr) CFRelease(options);
        if (prepared) {
          stage = ChainingStage::prepare_succeeded;
        } else {
          if (error_message.empty()) {
            error_message = description_from_object(prepare_error);
          }
          stage = ChainingStage::prepare_failed;
        }
      } else if (built_request && stage == ChainingStage::unavailable) {
        stage = ChainingStage::prepare_skipped;
      }

      if (call_enqueue_sets && has_output_set_enqueue_factory && has_enqueue_sets_selector) {
        using EnqueueBuildFn = id (*)(Class, SEL, unsigned int, unsigned int, unsigned long long, bool, bool);
        output_set_enqueue = reinterpret_cast<EnqueueBuildFn>(objc_msgSend)(
            output_set_enqueue_cls,
            sel_registerName("outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:"),
            0U,
            0U,
            1ULL,
            false,
            false);
        built_output_set_enqueue = output_set_enqueue != nil;
        if (!built_output_set_enqueue) {
          stage = ChainingStage::output_set_enqueue_build_failed;
        } else {
          auto options = create_empty_dictionary();
          id enqueue_error = nil;
          using EnqueueFn = bool (*)(id, SEL, id, id, id, unsigned int, id*);
          called_enqueue_sets = true;
          enqueue_sets_succeeded = reinterpret_cast<EnqueueFn>(objc_msgSend)(
              client,
              sel_registerName("enqueueSetsWithModel:outputSet:options:qos:error:"),
              client_model,
              output_set_enqueue,
              cf_as_id(options),
              kDefaultAneQos,
              &enqueue_error);
          if (options != nullptr) CFRelease(options);
          if (enqueue_sets_succeeded) {
            stage = ChainingStage::enqueue_sets_call_succeeded;
          } else {
            if (error_message.empty()) {
              error_message = description_from_object(enqueue_error);
            }
            stage = ChainingStage::enqueue_sets_call_failed;
          }
        }
      }

      if (call_buffers_ready && has_input_buffers_ready_factory && has_buffers_ready_selector) {
        using ReadyBuildFn = id (*)(Class, SEL, unsigned int, id, id, unsigned long long);
        input_buffers_ready = reinterpret_cast<ReadyBuildFn>(objc_msgSend)(
            input_buffers_ready_cls,
            sel_registerName("inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:"),
            0U,
            cf_as_id(input_buffer_info_index),
            cf_as_id(input_free_value),
            0ULL);
        built_input_buffers_ready = input_buffers_ready != nil;
        if (!built_input_buffers_ready) {
          stage = ChainingStage::input_buffers_ready_build_failed;
        } else {
          if (validate_request && responds_to_selector(input_buffers_ready, "validate")) {
            using ValidateFn = bool (*)(id, SEL);
            input_buffers_ready_validated = true;
            input_buffers_ready_valid = reinterpret_cast<ValidateFn>(objc_msgSend)(
                input_buffers_ready,
                sel_registerName("validate"));
            if (!input_buffers_ready_valid) {
              stage = ChainingStage::input_buffers_ready_validate_failed;
            }
          }
          if (stage != ChainingStage::input_buffers_ready_validate_failed) {
            auto options = create_empty_dictionary();
            id ready_error = nil;
            using ReadyFn = bool (*)(id, SEL, id, id, id, unsigned int, id*);
            called_buffers_ready = true;
            buffers_ready_succeeded = reinterpret_cast<ReadyFn>(objc_msgSend)(
                client,
                sel_registerName("buffersReadyWithModel:inputBuffers:options:qos:error:"),
                client_model,
                input_buffers_ready,
                cf_as_id(options),
                kDefaultAneQos,
                &ready_error);
            if (options != nullptr) CFRelease(options);
            if (buffers_ready_succeeded) {
              stage = ChainingStage::input_buffers_ready_call_succeeded;
            } else {
              if (error_message.empty()) {
                error_message = description_from_object(ready_error);
              }
              stage = ChainingStage::input_buffers_ready_call_failed;
            }
          }
        }
      }

      if (wrapped_inputs != nullptr) CFRelease(wrapped_inputs);
      if (wrapped_outputs != nullptr) CFRelease(wrapped_outputs);
      if (loopback_inputs != nullptr) CFRelease(loopback_inputs);
      if (loopback_outputs != nullptr) CFRelease(loopback_outputs);
      if (output_buffer_values != nullptr) CFRelease(output_buffer_values);
      if (output_sets != nullptr) CFRelease(output_sets);
      if (signal_events != nullptr) CFRelease(signal_events);
      if (input_buffer_info_index != nullptr) CFRelease(input_buffer_info_index);
      if (input_free_value != nullptr) CFRelease(input_free_value);
      if (stats_surface != nullptr) CFRelease(stats_surface);
    }
  }

  json << "{";
  append_bool(json, first, "has_chaining_request_class", chaining_cls != nullptr);
  append_bool(json, first, "has_prepare_selector", has_prepare_selector);
  append_bool(json, first, "has_output_sets_class", output_sets_cls != nullptr);
  append_bool(json, first, "has_output_sets_factory", has_output_sets_factory);
  append_bool(
      json,
      first,
      "has_output_set_enqueue_class",
      output_set_enqueue_cls != nullptr);
  append_bool(
      json,
      first,
      "has_output_set_enqueue_factory",
      has_output_set_enqueue_factory);
  append_bool(
      json,
      first,
      "has_input_buffers_ready_class",
      input_buffers_ready_cls != nullptr);
  append_bool(
      json,
      first,
      "has_input_buffers_ready_factory",
      has_input_buffers_ready_factory);
  append_bool(json, first, "has_buffer_factory", has_buffer_factory);
  append_bool(json, first, "has_shared_signal_event_class", shared_signal_cls != nullptr);
  append_bool(json, first, "has_iosurface_shared_event_class", io_shared_event_cls != nullptr);
  append_bool(json, first, "has_enqueue_sets_selector", has_enqueue_sets_selector);
  append_bool(json, first, "has_buffers_ready_selector", has_buffers_ready_selector);
  append_bool(json, first, "built_output_set", built_output_set);
  append_bool(json, first, "built_request", built_request);
  append_bool(json, first, "built_output_set_enqueue", built_output_set_enqueue);
  append_bool(json, first, "built_input_buffers_ready", built_input_buffers_ready);
  append_bool(json, first, "built_shared_signal_event", built_shared_signal);
  append_bool(json, first, "request_validated", request_validated);
  append_bool(json, first, "request_valid", request_valid);
  append_bool(
      json,
      first,
      "input_buffers_ready_validated",
      input_buffers_ready_validated);
  append_bool(json, first, "input_buffers_ready_valid", input_buffers_ready_valid);
  append_bool(json, first, "prepared", prepared);
  append_bool(json, first, "called_enqueue_sets", called_enqueue_sets);
  append_bool(json, first, "enqueue_sets_succeeded", enqueue_sets_succeeded);
  append_bool(json, first, "called_buffers_ready", called_buffers_ready);
  append_bool(json, first, "buffers_ready_succeeded", buffers_ready_succeeded);
  append_int(json, first, "stage", stage);
  append_string(json, first, "error", error_message);
  json << "}";
  return copy_c_string(json.str().c_str());
}
