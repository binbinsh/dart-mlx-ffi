#include "bridge.h"
#include "ane_p.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <objc/message.h>
#include <objc/runtime.h>

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

void set_error(const std::string& message) {
  dart_mlx_ane_private_set_error_message(message.c_str());
}

id retain_object(id value) {
  return value == nil ? nil : cf_as_id(CFRetain(value));
}

void release_object(id value) {
  if (value != nil) {
    CFRelease(value);
  }
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

void free_chain(DartMlxAnePrivateChainHandle* handle) {
  if (handle == nullptr) return;
  release_object(as_id(handle->input_buffers_ready));
  release_object(as_id(handle->output_set_enqueue));
  release_object(as_id(handle->chaining_request));
  release_object(as_id(handle->client_model));
  release_object(as_id(handle->client));
  delete handle;
}

} // namespace

extern "C" DartMlxAnePrivateChainHandle* dart_mlx_ane_private_chain_new(
    const DartMlxAnePrivateSessionHandle* session,
    bool validate_request,
    bool use_shared_signal_event,
    bool attempt_prepare) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (session == nullptr || session->model == nullptr || session->input_count == 0 ||
      session->output_count == 0) {
    set_error("Private ANE chain creation requires a populated session.");
    return nullptr;
  }

  auto client_cls = reinterpret_cast<Class>(objc_getClass("_ANEClient"));
  auto chaining_cls = reinterpret_cast<Class>(objc_getClass("_ANEChainingRequest"));
  auto buffer_cls = reinterpret_cast<Class>(objc_getClass("_ANEBuffer"));
  auto output_sets_cls = reinterpret_cast<Class>(objc_getClass("_ANEIOSurfaceOutputSets"));
  auto output_set_enqueue_cls = reinterpret_cast<Class>(objc_getClass("_ANEOutputSetEnqueue"));
  auto input_buffers_ready_cls = reinterpret_cast<Class>(objc_getClass("_ANEInputBuffersReady"));
  auto shared_signal_cls = reinterpret_cast<Class>(objc_getClass("_ANESharedSignalEvent"));
  auto io_shared_event_cls = reinterpret_cast<Class>(objc_getClass("IOSurfaceSharedEvent"));
  auto io_cls = reinterpret_cast<Class>(objc_getClass("_ANEIOSurfaceObject"));

  if (client_cls == nullptr || chaining_cls == nullptr || output_sets_cls == nullptr ||
      input_buffers_ready_cls == nullptr || output_set_enqueue_cls == nullptr ||
      io_cls == nullptr) {
    set_error("Private ANE chaining classes are unavailable.");
    return nullptr;
  }

  if (!has_class_method(
          chaining_cls,
          "chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:")) {
    set_error("Private ANE chaining request factory is unavailable.");
    return nullptr;
  }

  using GetterFn = id (*)(id, SEL);
  auto client = reinterpret_cast<GetterFn>(objc_msgSend)(
      as_id(session->model),
      sel_registerName("sharedConnection"));
  auto client_model = reinterpret_cast<GetterFn>(objc_msgSend)(
      as_id(session->model),
      sel_registerName("model"));
  if (client == nil || client_model == nil) {
    set_error("Private ANE chaining client objects are unavailable.");
    return nullptr;
  }

  auto wrapped_inputs = create_cf_array();
  auto output_buffer_values = create_cf_array();
  auto loopback_inputs = create_cf_array();
  auto loopback_outputs = create_cf_array();
  auto input_buffer_info_index = create_cf_array();
  auto input_free_value = create_cf_array();
  auto output_sets = create_cf_array();
  auto signal_events = create_cf_array();
  IOSurfaceRef stats_surface = create_surface(256);
  id shared_signal = nil;
  id output_set = nil;
  id output_set_enqueue = nil;
  id input_buffers_ready = nil;
  id chaining_request = nil;

  for (size_t index = 0; index < session->input_count; index++) {
    using WrapFn = id (*)(Class, SEL, IOSurfaceRef);
    auto wrapped = reinterpret_cast<WrapFn>(objc_msgSend)(
        io_cls,
        sel_registerName("objectWithIOSurface:"),
        session->inputs[index]);
    id value = wrapped;
    if (value != nil && has_class_method(buffer_cls, "bufferWithIOSurfaceObject:symbolIndex:source:")) {
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

  for (size_t index = 0; index < session->output_count; index++) {
    using WrapFn = id (*)(Class, SEL, IOSurfaceRef);
    auto wrapped = reinterpret_cast<WrapFn>(objc_msgSend)(
        io_cls,
        sel_registerName("objectWithIOSurface:"),
        session->outputs[index]);
    id value = wrapped;
    if (value != nil && has_class_method(buffer_cls, "bufferWithIOSurfaceObject:symbolIndex:source:")) {
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
  if (output_set == nil) {
    if (wrapped_inputs != nullptr) CFRelease(wrapped_inputs);
    if (output_buffer_values != nullptr) CFRelease(output_buffer_values);
    if (loopback_inputs != nullptr) CFRelease(loopback_inputs);
    if (loopback_outputs != nullptr) CFRelease(loopback_outputs);
    if (input_buffer_info_index != nullptr) CFRelease(input_buffer_info_index);
    if (input_free_value != nullptr) CFRelease(input_free_value);
    if (output_sets != nullptr) CFRelease(output_sets);
    if (signal_events != nullptr) CFRelease(signal_events);
    if (stats_surface != nullptr) CFRelease(stats_surface);
    set_error("Private ANE failed to build an output set.");
    return nullptr;
  }
  CFArrayAppendValue(output_sets, output_set);

  if (use_shared_signal_event && shared_signal_cls != nullptr &&
      io_shared_event_cls != nullptr &&
      has_class_method(shared_signal_cls, "signalEventWithValue:symbolIndex:eventType:sharedEvent:")) {
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
      if (shared_signal != nil) {
        CFArrayAppendValue(signal_events, shared_signal);
      }
    }
  }

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
  if (procedure_index != nullptr) CFRelease(procedure_index);
  if (transaction != nullptr) CFRelease(transaction);
  if (fw_delay != nullptr) CFRelease(fw_delay);
  if (pool_id != nullptr) CFRelease(pool_id);
  if (chaining_request == nil) {
    if (wrapped_inputs != nullptr) CFRelease(wrapped_inputs);
    if (output_buffer_values != nullptr) CFRelease(output_buffer_values);
    if (loopback_inputs != nullptr) CFRelease(loopback_inputs);
    if (loopback_outputs != nullptr) CFRelease(loopback_outputs);
    if (input_buffer_info_index != nullptr) CFRelease(input_buffer_info_index);
    if (input_free_value != nullptr) CFRelease(input_free_value);
    if (output_sets != nullptr) CFRelease(output_sets);
    if (signal_events != nullptr) CFRelease(signal_events);
    if (stats_surface != nullptr) CFRelease(stats_surface);
    set_error("Private ANE failed to build a chaining request.");
    return nullptr;
  }

  if (validate_request && responds_to_selector(chaining_request, "validate")) {
    using ValidateFn = bool (*)(id, SEL);
    const bool valid = reinterpret_cast<ValidateFn>(objc_msgSend)(
        chaining_request,
        sel_registerName("validate"));
    if (!valid) {
      if (wrapped_inputs != nullptr) CFRelease(wrapped_inputs);
      if (output_buffer_values != nullptr) CFRelease(output_buffer_values);
      if (loopback_inputs != nullptr) CFRelease(loopback_inputs);
      if (loopback_outputs != nullptr) CFRelease(loopback_outputs);
      if (input_buffer_info_index != nullptr) CFRelease(input_buffer_info_index);
      if (input_free_value != nullptr) CFRelease(input_free_value);
      if (output_sets != nullptr) CFRelease(output_sets);
      if (signal_events != nullptr) CFRelease(signal_events);
      if (stats_surface != nullptr) CFRelease(stats_surface);
      set_error("Private ANE chaining request validation failed.");
      return nullptr;
    }
  }

  using EnqueueBuildFn = id (*)(Class, SEL, unsigned int, unsigned int, unsigned long long, bool, bool);
  output_set_enqueue = reinterpret_cast<EnqueueBuildFn>(objc_msgSend)(
      output_set_enqueue_cls,
      sel_registerName("outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:"),
      0U,
      0U,
      1ULL,
      false,
      false);
  using ReadyBuildFn = id (*)(Class, SEL, unsigned int, id, id, unsigned long long);
  input_buffers_ready = reinterpret_cast<ReadyBuildFn>(objc_msgSend)(
      input_buffers_ready_cls,
      sel_registerName("inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:"),
      0U,
      cf_as_id(input_buffer_info_index),
      cf_as_id(input_free_value),
      0ULL);

  bool prepared = false;
  if (attempt_prepare) {
    if (!responds_to_selector(
            client,
            "prepareChainingWithModel:options:chainingReq:qos:error:")) {
      if (wrapped_inputs != nullptr) CFRelease(wrapped_inputs);
      if (output_buffer_values != nullptr) CFRelease(output_buffer_values);
      if (loopback_inputs != nullptr) CFRelease(loopback_inputs);
      if (loopback_outputs != nullptr) CFRelease(loopback_outputs);
      if (input_buffer_info_index != nullptr) CFRelease(input_buffer_info_index);
      if (input_free_value != nullptr) CFRelease(input_free_value);
      if (output_sets != nullptr) CFRelease(output_sets);
      if (signal_events != nullptr) CFRelease(signal_events);
      if (stats_surface != nullptr) CFRelease(stats_surface);
      set_error("Private ANE prepareChaining selector is unavailable.");
      return nullptr;
    }
    auto options = create_empty_dictionary();
    id error = nil;
    using PrepareFn = bool (*)(id, SEL, id, id, id, unsigned int, id*);
    prepared = reinterpret_cast<PrepareFn>(objc_msgSend)(
        client,
        sel_registerName("prepareChainingWithModel:options:chainingReq:qos:error:"),
        client_model,
        cf_as_id(options),
        chaining_request,
        kDefaultAneQos,
        &error);
    if (options != nullptr) CFRelease(options);
    if (!prepared) {
      if (wrapped_inputs != nullptr) CFRelease(wrapped_inputs);
      if (output_buffer_values != nullptr) CFRelease(output_buffer_values);
      if (loopback_inputs != nullptr) CFRelease(loopback_inputs);
      if (loopback_outputs != nullptr) CFRelease(loopback_outputs);
      if (input_buffer_info_index != nullptr) CFRelease(input_buffer_info_index);
      if (input_free_value != nullptr) CFRelease(input_free_value);
      if (output_sets != nullptr) CFRelease(output_sets);
      if (signal_events != nullptr) CFRelease(signal_events);
      if (stats_surface != nullptr) CFRelease(stats_surface);
      set_error(description_from_object(error).empty()
          ? "Private ANE prepareChaining failed."
          : description_from_object(error));
      return nullptr;
    }
  }

  if (wrapped_inputs != nullptr) CFRelease(wrapped_inputs);
  if (output_buffer_values != nullptr) CFRelease(output_buffer_values);
  if (loopback_inputs != nullptr) CFRelease(loopback_inputs);
  if (loopback_outputs != nullptr) CFRelease(loopback_outputs);
  if (input_buffer_info_index != nullptr) CFRelease(input_buffer_info_index);
  if (input_free_value != nullptr) CFRelease(input_free_value);
  if (output_sets != nullptr) CFRelease(output_sets);
  if (signal_events != nullptr) CFRelease(signal_events);
  if (stats_surface != nullptr) CFRelease(stats_surface);

  auto* handle = new DartMlxAnePrivateChainHandle;
  handle->client = reinterpret_cast<void*>(retain_object(client));
  handle->client_model = reinterpret_cast<void*>(retain_object(client_model));
  handle->chaining_request = reinterpret_cast<void*>(retain_object(chaining_request));
  handle->output_set_enqueue = reinterpret_cast<void*>(retain_object(output_set_enqueue));
  handle->input_buffers_ready = reinterpret_cast<void*>(retain_object(input_buffers_ready));
  handle->prepared = prepared;
  handle->has_output_set_enqueue = output_set_enqueue != nil;
  handle->has_input_buffers_ready = input_buffers_ready != nil;
  return handle;
}

extern "C" void dart_mlx_ane_private_chain_free(
    DartMlxAnePrivateChainHandle* handle) {
  free_chain(handle);
}

extern "C" int dart_mlx_ane_private_chain_is_prepared(
    const DartMlxAnePrivateChainHandle* handle) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE chain is closed.");
    return -1;
  }
  return handle->prepared ? 1 : 0;
}

extern "C" int dart_mlx_ane_private_chain_has_enqueue_sets(
    const DartMlxAnePrivateChainHandle* handle) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE chain is closed.");
    return -1;
  }
  return handle->has_output_set_enqueue ? 1 : 0;
}

extern "C" int dart_mlx_ane_private_chain_has_buffers_ready(
    const DartMlxAnePrivateChainHandle* handle) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE chain is closed.");
    return -1;
  }
  return handle->has_input_buffers_ready ? 1 : 0;
}

extern "C" int dart_mlx_ane_private_chain_enqueue_sets(
    DartMlxAnePrivateChainHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->client == nullptr || handle->client_model == nullptr ||
      handle->output_set_enqueue == nullptr) {
    set_error("Private ANE chain does not have enqueueSets state.");
    return -1;
  }
  if (!responds_to_selector(
          as_id(handle->client),
          "enqueueSetsWithModel:outputSet:options:qos:error:")) {
    set_error("Private ANE enqueueSets selector is unavailable.");
    return -1;
  }
  auto options = create_empty_dictionary();
  id error = nil;
  using EnqueueFn = bool (*)(id, SEL, id, id, id, unsigned int, id*);
  const bool ok = reinterpret_cast<EnqueueFn>(objc_msgSend)(
      as_id(handle->client),
      sel_registerName("enqueueSetsWithModel:outputSet:options:qos:error:"),
      as_id(handle->client_model),
      as_id(handle->output_set_enqueue),
      cf_as_id(options),
      kDefaultAneQos,
      &error);
  if (options != nullptr) CFRelease(options);
  if (!ok) {
    set_error(description_from_object(error).empty()
        ? "Private ANE enqueueSets failed."
        : description_from_object(error));
    return -1;
  }
  return 0;
}

extern "C" int dart_mlx_ane_private_chain_buffers_ready(
    DartMlxAnePrivateChainHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->client == nullptr || handle->client_model == nullptr ||
      handle->input_buffers_ready == nullptr) {
    set_error("Private ANE chain does not have buffersReady state.");
    return -1;
  }
  if (!responds_to_selector(
          as_id(handle->client),
          "buffersReadyWithModel:inputBuffers:options:qos:error:")) {
    set_error("Private ANE buffersReady selector is unavailable.");
    return -1;
  }
  auto options = create_empty_dictionary();
  id error = nil;
  using ReadyFn = bool (*)(id, SEL, id, id, id, unsigned int, id*);
  const bool ok = reinterpret_cast<ReadyFn>(objc_msgSend)(
      as_id(handle->client),
      sel_registerName("buffersReadyWithModel:inputBuffers:options:qos:error:"),
      as_id(handle->client_model),
      as_id(handle->input_buffers_ready),
      cf_as_id(options),
      kDefaultAneQos,
      &error);
  if (options != nullptr) CFRelease(options);
  if (!ok) {
    set_error(description_from_object(error).empty()
        ? "Private ANE buffersReady failed."
        : description_from_object(error));
    return -1;
  }
  return 0;
}
