#pragma once

#include "bridge.h"
#include "ane_p.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr const char* kAneFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
    "AppleNeuralEngine";
constexpr unsigned int kDefaultAneQos = 21;

id as_id(void* value) { return reinterpret_cast<id>(value); }
id cf_as_id(const void* value) { return reinterpret_cast<id>(const_cast<void*>(value)); }
void set_error(std::string message) { dart_mlx_ane_private_set_error_message(message.c_str()); }

void* ane_framework_handle() {
  static void* handle = dlopen(kAneFrameworkPath, RTLD_NOW | RTLD_LOCAL);
  return handle;
}

bool framework_loaded() { return ane_framework_handle() != nullptr; }

Class maybe_class(const char* name) {
  if (!framework_loaded()) {
    return nullptr;
  }
  return reinterpret_cast<Class>(objc_getClass(name));
}

bool has_class_method(Class cls, const char* selector_name) {
  return cls != nullptr &&
      class_getClassMethod(cls, sel_registerName(selector_name)) != nullptr;
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
    if (width_number != nullptr) {
      CFRelease(width_number);
    }
    if (height_number != nullptr) {
      CFRelease(height_number);
    }
    if (bpe_number != nullptr) {
      CFRelease(bpe_number);
    }
    if (bpr_number != nullptr) {
      CFRelease(bpr_number);
    }
    if (alloc_number != nullptr) {
      CFRelease(alloc_number);
    }
    if (pixel_number != nullptr) {
      CFRelease(pixel_number);
    }
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

void free_session(DartMlxAnePrivateSessionHandle* handle) {
  if (handle == nullptr) return;
  if (handle->realtime_loaded && handle->client != nullptr &&
      handle->client_model != nullptr) {
    auto options = create_empty_dictionary();
    id error = nil;
    if (responds_to_selector(
            as_id(handle->client),
            "unloadRealTimeModel:options:qos:error:")) {
      using Fn = bool (*)(id, SEL, id, id, unsigned int, id*);
      reinterpret_cast<Fn>(objc_msgSend)(
          as_id(handle->client),
          sel_registerName("unloadRealTimeModel:options:qos:error:"),
          as_id(handle->client_model),
          cf_as_id(options),
          kDefaultAneQos,
          &error);
    }
    if (responds_to_selector(as_id(handle->client), "endRealTimeTask")) {
      using EndFn = bool (*)(id, SEL);
      reinterpret_cast<EndFn>(objc_msgSend)(
          as_id(handle->client),
          sel_registerName("endRealTimeTask"));
    }
    if (options != nullptr) {
      CFRelease(options);
    }
    handle->realtime_loaded = false;
  }
  release_object(as_id(handle->client));
  release_object(as_id(handle->client_model));
  release_object(as_id(handle->request));
  release_object(as_id(handle->eval_options));
  release_object(as_id(handle->model));
  if (handle->inputs != nullptr) {
    for (size_t index = 0; index < handle->input_count; index++) {
      if (handle->inputs[index] != nullptr) {
        CFRelease(handle->inputs[index]);
      }
    }
    std::free(handle->inputs);
  }
  if (handle->outputs != nullptr) {
    for (size_t index = 0; index < handle->output_count; index++) {
      if (handle->outputs[index] != nullptr) {
        CFRelease(handle->outputs[index]);
      }
    }
    std::free(handle->outputs);
  }
  std::free(handle->input_bytes);
  std::free(handle->output_bytes);
  delete handle;
}

bool ensure_session_environment() {
  if (!framework_loaded()) {
    set_error("AppleNeuralEngine private framework is unavailable.");
    return false;
  }
  const auto request_cls = maybe_class("_ANERequest");
  const auto io_cls = maybe_class("_ANEIOSurfaceObject");
  if (request_cls == nullptr || io_cls == nullptr) {
    set_error("Private ANE request or IOSurface classes are unavailable.");
    return false;
  }
  if (!has_class_method(
          request_cls,
          "requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:")) {
    set_error("Private ANE request factory is unavailable.");
    return false;
  }
  if (!has_class_method(io_cls, "objectWithIOSurface:")) {
    set_error("Private ANE IOSurface wrapper factory is unavailable.");
    return false;
  }
  return true;
}

bool ensure_surface_write(
    IOSurfaceRef surface,
    const uint8_t* data,
    size_t len) {
  if (surface == nullptr) {
    set_error("Private ANE input surface is null.");
    return false;
  }
  if (IOSurfaceLock(surface, 0, nullptr) != 0) {
    set_error("Failed to lock a private ANE input surface for write.");
    return false;
  }
  auto* base = static_cast<uint8_t*>(IOSurfaceGetBaseAddress(surface));
  if (base == nullptr) {
    IOSurfaceUnlock(surface, 0, nullptr);
    set_error("Private ANE input surface has no base address.");
    return false;
  }
  if (len > 0) {
    std::memcpy(base, data, len);
  }
  IOSurfaceUnlock(surface, 0, nullptr);
  return true;
}

uint8_t* copy_surface_bytes(IOSurfaceRef surface, size_t len) {
  if (surface == nullptr) {
    set_error("Private ANE output surface is null.");
    return nullptr;
  }
  if (IOSurfaceLock(surface, 0, nullptr) != 0) {
    set_error("Failed to lock a private ANE output surface for read.");
    return nullptr;
  }
  auto* base = static_cast<uint8_t*>(IOSurfaceGetBaseAddress(surface));
  if (base == nullptr) {
    IOSurfaceUnlock(surface, 0, nullptr);
    set_error("Private ANE output surface has no base address.");
    return nullptr;
  }
  auto* copy = static_cast<uint8_t*>(std::malloc(len));
  if (copy == nullptr) {
    IOSurfaceUnlock(surface, 0, nullptr);
    set_error("Failed to allocate a private ANE output copy buffer.");
    return nullptr;
  }
  std::memcpy(copy, base, len);
  IOSurfaceUnlock(surface, 0, nullptr);
  return copy;
}

} // namespace
