#include "bridge.h"
#include "ane_p.h"

#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

constexpr unsigned int kDefaultAneQos = 21;
constexpr const char* kModelPathPrefix = "@model_path/";
constexpr const char* kAneFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
    "AppleNeuralEngine";

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

CFDictionaryRef create_empty_dictionary() {
  return CFDictionaryCreate(
      kCFAllocatorDefault,
      nullptr,
      nullptr,
      0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
}

void* ane_framework_handle() {
  static void* handle = dlopen(kAneFrameworkPath, RTLD_NOW | RTLD_LOCAL);
  return handle;
}

CFDictionaryRef create_weights_dictionary(
    const char* const* weight_paths,
    const uint8_t* const* weight_data,
    const size_t* weight_lens,
    const size_t* weight_offsets,
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
    const auto len = weight_lens[index];
    if (path == nullptr || path[0] == '\0') {
      CFRelease(weights);
      set_error("Private ANE weight paths must be non-empty.");
      return nullptr;
    }
    if (len > 0 && data == nullptr) {
      CFRelease(weights);
      set_error("Private ANE weight data cannot be null when length is non-zero.");
      return nullptr;
    }
    auto path_string =
        CFStringCreateWithCString(kCFAllocatorDefault, path, kCFStringEncodingUTF8);
    auto blob = CFDataCreate(kCFAllocatorDefault, data, static_cast<CFIndex>(len));
    const auto raw_offset =
        weight_offsets == nullptr ? static_cast<size_t>(0) : weight_offsets[index];
    const auto signed_offset = static_cast<int64_t>(raw_offset);
    auto offset =
        CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &signed_offset);
    if (path_string == nullptr || blob == nullptr || offset == nullptr) {
      if (path_string != nullptr) CFRelease(path_string);
      if (blob != nullptr) CFRelease(blob);
      if (offset != nullptr) CFRelease(offset);
      CFRelease(weights);
      set_error("Failed to allocate private ANE weight entry.");
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
      set_error("Failed to allocate private ANE weight dictionary entry.");
      return nullptr;
    }
    CFDictionarySetValue(weights, path_string, entry);
    CFRelease(entry);
    CFRelease(path_string);
  }
  return weights;
}

bool ensure_model_lifecycle_available() {
  if (ane_framework_handle() == nullptr) {
    set_error("AppleNeuralEngine private framework is unavailable.");
    return false;
  }
  auto ane_desc = reinterpret_cast<Class>(objc_getClass("_ANEInMemoryModelDescriptor"));
  auto ane_in_memory = reinterpret_cast<Class>(objc_getClass("_ANEInMemoryModel"));
  if (ane_desc == nullptr || ane_in_memory == nullptr) {
    set_error("Private ANE model classes are unavailable.");
    return false;
  }
  if (class_getClassMethod(ane_desc, sel_registerName("modelWithMILText:weights:optionsPlist:")) ==
      nullptr) {
    set_error("Private ANE descriptor factory is unavailable.");
    return false;
  }
  if (class_getClassMethod(ane_in_memory, sel_registerName("inMemoryModelWithDescriptor:")) ==
      nullptr) {
    set_error("Private ANE in-memory model factory is unavailable.");
    return false;
  }
  return true;
}

std::filesystem::path sanitized_relative_weight_path(const std::string& path) {
  if (!path.starts_with(kModelPathPrefix)) return {};
  const auto rel = path.substr(std::strlen(kModelPathPrefix));
  if (rel.empty()) return {};
  std::filesystem::path rel_path(rel);
  if (rel_path.is_absolute()) return {};
  for (const auto& part : rel_path) {
    const auto value = part.string();
    if (value.empty() || value == "." || value == "..") return {};
  }
  return rel_path;
}

bool ensure_model_files(DartMlxAnePrivateModelHandle* handle) {
  if (!handle->temp_dir.empty()) return true;
  if (!responds_to_selector(as_id(handle->model), "hexStringIdentifier")) {
    set_error("Private ANE model does not expose hexStringIdentifier.");
    return false;
  }
  using HexFn = id (*)(id, SEL);
  auto hex_obj = reinterpret_cast<HexFn>(objc_msgSend)(
      as_id(handle->model),
      sel_registerName("hexStringIdentifier"));
  const auto hex = string_from_nsstring(hex_obj);
  if (hex.empty()) {
    set_error("Private ANE model returned an empty hexStringIdentifier.");
    return false;
  }
  auto root = std::filesystem::temp_directory_path() / hex;
  std::error_code ec;
  std::filesystem::create_directories(root / "weights", ec);
  if (ec) {
    set_error("Failed to create private ANE temp directory.");
    return false;
  }
  {
    std::ofstream out(root / "model.mil", std::ios::binary);
    out.write(handle->mil_text.data(), static_cast<std::streamsize>(handle->mil_text.size()));
    if (!out) {
      set_error("Failed to write private ANE model.mil.");
      return false;
    }
  }
  for (const auto& weight : handle->weights) {
    const auto rel = sanitized_relative_weight_path(weight.path);
    if (rel.empty()) {
      set_error("Invalid private ANE weight path: " + weight.path);
      return false;
    }
    auto full = root / rel;
    std::filesystem::create_directories(full.parent_path(), ec);
    if (ec) {
      set_error("Failed to create a private ANE weight directory.");
      return false;
    }
    std::ofstream out(full, std::ios::binary);
    out.write(
        reinterpret_cast<const char*>(weight.data.data()),
        static_cast<std::streamsize>(weight.data.size()));
    if (!out) {
      set_error("Failed to write a private ANE weight blob.");
      return false;
    }
  }
  handle->temp_dir = root.string();
  return true;
}

id compute_model_options(id model) {
  auto empty_options = create_empty_dictionary();
  if (empty_options == nullptr) return nil;
  id final_options = cf_as_id(empty_options);
  if (responds_to_selector(model, "compiledModelExists") &&
      responds_to_selector(model, "compilerOptionsWithOptions:isCompiledModelCached:")) {
    using ExistsFn = bool (*)(id, SEL);
    using OptionsFn = id (*)(id, SEL, id, bool);
    const bool compiled_exists = reinterpret_cast<ExistsFn>(objc_msgSend)(
        model,
        sel_registerName("compiledModelExists"));
    auto computed = reinterpret_cast<OptionsFn>(objc_msgSend)(
        model,
        sel_registerName("compilerOptionsWithOptions:isCompiledModelCached:"),
        cf_as_id(empty_options),
        compiled_exists);
    if (computed != nil) {
      final_options = computed;
    }
  }
  return final_options;
}

int compile_or_load(
    DartMlxAnePrivateModelHandle* handle,
    const char* selector_name,
    const char* operation_name,
    id* applied_options_out) {
  if (handle == nullptr || handle->model == nullptr) {
    set_error(std::string("Private ANE ") + operation_name + " requires an open model.");
    return -1;
  }
  if (!ensure_model_files(handle)) return -1;
  if (!responds_to_selector(as_id(handle->model), selector_name)) {
    set_error(
        std::string("Private ANE selector is unavailable for ") + operation_name +
        ": " + selector_name);
    return -1;
  }
  auto empty_options = create_empty_dictionary();
  if (empty_options == nullptr) {
    set_error("Failed to allocate private ANE options.");
    return -1;
  }
  id final_options = compute_model_options(as_id(handle->model));
  if (final_options == nil) {
    final_options = cf_as_id(empty_options);
  }
  id error = nil;
  using Fn = bool (*)(id, SEL, unsigned int, id, id*);
  auto call = [&](id options) {
    return reinterpret_cast<Fn>(objc_msgSend)(
        as_id(handle->model),
        sel_registerName(selector_name),
        kDefaultAneQos,
        options,
        &error);
  };
  bool ok = call(final_options);
  if (!ok && final_options != cf_as_id(empty_options)) {
    error = nil;
    ok = call(cf_as_id(empty_options));
    if (ok) {
      final_options = cf_as_id(empty_options);
    }
  }
  if (ok && applied_options_out != nullptr) {
    *applied_options_out = final_options;
  }
  CFRelease(empty_options);
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

extern "C" DartMlxAnePrivateModelHandle* dart_mlx_ane_private_model_new_mil_ex(
    const char* mil_text,
    const char* const* weight_paths,
    const uint8_t* const* weight_data,
    const size_t* weight_lens,
    const size_t* weight_offsets,
    size_t weight_count);

extern "C" DartMlxAnePrivateModelHandle* dart_mlx_ane_private_model_new_mil(
    const char* mil_text,
    const char* const* weight_paths,
    const uint8_t* const* weight_data,
    const size_t* weight_lens,
    size_t weight_count) {
  return dart_mlx_ane_private_model_new_mil_ex(
      mil_text,
      weight_paths,
      weight_data,
      weight_lens,
      nullptr,
      weight_count);
}

extern "C" DartMlxAnePrivateModelHandle* dart_mlx_ane_private_model_new_mil_ex(
    const char* mil_text,
    const char* const* weight_paths,
    const uint8_t* const* weight_data,
    const size_t* weight_lens,
    const size_t* weight_offsets,
    size_t weight_count) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (mil_text == nullptr || mil_text[0] == '\0') {
    set_error("Private ANE MIL text must be non-empty.");
    return nullptr;
  }
  if (!ensure_model_lifecycle_available()) return nullptr;
  if (weight_count > 0 &&
      (weight_paths == nullptr || weight_data == nullptr || weight_lens == nullptr)) {
    set_error("Private ANE weight arrays must be non-null when weight_count > 0.");
    return nullptr;
  }

  auto mil_data = CFDataCreate(
      kCFAllocatorDefault,
      reinterpret_cast<const UInt8*>(mil_text),
      static_cast<CFIndex>(std::strlen(mil_text)));
  if (mil_data == nullptr) {
    set_error("Failed to allocate private ANE MIL data.");
    return nullptr;
  }
  auto weights = weight_count == 0
      ? create_empty_dictionary()
      : create_weights_dictionary(
            weight_paths,
            weight_data,
            weight_lens,
            weight_offsets,
            weight_count);
  if (weights == nullptr) {
    CFRelease(mil_data);
    return nullptr;
  }

  auto ane_desc = reinterpret_cast<Class>(objc_getClass("_ANEInMemoryModelDescriptor"));
  auto ane_in_memory = reinterpret_cast<Class>(objc_getClass("_ANEInMemoryModel"));
  using DescFn = id (*)(Class, SEL, id, id, id);
  auto descriptor = reinterpret_cast<DescFn>(objc_msgSend)(
      ane_desc,
      sel_registerName("modelWithMILText:weights:optionsPlist:"),
      cf_as_id(mil_data),
      cf_as_id(weights),
      nil);
  CFRelease(weights);
  CFRelease(mil_data);
  if (descriptor == nil) {
    set_error("Private ANE failed to create an in-memory model descriptor.");
    return nullptr;
  }
  using ModelFn = id (*)(Class, SEL, id);
  auto model = reinterpret_cast<ModelFn>(objc_msgSend)(
      ane_in_memory,
      sel_registerName("inMemoryModelWithDescriptor:"),
      descriptor);
  if (model == nil) {
    set_error("Private ANE failed to create an in-memory model.");
    return nullptr;
  }

  auto* handle = new DartMlxAnePrivateModelHandle;
  handle->model = reinterpret_cast<void*>(retain_object(model));
  handle->eval_options = nullptr;
  handle->loaded = false;
  handle->mil_text = mil_text;
  handle->weights.reserve(weight_count);
  for (size_t index = 0; index < weight_count; index++) {
    handle->weights.push_back(
        DartMlxAnePrivateWeight{
            std::string(weight_paths[index]),
            std::vector<uint8_t>(weight_data[index], weight_data[index] + weight_lens[index]),
            weight_offsets == nullptr ? static_cast<size_t>(0) : weight_offsets[index],
        });
  }
  return handle;
}

extern "C" void dart_mlx_ane_private_model_free(
    DartMlxAnePrivateModelHandle* handle) {
  if (handle == nullptr) return;
  release_object(as_id(handle->eval_options));
  release_object(as_id(handle->model));
  delete handle;
}

extern "C" int dart_mlx_ane_private_model_compile(
    DartMlxAnePrivateModelHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  id applied_options = nil;
  const auto status = compile_or_load(
      handle,
      "compileWithQoS:options:error:",
      "compile",
      &applied_options);
  if (status == 0) {
    release_object(as_id(handle->eval_options));
    handle->eval_options = reinterpret_cast<void*>(retain_object(applied_options));
  }
  return status;
}

extern "C" int dart_mlx_ane_private_model_load(
    DartMlxAnePrivateModelHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  id applied_options = nil;
  const auto status = compile_or_load(
      handle,
      "loadWithQoS:options:error:",
      "load",
      &applied_options);
  if (status == 0) {
    handle->loaded = true;
    release_object(as_id(handle->eval_options));
    handle->eval_options = reinterpret_cast<void*>(retain_object(applied_options));
  }
  return status;
}

extern "C" int dart_mlx_ane_private_model_unload(
    DartMlxAnePrivateModelHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->model == nullptr) {
    set_error("Private ANE model handle is closed.");
    return -1;
  }
  if (!responds_to_selector(as_id(handle->model), "unloadWithQoS:error:")) {
    set_error("Private ANE unload selector is unavailable.");
    return -1;
  }
  id error = nil;
  using Fn = bool (*)(id, SEL, unsigned int, id*);
  const bool ok = reinterpret_cast<Fn>(objc_msgSend)(
      as_id(handle->model),
      sel_registerName("unloadWithQoS:error:"),
      kDefaultAneQos,
      &error);
  if (!ok) {
    auto message = description_from_object(error);
    if (message.empty()) message = "Private ANE unload failed.";
    set_error(std::move(message));
    return -1;
  }
  handle->loaded = false;
  return 0;
}

extern "C" int dart_mlx_ane_private_model_is_loaded(
    const DartMlxAnePrivateModelHandle* handle) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr || handle->model == nullptr) {
    set_error("Private ANE model handle is closed.");
    return -1;
  }
  return handle->loaded ? 1 : 0;
}

extern "C" int dart_mlx_ane_private_model_compiled_exists(
    const DartMlxAnePrivateModelHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->model == nullptr) {
    set_error("Private ANE model handle is closed.");
    return -1;
  }
  if (!responds_to_selector(as_id(handle->model), "compiledModelExists")) return 0;
  using Fn = bool (*)(id, SEL);
  return reinterpret_cast<Fn>(objc_msgSend)(
             as_id(handle->model),
             sel_registerName("compiledModelExists"))
      ? 1
      : 0;
}

extern "C" char* dart_mlx_ane_private_model_hex_identifier_copy(
    const DartMlxAnePrivateModelHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->model == nullptr) {
    set_error("Private ANE model handle is closed.");
    return nullptr;
  }
  if (!responds_to_selector(as_id(handle->model), "hexStringIdentifier")) {
    return nullptr;
  }
  using Fn = id (*)(id, SEL);
  auto value = reinterpret_cast<Fn>(objc_msgSend)(
      as_id(handle->model),
      sel_registerName("hexStringIdentifier"));
  const auto text = string_from_nsstring(value);
  return text.empty() ? nullptr : copy_c_string(text.c_str());
}
