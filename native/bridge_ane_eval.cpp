#include "ane_eval_p.h"

extern "C" DartMlxAnePrivateSessionHandle* dart_mlx_ane_private_session_new(
    const DartMlxAnePrivateModelHandle* model_handle,
    const size_t* input_bytes,
    size_t input_count,
    const size_t* output_bytes,
    size_t output_count) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (model_handle == nullptr || model_handle->model == nullptr) {
    set_error("Private ANE session creation requires an open model handle.");
    return nullptr;
  }
  if ((input_count > 0 && input_bytes == nullptr) ||
      (output_count > 0 && output_bytes == nullptr)) {
    set_error("Private ANE session byte-size arrays must be non-null.");
    return nullptr;
  }
  if (!ensure_session_environment()) {
    return nullptr;
  }
  if (!responds_to_selector(
          as_id(model_handle->model),
          "evaluateWithQoS:options:request:error:")) {
    set_error("Private ANE model does not support evaluateWithQoS:options:request:error:.");
    return nullptr;
  }

  const auto request_cls = maybe_class("_ANERequest");
  const auto io_cls = maybe_class("_ANEIOSurfaceObject");
  auto* session = new DartMlxAnePrivateSessionHandle{};
  session->model = reinterpret_cast<void*>(retain_object(as_id(model_handle->model)));
  session->eval_options =
      reinterpret_cast<void*>(retain_object(as_id(model_handle->eval_options)));
  if (responds_to_selector(as_id(model_handle->model), "sharedConnection")) {
    using GetterFn = id (*)(id, SEL);
    auto client = reinterpret_cast<GetterFn>(objc_msgSend)(
        as_id(model_handle->model),
        sel_registerName("sharedConnection"));
    session->client = reinterpret_cast<void*>(retain_object(client));
  }
  if (responds_to_selector(as_id(model_handle->model), "model")) {
    using GetterFn = id (*)(id, SEL);
    auto client_model = reinterpret_cast<GetterFn>(objc_msgSend)(
        as_id(model_handle->model),
        sel_registerName("model"));
    session->client_model = reinterpret_cast<void*>(retain_object(client_model));
  }
  session->input_count = input_count;
  session->output_count = output_count;
  session->input_bytes = input_count == 0
      ? nullptr
      : static_cast<size_t*>(std::calloc(input_count, sizeof(size_t)));
  session->output_bytes = output_count == 0
      ? nullptr
      : static_cast<size_t*>(std::calloc(output_count, sizeof(size_t)));
  session->inputs = input_count == 0
      ? nullptr
      : static_cast<IOSurfaceRef*>(std::calloc(input_count, sizeof(IOSurfaceRef)));
  session->outputs = output_count == 0
      ? nullptr
      : static_cast<IOSurfaceRef*>(std::calloc(output_count, sizeof(IOSurfaceRef)));
  if ((input_count > 0 &&
          (session->input_bytes == nullptr || session->inputs == nullptr)) ||
      (output_count > 0 &&
          (session->output_bytes == nullptr || session->outputs == nullptr))) {
    set_error("Failed to allocate private ANE session metadata.");
    free_session(session);
    return nullptr;
  }

  auto wrapped_inputs = create_cf_array();
  auto input_indices = create_cf_array();
  auto wrapped_outputs = create_cf_array();
  auto output_indices = create_cf_array();
  if (wrapped_inputs == nullptr || input_indices == nullptr || wrapped_outputs == nullptr ||
      output_indices == nullptr) {
    if (wrapped_inputs != nullptr) {
      CFRelease(wrapped_inputs);
    }
    if (input_indices != nullptr) {
      CFRelease(input_indices);
    }
    if (wrapped_outputs != nullptr) {
      CFRelease(wrapped_outputs);
    }
    if (output_indices != nullptr) {
      CFRelease(output_indices);
    }
    set_error("Failed to allocate private ANE session arrays.");
    free_session(session);
    return nullptr;
  }

  for (size_t index = 0; index < input_count; index++) {
    session->input_bytes[index] = input_bytes[index];
    session->inputs[index] = create_surface(input_bytes[index]);
    if (session->inputs[index] == nullptr) {
      set_error("Failed to allocate a private ANE input IOSurface.");
      CFRelease(wrapped_inputs);
      CFRelease(input_indices);
      CFRelease(wrapped_outputs);
      CFRelease(output_indices);
      free_session(session);
      return nullptr;
    }
    using WrapFn = id (*)(Class, SEL, IOSurfaceRef);
    auto wrapped = reinterpret_cast<WrapFn>(objc_msgSend)(
        io_cls,
        sel_registerName("objectWithIOSurface:"),
        session->inputs[index]);
    if (wrapped == nil) {
      set_error("Private ANE failed to wrap an input IOSurface.");
      CFRelease(wrapped_inputs);
      CFRelease(input_indices);
      CFRelease(wrapped_outputs);
      CFRelease(output_indices);
      free_session(session);
      return nullptr;
    }
    auto number = create_cf_index(index);
    if (number == nullptr) {
      set_error("Failed to allocate a private ANE input index.");
      CFRelease(wrapped_inputs);
      CFRelease(input_indices);
      CFRelease(wrapped_outputs);
      CFRelease(output_indices);
      free_session(session);
      return nullptr;
    }
    CFArrayAppendValue(wrapped_inputs, wrapped);
    CFArrayAppendValue(input_indices, number);
    CFRelease(number);
  }

  for (size_t index = 0; index < output_count; index++) {
    session->output_bytes[index] = output_bytes[index];
    session->outputs[index] = create_surface(output_bytes[index]);
    if (session->outputs[index] == nullptr) {
      set_error("Failed to allocate a private ANE output IOSurface.");
      CFRelease(wrapped_inputs);
      CFRelease(input_indices);
      CFRelease(wrapped_outputs);
      CFRelease(output_indices);
      free_session(session);
      return nullptr;
    }
    using WrapFn = id (*)(Class, SEL, IOSurfaceRef);
    auto wrapped = reinterpret_cast<WrapFn>(objc_msgSend)(
        io_cls,
        sel_registerName("objectWithIOSurface:"),
        session->outputs[index]);
    if (wrapped == nil) {
      set_error("Private ANE failed to wrap an output IOSurface.");
      CFRelease(wrapped_inputs);
      CFRelease(input_indices);
      CFRelease(wrapped_outputs);
      CFRelease(output_indices);
      free_session(session);
      return nullptr;
    }
    auto number = create_cf_index(index);
    if (number == nullptr) {
      set_error("Failed to allocate a private ANE output index.");
      CFRelease(wrapped_inputs);
      CFRelease(input_indices);
      CFRelease(wrapped_outputs);
      CFRelease(output_indices);
      free_session(session);
      return nullptr;
    }
    CFArrayAppendValue(wrapped_outputs, wrapped);
    CFArrayAppendValue(output_indices, number);
    CFRelease(number);
  }

  int64_t zero = 0;
  auto procedure_index =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &zero);
  if (procedure_index == nullptr) {
    CFRelease(wrapped_inputs);
    CFRelease(input_indices);
    CFRelease(wrapped_outputs);
    CFRelease(output_indices);
    set_error("Failed to allocate a private ANE procedure index.");
    free_session(session);
    return nullptr;
  }

  using RequestFn = id (*)(Class, SEL, id, id, id, id, id);
  auto request = reinterpret_cast<RequestFn>(objc_msgSend)(
      request_cls,
      sel_registerName(
          "requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
      cf_as_id(wrapped_inputs),
      cf_as_id(input_indices),
      cf_as_id(wrapped_outputs),
      cf_as_id(output_indices),
      cf_as_id(procedure_index));
  CFRelease(procedure_index);
  CFRelease(wrapped_inputs);
  CFRelease(input_indices);
  CFRelease(wrapped_outputs);
  CFRelease(output_indices);
  if (request == nil) {
    set_error("Private ANE failed to create a request object.");
    free_session(session);
    return nullptr;
  }

  session->request = reinterpret_cast<void*>(retain_object(request));
  return session;
}

extern "C" void dart_mlx_ane_private_session_free(
    DartMlxAnePrivateSessionHandle* handle) {
  free_session(handle);
}

extern "C" int dart_mlx_ane_private_session_write_input_bytes(
    DartMlxAnePrivateSessionHandle* handle,
    size_t index,
    const uint8_t* data,
    size_t len) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE session is closed.");
    return -1;
  }
  if (index >= handle->input_count) {
    set_error("Private ANE input index is out of range.");
    return -1;
  }
  if (data == nullptr && len > 0) {
    set_error("Private ANE input data cannot be null when len > 0.");
    return -1;
  }
  if (len != handle->input_bytes[index]) {
    set_error("Private ANE input byte length does not match the declared size.");
    return -1;
  }
  return ensure_surface_write(handle->inputs[index], data, len) ? 0 : -1;
}

extern "C" int dart_mlx_ane_private_session_write_input_array_packed_f32(
    DartMlxAnePrivateSessionHandle* handle,
    size_t index,
    const DartMlxArrayHandle* input,
    size_t seq_len,
    size_t dim,
    size_t lane) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE session is closed.");
    return -1;
  }
  if (input == nullptr) {
    set_error("Private ANE packed input array is null.");
    return -1;
  }
  if (index >= handle->input_count) {
    set_error("Private ANE input index is out of range.");
    return -1;
  }
  if (seq_len == 0 || dim == 0 || lane == 0) {
    set_error("Private ANE packed input requires positive seq/dim/lane.");
    return -1;
  }
  if (seq_len > lane) {
    set_error("Private ANE packed input requires seq_len <= lane.");
    return -1;
  }
  const size_t expected_elements = seq_len * dim;
  const size_t expected_bytes = dim * lane * sizeof(float);
  if (mlx_array_size(input->value) != expected_elements) {
    set_error("Private ANE packed input element count mismatch.");
    return -1;
  }
  if (handle->input_bytes[index] != expected_bytes) {
    set_error("Private ANE packed input byte length mismatch.");
    return -1;
  }

  const int status = copy_array_data_contiguous(
      input, [&](mlx_array contiguous) {
        if (mlx_array_dtype(contiguous) != MLX_FLOAT32) {
          set_error("Private ANE packed input expects a float32 array.");
          return 1;
        }
        auto* src = mlx_array_data_float32(contiguous);
        if (src == nullptr) {
          set_error("Private ANE packed input failed to access float32 data.");
          return 1;
        }
        auto surface = handle->inputs[index];
        if (surface == nullptr) {
          set_error("Private ANE packed input surface is null.");
          return 1;
        }
        if (IOSurfaceLock(surface, 0, nullptr) != 0) {
          set_error("Failed to lock private ANE packed input surface.");
          return 1;
        }
        auto* dst = static_cast<float*>(IOSurfaceGetBaseAddress(surface));
        if (dst == nullptr) {
          IOSurfaceUnlock(surface, 0, nullptr);
          set_error("Private ANE packed input surface has no base address.");
          return 1;
        }
        std::memset(dst, 0, expected_bytes);
        for (size_t token = 0; token < seq_len; token++) {
          const size_t src_base = token * dim;
          for (size_t channel = 0; channel < dim; channel++) {
            dst[channel * lane + token] = src[src_base + channel];
          }
        }
        IOSurfaceUnlock(surface, 0, nullptr);
        return 0;
      });
  return status == 0 ? 0 : -1;
}

extern "C" uint8_t* dart_mlx_ane_private_session_read_output_bytes_copy(
    DartMlxAnePrivateSessionHandle* handle,
    size_t index,
    size_t* len_out) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE session is closed.");
    return nullptr;
  }
  if (len_out == nullptr) {
    set_error("Private ANE output length pointer is null.");
    return nullptr;
  }
  if (index >= handle->output_count) {
    set_error("Private ANE output index is out of range.");
    return nullptr;
  }
  *len_out = handle->output_bytes[index];
  return copy_surface_bytes(handle->outputs[index], handle->output_bytes[index]);
}

extern "C" int dart_mlx_ane_private_session_read_output_bytes(
    DartMlxAnePrivateSessionHandle* handle,
    size_t index,
    uint8_t* out,
    size_t len) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE session is closed.");
    return -1;
  }
  if (index >= handle->output_count) {
    set_error("Private ANE output index is out of range.");
    return -1;
  }
  if (len != handle->output_bytes[index]) {
    set_error("Private ANE output byte length does not match the declared size.");
    return -1;
  }
  if (out == nullptr && len > 0) {
    set_error("Private ANE output destination cannot be null when len > 0.");
    return -1;
  }

  auto surface = handle->outputs[index];
  if (surface == nullptr) {
    set_error("Private ANE output surface is null.");
    return -1;
  }
  if (IOSurfaceLock(surface, 0, nullptr) != 0) {
    set_error("Failed to lock a private ANE output surface for read.");
    return -1;
  }
  auto* base = static_cast<const uint8_t*>(IOSurfaceGetBaseAddress(surface));
  if (base == nullptr) {
    IOSurfaceUnlock(surface, 0, nullptr);
    set_error("Private ANE output surface has no base address.");
    return -1;
  }
  if (len > 0) {
    std::memcpy(out, base, len);
  }
  IOSurfaceUnlock(surface, 0, nullptr);
  return 0;
}

extern "C" int dart_mlx_ane_private_session_evaluate(
    DartMlxAnePrivateSessionHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->model == nullptr || handle->request == nullptr) {
    set_error("Private ANE session is incomplete.");
    return -1;
  }
  auto options = create_empty_dictionary();
  if (handle->eval_options == nullptr && options == nullptr) {
    set_error("Failed to allocate private ANE evaluation options.");
    return -1;
  }
  auto eval_options = handle->eval_options == nullptr
      ? cf_as_id(options)
      : as_id(handle->eval_options);
  id error = nil;
  bool ok = false;
  if (handle->client != nullptr && handle->client_model != nullptr &&
      responds_to_selector(
          as_id(handle->client),
          "evaluateWithModel:options:request:qos:error:")) {
    using ClientEvalFn = bool (*)(id, SEL, id, id, id, unsigned int, id*);
    ok = reinterpret_cast<ClientEvalFn>(objc_msgSend)(
        as_id(handle->client),
        sel_registerName("evaluateWithModel:options:request:qos:error:"),
        as_id(handle->client_model),
        eval_options,
        as_id(handle->request),
        kDefaultAneQos,
        &error);
  }
  if (!ok) {
    error = nil;
    using EvalFn = bool (*)(id, SEL, unsigned int, id, id, id*);
    ok = reinterpret_cast<EvalFn>(objc_msgSend)(
        as_id(handle->model),
        sel_registerName("evaluateWithQoS:options:request:error:"),
        kDefaultAneQos,
        eval_options,
        as_id(handle->request),
        &error);
  }
  if (options != nullptr) CFRelease(options);
  if (!ok) {
    auto message = description_from_object(error);
    if (message.empty()) {
      message = "Private ANE evaluate failed.";
    }
    set_error(std::move(message));
    return -1;
  }
  return 0;
}

extern "C" int dart_mlx_ane_private_session_prepare_realtime(
    DartMlxAnePrivateSessionHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->model == nullptr) {
    set_error("Private ANE session is incomplete.");
    return -1;
  }
  if (handle->realtime_loaded) {
    return 0;
  }
  if (!responds_to_selector(as_id(handle->model), "sharedConnection") ||
      !responds_to_selector(as_id(handle->model), "model")) {
    set_error("Private ANE realtime path is unavailable on this model.");
    return -1;
  }

  id client = as_id(handle->client);
  id client_model = as_id(handle->client_model);
  if (client == nil || client_model == nil) {
    using GetterFn = id (*)(id, SEL);
    client = reinterpret_cast<GetterFn>(objc_msgSend)(
        as_id(handle->model),
        sel_registerName("sharedConnection"));
    client_model = reinterpret_cast<GetterFn>(objc_msgSend)(
        as_id(handle->model),
        sel_registerName("model"));
  }
  if (client == nil || client_model == nil) {
    set_error("Private ANE realtime connection objects are unavailable.");
    return -1;
  }
  if (!responds_to_selector(client, "beginRealTimeTask") ||
      !responds_to_selector(client, "loadRealTimeModel:options:qos:error:") ||
      !responds_to_selector(client, "evaluateRealTimeWithModel:options:request:error:") ||
      !responds_to_selector(client, "unloadRealTimeModel:options:qos:error:") ||
      !responds_to_selector(client, "endRealTimeTask")) {
    set_error("Private ANE realtime selectors are unavailable.");
    return -1;
  }

  auto options = create_empty_dictionary();
  if (handle->eval_options == nullptr && options == nullptr) {
    set_error("Failed to allocate private ANE realtime options.");
    return -1;
  }
  auto eval_options = handle->eval_options == nullptr
      ? cf_as_id(options)
      : as_id(handle->eval_options);

  using BeginFn = bool (*)(id, SEL);
  reinterpret_cast<BeginFn>(objc_msgSend)(
      client,
      sel_registerName("beginRealTimeTask"));

  id error = nil;
  using LoadFn = bool (*)(id, SEL, id, id, unsigned int, id*);
  const bool ok = reinterpret_cast<LoadFn>(objc_msgSend)(
      client,
      sel_registerName("loadRealTimeModel:options:qos:error:"),
      client_model,
      eval_options,
      kDefaultAneQos,
      &error);
  if (options != nullptr) CFRelease(options);
  if (!ok) {
    if (responds_to_selector(client, "endRealTimeTask")) {
      using EndFn = bool (*)(id, SEL);
      reinterpret_cast<EndFn>(objc_msgSend)(
          client,
          sel_registerName("endRealTimeTask"));
    }
    auto message = description_from_object(error);
    if (message.empty()) {
      message = "Private ANE realtime load failed.";
    }
    set_error(std::move(message));
    return -1;
  }

  if (handle->client == nullptr) {
    handle->client = reinterpret_cast<void*>(retain_object(client));
  }
  if (handle->client_model == nullptr) {
    handle->client_model = reinterpret_cast<void*>(retain_object(client_model));
  }
  handle->realtime_loaded = true;
  return 0;
}

extern "C" int dart_mlx_ane_private_session_teardown_realtime(
    DartMlxAnePrivateSessionHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr) {
    set_error("Private ANE session is closed.");
    return -1;
  }
  if (!handle->realtime_loaded) {
    return 0;
  }

  auto options = create_empty_dictionary();
  if (handle->eval_options == nullptr && options == nullptr) {
    set_error("Failed to allocate private ANE realtime options.");
    return -1;
  }
  auto eval_options = handle->eval_options == nullptr
      ? cf_as_id(options)
      : as_id(handle->eval_options);

  id error = nil;
  using UnloadFn = bool (*)(id, SEL, id, id, unsigned int, id*);
  reinterpret_cast<UnloadFn>(objc_msgSend)(
      as_id(handle->client),
      sel_registerName("unloadRealTimeModel:options:qos:error:"),
      as_id(handle->client_model),
      eval_options,
      kDefaultAneQos,
      &error);
  if (options != nullptr) CFRelease(options);

  if (responds_to_selector(as_id(handle->client), "endRealTimeTask")) {
    using EndFn = bool (*)(id, SEL);
    reinterpret_cast<EndFn>(objc_msgSend)(
        as_id(handle->client),
        sel_registerName("endRealTimeTask"));
  }

  release_object(as_id(handle->client));
  release_object(as_id(handle->client_model));
  handle->client = nullptr;
  handle->client_model = nullptr;
  handle->realtime_loaded = false;
  return 0;
}

extern "C" int dart_mlx_ane_private_session_evaluate_realtime(
    DartMlxAnePrivateSessionHandle* handle) {
  dart_mlx_ane_private_clear_error();
  ScopedAutoreleasePool autorelease_pool;
  if (handle == nullptr || handle->client == nullptr || handle->client_model == nullptr ||
      handle->request == nullptr || !handle->realtime_loaded) {
    set_error("Private ANE realtime session is not prepared.");
    return -1;
  }

  auto options = create_empty_dictionary();
  if (handle->eval_options == nullptr && options == nullptr) {
    set_error("Failed to allocate private ANE realtime options.");
    return -1;
  }
  auto eval_options = handle->eval_options == nullptr
      ? cf_as_id(options)
      : as_id(handle->eval_options);

  id error = nil;
  using EvalFn = bool (*)(id, SEL, id, id, id, id*);
  const bool ok = reinterpret_cast<EvalFn>(objc_msgSend)(
      as_id(handle->client),
      sel_registerName("evaluateRealTimeWithModel:options:request:error:"),
      as_id(handle->client_model),
      eval_options,
      as_id(handle->request),
      &error);
  if (options != nullptr) CFRelease(options);
  if (!ok) {
    auto message = description_from_object(error);
    if (message.empty()) {
      message = "Private ANE realtime evaluate failed.";
    }
    set_error(std::move(message));
    return -1;
  }
  return 0;
}

extern "C" int dart_mlx_ane_private_session_realtime_is_loaded(
    const DartMlxAnePrivateSessionHandle* handle) {
  dart_mlx_ane_private_clear_error();
  if (handle == nullptr) {
    set_error("Private ANE session is closed.");
    return -1;
  }
  return handle->realtime_loaded ? 1 : 0;
}
