#include "bridge.h"

extern "C" char* dart_mlx_version_copy() {
  mlx_string value = mlx_string_new();
  if (mlx_version(&value) != 0) {
    return nullptr;
  }
  auto* copy = copy_c_string(mlx_string_data(value));
  mlx_string_free(value);
  return copy;
}

extern "C" void dart_mlx_string_free_copy(char* value) {
  std::free(value);
}

extern "C" DartMlxDeviceHandle* dart_mlx_default_device() {
  mlx_device value = mlx_device_new();
  if (mlx_get_default_device(&value) != 0) {
    return nullptr;
  }
  return wrap_device(value);
}

extern "C" int dart_mlx_device_is_available(const DartMlxDeviceHandle* handle) {
  bool available = false;
  if (mlx_device_is_available(&available, handle->value) != 0) {
    return -1;
  }
  return available ? 1 : 0;
}

extern "C" int dart_mlx_device_get_index(const DartMlxDeviceHandle* handle) {
  int index = 0;
  if (mlx_device_get_index(&index, handle->value) != 0) {
    return -1;
  }
  return index;
}

extern "C" int dart_mlx_device_get_type(const DartMlxDeviceHandle* handle) {
  auto type = static_cast<mlx_device_type>(0);
  if (mlx_device_get_type(&type, handle->value) != 0) {
    return -1;
  }
  return static_cast<int>(type);
}

extern "C" bool dart_mlx_device_equal(
    const DartMlxDeviceHandle* lhs,
    const DartMlxDeviceHandle* rhs) {
  return mlx_device_equal(lhs->value, rhs->value);
}

extern "C" int dart_mlx_device_count(int type) {
  int count = 0;
  if (mlx_device_count(&count, static_cast<mlx_device_type>(type)) != 0) {
    return -1;
  }
  return count;
}

extern "C" DartMlxClosureHandle* dart_mlx_function_from_callback(
    DartMlxClosureCallback callback) {
  auto* payload = new DartMlxClosurePayload{callback};
  auto value = mlx_closure_new_func_payload(
      dart_mlx_closure_trampoline, payload, free_closure_payload);
  if (value.ctx == nullptr) {
    delete payload;
    return nullptr;
  }
  return wrap_closure(value);
}

extern "C" DartMlxKwHandle* dart_mlx_kw_function_from_callback(
    DartMlxKwCallback callback) {
  auto* payload = new DartMlxKwPayload{callback};
  auto value = mlx_closure_kwargs_new_func_payload(
      dart_mlx_kw_trampoline, payload, free_kw_payload);
  if (value.ctx == nullptr) {
    delete payload;
    return nullptr;
  }
  return wrap_kw(value);
}

extern "C" void dart_mlx_kw_function_free(DartMlxKwHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_closure_kwargs_free(handle->value);
  delete handle;
}

extern "C" void dart_mlx_function_free(DartMlxClosureHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_closure_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_kw_function_apply(
    const DartMlxKwHandle* function,
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
  auto status = mlx_closure_kwargs_apply(&results, function->value, args, kwargs);
  mlx_vector_array_free(args);
  mlx_map_string_to_array_free(kwargs);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(results, outputs_out, outputs_len_out);
  mlx_vector_array_free(results);
  return status;
}

extern "C" int dart_mlx_function_apply(
    const DartMlxClosureHandle* function,
    DartMlxArrayHandle** inputs,
    size_t input_len,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto args = build_array_vector(inputs, input_len);
  auto results = mlx_vector_array_new();
  auto status = mlx_closure_apply(&results, function->value, args);
  mlx_vector_array_free(args);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(results, outputs_out, outputs_len_out);
  mlx_vector_array_free(results);
  return status;
}

extern "C" DartMlxClosureHandle* dart_mlx_function_checkpoint(
    const DartMlxClosureHandle* function) {
  auto out = mlx_closure_new();
  if (mlx_checkpoint(&out, function->value) != 0) {
    return nullptr;
  }
  return wrap_closure(out);
}

extern "C" DartMlxClosureHandle* dart_mlx_function_compile(
    const DartMlxClosureHandle* function,
    bool shapeless) {
  auto out = mlx_closure_new();
  if (mlx_compile(&out, function->value, shapeless) != 0) {
    return nullptr;
  }
  return wrap_closure(out);
}

extern "C" DartMlxCustomHandle* dart_mlx_custom_from_callback(
    DartMlxCustomCallback callback) {
  auto* payload = new DartMlxCustomPayload{callback};
  auto value = mlx_closure_custom_new_func_payload(
      dart_mlx_custom_trampoline, payload, free_custom_payload);
  if (value.ctx == nullptr) {
    delete payload;
    return nullptr;
  }
  return wrap_custom(value);
}

extern "C" void dart_mlx_custom_free(DartMlxCustomHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_closure_custom_free(handle->value);
  delete handle;
}

extern "C" DartMlxClosureHandle* dart_mlx_function_custom_vjp(
    const DartMlxClosureHandle* function,
    const DartMlxCustomHandle* custom) {
  auto out = mlx_closure_new();
  if (mlx_custom_vjp(&out, function->value, custom->value) != 0) {
    return nullptr;
  }
  return wrap_closure(out);
}

extern "C" DartMlxCustomJvpHandle* dart_mlx_custom_jvp_from_callback(
    DartMlxCustomJvpCallback callback) {
  auto* payload = new DartMlxCustomJvpPayload{callback};
  auto value = mlx_closure_custom_jvp_new_func_payload(
      dart_mlx_custom_jvp_trampoline, payload, free_custom_jvp_payload);
  if (value.ctx == nullptr) {
    delete payload;
    return nullptr;
  }
  return wrap_custom_jvp(value);
}

extern "C" void dart_mlx_custom_jvp_free(DartMlxCustomJvpHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_closure_custom_jvp_free(handle->value);
  delete handle;
}

extern "C" DartMlxClosureHandle* dart_mlx_function_custom(
    const DartMlxClosureHandle* function,
    const DartMlxCustomHandle* custom_vjp,
    const DartMlxCustomJvpHandle* custom_jvp) {
  auto out = mlx_closure_new();
  if (mlx_custom_function(
          &out,
          function->value,
          custom_vjp == nullptr ? mlx_closure_custom() : custom_vjp->value,
          custom_jvp == nullptr ? mlx_closure_custom_jvp()
                                : custom_jvp->value,
          mlx_closure_custom_vmap()) != 0) {
    return nullptr;
  }
  return wrap_closure(out);
}

extern "C" int dart_mlx_function_jvp(
    const DartMlxClosureHandle* function,
    DartMlxArrayHandle** primals,
    size_t primals_len,
    DartMlxArrayHandle** tangents,
    size_t tangents_len,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out,
    DartMlxArrayHandle*** tangents_out,
    size_t* tangents_len_out) {
  auto primals_vec = build_array_vector(primals, primals_len);
  auto tangents_vec = build_array_vector(tangents, tangents_len);
  auto out_values = mlx_vector_array_new();
  auto out_tangents = mlx_vector_array_new();
  auto status = mlx_jvp(
      &out_values, &out_tangents, function->value, primals_vec, tangents_vec);
  mlx_vector_array_free(primals_vec);
  mlx_vector_array_free(tangents_vec);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(out_values, outputs_out, outputs_len_out);
  if (status == 0) {
    status =
        export_vector_array(out_tangents, tangents_out, tangents_len_out);
  }
  mlx_vector_array_free(out_values);
  mlx_vector_array_free(out_tangents);
  return status;
}

extern "C" int dart_mlx_function_vjp(
    const DartMlxClosureHandle* function,
    DartMlxArrayHandle** primals,
    size_t primals_len,
    DartMlxArrayHandle** cotangents,
    size_t cotangents_len,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out,
    DartMlxArrayHandle*** cotangents_out,
    size_t* cotangents_len_out) {
  auto primals_vec = build_array_vector(primals, primals_len);
  auto cotangents_vec = build_array_vector(cotangents, cotangents_len);
  auto out_values = mlx_vector_array_new();
  auto out_cotangents = mlx_vector_array_new();
  auto status = mlx_vjp(
      &out_values,
      &out_cotangents,
      function->value,
      primals_vec,
      cotangents_vec);
  mlx_vector_array_free(primals_vec);
  mlx_vector_array_free(cotangents_vec);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(out_values, outputs_out, outputs_len_out);
  if (status == 0) {
    status = export_vector_array(
        out_cotangents, cotangents_out, cotangents_len_out);
  }
  mlx_vector_array_free(out_values);
  mlx_vector_array_free(out_cotangents);
  return status;
}

extern "C" int dart_mlx_function_value_and_grad(
    const DartMlxClosureHandle* function,
    const int* argnums,
    size_t argnums_len,
    DartMlxArrayHandle** inputs,
    size_t inputs_len,
    DartMlxArrayHandle*** values_out,
    size_t* values_len_out,
    DartMlxArrayHandle*** grads_out,
    size_t* grads_len_out) {
  auto fn = mlx_closure_value_and_grad_new();
  auto status =
      mlx_value_and_grad(&fn, function->value, argnums, argnums_len);
  if (status != 0) {
    return status;
  }
  auto inputs_vec = build_array_vector(inputs, inputs_len);
  auto out_values = mlx_vector_array_new();
  auto out_grads = mlx_vector_array_new();
  status = mlx_closure_value_and_grad_apply(&out_values, &out_grads, fn, inputs_vec);
  mlx_vector_array_free(inputs_vec);
  mlx_closure_value_and_grad_free(fn);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(out_values, values_out, values_len_out);
  if (status == 0) {
    status = export_vector_array(out_grads, grads_out, grads_len_out);
  }
  mlx_vector_array_free(out_values);
  mlx_vector_array_free(out_grads);
  return status;
}
