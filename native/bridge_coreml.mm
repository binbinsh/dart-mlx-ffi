#include "bridge.h"
#include "ane_p.h"

#import <CoreML/CoreML.h>
#import <mach/mach_time.h>

struct DartMlxCoreMlHandle {
  void* model;
  void* compiled_url;
  void* input_name;
  void* output_name;
  NSArray<NSNumber*>* input_shape;
  int output_count;
  uint64_t last_predict_ns;
};

namespace {

void set_coreml_error(const std::string& message) {
  dart_mlx_ane_private_set_error_message(message.c_str());
}

NSString* to_nsstring(const char* value) {
  return value == nullptr ? nil : [NSString stringWithUTF8String:value];
}

MLComputeUnits compute_units_from_int(int value) {
  switch (value) {
    case 0:
      return MLComputeUnitsCPUOnly;
    case 1:
      return MLComputeUnitsCPUAndGPU;
    case 2:
      return MLComputeUnitsAll;
    case 3:
    default:
      return MLComputeUnitsCPUAndNeuralEngine;
  }
}

uint64_t elapsed_ns(uint64_t start, uint64_t end) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  return (end - start) * info.numer / info.denom;
}

NSURL* compiled_url_for_path(NSString* path, NSError** error) {
  NSURL* source_url = [NSURL fileURLWithPath:path];
  if ([[path pathExtension] isEqualToString:@"mlmodelc"]) {
    return source_url;
  }
  return [MLModel compileModelAtURL:source_url error:error];
}

} // namespace

extern "C" DartMlxCoreMlHandle* dart_mlx_coreml_model_load(
    const char* path,
    const char* input_name,
    const char* output_name,
    const int* input_shape,
    size_t input_rank,
    int output_count,
    int compute_units) {
  @autoreleasepool {
    if (path == nullptr || input_name == nullptr || output_name == nullptr ||
        input_shape == nullptr || input_rank == 0 || output_count <= 0) {
      set_coreml_error("Core ML load arguments are invalid.");
      return nullptr;
    }

    NSString* ns_path = to_nsstring(path);
    NSString* ns_input = to_nsstring(input_name);
    NSString* ns_output = to_nsstring(output_name);
    if (ns_path == nil || ns_input == nil || ns_output == nil) {
      set_coreml_error("Core ML string conversion failed.");
      return nullptr;
    }

    NSError* error = nil;
    NSURL* compiled_url = compiled_url_for_path(ns_path, &error);
    if (compiled_url == nil) {
      set_coreml_error(error.localizedDescription.UTF8String ?: "Core ML compile failed.");
      return nullptr;
    }

    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:input_rank];
    for (size_t index = 0; index < input_rank; index++) {
      [shape addObject:@(input_shape[index])];
    }

    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = compute_units_from_int(compute_units);
    MLModel* model = [MLModel modelWithContentsOfURL:compiled_url configuration:config error:&error];
    if (model == nil) {
      set_coreml_error(error.localizedDescription.UTF8String ?: "Core ML model load failed.");
      return nullptr;
    }

    auto* handle = new DartMlxCoreMlHandle;
    handle->model = (__bridge_retained void*)model;
    handle->compiled_url = (__bridge_retained void*)compiled_url;
    handle->input_name = (__bridge_retained void*)ns_input;
    handle->output_name = (__bridge_retained void*)ns_output;
    handle->input_shape = [shape copy];
    handle->output_count = output_count;
    handle->last_predict_ns = 0;
    return handle;
  }
}

extern "C" void dart_mlx_coreml_model_free(DartMlxCoreMlHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  if (handle->model != nullptr) {
    CFRelease(handle->model);
  }
  if (handle->compiled_url != nullptr) {
    NSURL* url = (__bridge NSURL*)handle->compiled_url;
    NSString* ext = [[url path] pathExtension];
    if (![ext isEqualToString:@"mlmodelc"]) {
      [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
    }
    CFRelease(handle->compiled_url);
  }
  if (handle->input_name != nullptr) {
    CFRelease(handle->input_name);
  }
  if (handle->output_name != nullptr) {
    CFRelease(handle->output_name);
  }
  delete handle;
}

extern "C" float* dart_mlx_coreml_predict_f32_copy(
    DartMlxCoreMlHandle* handle,
    const float* input_values,
    size_t input_count,
    size_t* output_count_out) {
  @autoreleasepool {
    if (handle == nullptr || input_values == nullptr || output_count_out == nullptr) {
      set_coreml_error("Core ML predict arguments are invalid.");
      return nullptr;
    }
    size_t expected = 1;
    for (NSNumber* dim in handle->input_shape) {
      expected *= dim.unsignedIntegerValue;
    }
    if (input_count != expected) {
      set_coreml_error("Core ML input element count mismatch.");
      return nullptr;
    }

    NSError* error = nil;
    MLMultiArray* input = [[MLMultiArray alloc] initWithShape:handle->input_shape dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if (input == nil) {
      set_coreml_error(error.localizedDescription.UTF8String ?: "Core ML input allocation failed.");
      return nullptr;
    }
    std::memcpy(input.dataPointer, input_values, input_count * sizeof(float));

    NSString* input_name = (__bridge NSString*)handle->input_name;
    NSDictionary* dict = @{input_name: [MLFeatureValue featureValueWithMultiArray:input]};
    id<MLFeatureProvider> provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&error];
    if (provider == nil) {
      set_coreml_error(error.localizedDescription.UTF8String ?: "Core ML feature provider failed.");
      return nullptr;
    }

    MLModel* model = (__bridge MLModel*)handle->model;
    const auto start = mach_absolute_time();
    id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
    const auto end = mach_absolute_time();
    handle->last_predict_ns = elapsed_ns(start, end);
    if (prediction == nil) {
      set_coreml_error(error.localizedDescription.UTF8String ?: "Core ML prediction failed.");
      return nullptr;
    }

    NSString* output_name = (__bridge NSString*)handle->output_name;
    MLFeatureValue* feature = [prediction featureValueForName:output_name];
    MLMultiArray* output = feature.multiArrayValue;
    if (output == nil) {
      set_coreml_error("Core ML output is missing.");
      return nullptr;
    }

    const auto count = static_cast<size_t>(output.count);
    auto* copy = static_cast<float*>(std::malloc(count * sizeof(float)));
    if (copy == nullptr) {
      set_coreml_error("Core ML output allocation failed.");
      return nullptr;
    }
    if (output.dataType == MLMultiArrayDataTypeFloat32) {
      std::memcpy(copy, output.dataPointer, count * sizeof(float));
    } else {
      const double* src = static_cast<const double*>(output.dataPointer);
      for (size_t index = 0; index < count; index++) {
        copy[index] = static_cast<float>(src[index]);
      }
    }
    *output_count_out = count;
    return copy;
  }
}

extern "C" int64_t dart_mlx_coreml_last_predict_time_ns(DartMlxCoreMlHandle* handle) {
  if (handle == nullptr) {
    set_coreml_error("Core ML handle is closed.");
    return -1;
  }
  return static_cast<int64_t>(handle->last_predict_ns);
}
