#include "coreml_stateful.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "runtime_bridge.h"

#if defined(__APPLE__)
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#endif

namespace {

std::string lower(const std::string& value) {
  std::string out = value;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

}  // namespace

#if defined(__APPLE__)

MLComputeUnits dinf_coreml_parse_compute_units(
    const nlohmann::json& stage,
    MLComputeUnits fallback) {
  if (!stage.is_object() || !stage.contains("compute_units")) {
    return fallback;
  }
  const auto& value = stage.at("compute_units");
  if (!value.is_string()) {
    return fallback;
  }
  const std::string requested = lower(value.get<std::string>());
  if (requested == "cpu_and_ne" ||
      requested == "cpu_and_neural_engine" ||
      requested == "ane") {
    return MLComputeUnitsCPUAndNeuralEngine;
  }
  if (requested == "cpu_and_gpu" || requested == "gpu") {
    return MLComputeUnitsCPUAndGPU;
  }
  if (requested == "cpu_only" || requested == "cpu") {
    return MLComputeUnitsCPUOnly;
  }
  if (requested == "all") {
    return MLComputeUnitsAll;
  }
  return fallback;
}

#endif  // __APPLE__

bool dinf_coreml_stage_is_stateful(const nlohmann::json& stage) {
  if (!stage.is_object() || !stage.contains("stateful")) {
    return false;
  }
  const auto& value = stage.at("stateful");
  if (value.is_boolean()) {
    return value.get<bool>();
  }
  return false;
}

#if defined(__APPLE__)

@implementation DinfCoreMlStatefulStore {
  NSMutableDictionary<NSString*, MLState*>* _states;
}

- (instancetype)init {
  self = [super init];
  if (self != nil) {
    _states = [NSMutableDictionary dictionary];
  }
  return self;
}

- (MLState *)stateForStage:(NSString*)stageName fromModel:(MLModel*)model {
  if (stageName == nil || model == nil) {
    return nil;
  }
  MLState* existing = _states[stageName];
  if (existing != nil) {
    return existing;
  }
  if (@available(macOS 15.0, iOS 18.0, *)) {
    MLState* created = [model newState];
    if (created != nil) {
      _states[stageName] = created;
    }
    return created;
  }
  return nil;
}

- (void)reset {
  [_states removeAllObjects];
}

@end

id<MLFeatureProvider> dinf_coreml_predict_with_state(
    MLModel* model,
    id<MLFeatureProvider> inputs,
    DinfCoreMlStatefulStore* store,
    NSString* stageName,
    std::string* error) {
  if (model == nil || inputs == nil || store == nil || stageName == nil) {
    if (error != nullptr) {
      *error = "Stateful Core ML prediction requires model, inputs, store, "
               "and stage name";
    }
    return nil;
  }
  if (@available(macOS 15.0, iOS 18.0, *)) {
    MLState* state = [store stateForStage:stageName fromModel:model];
    if (state == nil) {
      if (error != nullptr) {
        *error = std::string("Failed to allocate MLState for stage '") +
                 stageName.UTF8String + "'";
      }
      return nil;
    }
    NSError* ns_error = nil;
    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
    id<MLFeatureProvider> prediction =
        [model predictionFromFeatures:inputs
                           usingState:state
                              options:options
                                error:&ns_error];
    if (prediction == nil && error != nullptr) {
      *error = ns_error == nil
          ? std::string("Stateful prediction failed for stage '") +
                stageName.UTF8String + "'"
          : ns_error.localizedDescription.UTF8String;
    }
    return prediction;
  }
  if (error != nullptr) {
    *error = "Stateful Core ML pipelines require iOS 18+/macOS 15+";
  }
  return nil;
}

#endif  // __APPLE__

extern "C" int32_t dinf_coreml_reset_state(
    DinfRuntimeSession* session,
    char** error) {
  if (session == nullptr) {
    if (error != nullptr) {
      *error = dinf_copy_string("dinf_coreml_reset_state: session is null");
    }
    return 1;
  }
  auto* resettable = dynamic_cast<DinfCoreMlResettable*>(session);
  if (resettable == nullptr) {
    if (error != nullptr) {
      *error = dinf_copy_string(
          "dinf_coreml_reset_state: session is not a Core ML session");
    }
    return 2;
  }
  std::string message;
  const int rc = resettable->ResetCoremlState(&message);
  if (rc != 0 && error != nullptr) {
    *error = dinf_copy_string(
        message.empty() ? "dinf_coreml_reset_state: reset failed" : message);
  }
  return rc;
}
