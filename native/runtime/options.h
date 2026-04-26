#pragma once

#include "runtime_bridge.h"

#include <string>

int dinf_option_int(const DinfOptions* options, const char* key, int fallback);

bool dinf_option_bool(
    const DinfOptions* options,
    const char* key,
    bool fallback);

std::string dinf_option_string(
    const DinfOptions* options,
    const char* key,
    const std::string& fallback = "");

bool dinf_options_contains_token(const DinfOptions* options, const char* token);
