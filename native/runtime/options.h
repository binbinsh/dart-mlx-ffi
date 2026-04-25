#pragma once

#include <string>

int dmf_option_int(const char* json, const char* key, int fallback);

bool dmf_option_bool(const char* json, const char* key, bool fallback);

std::string dmf_option_string(
    const char* json,
    const char* key,
    const std::string& fallback = "");

bool dmf_options_contains_token(const char* json, const char* token);
