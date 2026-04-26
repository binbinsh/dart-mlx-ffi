#pragma once

#include <string>

int dinf_option_int(const char* json, const char* key, int fallback);

bool dinf_option_bool(const char* json, const char* key, bool fallback);

std::string dinf_option_string(
    const char* json,
    const char* key,
    const std::string& fallback = "");

bool dinf_options_contains_token(const char* json, const char* token);
