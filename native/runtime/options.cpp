#include "options.h"

#include <cctype>
#include <cstring>
#include <cstdlib>
#include <string>

namespace {

const char* find_key(const char* json, const char* key) {
  if (json == nullptr || key == nullptr) {
    return nullptr;
  }
  const std::string needle = std::string("\"") + key + "\"";
  const std::string haystack(json);
  const auto pos = haystack.find(needle);
  if (pos == std::string::npos) {
    return nullptr;
  }
  const auto colon = haystack.find(':', pos + needle.size());
  if (colon == std::string::npos) {
    return nullptr;
  }
  const char* value = json + colon + 1;
  while (*value != '\0' && std::isspace(static_cast<unsigned char>(*value))) {
    ++value;
  }
  return value;
}

std::string unescape_json_string(const char* value) {
  std::string out;
  if (value == nullptr || *value != '"') {
    return out;
  }
  for (const char* cursor = value + 1; *cursor != '\0'; ++cursor) {
    if (*cursor == '"') {
      break;
    }
    if (*cursor == '\\' && cursor[1] != '\0') {
      ++cursor;
    }
    out.push_back(*cursor);
  }
  return out;
}

}  // namespace

int dinf_option_int(const char* json, const char* key, int fallback) {
  const char* value = find_key(json, key);
  if (value == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end == value ? fallback : static_cast<int>(parsed);
}

bool dinf_option_bool(const char* json, const char* key, bool fallback) {
  const char* value = find_key(json, key);
  if (value == nullptr) {
    return fallback;
  }
  if (std::strncmp(value, "true", 4) == 0) {
    return true;
  }
  if (std::strncmp(value, "false", 5) == 0) {
    return false;
  }
  return fallback;
}

std::string dinf_option_string(
    const char* json,
    const char* key,
    const std::string& fallback) {
  const char* value = find_key(json, key);
  const std::string parsed = unescape_json_string(value);
  return parsed.empty() ? fallback : parsed;
}

bool dinf_options_contains_token(const char* json, const char* token) {
  if (json == nullptr || token == nullptr) {
    return false;
  }
  const std::string needle = std::string("\"") + token + "\"";
  return std::string(json).find(needle) != std::string::npos;
}
