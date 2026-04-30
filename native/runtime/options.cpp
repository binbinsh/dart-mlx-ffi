#include "options.h"

#include <cstring>
#include <string>

namespace {

const DinfOptionEntry* find_entry(
    const DinfOptions* options,
    const char* key) {
  if (options == nullptr || options->entries == nullptr ||
      options->count <= 0 || key == nullptr) {
    return nullptr;
  }
  for (intptr_t i = 0; i < options->count; ++i) {
    const DinfOptionEntry& entry = options->entries[i];
    if (entry.path != nullptr && std::strcmp(entry.path, key) == 0) {
      return &entry;
    }
  }
  return nullptr;
}

std::string entry_text(const DinfOptionEntry& entry) {
  return entry.text == nullptr ? std::string() : std::string(entry.text);
}

}  // namespace

int dinf_option_int(
    const DinfOptions* options,
    const char* key,
    int fallback) {
  const DinfOptionEntry* entry = find_entry(options, key);
  if (entry == nullptr) {
    return fallback;
  }
  if (entry->kind == DINF_OPTION_INT) {
    return static_cast<int>(entry->int_value);
  }
  if (entry->kind == DINF_OPTION_DOUBLE) {
    return static_cast<int>(entry->double_value);
  }
  return fallback;
}

bool dinf_option_bool(
    const DinfOptions* options,
    const char* key,
    bool fallback) {
  const DinfOptionEntry* entry = find_entry(options, key);
  if (entry == nullptr || entry->kind != DINF_OPTION_BOOL) {
    return fallback;
  }
  return entry->bool_value != 0;
}

bool dinf_option_exists(const DinfOptions* options, const char* key) {
  return find_entry(options, key) != nullptr;
}

std::string dinf_option_string(
    const DinfOptions* options,
    const char* key,
    const std::string& fallback) {
  const DinfOptionEntry* entry = find_entry(options, key);
  if (entry == nullptr || entry->kind != DINF_OPTION_STRING) {
    return fallback;
  }
  const std::string value = entry_text(*entry);
  return value.empty() ? fallback : value;
}

bool dinf_options_contains_token(
    const DinfOptions* options,
    const char* token) {
  if (options == nullptr || options->entries == nullptr ||
      options->count <= 0 || token == nullptr) {
    return false;
  }
  for (intptr_t i = 0; i < options->count; ++i) {
    const DinfOptionEntry& entry = options->entries[i];
    if (entry.kind == DINF_OPTION_STRING && entry.text != nullptr &&
        std::strcmp(entry.text, token) == 0) {
      return true;
    }
  }
  return false;
}
