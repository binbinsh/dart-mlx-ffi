#include "runtime_bridge.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

#if defined(__linux__)
uint64_t proc_status_kb(const char* key) {
  std::ifstream file("/proc/self/status");
  std::string name;
  uint64_t value = 0;
  std::string unit;
  while (file >> name >> value >> unit) {
    if (name == key) {
      return value * 1024;
    }
  }
  return 0;
}
#endif

#if defined(__ANDROID__)
struct AndroidSmapsTotals {
  uint64_t pss = 0;
  uint64_t rss = 0;
  uint64_t native_heap_pss = 0;
  uint64_t java_heap_pss = 0;
  uint64_t native_heap_private_dirty = 0;
  uint64_t java_heap_private_dirty = 0;
};

bool starts_with_hex_address(const std::string& line) {
  if (line.empty()) {
    return false;
  }
  for (const char c : line) {
    if (c == '-') {
      return true;
    }
    const bool hex = (c >= '0' && c <= '9') ||
                     (c >= 'a' && c <= 'f') ||
                     (c >= 'A' && c <= 'F');
    if (!hex) {
      return false;
    }
  }
  return false;
}

bool contains_any(
    const std::string& value,
    const std::vector<const char*>& needles) {
  for (const char* needle : needles) {
    if (value.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool is_native_heap_mapping(const std::string& header) {
  return contains_any(
      header,
      {"[heap]", "libc_malloc", "scudo", "jemalloc", "malloc"});
}

bool is_java_heap_mapping(const std::string& header) {
  return contains_any(header, {"dalvik", "art heap", "zygote", "jit-code-cache"});
}

uint64_t smaps_kb_value(const std::string& line, const char* key) {
  if (line.rfind(key, 0) != 0) {
    return 0;
  }
  std::istringstream in(line.substr(std::string(key).size()));
  uint64_t value = 0;
  in >> value;
  return value * 1024;
}

AndroidSmapsTotals read_android_smaps() {
  std::ifstream file("/proc/self/smaps");
  AndroidSmapsTotals totals;
  std::string header;
  bool native_heap = false;
  bool java_heap = false;
  std::string line;
  while (std::getline(file, line)) {
    if (starts_with_hex_address(line)) {
      header = line;
      native_heap = is_native_heap_mapping(header);
      java_heap = is_java_heap_mapping(header);
      continue;
    }
    const uint64_t pss = smaps_kb_value(line, "Pss:");
    if (pss != 0) {
      totals.pss += pss;
      if (native_heap) {
        totals.native_heap_pss += pss;
      }
      if (java_heap) {
        totals.java_heap_pss += pss;
      }
      continue;
    }
    const uint64_t rss = smaps_kb_value(line, "Rss:");
    if (rss != 0) {
      totals.rss += rss;
      continue;
    }
    const uint64_t private_dirty = smaps_kb_value(line, "Private_Dirty:");
    if (private_dirty != 0) {
      if (native_heap) {
        totals.native_heap_private_dirty += private_dirty;
      }
      if (java_heap) {
        totals.java_heap_private_dirty += private_dirty;
      }
    }
  }
  return totals;
}

uint64_t update_peak(uint64_t value) {
  static uint64_t peak = 0;
  if (value > peak) {
    peak = value;
  }
  return peak;
}
#endif

}  // namespace

extern "C" void dinf_cpp_mem(DinfMemoryInfo* out) {
  if (out == nullptr) {
    return;
  }
  *out = DinfMemoryInfo{};
#if defined(__APPLE__)
  task_vm_info_data_t vm_info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  kern_return_t status = task_info(
      mach_task_self(),
      TASK_VM_INFO,
      reinterpret_cast<task_info_t>(&vm_info),
      &count);
  if (status == KERN_SUCCESS) {
    out->peak_memory_bytes = vm_info.phys_footprint;
    out->phys_footprint = vm_info.phys_footprint;
    out->resident_size = vm_info.resident_size;
    out->virtual_size = vm_info.virtual_size;
  }
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX counters{};
  if (GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters))) {
    out->peak_memory_bytes = counters.PeakWorkingSetSize;
    out->peak_working_set = counters.PeakWorkingSetSize;
    out->working_set = counters.WorkingSetSize;
  }
#elif defined(__ANDROID__)
  const AndroidSmapsTotals totals = read_android_smaps();
  const uint64_t peak_pss = update_peak(totals.pss);
  out->peak_memory_bytes = peak_pss;
  out->android_peak_pss = peak_pss;
  out->android_pss = totals.pss;
  out->android_rss = totals.rss;
  out->android_native_heap_pss = totals.native_heap_pss;
  out->android_java_heap_pss = totals.java_heap_pss;
  out->android_native_heap_private_dirty = totals.native_heap_private_dirty;
  out->android_java_heap_private_dirty = totals.java_heap_private_dirty;
#elif defined(__linux__)
  const uint64_t peak = proc_status_kb("VmHWM:");
  const uint64_t rss = proc_status_kb("VmRSS:");
  out->peak_memory_bytes = peak;
  out->vm_hwm = peak;
  out->vm_rss = rss;
#endif
}
