#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

DEVICE_ID="${DEVICE_ID:-${1:-}}"
BUNDLE_ID="${BUNDLE_ID:-com.example.dartMlxFfiProbe}"
REPORTS_SOURCE="${REPORTS_SOURCE:-Documents/paddle_ocr_vl_reports}"
OUT_DIR="${OUT_DIR:-/tmp/ios_resume_driver}"
MAX_LAUNCHES="${MAX_LAUNCHES:-12}"
APP_PATH="${APP_PATH:-example/build/ios/iphoneos/Runner.app}"
INSTALL_APP="${INSTALL_APP:-0}"
POLL_SECS="${POLL_SECS:-5}"
WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-90}"
TARGET_CASES="${TARGET_CASES:-}"
RESET_CASES="${RESET_CASES:-}"
SEED_DOCS_DIR="${SEED_DOCS_DIR:-}"
EXPECTED_CASES="${EXPECTED_CASES:-}"
AUTO_BUILD_SEED="${AUTO_BUILD_SEED:-0}"
AUTO_SEED_OUT_ROOT="${AUTO_SEED_OUT_ROOT:-/tmp/paddle_ios_seed}"
FORCE_SEED="${FORCE_SEED:-0}"

if [[ -z "${DEVICE_ID}" ]]; then
  echo "usage: DEVICE_ID=<udid> $0" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

declare -a TARGET_CASE_LIST=()
declare -a EXPECTED_CASE_LIST=()
declare -a MONITORED_CASE_LIST=()

if [[ -n "${TARGET_CASES}" ]]; then
  IFS=',' read -r -a TARGET_CASE_LIST <<<"${TARGET_CASES}"
fi

if [[ -n "${EXPECTED_CASES}" ]]; then
  IFS=',' read -r -a EXPECTED_CASE_LIST <<<"${EXPECTED_CASES}"
fi

install_app_if_needed() {
  if [[ "${INSTALL_APP}" != "1" ]]; then
    return
  fi
  if [[ ! -d "${APP_PATH}" ]]; then
    echo "skip install: app bundle not found at ${APP_PATH}" >&2
    return
  fi
  echo "installing ${APP_PATH}"
  xcrun devicectl device install app \
    --device "${DEVICE_ID}" \
    "${APP_PATH}" >/dev/null
}

collect_seed_cases() {
  local cases_root="$1"
  if [[ ! -d "${cases_root}" ]]; then
    return
  fi
  find "${cases_root}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
}

device_subdir_contains() {
  local subdir="$1"
  local needle="$2"
  local tmp="${OUT_DIR}/device_probe.txt"
  if ! xcrun devicectl device info files \
    --device "${DEVICE_ID}" \
    --domain-type appDataContainer \
    --domain-identifier "${BUNDLE_ID}" \
    --subdirectory "${subdir}" >"${tmp}" 2>/dev/null; then
    return 1
  fi
  grep -Fq "${needle}" "${tmp}"
}

device_has_seed_payload() {
  if [[ "${FORCE_SEED}" == "1" ]]; then
    return 1
  fi
  if ! device_subdir_contains "Documents/paddle_ocr_vl_model" "model.safetensors"; then
    return 1
  fi

  local required_cases=()
  if (( ${#TARGET_CASE_LIST[@]} > 0 )); then
    required_cases=("${TARGET_CASE_LIST[@]}")
  elif [[ -n "${SEED_DOCS_DIR}" && -d "${SEED_DOCS_DIR}/paddle_ocr_vl_cases" ]]; then
    while IFS= read -r case_name; do
      [[ -n "${case_name}" ]] || continue
      required_cases+=("${case_name}")
    done < <(collect_seed_cases "${SEED_DOCS_DIR}/paddle_ocr_vl_cases")
  fi

  local case_name
  for case_name in "${required_cases[@]}"; do
    if ! device_subdir_contains \
      "Documents/paddle_ocr_vl_cases/${case_name}" \
      "input_ids.npy"; then
      return 1
    fi
    if ! device_subdir_contains \
      "Documents/paddle_ocr_vl_cases/${case_name}" \
      "image_nhwc.npy"; then
      return 1
    fi
  done

  return 0
}

seed_documents_if_needed() {
  if [[ "${SEED_DOCS_DIR}" == "auto" ]]; then
    SEED_DOCS_DIR="${AUTO_SEED_OUT_ROOT}/Documents"
  fi
  if [[ "${AUTO_BUILD_SEED}" == "1" && -z "${SEED_DOCS_DIR}" ]]; then
    SEED_DOCS_DIR="${AUTO_SEED_OUT_ROOT}/Documents"
  fi
  if [[ -z "${SEED_DOCS_DIR}" ]]; then
    return
  fi
  if [[ ! -d "${SEED_DOCS_DIR}" ]]; then
    if [[ "${AUTO_BUILD_SEED}" == "1" || "${SEED_DOCS_DIR}" == "${AUTO_SEED_OUT_ROOT}/Documents" ]]; then
      local cases_value="${TARGET_CASES}"
      if [[ -z "${cases_value}" && -n "${EXPECTED_CASES}" ]]; then
        cases_value="${EXPECTED_CASES}"
      fi
      echo "building local seed into ${AUTO_SEED_OUT_ROOT}"
      if [[ -n "${cases_value}" ]]; then
        OUT_ROOT="${AUTO_SEED_OUT_ROOT}" CASES="${cases_value}" \
          "${SCRIPT_DIR}/build_ios_seed.sh"
      else
        OUT_ROOT="${AUTO_SEED_OUT_ROOT}" "${SCRIPT_DIR}/build_ios_seed.sh"
      fi
    fi
  fi
  if [[ ! -d "${SEED_DOCS_DIR}" ]]; then
    echo "seed docs dir not found: ${SEED_DOCS_DIR}" >&2
    exit 1
  fi
  if device_has_seed_payload; then
    echo "seed skipped: device already has required payload"
    return
  fi
  echo "seeding Documents from ${SEED_DOCS_DIR}"
  xcrun devicectl device copy to \
    --device "${DEVICE_ID}" \
    --source "${SEED_DOCS_DIR}" \
    --domain-type appDataContainer \
    --domain-identifier "${BUNDLE_ID}" \
    --destination Documents >/dev/null
}

load_expected_cases_from_seed() {
  if [[ -z "${SEED_DOCS_DIR}" ]]; then
    return
  fi
  local cases_root="${SEED_DOCS_DIR}/paddle_ocr_vl_cases"
  if [[ ! -d "${cases_root}" ]]; then
    return
  fi
  while IFS= read -r case_name; do
    [[ -n "${case_name}" ]] || continue
    EXPECTED_CASE_LIST+=("${case_name}")
  done < <(find "${cases_root}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
}

load_expected_cases_from_device() {
  local tmp="${OUT_DIR}/device_cases.txt"
  if ! xcrun devicectl device info files \
    --device "${DEVICE_ID}" \
    --domain-type appDataContainer \
    --domain-identifier "${BUNDLE_ID}" \
    --subdirectory Documents/paddle_ocr_vl_cases >"${tmp}" 2>/dev/null; then
    return
  fi
  while IFS= read -r case_name; do
    [[ -n "${case_name}" ]] || continue
    EXPECTED_CASE_LIST+=("${case_name}")
  done < <(
    awk '
      $0 ~ /Directory/ &&
      $1 != "Name" &&
      $1 != "-----------------------------------------------" &&
      $1 != "files:" {
        print $1
      }
    ' "${tmp}" | sort -u
  )
}

load_monitored_cases() {
  if (( ${#TARGET_CASE_LIST[@]} > 0 )); then
    MONITORED_CASE_LIST=("${TARGET_CASE_LIST[@]}")
    return
  fi
  if (( ${#EXPECTED_CASE_LIST[@]} == 0 )); then
    load_expected_cases_from_seed
  fi
  if (( ${#EXPECTED_CASE_LIST[@]} == 0 )); then
    load_expected_cases_from_device
  fi
  if (( ${#EXPECTED_CASE_LIST[@]} > 0 )); then
    MONITORED_CASE_LIST=()
    while IFS= read -r case_name; do
      [[ -n "${case_name}" ]] || continue
      MONITORED_CASE_LIST+=("${case_name}")
    done < <(printf '%s\n' "${EXPECTED_CASE_LIST[@]}" | awk 'NF{print}' | sort -u)
  fi
}

copy_reports() {
  local destination="$1"
  rm -rf "${destination}"
  mkdir -p "${destination}"
  if ! xcrun devicectl device copy from \
    --device "${DEVICE_ID}" \
    --domain-type appDataContainer \
    --domain-identifier "${BUNDLE_ID}" \
    --source "${REPORTS_SOURCE}" \
    --destination "${destination}" >/dev/null 2>&1; then
    return 0
  fi
}

snapshot_fingerprint() {
  local snapshot="$1"
  if (( ${#MONITORED_CASE_LIST[@]} > 0 )); then
    local case_name
    for case_name in "${MONITORED_CASE_LIST[@]}"; do
      local report="${snapshot}/${case_name}/report.json"
      if [[ -f "${report}" ]]; then
        jq -r '[
          .name,
          .status,
          (.generated_tokens // 0),
          (.launch_count // 0),
          (.cumulative_ms // 0)
        ] | @tsv' "${report}"
      else
        printf '%s\tmissing\t0\t0\t0\n' "${case_name}"
      fi
    done
    return
  fi

  find "${snapshot}" -name report.json | sort | while read -r report; do
    jq -r '[
      .name,
      .status,
      (.generated_tokens // 0),
      (.launch_count // 0),
      (.cumulative_ms // 0)
    ] | @tsv' "${report}"
  done
}

all_cases_ok() {
  local snapshot="$1"
  if (( ${#MONITORED_CASE_LIST[@]} > 0 )); then
    local case_name
    for case_name in "${MONITORED_CASE_LIST[@]}"; do
      local report="${snapshot}/${case_name}/report.json"
      if [[ ! -f "${report}" ]]; then
        return 1
      fi
      if [[ "$(jq -r '.status' "${report}")" != "ok" ]]; then
        return 1
      fi
    done
    return 0
  fi

  local report_count
  report_count="$(find "${snapshot}" -name report.json | wc -l | tr -d ' ')"
  if [[ "${report_count}" == "0" ]]; then
    return 1
  fi
  local pending
  pending="$(
    find "${snapshot}" -name report.json -print0 |
      xargs -0 jq -r 'select(.status != "ok") | .name'
  )"
  [[ -z "${pending}" ]]
}

print_summary() {
  local snapshot="$1"
  echo "snapshot: ${snapshot}"
  if (( ${#MONITORED_CASE_LIST[@]} > 0 )); then
    local case_name
    for case_name in "${MONITORED_CASE_LIST[@]}"; do
      local report="${snapshot}/${case_name}/report.json"
      if [[ -f "${report}" ]]; then
        jq -r '[
          .name,
          .status,
          (.generated_tokens|tostring),
          ((.generated_this_launch // 0)|tostring),
          ((.launch_count // 0)|tostring),
          ((.peak_bytes // -1)|tostring)
        ] | @tsv' "${report}" |
          awk -F '\t' '{printf("  %s\tstatus=%s\tgenerated=%s\tthis_launch=%s\tlaunches=%s\tpeak=%s\n",$1,$2,$3,$4,$5,$6)}'
      else
        printf '  %s\tstatus=missing\tgenerated=0\tthis_launch=0\tlaunches=0\tpeak=-1\n' "${case_name}"
      fi
    done
    return
  fi
  find "${snapshot}" -name report.json | sort | while read -r report; do
    jq -r '[
      .name,
      .status,
      (.generated_tokens|tostring),
      ((.generated_this_launch // 0)|tostring),
      ((.launch_count // 0)|tostring),
      ((.peak_bytes // -1)|tostring)
    ] | @tsv' "${report}" |
      awk -F '\t' '{printf("  %s\tstatus=%s\tgenerated=%s\tthis_launch=%s\tlaunches=%s\tpeak=%s\n",$1,$2,$3,$4,$5,$6)}'
  done
}

launch_app() {
  local reset_value="$1"
  local env_json='{}'
  if [[ -n "${TARGET_CASES}" || -n "${reset_value}" ]]; then
    env_json="$(
      jq -nc \
        --arg target "${TARGET_CASES}" \
        --arg reset "${reset_value}" \
        '
        {}
        + (if $target != "" then {"POCR_TARGET_CASES": $target} else {} end)
        + (if $reset != "" then {"POCR_RESET_CASES": $reset} else {} end)
        '
    )"
  fi

  if [[ "${env_json}" == "{}" ]]; then
    xcrun devicectl device process launch \
      --device "${DEVICE_ID}" \
      --terminate-existing \
      "${BUNDLE_ID}" >/dev/null
    return
  fi

  xcrun devicectl device process launch \
    --device "${DEVICE_ID}" \
    --terminate-existing \
    --environment-variables "${env_json}" \
    "${BUNDLE_ID}" >/dev/null
}

wait_for_progress() {
  local baseline="$1"
  local poll_snapshot="$2"
  local waited=0

  while (( waited < WAIT_TIMEOUT_SECS )); do
    sleep "${POLL_SECS}"
    copy_reports "${poll_snapshot}"
    local current
    current="$(snapshot_fingerprint "${poll_snapshot}")"
    if all_cases_ok "${poll_snapshot}"; then
      return 0
    fi
    if [[ -n "${current}" && "${current}" != "${baseline}" ]]; then
      return 0
    fi
    waited=$((waited + POLL_SECS))
  done

  return 1
}

install_app_if_needed
seed_documents_if_needed
load_monitored_cases

initial_snapshot="${OUT_DIR}/initial"
copy_reports "${initial_snapshot}"
print_summary "${initial_snapshot}"
if [[ -z "${RESET_CASES}" ]] && all_cases_ok "${initial_snapshot}"; then
  echo "all cases already completed"
  exit 0
fi

baseline_fingerprint="$(snapshot_fingerprint "${initial_snapshot}")"

if (( MAX_LAUNCHES <= 0 )); then
  echo "MAX_LAUNCHES <= 0, stopping after initial snapshot"
  exit 0
fi

for launch in $(seq 1 "${MAX_LAUNCHES}"); do
  echo "=== launch ${launch}/${MAX_LAUNCHES} ==="
  launch_reset=''
  if [[ "${launch}" == "1" ]]; then
    if [[ "${RESET_CASES}" == "target" ]]; then
      launch_reset="${TARGET_CASES}"
    else
      launch_reset="${RESET_CASES}"
    fi
  fi
  launch_app "${launch_reset}"

  poll_snapshot="${OUT_DIR}/poll_${launch}"
  if ! wait_for_progress "${baseline_fingerprint}" "${poll_snapshot}"; then
    echo "no report progress detected within ${WAIT_TIMEOUT_SECS}s" >&2
    print_summary "${poll_snapshot}"
    exit 2
  fi

  snapshot="${OUT_DIR}/launch_${launch}"
  rm -rf "${snapshot}"
  mv "${poll_snapshot}" "${snapshot}"
  print_summary "${snapshot}"
  baseline_fingerprint="$(snapshot_fingerprint "${snapshot}")"

  if all_cases_ok "${snapshot}"; then
    echo "all cases completed"
    exit 0
  fi
done

echo "max launches reached without completing all cases" >&2
exit 2
