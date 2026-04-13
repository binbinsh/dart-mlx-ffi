#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  tool/ios_pocr.sh --device <udid> [options]

Options:
  --device <udid>           Required iPhone/iPad device id.
  --seed <auto|path|skip>   Seed strategy. Default: skip.
  --case <name[,name...]>   Limit to specific cases.
  --reset <target|all|list> Reset reports before running.
  --status                  Show current device status only.
  --force-seed              Force seeding even if payload looks present.
  --install-app             Reinstall Runner.app before running.
  --max-launches <n>        Maximum launch count for resume driver.
  --poll-secs <n>           Poll interval in seconds.
  --wait-timeout <n>        Timeout waiting for progress in seconds.
  --out-dir <path>          Snapshot output directory.
  --build-seed-only         Build seed and stop.
  --help                    Show this message.

Examples:
  tool/ios_pocr.sh --device <udid> --seed auto
  tool/ios_pocr.sh --device <udid> --case photo_render_512 --reset target
  tool/ios_pocr.sh --seed auto --build-seed-only
EOF
}

DEVICE_ID=""
SEED_MODE="skip"
TARGET_CASES=""
RESET_CASES=""
STATUS_ONLY="0"
FORCE_SEED="0"
INSTALL_APP="0"
MAX_LAUNCHES=""
POLL_SECS=""
WAIT_TIMEOUT_SECS=""
OUT_DIR=""
BUILD_SEED_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE_ID="${2:-}"
      shift 2
      ;;
    --seed)
      SEED_MODE="${2:-}"
      shift 2
      ;;
    --case)
      TARGET_CASES="${2:-}"
      shift 2
      ;;
    --reset)
      RESET_CASES="${2:-}"
      shift 2
      ;;
    --status)
      STATUS_ONLY="1"
      shift
      ;;
    --force-seed)
      FORCE_SEED="1"
      shift
      ;;
    --install-app)
      INSTALL_APP="1"
      shift
      ;;
    --max-launches)
      MAX_LAUNCHES="${2:-}"
      shift 2
      ;;
    --poll-secs)
      POLL_SECS="${2:-}"
      shift 2
      ;;
    --wait-timeout)
      WAIT_TIMEOUT_SECS="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --build-seed-only)
      BUILD_SEED_ONLY="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${SEED_MODE}" in
  skip|auto)
    ;;
  *)
    if [[ ! -d "${SEED_MODE}" ]]; then
      echo "seed path not found: ${SEED_MODE}" >&2
      exit 1
    fi
    ;;
esac

if [[ "${BUILD_SEED_ONLY}" == "1" ]]; then
  if [[ "${SEED_MODE}" == "skip" ]]; then
    SEED_MODE="auto"
  fi
  if [[ "${SEED_MODE}" == "auto" ]]; then
    "${SCRIPT_DIR}/build_ios_seed.sh"
  else
    echo "seed path already provided: ${SEED_MODE}" >&2
  fi
  exit 0
fi

if [[ -z "${DEVICE_ID}" ]]; then
  echo "--device is required unless --build-seed-only is used" >&2
  usage >&2
  exit 1
fi

if [[ "${STATUS_ONLY}" == "1" && -z "${MAX_LAUNCHES}" ]]; then
  MAX_LAUNCHES="0"
fi

SEED_DOCS_DIR=""
if [[ "${SEED_MODE}" == "auto" ]]; then
  SEED_DOCS_DIR="auto"
elif [[ "${SEED_MODE}" != "skip" ]]; then
  SEED_DOCS_DIR="${SEED_MODE}"
fi

env_args=()
env_args+=("DEVICE_ID=${DEVICE_ID}")
env_args+=("INSTALL_APP=${INSTALL_APP}")
env_args+=("FORCE_SEED=${FORCE_SEED}")

if [[ -n "${SEED_DOCS_DIR}" ]]; then
  env_args+=("SEED_DOCS_DIR=${SEED_DOCS_DIR}")
fi
if [[ -n "${TARGET_CASES}" ]]; then
  env_args+=("TARGET_CASES=${TARGET_CASES}")
fi
if [[ -n "${RESET_CASES}" ]]; then
  env_args+=("RESET_CASES=${RESET_CASES}")
fi
if [[ -n "${MAX_LAUNCHES}" ]]; then
  env_args+=("MAX_LAUNCHES=${MAX_LAUNCHES}")
fi
if [[ -n "${POLL_SECS}" ]]; then
  env_args+=("POLL_SECS=${POLL_SECS}")
fi
if [[ -n "${WAIT_TIMEOUT_SECS}" ]]; then
  env_args+=("WAIT_TIMEOUT_SECS=${WAIT_TIMEOUT_SECS}")
fi
if [[ -n "${OUT_DIR}" ]]; then
  env_args+=("OUT_DIR=${OUT_DIR}")
fi

env "${env_args[@]}" "${SCRIPT_DIR}/ios_resume.sh"
