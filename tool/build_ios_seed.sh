#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PLATFORM_ROOT="$(cd "${PACKAGE_DIR}/../../.." && pwd)"
DEFAULT_OCR_FIXTURE="${MODEL_PLATFORM_ROOT}/models/validation/runtime/fixtures/image.png"

MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--mlx-community--PaddleOCR-VL-1.5-8bit/snapshots/37d4c85284434b6e6fd4c03f8b719b1aefaa013c}"
OUT_ROOT="${OUT_ROOT:-/tmp/paddle_ios_seed}"
DOCS_ROOT="${OUT_ROOT}/Documents"
MODEL_OUT="${DOCS_ROOT}/paddle_ocr_vl_model"
CASES_OUT="${DOCS_ROOT}/paddle_ocr_vl_cases"

PHOTO_IMAGE="${PHOTO_IMAGE:-${DEFAULT_OCR_FIXTURE}}"
PHOTO_MAX_PIXELS="${PHOTO_MAX_PIXELS:-401408}"

RECIPE_IMAGE="${RECIPE_IMAGE:-${DEFAULT_OCR_FIXTURE}}"
RECIPE_MAX_PIXELS="${RECIPE_MAX_PIXELS:-501760}"

CASES="${CASES:-photo_render_512,recipe_ref_501760}"

if [[ ! -d "${MODEL_SNAPSHOT}" ]]; then
  echo "model snapshot not found: ${MODEL_SNAPSHOT}" >&2
  exit 1
fi

rm -rf "${OUT_ROOT}"
mkdir -p "${MODEL_OUT}" "${CASES_OUT}"

echo "copying model snapshot to ${MODEL_OUT}"
cp -RL "${MODEL_SNAPSHOT}/." "${MODEL_OUT}/"

build_case() {
  local case_name="$1"
  local image_path="$2"
  local max_pixels="$3"

  if [[ ! -f "${image_path}" ]]; then
    echo "image not found for ${case_name}: ${image_path}" >&2
    exit 1
  fi

  local out_dir="${CASES_OUT}/${case_name}"
  mkdir -p "${out_dir}"
  echo "building ${case_name} from ${image_path}"
  uv run --no-project --with mlx-vlm --with pillow \
    python tool/dump_paddle_v15_reference.py \
    --image "${image_path}" \
    --out-dir "${out_dir}" \
    --max-pixels "${max_pixels}"
}

IFS=',' read -r -a case_list <<<"${CASES}"
for case_name in "${case_list[@]}"; do
  case_name="$(echo "${case_name}" | xargs)"
  case "${case_name}" in
    photo_render_512)
      build_case "${case_name}" "${PHOTO_IMAGE}" "${PHOTO_MAX_PIXELS}"
      ;;
    recipe_ref_501760)
      build_case "${case_name}" "${RECIPE_IMAGE}" "${RECIPE_MAX_PIXELS}"
      ;;
    '')
      ;;
    *)
      echo "unsupported case: ${case_name}" >&2
      exit 1
      ;;
  esac
done

echo "seed ready at ${DOCS_ROOT}"
