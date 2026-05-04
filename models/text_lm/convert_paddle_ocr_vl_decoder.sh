#!/usr/bin/env bash
# Wrapper around models/text_lm/convert_unsloth_mlx.py that produces a
# decoder-only MLX 4-bit snapshot of PaddlePaddle/PaddleOCR-VL-1.5 for the
# hybrid OCR runner (issue #1).
#
# What this does (high level):
#   1. Calls convert_unsloth_mlx.py with --skip-prefix visual. so the ViT
#      weights are dropped before MLX quantization. The result is just the
#      ERNIE-4.5 language backbone in 4-bit, plus tokenizer files.
#   2. Patches config.json so HF/MLX loaders treat it as a plain
#      Ernie4_5ForCausalLM (model_type=ernie4_5). The original
#      PaddleOCRVLForConditionalGeneration architectures string would
#      otherwise refuse to load against the stripped weights.
#   3. Prints next-step instructions for verifying with mlx_lm.generate
#      and uploading to Hugging Face. NEVER pushes anything itself.
#
# IDEMPOTENT:
#   - If --output-dir already contains *.safetensors and config.json, the
#     conversion is skipped. Use --force to rerun.
#   - The config.json patch is rerun on every invocation unless
#     --skip-config-patch is given. (Idempotent: patching an
#     already-patched config is a no-op apart from a stable rewrite.)
#
# This script does NOT auto-run heavy steps in a fresh repo — you must
# have HF cache populated or be willing to download ~5 GB. Run manually.
#
# Usage:
#   models/text_lm/convert_paddle_ocr_vl_decoder.sh \
#       [--output-dir DIR] [--input REPO_OR_DIR] \
#       [--imatrix-repo REPO --imatrix-file FILE] \
#       [--q-bits 4] [--q-group-size 64] \
#       [--force] [--skip-config-patch] [--dry-run]
#
# Defaults match the manifest expectation:
#   --input       PaddlePaddle/PaddleOCR-VL-1.5
#   --output-dir  ~/snapshots/paddleocr-vl-ernie-mlx-4bit
#   --q-bits 4 --q-group-size 64
#
# Refs: models/text_lm/convert_unsloth_mlx.py (CLI surface), and
# models/text_lm/test_convert_unsloth_mlx.py (--skip-prefix semantics).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT="PaddlePaddle/PaddleOCR-VL-1.5"
OUTPUT_DIR="${HOME}/snapshots/paddleocr-vl-ernie-mlx-4bit"
IMATRIX_REPO=""
IMATRIX_FILE=""
IMATRIX_PATH=""
Q_BITS=4
Q_GROUP_SIZE=64
DTYPE=""
MLX_CLI="mlx"
FORCE=0
SKIP_CONFIG_PATCH=0
DRY_RUN=0
NO_QUANTIZE=0

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)            INPUT="$2"; shift 2;;
    --output-dir)       OUTPUT_DIR="$2"; shift 2;;
    --imatrix-repo)     IMATRIX_REPO="$2"; shift 2;;
    --imatrix-file)     IMATRIX_FILE="$2"; shift 2;;
    --imatrix-path)     IMATRIX_PATH="$2"; shift 2;;
    --q-bits)           Q_BITS="$2"; shift 2;;
    --q-group-size)     Q_GROUP_SIZE="$2"; shift 2;;
    --dtype)            DTYPE="$2"; shift 2;;
    --mlx-cli)          MLX_CLI="$2"; shift 2;;
    --force)            FORCE=1; shift;;
    --skip-config-patch) SKIP_CONFIG_PATCH=1; shift;;
    --dry-run)          DRY_RUN=1; shift;;
    --no-quantize)      NO_QUANTIZE=1; shift;;
    -h|--help)          usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

needs_conversion=1
if [[ -f "${OUTPUT_DIR}/config.json" ]] \
   && compgen -G "${OUTPUT_DIR}/*.safetensors" >/dev/null; then
  needs_conversion=0
fi

if [[ "${FORCE}" -eq 1 ]]; then
  needs_conversion=1
fi

if [[ "${needs_conversion}" -eq 1 ]]; then
  echo "[convert_paddle_ocr_vl_decoder] running convert_unsloth_mlx.py"
  args=(
    --input "${INPUT}"
    --output-dir "${OUTPUT_DIR}"
    --skip-prefix "visual."
    --mlx-cli "${MLX_CLI}"
  )
  if [[ "${NO_QUANTIZE}" -eq 1 ]]; then
    args+=(--no-quantize)
  else
    args+=(--q-bits "${Q_BITS}" --q-group-size "${Q_GROUP_SIZE}")
    if [[ -n "${IMATRIX_PATH}" ]]; then
      args+=(--imatrix-path "${IMATRIX_PATH}")
    elif [[ -n "${IMATRIX_REPO}" && -n "${IMATRIX_FILE}" ]]; then
      args+=(--imatrix-repo "${IMATRIX_REPO}" --imatrix-file "${IMATRIX_FILE}")
    elif [[ -n "${IMATRIX_REPO}" ]]; then
      args+=(--imatrix-repo "${IMATRIX_REPO}")
    else
      cat >&2 <<'EOF'
[convert_paddle_ocr_vl_decoder] error: quantized conversion needs an imatrix.
Pass --imatrix-path PATH, or --imatrix-repo REPO [--imatrix-file FILE], or
re-run with --no-quantize for a non-quantized smoke test.
EOF
      exit 2
    fi
  fi
  if [[ -n "${DTYPE}" ]]; then
    args+=(--dtype "${DTYPE}")
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    args+=(--dry-run)
  fi

  uv run --no-project --with huggingface_hub --with safetensors --with numpy \
    python "${HERE}/convert_unsloth_mlx.py" "${args[@]}"
else
  echo "[convert_paddle_ocr_vl_decoder] reusing existing snapshot at ${OUTPUT_DIR}"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[convert_paddle_ocr_vl_decoder] dry-run: skipping config.json patch"
  exit 0
fi

if [[ "${SKIP_CONFIG_PATCH}" -eq 0 ]]; then
  echo "[convert_paddle_ocr_vl_decoder] patching config.json -> Ernie4_5ForCausalLM"
  uv run --no-project python "${HERE}/patch_decoder_config.py" \
    --config "${OUTPUT_DIR}/config.json"
else
  echo "[convert_paddle_ocr_vl_decoder] --skip-config-patch: leaving config.json untouched"
fi

cat <<EOF

[convert_paddle_ocr_vl_decoder] done. Snapshot at:
  ${OUTPUT_DIR}

Next steps (run manually — this script will not push):

  # 1. Smoke-test load with mlx_lm. The patched config makes this work
  #    without trust_remote_code.
  uv run --no-project --with mlx-lm python -c \\
    "from mlx_lm import load, generate; m, t = load('${OUTPUT_DIR}'); \\
     print(generate(m, t, prompt='Hello', max_tokens=8))"

  # 2. Inspect tensor keys to confirm visual.* is gone.
  uv run --no-project --with safetensors python -c \\
    "from safetensors import safe_open; import glob; \\
     [print(k) for f in glob.glob('${OUTPUT_DIR}/*.safetensors') \\
      for k in safe_open(f, framework='numpy').keys() if k.startswith('visual.')]"
  # ^ should print nothing.

  # 3. Upload to your HF namespace (replace <USER>):
  huggingface-cli upload <USER>/paddleocr-vl-ernie-mlx-4bit \\
    "${OUTPUT_DIR}" . --repo-type model

  # 4. After upload, edit lib/src/models/shared/manifest.dart per
  #    docs/refactor/13-manifest-patch.md (mlxRepo + runner + tags).
EOF
