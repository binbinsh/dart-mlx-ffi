"""MLX-decoder vs HF-reference per-step top-1 parity audit.

This script closes the test-coverage gap that opens once commit #11 of the
hybrid-OCR refactor (issue #1) retires the CoreML decode stage and the
existing Stage C parity check in
``models/validation/runtime/converters/paddleocr_vl_coreml/parity.py``. After that
removal the only path through the decoder is the new MLX runner driven by
``PaddleOcrVlHybridRunner``; this script asserts numerical agreement of
that path against the HF PyTorch reference for K greedy decode steps.

Pipeline
--------
1. Resolve a HF snapshot for the given model id (or honour an explicit
   ``--snapshot`` path).
2. Load the HF model + processor in fp32 and run ONE forward + K greedy
   continuations to capture per-step argmax tokens. The HF forward mirrors
   ``_compute_hf_reference`` in ``parity.py`` with one twist: we keep the
   natural HF input-id layout (no image-first reshuffle) so Dart's
   ``generateFromVisionFeaturesDetailed`` — which scatters image features
   in-place at placeholder positions — sees the same effective layout.
3. Dump the projector output (``image_embeds``) and prompt token ids to
   ``.npy`` files in a temp directory, then spawn ``dart_decode_dump.dart``
   to run the same K decode steps through the MLX runner. The Dart entry
   loads the decoder with ``keepVisionWeights: false`` (the hybrid load
   mode) and consumes the HF-projected image features directly via
   ``PaddleOcrVlRunner.generateFromVisionFeaturesDetailed`` — so we are
   genuinely measuring the decoder, not the ViT.
4. Compare HF top-1 vs MLX top-1 per step. Hard-fail if any of the first
   ``--strict-steps`` (default 3) steps disagree; warn-only for later
   steps where 8-bit MLX quantization can introduce divergence.

Soft-skip behaviour
-------------------
The script is designed to be invoked from CI without preflight: if the HF
deps are unimportable, the snapshot is missing, or the benchmark image is
not on disk, it exits 0 with a JSON payload containing
``{"skipped": true, "reason": ...}``. Real numerical mismatches still
exit 1.

CLI
---
::

    uv run --no-project --with mlx-lm --with pillow --with mlx-vlm \\
        python benchmark/paddle_ocr_vl/mlx_decode_parity.py \\
        --model-id PaddlePaddle/PaddleOCR-VL-1.5 \\
        --image benchmark/assets/paddle_ocr_vl_test.jpg \\
        --steps 8 \\
        --out /tmp/mlx_decode_parity.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths / constants — kept in sync with parity.py.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_IMAGE = ROOT / "benchmark" / "assets" / "paddle_ocr_vl_test.jpg"
DART_DUMP_SCRIPT = (
    Path(__file__).resolve().parent / "dart_decode_dump.dart"
)
IMAGE_TOKEN_ID = 100295  # parity.py mirror

# Default decode budget. ``parity.py`` defaults to 8 too.
DEFAULT_STEPS = 8
DEFAULT_STRICT_STEPS = 3

# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class ParityReport:
    steps: int
    strict_steps: int
    hf_tokens: list[int] = field(default_factory=list)
    mlx_tokens: list[int] = field(default_factory=list)
    agreement_indices: list[int] = field(default_factory=list)
    mismatch_indices: list[int] = field(default_factory=list)
    strict_passed: bool = False
    soft_passed: bool = False
    snapshot: str = ""
    image: str = ""
    notes: str = ""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "HF model id to resolve via benchmark/common.find_cached_snapshot. "
            "Ignored when --snapshot is given."
        ),
    )
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Explicit path to a HF/MLX snapshot directory.",
    )
    ap.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to a JPG/PNG image used as the parity input.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of greedy decode steps to compare (default: 8).",
    )
    ap.add_argument(
        "--strict-steps",
        type=int,
        default=DEFAULT_STRICT_STEPS,
        help=(
            "Hard-fail if any of the first N steps mismatch. Later steps "
            "warn-only (default: 3)."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON parity report.",
    )
    ap.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the intermediate .npy files in --tmp-dir.",
    )
    ap.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Reuse a fixed scratch directory instead of mkdtemp.",
    )
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override processor.image_processor.max_pixels (test-only).",
    )
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Soft-skip helpers
# --------------------------------------------------------------------------- #
def _emit(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


def _skip(reason: str, out: Path | None) -> int:
    _emit({"skipped": True, "reason": reason}, out)
    return 0


def _resolve_snapshot(model_id: str, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    try:
        sys.path.insert(0, str(ROOT / "benchmark"))
        from common import find_cached_snapshot  # type: ignore[import-not-found]
    except Exception:
        return None
    return find_cached_snapshot(model_id)


# --------------------------------------------------------------------------- #
# HF reference forward — narrow port of parity.py:_compute_hf_reference.
# --------------------------------------------------------------------------- #
def _hf_reference(
    snapshot: Path,
    image_path: Path,
    steps: int,
    max_pixels: int | None,
) -> dict[str, Any]:
    """Drive the HF PaddleOCR-VL model end-to-end and return:

    - ``image_embeds`` (M, hidden) fp32 numpy
    - ``input_ids`` (S,) int64 numpy — natural HF layout, no padding
    - ``grid_thw`` list[int] of length 3
    - ``decode_tokens`` list[int] of length steps + 1 (seed + K)
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import (  # type: ignore[import-not-found]
        AutoProcessor,
        PaddleOCRVLForConditionalGeneration,
    )
    from transformers.cache_utils import (  # type: ignore[import-not-found]
        DynamicCache,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    processor = AutoProcessor.from_pretrained(str(snapshot))
    if max_pixels is not None:
        processor.image_processor.max_pixels = max_pixels

    model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        str(snapshot),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inner = model.model

    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    proc = processor(images=[img], text=[text], return_tensors="pt")
    input_ids = proc["input_ids"]                # (1, S) int64
    pixel_values = proc["pixel_values"]          # (N, 3, P, P) fp32
    image_grid_thw = proc["image_grid_thw"]      # (1, 3) int64

    # Vision -> projector -> image_embeds (M, hidden) fp32.
    image_outputs = inner.get_image_features(
        pixel_values=pixel_values.detach().clone(),
        image_grid_thw=image_grid_thw,
        return_dict=True,
    )
    image_embeds = image_outputs.pooler_output.to(torch.float32)
    img_mask = (input_ids == IMAGE_TOKEN_ID)
    n_img = int(img_mask.sum().item())
    if n_img != image_embeds.shape[0]:
        raise RuntimeError(
            f"image-token count mismatch: prompt has {n_img} placeholders "
            f"but projector produced {image_embeds.shape[0]} features"
        )

    # Scatter image_embeds into text-embed buffer at placeholder positions.
    text_embeds = inner.language_model.embed_tokens(input_ids)
    fused = text_embeds.clone()
    fused[img_mask] = image_embeds.to(text_embeds.dtype)

    # HF-native position_ids via get_rope_index (matches what the model
    # itself feeds its rotary_emb in the upstream forward).
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    mm_token_type_ids[img_mask] = 1
    attention_mask = torch.ones_like(input_ids)
    position_ids, _rope_deltas_t = inner.get_rope_index(
        input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    real_len = int(input_ids.shape[1])

    # Prefill via language_model with DynamicCache; capture seed token.
    cache = DynamicCache()
    out = inner.language_model(
        input_ids=None,
        inputs_embeds=fused.to(torch.float32),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    last_logits = (
        model.lm_head(out.last_hidden_state)[0, -1].to(torch.float32).numpy()
    )
    seed_token = int(np.argmax(last_logits))
    past_kv = out.past_key_values

    # K greedy decode steps, mirroring parity.py Stage C.
    rope_deltas_int = (
        int(position_ids[..., :real_len].max().item()) + 1 - real_len
    )
    decode_tokens: list[int] = [seed_token]
    cur = seed_token
    for step in range(steps):
        tok_embed = inner.language_model.embed_tokens(
            torch.tensor([[cur]], dtype=torch.int64)
        ).to(torch.float32)
        cache_pos = real_len + step
        pos_int = cache_pos + rope_deltas_int
        position_ids_step = (
            torch.tensor([[[pos_int]]], dtype=torch.int64)
            .expand(3, 1, 1)
            .contiguous()
        )
        attn_step = torch.ones(1, real_len + step + 1, dtype=torch.int64)
        out = inner.language_model(
            input_ids=None,
            inputs_embeds=tok_embed,
            attention_mask=attn_step,
            position_ids=position_ids_step,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
            cache_position=torch.tensor([cache_pos], dtype=torch.int64),
        )
        past_kv = out.past_key_values
        step_logits = (
            model.lm_head(out.last_hidden_state)[0, 0]
            .to(torch.float32)
            .numpy()
        )
        cur = int(np.argmax(step_logits))
        decode_tokens.append(cur)

    return {
        "image_embeds": image_embeds.detach().cpu().numpy(),
        "input_ids": input_ids[0].detach().cpu().numpy().astype("int32"),
        "grid_thw": [int(v) for v in image_grid_thw[0].tolist()],
        "decode_tokens": decode_tokens,
    }


# --------------------------------------------------------------------------- #
# Dart driver
# --------------------------------------------------------------------------- #
def _run_dart_dump(
    *,
    snapshot: Path,
    image_embeds_path: Path,
    prompt_ids_path: Path,
    grid_thw: list[int],
    steps: int,
    out_json: Path,
) -> dict[str, Any]:
    cmd = [
        "dart",
        "run",
        str(DART_DUMP_SCRIPT),
        f"--snapshot={snapshot}",
        f"--image-embeds={image_embeds_path}",
        f"--prompt-ids={prompt_ids_path}",
        f"--grid-thw={grid_thw[0]},{grid_thw[1]},{grid_thw[2]}",
        f"--steps={steps}",
        f"--out={out_json}",
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "dart_decode_dump.dart failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{completed.stdout}\n"
            f"  stderr:\n{completed.stderr}"
        )
    return json.loads(out_json.read_text())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = _parse_args()

    if not args.image.exists():
        return _skip(f"image not found: {args.image}", args.out)

    snapshot = _resolve_snapshot(args.model_id, args.snapshot)
    if snapshot is None or not snapshot.exists():
        return _skip(
            f"snapshot not found for model_id={args.model_id} "
            f"(explicit={args.snapshot})",
            args.out,
        )

    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return _skip(f"HF deps unavailable: {e!r}", args.out)

    if not DART_DUMP_SCRIPT.exists():
        return _skip(
            f"dart_decode_dump.dart not found at {DART_DUMP_SCRIPT}",
            args.out,
        )

    rep = ParityReport(
        steps=int(args.steps),
        strict_steps=int(args.strict_steps),
        snapshot=str(snapshot),
        image=str(args.image),
    )

    t0 = time.time()
    try:
        hf = _hf_reference(
            snapshot=snapshot,
            image_path=args.image,
            steps=args.steps,
            max_pixels=args.max_pixels,
        )
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        rep.notes = f"HF reference failed: {e!r}\n{tb}"
        _emit({"error": rep.notes, **asdict(rep)}, args.out)
        return 1
    print(f"[parity] HF reference computed in {time.time()-t0:.2f}s", flush=True)

    rep.hf_tokens = list(map(int, hf["decode_tokens"]))

    tmp_root = (
        Path(args.tmp_dir).resolve()
        if args.tmp_dir is not None
        else Path(tempfile.mkdtemp(prefix="mlx_decode_parity_"))
    )
    tmp_root.mkdir(parents=True, exist_ok=True)
    image_embeds_path = tmp_root / "image_embeds.npy"
    prompt_ids_path = tmp_root / "prompt_ids.npy"
    dart_out_path = tmp_root / "dart_tokens.json"

    import numpy as np

    np.save(image_embeds_path, hf["image_embeds"].astype("float32"))
    np.save(prompt_ids_path, hf["input_ids"].astype("int32"))

    try:
        t0 = time.time()
        dart_payload = _run_dart_dump(
            snapshot=snapshot,
            image_embeds_path=image_embeds_path,
            prompt_ids_path=prompt_ids_path,
            grid_thw=hf["grid_thw"],
            steps=args.steps,
            out_json=dart_out_path,
        )
        print(
            f"[parity] Dart MLX decode completed in {time.time()-t0:.2f}s",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        rep.notes = f"Dart decode failed: {e!r}"
        _emit({"error": rep.notes, **asdict(rep)}, args.out)
        return 1
    finally:
        if not args.keep_tmp and args.tmp_dir is None:
            for p in (image_embeds_path, prompt_ids_path, dart_out_path):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
            try:
                tmp_root.rmdir()
            except OSError:
                pass

    rep.mlx_tokens = list(map(int, dart_payload.get("decode_tokens", [])))
    if len(rep.mlx_tokens) < args.steps + 1:
        rep.notes = (
            f"Dart returned {len(rep.mlx_tokens)} tokens, expected "
            f"{args.steps + 1} (seed + {args.steps})"
        )
        _emit({"error": rep.notes, **asdict(rep)}, args.out)
        return 1

    # Compare position-by-position. Index 0 is the seed (prefill last
    # token); 1..K are the K greedy continuations.
    horizon = min(len(rep.hf_tokens), len(rep.mlx_tokens), args.steps + 1)
    for i in range(horizon):
        if rep.hf_tokens[i] == rep.mlx_tokens[i]:
            rep.agreement_indices.append(i)
        else:
            rep.mismatch_indices.append(i)

    strict_horizon = min(args.strict_steps + 1, horizon)  # +1 to cover seed
    rep.strict_passed = all(
        rep.hf_tokens[i] == rep.mlx_tokens[i] for i in range(strict_horizon)
    )
    rep.soft_passed = len(rep.mismatch_indices) == 0

    summary = {
        "skipped": False,
        **asdict(rep),
        "summary": (
            f"{len(rep.agreement_indices)}/{horizon} steps agree; "
            f"strict[0..{strict_horizon - 1}] "
            f"{'PASS' if rep.strict_passed else 'FAIL'}"
        ),
    }
    _emit(summary, args.out)

    if not rep.strict_passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
