"""Per-stage logits parity between PyTorch HF reference and CoreML mlpackages.

Phase A v2 — subprocess-isolated CoreML predicts.

We compare three stages of the converted pipeline against the HF PyTorch
reference (FP32) on identical image-first inputs:

    Stage A  vision_embed       — projector output (image_embeds) BEFORE
                                  scatter into prompt embeddings.
    Stage B  prefill_decoder    — last-token logits after one prefill step.
    Stage C  decode_decoder     — per-step logits for K greedy decode steps.

Hybrid-OCR transition (issue #1, commit #5):
    vision_embed.mlpackage now emits ``image_embeds`` of shape
    ``[num_image_tokens, hidden]``. Stages B and C still expect a fused
    ``inputs_embeds`` so this harness contains a transitional adapter
    (``_scatter_image_embeds_into_prompt``) that performs the scatter in
    PyTorch using HF ``embed_tokens``. The adapter is removed in commit
    #11 when Stages B and C are deleted from the pipeline.

CoreML predicts run in spawn-based subprocesses (see ``_coreml_subprocess``),
which sidesteps the coremltools↔torch memory-corruption SIGSEGVs we hit in v1.

Inputs match what coreml_runner.dart will actually feed:
  * Image-first prompt: image tokens occupy positions [0, M), text tail
    follows, right-padded to ``--prompt-len`` (default 256).
  * mRoPE 3-axis position_ids (T,H,W) computed from the image grid.

Outputs land in ``--report`` JSON. Tolerance defaults: vision MAE < 5e-3,
prefill last-token MAE < 5e-2 with top1 match, decode per-step matches HF
top1 in every step (mean MAE recorded; not gated for image-invariance debug).

Run:
  .venv/bin/python -m benchmark.runtime.converters.paddleocr_vl_coreml.parity \\
    --hf-snapshot /tmp/.../snapshots/<sha> \\
    --coreml-dir  /tmp/paddleocr-vl-coreml-rebuild/converted \\
    --images img1.jpg img2.jpg \\
    --report /tmp/paddleocr-vl-coreml-rebuild/parity_report.json \\
    --decode-steps 16
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore")

from ._coreml_subprocess import predict_isolated, predict_isolated_chain  # noqa: E402

# ---- pipeline.json constants --------------------------------------------- #
PATCH_SIZE = 14
SPATIAL_MERGE = 2
DEFAULT_PROMPT_LEN = 256          # converters/enumerated_shapes PROMPT_BUCKETS[1]
DEFAULT_DECODE_STEPS = 8
IMAGE_TOKEN_ID = 100295
VISION_START_ID = 101305
VISION_END_ID = 101306
HEAD_DIM = 128
HIDDEN_SIZE = 1024
NUM_LAYERS = 18
MROPE_SECTION = [16, 24, 24]
ROPE_THETA = 500_000.0
DEFAULT_VISION_TOL = 5e-3
DEFAULT_LOGITS_TOL = 5e-2


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class StageAReport:
    mae: float
    max_abs: float
    mse: float
    cosine: float
    shape: list[int]
    any_nan_pt: bool
    any_nan_cm: bool
    passed: bool
    tolerance: float


@dataclass
class StageBReport:
    pt_top1: int
    cm_top1: int
    top1_match: bool
    logits_mae: float
    logits_max_abs: float
    logits_cosine: float
    top32_mae: float
    any_nan_pt: bool
    any_nan_cm: bool
    passed: bool
    tolerance: float


@dataclass
class DecodeStep:
    step: int
    pt_top1: int
    cm_top1: int
    top1_match: bool
    logits_mae: float
    logits_max_abs: float
    logits_cosine: float


@dataclass
class StageCReport:
    steps: list[DecodeStep] = field(default_factory=list)
    top1_match_ratio: float = 0.0
    summary_mae: float = 0.0
    max_step_mae: float = 0.0
    passed: bool = False


@dataclass
class ImageReport:
    path: str
    grid_thw: list[int]
    bucket_used: int | None
    prompt_len_used: int
    n_image_tokens: int
    notes: str = ""
    error: str | None = None
    stage_a_vision_embed: StageAReport | None = None
    stage_b_prefill: StageBReport | None = None
    stage_c_decode: StageCReport | None = None
    overall_passed: bool = False


# --------------------------------------------------------------------------- #
# mRoPE helpers — kept verbatim from v1 so e2e_token_golden imports still work
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def build_inv_freq(head_dim: int, base: float = ROPE_THETA) -> torch.Tensor:
    return 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )


def rotary_cos_sin_3d(
    inv_freq: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D (T,H,W) cos/sin tables.  position_ids: (3, B, P) → cos/sin (3,B,P,d)."""
    inv = inv_freq.to(torch.float32)[None, None, :, None]
    inv = inv.expand(3, position_ids.shape[1], -1, 1)
    pos = position_ids[:, :, None, :].to(torch.float32)
    freqs = (inv @ pos).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def select_mrope(
    cos: torch.Tensor, sin: torch.Tensor, mrope_section: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    sec = list(mrope_section) * 2
    cos_chunks = list(cos.split(sec, dim=-1))
    sin_chunks = list(sin.split(sec, dim=-1))
    cos_sel = torch.cat([m[i % 3] for i, m in enumerate(cos_chunks)], dim=-1)
    sin_sel = torch.cat([m[i % 3] for i, m in enumerate(sin_chunks)], dim=-1)
    return cos_sel, sin_sel


def build_image_first_input_ids(
    text_input_ids: torch.Tensor,
    n_image_tokens: int,
    prompt_len: int,
) -> tuple[torch.Tensor, int]:
    ids = text_input_ids[0].tolist()
    text_tail = [t for t in ids if t != IMAGE_TOKEN_ID]
    real_len = n_image_tokens + len(text_tail)
    if real_len > prompt_len:
        raise ValueError(
            f"prompt overflow: {real_len} tokens but bucket={prompt_len}"
        )
    out = (
        [IMAGE_TOKEN_ID] * n_image_tokens
        + text_tail
        + [0] * (prompt_len - real_len)
    )
    return torch.tensor(out, dtype=torch.int64).unsqueeze(0), real_len


def fit_bucket(
    grid_thw: tuple[int, int, int],
    available_buckets: list[tuple[int, int, int]],
) -> int | None:
    t, h, w = grid_thw
    candidates = []
    for i, (bt, bh, bw) in enumerate(available_buckets):
        if bt >= t and bh >= h and bw >= w:
            candidates.append((bt * bh * bw, i))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf)) or 1.0
    return float(np.dot(af, bf) / denom)


def _diff_stats(pt: np.ndarray, cm: np.ndarray) -> tuple[float, float, float, float]:
    pt32 = pt.astype(np.float32)
    cm32 = cm.astype(np.float32)
    diff = np.abs(pt32 - cm32)
    return (
        float(diff.mean()),
        float(diff.max()),
        float((diff ** 2).mean()),
        _cosine(pt32, cm32),
    )


def _build_position_ids_image_first(
    grid_thw: list[int],
    real_len: int,
    prompt_len: int,
) -> torch.Tensor:
    """3D mRoPE position_ids for the image-first layout.  Shape (3, 1, prompt_len)."""
    t, h, w = grid_thw
    gt, gh, gw = t, h // SPATIAL_MERGE, w // SPATIAL_MERGE
    n_img = gt * gh * gw
    t_index = torch.arange(gt).view(-1, 1).expand(-1, gh * gw).flatten()
    h_index = torch.arange(gh).view(1, -1, 1).expand(gt, -1, gw).flatten()
    w_index = torch.arange(gw).view(1, 1, -1).expand(gt, gh, -1).flatten()
    img_pos = torch.stack([t_index, h_index, w_index])  # (3, n_img)

    max_img = int(img_pos.max().item())
    n_text = real_len - n_img
    text_pos_1d = torch.arange(n_text, dtype=torch.long) + (max_img + 1)
    text_pos = text_pos_1d.unsqueeze(0).expand(3, -1)

    n_pad = prompt_len - real_len
    if n_pad > 0:
        pad_val = max_img + 1 + n_text
        pad_pos = torch.full((3, n_pad), pad_val, dtype=torch.long)
        position_ids = torch.cat([img_pos, text_pos, pad_pos], dim=1)
    else:
        position_ids = torch.cat([img_pos, text_pos], dim=1)
    return position_ids.unsqueeze(1)  # (3, 1, P)


def _compute_rope_deltas(position_ids: torch.Tensor, real_len: int) -> int:
    real_pos = position_ids[..., :real_len]
    return int(real_pos.max().item()) + 1 - real_len


# --------------------------------------------------------------------------- #
# Phase E v9 — HF-native input construction (Bug L fix)
#
# The legacy `build_image_first_input_ids` + `_build_position_ids_image_first`
# pair reshuffles input_ids so image tokens occupy [0, M) and computes a
# custom 3D mRoPE table. Validation (diag_path_a_vs_b.py) shows this layout
# diverges from HF's actual mrope positions (cosine ~0.93, top-1 wrong).
#
# This block constructs prefill inputs using HF's exact path:
#   * input_ids straight from `processor(images=, text=)` (image tokens stay
#     at the chat-template positions HF expects).
#   * position_ids via `model.get_rope_index(...)`  → (3, B, S) int64.
#   * cos/sin via the model's own `rotary_emb(x, position_ids)` → (3,B,S,D),
#     then collapsed via the `mrope_section` chunking convention to (B,S,D).
#   * Image features come from our `vision_embed.mlpackage` (with the
#     image-first layout it was traced for) but only the first M rows of
#     the projected feature stream are kept; those are then scattered into
#     the HF-native text-embed buffer at the natural image-token positions.
# --------------------------------------------------------------------------- #
def collapse_mrope_3axis(
    cos_3: torch.Tensor,
    sin_3: torch.Tensor,
    mrope_section: list[int] = MROPE_SECTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse rotary cos/sin from (3, B, S, D) to (B, S, D) by selecting
    axis i % 3 from each chunk of size `sec[i]`. `sec` is `mrope_section * 2`
    so that the doubled rotary layout (real|imag halves) lines up.

    This matches the HF model's per-layer call:
        cos = cos[mrope_section[0]] + cos[mrope_section[1]] + ...
    """
    sec = list(mrope_section) * 2  # e.g. [16,24,24,16,24,24]
    cos_chunks = list(cos_3.split(sec, dim=-1))
    sin_chunks = list(sin_3.split(sec, dim=-1))
    cos_sel = torch.cat([c[i % 3] for i, c in enumerate(cos_chunks)], dim=-1)
    sin_sel = torch.cat([c[i % 3] for i, c in enumerate(sin_chunks)], dim=-1)
    return cos_sel, sin_sel


def build_hf_native_prefill_inputs(
    *,
    image,                             # PIL.Image (already bucket-resized)
    text: str,                         # chat-templated prompt
    processor,                         # HF AutoProcessor
    model,                             # HF SHIPPED model (for get_rope_index + rotary_emb)
    embed_tokens,                      # nn.Embedding (model.model.language_model.embed_tokens)
    image_features_projected: torch.Tensor,  # (M, hidden) projected image embeds, fp16/fp32
    prompt_len: int,
    head_dim: int = HEAD_DIM,
    mrope_section: list[int] = MROPE_SECTION,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> dict[str, Any]:
    """Build prefill inputs the way HF's own forward does.

    Returns a dict ready to feed `prefill_decoder.mlpackage`:
        inputs_embeds   (1, P, hidden) fp16  — text embeds w/ image features scattered
        attention_mask  (1, P) int32         — 1 over real_len, 0 over pad
        rope_cos        (1, 1, P, head_dim) fp16
        rope_sin        (1, 1, P, head_dim) fp16
        prompt_len_used (1,) int32
        position_ids    (3, 1, P) int64       — for decode-step extension
        rope_deltas     int                   — max_pos+1 - real_len, for decode
        real_len        int
        input_ids       (1, real_len) int64   — natural HF layout (no padding)
    """
    proc = processor(images=[image], text=[text], return_tensors="pt")
    input_ids = proc["input_ids"]                      # (1, S) int64
    image_grid_thw = proc["image_grid_thw"]            # (1, 3)
    attention_mask_real = proc.get("attention_mask")
    if attention_mask_real is None:
        attention_mask_real = torch.ones_like(input_ids)
    real_len = int(input_ids.shape[1])
    if real_len > prompt_len:
        raise ValueError(
            f"prompt overflow: HF produced {real_len} tokens; bucket={prompt_len}"
        )

    # ---- 1) text embeds (B, S, H) and scatter projected image features --- #
    text_embeds = embed_tokens(input_ids)              # fp32 typically
    img_mask_real = (input_ids == image_token_id)      # (1, S) bool
    n_img = int(img_mask_real.sum().item())
    if n_img != image_features_projected.shape[0]:
        raise ValueError(
            f"image feature count mismatch: HF expects {n_img} tokens but "
            f"got {image_features_projected.shape[0]} projected features"
        )
    fused_embeds = text_embeds.clone()
    fused_embeds[img_mask_real] = image_features_projected.to(text_embeds.dtype)

    # ---- 2) HF's get_rope_index → (3, 1, S) ----------------------------- #
    # The LIB-version PaddleOCRVLModel.get_rope_index requires
    # `mm_token_type_ids` (text=0, image=1, video=2). We synthesize it the
    # way the processor does (zeros, then 1 at every image_token slot).
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    mm_token_type_ids[img_mask_real] = 1
    inner = model.model  # PaddleOCRVLModel
    position_ids_real, _rope_deltas_t = inner.get_rope_index(
        input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask_real,
    )
    rope_deltas = int(position_ids_real[..., :real_len].max().item()) + 1 - real_len

    # ---- 3) HF's rotary_emb → cos/sin (3, 1, S, head_dim) -------------- #
    rotary_emb = inner.language_model.rotary_emb
    cos3_real, sin3_real = rotary_emb(fused_embeds, position_ids_real)

    # ---- 4) Collapse mrope axes → (1, S, head_dim) ---------------------- #
    cos_sel_real, sin_sel_real = collapse_mrope_3axis(
        cos3_real, sin3_real, mrope_section
    )

    # ---- 5) Pad everything from S → P ---------------------------------- #
    P = prompt_len
    H = fused_embeds.shape[-1]

    embeds_pad = torch.zeros(1, P, H, dtype=torch.float16)
    embeds_pad[:, :real_len, :] = fused_embeds.to(torch.float16)

    attn_pad = torch.zeros(1, P, dtype=torch.int32)
    attn_pad[0, :real_len] = 1

    cos_pad = torch.zeros(1, P, head_dim, dtype=torch.float16)
    sin_pad = torch.zeros(1, P, head_dim, dtype=torch.float16)
    cos_pad[:, :real_len, :] = cos_sel_real.to(torch.float16)
    sin_pad[:, :real_len, :] = sin_sel_real.to(torch.float16)
    rope_cos = cos_pad.unsqueeze(1)  # (1, 1, P, D)
    rope_sin = sin_pad.unsqueeze(1)

    # Pad position_ids to P with the next sequential id along the same axis
    # (only used to derive per-step decode positions; pad value is harmless
    # because attention_mask covers real_len only).
    pad_pos_val = int(position_ids_real[..., :real_len].max().item()) + 1
    pos_pad = torch.full((3, 1, P), pad_pos_val, dtype=torch.int64)
    pos_pad[:, :, :real_len] = position_ids_real

    return dict(
        inputs_embeds=embeds_pad.numpy(),
        attention_mask=attn_pad.numpy(),
        rope_cos=rope_cos.numpy(),
        rope_sin=rope_sin.numpy(),
        prompt_len_used=np.array([real_len], dtype=np.int32),
        position_ids=pos_pad,
        rope_deltas=rope_deltas,
        real_len=real_len,
        input_ids=input_ids,
    )


def hf_native_step_rope(
    *,
    cache_pos: int,
    rope_deltas: int,
    inv_freq: torch.Tensor,
    head_dim: int = HEAD_DIM,
    mrope_section: list[int] = MROPE_SECTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (rope_cos, rope_sin) at shape (1, 1, 1, head_dim) for a single
    decode step at sequence index `cache_pos`. Replicates the per-axis
    formula HF uses post-prefill: each of the 3 rope axes carries the same
    1D position `cache_pos + rope_deltas`, then mrope-section collapse picks
    axis i%3 chunk-by-chunk — yielding identical cos/sin since all three axes
    share the same id."""
    pos_int = cache_pos + rope_deltas
    pos_p = (
        torch.tensor([[[pos_int]]], dtype=torch.int64).expand(3, 1, 1).contiguous()
    )
    cos3, sin3 = rotary_cos_sin_3d(inv_freq, pos_p)
    cos_sel, sin_sel = collapse_mrope_3axis(cos3, sin3, mrope_section)
    rope_cos = cos_sel.unsqueeze(1).to(torch.float16).numpy()
    rope_sin = sin_sel.unsqueeze(1).to(torch.float16).numpy()
    return rope_cos, rope_sin


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class ParityHarness:
    def __init__(
        self,
        hf_snapshot: Path,
        coreml_dir: Path,
        decode_steps: int,
        prompt_len: int,
        vision_tol: float,
        logits_tol: float,
        progress_path: Path | None = None,
    ) -> None:
        self.hf_snapshot = hf_snapshot
        self.coreml_dir = coreml_dir
        self.decode_steps = int(decode_steps)
        self.prompt_len = int(prompt_len)
        self.vision_tol = float(vision_tol)
        self.logits_tol = float(logits_tol)

        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_grad_enabled(False)

        self._progress_fp = None
        if progress_path is not None:
            try:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                self._progress_fp = open(progress_path, "a", buffering=1)
                self._log(f"=== ParityHarness init pid={os.getpid()} ===")
            except Exception:
                self._progress_fp = None

        with open(coreml_dir / "pipeline.json") as f:
            self.pipeline = json.load(f)
        self.cfg = self.pipeline["config"]
        self.image_buckets = [tuple(g) for g in self.pipeline["buckets"]["image_grids"]]
        self.merged_token_counts = list(
            self.pipeline["buckets"]["merged_token_counts"]
        )
        # Phase 1: only bucket 0 is traced.
        self.active_bucket_idx = 0
        self.active_bucket = self.image_buckets[0]
        self.active_merged = self.merged_token_counts[0]

        self.head_dim = int(self.cfg["head_dim"])
        self.hidden_size = int(self.cfg["hidden_size"])
        self.num_layers = int(self.cfg["num_layers"])
        self.inv_freq = build_inv_freq(self.head_dim, ROPE_THETA)

        self.vision_path = str(coreml_dir / "vision_embed.mlpackage")
        self.prefill_path = str(coreml_dir / "prefill_decoder.mlpackage")
        self.decode_path = str(coreml_dir / "decode_decoder.mlpackage")

        self.model = None
        self.processor = None

    # ------------------------------------------------------------------ logs
    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        fp = getattr(self, "_progress_fp", None)
        if fp is not None:
            try:
                fp.write(line + "\n")
                fp.flush()
            except Exception:
                pass

    # -------------------------------------------------------------- HF load
    def load_hf(self) -> None:
        from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

        self._log(f"[hf] loading model from {self.hf_snapshot}")
        self.processor = AutoProcessor.from_pretrained(str(self.hf_snapshot))
        self.processor.image_processor.min_pixels = PATCH_SIZE * PATCH_SIZE
        max_grid = max(h * w for _, h, w in self.image_buckets)
        self.processor.image_processor.max_pixels = max_grid * PATCH_SIZE * PATCH_SIZE
        self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            str(self.hf_snapshot),
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._log("[hf] model + processor loaded")

    # ------------------------------------------------------- per-image driver
    def run_image(self, image_path: Path) -> ImageReport:
        from PIL import Image

        rep = ImageReport(
            path=str(image_path),
            grid_thw=[],
            bucket_used=None,
            prompt_len_used=0,
            n_image_tokens=0,
        )
        try:
            self._log(f"\n=== {image_path.name} ===")
            img = Image.open(image_path).convert("RGB")
            self._log(f"  source size {img.size}")

            bucket = self.active_bucket
            t, gh, gw = bucket
            target_pix = (gw * PATCH_SIZE, gh * PATCH_SIZE)  # PIL: (W, H)
            img_resized = img.resize(target_pix, Image.BICUBIC)
            self._log(
                f"  resized to {img_resized.size} for bucket {bucket} "
                f"(merged={self.active_merged})"
            )
            rep.notes = (
                f"Resized to bucket {bucket}; Phase-1 mlpackage only ships "
                f"bucket index 0."
            )
            rep.bucket_used = self.active_bucket_idx

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "OCR:"},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            proc = self.processor(
                images=[img_resized], text=[text], return_tensors="pt"
            )
            grid = proc["image_grid_thw"].tolist()[0]
            rep.grid_thw = list(grid)
            n_image_tokens = int((proc["input_ids"] == IMAGE_TOKEN_ID).sum().item())
            rep.n_image_tokens = n_image_tokens
            self._log(
                f"  processor grid_thw={grid} n_image_tokens={n_image_tokens}"
            )
            assert tuple(grid) == bucket, f"grid {grid} != bucket {bucket}"
            assert n_image_tokens == self.active_merged

            input_ids_first, real_len = build_image_first_input_ids(
                proc["input_ids"], n_image_tokens, self.prompt_len
            )
            rep.prompt_len_used = real_len
            self._log(f"  input_ids: real_len={real_len} padded={self.prompt_len}")

            attention_mask = torch.zeros(1, self.prompt_len, dtype=torch.int64)
            attention_mask[0, :real_len] = 1

            pv = proc["pixel_values"]  # (N, 3, 14, 14) fp32
            n_patches = pv.shape[0]
            assert n_patches == bucket[0] * bucket[1] * bucket[2]
            pv_flat_fp16 = pv.reshape(
                1, n_patches, 3 * PATCH_SIZE * PATCH_SIZE
            ).to(torch.float16)

            position_ids_full = _build_position_ids_image_first(
                list(grid), real_len, self.prompt_len
            )

            # ---- HF reference: ONE forward pass via the public language_model
            # API. We feed the SAME image-first inputs that the CoreML
            # mlpackage will see. Hidden states & logits come from a single
            # call; KV-cache survives for Stage C.
            hf_ref = self._compute_hf_reference(
                input_ids_first=input_ids_first,
                pixel_values_fp32=pv,
                grid_thw=grid,
                attention_mask=attention_mask,
                position_ids_full=position_ids_full,
                real_len=real_len,
            )

            # ============================================================ A
            #
            # Stage A now compares CoreML image_embeds (projector output,
            # rank-2 [M, hidden]) against the HF projector output before
            # scatter. Stages B and C still consume the fused inputs_embeds
            # — they are fed from the HF reference path
            # (``hf_ref["fused_embeds"]``), which is the transitional
            # adapter for the hybrid-OCR refactor: HF embed_tokens scatters
            # the projected image features into the prompt-embedding
            # buffer in PyTorch so Stages B/C see a valid input. This
            # adapter goes away in commit #11 when Stages B and C are
            # deleted from the pipeline (the Dart-side scatter becomes the
            # only path).
            rep.stage_a_vision_embed = self._stage_a(
                pv_flat_fp16=pv_flat_fp16,
                grid_thw=list(grid),
                hf_image_embeds_projected=hf_ref["image_embeds_projected"],
            )

            # ============================================================ B
            rep.stage_b_prefill = self._stage_b(
                hf_fused_embeds=hf_ref["fused_embeds"],
                attention_mask=attention_mask,
                position_ids_full=position_ids_full,
                real_len=real_len,
                hf_last_logits=hf_ref["last_logits"],
            )

            # ============================================================ C
            rep.stage_c_decode = self._stage_c(
                hf_fused_embeds=hf_ref["fused_embeds"],
                attention_mask=attention_mask,
                position_ids_full=position_ids_full,
                real_len=real_len,
                hf_per_step_logits=hf_ref["per_step_logits"],
                hf_decode_seed_token=hf_ref["seed_token"],
                hf_decode_tokens=hf_ref["decode_tokens"],
            )

            rep.overall_passed = (
                bool(rep.stage_a_vision_embed and rep.stage_a_vision_embed.passed)
                and bool(rep.stage_b_prefill and rep.stage_b_prefill.passed)
                and bool(rep.stage_c_decode and rep.stage_c_decode.passed)
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            rep.error = f"{type(e).__name__}: {e}\n{tb}"
            self._log(f"  [ERROR] {e}")
            print(tb, flush=True)
        return rep

    # ------------------------------------------------------ HF reference pass
    def _compute_hf_reference(
        self,
        *,
        input_ids_first: torch.Tensor,
        pixel_values_fp32: torch.Tensor,
        grid_thw: list[int],
        attention_mask: torch.Tensor,
        position_ids_full: torch.Tensor,
        real_len: int,
    ) -> dict[str, Any]:
        """Single HF forward producing fused_embeds, prefill last-token logits,
        and K decode-step logits (greedy continuation)."""
        from transformers.cache_utils import DynamicCache

        m = self.model
        inner = m.model

        # 1) fused embeds via the same path as upstream forward
        self._log("[hf] computing fused embeds")
        t0 = time.time()
        ids = input_ids_first
        text_embeds = inner.language_model.embed_tokens(ids)  # (1, P, hidden)
        image_outputs = inner.get_image_features(
            pixel_values=pixel_values_fp32.detach().clone(),
            image_grid_thw=torch.tensor([list(grid_thw)], dtype=torch.long),
            return_dict=True,
        )
        image_embeds = image_outputs.pooler_output.to(text_embeds.dtype)
        mask = (ids == IMAGE_TOKEN_ID)
        assert int(mask.sum().item()) == image_embeds.shape[0]
        mask_exp = mask.unsqueeze(-1).expand_as(text_embeds)
        fused_embeds = text_embeds.masked_scatter(mask_exp, image_embeds)
        self._log(
            f"  fused embeds shape={tuple(fused_embeds.shape)} "
            f"any_nan={torch.isnan(fused_embeds).any().item()} "
            f"abs_max={fused_embeds.abs().max().item():.3e} "
            f"in {time.time()-t0:.2f}s"
        )

        # 2) prefill via language_model on the trimmed (real_len) inputs
        self._log("[hf] prefill forward")
        embeds_real = fused_embeds[:, :real_len, :].contiguous().to(torch.float32)
        attn_real = torch.ones(1, real_len, dtype=torch.int64)
        position_ids_real = position_ids_full[:, :, :real_len].contiguous()
        cache = DynamicCache()
        t0 = time.time()
        out = inner.language_model(
            input_ids=None,
            inputs_embeds=embeds_real,
            attention_mask=attn_real,
            position_ids=position_ids_real,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        last_logits = m.lm_head(out.last_hidden_state)[0, -1].to(torch.float32).numpy()
        seed_token = int(np.argmax(last_logits))
        past_kv = out.past_key_values
        self._log(
            f"  prefill {time.time()-t0:.2f}s seed_token={seed_token} "
            f"any_nan={bool(np.isnan(last_logits).any())}"
        )

        # 3) K greedy decode steps via language_model
        K = self.decode_steps
        rope_deltas_int = _compute_rope_deltas(position_ids_full, real_len)
        per_step_logits: list[np.ndarray] = []
        decode_tokens: list[int] = [seed_token]
        cur = seed_token
        for step in range(K):
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
                m.lm_head(out.last_hidden_state)[0, 0].to(torch.float32).numpy()
            )
            per_step_logits.append(step_logits)
            cur = int(np.argmax(step_logits))
            decode_tokens.append(cur)
        self._log(f"[hf] decode tokens (seed + {K}): {decode_tokens}")

        return {
            "fused_embeds": fused_embeds,
            "image_embeds_projected": image_embeds,
            "image_token_mask": mask,
            "last_logits": last_logits,
            "seed_token": seed_token,
            "per_step_logits": per_step_logits,
            "decode_tokens": decode_tokens,
        }

    # ----------------------------------------------------------------- Stage A
    def _stage_a(
        self,
        *,
        pv_flat_fp16: torch.Tensor,
        grid_thw: list[int],
        hf_image_embeds_projected: torch.Tensor,
    ) -> StageAReport:
        """Compare CoreML ``image_embeds`` against the HF projector output.

        Hybrid-OCR contract (issue #1): vision_embed.mlpackage emits the
        projector output directly, before any scatter into prompt embeds.
        We compare against ``image_outputs.pooler_output`` from
        ``inner.get_image_features(...)`` — same path HF takes internally
        before the masked_scatter into text embeds.
        """
        self._log("[A] vision_embed (subprocess) → image_embeds")
        cm_in = {
            "pixel_values": pv_flat_fp16.numpy().astype(np.float16),
            "image_grid_thw": np.asarray(grid_thw, dtype=np.int32),
        }
        t0 = time.time()
        cm_out = predict_isolated(self.vision_path, cm_in, stateful=False)
        self._log(f"  predict {time.time()-t0:.2f}s")
        cm_embeds = np.asarray(cm_out["image_embeds"]).astype(np.float32)
        # Accept either rank-2 [M, H] or rank-3 [1, M, H] from the package.
        if cm_embeds.ndim == 3 and cm_embeds.shape[0] == 1:
            cm_embeds = cm_embeds[0]
        pt = hf_image_embeds_projected.detach().to(torch.float32).numpy()
        if pt.ndim == 3 and pt.shape[0] == 1:
            pt = pt[0]
        if cm_embeds.shape != pt.shape:
            raise ValueError(
                f"vision shape mismatch: pt={pt.shape} vs cm={cm_embeds.shape}"
            )
        mae, mx, mse, cos = _diff_stats(pt, cm_embeds)
        any_nan_pt = bool(np.isnan(pt).any())
        any_nan_cm = bool(np.isnan(cm_embeds).any())
        passed = (mae < self.vision_tol) and not (any_nan_pt or any_nan_cm)
        self._log(
            f"  Stage A: shape={list(cm_embeds.shape)} "
            f"mae={mae:.4e} max={mx:.4e} mse={mse:.4e} cos={cos:.5f} "
            f"nan_pt={any_nan_pt} nan_cm={any_nan_cm} → "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            self._dump_worst_positions(pt, cm_embeds, label="stage_a")
        return StageAReport(
            mae=mae, max_abs=mx, mse=mse, cosine=cos,
            shape=list(cm_embeds.shape),
            any_nan_pt=any_nan_pt, any_nan_cm=any_nan_cm,
            passed=passed, tolerance=self.vision_tol,
        )

    # ----------------------------------------------------------------- Stage B
    def _stage_b(
        self,
        *,
        hf_fused_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_full: torch.Tensor,
        real_len: int,
        hf_last_logits: np.ndarray,
    ) -> StageBReport:
        self._log("[B] prefill_decoder (subprocess)")
        cos3, sin3 = rotary_cos_sin_3d(self.inv_freq, position_ids_full)
        cos_sel, sin_sel = select_mrope(cos3, sin3, MROPE_SECTION)
        rope_cos = cos_sel.unsqueeze(1).to(torch.float16).numpy()
        rope_sin = sin_sel.unsqueeze(1).to(torch.float16).numpy()

        cm_in = {
            "inputs_embeds": hf_fused_embeds.to(torch.float16).numpy(),
            "attention_mask": attention_mask.to(torch.int32).numpy(),
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "prompt_len_used": np.array([real_len], dtype=np.int32),
        }
        t0 = time.time()
        cm_out = predict_isolated(self.prefill_path, cm_in, stateful=True)
        self._log(f"  predict {time.time()-t0:.2f}s")
        cm_logits = cm_out["logits"].astype(np.float32).reshape(-1)

        any_nan_pt = bool(np.isnan(hf_last_logits).any())
        any_nan_cm = bool(np.isnan(cm_logits).any())
        if any_nan_cm:
            mae = float("nan"); mx = float("nan"); cos = float("nan"); top32 = float("nan")
            cm_top1 = int(np.nanargmax(cm_logits) if not np.isnan(cm_logits).all() else 0)
        else:
            mae, mx, _mse, cos = _diff_stats(hf_last_logits, cm_logits)
            top32_idx = np.argsort(-hf_last_logits)[:32]
            top32 = float(np.abs(hf_last_logits - cm_logits)[top32_idx].mean())
            cm_top1 = int(np.argmax(cm_logits))
        pt_top1 = int(np.argmax(hf_last_logits)) if not any_nan_pt else 0
        passed = (
            (pt_top1 == cm_top1)
            and not (any_nan_pt or any_nan_cm)
            and (top32 < self.logits_tol if not np.isnan(top32) else False)
        )
        self._log(
            f"  Stage B: pt_top1={pt_top1} cm_top1={cm_top1} "
            f"mae={mae:.4e} max={mx:.4e} top32_mae={top32:.4e} cos={cos:.5f} "
            f"nan_pt={any_nan_pt} nan_cm={any_nan_cm} → "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed and not any_nan_cm:
            self._dump_worst_logits(hf_last_logits, cm_logits, label="stage_b")
        return StageBReport(
            pt_top1=pt_top1, cm_top1=cm_top1,
            top1_match=(pt_top1 == cm_top1),
            logits_mae=mae, logits_max_abs=mx,
            logits_cosine=cos, top32_mae=top32,
            any_nan_pt=any_nan_pt, any_nan_cm=any_nan_cm,
            passed=passed, tolerance=self.logits_tol,
        )

    # ----------------------------------------------------------------- Stage C
    def _stage_c(
        self,
        *,
        hf_fused_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_full: torch.Tensor,
        real_len: int,
        hf_per_step_logits: list[np.ndarray],
        hf_decode_seed_token: int,
        hf_decode_tokens: list[int],
    ) -> StageCReport:
        """Per-step decode parity.

        decode_decoder owns its OWN state separate from prefill's state. To
        warm decode's state we replay the entire image-first prompt through
        decode itself (P calls), then run K decode steps. All P + K calls
        live in a single subprocess so we don't pickle CoreML state objects.
        """
        K = self.decode_steps
        self._log(f"[C] decode_decoder ({K} steps, warmup={real_len})")

        rope_deltas_int = _compute_rope_deltas(position_ids_full, real_len)

        # Build the full plan as one chain.
        plan: list[tuple[str, str, dict[str, np.ndarray] | None, bool]] = []
        plan.append(("load_stateful", self.decode_path, None, False))

        # Warmup: P uncaptured predicts replaying the prompt
        embeds_fp16 = hf_fused_embeds.to(torch.float16).numpy()
        for p in range(real_len):
            embed_p = embeds_fp16[:, p:p + 1, :]
            pos_p = position_ids_full[:, :, p:p + 1].contiguous()
            cos3, sin3 = rotary_cos_sin_3d(self.inv_freq, pos_p)
            cos_sel, sin_sel = select_mrope(cos3, sin3, MROPE_SECTION)
            rope_cos = cos_sel.unsqueeze(1).to(torch.float16).numpy()
            rope_sin = sin_sel.unsqueeze(1).to(torch.float16).numpy()
            plan.append((
                "predict", self.decode_path, {
                    "inputs_embeds": embed_p,
                    "rope_cos": rope_cos,
                    "rope_sin": rope_sin,
                    "cur_len": np.array([p], dtype=np.int32),
                    "kv_len": np.array([p + 1], dtype=np.int32),
                }, False,
            ))

        # K decode steps: feed HF's tokens to keep parity meaningful even if
        # CoreML's first decode disagrees (we want per-step input invariance).
        embed_tokens = self.model.model.language_model.embed_tokens
        for step in range(K):
            tok = hf_decode_tokens[step]  # seed at step 0, then HF's argmaxes
            tok_embed = embed_tokens(
                torch.tensor([[tok]], dtype=torch.int64)
            ).to(torch.float16).numpy()
            cache_pos = real_len + step
            pos_int = cache_pos + rope_deltas_int
            pos_p = (
                torch.tensor([[[pos_int]]], dtype=torch.int64)
                .expand(3, 1, 1).contiguous()
            )
            cos3, sin3 = rotary_cos_sin_3d(self.inv_freq, pos_p)
            cos_sel, sin_sel = select_mrope(cos3, sin3, MROPE_SECTION)
            rope_cos = cos_sel.unsqueeze(1).to(torch.float16).numpy()
            rope_sin = sin_sel.unsqueeze(1).to(torch.float16).numpy()
            plan.append((
                "predict", self.decode_path, {
                    "inputs_embeds": tok_embed,
                    "rope_cos": rope_cos,
                    "rope_sin": rope_sin,
                    "cur_len": np.array([cache_pos], dtype=np.int32),
                    "kv_len": np.array([cache_pos + 1], dtype=np.int32),
                }, True,
            ))

        t0 = time.time()
        results = predict_isolated_chain(plan, timeout_s=1800)
        self._log(f"  decode chain ({real_len + K} predicts) in {time.time()-t0:.1f}s")

        # results aligns with predict entries in order. Last K are captured.
        captured = [r for r in results if r is not None]
        assert len(captured) == K, f"got {len(captured)} captured, want {K}"

        steps: list[DecodeStep] = []
        for i, cm_out in enumerate(captured):
            cm_logits = cm_out["logits"].astype(np.float32).reshape(-1)
            hf_logits = hf_per_step_logits[i]
            if np.isnan(cm_logits).any():
                mae = float("nan"); mx = float("nan"); cos = float("nan")
                cm_top1 = 0
            else:
                mae, mx, _, cos = _diff_stats(hf_logits, cm_logits)
                cm_top1 = int(np.argmax(cm_logits))
            pt_top1 = int(np.argmax(hf_logits))
            steps.append(DecodeStep(
                step=i, pt_top1=pt_top1, cm_top1=cm_top1,
                top1_match=(pt_top1 == cm_top1),
                logits_mae=mae, logits_max_abs=mx, logits_cosine=cos,
            ))
            self._log(
                f"  step {i:>2}: pt_top1={pt_top1} cm_top1={cm_top1} "
                f"mae={mae:.3e} max={mx:.3e} cos={cos:.4f} "
                f"{'OK' if pt_top1 == cm_top1 else 'MISMATCH'}"
            )
        n_match = sum(1 for s in steps if s.top1_match)
        ratio = n_match / len(steps) if steps else 0.0
        valid_maes = [s.logits_mae for s in steps if not np.isnan(s.logits_mae)]
        summary_mae = float(np.mean(valid_maes)) if valid_maes else float("nan")
        max_step_mae = float(np.max(valid_maes)) if valid_maes else float("nan")
        passed = (ratio == 1.0) and (
            not np.isnan(summary_mae) and summary_mae < self.logits_tol
        )
        return StageCReport(
            steps=steps,
            top1_match_ratio=ratio,
            summary_mae=summary_mae,
            max_step_mae=max_step_mae,
            passed=passed,
        )

    # ---------------------------------------------------------- diagnostics
    def _dump_worst_positions(
        self, pt: np.ndarray, cm: np.ndarray, *, label: str
    ) -> None:
        diff = np.abs(pt - cm).reshape(-1)
        worst = np.argsort(-diff)[:8]
        self._log(f"  [{label}] worst positions:")
        for idx in worst:
            self._log(
                f"    flat_idx={int(idx)} pt={float(pt.reshape(-1)[idx]):.6f} "
                f"cm={float(cm.reshape(-1)[idx]):.6f} "
                f"diff={float(diff[idx]):.6f}"
            )

    def _dump_worst_logits(
        self, pt: np.ndarray, cm: np.ndarray, *, label: str
    ) -> None:
        pt_top = np.argsort(-pt)[:5]
        cm_top = np.argsort(-cm)[:5]
        self._log(f"  [{label}] PT top5: {[(int(i), float(pt[i])) for i in pt_top]}")
        self._log(f"  [{label}] CM top5: {[(int(i), float(cm[i])) for i in cm_top]}")


# --------------------------------------------------------------------------- #
# JSON helpers + main
# --------------------------------------------------------------------------- #
def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def _max_mae_in_report(r: ImageReport) -> float:
    candidates: list[float] = []
    if r.stage_a_vision_embed:
        candidates.append(r.stage_a_vision_embed.mae)
    if r.stage_b_prefill and not np.isnan(r.stage_b_prefill.top32_mae):
        candidates.append(r.stage_b_prefill.top32_mae)
    if r.stage_c_decode and not np.isnan(r.stage_c_decode.summary_mae):
        candidates.append(r.stage_c_decode.summary_mae)
    return max(candidates) if candidates else 0.0


def _image_invariance_summary(reports: list[ImageReport]) -> dict[str, Any]:
    """If we have ≥2 images, report whether each stage's CoreML output
    differs across images. This localizes the image-invariance bug."""
    out: dict[str, Any] = {"n_images": len(reports)}
    if len(reports) < 2:
        return out
    # Stage A: compare cosine between consecutive images' shapes only as a
    # proxy via reported per-image cosine to HF (different inputs ⇒ different
    # cosine if vision embed is image-dependent). A flat cosine across images
    # would be highly suspicious but we can't do better without storing the
    # actual tensors.
    a_cosines = [r.stage_a_vision_embed.cosine for r in reports
                 if r.stage_a_vision_embed]
    out["stage_a_cosines_to_hf"] = a_cosines
    b_top1 = [r.stage_b_prefill.cm_top1 for r in reports if r.stage_b_prefill]
    out["stage_b_cm_top1_per_image"] = b_top1
    out["stage_b_top1_all_same"] = (len(set(b_top1)) <= 1) if b_top1 else None
    if all(r.stage_c_decode for r in reports):
        c_seqs = [
            [s.cm_top1 for s in r.stage_c_decode.steps] for r in reports
        ]
        out["stage_c_cm_token_sequences"] = c_seqs
        out["stage_c_sequences_all_identical"] = (
            len({tuple(s) for s in c_seqs}) <= 1
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-snapshot", type=Path, required=True)
    ap.add_argument("--coreml-dir", type=Path, required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--vision-tolerance", type=float, default=DEFAULT_VISION_TOL)
    ap.add_argument("--logits-tolerance", type=float, default=DEFAULT_LOGITS_TOL)
    ap.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    ap.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    ap.add_argument(
        "--progress-log",
        type=Path,
        default=Path("/tmp/paddleocr-vl-coreml-rebuild/parity_progress.log"),
    )
    args = ap.parse_args()

    started = time.time()
    print(
        f"[parity] vision_tol={args.vision_tolerance:.1e} "
        f"logits_tol={args.logits_tolerance:.1e} "
        f"decode_steps={args.decode_steps} prompt_len={args.prompt_len}",
        flush=True,
    )
    h = ParityHarness(
        hf_snapshot=args.hf_snapshot,
        coreml_dir=args.coreml_dir,
        decode_steps=args.decode_steps,
        prompt_len=args.prompt_len,
        vision_tol=args.vision_tolerance,
        logits_tol=args.logits_tolerance,
        progress_path=args.progress_log,
    )
    h.load_hf()

    image_reports: list[ImageReport] = []
    for img in args.images:
        rep = h.run_image(img)
        image_reports.append(rep)
        gc.collect()

    summary = {
        "total_images": len(image_reports),
        "all_passed": all(r.overall_passed for r in image_reports),
        "max_mae_observed": max(
            (_max_mae_in_report(r) for r in image_reports), default=0.0
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compute_units": "CPU_ONLY",
        "pytorch_dtype": "float32",
        "decode_steps": args.decode_steps,
        "prompt_len": args.prompt_len,
        "vision_tolerance": args.vision_tolerance,
        "logits_tolerance": args.logits_tolerance,
        "images": [_to_jsonable(r) for r in image_reports],
        "image_invariance": _image_invariance_summary(image_reports),
        "summary": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[parity] wrote {args.report}", flush=True)
    print(f"[parity] summary: {json.dumps(summary, indent=2)}", flush=True)
    for r in image_reports:
        a = r.stage_a_vision_embed
        b = r.stage_b_prefill
        c = r.stage_c_decode
        a_str = (
            f"A={'PASS' if a.passed else 'FAIL'}(mae={a.mae:.2e})"
            if a else "A=?"
        )
        b_str = (
            f"B={'PASS' if b.passed else 'FAIL'}"
            f"(top32={b.top32_mae:.2e},top1_match={b.top1_match})"
            if b else "B=?"
        )
        c_str = (
            f"C={'PASS' if c.passed else 'FAIL'}"
            f"(ratio={c.top1_match_ratio:.2f},mae={c.summary_mae:.2e})"
            if c else "C=?"
        )
        print(f"  {Path(r.path).name}: {a_str} {b_str} {c_str}", flush=True)
        if r.error:
            print(f"    error: {r.error.splitlines()[0]}", flush=True)
    return 0 if summary["all_passed"] else 1


# Backwards-compatible shims for pipeline.py.
from .parity_compat import ParityReport, compare_logits, write_report  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
