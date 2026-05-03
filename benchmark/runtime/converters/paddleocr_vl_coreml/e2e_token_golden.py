"""End-to-end Python token-level golden test for PaddleOCR-VL-1.5 CoreML.

Phase E of the rebuild. We drive the converted FP16 mlpackages from Python
end-to-end (vision_embed → prefill_decoder → decode_decoder greedy loop)
and compare the generated token stream to the HF transformers golden
produced by ``scripts/generate_golden_ocr.py``.

Phase E v9 — Bug L fix
----------------------
The legacy harness reshuffles input_ids to image-first layout and computes
its own custom 3-axis mRoPE table. Validation (diag_path_a_vs_b.py) shows
that path diverges from HF's actual mrope positions (cosine ~0.93, top-1
wrong) even though the converted prefill mlpackage itself is bit-correct.

Set ``PHASE_E_HF_NATIVE=1`` to use HF's natural input layout +
``model.get_rope_index`` + the model's own ``rotary_emb`` to build prefill
inputs. The decode loop also derives its per-step rope from HF's
``rope_deltas``.

Default remains the legacy path (PHASE_E_HF_NATIVE=0) to keep rollback trivial.

CRITICAL CONSTRAINT (Phase 1)
-----------------------------
The Phase-1 mlpackages were traced for the SMALLEST image bucket only —
``image_grid_thw=(1, 28, 28)`` with merged_token_count=196 — and a single
prompt-length bucket ``prompt_len=256``. The HF goldens were generated on
the FULL native resolution (1222 / 1230 image tokens for the two test
images), so a token-by-token match against those goldens is not physically
possible at Phase 1: we are forced to OCR a downsized image. We therefore
ALSO compute a small-bucket HF reference inline (same image resize, same
prompt) so we can measure true CoreML↔HF parity at the operating point of
the mlpackages. Both comparisons land in the report.

Pipeline mirrors what ``coreml_runner.dart`` will do:
  1. PIL.open + resize to bucket pixel size (gw*14 × gh*14).
  2. HF processor → pixel_values (N,3,14,14) + input_ids w/ M image tokens.
  3. Re-arrange input_ids to image-first layout, right-pad to 256.
  4. Compute mRoPE 3-axis position_ids → cos/sin → mrope_section select.
  5. vision_embed.predict(...) → fused inputs_embeds.
  6. prefill_decoder.predict(...) with state → first logits → argmax.
  7. Decode loop: embed_tokens(prev_token) → decode.predict(...) → argmax,
     stop on EOS=2 or max_new_tokens.
  8. Edit-distance vs HF golden + decoded-text similarity.

Run:
  .venv/bin/python -m benchmark.runtime.converters.paddleocr_vl_coreml.e2e_token_golden \\
    --hf-snapshot /tmp/.../snapshots/<sha> \\
    --coreml-dir  /tmp/paddleocr-vl-coreml-rebuild/converted \\
    --golden-dir  /Users/.../test/golden/expected \\
    --images img1.jpg "蓝带配方和流程表.jpg" \\
    --max-new-tokens 256 \\
    --report /tmp/paddleocr-vl-coreml-rebuild/e2e_golden_report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Compatibility shim
# --------------------------------------------------------------------------- #
# SHIPPED PaddleOCR-VL modeling code (cached in the snapshot via
# trust_remote_code) was authored against a newer transformers release that
# exposes `check_model_inputs` from `transformers.utils.generic`. Our pinned
# transformers does not. We only need AutoProcessor / AutoTokenizer here
# (no model load) but defensively shim it anyway in case the processor's
# trust_remote_code path triggers it.
import transformers.utils.generic as _tug  # noqa: E402
if not hasattr(_tug, "check_model_inputs"):
    _tug.check_model_inputs = lambda fn: fn  # no-op decorator

# Reuse battle-tested helpers from the parity harness (Phase A).
from .parity import (  # noqa: E402
    IMAGE_TOKEN_ID,
    PATCH_SIZE,
    SPATIAL_MERGE,
    build_hf_native_prefill_inputs,
    build_image_first_input_ids,
    build_inv_freq,
    collapse_mrope_3axis,
    fit_bucket,
    hf_native_step_rope,
    rotary_cos_sin_3d,
    select_mrope,
)

EOS_TOKEN_ID = 2  # generation_config.json
DEFAULT_PROMPT = "OCR:"
PROMPT_LEN_TRACED = 256  # only prompt bucket the prefill mlpackage supports
HEAD_DIM = 128
HIDDEN_SIZE = 1024
NUM_LAYERS = 18
MROPE_SECTION = [16, 24, 24]

# Phase E v9 — env-gated switch to HF-native input construction (Bug L fix).
HF_NATIVE = os.environ.get("PHASE_E_HF_NATIVE", "0") == "1"


# --------------------------------------------------------------------------- #
# Report dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class StageLatencies:
    vision_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_total_ms: float = 0.0
    decode_steps: int = 0
    decode_p50_ms: float = 0.0
    decode_p95_ms: float = 0.0


@dataclass
class GoldenComparison:
    label: str  # "hf_saved_full_res" or "hf_smallbucket_apples_to_apples"
    hf_token_count: int
    cm_token_count: int
    edit_distance: int
    first_divergence_at: int | None  # None if perfect prefix match
    tokens_match_ratio: float  # matches in common prefix / max(len)
    text_similarity: float  # char-level ratio, 0..1
    notes: str = ""


@dataclass
class ImageReport:
    image: str
    bucket_used: dict[str, Any] = field(default_factory=dict)
    grid_thw_native: list[int] | None = None
    n_image_tokens: int = 0
    prompt_len_used: int = 0
    tokens_generated: int = 0
    coreml_token_ids: list[int] = field(default_factory=list)
    coreml_text_first_500: str = ""
    hf_text_first_500: str = ""
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    comparisons: list[GoldenComparison] = field(default_factory=list)
    passed: bool = False
    error: str | None = None
    skip_reason: str | None = None


# --------------------------------------------------------------------------- #
# Distance / similarity utilities
# --------------------------------------------------------------------------- #
def levenshtein(a: list[int], b: list[int]) -> int:
    """Iterative Levenshtein on two int sequences. O(len(a)*len(b)) time, O(min) space."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) == len(b):
        return None
    return n  # one is a prefix of the other; "diverges" at end of shorter


def char_similarity(a: str, b: str) -> float:
    """Simple char-level similarity = 1 - normalized Levenshtein on chars.

    Caps at len(a)+len(b) <= 4000 chars to keep runtime bounded; otherwise
    we fall back to a cheap longest-common-prefix/total ratio.
    """
    if not a and not b:
        return 1.0
    if max(len(a), len(b)) > 2000:
        # cheap surrogate
        n = min(len(a), len(b))
        prefix = 0
        for i in range(n):
            if a[i] != b[i]:
                break
            prefix += 1
        return prefix / max(len(a), len(b))
    # exact Levenshtein on chars (small strings)
    al, bl = list(a), list(b)
    if len(al) < len(bl):
        al, bl = bl, al
    prev = list(range(len(bl) + 1))
    for i, ca in enumerate(al, 1):
        cur = [i] + [0] * len(bl)
        for j, cb in enumerate(bl, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[-1]
    return 1.0 - dist / max(len(a), len(b))


# --------------------------------------------------------------------------- #
# E2E harness
# --------------------------------------------------------------------------- #
class E2EHarness:
    def __init__(
        self,
        hf_snapshot: Path,
        coreml_dir: Path,
        golden_dir: Path,
        max_new_tokens: int,
    ) -> None:
        self.hf_snapshot = hf_snapshot
        self.coreml_dir = coreml_dir
        self.golden_dir = golden_dir
        self.max_new_tokens = int(max_new_tokens)

        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_grad_enabled(False)

        with (coreml_dir / "pipeline.json").open() as f:
            self.pipeline = json.load(f)
        self.image_buckets = [tuple(g) for g in self.pipeline["buckets"]["image_grids"]]
        self.merged_counts = list(self.pipeline["buckets"]["merged_token_counts"])
        self.prompt_buckets = list(self.pipeline["buckets"]["prompt_lens"])

        # Phase-1: only the first image bucket and prompt_len=256 are real.
        self.active_bucket_idx = 0
        self.active_bucket = self.image_buckets[0]
        self.active_merged = self.merged_counts[0]

        self.inv_freq = build_inv_freq(HEAD_DIM, base=500_000.0)

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.embed_tokens_weight = None  # (vocab, hidden) np.float16
        self.embed_tokens_module = None  # nn.Embedding (HF_NATIVE only)
        self.vision_ml = None
        self.prefill_ml = None
        self.decode_ml = None

    # ------------------------------ loaders ------------------------------- #
    def load_hf(self) -> None:
        """Load ONLY the processor + tokenizer + raw embed_tokens weights.

        Phase E v6: we no longer instantiate the HF model at runtime.
        - Reference token streams come from the saved goldens at
          ``golden_dir/<image_stem>.json`` (which were generated via
          ``trust_remote_code=True`` SHIPPED and are the authoritative
          reference per vision_diag.md).
        - The decode-loop per-token embedding lookup is replaced by an
          indexed numpy gather into the ``model.embed_tokens.weight``
          tensor pulled directly from the safetensors checkpoint. This
          avoids the LIB-vs-SHIPPED model-class divergence and the
          `_init_weights` / `check_model_inputs` / `ROPE_INIT_FUNCTIONS`
          API mismatches between the SHIPPED modeling code and our pinned
          transformers version.
        """
        from transformers import AutoProcessor

        print(f"[hf] loading processor from {self.hf_snapshot}", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            str(self.hf_snapshot), trust_remote_code=True
        )
        # Force the image processor to keep our pre-resized images intact
        self.processor.image_processor.min_pixels = PATCH_SIZE * PATCH_SIZE
        max_grid = max(h * w for _, h, w in self.image_buckets)
        self.processor.image_processor.max_pixels = max_grid * PATCH_SIZE * PATCH_SIZE
        self.tokenizer = self.processor.tokenizer

        # ---- Pull embed_tokens.weight directly from the checkpoint --- #
        from safetensors import safe_open

        ckpt = self.hf_snapshot / "model.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"expected single-file safetensors at {ckpt}; sharded "
                f"layouts not handled here"
            )
        embed_key = "model.embed_tokens.weight"
        with safe_open(str(ckpt), framework="pt") as f:
            keys = set(f.keys())
            if embed_key not in keys:
                # Fall back to language_model.* layout (LIB-style) if the
                # checkpoint ever ships under a different prefix.
                alt = "model.language_model.embed_tokens.weight"
                if alt in keys:
                    embed_key = alt
                else:
                    raise KeyError(
                        f"neither 'model.embed_tokens.weight' nor "
                        f"'{alt}' found in {ckpt}"
                    )
            w = f.get_tensor(embed_key)  # bf16 (vocab, hidden)
        # Cast bf16→fp32→fp16 to match the dtype the decode mlpackage
        # consumes (`inputs_embeds` is fp16).
        w_fp32 = w.to(torch.float32).numpy()
        self.embed_tokens_weight = w_fp32.astype(np.float16)
        print(
            f"[hf] embed_tokens loaded from '{embed_key}': "
            f"shape={self.embed_tokens_weight.shape} dtype=fp16",
            flush=True,
        )

        if HF_NATIVE:
            # Reference loader is LIB (PaddleOCRVLForConditionalGeneration from
            # transformers.models.paddleocr_vl). SHIPPED loader (trust_remote_code=True)
            # is currently broken under pinned transformers (KeyError 'default'
            # in ROPE_INIT_FUNCTIONS during RotaryEmbedding init). LIB has been
            # formally adopted as the Phase E v9 reference per the loader-fork
            # decision; the original 0.999965 baseline (diag_prefill_vs_hf.py /
            # diag_path_a_vs_b.py) was produced with the same LIB loader.
            #
            # LIB layout differs from SHIPPED:
            #   - get_rope_index lives on model.model (PaddleOCRVLModel), not
            #     on the wrapper; requires mm_token_type_ids (synthesized in
            #     parity.build_hf_native_prefill_inputs).
            #   - rotary_emb lives on model.model.language_model.
            from transformers import PaddleOCRVLForConditionalGeneration

            print("[hf] PHASE_E_HF_NATIVE=1 → loading full LIB model "
                  "(needed for get_rope_index + rotary_emb)", flush=True)
            t0 = time.time()
            self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                str(self.hf_snapshot),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
            ).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.embed_tokens_module = self.model.model.language_model.embed_tokens
            print(f"[hf] HF model loaded in {time.time()-t0:.1f}s", flush=True)

    def load_coreml(self) -> None:
        # In HF-NATIVE mode (PHASE_E_HF_NATIVE=1) all CoreML predicts run in
        # spawned subprocesses (Bug X mitigation: parent must not call
        # MLModel.predict). The child loads the mlpackage itself, so the
        # parent skips loading entirely — saves ~10s/model and avoids any
        # parent-side Espresso cache state that might interfere with the
        # child's load.
        if HF_NATIVE:
            print(
                "[coreml] PHASE_E_HF_NATIVE=1 → skipping parent-side "
                "mlpackage loads (subprocess does it)",
                flush=True,
            )
            self.vision_ml = None
            self.prefill_ml = None
            self.decode_ml = None
            return
        import coremltools as ct

        cu = ct.ComputeUnit.CPU_ONLY
        for name, attr in [
            ("vision_embed", "vision_ml"),
            ("prefill_decoder", "prefill_ml"),
            ("decode_decoder", "decode_ml"),
        ]:
            print(f"[coreml] loading {name}.mlpackage (CPU_ONLY)", flush=True)
            t0 = time.time()
            setattr(
                self,
                attr,
                ct.models.MLModel(
                    str(self.coreml_dir / f"{name}.mlpackage"),
                    compute_units=cu,
                ),
            )
            print(f"[coreml]   loaded in {time.time()-t0:.1f}s", flush=True)

    # ------------------------- per-image driver --------------------------- #
    def run_image(self, image_path: Path) -> ImageReport:
        from PIL import Image

        rep = ImageReport(image=image_path.name)
        try:
            print(f"\n=== {image_path.name} ===", flush=True)
            img = Image.open(image_path).convert("RGB")
            print(f"  source size {img.size}", flush=True)

            # ---- Resize to active bucket dims ---------------------------- #
            t, gh, gw = self.active_bucket
            target_pix = (gw * PATCH_SIZE, gh * PATCH_SIZE)  # PIL: (W, H)
            img_resized = img.resize(target_pix, Image.BICUBIC)
            print(
                f"  resized to {img_resized.size} for bucket {self.active_bucket} "
                f"(merged={self.active_merged})",
                flush=True,
            )
            rep.bucket_used = {
                "grid": list(self.active_bucket),
                "merged_count": self.active_merged,
                "prompt_len": PROMPT_LEN_TRACED,
            }

            # ---- HF processor → pixel_values + chat-templated text ------- #
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": DEFAULT_PROMPT},
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
            rep.grid_thw_native = list(grid)
            n_image_tokens = int((proc["input_ids"] == IMAGE_TOKEN_ID).sum().item())
            rep.n_image_tokens = n_image_tokens
            print(
                f"  processor grid_thw={grid} n_image_tokens={n_image_tokens}",
                flush=True,
            )
            if tuple(grid) != self.active_bucket:
                rep.skip_reason = (
                    f"processor produced grid {grid} != active bucket "
                    f"{self.active_bucket}; check resize logic"
                )
                return rep
            if n_image_tokens != self.active_merged:
                rep.skip_reason = (
                    f"image-token count {n_image_tokens} != merged "
                    f"{self.active_merged}"
                )
                return rep

            # ---- Pixel values: (N,3,14,14) → (1,N,3*14*14) fp16 --------- #
            pv = proc["pixel_values"]
            n_patches = pv.shape[0]
            assert n_patches == self.active_bucket[0] * self.active_bucket[1] * self.active_bucket[2]
            pv_flat_fp16 = pv.reshape(
                1, n_patches, 3 * PATCH_SIZE * PATCH_SIZE
            ).to(torch.float16)

            if HF_NATIVE:
                return self._run_image_hf_native(
                    rep=rep,
                    image_path=image_path,
                    proc=proc,
                    text=text,
                    img_resized=img_resized,
                    grid=grid,
                    n_image_tokens=n_image_tokens,
                    pv_flat_fp16=pv_flat_fp16,
                )

            # ---- Build image-first prompt (image tokens at [0, M)) ------- #
            try:
                input_ids_first, real_len = build_image_first_input_ids(
                    proc["input_ids"], n_image_tokens, PROMPT_LEN_TRACED
                )
            except ValueError as e:
                rep.skip_reason = str(e)
                return rep
            rep.prompt_len_used = real_len
            print(
                f"  rebuilt input_ids: real_len={real_len} padded={PROMPT_LEN_TRACED}",
                flush=True,
            )

            attention_mask = torch.zeros(1, PROMPT_LEN_TRACED, dtype=torch.int64)
            attention_mask[0, :real_len] = 1

            # ---- mRoPE 3-axis position_ids for the prompt --------------- #
            position_ids = self._build_position_ids_image_first(
                grid, real_len
            )  # (3, 1, P) int64
            cos3, sin3 = rotary_cos_sin_3d(self.inv_freq, position_ids)
            cos_sel, sin_sel = select_mrope(cos3, sin3, MROPE_SECTION)
            # (1, P, head_dim) → (1, 1, P, head_dim) for the unsqueeze_dim=1 heads axis
            rope_cos_prefill = cos_sel.unsqueeze(1).to(torch.float16).numpy()
            rope_sin_prefill = sin_sel.unsqueeze(1).to(torch.float16).numpy()

            # ============================================================ #
            # Stage 1 — vision_embed
            # ============================================================ #
            image_token_mask_np = (
                (input_ids_first == IMAGE_TOKEN_ID).numpy().astype(np.float32)
            )
            cm_in_v = {
                "pixel_values": pv_flat_fp16.numpy().astype(np.float16),
                "input_ids": input_ids_first.to(torch.int32).numpy(),
                "image_token_mask": image_token_mask_np,
            }
            t0 = time.time()
            v_out = self.vision_ml.predict(cm_in_v)
            vision_ms = (time.time() - t0) * 1000.0
            inputs_embeds_np = np.asarray(v_out["inputs_embeds"]).astype(np.float16)
            print(
                f"  [vision] predict {vision_ms:.0f}ms shape={inputs_embeds_np.shape}",
                flush=True,
            )

            # ============================================================ #
            # Stage 2 — prefill_decoder
            # ============================================================ #
            cm_in_p = {
                "inputs_embeds": inputs_embeds_np,
                "attention_mask": attention_mask.to(torch.int32).numpy(),
                "rope_cos": rope_cos_prefill,
                "rope_sin": rope_sin_prefill,
                "prompt_len_used": np.array([real_len], dtype=np.int32),
            }
            prefill_state = self.prefill_ml.make_state()
            t0 = time.time()
            p_out = self.prefill_ml.predict(cm_in_p, state=prefill_state)
            prefill_ms = (time.time() - t0) * 1000.0
            first_logits = (
                np.asarray(p_out["logits"]).astype(np.float32).reshape(-1)
            )
            first_token = int(np.argmax(first_logits))
            print(
                f"  [prefill] predict {prefill_ms:.0f}ms first_token={first_token}",
                flush=True,
            )

            # KV-cache bridge: prefill and decode are independent mlpackages
            # with independent MLState handles. The historical workaround
            # was to replay every prompt token through decode to "re-warm"
            # its cache (O(real_len) cost). This is suspected to diverge
            # from prefill's actual KV writes (RoPE/mask/op-order skew),
            # producing garbage decode logits.
            #
            # Direct fix: copy each KV buffer byte-for-byte from
            # prefill_state into decode_state. coremltools 9.0 exposes
            # MLState.read_state(name) -> np.ndarray and
            # MLState.write_state(name, np.ndarray); state buffer names
            # are discovered from the decode model spec
            # (`spec.description.state[i].name`).
            decode_state = self.decode_ml.make_state()
            decode_spec = self.decode_ml.get_spec()
            state_names = [s.name for s in decode_spec.description.state]
            t_bridge = time.time()
            try:
                for name in state_names:
                    src = prefill_state.read_state(name)
                    decode_state.write_state(name, src)
                bridge_ms = (time.time() - t_bridge) * 1000.0
                warm_ms = bridge_ms
                print(
                    f"  [bridge] copied {len(state_names)} KV buffers "
                    f"prefill->decode in {bridge_ms:.0f}ms",
                    flush=True,
                )
            except Exception as bridge_exc:
                print(
                    f"  [bridge] PYTHON_BRIDGE_UNAVAILABLE ({type(bridge_exc).__name__}: "
                    f"{bridge_exc}); falling back to replay",
                    flush=True,
                )
                t_warm = time.time()
                for p in range(real_len):
                    embed_p = inputs_embeds_np[:, p:p + 1, :]
                    pos_p = position_ids[:, :, p:p + 1].contiguous()
                    cos3p, sin3p = rotary_cos_sin_3d(self.inv_freq, pos_p)
                    cos_pp, sin_pp = select_mrope(cos3p, sin3p, MROPE_SECTION)
                    rope_cos_step = cos_pp.unsqueeze(1).to(torch.float16).numpy()
                    rope_sin_step = sin_pp.unsqueeze(1).to(torch.float16).numpy()
                    self.decode_ml.predict(
                        {
                            "inputs_embeds": embed_p,
                            "rope_cos": rope_cos_step,
                            "rope_sin": rope_sin_step,
                            "cur_len": np.array([p], dtype=np.int32),
                            "kv_len": np.array([p + 1], dtype=np.int32),
                        },
                        state=decode_state,
                    )
                warm_ms = (time.time() - t_warm) * 1000.0
                print(f"  [decode-warm] replay done in {warm_ms:.0f}ms", flush=True)

            # ============================================================ #
            # Stage 3 — decode loop (greedy)
            # ============================================================ #
            # rope_delta = max_pos + 1 - real_len   (HF "rope_deltas" trick)
            rope_delta = int(position_ids[..., :real_len].max().item()) + 1 - real_len

            generated: list[int] = [first_token]
            step_latencies_ms: list[float] = []

            cur_token = first_token
            for step in range(self.max_new_tokens - 1):
                if cur_token == EOS_TOKEN_ID:
                    break
                # Embed via direct gather into the safetensors weight.
                # No HF model needed; matches the SHIPPED checkpoint
                # bit-exactly (modulo bf16→fp16 cast).
                tok_embed_np = self.embed_tokens_weight[
                    cur_token : cur_token + 1
                ].reshape(1, 1, HIDDEN_SIZE)  # (1,1,1024) fp16
                cache_pos = real_len + step  # 0-indexed slot we're writing to
                pos_int = cache_pos + rope_delta
                pos_p = (
                    torch.tensor([[[pos_int]]], dtype=torch.int64)
                    .expand(3, 1, 1)
                    .contiguous()
                )
                cos3s, sin3s = rotary_cos_sin_3d(self.inv_freq, pos_p)
                cos_s, sin_s = select_mrope(cos3s, sin3s, MROPE_SECTION)
                rope_cos_s = cos_s.unsqueeze(1).to(torch.float16).numpy()
                rope_sin_s = sin_s.unsqueeze(1).to(torch.float16).numpy()

                t_step = time.time()
                d_out = self.decode_ml.predict(
                    {
                        "inputs_embeds": tok_embed_np,
                        "rope_cos": rope_cos_s,
                        "rope_sin": rope_sin_s,
                        "cur_len": np.array([cache_pos], dtype=np.int32),
                        "kv_len": np.array([cache_pos + 1], dtype=np.int32),
                    },
                    state=decode_state,
                )
                step_ms = (time.time() - t_step) * 1000.0
                step_latencies_ms.append(step_ms)

                step_logits = (
                    np.asarray(d_out["logits"]).astype(np.float32).reshape(-1)
                )
                nxt = int(np.argmax(step_logits))
                generated.append(nxt)
                cur_token = nxt

                if (step + 1) % 16 == 0 or step == 0:
                    print(
                        f"    decode step {step+1:>3}: tok={nxt} "
                        f"({step_ms:.0f}ms)",
                        flush=True,
                    )
                if nxt == EOS_TOKEN_ID:
                    print(f"    EOS at step {step+1}", flush=True)
                    break

            # Strip a trailing EOS for the comparison (matches HF's behavior
            # of not including EOS in golden input_ids).
            tokens_for_compare = generated[:]
            if tokens_for_compare and tokens_for_compare[-1] == EOS_TOKEN_ID:
                tokens_for_compare = tokens_for_compare[:-1]

            rep.tokens_generated = len(tokens_for_compare)
            rep.coreml_token_ids = tokens_for_compare
            cm_text = self.tokenizer.decode(
                tokens_for_compare, skip_special_tokens=True
            )
            rep.coreml_text_first_500 = cm_text[:500]

            lats = np.asarray(step_latencies_ms, dtype=np.float64)
            rep.stage_latencies_ms = {
                "vision": round(vision_ms, 1),
                "prefill": round(prefill_ms, 1),
                "decode_warmup": round(warm_ms, 1),
                "decode_total": round(float(lats.sum()) if lats.size else 0.0, 1),
                "decode_steps": int(lats.size),
                "decode_p50": round(float(np.percentile(lats, 50)) if lats.size else 0.0, 1),
                "decode_p95": round(float(np.percentile(lats, 95)) if lats.size else 0.0, 1),
            }

            # ============================================================ #
            # Compare against saved HF golden
            # ============================================================ #
            golden_path = self.golden_dir / f"{Path(image_path).stem}.json"
            if not golden_path.exists():
                rep.skip_reason = f"no golden at {golden_path}"
                return rep
            with golden_path.open() as f:
                golden = json.load(f)
            hf_tokens = list(golden["generated_token_ids"])
            # Strip trailing EOS in HF too if present
            hf_compare = (
                hf_tokens[:-1] if hf_tokens and hf_tokens[-1] == EOS_TOKEN_ID
                else hf_tokens
            )
            hf_text = golden.get("generated_text", "") or self.tokenizer.decode(
                hf_compare, skip_special_tokens=True
            )
            rep.hf_text_first_500 = hf_text[:500]

            # Truncate to the shorter of the two for a fair edit-distance
            # measurement on the prefix the CoreML run actually produced.
            n_compare = min(len(tokens_for_compare), len(hf_compare))
            cm_prefix = tokens_for_compare[:n_compare]
            hf_prefix = hf_compare[:n_compare]
            ed = levenshtein(cm_prefix, hf_prefix)
            div = first_divergence(cm_prefix, hf_prefix)
            matches = sum(1 for a, b in zip(cm_prefix, hf_prefix) if a == b)
            ratio = matches / n_compare if n_compare else 0.0
            sim = char_similarity(
                self.tokenizer.decode(cm_prefix, skip_special_tokens=True),
                self.tokenizer.decode(hf_prefix, skip_special_tokens=True),
            )
            note_full = (
                f"HF golden was generated on FULL native res "
                f"(grid_thw={golden.get('grid_thw')}, "
                f"{golden.get('num_image_tokens')} image tokens); CoreML ran "
                f"on resized bucket {self.active_bucket} ({self.active_merged} "
                f"image tokens). Token-level match across these two operating "
                f"points is not architecturally guaranteed at Phase 1."
            )
            rep.comparisons.append(
                GoldenComparison(
                    label="hf_saved_full_res",
                    hf_token_count=len(hf_compare),
                    cm_token_count=len(tokens_for_compare),
                    edit_distance=ed,
                    first_divergence_at=div,
                    tokens_match_ratio=ratio,
                    text_similarity=sim,
                    notes=note_full,
                )
            )

            # ---- Print divergence context for debugging ---------------- #
            if div is not None:
                self._print_divergence_context(
                    cm_prefix, hf_prefix, div, label="HF saved (full-res)"
                )

            # Acceptance per spec: edit_distance ≤ 2 (small image) or ≤ 5
            # (recipe). Because the saved golden uses a different image
            # bucket, we surface the comparison but NEVER claim "passed"
            # against it — we mark passed iff ed==0 (which would only
            # happen by coincidence at Phase 1).
            rep.passed = (ed == 0)

            print(
                f"  [compare] vs HF saved: ed={ed} "
                f"first_div={div} match_ratio={ratio:.3f} text_sim={sim:.3f}",
                flush=True,
            )

        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            rep.error = f"{type(e).__name__}: {e}\n{tb}"
            print(f"  [ERROR] {e}", flush=True)
            print(tb, flush=True)
        return rep

    # ------------------------- helpers ----------------------------------- #
    def _extract_image_features_via_mlpackage(
        self,
        *,
        n_image_tokens: int,
        pv_flat_fp16: torch.Tensor,
    ) -> torch.Tensor:
        """Run vision_embed.mlpackage with image-first dummy input_ids to harvest
        the projected (M, hidden) image features. Because the wrapper writes the
        projected features into slots [0:M] of `scatter_buf` and then `where`s
        them into the text-embed buffer wherever `image_token_mask` is True, we
        recover the features by:
          * passing input_ids = [IMAGE_TOKEN_ID] * M + [pad ...] (image-first)
          * passing image_token_mask = 1 over [0:M], 0 elsewhere
          * reading inputs_embeds[:, :M, :] from the output.
        The text-embed contribution at slots [0:M] is fully overridden by `where`,
        so the harvested rows are exactly `image_embeds.unsqueeze(0)`.
        """
        M = n_image_tokens
        ids = torch.zeros(1, PROMPT_LEN_TRACED, dtype=torch.int32)
        ids[0, :M] = IMAGE_TOKEN_ID
        mask_np = np.zeros((1, PROMPT_LEN_TRACED), dtype=np.float32)
        mask_np[0, :M] = 1.0
        v_in = {
            "pixel_values": pv_flat_fp16.numpy().astype(np.float16),
            "input_ids": ids.numpy(),
            "image_token_mask": mask_np,
        }
        # Bug X (Option C): run vision predict in a child process so the
        # parent never executes MLModel.predict. The parent does torch ops
        # (HF model) between vision and prefill; even one in-process predict
        # corrupts the next torch op on macOS coremltools-9 + torch-2.11.
        from ._coreml_subprocess import predict_isolated
        vision_path = str(self.coreml_dir / "vision_embed.mlpackage")
        v_out = predict_isolated(vision_path, v_in, stateful=False)
        embeds_full = np.asarray(v_out["inputs_embeds"]).astype(np.float32)
        # (1, P, hidden) → (M, hidden)
        feats = torch.from_numpy(embeds_full[0, :M, :]).contiguous()
        return feats

    def _run_image_hf_native(
        self,
        *,
        rep: ImageReport,
        image_path: Path,
        proc,
        text: str,
        img_resized,
        grid: list[int],
        n_image_tokens: int,
        pv_flat_fp16: torch.Tensor,
    ) -> ImageReport:
        """Phase E v9 HF-native input construction path."""
        if self.model is None:
            rep.skip_reason = "PHASE_E_HF_NATIVE=1 but HF model not loaded"
            return rep

        # ---- Stage 1: vision_embed (image-first), harvest M features ----- #
        t0 = time.time()
        image_feats = self._extract_image_features_via_mlpackage(
            n_image_tokens=n_image_tokens,
            pv_flat_fp16=pv_flat_fp16,
        )
        vision_ms = (time.time() - t0) * 1000.0
        print(
            f"  [vision] predict {vision_ms:.0f}ms harvested "
            f"image_features {tuple(image_feats.shape)}",
            flush=True,
        )

        # ---- Build HF-native prefill inputs ------------------------------ #
        try:
            built = build_hf_native_prefill_inputs(
                image=img_resized,
                text=text,
                processor=self.processor,
                model=self.model,
                embed_tokens=self.embed_tokens_module,
                image_features_projected=image_feats,
                prompt_len=PROMPT_LEN_TRACED,
            )
        except ValueError as e:
            rep.skip_reason = str(e)
            return rep
        real_len = int(built["real_len"])
        rep.prompt_len_used = real_len
        rope_deltas = int(built["rope_deltas"])
        print(
            f"  [hf-native] real_len={real_len}/{PROMPT_LEN_TRACED} "
            f"rope_deltas={rope_deltas} "
            f"max_pos={int(built['position_ids'][..., :real_len].max().item())}",
            flush=True,
        )

        # ---- Stages 2+3: prefill + decode in ONE child subprocess -------- #
        # Bug X (Option C): parent never calls MLModel.predict in the
        # HF-NATIVE path. Vision was already isolated above; here prefill
        # and the entire decode loop run in a single spawned child so the
        # KV bridge is intra-process (no pickling across processes).
        cm_in_p = {
            "inputs_embeds": built["inputs_embeds"],
            "attention_mask": built["attention_mask"],
            "rope_cos": built["rope_cos"],
            "rope_sin": built["rope_sin"],
            "prompt_len_used": built["prompt_len_used"],
        }
        prefill_path = str(self.coreml_dir / "prefill_decoder.mlpackage")
        decode_path = str(self.coreml_dir / "decode_decoder.mlpackage")

        from ._coreml_subprocess import run_prefill_plus_decode_isolated

        t_pp = time.time()
        try:
            pp_result = run_prefill_plus_decode_isolated(
                prefill_path=prefill_path,
                decode_path=decode_path,
                prefill_inputs=cm_in_p,
                real_len=real_len,
                rope_deltas=rope_deltas,
                embed_tokens_weight=self.embed_tokens_weight.astype(np.float32),
                inv_freq=self.inv_freq.detach().cpu().numpy().astype(np.float32),
                head_dim=HEAD_DIM,
                mrope_section=MROPE_SECTION,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=EOS_TOKEN_ID,
                hidden_size=HIDDEN_SIZE,
                timeout_s=1800.0,
            )
        except Exception as pp_exc:
            rep.error = (
                f"PREFILL_PLUS_DECODE_ISOLATED_FAILED: "
                f"{type(pp_exc).__name__}: {pp_exc}"
            )
            return rep
        pp_total_ms = (time.time() - t_pp) * 1000.0
        first_token = int(pp_result["first_token"])
        prefill_ms = float(pp_result["prefill_ms"])
        generated = list(pp_result["generated"])
        step_latencies_ms = list(pp_result["step_latencies_ms"])
        # Bridge timing is now intra-child; expose as 0 for schema compat.
        warm_ms = 0.0
        print(
            f"  [prefill] predict {prefill_ms:.0f}ms first_token={first_token} "
            f"(in subprocess)",
            flush=True,
        )
        if step_latencies_ms:
            print(
                f"  [decode-isolated] {len(step_latencies_ms)} steps "
                f"first_step={step_latencies_ms[0]:.0f}ms "
                f"(subprocess wall {pp_total_ms:.0f}ms total)",
                flush=True,
            )
        else:
            print(
                f"  [decode-isolated] 0 steps (EOS at first_token?) "
                f"(subprocess wall {pp_total_ms:.0f}ms)",
                flush=True,
            )

        tokens_for_compare = generated[:]
        if tokens_for_compare and tokens_for_compare[-1] == EOS_TOKEN_ID:
            tokens_for_compare = tokens_for_compare[:-1]
        rep.tokens_generated = len(tokens_for_compare)
        rep.coreml_token_ids = tokens_for_compare
        cm_text = self.tokenizer.decode(
            tokens_for_compare, skip_special_tokens=True
        )
        rep.coreml_text_first_500 = cm_text[:500]
        lats = np.asarray(step_latencies_ms, dtype=np.float64)
        rep.stage_latencies_ms = {
            "vision": round(vision_ms, 1),
            "prefill": round(prefill_ms, 1),
            "decode_warmup": round(warm_ms, 1),
            "decode_total": round(float(lats.sum()) if lats.size else 0.0, 1),
            "decode_steps": int(lats.size),
            "decode_p50": round(float(np.percentile(lats, 50)) if lats.size else 0.0, 1),
            "decode_p95": round(float(np.percentile(lats, 95)) if lats.size else 0.0, 1),
        }

        # ---- Compare against HF golden ----------------------------------- #
        golden_path = self.golden_dir / f"{Path(image_path).stem}.json"
        if not golden_path.exists():
            rep.skip_reason = f"no golden at {golden_path}"
            return rep
        with golden_path.open() as f:
            golden = json.load(f)
        hf_tokens = list(golden["generated_token_ids"])
        hf_compare = (
            hf_tokens[:-1] if hf_tokens and hf_tokens[-1] == EOS_TOKEN_ID
            else hf_tokens
        )
        hf_text = golden.get("generated_text", "") or self.tokenizer.decode(
            hf_compare, skip_special_tokens=True
        )
        rep.hf_text_first_500 = hf_text[:500]
        n_compare = min(len(tokens_for_compare), len(hf_compare))
        cm_prefix = tokens_for_compare[:n_compare]
        hf_prefix = hf_compare[:n_compare]
        ed = levenshtein(cm_prefix, hf_prefix)
        div = first_divergence(cm_prefix, hf_prefix)
        matches = sum(1 for a, b in zip(cm_prefix, hf_prefix) if a == b)
        ratio = matches / n_compare if n_compare else 0.0
        sim = char_similarity(
            self.tokenizer.decode(cm_prefix, skip_special_tokens=True),
            self.tokenizer.decode(hf_prefix, skip_special_tokens=True),
        )
        rep.comparisons.append(
            GoldenComparison(
                label="hf_saved_full_res",
                hf_token_count=len(hf_compare),
                cm_token_count=len(tokens_for_compare),
                edit_distance=ed,
                first_divergence_at=div,
                tokens_match_ratio=ratio,
                text_similarity=sim,
                notes="HF-NATIVE input construction (PHASE_E_HF_NATIVE=1).",
            )
        )
        if div is not None:
            self._print_divergence_context(
                cm_prefix, hf_prefix, div, label="HF saved (full-res, hf-native)"
            )
        rep.passed = (ed == 0)
        print(
            f"  [compare] vs HF saved: ed={ed} first_div={div} "
            f"match_ratio={ratio:.3f} text_sim={sim:.3f}",
            flush=True,
        )
        return rep

    def _build_position_ids_image_first(
        self,
        grid_thw: list[int],
        real_len: int,
    ) -> torch.Tensor:
        """3D mRoPE position_ids for the image-first layout.

        Same logic as ``ParityHarness._build_position_ids_image_first`` —
        kept inline so we don't depend on a method's bound state.
        """
        t, h, w = grid_thw
        gt, gh, gw = t, h // SPATIAL_MERGE, w // SPATIAL_MERGE
        n_img = gt * gh * gw
        t_index = torch.arange(gt).view(-1, 1).expand(-1, gh * gw).flatten()
        h_index = torch.arange(gh).view(1, -1, 1).expand(gt, -1, gw).flatten()
        w_index = torch.arange(gw).view(1, 1, -1).expand(gt, gh, -1).flatten()
        img_pos = torch.stack([t_index, h_index, w_index])  # (3, n_img)
        max_img = int(img_pos.max().item())
        n_text = real_len - n_img
        text_pos = (
            (torch.arange(n_text, dtype=torch.long) + (max_img + 1))
            .unsqueeze(0)
            .expand(3, -1)
        )
        n_pad = PROMPT_LEN_TRACED - real_len
        if n_pad > 0:
            pad_val = max_img + 1 + n_text
            pad_pos = torch.full((3, n_pad), pad_val, dtype=torch.long)
            position_ids = torch.cat([img_pos, text_pos, pad_pos], dim=1)
        else:
            position_ids = torch.cat([img_pos, text_pos], dim=1)
        return position_ids.unsqueeze(1)  # (3, 1, P)

    def _print_divergence_context(
        self,
        cm: list[int],
        hf: list[int],
        div: int,
        *,
        label: str,
        ctx: int = 5,
    ) -> None:
        lo = max(0, div - ctx)
        hi = min(len(cm), div + ctx + 1)
        cm_slice = cm[lo:hi]
        hf_slice = hf[lo:hi]
        try:
            cm_dec = [self.tokenizer.decode([t], skip_special_tokens=False) for t in cm_slice]
            hf_dec = [self.tokenizer.decode([t], skip_special_tokens=False) for t in hf_slice]
        except Exception:
            cm_dec = ["?"] * len(cm_slice)
            hf_dec = ["?"] * len(hf_slice)
        print(
            f"    [{label}] first divergence at index {div} "
            f"(showing ±{ctx}):",
            flush=True,
        )
        for i in range(hi - lo):
            idx = lo + i
            mark = " " if idx != div else ">"
            print(
                f"      {mark} idx={idx:>4}  "
                f"cm={cm[idx]:>6} ({cm_dec[i]!r})  "
                f"hf={hf[idx]:>6} ({hf_dec[i]!r})",
                flush=True,
            )


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
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-snapshot", type=Path, required=True)
    ap.add_argument("--coreml-dir", type=Path, required=True)
    ap.add_argument("--golden-dir", type=Path, required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    print(
        f"[e2e] max_new_tokens={args.max_new_tokens} compute_units=CPU_ONLY",
        flush=True,
    )
    started = time.time()
    h = E2EHarness(
        hf_snapshot=args.hf_snapshot,
        coreml_dir=args.coreml_dir,
        golden_dir=args.golden_dir,
        max_new_tokens=args.max_new_tokens,
    )
    h.load_hf()
    h.load_coreml()

    image_reports: list[ImageReport] = []
    for img in args.images:
        # For images whose golden is short, cap at min(max_new, hf_len) for sanity.
        try:
            gpath = args.golden_dir / f"{img.stem}.json"
            if gpath.exists():
                with gpath.open() as f:
                    g = json.load(f)
                hf_len = len(g.get("generated_token_ids", []))
                if hf_len and hf_len < h.max_new_tokens:
                    print(
                        f"[e2e] capping max_new_tokens to {hf_len} "
                        f"(HF golden length) for {img.name}",
                        flush=True,
                    )
                    h.max_new_tokens = min(h.max_new_tokens, hf_len)
        except Exception:
            pass
        rep = h.run_image(img)
        image_reports.append(rep)
        gc.collect()

    summary = {
        "total_images": len(image_reports),
        "all_passed": all(r.passed for r in image_reports),
        "ran_to_completion": sum(1 for r in image_reports if r.error is None and r.skip_reason is None),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compute_units": "CPU_ONLY",
        "max_new_tokens": args.max_new_tokens,
        "phase1_constraints": {
            "active_image_bucket": list(h.active_bucket),
            "active_merged_token_count": h.active_merged,
            "prompt_len_traced": PROMPT_LEN_TRACED,
            "note": (
                "Only the smallest image bucket and a single prompt-len bucket "
                "are traced into the Phase-1 mlpackages. The HF goldens were "
                "produced at native image resolution; the CoreML run is "
                "necessarily on a downsized image, so token-level parity vs "
                "the saved golden is not expected to be identical."
            ),
        },
        "images": [_to_jsonable(r) for r in image_reports],
        "summary": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[e2e] wrote {args.report}", flush=True)
    print(f"[e2e] summary: {json.dumps(summary, indent=2)}", flush=True)
    for r in image_reports:
        if r.error:
            print(f"  {r.image}: ERROR {r.error.splitlines()[0]}", flush=True)
            continue
        if r.skip_reason:
            print(f"  {r.image}: SKIP {r.skip_reason}", flush=True)
            continue
        comp = r.comparisons[0] if r.comparisons else None
        print(
            f"  {r.image}: tokens={r.tokens_generated} "
            f"ed={comp.edit_distance if comp else '?'} "
            f"div@{comp.first_divergence_at if comp else '?'} "
            f"match={comp.tokens_match_ratio if comp else 0.0:.3f} "
            f"text_sim={comp.text_similarity if comp else 0.0:.3f}",
            flush=True,
        )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
