"""Vision-stage logits parity between PyTorch HF reference and CoreML.

Phase A — vision_embed parity.

We compare the projector output of the converted pipeline against the HF
PyTorch reference (FP32) on identical image inputs:

    Stage A  vision_embed       — projector output (image_embeds) BEFORE
                                  scatter into prompt embeddings.

Hybrid-OCR contract (issue #1, commit #11):
    vision_embed.mlpackage emits ``image_embeds`` of shape
    ``[num_image_tokens, hidden]``. The legacy 4-stage CoreML pipeline
    (token_embed/prefill_decoder/decode_decoder) was removed in commit
    #11 — the decoder now runs in MLX via ``PaddleOcrVlHybridRunner``.
    Stages B and C and their subprocess-isolated CoreML predicts have
    therefore been deleted from this harness.

Outputs land in ``--report`` JSON. Tolerance default: vision MAE < 5e-3.

Run:
  .venv/bin/python -m benchmark.runtime.converters.paddleocr_vl_coreml.parity \\
    --hf-snapshot /tmp/.../snapshots/<sha> \\
    --coreml-dir  /tmp/paddleocr-vl-coreml-rebuild/converted \\
    --images img1.jpg img2.jpg \\
    --report /tmp/paddleocr-vl-coreml-rebuild/parity_report.json
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---- pipeline.json constants --------------------------------------------- #
PATCH_SIZE = 14
SPATIAL_MERGE = 2
IMAGE_TOKEN_ID = 100295
DEFAULT_VISION_TOL = 5e-3


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
class ImageReport:
    path: str
    grid_thw: list[int]
    bucket_used: int | None
    n_image_tokens: int
    notes: str = ""
    error: str | None = None
    stage_a_vision_embed: StageAReport | None = None
    overall_passed: bool = False


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


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class ParityHarness:
    def __init__(
        self,
        hf_snapshot: Path,
        coreml_dir: Path,
        vision_tol: float,
        progress_path: Path | None = None,
    ) -> None:
        self.hf_snapshot = hf_snapshot
        self.coreml_dir = coreml_dir
        self.vision_tol = float(vision_tol)

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

        self.vision_path = str(coreml_dir / "vision_embed.mlpackage")

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

            pv = proc["pixel_values"]  # (N, 3, 14, 14) fp32
            n_patches = pv.shape[0]
            assert n_patches == bucket[0] * bucket[1] * bucket[2]
            pv_flat_fp16 = pv.reshape(
                1, n_patches, 3 * PATCH_SIZE * PATCH_SIZE
            ).to(torch.float16)

            # HF reference: projector output BEFORE scatter into prompt
            # embeddings. Hybrid-OCR contract (issue #1, commit #11) — the
            # decoder runs in MLX so we no longer need the fused embeds /
            # prefill / decode reference paths the legacy 4-stage harness
            # carried.
            hf_image_embeds_projected = self._compute_hf_vision_reference(
                pixel_values_fp32=pv,
                grid_thw=grid,
            )

            # ============================================================ A
            #
            # Stage A compares CoreML image_embeds (projector output,
            # rank-2 [M, hidden]) against the HF projector output before
            # scatter. The Dart-side scatter (paddleOcrVlScatterImageEmbeddings
            # in embed.dart) is the only consumer of these features at
            # runtime — see PaddleOcrVlHybridRunner (commit #8).
            rep.stage_a_vision_embed = self._stage_a(
                pv_flat_fp16=pv_flat_fp16,
                grid_thw=list(grid),
                hf_image_embeds_projected=hf_image_embeds_projected,
            )

            rep.overall_passed = bool(
                rep.stage_a_vision_embed and rep.stage_a_vision_embed.passed
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            rep.error = f"{type(e).__name__}: {e}\n{tb}"
            self._log(f"  [ERROR] {e}")
            print(tb, flush=True)
        return rep

    # ------------------------------------------------------ HF reference pass
    def _compute_hf_vision_reference(
        self,
        *,
        pixel_values_fp32: torch.Tensor,
        grid_thw: list[int],
    ) -> torch.Tensor:
        """HF projector output before scatter — Stage A reference.

        Hybrid-OCR (issue #1, commit #11): the decoder runs in MLX so we
        only need the vision tower's projected features here. This returns
        ``image_outputs.pooler_output`` (shape ``[num_image_tokens, hidden]``),
        i.e. the same tensor HF would scatter into the text-embed buffer.
        """
        m = self.model
        inner = m.model
        self._log("[hf] computing image_embeds (projector output)")
        t0 = time.time()
        image_outputs = inner.get_image_features(
            pixel_values=pixel_values_fp32.detach().clone(),
            image_grid_thw=torch.tensor([list(grid_thw)], dtype=torch.long),
            return_dict=True,
        )
        image_embeds = image_outputs.pooler_output
        self._log(
            f"  image_embeds shape={tuple(image_embeds.shape)} "
            f"any_nan={torch.isnan(image_embeds).any().item()} "
            f"abs_max={image_embeds.abs().max().item():.3e} "
            f"in {time.time()-t0:.2f}s"
        )
        return image_embeds

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
        self._log("[A] vision_embed → image_embeds")
        cm_in = {
            "pixel_values": pv_flat_fp16.numpy().astype(np.float16),
            "image_grid_thw": np.asarray(grid_thw, dtype=np.int32),
        }
        # In-process CoreML predict. The legacy harness used spawn-based
        # subprocesses (``_coreml_subprocess``) to sidestep coremltools↔torch
        # SIGSEGVs that surfaced when the prefill/decode stages were run back
        # to back. With Stages B and C removed (commit #11) the vision-only
        # predict is safe in-process.
        import coremltools as ct  # noqa: PLC0415

        mlmodel = ct.models.MLModel(self.vision_path)
        t0 = time.time()
        cm_out = mlmodel.predict(cm_in)
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
    if r.stage_a_vision_embed:
        return r.stage_a_vision_embed.mae
    return 0.0


def _image_invariance_summary(reports: list[ImageReport]) -> dict[str, Any]:
    """If we have ≥2 images, report Stage A cosine spread across them.

    Stage A's per-image cosine to HF should vary with the input — a flat
    cosine across distinct images would be highly suspicious (it was the
    original image-invariance bug we shipped a regression for). We can't
    do better here without storing the actual tensors.
    """
    out: dict[str, Any] = {"n_images": len(reports)}
    if len(reports) < 2:
        return out
    a_cosines = [r.stage_a_vision_embed.cosine for r in reports
                 if r.stage_a_vision_embed]
    out["stage_a_cosines_to_hf"] = a_cosines
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-snapshot", type=Path, required=True)
    ap.add_argument("--coreml-dir", type=Path, required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--vision-tolerance", type=float, default=DEFAULT_VISION_TOL)
    ap.add_argument(
        "--progress-log",
        type=Path,
        default=Path("/tmp/paddleocr-vl-coreml-rebuild/parity_progress.log"),
    )
    args = ap.parse_args()

    started = time.time()
    print(
        f"[parity] vision_tol={args.vision_tolerance:.1e}",
        flush=True,
    )
    h = ParityHarness(
        hf_snapshot=args.hf_snapshot,
        coreml_dir=args.coreml_dir,
        vision_tol=args.vision_tolerance,
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
        "vision_tolerance": args.vision_tolerance,
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
        a_str = (
            f"A={'PASS' if a.passed else 'FAIL'}(mae={a.mae:.2e})"
            if a else "A=?"
        )
        print(f"  {Path(r.path).name}: {a_str}", flush=True)
        if r.error:
            print(f"    error: {r.error.splitlines()[0]}", flush=True)
    return 0 if summary["all_passed"] else 1


# Backwards-compatible shims for pipeline.py.
from .parity_compat import ParityReport, compare_logits, write_report  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
