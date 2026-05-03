from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_LOGITS_TOL = 5e-2


@dataclass
class ParityReport:
    bucket: tuple[int, int, int]
    prompt_len: int
    max_abs_err_top32: float
    max_abs_err_full: float
    cosine_sim: float
    pt_top1_token: int
    cm_top1_token: int
    pt_top1_logit: float
    cm_top1_logit: float
    passed: bool
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compare_logits(
    pt_logits: torch.Tensor | np.ndarray,
    cm_logits: torch.Tensor | np.ndarray,
    *,
    bucket: tuple[int, int, int],
    prompt_len: int,
    tolerance: float = DEFAULT_LOGITS_TOL,
) -> ParityReport:
    pt = (
        pt_logits.detach().to(torch.float32).cpu().numpy()
        if isinstance(pt_logits, torch.Tensor)
        else np.asarray(pt_logits)
    ).astype(np.float32).ravel()
    cm = (
        cm_logits.detach().to(torch.float32).cpu().numpy()
        if isinstance(cm_logits, torch.Tensor)
        else np.asarray(cm_logits)
    ).astype(np.float32).ravel()
    if pt.shape != cm.shape:
        raise ValueError(f"logit shape mismatch: pt={pt.shape} vs cm={cm.shape}")
    diff = np.abs(pt - cm)
    top32_idx = np.argsort(-pt)[:32]
    top32_err = float(diff[top32_idx].max())
    full_err = float(diff.max())
    denom = float(np.linalg.norm(pt) * np.linalg.norm(cm)) or 1.0
    cosine = float(np.dot(pt, cm) / denom)
    pt_top1 = int(np.argmax(pt))
    cm_top1 = int(np.argmax(cm))
    return ParityReport(
        bucket=bucket,
        prompt_len=prompt_len,
        max_abs_err_top32=top32_err,
        max_abs_err_full=full_err,
        cosine_sim=cosine,
        pt_top1_token=pt_top1,
        cm_top1_token=cm_top1,
        pt_top1_logit=float(pt[pt_top1]),
        cm_top1_logit=float(cm[cm_top1]),
        passed=(top32_err < tolerance) and (pt_top1 == cm_top1),
        tolerance=tolerance,
    )


def write_report(report: ParityReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
