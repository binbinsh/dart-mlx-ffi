from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine baseline and candidate runtime reports."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-id")
    args = parser.parse_args()

    baseline = _read(args.baseline)
    candidate = _read(args.candidate)
    payload = combine_reports(
        baseline,
        candidate,
        model_id=args.model_id,
        baseline_path=args.baseline,
        candidate_path=args.candidate,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def combine_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    model_id: str | None = None,
    baseline_path: Path | None = None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    resolved_model = model_id or candidate.get("model_id") or baseline.get("model_id")
    mismatches = []
    for field in ("model_id", "platform", "task"):
        left = baseline.get(field)
        right = candidate.get(field)
        if left and right and left != right:
            mismatches.append({"field": field, "baseline": left, "candidate": right})
    baseline_signature = _input_signature(baseline)
    candidate_signature = _input_signature(candidate)
    if (
        baseline_signature is not None
        and candidate_signature is not None
        and baseline_signature != candidate_signature
    ):
        mismatches.append(
            {
                "field": "input_signature",
                "baseline": baseline_signature,
                "candidate": candidate_signature,
            }
        )
    baseline_run_config = baseline.get("run_config")
    candidate_run_config = candidate.get("run_config")
    if (
        baseline_run_config is not None
        and candidate_run_config is not None
        and baseline_run_config != candidate_run_config
    ):
        mismatches.append(
            {
                "field": "run_config",
                "baseline": baseline_run_config,
                "candidate": candidate_run_config,
            }
        )
    return {
        "model_id": resolved_model,
        "platform": candidate.get("platform") or baseline.get("platform"),
        "baseline": baseline,
        "candidate": candidate,
        "sources": {
            "baseline": str(baseline_path) if baseline_path else None,
            "candidate": str(candidate_path) if candidate_path else None,
        },
        "mismatches": mismatches,
    }


def _read(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Report must be a JSON object: {path}")
    return decoded


def _input_signature(report: dict[str, Any]) -> Any:
    signature = report.get("input_signature")
    if signature is not None:
        return signature
    digest = report.get("input_digest")
    if digest is not None:
        return {"digest": digest}
    return None


if __name__ == "__main__":
    main()
