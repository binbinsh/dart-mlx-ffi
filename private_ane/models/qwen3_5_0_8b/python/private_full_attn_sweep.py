from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def run_once(layers: list[int], *, token_limit: int, max_new_tokens: int) -> dict[str, object]:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "private_ane/models/qwen3_5_0_8b/python/private_full_attn_infer.py",
            "--token-limit",
            str(token_limit),
            "--max-new-tokens",
            str(max_new_tokens),
            "--attn-layers",
            ",".join(str(layer) for layer in layers),
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "layers": layers,
            "ok": False,
            "error": (proc.stderr or proc.stdout).strip(),
        }
    return {**json.loads(proc.stdout), "ok": True}


def main() -> None:
    args = parse_args()
    reports = []
    for end in range(1, len(FULL_ATTN_LAYERS) + 1):
        layers = FULL_ATTN_LAYERS[:end]
        reports.append(
            run_once(
                layers,
                token_limit=args.token_limit,
                max_new_tokens=args.max_new_tokens,
            )
        )

    safe_prefix = 0
    for report in reports:
        if report.get("ok") and report.get("token_match"):
            safe_prefix = len(report["attn_layers"])
        else:
            break

    payload = {
        "runtime": "qwen35_private_full_attn_sweep",
        "token_limit": args.token_limit,
        "max_new_tokens": args.max_new_tokens,
        "full_attn_layers": FULL_ATTN_LAYERS,
        "safe_prefix_count": safe_prefix,
        "safe_prefix_layers": FULL_ATTN_LAYERS[:safe_prefix],
        "reports": reports,
    }

    if args.json:
        print(json.dumps(payload))
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
