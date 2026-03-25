from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

try:
    from ..common import find_cached_snapshot
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import find_cached_snapshot


ROOT = Path(__file__).resolve().parents[4]
MODEL_ID = "mlx-community/Qwen3.5-0.8B-4bit"
DEFAULT_PROMPT = (
    "Explain why MLX on Apple Silicon is useful for local inference, "
    "and mention latency, memory efficiency, and developer ergonomics."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--token-limit", type=int, default=16)
    parser.add_argument("--private-ane", action="store_true")
    parser.add_argument("--online-fast", action="store_true")
    parser.add_argument("--attn-post", action="store_true")
    parser.add_argument("--attn-layers")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def resolve_snapshot(snapshot_dir: str | None) -> Path:
    if snapshot_dir:
        return Path(snapshot_dir)
    cached = find_cached_snapshot(MODEL_ID)
    if cached is None:
        raise SystemExit(f"Unable to find a cached snapshot for {MODEL_ID}.")
    return cached


def main() -> None:
    args = parse_args()
    snapshot_dir = resolve_snapshot(args.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=True,
    )
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)[: args.token_limit]
    if not token_ids:
        raise SystemExit("Tokenizer returned an empty prompt.")

    own_artifacts_dir = None
    artifacts_dir = args.artifacts_dir
    if args.private_ane and artifacts_dir is None:
        own_artifacts_dir = tempfile.mkdtemp(prefix="qwen35_full_private_ane_")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "private_ane" / "models" / "qwen3_5_0_8b" / "export" / "make_private_ffn.py"),
                "--snapshot-dir",
                str(snapshot_dir),
                "--out-dir",
                own_artifacts_dir,
            ]
        )
        artifacts_dir = own_artifacts_dir

    work_dir = Path(tempfile.mkdtemp(prefix="qwen35_full_infer_"))
    try:
        token_ids_path = work_dir / "token_ids.json"
        token_ids_path.write_text(
            json.dumps(
                {
                    "token_ids": token_ids,
                    "eos_token_id": tokenizer.eos_token_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        cmd = [
            "dart",
            "run",
            "tool/qwen35_full_infer.dart",
            "--snapshot-dir",
            str(snapshot_dir),
            "--token-ids-file",
            str(token_ids_path),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--json",
        ]
        if artifacts_dir is not None:
            cmd.extend(["--private-ane-artifacts-dir", str(artifacts_dir)])

        env = None
        if args.private_ane and args.online_fast:
            env = dict(**os.environ, QWEN35_PRIVATE_ANE_LAYER_PRESET="online-fast")
        if args.private_ane and args.attn_post:
            env = dict(**(env or os.environ), QWEN35_PRIVATE_ANE_ATTN_POST="1")
        if args.private_ane and args.attn_layers:
            env = dict(**(env or os.environ), QWEN35_PRIVATE_ANE_ATTN_LAYER_FILTER=args.attn_layers)
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr or proc.stdout)
        payload = proc.stdout
        start = payload.find("{")
        if start < 0:
            raise SystemExit(f"Missing JSON payload from Dart inference.\nstdout={proc.stdout}")
        report = json.loads(payload[start:])
        generated_token_ids = list(report["generated_token_ids"])
        report["model_id"] = MODEL_ID
        report["prompt"] = args.prompt
        report["generated_text"] = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=False,
        )
        report["new_text"] = tokenizer.decode(
            generated_token_ids[len(token_ids) :],
            skip_special_tokens=False,
        )
        report["private_ane"] = args.private_ane
        report["online_fast"] = args.online_fast
        report["attn_post"] = args.attn_post
        report["attn_layers"] = args.attn_layers

        if args.json:
            print(json.dumps(report))
            return

        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if own_artifacts_dir is not None:
            shutil.rmtree(own_artifacts_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
