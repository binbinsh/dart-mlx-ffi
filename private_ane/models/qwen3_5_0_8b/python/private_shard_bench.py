from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--token-limit", type=int, default=16)
    parser.add_argument("--lane", type=int, default=32)
    parser.add_argument("--layers")
    parser.add_argument("--shard-size", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def resolve_snapshot(snapshot_dir: str | None) -> Path:
    if snapshot_dir:
        return Path(snapshot_dir)
    cached = find_cached_snapshot(MODEL_ID)
    if cached is None:
        raise SystemExit(f"Unable to find a cached snapshot for {MODEL_ID}.")
    return cached


def tokenize_prompt(snapshot_dir: Path, prompt: str, token_limit: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=True,
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return token_ids[:token_limit]


def main() -> None:
    args = parse_args()
    snapshot_dir = resolve_snapshot(args.snapshot_dir)
    token_ids = tokenize_prompt(snapshot_dir, args.prompt, args.token_limit)
    if not token_ids:
        raise SystemExit("Tokenizer returned an empty prompt.")

    work_dir = Path(tempfile.mkdtemp(prefix="qwen35_private_ane_shard_"))
    try:
        artifacts_dir = work_dir / "artifacts"
        token_ids_path = work_dir / "token_ids.json"
        token_ids_path.write_text(
            json.dumps({"prompt": args.prompt, "token_ids": token_ids}, indent=2),
            encoding="utf-8",
        )

        make_cmd = [
            sys.executable,
            str(ROOT / "private_ane" / "models" / "qwen3_5_0_8b" / "export" / "make_private_ffn.py"),
            "--snapshot-dir",
            str(snapshot_dir),
            "--out-dir",
            str(artifacts_dir),
            "--lane",
            str(args.lane),
            "--shard-size",
            str(args.shard_size),
        ]
        if args.layers:
            make_cmd.extend(["--layers", args.layers])
        subprocess.check_call(make_cmd)

        proc = subprocess.run(
            [
                "dart",
                "run",
                "tool/qwen35_private_ane_shard_bench.dart",
                "--snapshot-dir",
                str(snapshot_dir),
                "--artifacts-dir",
                str(artifacts_dir),
                "--token-ids-file",
                str(token_ids_path),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr or proc.stdout)
        payload = proc.stdout
        start = payload.find("{")
        if start < 0:
            raise SystemExit(f"Missing JSON payload from Dart benchmark.\nstdout={proc.stdout}")
        report = json.loads(payload[start:])
        report["model_id"] = MODEL_ID
        report["lane"] = args.lane
        report["layer_filter"] = args.layers
        report["shard_size"] = args.shard_size

        if args.json:
            print(json.dumps(report))
            return

        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
