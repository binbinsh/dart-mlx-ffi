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
    parser.add_argument("--token-limit", type=int, default=16)
    parser.add_argument("--lane", type=int, default=32)
    parser.add_argument("--layers")
    parser.add_argument("--online-fast", action="store_true")
    parser.add_argument("--attn-post", action="store_true")
    parser.add_argument("--attn-layers")
    parser.add_argument("--autotune", action="store_true")
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


def _run_json(cmd: list[str], *, env: dict[str, str] | None = None) -> dict[str, object]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    payload = proc.stdout
    start = payload.find("{")
    if start < 0:
        raise SystemExit(f"Missing JSON payload.\nstdout={proc.stdout}")
    return json.loads(payload[start:])


def _autotune_layers(snapshot_dir: Path, artifacts_dir: Path, token_ids_path: Path) -> str:
    layer_report = _run_json(
        [
            "dart",
            "run",
            "tool/qwen35_private_ane_layer_bench.dart",
            "--snapshot-dir",
            str(snapshot_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--token-ids-file",
            str(token_ids_path),
            "--warmup",
            "1",
            "--iters",
            "3",
            "--json",
        ]
    )
    baseline_by_layer = {
        int(row["layer"]): float(row["baseline_dense_ms"])
        for row in layer_report["layers"]
    }

    env = dict(**os.environ, QWEN35_PRIVATE_ANE_PROFILE="1")
    proc = subprocess.run(
        [
            "dart",
            "run",
            "tool/qwen35_private_ane_bench.dart",
            "--snapshot-dir",
            str(snapshot_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--token-ids-file",
            str(token_ids_path),
            "--warmup",
            "1",
            "--iters",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)

    selected: list[int] = []
    for line in proc.stderr.splitlines():
        if not line.startswith("qwen35_private_ane_profile: "):
            continue
        payload = line.split(": ", 1)[1]
        if payload.startswith("layer,"):
            continue
        layer_s, calls_s, cast_s, copy_s, pack_s, ane_s, unpack_s, output_s, total_s = payload.split(",")
        layer = int(layer_s)
        calls = max(int(calls_s), 1)
        total_ms = float(total_s) / calls
        baseline_ms = baseline_by_layer.get(layer)
        if baseline_ms is None:
            continue
        if total_ms < baseline_ms * 0.95:
            selected.append(layer)

    return ",".join(str(layer) for layer in sorted(selected))


def main() -> None:
    args = parse_args()
    if args.autotune and args.layers:
        raise SystemExit("--autotune and --layers are mutually exclusive.")
    snapshot_dir = resolve_snapshot(args.snapshot_dir)
    token_ids = tokenize_prompt(snapshot_dir, args.prompt, args.token_limit)
    if not token_ids:
        raise SystemExit("Tokenizer returned an empty prompt.")

    work_dir = Path(tempfile.mkdtemp(prefix="qwen35_private_ane_"))
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
        ]
        if args.layers:
            make_cmd.extend(["--layers", args.layers])
        subprocess.check_call(make_cmd)

        env = None
        autotuned_layers = None
        if args.autotune:
            autotuned_layers = _autotune_layers(snapshot_dir, artifacts_dir, token_ids_path)
            if autotuned_layers:
                env = dict(**os.environ, QWEN35_PRIVATE_ANE_LAYER_FILTER=autotuned_layers)
        elif args.online_fast:
            env = dict(**os.environ, QWEN35_PRIVATE_ANE_LAYER_PRESET="online-fast")
        if args.attn_post:
            env = dict(**(env or os.environ), QWEN35_PRIVATE_ANE_ATTN_POST="1")
        if args.attn_layers:
            env = dict(**(env or os.environ), QWEN35_PRIVATE_ANE_ATTN_LAYER_FILTER=args.attn_layers)
        proc = subprocess.run(
            [
                "dart",
                "run",
                "tool/qwen35_private_ane_bench.dart",
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
            env=env,
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
        report["online_fast"] = args.online_fast
        report["attn_post"] = args.attn_post
        report["attn_layers"] = args.attn_layers
        report["autotune"] = args.autotune
        report["autotuned_layers"] = autotuned_layers

        if args.json:
            print(json.dumps(report))
            return

        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
