from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import requests
from transformers import AutoTokenizer

try:
    from ..common import cleanup_mlx, find_cached_snapshot, slug
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import cleanup_mlx, find_cached_snapshot, slug

ROOT = Path(__file__).resolve().parents[2]
VENDORS = ROOT / "vendors"
if str(VENDORS / "mlx-vlm") not in sys.path:
    sys.path.insert(0, str(VENDORS / "mlx-vlm"))

import mlx.core as mx
from mlx_vlm.utils import load_model

DEFAULT_MLX_MODEL_ID = "Brooooooklyn/Qwen3.5-9B-unsloth-mlx"
DEFAULT_GGUF_REPO = "unsloth/Qwen3.5-9B-GGUF"
DEFAULT_GGUF_QUANT = "Q4_K_M"
DEFAULT_PROMPT = (
    "Explain why MLX on Apple Silicon is useful for local inference, "
    "and mention latency, memory efficiency, and developer ergonomics."
)
DIRECT_DOWNLOAD_MIN_BYTES = 512 * 1024 * 1024
DIRECT_DOWNLOAD_WORKERS = 8
DIRECT_DOWNLOAD_CHUNK_BYTES = 64 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--mlx-model-id", default=DEFAULT_MLX_MODEL_ID)
    parser.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO)
    parser.add_argument("--gguf-quant", default=DEFAULT_GGUF_QUANT)
    parser.add_argument("--gguf-file")
    parser.add_argument("--hf-endpoint")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--token-limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def resolve_snapshot(
    snapshot_dir: str | None,
    model_id: str,
    *,
    hf_endpoint: str | None,
) -> Path:
    if snapshot_dir:
        return Path(snapshot_dir).expanduser().resolve()
    if hf_endpoint:
        return download_snapshot_via_endpoint(model_id, hf_endpoint=hf_endpoint)
    cached = find_cached_snapshot(model_id)
    if cached is not None:
        return cached
    return Path(
        snapshot_download(
            model_id,
            repo_type="model",
            endpoint=hf_endpoint,
        )
    ).resolve()


def resolve_gguf_model(
    repo: str,
    *,
    quant: str,
    gguf_file: str | None,
    hf_endpoint: str | None,
) -> tuple[Path, str]:
    if gguf_file is not None and Path(gguf_file).exists():
        path = Path(gguf_file).expanduser().resolve()
        return path, path.name

    api = HfApi(endpoint=hf_endpoint)
    files = [name for name in api.list_repo_files(repo_id=repo, repo_type="model") if name.endswith(".gguf")]
    if not files:
        raise SystemExit(f"No GGUF files found in {repo}.")

    selected = gguf_file
    if selected is None:
        lower_quant = quant.lower()
        matches = [name for name in files if lower_quant in Path(name).name.lower()]
        if not matches:
            raise SystemExit(
                f"Unable to find a GGUF file matching quant {quant!r} in {repo}.\n"
                f"Available files: {files}"
            )
        matches.sort(key=lambda name: (len(name), name))
        selected = matches[0]

    if hf_endpoint:
        out_dir = ROOT / "benchmark" / "out" / "hf_downloads" / slug(repo)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / Path(selected).name
        download_repo_file_via_endpoint(
            repo,
            selected,
            target,
            hf_endpoint=hf_endpoint,
        )
        return target.resolve(), selected

    downloaded = hf_hub_download(
        repo_id=repo,
        filename=selected,
        repo_type="model",
        endpoint=hf_endpoint,
    )
    return Path(downloaded).resolve(), selected


def download_snapshot_via_endpoint(model_id: str, *, hf_endpoint: str) -> Path:
    out_dir = ROOT / "benchmark" / "out" / "hf_downloads" / slug(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=hf_endpoint)
    files = api.list_repo_files(repo_id=model_id, repo_type="model")
    for name in files:
      target = out_dir / name
      target.parent.mkdir(parents=True, exist_ok=True)
      if name.endswith(".safetensors"):
          download_repo_file_via_endpoint(
              model_id,
              name,
              target,
              hf_endpoint=hf_endpoint,
          )
      else:
          hf_hub_download(
              repo_id=model_id,
              filename=name,
              repo_type="model",
              endpoint=hf_endpoint,
              local_dir=out_dir,
          )
    return out_dir.resolve()


def download_repo_file_via_endpoint(
    repo_id: str,
    filename: str,
    target: Path,
    *,
    hf_endpoint: str,
) -> None:
    url = build_resolve_url(hf_endpoint, repo_id, filename)
    session = requests.Session()
    response = session.head(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    final_url = response.url
    total_size = int(response.headers.get("Content-Length", "0"))
    accept_ranges = response.headers.get("Accept-Ranges", "").lower()
    if target.exists() and total_size > 0 and target.stat().st_size == total_size:
        return
    if total_size < DIRECT_DOWNLOAD_MIN_BYTES or "bytes" not in accept_ranges:
        download_file_single(session, final_url, target)
        return
    download_file_ranges(session, final_url, target, total_size)


def build_resolve_url(hf_endpoint: str, repo_id: str, filename: str) -> str:
    endpoint = hf_endpoint.rstrip("/")
    encoded = "/".join(quote(part) for part in filename.split("/"))
    return f"{endpoint}/{repo_id}/resolve/main/{encoded}"


def download_file_single(session: requests.Session, url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    with session.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(target)


def download_file_ranges(
    session: requests.Session,
    url: str,
    target: Path,
    total_size: int,
) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.truncate(total_size)

    ranges = []
    start = 0
    while start < total_size:
        end = min(total_size - 1, start + DIRECT_DOWNLOAD_CHUNK_BYTES - 1)
        ranges.append((start, end))
        start = end + 1

    def worker(byte_range: tuple[int, int]) -> None:
        start, end = byte_range
        with requests.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=30,
        ) as response:
            response.raise_for_status()
            with tmp.open("r+b") as handle:
                handle.seek(start)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=DIRECT_DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(worker, byte_range) for byte_range in ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    tmp.replace(target)


def encode_prompt(
    snapshot_dir: Path,
    prompt: str,
    *,
    token_limit: int,
    trust_remote_code: bool,
) -> tuple[AutoTokenizer, list[int], str]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=trust_remote_code,
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)[:token_limit]
    if not token_ids:
        raise SystemExit("Tokenizer returned an empty prompt.")
    normalized_prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    return tokenizer, token_ids, normalized_prompt


def run_dart_generation(
    snapshot_dir: Path,
    *,
    token_ids: list[int],
    eos_token_id: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    work_dir = Path(tempfile.mkdtemp(prefix="qwen35_unsloth_dart_"))
    try:
        token_ids_path = work_dir / "token_ids.json"
        token_ids_path.write_text(
            json.dumps(
                {
                    "token_ids": token_ids,
                    "eos_token_id": eos_token_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                "dart",
                "run",
                "tool/qwen35_full_infer.dart",
                "--snapshot-dir",
                str(snapshot_dir),
                "--token-ids-file",
                str(token_ids_path),
                "--max-new-tokens",
                str(max_new_tokens),
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr or proc.stdout)
        start = proc.stdout.find("{")
        if start < 0:
            raise SystemExit(f"Missing JSON payload from Dart inference.\n{proc.stdout}")
        return json.loads(proc.stdout[start:])
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_python_mlx_generation(
    snapshot_dir: Path,
    *,
    token_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    model = load_model(snapshot_dir, lazy=False)
    lm = model.language_model
    lm._position_ids = None
    lm._rope_deltas = None
    cache = lm.make_cache()
    prompt = mx.array([token_ids], dtype=mx.int32)
    started = time.perf_counter()
    out = _run_python_mlx_text_logits(lm, prompt, cache)[:, -1, :].astype(mx.float32)
    mx.eval(out)
    mx.synchronize()
    first_logits = np.asarray(out).astype(np.float32, copy=False)[0]
    next_token = int(np.argmax(first_logits))
    generated = list(token_ids)
    generated.append(next_token)
    for _ in range(max_new_tokens - 1):
        step = mx.array([[next_token]], dtype=mx.int32)
        out = _run_python_mlx_text_logits(lm, step, cache)[:, -1, :].astype(mx.float32)
        mx.eval(out)
        mx.synchronize()
        next_token = int(np.argmax(np.asarray(out).astype(np.float32, copy=False)[0]))
        generated.append(next_token)
    total_ms = (time.perf_counter() - started) * 1000.0
    try:
        return {
            "generated_token_ids": generated,
            "new_token_ids": generated[len(token_ids) :],
            "prompt_length": len(token_ids),
            "generated_length": len(generated),
            "generate_ms": total_ms,
            "per_new_token_ms": total_ms / max(1, len(generated) - len(token_ids)),
            "logits16": [float(v) for v in first_logits[:16]],
        }
    finally:
        del prompt
        del cache
        del model
        cleanup_mlx(mx)


def _run_python_mlx_text_logits(model, input_ids, cache):
    batch_size, seq_length = input_ids.shape
    position_ids = None
    if cache and cache[model.model.fa_idx] is not None:
        cache_offset = cache[model.model.fa_idx].offset
        delta = mx.array(
            cache_offset + model._rope_deltas if model._rope_deltas is not None else cache_offset
        )
        position_ids = mx.arange(seq_length).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
        if delta.ndim == 0:
            delta = mx.expand_dims(delta, axis=0)
        if delta.shape[0] < batch_size:
            delta = mx.tile(delta, (batch_size, 1))
        else:
            delta = delta[:batch_size]
        position_ids = mx.add(position_ids, delta)[None, ...]
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_length))
    else:
        position_ids, rope_deltas = model.get_rope_index(input_ids)
        model._rope_deltas = rope_deltas
        model._position_ids = position_ids

    hidden = model.model(input_ids, cache=cache, position_ids=position_ids)
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(hidden)
    return model.lm_head(hidden)


def run_gguf_generation(
    *,
    model_path: Path,
    prompt: str,
    max_new_tokens: int,
    ctx_size: int,
    threads: int | None,
) -> dict[str, Any]:
    cmd = [
        "llama-completion",
        "--model",
        str(model_path),
        "--predict",
        str(max_new_tokens),
        "--prompt",
        prompt,
        "--temp",
        "0",
        "--seed",
        "0",
        "--ctx-size",
        str(ctx_size),
        "--no-display-prompt",
        "--simple-io",
        "-no-cnv",
        "--perf",
    ]
    if threads is not None:
        cmd.extend(["--threads", str(threads)])
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    timing_source = "\n".join([proc.stdout, proc.stderr])
    timings = parse_llama_perf(timing_source)
    return {
        "command": cmd,
        "new_text": strip_ansi(proc.stdout).strip(),
        "stderr": proc.stderr,
        **timings,
    }


def parse_llama_perf(text: str) -> dict[str, Any]:
    patterns = {
        "load_ms": r"load time\s*=\s*([0-9.]+)\s*ms",
        "prompt_eval": (
            r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens"
            r".*?\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
        ),
        "eval": (
            r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs?"
            r".*?\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
        ),
        "total_ms": r"total time\s*=\s*([0-9.]+)\s*ms",
    }
    out: dict[str, Any] = {}
    if match := re.search(patterns["load_ms"], text):
        out["load_ms"] = float(match.group(1))
    if match := re.search(patterns["prompt_eval"], text, re.IGNORECASE | re.DOTALL):
        out["prompt_eval_ms"] = float(match.group(1))
        out["prompt_eval_tokens"] = int(match.group(2))
        out["prompt_eval_ms_per_token"] = float(match.group(3))
        out["prompt_eval_tps"] = float(match.group(4))
    if match := re.search(patterns["eval"], text, re.IGNORECASE | re.DOTALL):
        out["eval_ms"] = float(match.group(1))
        out["eval_runs"] = int(match.group(2))
        out["eval_ms_per_token"] = float(match.group(3))
        out["eval_tps"] = float(match.group(4))
    if match := re.search(patterns["total_ms"], text):
        out["total_ms"] = float(match.group(1))
    return out


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def compare_token_ids(lhs: list[int], rhs: list[int]) -> dict[str, Any]:
    mismatch_index = None
    for index, (a, b) in enumerate(zip(lhs, rhs)):
        if a != b:
            mismatch_index = index
            break
    if mismatch_index is None and len(lhs) != len(rhs):
        mismatch_index = min(len(lhs), len(rhs))
    return {
        "exact_match": lhs == rhs,
        "lhs_length": len(lhs),
        "rhs_length": len(rhs),
        "first_mismatch_index": mismatch_index,
    }


def attach_decoded_text(
    report: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    generated = list(report["generated_token_ids"])
    prompt_length = int(report["prompt_length"])
    report["generated_text"] = tokenizer.decode(
        generated,
        skip_special_tokens=False,
    )
    report["new_text"] = tokenizer.decode(
        generated[prompt_length:],
        skip_special_tokens=False,
    )
    return report


def main() -> None:
    args = parse_args()
    hf_endpoint = args.hf_endpoint or os.environ.get("HF_ENDPOINT")
    snapshot_dir = resolve_snapshot(
        args.snapshot_dir,
        args.mlx_model_id,
        hf_endpoint=hf_endpoint,
    )
    gguf_model_path, resolved_gguf_file = resolve_gguf_model(
        args.gguf_repo,
        quant=args.gguf_quant,
        gguf_file=args.gguf_file,
        hf_endpoint=hf_endpoint,
    )
    tokenizer, token_ids, normalized_prompt = encode_prompt(
        snapshot_dir,
        args.prompt,
        token_limit=args.token_limit,
        trust_remote_code=args.trust_remote_code,
    )
    dart_report = attach_decoded_text(
        run_dart_generation(
            snapshot_dir,
            token_ids=token_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
        ),
        tokenizer,
    )
    python_report = attach_decoded_text(
        run_python_mlx_generation(
            snapshot_dir,
            token_ids=token_ids,
            max_new_tokens=args.max_new_tokens,
        ),
        tokenizer,
    )
    gguf_report = run_gguf_generation(
        model_path=gguf_model_path,
        prompt=normalized_prompt,
        max_new_tokens=args.max_new_tokens,
        ctx_size=max(args.ctx_size, len(token_ids) + args.max_new_tokens + 16),
        threads=args.threads,
    )

    report = {
        "kind": "qwen3_5_unsloth_compare",
        "mlx_model_id": args.mlx_model_id,
        "hf_endpoint": hf_endpoint,
        "gguf_repo": args.gguf_repo,
        "gguf_quant": args.gguf_quant,
        "gguf_file": resolved_gguf_file,
        "gguf_model_path": str(gguf_model_path),
        "snapshot_dir": str(snapshot_dir),
        "prompt": args.prompt,
        "normalized_prompt": normalized_prompt,
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "max_new_tokens": args.max_new_tokens,
        "dart": dart_report,
        "python_mlx": python_report,
        "gguf": gguf_report,
        "dart_vs_python": compare_token_ids(
            list(dart_report["generated_token_ids"]),
            list(python_report["generated_token_ids"]),
        ),
    }

    out_path = (
        Path(args.out)
        if args.out
        else ROOT
        / "benchmark"
        / "out"
        / "qwen3_5_unsloth_compare"
        / f"{slug(args.mlx_model_id)}__{slug(args.gguf_quant)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False))
        return

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report_path={out_path}")


if __name__ == "__main__":
    main()
