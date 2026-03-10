from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

try:
    from .common import add_vendor_to_path, cleanup_mlx, resolve_model_path
    from .recent_text_sweep import slug
    from .recent_vlm_sweep import export_model as export_vlm_model
    from .recent_vlm_sweep import extract_logits as vlm_extract_logits
    from .recent_vlm_sweep import prepare_model_inputs as prepare_vlm_inputs
    from .recent_vlm_sweep import dart_forward as vlm_dart_forward
    from .recent_tts_sweep import export_model as export_ming_tts_model
    from .recent_tts_sweep import python_forward as ming_python_forward
    from .recent_tts_sweep import dart_forward as ming_dart_forward
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import add_vendor_to_path, cleanup_mlx, resolve_model_path
    from recent_text_sweep import slug
    from recent_vlm_sweep import export_model as export_vlm_model
    from recent_vlm_sweep import extract_logits as vlm_extract_logits
    from recent_vlm_sweep import prepare_model_inputs as prepare_vlm_inputs
    from recent_vlm_sweep import dart_forward as vlm_dart_forward
    from recent_tts_sweep import export_model as export_ming_tts_model
    from recent_tts_sweep import python_forward as ming_python_forward
    from recent_tts_sweep import dart_forward as ming_dart_forward

import mlx.core as mx
from mlx_lm import load as load_lm

ROOT = Path(__file__).resolve().parents[1]
TEXT_PROMPT = "Explain why MLX is useful for local inference on Apple Silicon."
VLM_PROMPT = "Describe this image briefly."


def run_script_capture(cmd: list[str], *, env: dict[str, str]) -> str:
    temp_dir = Path(tempfile.mkdtemp())
    stdout_path = temp_dir / "runner.stdout"
    subprocess.run(
        ["script", "-q", str(stdout_path), *cmd],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
    )
    return stdout_path.read_text(encoding="utf-8")


def parse_last_json(raw: str) -> dict[str, object]:
    matches = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
    if not matches:
        raise RuntimeError(f"No JSON payload found in output:\n{raw}")
    return json.loads(matches[-1])


def compare_lists(a: list[float], b: list[float]) -> tuple[float, float]:
    diffs = [abs(x - y) for x, y in zip(a, b)]
    return (
        max(diffs) if diffs else 0.0,
        (sum(diffs) / len(diffs)) if diffs else 0.0,
    )


def _text_load_kwargs(model_id: str) -> dict[str, object]:
    if "IQuest-Coder" in model_id:
        return {
            "tokenizer_config": {"trust_remote_code": True},
            "model_config": {"trust_remote_code": True},
        }
    return {}


def text_bench(model_id: str, *, warmup: int = 3, iters: int = 10) -> dict[str, object]:
    load_kwargs = _text_load_kwargs(model_id)
    model, tokenizer = load_lm(model_id, lazy=False, **load_kwargs)
    token_ids = tokenizer.encode(TEXT_PROMPT)[:24]
    tokens = mx.array([token_ids], dtype=mx.int32)

    def forward():
        out = model(tokens)
        logits = (out.logits if hasattr(out, "logits") else out)[:, -1, :16].astype(mx.float32)
        mx.eval(logits)
        mx.synchronize()
        return logits

    for _ in range(warmup):
        forward()
    started = time.perf_counter()
    last = None
    for _ in range(iters):
        last = forward()
    py_ms = (time.perf_counter() - started) * 1000.0 / iters
    py_values = [float(v) for v in last.reshape([-1]).tolist()]
    export_dir = ROOT / "benchmark" / "out" / "publish" / slug(model_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "function.mlxfn"
    input_path = export_dir / "inputs.safetensors"
    model2, _tokenizer2 = load_lm(model_id, lazy=False, **load_kwargs)
    tokens2 = mx.array([token_ids], dtype=mx.int32)

    def export_forward(input_ids):
        out = model2(input_ids)
        return (out.logits if hasattr(out, "logits") else out)[:, -1, :16].astype(mx.float32)

    if export_path.exists():
        export_path.unlink()
    mx.export_function(str(export_path), export_forward, tokens2)
    mx.save_safetensors(str(input_path), {"input_ids": tokens2})
    temp_dir = Path(tempfile.mkdtemp())
    values_path = temp_dir / "out.safetensors"
    env = dict(os.environ)
    env["GENERIC_VALUES_PATH"] = str(values_path)
    env["GENERIC_WARMUP"] = str(warmup)
    env["GENERIC_ITERS"] = str(iters)
    raw = run_script_capture(
        ["dart", "run", "benchmark/generic_import_run.dart", str(export_path), str(input_path)],
        env=env,
    )
    payload = parse_last_json(raw)
    dart_values = [
        float(v)
        for v in mx.load(str(values_path))["output"].reshape([-1]).astype(mx.float32).tolist()
    ]
    max_diff, mean_diff = compare_lists(py_values, dart_values)
    cleanup_mlx(mx)
    return {
        "model_id": model_id,
        "kind": "text",
        "input_desc": f"{len(token_ids)} text tokens",
        "comparison": "last-token logits[:16]",
        "python_ms": py_ms,
        "dart_ms": float(payload["per_iter_ms"]),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def vlm_bench(model_id: str, *, warmup: int = 3, iters: int = 10) -> dict[str, object]:
    export_dir = ROOT / "benchmark" / "out" / "publish" / slug(model_id)
    export_path, input_path, input_names = export_vlm_model(model_id, export_dir)

    # Python timed loop
    model, processor, inputs = prepare_vlm_inputs(model_id)
    call_inputs = {
        name: value
        for name, value in inputs.items()
        if isinstance(value, mx.array) and value.size > 0
    }
    input_ids = call_inputs["input_ids"]
    attention_mask = call_inputs.get("attention_mask")
    pixel_values = call_inputs.get("pixel_values")
    kwargs = {
        key: value
        for key, value in call_inputs.items()
        if key not in {"input_ids", "attention_mask", "pixel_values"}
    }

    def forward():
        out = model(input_ids, pixel_values, attention_mask, **kwargs)
        logits = vlm_extract_logits(out)[:, -1, :16].astype(mx.float32)
        mx.eval(logits)
        mx.synchronize()
        return logits

    for _ in range(warmup):
        forward()
    started = time.perf_counter()
    last = None
    for _ in range(iters):
        last = forward()
    py_ms = (time.perf_counter() - started) * 1000.0 / iters
    py_values = [float(v) for v in last.reshape([-1]).tolist()]

    temp_dir = Path(tempfile.mkdtemp())
    values_path = temp_dir / "out.safetensors"
    env = dict(os.environ)
    env["GENERIC_VALUES_PATH"] = str(values_path)
    env["GENERIC_WARMUP"] = str(warmup)
    env["GENERIC_ITERS"] = str(iters)
    raw = run_script_capture(
        [
            "dart",
            "run",
            "benchmark/generic_import_run.dart",
            str(export_path),
            str(input_path),
            json.dumps(input_names),
        ],
        env=env,
    )
    payload = parse_last_json(raw)
    dart_values = [
        float(v)
        for v in mx.load(str(values_path))["output"].reshape([-1]).astype(mx.float32).tolist()
    ]
    max_diff, mean_diff = compare_lists(py_values, dart_values)
    cleanup_mlx(mx)
    return {
        "model_id": model_id,
        "kind": "vlm",
        "input_desc": "1 synthetic image + text prompt",
        "comparison": "last-token logits[:16]",
        "python_ms": py_ms,
        "dart_ms": float(payload["per_iter_ms"]),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def ming_tts_bench(model_id: str, *, warmup: int = 3, iters: int = 10) -> dict[str, object]:
    export_dir = ROOT / "benchmark" / "out" / "publish" / slug(model_id)
    export_path, input_path, input_names = export_ming_tts_model(model_id, export_dir)

    model, inputs, _, _ = ming_python_forward(model_id)

    def forward():
        out = model.flowloss.cfm.model.forward_with_cfg(
            inputs["x"],
            inputs["t"],
            inputs["c"],
            cfg_scale=0.3,
            latent_history=inputs["latent_history"],
            patch_size=4,
        ).astype(mx.float32)
        mx.eval(out)
        mx.synchronize()
        return out

    for _ in range(warmup):
        forward()
    started = time.perf_counter()
    last = None
    for _ in range(iters):
        last = forward()
    py_ms = (time.perf_counter() - started) * 1000.0 / iters
    py_values = [float(v) for v in last.reshape([-1]).tolist()]

    dart_values, dart_ms = ming_dart_forward(export_path, input_path, input_names)
    max_diff, mean_diff = compare_lists(py_values, dart_values)
    cleanup_mlx(mx)
    return {
        "model_id": model_id,
        "kind": "tts",
        "input_desc": "deterministic DiT subgraph tensors",
        "comparison": "forward_with_cfg output",
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def kitten_bench(*, warmup: int = 3, iters: int = 10) -> dict[str, object]:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "benchmark/kitten_tts/mlx_audio_compare.py",
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    return {
        "model_id": "mlx-community/kitten-tts-nano-0.8-6bit",
        "kind": "tts",
        "input_desc": "fixed text + voice",
        "comparison": "full waveform",
        "python_ms": payload["python_ms"],
        "dart_ms": payload["dart_ms"],
        "max_abs_diff": payload["max_abs_diff"],
        "mean_abs_diff": payload["mean_abs_diff"],
    }


def main() -> None:
    out = ROOT / "benchmark" / "out" / "publish_report.json"
    partial = ROOT / "benchmark" / "out" / "publish_report.partial.json"
    done = {}
    if partial.exists():
        for item in json.loads(partial.read_text(encoding="utf-8")):
            done[item["model_id"]] = item
    report = list(done.values())
    recent_models = json.loads((ROOT / "benchmark" / "recent_unique_models.json").read_text())
    for item in recent_models:
        mid = item["model_id"]
        if mid in done:
            continue
        kind = item["kind"]
        if kind == "text":
            report.append(text_bench(mid))
        elif kind == "vlm":
            report.append(vlm_bench(mid))
        elif kind == "tts":
            report.append(ming_tts_bench(mid))
        partial.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for mid in [
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        "mlx-community/Qwen3.5-35B-A3B-4bit",
        "mlx-community/kitten-tts-nano-0.8-6bit",
    ]:
        if mid in done or any(item["model_id"] == mid for item in report):
            continue
        if mid == "mlx-community/kitten-tts-nano-0.8-6bit":
            report.append(kitten_bench())
        else:
            report.append(text_bench(mid))
        partial.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
