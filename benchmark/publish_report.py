from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .common import (
        add_vendor_to_path,
        benchmark_dart_export,
        cleanup_mlx,
        compare_lists,
        resolve_model_path,
        slug,
    )
    from .parakeet_tdt_sweep import asr_bench
    from .vlm_export_sweep import export_model as export_vlm_model
    from .vlm_export_sweep import extract_logits as vlm_extract_logits
    from .vlm_export_sweep import prepare_model_inputs as prepare_vlm_inputs
    from .tts_export_sweep import export_model as export_ming_tts_model
    from .tts_export_sweep import python_forward as ming_python_forward
    from .tts_export_sweep import dart_forward as ming_dart_forward
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (
        add_vendor_to_path,
        benchmark_dart_export,
        cleanup_mlx,
        compare_lists,
        resolve_model_path,
        slug,
    )
    from parakeet_tdt_sweep import asr_bench
    from vlm_export_sweep import export_model as export_vlm_model
    from vlm_export_sweep import extract_logits as vlm_extract_logits
    from vlm_export_sweep import prepare_model_inputs as prepare_vlm_inputs
    from tts_export_sweep import export_model as export_ming_tts_model
    from tts_export_sweep import python_forward as ming_python_forward
    from tts_export_sweep import dart_forward as ming_dart_forward

import mlx.core as mx
from mlx_lm import load as load_lm

ROOT = Path(__file__).resolve().parents[1]
TEXT_PROMPT = "Explain why MLX is useful for local inference on Apple Silicon."
VLM_PROMPT = "Describe this image briefly."
UNSLOTH_GEMMA4_VENV = (
    ROOT / "output" / "gemma-4-e2b-it" / "unsloth-mlx-venv"
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
    del model, tokenizer, tokens, last
    del model2, _tokenizer2, tokens2
    cleanup_mlx(mx)
    dart_values, dart_ms = benchmark_dart_export(
        export_path=export_path,
        input_path=input_path,
        mx_module=mx,
        warmup=warmup,
        iters=iters,
    )
    max_diff, mean_diff = compare_lists(py_values, dart_values)
    cleanup_mlx(mx)
    return {
        "model_id": model_id,
        "kind": "text",
        "input_desc": f"{len(token_ids)} text tokens",
        "comparison": "last-token logits[:16]",
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


def unsloth_mlx_text_bench(
    model_id: str,
    *,
    warmup: int = 3,
    iters: int = 10,
) -> dict[str, object]:
    python_bin = UNSLOTH_GEMMA4_VENV / "bin" / "python"
    if not python_bin.exists():
        raise RuntimeError(
            "Missing Unsloth Gemma4 MLX environment. "
            f"Expected: {python_bin}"
        )

    export_dir = ROOT / "benchmark" / "out" / "publish" / slug(model_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    report_path = export_dir / "python_report.json"
    script = """
from pathlib import Path
import json
import time

import mlx.core as mx
from mlx_lm import load

MODEL_ID = {model_id!r}
PROMPT = {prompt!r}
EXPORT_DIR = Path({export_dir!r})


def extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


model, tokenizer = load(MODEL_ID, lazy=False)
token_ids = tokenizer.encode(PROMPT)[:24]
tokens = mx.array([token_ids], dtype=mx.int32)

for _ in range({warmup}):
    logits = extract_logits(model(tokens))[:, -1, :16].astype(mx.float32)
    mx.eval(logits)
    mx.synchronize()

started = time.perf_counter()
last = None
for _ in range({iters}):
    last = extract_logits(model(tokens))[:, -1, :16].astype(mx.float32)
    mx.eval(last)
    mx.synchronize()
py_ms = (time.perf_counter() - started) * 1000.0 / {iters}
py_values = [float(v) for v in last.reshape([-1]).tolist()]

export_path = EXPORT_DIR / "function.mlxfn"
input_path = EXPORT_DIR / "inputs.safetensors"
if export_path.exists():
    export_path.unlink()
if input_path.exists():
    input_path.unlink()


def forward(input_ids):
    output = extract_logits(model(input_ids))
    return output[:, -1, :16].astype(mx.float32)


mx.export_function(str(export_path), forward, tokens)
mx.save_safetensors(str(input_path), {{"input_ids": tokens}})

payload = {{
    "model_id": MODEL_ID,
    "token_count": len(token_ids),
    "python_ms": py_ms,
    "python_values": py_values,
    "export_path": str(export_path),
    "input_path": str(input_path),
}}
Path({report_path!r}).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload))
""".format(
        model_id=model_id,
        prompt=TEXT_PROMPT,
        export_dir=str(export_dir),
        report_path=str(report_path),
        warmup=warmup,
        iters=iters,
    )
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "1"
    completed = subprocess.run(
        [str(python_bin), "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    py_values = [float(v) for v in payload["python_values"]]
    dart_values, dart_ms = benchmark_dart_export(
        export_path=Path(payload["export_path"]),
        input_path=Path(payload["input_path"]),
        mx_module=mx,
        warmup=warmup,
        iters=iters,
    )
    max_diff, mean_diff = compare_lists(py_values, dart_values)
    cleanup_mlx(mx)
    return {
        "model_id": model_id,
        "kind": "text",
        "input_desc": f"{payload['token_count']} text tokens",
        "comparison": "last-token logits[:16]",
        "python_ms": float(payload["python_ms"]),
        "dart_ms": dart_ms,
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
    del model, processor, inputs, call_inputs, input_ids, attention_mask, pixel_values, kwargs, last
    cleanup_mlx(mx)

    dart_values, dart_ms = benchmark_dart_export(
        export_path=export_path,
        input_path=input_path,
        input_names=input_names,
        mx_module=mx,
        warmup=warmup,
        iters=iters,
    )
    max_diff, mean_diff = compare_lists(py_values, dart_values)
    cleanup_mlx(mx)
    return {
        "model_id": model_id,
        "kind": "vlm",
        "input_desc": "1 synthetic image + text prompt",
        "comparison": "last-token logits[:16]",
        "python_ms": py_ms,
        "dart_ms": dart_ms,
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
    del model, inputs, last
    cleanup_mlx(mx)

    dart_values, dart_ms = ming_dart_forward(
        export_path,
        input_path,
        input_names,
        warmup=warmup,
        iters=iters,
    )
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
    resume = os.environ.get("PUBLISH_RESUME") == "1"
    done = {}
    if resume and partial.exists():
        for item in json.loads(partial.read_text(encoding="utf-8")):
            done[item["model_id"]] = item
    report = list(done.values())
    model_specs = json.loads((ROOT / "benchmark" / "publish_model_list.json").read_text())
    for item in model_specs:
        mid = item["model_id"]
        if mid in done:
            continue
        runner = item.get("runner")
        kind = item["kind"]
        if runner == "kitten":
            report.append(kitten_bench())
        elif runner == "unsloth_mlx":
            report.append(unsloth_mlx_text_bench(mid))
        elif kind == "text":
            report.append(text_bench(mid))
        elif kind == "vlm":
            report.append(vlm_bench(mid))
        elif kind == "tts":
            report.append(ming_tts_bench(mid))
        elif kind == "asr":
            report.append(asr_bench(mid))
        else:
            raise ValueError(f"Unsupported benchmark spec: {item}")
        partial.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
