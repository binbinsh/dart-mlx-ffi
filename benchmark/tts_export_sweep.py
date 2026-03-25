from __future__ import annotations

import json
from pathlib import Path
import sys
import time

try:
    from .common import add_vendor_to_path, benchmark_dart_export, cleanup_mlx, resolve_model_path, slug
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import add_vendor_to_path, benchmark_dart_export, cleanup_mlx, resolve_model_path, slug

add_vendor_to_path("mlx-audio")

import mlx.core as mx
from mlx_audio.tts.utils import get_model_path, load

def _dit_inputs():
    return {
        "x": mx.zeros((1, 4, 64), dtype=mx.float32),
        "t": mx.array([0.5], dtype=mx.float32),
        "c": mx.zeros((1, 1, 896), dtype=mx.float32),
        "latent_history": mx.zeros((1, 32, 64), dtype=mx.float32),
    }


def python_forward(model_id: str):
    model_path = resolve_model_path(model_id, get_model_path)
    model = load(str(model_path), lazy=False)
    inputs = _dit_inputs()

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

    started = time.perf_counter()
    out = forward()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    values = [float(v) for v in out.reshape([-1]).tolist()]
    return model, inputs, values, elapsed_ms


def export_model(model_id: str, export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "function.mlxfn"
    input_path = export_dir / "inputs.safetensors"
    input_names = ["x", "t", "c", "latent_history"]
    input_names_path = export_dir / "input_names.json"
    if export_path.exists() and input_path.exists() and input_names_path.exists():
        cached_names = json.loads(input_names_path.read_text(encoding="utf-8"))
        return export_path, input_path, cached_names

    model_path = resolve_model_path(model_id, get_model_path)
    model = load(str(model_path), lazy=False)
    inputs = _dit_inputs()

    def forward(x, t, c, latent_history):
        return model.flowloss.cfm.model.forward_with_cfg(
            x,
            t,
            c,
            cfg_scale=0.3,
            latent_history=latent_history,
            patch_size=4,
        ).astype(mx.float32)

    if export_path.exists():
        export_path.unlink()
    mx.export_function(
        str(export_path),
        forward,
        *[inputs[name] for name in input_names],
    )
    mx.save_safetensors(str(input_path), {name: inputs[name] for name in input_names})
    input_names_path.write_text(json.dumps(input_names), encoding="utf-8")
    cleanup_mlx(mx)
    return export_path, input_path, input_names


def dart_forward(
    export_path: Path,
    input_path: Path,
    input_names: list[str],
    *,
    warmup: int = 0,
    iters: int = 1,
):
    return benchmark_dart_export(
        export_path=export_path,
        input_path=input_path,
        input_names=input_names,
        mx_module=mx,
        warmup=warmup,
        iters=iters,
    )
