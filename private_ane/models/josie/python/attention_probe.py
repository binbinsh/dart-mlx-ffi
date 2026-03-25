from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "shared" / "benchmark"))

from ane_private_mlprogram_bench import bench_one, build_helper


def make_attention_model(
    out_dir: Path,
    *,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    lane: int,
    seed: int,
) -> Path:
    qdim = num_heads * head_dim
    kdim = num_kv_heads * head_dim
    rng = np.random.default_rng(seed)
    wq = rng.normal(0.0, 0.02, size=(qdim, dim, 1, 1)).astype(np.float32)
    wk = rng.normal(0.0, 0.02, size=(kdim, dim, 1, 1)).astype(np.float32)
    wv = rng.normal(0.0, 0.02, size=(kdim, dim, 1, 1)).astype(np.float32)
    wo = rng.normal(0.0, 0.02, size=(dim, qdim, 1, 1)).astype(np.float32)
    scale = np.array(1.0 / np.sqrt(head_dim), dtype=np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, lane), dtype=types.fp32)])
    def prog(x):
        q = mb.conv(x=x, weight=wq, pad_type="valid", strides=[1, 1], dilations=[1, 1], groups=1)
        k = mb.conv(x=x, weight=wk, pad_type="valid", strides=[1, 1], dilations=[1, 1], groups=1)
        v = mb.conv(x=x, weight=wv, pad_type="valid", strides=[1, 1], dilations=[1, 1], groups=1)
        q = mb.reshape(x=q, shape=[1, num_heads, head_dim, lane])
        q = mb.transpose(x=q, perm=[0, 1, 3, 2])
        k = mb.reshape(x=k, shape=[1, num_kv_heads, head_dim, lane])
        k = mb.transpose(x=k, perm=[0, 1, 3, 2])
        v = mb.reshape(x=v, shape=[1, num_kv_heads, head_dim, lane])
        v = mb.transpose(x=v, perm=[0, 1, 3, 2])
        if num_heads != num_kv_heads:
            reps = mb.const(
                val=np.array([1, num_heads // num_kv_heads, 1, 1], dtype=np.int32)
            )
            k = mb.tile(x=k, reps=reps)
            v = mb.tile(x=v, reps=reps)
        kt = mb.transpose(x=k, perm=[0, 1, 3, 2])
        scores = mb.matmul(x=q, y=kt)
        scores = mb.mul(x=scores, y=mb.const(val=scale))
        probs = mb.softmax(x=scores, axis=-1)
        ctx = mb.matmul(x=probs, y=v)
        ctx = mb.transpose(x=ctx, perm=[0, 1, 3, 2])
        ctx = mb.reshape(x=ctx, shape=[1, qdim, 1, lane])
        out = mb.conv(x=ctx, weight=wo, pad_type="valid", strides=[1, 1], dilations=[1, 1], groups=1)
        return mb.add(x=x, y=out)

    model = ct.convert(prog, convert_to="mlprogram")
    pkg = out_dir / f"attn_{dim}_{num_heads}_{num_kv_heads}_{head_dim}_{lane}.mlpackage"
    model.save(str(pkg))
    return Path(ct.models.utils.compile_model(str(pkg)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    cases = [
        {
            "name": "tiny-attn-512-l8",
            "dim": 512,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 64,
            "lane": 8,
            "seed": 7,
        },
        {
            "name": "josie-attn-2560-l8",
            "dim": 2560,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "lane": 8,
            "seed": 11,
        },
    ]

    work_dir = Path(tempfile.mkdtemp(prefix="josie_attn_probe_"))
    try:
        helper = build_helper(work_dir)
        report = {"runtime": "josie_attention_probe", "models": []}
        for case in cases:
            compiled = make_attention_model(
                work_dir,
                dim=case["dim"],
                num_heads=case["num_heads"],
                num_kv_heads=case["num_kv_heads"],
                head_dim=case["head_dim"],
                lane=case["lane"],
                seed=case["seed"],
            )
            result = bench_one(
                helper,
                compiled,
                channels=case["dim"],
                spatial=case["lane"],
                iters=args.iters,
            )
            report["models"].append({**case, **result})

        if args.json:
            print(json.dumps(report))
            return

        print("josie attention probe")
        for model in report["models"]:
            if model.get("ok"):
                print(f"{model['name']}: {model['per_iter_ms']:.4f} ms/iter")
            else:
                print(f"{model['name']}: failed at {model.get('stage')} | {model.get('error','')}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
