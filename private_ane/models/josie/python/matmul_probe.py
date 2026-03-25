from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from ane_ops import build_server, close_server, run_server


NUM_HEADS = 32
HEAD_DIM = 128


def make_direct_mil(key_len: int, *, first: str, second: str) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {NUM_HEADS}, 1, {HEAD_DIM}]> {first},
        tensor<fp16, [1, {NUM_HEADS}, {key_len}, {HEAD_DIM}]> {second}
    ) {{
        bool mm_tx = const()[name = string("mm_tx"), val = bool(false)];
        bool mm_ty = const()[name = string("mm_ty"), val = bool(true)];
        tensor<fp16, [1, {NUM_HEADS}, 1, {key_len}]> out =
            matmul(transpose_x = mm_tx, transpose_y = mm_ty, x = {first}, y = {second})[name = string("mm")];
    }} -> (out);
}}
'''


def make_native_mil(key_len: int, *, first: str, second: str) -> str:
    model_dim = NUM_HEADS * HEAD_DIM
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {model_dim}, 1, 1]> {first},
        tensor<fp16, [1, {model_dim}, 1, {key_len}]> {second}
    ) {{
        tensor<int32, [4]> q_rsh = const()[name = string("q_rsh"), val = tensor<int32, [4]>([1,{NUM_HEADS},{HEAD_DIM},1])];
        tensor<int32, [4]> k_rsh = const()[name = string("k_rsh"), val = tensor<int32, [4]>([1,{NUM_HEADS},{HEAD_DIM},{key_len}])];
        tensor<int32, [4]> pm = const()[name = string("pm"), val = tensor<int32, [4]>([0,1,3,2])];
        tensor<fp16, [1, {NUM_HEADS}, {HEAD_DIM}, 1]> qr =
            reshape(shape = q_rsh, x = {first})[name = string("qr")];
        tensor<fp16, [1, {NUM_HEADS}, 1, {HEAD_DIM}]> q =
            transpose(perm = pm, x = qr)[name = string("q")];
        tensor<fp16, [1, {NUM_HEADS}, {HEAD_DIM}, {key_len}]> kr =
            reshape(shape = k_rsh, x = {second})[name = string("kr")];
        tensor<fp16, [1, {NUM_HEADS}, {key_len}, {HEAD_DIM}]> k =
            transpose(perm = pm, x = kr)[name = string("k")];
        bool mm_tx = const()[name = string("mm_tx"), val = bool(false)];
        bool mm_ty = const()[name = string("mm_ty"), val = bool(true)];
        tensor<fp16, [1, {NUM_HEADS}, 1, {key_len}]> out =
            matmul(transpose_x = mm_tx, transpose_y = mm_ty, x = q, y = k)[name = string("mm")];
    }} -> (out);
}}
'''


def make_inputs(key_len: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7 + key_len)
    q = rng.normal(0.0, 0.1, size=(1, NUM_HEADS, 1, HEAD_DIM)).astype(np.float16)
    k = rng.normal(0.0, 0.1, size=(1, NUM_HEADS, key_len, HEAD_DIM)).astype(np.float16)
    return q, k


def pack_native_query(q: np.ndarray) -> np.ndarray:
    return np.transpose(q, (0, 1, 3, 2)).reshape(1, NUM_HEADS * HEAD_DIM, 1, 1)


def pack_native_key(k: np.ndarray, key_len: int) -> np.ndarray:
    return np.transpose(k, (0, 1, 3, 2)).reshape(1, NUM_HEADS * HEAD_DIM, 1, key_len)


def expected_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    out = np.matmul(q.astype(np.float32), np.swapaxes(k.astype(np.float32), -1, -2))
    return out.astype(np.float32)


def run_case(key_len: int, *, layout: str, names: tuple[str, str], order: str) -> dict[str, object]:
    q, k = make_inputs(key_len)
    if layout == "direct":
        mil_text = make_direct_mil(key_len, first=names[0], second=names[1])
        q_blob = q.tobytes()
        k_blob = k.tobytes()
        q_bytes = q.nbytes
        k_bytes = k.nbytes
    else:
        mil_text = make_native_mil(key_len, first=names[0], second=names[1])
        q_native = pack_native_query(q)
        k_native = pack_native_key(k, key_len)
        q_blob = q_native.tobytes()
        k_blob = k_native.tobytes()
        q_bytes = q_native.nbytes
        k_bytes = k_native.nbytes

    if order == "declared":
        input_bytes = [q_bytes, k_bytes]
        blobs = [q_blob, k_blob]
    else:
        input_bytes = [k_bytes, q_bytes]
        blobs = [k_blob, q_blob]

    work_dir = None
    proc = None
    try:
        work_dir, proc = build_server(
            mil_text,
            input_bytes,
            NUM_HEADS * key_len * 2,
            prefix=f"josie_mm_{layout}_{key_len}_",
        )
        out_blob = run_server(proc, blobs, NUM_HEADS * key_len * 2)
        out = np.frombuffer(out_blob, dtype=np.float16).astype(np.float32)
        out = out.reshape(1, NUM_HEADS, 1, key_len)
        ref = expected_scores(q, k)
        diffs = np.abs(out - ref)
        return {
            "ok": True,
            "layout": layout,
            "key_len": key_len,
            "names": list(names),
            "order": order,
            "max_abs_diff": float(np.max(diffs)),
            "mean_abs_diff": float(np.mean(diffs)),
        }
    except Exception as error:
        return {
            "ok": False,
            "layout": layout,
            "key_len": key_len,
            "names": list(names),
            "order": order,
            "error": str(error),
        }
    finally:
        if work_dir is not None and proc is not None:
            close_server(work_dir, proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = []
    for key_len in (1, 4):
        for layout in ("direct", "native"):
            for names in (("a", "b"), ("qf", "kf")):
                orders = ("declared",) if names == ("a", "b") else ("declared", "alphabetical")
                for order in orders:
                    cases.append(run_case(key_len, layout=layout, names=names, order=order))

    report = {"runtime": "josie_private_ane_matmul_probe", "cases": cases}
    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
