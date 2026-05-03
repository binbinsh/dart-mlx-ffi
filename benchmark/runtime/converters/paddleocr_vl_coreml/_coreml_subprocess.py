"""Subprocess-isolated CoreML predict helpers for Phase A parity.

Background
----------
Calling ``coremltools.MLModel.predict()`` in the same process as PyTorch
corrupts shared memory in ways that cause SIGSEGV inside unrelated torch
kernels (Conv2d, silu, split) on the *next* torch op. We've reproduced this
across Conv1d, Conv2d, silu, and torch.split. It is not specific to a single
operator; it looks like an ABI / allocator interaction between coremltools 9.0
and torch 2.11 + numpy 2.4 on macOS aarch64.

The mitigation is process isolation: every CoreML predict runs in a fresh
``spawn``-based child process. Numpy outputs are pickled back to the parent.

Two flavors:

* :func:`predict_isolated` – one predict call, optional state.
* :func:`predict_isolated_chain` – sequence of (mlpackage, inputs) calls
  sharing one stateful model instance. Required for stateful prefill →
  decode loops because CoreML state objects are opaque C++ pointers and
  cannot be pickled across processes.

Typing is loose (numpy arrays in / out) because that's what
``MLModel.predict`` accepts and returns.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle  # noqa: F401 — implicit via mp.Pipe
import traceback
from typing import Any

import numpy as np


def _ensure_numpy(d: dict[str, Any]) -> dict[str, np.ndarray]:
    """Coerce inputs to contiguous numpy arrays.

    Defensive copy: avoids any chance of a torch-owned buffer being aliased
    into the child via shared memory (which is the suspected segfault source).
    """
    out = {}
    for k, v in d.items():
        a = np.asarray(v)
        out[k] = np.ascontiguousarray(a).copy()
    return out


def _single_worker(
    mlpackage_path: str,
    inputs_dict: dict[str, np.ndarray],
    stateful: bool,
    conn: Any,
) -> None:
    try:
        import coremltools as ct

        m = ct.models.MLModel(
            mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY
        )
        if stateful:
            state = m.make_state()
            out = m.predict(inputs_dict, state=state)
        else:
            out = m.predict(inputs_dict)
        out_np = {k: np.asarray(v).copy() for k, v in out.items()}
        conn.send(("ok", out_np))
    except Exception as e:  # noqa: BLE001
        conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def predict_isolated(
    mlpackage_path: str,
    inputs_dict: dict[str, np.ndarray],
    *,
    stateful: bool = False,
    timeout_s: float = 180.0,
) -> dict[str, np.ndarray]:
    """Run one CoreML predict in a child process; return numpy outputs."""
    inputs_dict = _ensure_numpy(inputs_dict)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(
        target=_single_worker,
        args=(str(mlpackage_path), inputs_dict, bool(stateful), child_conn),
    )
    p.start()
    try:
        if not parent_conn.poll(timeout_s):
            p.terminate()
            p.join(5)
            raise RuntimeError(
                f"predict_isolated timeout ({timeout_s}s) for {mlpackage_path}"
            )
        status, payload = parent_conn.recv()
    finally:
        p.join(timeout_s)
        if p.is_alive():
            p.terminate()
            p.join(5)
        parent_conn.close()
    if p.exitcode not in (0, None):
        # The child sent us a payload before exiting; that's the real error.
        # But if status was 'ok' yet exitcode!=0, something went wrong AFTER
        # send (not our problem — we already have the data).
        if status == "err":
            raise RuntimeError(
                f"predict_isolated child exited {p.exitcode}: {payload}"
            )
    if status == "err":
        raise RuntimeError(f"predict_isolated failed: {payload}")
    return payload


# --------------------------------------------------------------------------- #
# Stateful chain — one process, one state, many predicts.
# --------------------------------------------------------------------------- #
def _chain_worker(
    plan: list[tuple[str, str, dict[str, np.ndarray], bool]],
    conn: Any,
) -> None:
    """Plan entry: (op, mlpackage_path, inputs_dict, capture_output).

    op is one of:
        "load_stateful"   – load the mlpackage, m.make_state(); inputs ignored
        "load_stateless"  – load the mlpackage; inputs ignored
        "predict"         – run predict on the most recently loaded model

    capture_output: if True, append outputs to the result list. If False,
    skip pickling (used to keep replay-warmup memory bounded).

    Models are kept in a dict keyed by mlpackage_path so the loop can
    alternate between (e.g.) prefill_decoder and decode_decoder while sharing
    the SAME state object — which is required because the prefill mlpackage
    and decode mlpackage own separate state buffers in the current pipeline.
    """
    try:
        import coremltools as ct

        loaded: dict[str, Any] = {}
        states: dict[str, Any] = {}
        last_path: str | None = None
        results: list[dict[str, np.ndarray] | None] = []

        for op, path, inputs, capture in plan:
            if op == "load_stateful":
                if path not in loaded:
                    loaded[path] = ct.models.MLModel(
                        path, compute_units=ct.ComputeUnit.CPU_ONLY
                    )
                    states[path] = loaded[path].make_state()
                last_path = path
            elif op == "load_stateless":
                if path not in loaded:
                    loaded[path] = ct.models.MLModel(
                        path, compute_units=ct.ComputeUnit.CPU_ONLY
                    )
                last_path = path
            elif op == "predict":
                if path != last_path and path not in loaded:
                    raise RuntimeError(
                        f"chain: predict on unloaded model {path}"
                    )
                m = loaded[path]
                state = states.get(path)
                if state is not None:
                    out = m.predict(inputs, state=state)
                else:
                    out = m.predict(inputs)
                if capture:
                    results.append({k: np.asarray(v).copy() for k, v in out.items()})
                else:
                    results.append(None)
            else:
                raise ValueError(f"chain: unknown op {op!r}")

        conn.send(("ok", results))
    except Exception as e:  # noqa: BLE001
        conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def predict_isolated_chain(
    plan: list[tuple[str, str, dict[str, np.ndarray] | None, bool]],
    *,
    timeout_s: float = 1800.0,
) -> list[dict[str, np.ndarray] | None]:
    """Execute a stateful chain of CoreML predicts in a single child process.

    plan entries:
        ("load_stateful", path, None, False)
        ("load_stateless", path, None, False)
        ("predict", path, inputs, capture_output_bool)

    Returns a list aligned with the *predict* entries (load entries return
    no element). Captured entries are dicts of numpy arrays; uncaptured
    entries are None.
    """
    sanitized: list[tuple[str, str, dict[str, np.ndarray], bool]] = []
    for op, path, inputs, capture in plan:
        if op == "predict":
            if inputs is None:
                raise ValueError("chain predict entry needs inputs dict")
            sanitized.append((op, str(path), _ensure_numpy(inputs), bool(capture)))
        else:
            sanitized.append((op, str(path), {}, False))

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=_chain_worker, args=(sanitized, child_conn))
    p.start()
    try:
        if not parent_conn.poll(timeout_s):
            p.terminate()
            p.join(5)
            raise RuntimeError(
                f"predict_isolated_chain timeout ({timeout_s}s)"
            )
        status, payload = parent_conn.recv()
    finally:
        p.join(timeout_s)
        if p.is_alive():
            p.terminate()
            p.join(5)
        parent_conn.close()
    if status == "err":
        raise RuntimeError(f"predict_isolated_chain failed: {payload}")

    # Filter to only "predict" entries to align with caller expectations.
    out: list[dict[str, np.ndarray] | None] = []
    pi = 0
    for op, *_ in plan:
        if op == "predict":
            out.append(payload[pi])
            pi += 1
    return out


# --------------------------------------------------------------------------- #
# Autoregressive decode loop — isolated (Bug X fix for fp32 decode).
# --------------------------------------------------------------------------- #
# The chain helper above can't host a feedback loop (each step's output
# selects the next step's input). For the production e2e path we need to
# run the entire greedy decode loop inside one child process so the
# coremltools-9 / torch-2.11 ABI segfault never reaches the parent.
#
# The child:
#   1. Loads decode mlpackage, makes a fresh state.
#   2. Restores KV state from buffers handed in by the parent (parent
#      already ran prefill in-process — fp16 prefill is segfault-safe;
#      it's the fp32 decode predict followed by torch ops that bites).
#   3. Runs the greedy loop, computing per-step rope via torch INSIDE
#      the child (so the post-predict torch op never touches the parent).
#   4. Returns generated token ids + per-step latencies.

def _decode_loop_worker(
    decode_path: str,
    kv_npz_path: str,                 # mmap'd from disk to avoid pickling KV state
    first_token: int,
    real_len: int,
    rope_deltas: int,
    embed_tokens_npy_path: str,       # mmap'd from disk to avoid pipe-pickling 200MB
    inv_freq_np: np.ndarray,          # (head_dim/2,) fp32 — small, pass inline
    head_dim: int,
    mrope_section: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    hidden_size: int,
    conn: Any,
) -> None:
    # Open a debug log inside the child so its trace survives even if the
    # parent dies and its stdout/stderr fd is closed before we can flush.
    import os as _os
    _dbg_path = f"/tmp/decode_worker_{_os.getpid()}.log"
    _f = open(_dbg_path, "w", buffering=1)
    def _wlog(msg: str) -> None:
        _f.write(f"[child {_os.getpid()}] {msg}\n")
        _f.flush()
    _wlog(f"entered worker; dbg log = {_dbg_path}")
    try:
        import time as _time
        _wlog("importing coremltools")
        import coremltools as ct
        _wlog("importing torch")
        import torch  # noqa: F401  — used by hf_native_step_rope

        # Re-import inside child so we don't pickle module objects.
        _wlog("importing parity")
        from benchmark.runtime.converters.paddleocr_vl_coreml.parity import (
            hf_native_step_rope,
        )
        _wlog("imports done")

        # mmap the embed table — vocab × hidden × 2B fp16 ≈ 200 MB; pickling
        # this through mp.Pipe truncates the connection on macOS.
        _wlog(f"mmap embed from {embed_tokens_npy_path}")
        embed_tokens_weight = np.load(embed_tokens_npy_path, mmap_mode="r")
        _wlog(f"embed mmap'd shape={embed_tokens_weight.shape}")

        # Load KV buffers — also via disk to avoid bloating the spawn pickle.
        _wlog(f"loading kv from {kv_npz_path}")
        with np.load(kv_npz_path) as kv_loaded:
            kv_buffers = {k: kv_loaded[k].copy() for k in kv_loaded.files}
        _wlog(f"kv loaded {len(kv_buffers)} buffers")

        _wlog(f"loading decode mlpackage {decode_path}")
        m = ct.models.MLModel(decode_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        _wlog("decode loaded; making state")
        state = m.make_state()
        _wlog("state made; writing kv buffers")
        for name, buf in kv_buffers.items():
            arr = np.asarray(state.read_state(name), copy=False)
            arr[...] = buf
        _wlog("kv buffers written")

        inv_freq = torch.from_numpy(inv_freq_np).to(torch.float32)

        generated: list[int] = [int(first_token)]
        step_latencies_ms: list[float] = []
        cur_token = int(first_token)
        for step in range(max_new_tokens - 1):
            if cur_token == eos_token_id:
                break
            tok_row = np.asarray(
                embed_tokens_weight[cur_token : cur_token + 1]
            ).astype(np.float16, copy=False)
            tok_embed_np = tok_row.reshape(1, 1, hidden_size)
            cache_pos = real_len + step
            rope_cos_s, rope_sin_s = hf_native_step_rope(
                cache_pos=cache_pos,
                rope_deltas=rope_deltas,
                inv_freq=inv_freq,
                head_dim=head_dim,
                mrope_section=mrope_section,
            )
            t0 = _time.time()
            d_out = m.predict(
                {
                    "inputs_embeds": tok_embed_np,
                    "rope_cos": rope_cos_s,
                    "rope_sin": rope_sin_s,
                    "cur_len": np.array([cache_pos], dtype=np.int32),
                    "kv_len": np.array([cache_pos + 1], dtype=np.int32),
                },
                state=state,
            )
            step_latencies_ms.append((_time.time() - t0) * 1000.0)
            step_logits = (
                np.asarray(d_out["logits"]).astype(np.float32).reshape(-1)
            )
            nxt = int(np.argmax(step_logits))
            generated.append(nxt)
            cur_token = nxt
            if nxt == eos_token_id:
                break

        conn.send(("ok", {
            "generated": generated,
            "step_latencies_ms": step_latencies_ms,
        }))
    except Exception as e:  # noqa: BLE001
        conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def run_decode_loop_isolated(
    *,
    decode_path: str,
    kv_buffers: dict[str, np.ndarray],
    first_token: int,
    real_len: int,
    rope_deltas: int,
    embed_tokens_weight: np.ndarray,
    inv_freq: np.ndarray,
    head_dim: int,
    mrope_section: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    hidden_size: int,
    timeout_s: float = 1800.0,
) -> tuple[list[int], list[float]]:
    """Run greedy autoregressive decode in a spawned child process.

    KV buffers + inv_freq are passed inline (small). The embed table is
    written to a temporary ``.npy`` file and mmap'd by the child to avoid
    truncating ``mp.Pipe`` with a 400 MB pickle (macOS limit).
    Returns ``(generated_token_ids, per_step_latencies_ms)``.
    """
    import os as _os
    import sys as _sys
    import tempfile

    def _dbg(msg: str) -> None:
        print(f"[isolated-decode] {msg}", flush=True, file=_sys.stderr)

    _dbg("entering")
    # Persist KV buffers to a tmp .npz — avoids pickling them through the
    # spawn pipe (which has crashed the parent on macOS for sizes <10 MB
    # combined; suspect ABI corruption from coremltools-9 + spawn).
    kv_clean = {
        str(k): np.ascontiguousarray(np.asarray(v))
        for k, v in kv_buffers.items()
    }
    fd_kv, kv_npz = tempfile.mkstemp(prefix="kv_buffers_", suffix=".npz")
    _os.close(fd_kv)
    np.savez(kv_npz, **kv_clean)
    _dbg(f"kv saved to {kv_npz} ({len(kv_clean)} bufs)")

    inv_clean = np.ascontiguousarray(np.asarray(inv_freq)).copy()

    # Persist embed table to a tmpfile for mmap in child.
    fd, embed_npy = tempfile.mkstemp(prefix="embed_tokens_", suffix=".npy")
    _os.close(fd)
    embed_arr = np.ascontiguousarray(np.asarray(embed_tokens_weight))
    _dbg(f"saving embed {embed_arr.shape} {embed_arr.dtype} to {embed_npy}")
    np.save(embed_npy, embed_arr, allow_pickle=False)
    _dbg("embed saved")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    _dbg("pipe + ctx ready, spawning")
    p = ctx.Process(
        target=_decode_loop_worker,
        args=(
            str(decode_path),
            kv_npz,
            int(first_token),
            int(real_len),
            int(rope_deltas),
            embed_npy,
            inv_clean,
            int(head_dim),
            list(mrope_section),
            int(max_new_tokens),
            int(eos_token_id),
            int(hidden_size),
            child_conn,
        ),
    )
    p.start()
    _dbg("spawn started, polling")
    try:
        if not parent_conn.poll(timeout_s):
            p.terminate()
            p.join(5)
            raise RuntimeError(
                f"run_decode_loop_isolated timeout ({timeout_s}s)"
            )
        status, payload = parent_conn.recv()
    finally:
        p.join(timeout_s)
        if p.is_alive():
            p.terminate()
            p.join(5)
        parent_conn.close()
        for _path in (embed_npy, kv_npz):
            try:
                _os.unlink(_path)
            except OSError:
                pass
    if status == "err":
        raise RuntimeError(f"run_decode_loop_isolated failed: {payload}")
    return payload["generated"], payload["step_latencies_ms"]


# --------------------------------------------------------------------------- #
# Combined prefill + decode — Bug X parent-protection (Option C).
# --------------------------------------------------------------------------- #
# The parent process (e2e_token_golden HF-NATIVE path) cannot call any
# MLModel.predict itself, because coremltools-9 + torch-2.11 ABI corruption
# segfaults the parent during the next torch op. The previous mitigation —
# only isolating the decode loop — still required parent to run vision and
# prefill in-process, which trips the bug.
#
# This worker accepts a fully-prepared prefill input dict (parent built it
# torch-side using HF model + harvested image features) and runs:
#   1. prefill_decoder.mlpackage.predict(...) → first_token + KV state
#   2. decode_decoder.mlpackage.predict(...) loop → generated tokens
# All MLModel.predict calls live in this child; parent never touches CoreML.
def _prefill_plus_decode_worker(
    prefill_path: str,
    decode_path: str,
    prefill_inputs_npz_path: str,
    real_len: int,
    rope_deltas: int,
    embed_tokens_npy_path: str,
    inv_freq_np: np.ndarray,
    head_dim: int,
    mrope_section: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    hidden_size: int,
    conn: Any,
) -> None:
    import os as _os
    _dbg_path = f"/tmp/pp_decode_worker_{_os.getpid()}.log"
    _f = open(_dbg_path, "w", buffering=1)
    def _wlog(msg: str) -> None:
        _f.write(f"[child {_os.getpid()}] {msg}\n")
        _f.flush()
    _wlog(f"entered worker; dbg log = {_dbg_path}")
    try:
        import time as _time
        _wlog("importing coremltools")
        import coremltools as ct
        _wlog("importing torch")
        import torch  # noqa: F401  — used by hf_native_step_rope

        _wlog("importing parity")
        from benchmark.runtime.converters.paddleocr_vl_coreml.parity import (
            hf_native_step_rope,
        )
        _wlog("imports done")

        _wlog(f"mmap embed from {embed_tokens_npy_path}")
        embed_tokens_weight = np.load(embed_tokens_npy_path, mmap_mode="r")
        _wlog(f"embed mmap'd shape={embed_tokens_weight.shape}")

        _wlog(f"loading prefill inputs from {prefill_inputs_npz_path}")
        with np.load(prefill_inputs_npz_path) as p_loaded:
            prefill_inputs = {k: p_loaded[k].copy() for k in p_loaded.files}
        _wlog(f"prefill inputs loaded: {list(prefill_inputs.keys())}")

        # ---- Prefill --------------------------------------------------- #
        _wlog(f"loading prefill mlpackage {prefill_path}")
        p_ml = ct.models.MLModel(prefill_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        _wlog("prefill loaded; making state")
        p_state = p_ml.make_state()
        _wlog("running prefill.predict")
        t_prefill = _time.time()
        p_out = p_ml.predict(prefill_inputs, state=p_state)
        prefill_ms = (_time.time() - t_prefill) * 1000.0
        first_logits = (
            np.asarray(p_out["logits"]).astype(np.float32).reshape(-1)
        )
        first_token = int(np.argmax(first_logits))
        _wlog(f"prefill done in {prefill_ms:.0f}ms first_token={first_token}")

        # ---- Bridge KV state from prefill → decode --------------------- #
        state_names = [s.name for s in p_ml.get_spec().description.state]
        _wlog(f"reading {len(state_names)} KV buffers from prefill state")
        kv_buffers = {
            name: np.asarray(p_state.read_state(name)).copy()
            for name in state_names
        }
        # Drop prefill model + state before loading decode (release Espresso
        # caches; conserve RAM).
        p_state = None
        p_ml = None
        import gc as _gc
        _gc.collect()

        _wlog(f"loading decode mlpackage {decode_path}")
        m = ct.models.MLModel(decode_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        _wlog("decode loaded; making state")
        d_state = m.make_state()
        _wlog("writing kv buffers into decode state")
        for name, buf in kv_buffers.items():
            d_state.write_state(name, buf)
        kv_buffers = None
        _gc.collect()
        _wlog("kv buffers written; entering decode loop")

        inv_freq = torch.from_numpy(inv_freq_np).to(torch.float32)

        generated: list[int] = [int(first_token)]
        step_latencies_ms: list[float] = []
        cur_token = int(first_token)
        for step in range(max_new_tokens - 1):
            if cur_token == eos_token_id:
                break
            tok_row = np.asarray(
                embed_tokens_weight[cur_token : cur_token + 1]
            ).astype(np.float16, copy=False)
            tok_embed_np = tok_row.reshape(1, 1, hidden_size)
            cache_pos = real_len + step
            rope_cos_s, rope_sin_s = hf_native_step_rope(
                cache_pos=cache_pos,
                rope_deltas=rope_deltas,
                inv_freq=inv_freq,
                head_dim=head_dim,
                mrope_section=mrope_section,
            )
            t0 = _time.time()
            d_out = m.predict(
                {
                    "inputs_embeds": tok_embed_np,
                    "rope_cos": rope_cos_s,
                    "rope_sin": rope_sin_s,
                    "cur_len": np.array([cache_pos], dtype=np.int32),
                    "kv_len":  np.array([cache_pos + 1], dtype=np.int32),
                },
                state=d_state,
            )
            step_latencies_ms.append((_time.time() - t0) * 1000.0)
            step_logits = (
                np.asarray(d_out["logits"]).astype(np.float32).reshape(-1)
            )
            nxt = int(np.argmax(step_logits))
            if step < 2:
                _top = np.argsort(step_logits)[-5:][::-1]
                _wlog(
                    f"  step {step} logits sum={float(step_logits.sum()):.6f} "
                    f"top5={[(int(i), float(step_logits[i])) for i in _top]}"
                )
            generated.append(nxt)
            cur_token = nxt
            if nxt == eos_token_id:
                break

        _wlog(f"decode loop done; {len(generated)} tokens generated")
        conn.send(("ok", {
            "first_token": first_token,
            "prefill_ms": prefill_ms,
            "generated": generated,
            "step_latencies_ms": step_latencies_ms,
        }))
    except Exception as e:  # noqa: BLE001
        _wlog(f"EXCEPTION: {type(e).__name__}: {e}")
        conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        try:
            _f.close()
        except Exception:
            pass
        conn.close()


def run_prefill_plus_decode_isolated(
    *,
    prefill_path: str,
    decode_path: str,
    prefill_inputs: dict[str, np.ndarray],
    real_len: int,
    rope_deltas: int,
    embed_tokens_weight: np.ndarray,
    inv_freq: np.ndarray,
    head_dim: int,
    mrope_section: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    hidden_size: int,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Run prefill + decode loop entirely inside one spawned child process.

    Parent must NOT have called MLModel.predict in this process before
    invoking this helper (Bug X: predict-then-torch corrupts allocator).

    Parent prepares ``prefill_inputs`` torch-side and provides the embed
    table + inv_freq + mRoPE config. Returns a dict with ``first_token``,
    ``prefill_ms``, ``generated`` (list[int] including first_token),
    ``step_latencies_ms``.
    """
    import os as _os
    import sys as _sys
    import tempfile

    def _dbg(msg: str) -> None:
        print(f"[isolated-pp-decode] {msg}", flush=True, file=_sys.stderr)

    _dbg("entering")

    # Persist prefill inputs to .npz (avoid pipe pickle truncation on macOS).
    p_clean = {
        str(k): np.ascontiguousarray(np.asarray(v))
        for k, v in prefill_inputs.items()
    }
    fd_p, p_npz = tempfile.mkstemp(prefix="prefill_inputs_", suffix=".npz")
    _os.close(fd_p)
    np.savez(p_npz, **p_clean)
    _dbg(f"prefill inputs saved to {p_npz}")

    inv_clean = np.ascontiguousarray(np.asarray(inv_freq)).copy()

    fd, embed_npy = tempfile.mkstemp(prefix="embed_tokens_", suffix=".npy")
    _os.close(fd)
    embed_arr = np.ascontiguousarray(np.asarray(embed_tokens_weight))
    _dbg(f"saving embed {embed_arr.shape} {embed_arr.dtype} to {embed_npy}")
    np.save(embed_npy, embed_arr, allow_pickle=False)
    _dbg("embed saved")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    _dbg("pipe + ctx ready, spawning")
    p = ctx.Process(
        target=_prefill_plus_decode_worker,
        args=(
            str(prefill_path),
            str(decode_path),
            p_npz,
            int(real_len),
            int(rope_deltas),
            embed_npy,
            inv_clean,
            int(head_dim),
            list(mrope_section),
            int(max_new_tokens),
            int(eos_token_id),
            int(hidden_size),
            child_conn,
        ),
    )
    p.start()
    _dbg(f"spawn started (pid={p.pid}), polling")
    try:
        if not parent_conn.poll(timeout_s):
            p.terminate()
            p.join(5)
            raise RuntimeError(
                f"run_prefill_plus_decode_isolated timeout ({timeout_s}s)"
            )
        status, payload = parent_conn.recv()
    finally:
        p.join(timeout_s)
        if p.is_alive():
            p.terminate()
            p.join(5)
        parent_conn.close()
        for _path in (embed_npy, p_npz):
            try:
                _os.unlink(_path)
            except OSError:
                pass
    if status == "err":
        raise RuntimeError(f"run_prefill_plus_decode_isolated failed: {payload}")
    return payload
