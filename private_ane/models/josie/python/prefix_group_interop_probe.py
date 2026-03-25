from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from full_model_compare import PROMPT, repeat_kv
from manual_causal_heads_compare import make_ctx_prefix_group_mil

HELPER_SRC = r'''
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "ane_interop.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

static double ticks_to_ns(uint64_t t) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)t * tb.numer / tb.denom;
}

static bool read_exact(FILE *stream, void *buffer, size_t len) {
    uint8_t *p = (uint8_t *)buffer;
    size_t got = 0;
    while (got < len) {
        size_t n = fread(p + got, 1, len - got, stream);
        if (n == 0) return false;
        got += n;
    }
    return true;
}

static bool write_exact(FILE *stream, const void *buffer, size_t len) {
    const uint8_t *p = (const uint8_t *)buffer;
    size_t sent = 0;
    while (sent < len) {
        size_t n = fwrite(p + sent, 1, len - sent, stream);
        if (n == 0) return false;
        sent += n;
    }
    fflush(stream);
    return true;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 5) return 2;
    NSString *root = [NSString stringWithUTF8String:argv[1]];
    int seqLen = atoi(argv[2]);
    int headDim = atoi(argv[3]);
    int iters = atoi(argv[4]);
    size_t inputBytes = (size_t)seqLen * (size_t)headDim * 2;
    int outputCount = seqLen - 1;
    size_t outputBytes = (size_t)headDim * 2;

    NSData *mil = [[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"]
                                             encoding:NSUTF8StringEncoding
                                                error:nil] dataUsingEncoding:NSUTF8StringEncoding];
    size_t inSizes[3] = {inputBytes, inputBytes, inputBytes};
    size_t *outSizes = (size_t *)malloc((size_t)outputCount * sizeof(size_t));
    for (int i = 0; i < outputCount; ++i) outSizes[i] = outputBytes;

    ANEHandle *h = ane_interop_compile(
        (const uint8_t *)mil.bytes,
        mil.length,
        NULL,
        NULL,
        NULL,
        0,
        3,
        inSizes,
        outputCount,
        outSizes);
    if (!h) {
      printf("{\"ok\":false,\"stage\":\"compile\",\"code\":%d}\n", ane_interop_last_compile_error());
      free(outSizes);
      return 3;
    }

    IOSurfaceRef in0 = ane_interop_copy_input(h, 0);
    IOSurfaceRef in1 = ane_interop_copy_input(h, 1);
    IOSurfaceRef in2 = ane_interop_copy_input(h, 2);
    void *buf0 = malloc(inputBytes);
    void *buf1 = malloc(inputBytes);
    void *buf2 = malloc(inputBytes);

    while (read_exact(stdin, buf0, inputBytes) &&
           read_exact(stdin, buf1, inputBytes) &&
           read_exact(stdin, buf2, inputBytes)) {
      memcpy(IOSurfaceGetBaseAddress(in0), buf0, inputBytes);
      memcpy(IOSurfaceGetBaseAddress(in1), buf1, inputBytes);
      memcpy(IOSurfaceGetBaseAddress(in2), buf2, inputBytes);

      for (int i = 0; i < 3; ++i) {
        ane_interop_eval(h);
      }

      uint64_t totalHw = 0;
      bool ok = true;
      uint64_t t0 = mach_absolute_time();
      for (int i = 0; i < iters; ++i) {
        ok = ane_interop_eval(h);
        totalHw += ane_interop_last_hw_execution_time_ns(h);
        if (!ok) break;
      }
      uint64_t t1 = mach_absolute_time();
      if (!ok) {
        printf("{\"ok\":false,\"stage\":\"eval\"}\n");
        ane_interop_free(h);
        free(outSizes);
        free(buf0);
        free(buf1);
        free(buf2);
        return 7;
      }

      NSMutableDictionary *meta = [@{
        @"ok": @YES,
        @"hw_ns": @(totalHw / (uint64_t)iters),
        @"wall_ns": @((uint64_t)(ticks_to_ns(t1 - t0) / (double)iters))
      } mutableCopy];
      NSData *json = [NSJSONSerialization dataWithJSONObject:meta options:0 error:nil];
      write_exact(stdout, json.bytes, json.length);
      write_exact(stdout, "\n", 1);
      for (int i = 0; i < outputCount; ++i) {
        IOSurfaceRef out = ane_interop_copy_output(h, i);
        write_exact(stdout, IOSurfaceGetBaseAddress(out), outputBytes);
        CFRelease(out);
      }
    }

    ane_interop_free(h);
    free(outSizes);
    free(buf0);
    free(buf1);
    free(buf2);
    return 0;
  }
}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    token_ids = tokenizer.encode(PROMPT)[: args.token_limit]
    tokens = mx.array([token_ids], dtype=mx.int32)
    layer = model.model.layers[0]

    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()
    x = layer.input_layernorm(h)
    q = layer.self_attn.q_proj(x)
    k = layer.self_attn.k_proj(x)
    v = layer.self_attn.v_proj(x)
    q = layer.self_attn.q_norm(q.reshape(1, args.token_limit, num_heads, -1)).transpose(0, 2, 1, 3)
    k = layer.self_attn.k_norm(k.reshape(1, args.token_limit, num_kv_heads, -1)).transpose(0, 2, 1, 3)
    v = v.reshape(1, args.token_limit, num_kv_heads, -1).transpose(0, 2, 1, 3)
    q = layer.self_attn.rope(q)
    k = layer.self_attn.rope(k)
    v = repeat_kv(v, num_heads, num_kv_heads, args.token_limit, head_dim)
    k = repeat_kv(k, num_heads, num_kv_heads, args.token_limit, head_dim)
    q_np = np.array(q, copy=False).astype(np.float16, copy=False)
    k_np = np.array(k, copy=False).astype(np.float16, copy=False)
    v_np = np.array(v, copy=False).astype(np.float16, copy=False)

    work = Path(tempfile.mkdtemp(prefix="josie_prefix_group_interop_"))
    try:
        root = work / "artifact"
        root.mkdir(parents=True, exist_ok=True)
        (root / "model.mil").write_text(
            make_ctx_prefix_group_mil(args.token_limit, head_dim),
            encoding="utf-8",
        )
        src = work / "probe.m"
        exe = work / "probe"
        src.write_text(HELPER_SRC, encoding="utf-8")
        subprocess.check_call(
            [
                "clang",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "IOSurface",
                "-ldl",
                "-Ithird_party/espresso_ane/include",
                "third_party/espresso_ane/ane_interop.m",
                "third_party/espresso_ane/neon_convert.c",
                "third_party/espresso_ane/surface_io.c",
                str(src),
                "-o",
                str(exe),
            ]
        )

        reports = []
        for path_name in ("inmem", "client", "clientDirect", "realtime"):
            env = dict(os.environ)
            env["ANE_EVAL_PATH"] = path_name
            proc = subprocess.Popen(
                [str(exe), str(root), str(args.token_limit), str(head_dim), str(args.iters)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            try:
                qh = q_np[:, 0:1, :, :]
                kh = k_np[:, 0:1, :, :]
                vh = v_np[:, 0:1, :, :]
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(qh.tobytes())
                proc.stdin.write(kh.tobytes())
                proc.stdin.write(vh.tobytes())
                proc.stdin.flush()
                meta = json.loads(proc.stdout.readline().decode("utf-8"))
                blobs = []
                for _ in range(args.token_limit - 1):
                    blobs.append(proc.stdout.read(head_dim * 2))
                ref_reports = []
                for prefix, blob in enumerate(blobs, start=2):
                    ref = mx.fast.scaled_dot_product_attention(
                        q[:, 0:1, prefix - 1 : prefix, :],
                        k[:, 0:1, :prefix, :],
                        v[:, 0:1, :prefix, :],
                        scale=head_dim ** -0.5,
                    )
                    mx.eval(ref)
                    mx.synchronize()
                    ref_np = np.array(ref, copy=False).astype(np.float32)
                    got = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(1, 1, 1, head_dim)
                    diffs = np.abs(got - ref_np)
                    ref_reports.append(
                        {
                            "prefix": prefix,
                            "max_abs_diff": float(np.max(diffs)),
                            "mean_abs_diff": float(np.mean(diffs)),
                        }
                    )
                reports.append({"path": path_name, "meta": meta, "prefixes": ref_reports})
            finally:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

        payload = {"runtime": "josie_prefix_group_interop_probe", "reports": reports}
        if args.json:
            print(json.dumps(payload))
            return
        print(json.dumps(payload, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
