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
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static double ticks_to_ns(uint64_t t) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)t * tb.numer / tb.denom;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 6) return 2;
    NSString *root = [NSString stringWithUTF8String:argv[1]];
    int seqLen = atoi(argv[2]);
    int headDim = atoi(argv[3]);
    int numHeads = atoi(argv[4]);
    int iters = atoi(argv[5]);
    size_t inputBytes = (size_t)seqLen * (size_t)headDim * 2;
    int outputCount = seqLen - 1;
    size_t outputBytes = (size_t)headDim * 2;
    size_t totalInputBytes = (size_t)numHeads * inputBytes;
    size_t totalOutputBytes = (size_t)numHeads * (size_t)outputCount * outputBytes;

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
    uint8_t *qAll = (uint8_t *)malloc(totalInputBytes);
    uint8_t *kAll = (uint8_t *)malloc(totalInputBytes);
    uint8_t *vAll = (uint8_t *)malloc(totalInputBytes);
    uint8_t *outAll = (uint8_t *)malloc(totalOutputBytes);

    while (read_exact(stdin, qAll, totalInputBytes) &&
           read_exact(stdin, kAll, totalInputBytes) &&
           read_exact(stdin, vAll, totalInputBytes)) {
      for (int i = 0; i < 3; ++i) {
        size_t off = (size_t)i * 0;
        (void)off;
        memcpy(IOSurfaceGetBaseAddress(in0), qAll, inputBytes);
        memcpy(IOSurfaceGetBaseAddress(in1), kAll, inputBytes);
        memcpy(IOSurfaceGetBaseAddress(in2), vAll, inputBytes);
        ane_interop_eval(h);
      }

      uint64_t t0 = mach_absolute_time();
      for (int iter = 0; iter < iters; ++iter) {
        for (int head = 0; head < numHeads; ++head) {
          size_t off = (size_t)head * inputBytes;
          memcpy(IOSurfaceGetBaseAddress(in0), qAll + off, inputBytes);
          memcpy(IOSurfaceGetBaseAddress(in1), kAll + off, inputBytes);
          memcpy(IOSurfaceGetBaseAddress(in2), vAll + off, inputBytes);
          if (!ane_interop_eval(h)) {
            printf("{\"ok\":false,\"stage\":\"eval\"}\n");
            ane_interop_free(h);
            free(outSizes);
            free(qAll);
            free(kAll);
            free(vAll);
            free(outAll);
            return 7;
          }
        }
      }
      uint64_t t1 = mach_absolute_time();

      for (int head = 0; head < numHeads; ++head) {
        size_t base = (size_t)head * (size_t)outputCount * outputBytes;
        size_t off = (size_t)head * inputBytes;
        memcpy(IOSurfaceGetBaseAddress(in0), qAll + off, inputBytes);
        memcpy(IOSurfaceGetBaseAddress(in1), kAll + off, inputBytes);
        memcpy(IOSurfaceGetBaseAddress(in2), vAll + off, inputBytes);
        if (!ane_interop_eval(h)) {
          printf("{\"ok\":false,\"stage\":\"eval\"}\n");
          ane_interop_free(h);
          free(outSizes);
          free(qAll);
          free(kAll);
          free(vAll);
          free(outAll);
          return 8;
        }
        for (int outIndex = 0; outIndex < outputCount; ++outIndex) {
          IOSurfaceRef out = ane_interop_copy_output(h, outIndex);
          memcpy(outAll + base + ((size_t)outIndex * outputBytes), IOSurfaceGetBaseAddress(out), outputBytes);
          CFRelease(out);
        }
      }

      NSDictionary *meta = @{
        @"ok": @YES,
        @"wall_ns": @((uint64_t)(ticks_to_ns(t1 - t0) / (double)iters))
      };
      NSData *json = [NSJSONSerialization dataWithJSONObject:meta options:0 error:nil];
      write_exact(stdout, json.bytes, json.length);
      write_exact(stdout, "\n", 1);
      write_exact(stdout, outAll, totalOutputBytes);
    }

    ane_interop_free(h);
    free(outSizes);
    free(qAll);
    free(kAll);
    free(vAll);
    free(outAll);
    return 0;
  }
}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
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

    work = Path(tempfile.mkdtemp(prefix="josie_prefix_group_interop_batch_"))
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

        env = dict(os.environ)
        env["ANE_EVAL_PATH"] = "client"
        proc = subprocess.Popen(
            [str(exe), str(root), str(args.token_limit), str(head_dim), str(num_heads), str(args.iters)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(q_np.tobytes())
            proc.stdin.write(k_np.tobytes())
            proc.stdin.write(v_np.tobytes())
            proc.stdin.flush()
            meta = json.loads(proc.stdout.readline().decode("utf-8"))
            total_bytes = (args.token_limit - 1) * num_heads * head_dim * 2
            blob = proc.stdout.read(total_bytes)
            reports = []
            raw = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
            raw = raw.reshape(num_heads, args.token_limit - 1, 1, head_dim)
            for prefix in range(2, args.token_limit + 1):
                chunk = raw[:, prefix - 2, :, :]
                ref = mx.fast.scaled_dot_product_attention(
                    q[:, :, prefix - 1 : prefix, :],
                    k[:, :, :prefix, :],
                    v[:, :, :prefix, :],
                    scale=head_dim ** -0.5,
                )
                mx.eval(ref)
                mx.synchronize()
                ref_np = np.array(ref, copy=False).astype(np.float32)[0]
                diffs = np.abs(chunk - ref_np)
                reports.append(
                    {
                        "prefix": prefix,
                        "max_abs_diff": float(np.max(diffs)),
                        "mean_abs_diff": float(np.mean(diffs)),
                    }
                )
            payload = {"runtime": "josie_prefix_group_interop_batch_probe", "meta": meta, "prefixes": reports}
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

        if args.json:
            print(json.dumps(payload))
            return
        print(json.dumps(payload, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
