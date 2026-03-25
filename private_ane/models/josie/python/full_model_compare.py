from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


PROMPT = "Explain why MLX is useful for local inference."

HELPER_SRC = r'''
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdint.h>
#include <string.h>

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
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
        if (argc < 6) return 2;
        NSString *milPath = [NSString stringWithUTF8String:argv[1]];
        int qBytes = atoi(argv[2]);
        int kBytes = atoi(argv[3]);
        int vBytes = atoi(argv[4]);
        int outBytes = atoi(argv[5]);
        int allocBytes = qBytes;
        if (kBytes > allocBytes) allocBytes = kBytes;
        if (vBytes > allocBytes) allocBytes = vBytes;
        if (outBytes > allocBytes) allocBytes = outBytes;

        NSData *milData = [[NSString stringWithContentsOfFile:milPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:nil] dataUsingEncoding:NSUTF8StringEncoding];

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        NSError *e = nil;

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
        if (!desc) return 3;
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return 4;

        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:tmpDir withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) return 5;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) return 6;

        IOSurfaceRef ioQ = make_surface(allocBytes);
        IOSurfaceRef ioK = make_surface(allocBytes);
        IOSurfaceRef ioV = make_surface(allocBytes);
        IOSurfaceRef ioO = make_surface(allocBytes);

        id wQ = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioQ);
        id wK = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioK);
        id wV = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioV);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioO);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wQ, wK, wV], @[@0, @1, @2], @[wO], @[@0], nil, nil, @0);

        void *qBuf = malloc(qBytes);
        void *kBuf = malloc(kBytes);
        void *vBuf = malloc(vBytes);

        while (read_exact(stdin, qBuf, qBytes) &&
               read_exact(stdin, kBuf, kBytes) &&
               read_exact(stdin, vBuf, vBytes)) {
            IOSurfaceLock(ioQ, 0, NULL);
            memset(IOSurfaceGetBaseAddress(ioQ), 0, allocBytes);
            memcpy(IOSurfaceGetBaseAddress(ioQ), qBuf, qBytes);
            IOSurfaceUnlock(ioQ, 0, NULL);

            IOSurfaceLock(ioK, 0, NULL);
            memset(IOSurfaceGetBaseAddress(ioK), 0, allocBytes);
            memcpy(IOSurfaceGetBaseAddress(ioK), kBuf, kBytes);
            IOSurfaceUnlock(ioK, 0, NULL);

            IOSurfaceLock(ioV, 0, NULL);
            memset(IOSurfaceGetBaseAddress(ioV), 0, allocBytes);
            memcpy(IOSurfaceGetBaseAddress(ioV), vBuf, vBytes);
            IOSurfaceUnlock(ioV, 0, NULL);

            e = nil;
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            if (!ok) return 7;

            IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
            const void *outPtr = IOSurfaceGetBaseAddress(ioO);
            bool wrote = write_exact(stdout, outPtr, outBytes);
            IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);
            if (!wrote) return 8;
        }

        free(qBuf);
        free(kBuf);
        free(vBuf);
        return 0;
    }
}
'''


def make_mil(
    num_heads: int,
    seq_len: int,
    head_dim: int,
    *,
    query_len: int | None = None,
    key_len: int | None = None,
) -> str:
    query_len = seq_len if query_len is None else query_len
    key_len = seq_len if key_len is None else key_len
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {num_heads}, {query_len}, {head_dim}]> q,
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> k,
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> v
    ) {{
        tensor<fp16, [1, {num_heads}, {query_len}, {head_dim}]> out =
            scaled_dot_product_attention(query = q, key = k, value = v)[name = string("sdpa")];
    }} -> (out);
}}
'''


def make_sdpa_server(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    query_len: int | None = None,
    key_len: int | None = None,
):
    work_dir = Path(tempfile.mkdtemp(prefix="josie_full_model_"))
    mil_path = work_dir / "sdpa.mil"
    mil_path.write_text(
        make_mil(
            num_heads,
            seq_len,
            head_dim,
            query_len=query_len,
            key_len=key_len,
        ),
        encoding="utf-8",
    )
    src = work_dir / "ane_sdpa_server.m"
    exe = work_dir / "ane_sdpa_server"
    src.write_text(HELPER_SRC, encoding="utf-8")
    subprocess.check_call(
        [
            "clang",
            "-fobjc-arc",
            "-framework",
            "Foundation",
            "-framework",
            "CoreML",
            "-framework",
            "IOSurface",
            "-ldl",
            "-o",
            str(exe),
            str(src),
        ]
    )
    return work_dir, exe, mil_path


def repeat_kv(tensor, num_heads: int, num_kv_heads: int, seq_len: int, head_dim: int):
    repeat = num_heads // num_kv_heads
    return mx.broadcast_to(
        mx.expand_dims(tensor, axis=2),
        (1, num_kv_heads, repeat, seq_len, head_dim),
    ).reshape(1, num_heads, seq_len, head_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--token-limit", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    seq_len = len(token_ids)
    tokens = mx.array([token_ids], dtype=mx.int32)

    def mlx_forward():
        out = model(tokens)
        mx.eval(out)
        mx.synchronize()
        return out

    for _ in range(args.warmup):
        mlx_forward()
    started = time.perf_counter()
    mlx_last = None
    for _ in range(args.iters):
        mlx_last = mlx_forward()
    mlx_ms = (time.perf_counter() - started) * 1000.0 / args.iters
    mlx_out = np.array(mlx_last.astype(mx.float32).tolist(), dtype=np.float32)

    def fp16_reference_forward():
        h = model.model.embed_tokens(tokens).astype(mx.float16)
        mx.eval(h)
        mx.synchronize()
        for layer in model.model.layers:
            x_norm = layer.input_layernorm(h).astype(mx.float16)
            q = layer.self_attn.q_proj(x_norm).astype(mx.float16)
            k = layer.self_attn.k_proj(x_norm).astype(mx.float16)
            v = layer.self_attn.v_proj(x_norm).astype(mx.float16)

            q = layer.self_attn.q_norm(q.reshape(1, seq_len, num_heads, -1)).transpose(0, 2, 1, 3).astype(mx.float16)
            k = layer.self_attn.k_norm(k.reshape(1, seq_len, num_kv_heads, -1)).transpose(0, 2, 1, 3).astype(mx.float16)
            v = v.reshape(1, seq_len, num_kv_heads, -1).transpose(0, 2, 1, 3).astype(mx.float16)

            q = layer.self_attn.rope(q).astype(mx.float16)
            k = layer.self_attn.rope(k).astype(mx.float16)
            k = repeat_kv(k, num_heads, num_kv_heads, seq_len, head_dim).astype(mx.float16)
            v = repeat_kv(v, num_heads, num_kv_heads, seq_len, head_dim).astype(mx.float16)

            ctx = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=head_dim ** -0.5,
            ).astype(mx.float16)
            attn_out = layer.self_attn.o_proj(
                ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
            ).astype(mx.float16)
            h = (h + attn_out).astype(mx.float16)

            ffn_in = layer.post_attention_layernorm(h)
            ffn_out = layer.mlp(ffn_in)
            h = (h + ffn_out).astype(mx.float16)

        h = model.model.norm(h).astype(mx.float16)
        out = model.model.embed_tokens.as_linear(h)
        mx.eval(out)
        mx.synchronize()
        return out

    fp16_last = fp16_reference_forward()
    fp16_out = np.array(fp16_last.astype(mx.float32).tolist(), dtype=np.float32)

    server_dir, server_exe, mil_path = make_sdpa_server(seq_len, num_heads, head_dim)
    proc = subprocess.Popen(
        [
            str(server_exe),
            str(mil_path),
            str(num_heads * seq_len * head_dim * 2),
            str(num_heads * seq_len * head_dim * 2),
            str(num_heads * seq_len * head_dim * 2),
            str(num_heads * seq_len * head_dim * 2),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        def ane_sdpa_forward():
            h = model.model.embed_tokens(tokens).astype(mx.float16)
            mx.eval(h)
            mx.synchronize()
            for layer in model.model.layers:
                x_norm = layer.input_layernorm(h)
                q = layer.self_attn.q_proj(x_norm)
                k = layer.self_attn.k_proj(x_norm)
                v = layer.self_attn.v_proj(x_norm)

                q = layer.self_attn.q_norm(
                    q.reshape(1, seq_len, num_heads, -1)
                ).transpose(0, 2, 1, 3)
                k = layer.self_attn.k_norm(
                    k.reshape(1, seq_len, num_kv_heads, -1)
                ).transpose(0, 2, 1, 3)
                v = v.reshape(1, seq_len, num_kv_heads, -1).transpose(0, 2, 1, 3)

                q = layer.self_attn.rope(q)
                k = layer.self_attn.rope(k)
                k = repeat_kv(k, num_heads, num_kv_heads, seq_len, head_dim)
                v = repeat_kv(v, num_heads, num_kv_heads, seq_len, head_dim)

                q_np = np.array(q, copy=False).astype(np.float16, copy=False)
                k_np = np.array(k, copy=False).astype(np.float16, copy=False)
                v_np = np.array(v, copy=False).astype(np.float16, copy=False)

                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(q_np.tobytes())
                proc.stdin.write(k_np.tobytes())
                proc.stdin.write(v_np.tobytes())
                proc.stdin.flush()

                out_bytes = q_np.nbytes
                chunk = proc.stdout.read(out_bytes)
                if chunk is None or len(chunk) != out_bytes:
                    stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                    raise RuntimeError(f"ANE SDPA server failed. stderr={stderr}")

                ctx = np.frombuffer(chunk, dtype=np.float16).astype(np.float32)
                ctx = ctx.reshape(1, num_heads, seq_len, head_dim)
                ctx = mx.array(ctx.astype(np.float16), dtype=mx.float16)
                attn_out = layer.self_attn.o_proj(
                    ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
                )
                h = h + attn_out

                ffn_in = layer.post_attention_layernorm(h)
                ffn_out = layer.mlp(ffn_in)
                h = h + ffn_out

            h = model.model.norm(h)
            out = model.model.embed_tokens.as_linear(h)
            mx.eval(out)
            mx.synchronize()
            return out

        for _ in range(args.warmup):
            ane_sdpa_forward()
        started = time.perf_counter()
        ane_last = None
        for _ in range(args.iters):
            ane_last = ane_sdpa_forward()
        ane_ms = (time.perf_counter() - started) * 1000.0 / args.iters
        ane_out = np.array(ane_last.astype(mx.float32).tolist(), dtype=np.float32)
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
        shutil.rmtree(server_dir, ignore_errors=True)

    diffs = np.abs(ane_out - mlx_out)
    fp16_diffs = np.abs(ane_out - fp16_out)
    fp16_ref_diffs = np.abs(fp16_out - mlx_out)
    ane_argmax = int(np.argmax(ane_out[0, -1]))
    mlx_argmax = int(np.argmax(mlx_out[0, -1]))
    fp16_argmax = int(np.argmax(fp16_out[0, -1]))
    report = {
        "runtime": "josie_full_model_hybrid_private_vs_mlx",
        "prompt": args.prompt,
        "token_count": seq_len,
        "ane_per_iter_ms": ane_ms,
        "mlx_per_iter_ms": mlx_ms,
        "ane_speedup_vs_mlx": mlx_ms / ane_ms,
        "max_abs_diff": float(np.max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
        "last_token_argmax_match": ane_argmax == mlx_argmax,
        "ane_last_token_argmax": ane_argmax,
        "mlx_last_token_argmax": mlx_argmax,
        "fp16_ref_last_token_argmax": fp16_argmax,
        "last_token_argmax_match_vs_fp16_ref": ane_argmax == fp16_argmax,
        "max_abs_diff_vs_fp16_ref": float(np.max(fp16_diffs)),
        "mean_abs_diff_vs_fp16_ref": float(np.mean(fp16_diffs)),
        "fp16_ref_max_abs_diff_vs_mlx": float(np.max(fp16_ref_diffs)),
        "fp16_ref_mean_abs_diff_vs_mlx": float(np.mean(fp16_ref_diffs)),
    }

    if args.json:
        print(json.dumps(report))
        return

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
