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


HELPER_SRC = r'''
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdint.h>
#include <string.h>
#include <mach/mach_time.h>

static double ticks_to_ms(uint64_t t) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)t * tb.numer / tb.denom / 1e6;
}

static void print_json_error(const char *stage, const char *error) {
    NSMutableString *escaped = [NSMutableString string];
    const char *src = error ? error : "";
    for (const char *p = src; *p; ++p) {
        switch (*p) {
            case '\\': [escaped appendString:@"\\\\"]; break;
            case '\"': [escaped appendString:@"\\\""]; break;
            case '\n': [escaped appendString:@"\\n"]; break;
            case '\r': [escaped appendString:@"\\r"]; break;
            case '\t': [escaped appendString:@"\\t"]; break;
            default: [escaped appendFormat:@"%c", *p]; break;
        }
    }
    printf("{\"ok\":false,\"stage\":\"%s\",\"error\":\"%s\"}\n", stage, escaped.UTF8String);
}

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

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 8) return 2;
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        int qBytes = atoi(argv[2]);
        int kBytes = atoi(argv[3]);
        int vBytes = atoi(argv[4]);
        int outBytes = atoi(argv[5]);
        int iters = atoi(argv[6]);
        NSString *outPath = [NSString stringWithUTF8String:argv[7]];

        NSData *milData = [[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"]
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
        if (!desc) {
            print_json_error("descriptor", "nil");
            return 3;
        }
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            print_json_error("model", "nil");
            return 4;
        }

        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:tmpDir withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            print_json_error("compile", e ? [[e description] UTF8String] : "");
            return 5;
        }

        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            print_json_error("load", e ? [[e description] UTF8String] : "");
            return 6;
        }

        IOSurfaceRef ioQ = make_surface(qBytes);
        IOSurfaceRef ioK = make_surface(kBytes);
        IOSurfaceRef ioV = make_surface(vBytes);
        IOSurfaceRef ioO = make_surface(outBytes);

        NSData *q = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"q.bin"]];
        NSData *k = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"k.bin"]];
        NSData *v = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"v.bin"]];

        IOSurfaceLock(ioQ, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(ioQ), q.bytes, q.length);
        IOSurfaceUnlock(ioQ, 0, NULL);
        IOSurfaceLock(ioK, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(ioK), k.bytes, k.length);
        IOSurfaceUnlock(ioK, 0, NULL);
        IOSurfaceLock(ioV, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(ioV), v.bytes, v.length);
        IOSurfaceUnlock(ioV, 0, NULL);

        id wQ = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioQ);
        id wK = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioK);
        id wV = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioV);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioO);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wQ, wK, wV], @[@0, @1, @2], @[wO], @[@0], nil, nil, @0);

        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) {
            print_json_error("eval", e ? [[e description] UTF8String] : "");
            return 7;
        }

        for (int i = 0; i < 5; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        }
        uint64_t t1 = mach_absolute_time();

        IOSurfaceLock(ioO, kIOSurfaceLockReadOnly, NULL);
        NSData *out = [NSData dataWithBytes:IOSurfaceGetBaseAddress(ioO) length:outBytes];
        [out writeToFile:outPath atomically:YES];
        IOSurfaceUnlock(ioO, kIOSurfaceLockReadOnly, NULL);

        printf("{\"ok\":true,\"per_iter_ms\":%.6f}\n", ticks_to_ms(t1 - t0) / iters);
        return 0;
    }
}
'''


def make_mil(num_heads: int, seq_len: int, head_dim: int) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {num_heads}, {seq_len}, {head_dim}]> q,
        tensor<fp16, [1, {num_heads}, {seq_len}, {head_dim}]> k,
        tensor<fp16, [1, {num_heads}, {seq_len}, {head_dim}]> v
    ) {{
        tensor<fp16, [1, {num_heads}, {seq_len}, {head_dim}]> out =
            scaled_dot_product_attention(query = q, key = k, value = v)[name = string("sdpa")];
    }} -> (out);
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, _tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    attn = model.model.layers[0].self_attn
    dim = 2560
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    repeat = num_heads // num_kv_heads

    x = mx.random.normal(shape=(1, args.seq_len, dim), dtype=mx.float16) * 0.1

    def repeat_kv(tensor):
        return mx.broadcast_to(
            mx.expand_dims(tensor, axis=2),
            (1, num_kv_heads, repeat, args.seq_len, head_dim),
        ).reshape(1, num_heads, args.seq_len, head_dim)

    def project():
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        q = attn.q_norm(q.reshape(1, args.seq_len, num_heads, -1)).transpose(0, 2, 1, 3)
        k = attn.k_norm(k.reshape(1, args.seq_len, num_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(1, args.seq_len, num_kv_heads, -1).transpose(0, 2, 1, 3)
        q = attn.rope(q)
        k = attn.rope(k)
        k = repeat_kv(k)
        v = repeat_kv(v)
        return q, k, v

    q, k, v = project()
    q_np = np.array(q, copy=False).astype(np.float16, copy=False)
    k_np = np.array(k, copy=False).astype(np.float16, copy=False)
    v_np = np.array(v, copy=False).astype(np.float16, copy=False)

    def mlx_forward():
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=head_dim ** -0.5,
        )
        out = out.transpose(0, 2, 1, 3).reshape(1, args.seq_len, -1)
        out = attn.o_proj(out)
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
    mlx_out = np.array(mlx_last, copy=False).astype(np.float32, copy=False)

    work_dir = Path(tempfile.mkdtemp(prefix="josie_hybrid_attn_"))
    try:
        root = work_dir / "artifact"
        root.mkdir(parents=True, exist_ok=True)
        (root / "q.bin").write_bytes(q_np.tobytes())
        (root / "k.bin").write_bytes(k_np.tobytes())
        (root / "v.bin").write_bytes(v_np.tobytes())
        (root / "model.mil").write_text(
            make_mil(num_heads=num_heads, seq_len=args.seq_len, head_dim=head_dim),
            encoding="utf-8",
        )
        src = work_dir / "ane_sdpa_helper.m"
        exe = work_dir / "ane_sdpa_helper"
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

        q_bytes = q_np.nbytes
        k_bytes = k_np.nbytes
        v_bytes = v_np.nbytes
        out_bytes = q_np.nbytes
        out_path = work_dir / "out.bin"
        proc = subprocess.run(
            [
                str(exe),
                str(root),
                str(q_bytes),
                str(k_bytes),
                str(v_bytes),
                str(out_bytes),
                str(args.iters),
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        payload = json.loads(proc.stdout.strip() or "{}")
        payload["returncode"] = proc.returncode
        if not payload.get("ok"):
            raise SystemExit(
                f"ANE sdpa failed at stage={payload.get('stage')} error={payload.get('error','')}"
            )

        ane_ctx = np.frombuffer(out_path.read_bytes(), dtype=np.float16).astype(np.float32)
        ane_ctx = ane_ctx.reshape(1, num_heads, args.seq_len, head_dim)
        ane_ctx = mx.array(ane_ctx.astype(np.float16), dtype=mx.float16)
        ane_ctx = ane_ctx.transpose(0, 2, 1, 3).reshape(1, args.seq_len, -1)
        ane_out = attn.o_proj(ane_ctx)
        mx.eval(ane_out)
        mx.synchronize()
        ane_out_np = np.array(ane_out, copy=False).astype(np.float32, copy=False)

        diffs = np.abs(ane_out_np - mlx_out)
        report = {
            "runtime": "josie_attention_hybrid_private_vs_mlx",
            "seq_len": args.seq_len,
            "ane_sdpa_per_iter_ms": float(payload["per_iter_ms"]),
            "mlx_attention_per_iter_ms": mlx_ms,
            "ane_sdpa_speedup_vs_mlx_attention": mlx_ms / float(payload["per_iter_ms"]),
            "max_abs_diff": float(np.max(diffs)),
            "mean_abs_diff": float(np.mean(diffs)),
        }
        if args.json:
            print(json.dumps(report))
            return
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
