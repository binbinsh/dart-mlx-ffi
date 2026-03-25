from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from full_model_compare import PROMPT, repeat_kv

JOSIE_REPO = "mlx-community/JOSIE-1.1-4B-Instruct-4bit"
JOSIE_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--JOSIE-1.1-4B-Instruct-4bit"
    / "snapshots"
)

COREML_HELPER_SRC = r'''
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

static void print_error(const char *stage, NSError *e) {
    const char *msg = e ? [[e description] UTF8String] : "unknown";
    fprintf(stderr, "%s: %s\n", stage, msg);
    fflush(stderr);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 5) return 2;
        NSString *milPath = [NSString stringWithUTF8String:argv[1]];
        int seqLen = atoi(argv[2]);
        int headDim = atoi(argv[3]);
        int numHeads = atoi(argv[4]);
        int outputCount = seqLen - 1;
        size_t inputBytes = (size_t)seqLen * (size_t)headDim * 2;
        size_t totalInputBytes = (size_t)numHeads * inputBytes;
        size_t outputBytes = (size_t)headDim * 2;
        size_t totalOutputBytes = (size_t)numHeads * (size_t)outputCount * outputBytes;
        size_t allocBytes = inputBytes > (49 * 1024) ? inputBytes : (49 * 1024);

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
        [[NSFileManager defaultManager] createDirectoryAtPath:tmpDir
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            print_error("compile", e);
            return 5;
        }
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            print_error("load", e);
            return 6;
        }

        IOSurfaceRef inQ = make_surface(allocBytes);
        IOSurfaceRef inK = make_surface(allocBytes);
        IOSurfaceRef inV = make_surface(allocBytes);
        id wQ = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), inQ);
        id wK = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), inK);
        id wV = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), inV);

        NSMutableArray *outSurfaces = [NSMutableArray array];
        NSMutableArray *outputs = [NSMutableArray array];
        NSMutableArray *outputIndices = [NSMutableArray array];
        for (int i = 0; i < outputCount; ++i) {
            IOSurfaceRef io = make_surface(allocBytes);
            [outSurfaces addObject:(__bridge id)io];
            [outputs addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                AIO, @selector(objectWithIOSurface:), io)];
            [outputIndices addObject:@(i)];
        }

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wQ, wK, wV], @[@0, @1, @2], outputs, outputIndices, nil, nil, @0);

        uint8_t *qAll = (uint8_t *)malloc(totalInputBytes);
        uint8_t *kAll = (uint8_t *)malloc(totalInputBytes);
        uint8_t *vAll = (uint8_t *)malloc(totalInputBytes);
        uint8_t *outAll = (uint8_t *)malloc(totalOutputBytes);

        while (read_exact(stdin, qAll, totalInputBytes) &&
               read_exact(stdin, kAll, totalInputBytes) &&
               read_exact(stdin, vAll, totalInputBytes)) {
            memset(outAll, 0, totalOutputBytes);
            for (int head = 0; head < numHeads; ++head) {
                size_t off = (size_t)head * inputBytes;
                IOSurfaceLock(inQ, 0, NULL);
                memset(IOSurfaceGetBaseAddress(inQ), 0, allocBytes);
                memcpy(IOSurfaceGetBaseAddress(inQ), qAll + off, inputBytes);
                IOSurfaceUnlock(inQ, 0, NULL);

                IOSurfaceLock(inK, 0, NULL);
                memset(IOSurfaceGetBaseAddress(inK), 0, allocBytes);
                memcpy(IOSurfaceGetBaseAddress(inK), kAll + off, inputBytes);
                IOSurfaceUnlock(inK, 0, NULL);

                IOSurfaceLock(inV, 0, NULL);
                memset(IOSurfaceGetBaseAddress(inV), 0, allocBytes);
                memcpy(IOSurfaceGetBaseAddress(inV), vAll + off, inputBytes);
                IOSurfaceUnlock(inV, 0, NULL);

                for (id surfaceObj in outSurfaces) {
                    IOSurfaceRef io = (__bridge IOSurfaceRef)surfaceObj;
                    IOSurfaceLock(io, 0, NULL);
                    memset(IOSurfaceGetBaseAddress(io), 0, allocBytes);
                    IOSurfaceUnlock(io, 0, NULL);
                }

                e = nil;
                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                if (!ok) {
                    print_error("eval", e);
                    return 7;
                }

                for (int outIndex = 0; outIndex < outputCount; ++outIndex) {
                    IOSurfaceRef io = (__bridge IOSurfaceRef)[outSurfaces objectAtIndex:outIndex];
                    IOSurfaceLock(io, kIOSurfaceLockReadOnly, NULL);
                    const uint8_t *src = (const uint8_t *)IOSurfaceGetBaseAddress(io);
                    size_t dst = ((size_t)head * (size_t)outputCount + (size_t)outIndex) * outputBytes;
                    memcpy(outAll + dst, src, outputBytes);
                    IOSurfaceUnlock(io, kIOSurfaceLockReadOnly, NULL);
                }
            }

            if (!write_exact(stdout, outAll, totalOutputBytes)) {
                return 8;
            }
        }

        free(qAll);
        free(kAll);
        free(vAll);
        free(outAll);
        return 0;
    }
}
'''

INTEROP_HELPER_SRC = r'''
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "ane_interop.h"
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

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 5) return 2;
        NSString *milPath = [NSString stringWithUTF8String:argv[1]];
        int seqLen = atoi(argv[2]);
        int headDim = atoi(argv[3]);
        int numHeads = atoi(argv[4]);
        int outputCount = seqLen - 1;
        size_t inputBytes = (size_t)seqLen * (size_t)headDim * 2;
        size_t totalInputBytes = (size_t)numHeads * inputBytes;
        size_t outputBytes = (size_t)headDim * 2;
        size_t totalOutputBytes = (size_t)numHeads * (size_t)outputCount * outputBytes;

        NSData *mil = [[NSString stringWithContentsOfFile:milPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:nil] dataUsingEncoding:NSUTF8StringEncoding];
        size_t inSizes[3] = {inputBytes, inputBytes, inputBytes};
        size_t *outSizes = (size_t *)malloc((size_t)outputCount * sizeof(size_t));
        for (int i = 0; i < outputCount; ++i) {
            outSizes[i] = outputBytes;
        }

        ANEHandle *handle = ane_interop_compile(
            (const uint8_t *)mil.bytes,
            mil.length,
            NULL,
            NULL,
            NULL,
            0,
            3,
            inSizes,
            outputCount,
            outSizes
        );
        if (!handle) {
            free(outSizes);
            return 3;
        }

        IOSurfaceRef inQ = ane_interop_copy_input(handle, 0);
        IOSurfaceRef inK = ane_interop_copy_input(handle, 1);
        IOSurfaceRef inV = ane_interop_copy_input(handle, 2);
        uint8_t *qAll = (uint8_t *)malloc(totalInputBytes);
        uint8_t *kAll = (uint8_t *)malloc(totalInputBytes);
        uint8_t *vAll = (uint8_t *)malloc(totalInputBytes);
        uint8_t *outAll = (uint8_t *)malloc(totalOutputBytes);

        while (read_exact(stdin, qAll, totalInputBytes) &&
               read_exact(stdin, kAll, totalInputBytes) &&
               read_exact(stdin, vAll, totalInputBytes)) {
            for (int head = 0; head < numHeads; ++head) {
                size_t off = (size_t)head * inputBytes;
                memcpy(IOSurfaceGetBaseAddress(inQ), qAll + off, inputBytes);
                memcpy(IOSurfaceGetBaseAddress(inK), kAll + off, inputBytes);
                memcpy(IOSurfaceGetBaseAddress(inV), vAll + off, inputBytes);
                if (!ane_interop_eval(handle)) {
                    ane_interop_free(handle);
                    free(outSizes);
                    free(qAll);
                    free(kAll);
                    free(vAll);
                    free(outAll);
                    return 7;
                }
                size_t base = (size_t)head * (size_t)outputCount * outputBytes;
                for (int outIndex = 0; outIndex < outputCount; ++outIndex) {
                    IOSurfaceRef out = ane_interop_copy_output(handle, outIndex);
                    memcpy(
                        outAll + base + ((size_t)outIndex * outputBytes),
                        IOSurfaceGetBaseAddress(out),
                        outputBytes
                    );
                    CFRelease(out);
                }
            }

            if (!write_exact(stdout, outAll, totalOutputBytes)) {
                ane_interop_free(handle);
                free(outSizes);
                free(qAll);
                free(kAll);
                free(vAll);
                free(outAll);
                return 8;
            }
        }

        ane_interop_free(handle);
        free(outSizes);
        free(qAll);
        free(kAll);
        free(vAll);
        free(outAll);
        return 0;
    }
}
'''


def load_josie(*, lazy: bool):
    if JOSIE_CACHE.exists():
        snapshots = sorted(
            (path for path in JOSIE_CACHE.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return load(str(snapshots[0]), lazy=lazy)
    return load(JOSIE_REPO, lazy=lazy)


def make_ctx_prefix_group_mil(seq_len: int, head_dim: int) -> str:
    bodies = []
    outputs = []
    scale = head_dim ** -0.5
    for prefix in range(2, seq_len + 1):
        q_begin = [0, 0, prefix - 1, 0]
        q_end = [1, 1, prefix, head_dim]
        kv_begin = [0, 0, 0, 0]
        kv_end = [1, 1, prefix, head_dim]
        out_name = f"z{prefix}"
        outputs.append(out_name)
        bodies.append(
            f'''        tensor<int32, [4]> q{prefix}_b = const()[name = string("q{prefix}_b"), val = tensor<int32, [4]>([{q_begin[0]},{q_begin[1]},{q_begin[2]},{q_begin[3]}])];
        tensor<int32, [4]> q{prefix}_e = const()[name = string("q{prefix}_e"), val = tensor<int32, [4]>([{q_end[0]},{q_end[1]},{q_end[2]},{q_end[3]}])];
        tensor<int32, [4]> k{prefix}_b = const()[name = string("k{prefix}_b"), val = tensor<int32, [4]>([{kv_begin[0]},{kv_begin[1]},{kv_begin[2]},{kv_begin[3]}])];
        tensor<int32, [4]> k{prefix}_e = const()[name = string("k{prefix}_e"), val = tensor<int32, [4]>([{kv_end[0]},{kv_end[1]},{kv_end[2]},{kv_end[3]}])];
        tensor<fp16, [1, 1, 1, {head_dim}]> q{prefix} =
            slice_by_index(begin = q{prefix}_b, end = q{prefix}_e, x = a)[name = string("q{prefix}")];
        tensor<fp16, [1, 1, {prefix}, {head_dim}]> k{prefix} =
            slice_by_index(begin = k{prefix}_b, end = k{prefix}_e, x = b)[name = string("k{prefix}")];
        tensor<fp16, [1, 1, {prefix}, {head_dim}]> v{prefix} =
            slice_by_index(begin = k{prefix}_b, end = k{prefix}_e, x = c)[name = string("v{prefix}")];
        bool s{prefix}_tx = const()[name = string("s{prefix}_tx"), val = bool(false)];
        bool s{prefix}_ty = const()[name = string("s{prefix}_ty"), val = bool(true)];
        int32 p{prefix}_ax = const()[name = string("p{prefix}_ax"), val = int32(-1)];
        fp16 p{prefix}_sc = const()[name = string("p{prefix}_sc"), val = fp16({scale:.8f})];
        tensor<fp16, [1, 1, 1, {prefix}]> s{prefix} =
            matmul(transpose_x = s{prefix}_tx, transpose_y = s{prefix}_ty, x = q{prefix}, y = k{prefix})[name = string("s{prefix}")];
        tensor<fp16, [1, 1, 1, {prefix}]> u{prefix} =
            mul(x = s{prefix}, y = p{prefix}_sc)[name = string("u{prefix}")];
        tensor<fp16, [1, 1, 1, {prefix}]> p{prefix} =
            softmax(axis = p{prefix}_ax, x = u{prefix})[name = string("p{prefix}")];
        tensor<fp16, [1, 1, 1, {head_dim}]> {out_name} =
            matmul(transpose_x = s{prefix}_tx, transpose_y = s{prefix}_tx, x = p{prefix}, y = v{prefix})[name = string("{out_name}")];'''
        )

    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, 1, {seq_len}, {head_dim}]> a,
        tensor<fp16, [1, 1, {seq_len}, {head_dim}]> b,
        tensor<fp16, [1, 1, {seq_len}, {head_dim}]> c
    ) {{
{chr(10).join(bodies)}
    }} -> ({", ".join(outputs)});
}}
'''


def _head_helper_backend() -> str:
    raw = os.environ.get("JOSIE_HEAD_HELPER_BACKEND", "coreml").strip().lower()
    return raw if raw in {"interop", "coreml"} else "coreml"


def _compile_interop_helper(src: Path, exe: Path) -> None:
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
        ],
        cwd=ROOT,
    )


def _build_coreml_head_server(work_dir: Path, seq_len: int, *, head_dim: int):
    mil_path = work_dir / "model.mil"
    src = work_dir / "helper.m"
    exe = work_dir / "helper"
    mil_path.write_text(make_ctx_prefix_group_mil(seq_len, head_dim), encoding="utf-8")
    src.write_text(COREML_HELPER_SRC, encoding="utf-8")
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
        ],
        cwd=ROOT,
    )
    proc = subprocess.Popen(
        [str(exe), str(mil_path), str(seq_len), str(head_dim), "32"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
    )
    return work_dir, proc, "coreml"


def _build_interop_head_server(work_dir: Path, seq_len: int, *, head_dim: int):
    mil_path = work_dir / "model.mil"
    src = work_dir / "helper.m"
    exe = work_dir / "helper"
    mil_path.write_text(make_ctx_prefix_group_mil(seq_len, head_dim), encoding="utf-8")
    src.write_text(INTEROP_HELPER_SRC, encoding="utf-8")
    _compile_interop_helper(src, exe)
    proc = subprocess.Popen(
        [str(exe), str(mil_path), str(seq_len), str(head_dim), "32"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
    )
    return work_dir, proc, "ane_interop"


def build_head_server(seq_len: int, *, head_dim: int):
    if seq_len <= 1:
        return None
    work_dir = Path(tempfile.mkdtemp(prefix=f"josie_hs_group_{seq_len}_"))
    backend = _head_helper_backend()
    if backend == "coreml":
        return _build_coreml_head_server(work_dir, seq_len, head_dim=head_dim)
    try:
        return _build_interop_head_server(work_dir, seq_len, head_dim=head_dim)
    except subprocess.CalledProcessError:
        shutil.rmtree(work_dir, ignore_errors=True)
        retry_dir = Path(tempfile.mkdtemp(prefix=f"josie_hs_group_{seq_len}_"))
        return _build_coreml_head_server(retry_dir, seq_len, head_dim=head_dim)


def close_head_server(server) -> None:
    if server is None:
        return
    work_dir, proc, _backend = server
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
    shutil.rmtree(work_dir, ignore_errors=True)


def manual_heads_forward(
    model,
    tokens,
    server,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    seq_len = tokens.shape[1]
    _work_dir, proc, _backend = server
    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()
    prefix_count = seq_len - 1
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

        q_np = np.array(q, copy=False).astype(np.float16, copy=False)
        k_np = np.array(k, copy=False).astype(np.float16, copy=False)
        v_np = np.array(v, copy=False).astype(np.float16, copy=False)

        ctx_all = np.empty((1, num_heads, seq_len, head_dim), dtype=np.float16)
        ctx_all[:, :, 0:1, :] = v_np[:, :, :1, :]
        if prefix_count > 0:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(q_np.tobytes())
            proc.stdin.write(k_np.tobytes())
            proc.stdin.write(v_np.tobytes())
            proc.stdin.flush()
            total_bytes = num_heads * prefix_count * head_dim * 2
            blob = proc.stdout.read(total_bytes)
            if blob is None or len(blob) != total_bytes:
                stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                raise RuntimeError(f"ANE grouped-head helper failed. stderr={stderr}")
            raw = np.frombuffer(blob, dtype=np.float16).reshape(num_heads, prefix_count, head_dim)
            ctx_all[0, :, 1:, :] = raw

        ctx_mx = mx.array(ctx_all, dtype=mx.float16)
        attn_out = layer.self_attn.o_proj(
            ctx_mx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        ).astype(mx.float16)
        h = (h + attn_out).astype(mx.float16)
        h = (h + layer.mlp(layer.post_attention_layernorm(h)).astype(mx.float16)).astype(mx.float16)

    h = model.model.norm(h).astype(mx.float16)
    out = model.model.embed_tokens.as_linear(h)
    mx.eval(out)
    mx.synchronize()
    return out.astype(mx.float32)


def standard_sequence_forward(model, tokens):
    out = model(tokens)
    mx.eval(out)
    mx.synchronize()
    out = out.astype(mx.float32)
    return np.array(out, copy=False).astype(np.float32, copy=False)[0]


def manual_heads_sequence_forward(
    model,
    tokens,
    server,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    out = manual_heads_forward(
        model,
        tokens,
        server,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return np.array(out, copy=False).astype(np.float32, copy=False)[0]


def run_sequence(fn, *, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    last = None
    for _ in range(iters):
        last = fn()
    total_ms = (time.perf_counter() - started) * 1000.0 / iters
    return last, total_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_josie(lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    tokens = mx.array([token_ids], dtype=mx.int32)

    mlx_out, mlx_ms = run_sequence(
        lambda: standard_sequence_forward(model, tokens),
        warmup=args.warmup,
        iters=args.iters,
    )

    server = build_head_server(tokens.shape[1], head_dim=head_dim)
    try:
        ane_out, ane_ms = run_sequence(
            lambda: manual_heads_sequence_forward(
                model,
                tokens,
                server,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    finally:
        if server is not None:
            close_head_server(server)

    diffs = np.abs(ane_out - mlx_out)
    ane_argmax = [int(np.argmax(row)) for row in ane_out]
    mlx_argmax = [int(np.argmax(row)) for row in mlx_out]
    report = {
        "runtime": "josie_manual_causal_split_heads_private_vs_mlx",
        "prompt": args.prompt,
        "token_count": len(token_ids),
        "ane_backend": None if server is None else server[2],
        "ane_total_ms": ane_ms,
        "mlx_total_ms": mlx_ms,
        "ane_speedup_vs_mlx": mlx_ms / ane_ms if ane_ms else None,
        "max_abs_diff": float(np.max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
        "argmax_matches_mlx": ane_argmax == mlx_argmax,
        "ane_argmax": ane_argmax,
        "mlx_argmax": mlx_argmax,
    }

    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
