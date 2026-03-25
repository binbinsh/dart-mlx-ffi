from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import mlx.core as mx
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "shared" / "benchmark"))

JOSIE_REPO = "mlx-community/JOSIE-1.1-4B-Instruct-4bit"
JOSIE_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--JOSIE-1.1-4B-Instruct-4bit"
    / "snapshots"
)
MIN_ALLOC_BYTES = 49 * 1024

HELPER_SRC = r'''
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

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
        (id)kIOSurfaceWidth:@(bytes),
        (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,
        (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes),
        (id)kIOSurfacePixelFormat:@0});
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 5) return 2;
        NSString *compiled = [NSString stringWithUTF8String:argv[1]];
        int channels = atoi(argv[2]);
        int spatial = atoi(argv[3]);
        int iters = atoi(argv[4]);
        NSUInteger logicalBytes = (NSUInteger)channels * (NSUInteger)spatial * 4;
        NSUInteger allocBytes = logicalBytes;
        if (allocBytes < (49 * 1024)) allocBytes = (49 * 1024);

        NSData *milData = [[NSString stringWithContentsOfFile:[compiled stringByAppendingPathComponent:@"model.mil"]
                                                     encoding:NSUTF8StringEncoding
                                                        error:nil] dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = [NSData dataWithContentsOfFile:[compiled stringByAppendingPathComponent:@"weights/weight.bin"]];

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

        NSError *e = nil;
        NSDictionary *wdict = @{
            @"@model_path/weights/weight.bin": @{@"offset": @64, @"data": weightBlob}
        };
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (!desc) {
            print_json_error("descriptor", "nil");
            return 3;
        }

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            print_json_error("model", "nil");
            return 4;
        }

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

        IOSurfaceRef ioIn = make_surface(allocBytes);
        IOSurfaceRef ioOut = make_surface(allocBytes);
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        float *buf = (float*)malloc(logicalBytes);
        for (int i = 0; i < channels * spatial; i++) {
            buf[i] = (float)(i % 97) / 97.0f;
        }
        IOSurfaceLock(ioIn, 0, NULL);
        memset(IOSurfaceGetBaseAddress(ioIn), 0, allocBytes);
        memcpy(IOSurfaceGetBaseAddress(ioIn), buf, logicalBytes);
        IOSurfaceUnlock(ioIn, 0, NULL);
        free(buf);

        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) {
            print_json_error("eval", e ? [[e description] UTF8String] : "");
            return 7;
        }

        for (int i = 0; i < 10; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        }
        uint64_t t1 = mach_absolute_time();
        printf("{\"ok\":true,\"per_iter_ms\":%.6f}\n", ticks_to_ms(t1 - t0) / iters);
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


def dequantize_linear_weight(linear) -> np.ndarray:
    weight = mx.dequantize(
        linear.weight,
        scales=linear.scales,
        biases=linear.biases,
        group_size=linear.group_size,
        bits=linear.bits,
        mode=linear.mode,
    )
    mx.eval(weight)
    mx.synchronize()
    return np.array(weight.astype(mx.float32).tolist(), dtype=np.float32)


def compile_linear(
    out_dir: Path,
    *,
    name: str,
    input_channels: int,
    output_channels: int,
    spatial: int,
    weight: np.ndarray,
) -> Path:
    kernel = weight.reshape(output_channels, input_channels, 1, 1)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, input_channels, 1, spatial), dtype=types.fp32),
        ]
    )
    def prog(x):
        return mb.conv(
            x=x,
            weight=kernel,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
        )

    model = ct.convert(prog, convert_to="mlprogram")
    pkg = out_dir / f"{name}.mlpackage"
    model.save(str(pkg))
    return Path(ct.models.utils.compile_model(str(pkg)))


def compile_ffn(
    out_dir: Path,
    *,
    name: str,
    dim: int,
    hidden: int,
    spatial: int,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
) -> Path:
    gate_proj = w1.reshape(hidden, dim, 1, 1)
    up_proj = w3.reshape(hidden, dim, 1, 1)
    down_proj = w2.reshape(dim, hidden, 1, 1)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, dim, 1, spatial), dtype=types.fp32)]
    )
    def prog(x):
        h1 = mb.conv(
            x=x,
            weight=gate_proj,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
        )
        h3 = mb.conv(
            x=x,
            weight=up_proj,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
        )
        sig = mb.sigmoid(x=h1)
        silu = mb.mul(x=h1, y=sig)
        gate = mb.mul(x=silu, y=h3)
        return mb.conv(
            x=gate,
            weight=down_proj,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
        )

    model = ct.convert(prog, convert_to="mlprogram")
    pkg = out_dir / f"{name}.mlpackage"
    model.save(str(pkg))
    return Path(ct.models.utils.compile_model(str(pkg)))


def build_helper(out_dir: Path) -> Path:
    src = out_dir / "josie_layer_ops_helper.m"
    exe = out_dir / "josie_layer_ops_helper"
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
    return exe


def probe_one(
    helper: Path,
    compiled: Path,
    *,
    name: str,
    channels: int,
    spatial: int,
    layer: int,
) -> dict[str, object]:
    proc = subprocess.run(
        [str(helper), str(compiled), str(channels), str(spatial), "5"],
        capture_output=True,
        text=True,
        check=False,
    )
    result = json.loads(proc.stdout.strip() or "{}")
    result["returncode"] = proc.returncode
    return {
        "name": name,
        "layer": layer,
        "channels": channels,
        "spatial": spatial,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--spatial", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, _tokenizer = load_josie(lazy=True)
    layer = model.model.layers[args.layer]
    attn = layer.self_attn
    mlp = layer.mlp

    work_dir = Path(tempfile.mkdtemp(prefix="josie_layer_ops_probe_"))
    try:
        helper = build_helper(work_dir)
        reports = []

        linear_specs = [
            ("q_proj", 2560, 4096, dequantize_linear_weight(attn.q_proj)),
            ("k_proj", 2560, 1024, dequantize_linear_weight(attn.k_proj)),
            ("v_proj", 2560, 1024, dequantize_linear_weight(attn.v_proj)),
            ("o_proj", 4096, 2560, dequantize_linear_weight(attn.o_proj)),
        ]
        for name, input_channels, output_channels, weight in linear_specs:
            compiled = compile_linear(
                work_dir,
                name=f"layer{args.layer}_{name}",
                input_channels=input_channels,
                output_channels=output_channels,
                spatial=args.spatial,
                weight=weight,
            )
            reports.append(
                probe_one(
                    helper,
                    compiled,
                    name=name,
                    channels=input_channels,
                    spatial=args.spatial,
                    layer=args.layer,
                )
            )

        gate_proj = dequantize_linear_weight(mlp.gate_proj)
        up_proj = dequantize_linear_weight(mlp.up_proj)
        down_proj = dequantize_linear_weight(mlp.down_proj)
        compiled = compile_ffn(
            work_dir,
            name=f"layer{args.layer}_ffn",
            dim=gate_proj.shape[1],
            hidden=gate_proj.shape[0],
            spatial=args.spatial,
            w1=gate_proj,
            w3=up_proj,
            w2=down_proj,
        )
        reports.append(
            probe_one(
                helper,
                compiled,
                name="ffn",
                channels=gate_proj.shape[1],
                spatial=args.spatial,
                layer=args.layer,
            )
        )

        report = {
            "runtime": "josie_layer_ops_probe",
            "layer": args.layer,
            "spatial": args.spatial,
            "models": reports,
        }
        if args.json:
            print(json.dumps(report))
            return
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
