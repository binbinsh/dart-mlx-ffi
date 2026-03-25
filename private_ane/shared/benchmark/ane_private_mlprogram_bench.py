from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


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
            case '\\': [escaped appendString:@"\\\\\\\\"]; break;
            case '\"': [escaped appendString:@"\\\\\\\""]; break;
            case '\n': [escaped appendString:@"\\\\n"]; break;
            case '\r': [escaped appendString:@"\\\\r"]; break;
            case '\t': [escaped appendString:@"\\\\t"]; break;
            default: [escaped appendFormat:@"%c", *p]; break;
        }
    }
    printf("{\"ok\":false,\"stage\":\"%s\",\"error\":\"%s\"}\n", stage, escaped.UTF8String);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 4) return 2;
        NSString *compiled = [NSString stringWithUTF8String:argv[1]];
        int channels = atoi(argv[2]);
        int spatial = atoi(argv[3]);
        int iters = argc >= 5 ? atoi(argv[4]) : 50;

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
            printf("{\"ok\":false,\"stage\":\"descriptor\"}\n");
            return 3;
        }

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            printf("{\"ok\":false,\"stage\":\"model\"}\n");
            return 4;
        }

        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

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

        NSUInteger bytes = (NSUInteger)channels * (NSUInteger)spatial * 4;
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),
            (id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,
            (id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),
            (id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),
            (id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,
            (id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),
            (id)kIOSurfacePixelFormat:@0});
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        float *buf = (float*)malloc(bytes);
        for (int i = 0; i < channels * spatial; i++) {
            buf[i] = (float)(i % 97) / 97.0f;
        }
        IOSurfaceLock(ioIn, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(ioIn), buf, bytes);
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


def make_model(out_dir: Path, *, channels: int, spatial: int) -> Path:
    W = np.eye(channels, dtype=np.float32).reshape(channels, channels, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, channels, 1, spatial), dtype=types.fp32)])
    def prog(x):
        return mb.conv(
            x=x,
            weight=W,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
        )

    model = ct.convert(prog, convert_to="mlprogram")
    pkg = out_dir / f"conv_{channels}x{spatial}.mlpackage"
    model.save(str(pkg))
    compiled = ct.models.utils.compile_model(str(pkg))
    return Path(compiled)


def build_helper(out_dir: Path) -> Path:
    src = out_dir / "ane_private_helper.m"
    exe = out_dir / "ane_private_helper"
    src.write_text(HELPER_SRC)
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


def bench_one(helper: Path, compiled: Path, *, channels: int, spatial: int, iters: int) -> dict[str, object]:
    proc = subprocess.run(
        [str(helper), str(compiled), str(channels), str(spatial), str(iters)],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(proc.stdout.strip() or "{}")
    payload["returncode"] = proc.returncode
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-ms", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix="ane_private_mlprogram_"))
    try:
        helper = build_helper(work_dir)
        report = {"runtime": "private_ane_mlprogram", "models": []}
        for channels, spatial in [(256, 64), (512, 64)]:
            compiled = make_model(work_dir, channels=channels, spatial=spatial)
            result = bench_one(
                helper,
                compiled,
                channels=channels,
                spatial=spatial,
                iters=args.iters,
            )
            result["name"] = f"conv-{channels}x{spatial}"
            report["models"].append(result)
            if result.get("ok") and float(result["per_iter_ms"]) > args.max_ms:
                raise SystemExit(
                    f"{result['name']} exceeded max ms/iter: {result['per_iter_ms']} > {args.max_ms}"
                )
            if not result.get("ok"):
                raise SystemExit(
                    f"{result['name']} failed at stage={result.get('stage')} error={result.get('error','')}"
                )

        if args.json:
            print(json.dumps(report))
            return

        print("private ane mlprogram benchmark")
        for model in report["models"]:
            print(f"{model['name']}: {model['per_iter_ms']:.4f} ms/iter")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
