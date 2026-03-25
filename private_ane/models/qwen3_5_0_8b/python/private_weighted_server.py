from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


HELPER_SRC = r'''
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
        if (argc < 4) return 2;
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        int inputBytes = atoi(argv[2]);
        int outputBytes = atoi(argv[3]);
        int allocBytes = inputBytes > outputBytes ? inputBytes : outputBytes;

        NSData *milData = [[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"]
                                                     encoding:NSUTF8StringEncoding
                                                        error:nil] dataUsingEncoding:NSUTF8StringEncoding];
        NSString *weightsDir = [root stringByAppendingPathComponent:@"weights"];
        NSArray<NSString *> *files = [[[NSFileManager defaultManager]
            contentsOfDirectoryAtPath:weightsDir
                                 error:nil] sortedArrayUsingSelector:@selector(compare:)];

        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (NSString *name in files) {
            NSString *path = [weightsDir stringByAppendingPathComponent:name];
            NSData *blob = [NSData dataWithContentsOfFile:path];
            if (blob == nil) continue;
            NSString *modelPath = [@"@model_path/weights/" stringByAppendingString:name];
            [wdict setObject:@{@"offset": @64, @"data": blob} forKey:modelPath];
        }

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        NSError *e = nil;

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (!desc) {
            print_error("descriptor", e);
            return 3;
        }
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            print_error("model", e);
            return 4;
        }

        id hx = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (NSString *name in files) {
            NSString *src = [weightsDir stringByAppendingPathComponent:name];
            NSString *dst = [[tmpDir stringByAppendingPathComponent:@"weights"] stringByAppendingPathComponent:name];
            [fm removeItemAtPath:dst error:nil];
            [fm copyItemAtPath:src toPath:dst error:nil];
        }

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

        IOSurfaceRef ioIn = make_surface(allocBytes);
        IOSurfaceRef ioOut = make_surface(allocBytes);
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        void *buf = malloc(inputBytes);
        while (read_exact(stdin, buf, inputBytes)) {
            IOSurfaceLock(ioIn, 0, NULL);
            memset(IOSurfaceGetBaseAddress(ioIn), 0, allocBytes);
            memcpy(IOSurfaceGetBaseAddress(ioIn), buf, inputBytes);
            IOSurfaceUnlock(ioIn, 0, NULL);

            IOSurfaceLock(ioOut, 0, NULL);
            memset(IOSurfaceGetBaseAddress(ioOut), 0, allocBytes);
            IOSurfaceUnlock(ioOut, 0, NULL);

            e = nil;
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            if (!ok) {
                print_error("eval", e);
                return 7;
            }

            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            const void *outPtr = IOSurfaceGetBaseAddress(ioOut);
            bool wrote = write_exact(stdout, outPtr, outputBytes);
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
            if (!wrote) return 8;
        }

        free(buf);
        return 0;
    }
}
'''


def build_weighted_helper(prefix: str) -> tuple[Path, Path]:
    work_dir = Path(tempfile.mkdtemp(prefix=prefix))
    src = work_dir / "ane_weighted_server.m"
    exe = work_dir / "ane_weighted_server"
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
    return work_dir, exe


def build_weighted_server(
    artifact_dir: Path,
    *,
    input_bytes: int,
    output_bytes: int,
    prefix: str,
) -> tuple[Path, subprocess.Popen[bytes]]:
    work_dir, exe = build_weighted_helper(prefix)
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [
            str(exe),
            str(artifact_dir),
            str(input_bytes),
            str(output_bytes),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return work_dir, proc


def spawn_weighted_server(
    exe: Path,
    artifact_dir: Path,
    *,
    input_bytes: int,
    output_bytes: int,
) -> subprocess.Popen[bytes]:
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [
            str(exe),
            str(artifact_dir),
            str(input_bytes),
            str(output_bytes),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def close_weighted_server(work_dir: Path, proc: subprocess.Popen[bytes]) -> None:
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


def run_weighted_server(
    proc: subprocess.Popen[bytes],
    input_blob: bytes,
    output_bytes: int,
) -> bytes:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(input_blob)
    proc.stdin.flush()
    chunk = proc.stdout.read(output_bytes)
    if chunk is None or len(chunk) != output_bytes:
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        raise RuntimeError(f"ANE weighted server failed. stderr={stderr}")
    return chunk
