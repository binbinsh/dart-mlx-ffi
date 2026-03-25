from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


MIN_ALLOC_BYTES = 49 * 1024

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
        if (argc < 6) return 2;
        NSString *milPath = [NSString stringWithUTF8String:argv[1]];
        int inputCount = atoi(argv[2]);
        int outputCount = atoi(argv[3]);
        int allocBytes = atoi(argv[4]);
        if (argc < 5 + inputCount + outputCount) return 3;

        NSMutableArray<NSNumber *> *inputBytes = [NSMutableArray array];
        for (int i = 0; i < inputCount; ++i) {
            [inputBytes addObject:@(atoi(argv[5 + i]))];
        }
        NSMutableArray<NSNumber *> *outputBytes = [NSMutableArray array];
        for (int i = 0; i < outputCount; ++i) {
            [outputBytes addObject:@(atoi(argv[5 + inputCount + i]))];
        }

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
        if (!desc) return 4;
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return 5;

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
            return 6;
        }
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            print_error("load", e);
            return 7;
        }

        NSMutableArray *surfaces = [NSMutableArray array];
        NSMutableArray *inputs = [NSMutableArray array];
        NSMutableArray *inputIndices = [NSMutableArray array];
        NSMutableArray *buffers = [NSMutableArray array];
        for (int i = 0; i < inputCount; ++i) {
            IOSurfaceRef io = make_surface(allocBytes);
            [surfaces addObject:(__bridge id)io];
            [inputs addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                AIO, @selector(objectWithIOSurface:), io)];
            [inputIndices addObject:@(i)];
            void *buf = malloc([inputBytes[i] intValue]);
            [buffers addObject:[NSValue valueWithPointer:buf]];
        }

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
            inputs, inputIndices, outputs, outputIndices, nil, nil, @0);

        bool eof = false;
        while (!eof) {
            for (int i = 0; i < inputCount; ++i) {
                void *buf = [[buffers objectAtIndex:i] pointerValue];
                int bytes = [inputBytes[i] intValue];
                if (!read_exact(stdin, buf, bytes)) {
                    eof = true;
                    break;
                }
            }
            if (eof) break;

            for (int i = 0; i < inputCount; ++i) {
                IOSurfaceRef io = (__bridge IOSurfaceRef)[surfaces objectAtIndex:i];
                void *buf = [[buffers objectAtIndex:i] pointerValue];
                int bytes = [inputBytes[i] intValue];
                IOSurfaceLock(io, 0, NULL);
                memset(IOSurfaceGetBaseAddress(io), 0, allocBytes);
                memcpy(IOSurfaceGetBaseAddress(io), buf, bytes);
                IOSurfaceUnlock(io, 0, NULL);
            }

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
                return 8;
            }

            for (int i = 0; i < outputCount; ++i) {
                IOSurfaceRef io = (__bridge IOSurfaceRef)[outSurfaces objectAtIndex:i];
                IOSurfaceLock(io, kIOSurfaceLockReadOnly, NULL);
                const void *outPtr = IOSurfaceGetBaseAddress(io);
                bool wrote = write_exact(stdout, outPtr, [outputBytes[i] intValue]);
                IOSurfaceUnlock(io, kIOSurfaceLockReadOnly, NULL);
                if (!wrote) return 9;
            }
        }

        for (NSValue *value in buffers) {
            free([value pointerValue]);
        }
        return 0;
    }
}
'''


def build_helper_exe(prefix: str) -> tuple[Path, Path]:
    work_dir = Path(tempfile.mkdtemp(prefix=prefix))
    src = work_dir / "ane_server.m"
    exe = work_dir / "ane_server"
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


def build_server(
    mil_text: str,
    input_bytes: list[int],
    out_bytes: int,
    *,
    prefix: str,
) -> tuple[Path, subprocess.Popen[bytes]]:
    return build_multi_server(
        mil_text,
        input_bytes,
        [out_bytes],
        prefix=prefix,
    )


def build_multi_server(
    mil_text: str,
    input_bytes: list[int],
    output_bytes: list[int],
    *,
    prefix: str,
) -> tuple[Path, subprocess.Popen[bytes]]:
    helper_dir, exe = build_helper_exe(prefix)
    work_dir = helper_dir
    mil_path = work_dir / "model.mil"
    mil_path.write_text(mil_text, encoding="utf-8")
    alloc_bytes = max(MIN_ALLOC_BYTES, *output_bytes, *input_bytes)
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [
            str(exe),
            str(mil_path),
            str(len(input_bytes)),
            str(len(output_bytes)),
            str(alloc_bytes),
            *[str(item) for item in input_bytes],
            *[str(item) for item in output_bytes],
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return work_dir, proc


def spawn_multi_server(
    exe: Path,
    mil_text: str,
    input_bytes: list[int],
    output_bytes: list[int],
    *,
    prefix: str,
) -> tuple[Path, subprocess.Popen[bytes]]:
    work_dir = Path(tempfile.mkdtemp(prefix=prefix))
    mil_path = work_dir / "model.mil"
    mil_path.write_text(mil_text, encoding="utf-8")
    alloc_bytes = max(MIN_ALLOC_BYTES, *output_bytes, *input_bytes)
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [
            str(exe),
            str(mil_path),
            str(len(input_bytes)),
            str(len(output_bytes)),
            str(alloc_bytes),
            *[str(item) for item in input_bytes],
            *[str(item) for item in output_bytes],
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return work_dir, proc


def close_server(work_dir: Path, proc: subprocess.Popen[bytes]) -> None:
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


def run_server(
    proc: subprocess.Popen[bytes],
    inputs: list[bytes],
    out_bytes: int,
) -> bytes:
    return run_multi_server(proc, inputs, [out_bytes])[0]


def run_multi_server(
    proc: subprocess.Popen[bytes],
    inputs: list[bytes],
    output_bytes: list[int],
) -> list[bytes]:
    assert proc.stdin is not None and proc.stdout is not None
    try:
        for blob in inputs:
            proc.stdin.write(blob)
        proc.stdin.flush()
    except BrokenPipeError as error:
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        raise RuntimeError(f"ANE op server pipe failed. stderr={stderr}") from error
    outputs: list[bytes] = []
    for out_bytes in output_bytes:
        chunk = proc.stdout.read(out_bytes)
        if chunk is None or len(chunk) != out_bytes:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"ANE op server failed. stderr={stderr}")
        outputs.append(chunk)
    return outputs
