from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


HELPER_SRC = r'''
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "ane_interop.h"

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) return 2;
    NSString *root = [NSString stringWithUTF8String:argv[1]];
    int bytes = atoi(argv[2]);

    NSData *mil = [[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"]
                                             encoding:NSUTF8StringEncoding
                                                error:nil] dataUsingEncoding:NSUTF8StringEncoding];
    NSData *w1 = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"weights/w1.bin"]];
    NSData *w3 = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"weights/w3.bin"]];
    NSData *w2 = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"weights/w2.bin"]];
    const char *paths[] = {
      "@model_path/weights/w1.bin",
      "@model_path/weights/w3.bin",
      "@model_path/weights/w2.bin",
    };
    const uint8_t *datas[] = {
      (const uint8_t *)w1.bytes,
      (const uint8_t *)w3.bytes,
      (const uint8_t *)w2.bytes,
    };
    size_t lens[] = {w1.length, w3.length, w2.length};
    size_t inb = (size_t)bytes;
    size_t outb = (size_t)bytes;
    ANEHandle *h = ane_interop_compile(
        (const uint8_t *)mil.bytes,
        mil.length,
        paths,
        datas,
        lens,
        3,
        1,
        &inb,
        1,
        &outb);
    if (!h) {
      printf("{\"ok\":false,\"stage\":\"compile\",\"code\":%d}\n", ane_interop_last_compile_error());
      return 3;
    }

    IOSurfaceRef inp = ane_interop_copy_input(h, 0);
    NSData *input = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"input.bin"]];
    memcpy(IOSurfaceGetBaseAddress(inp), input.bytes, input.length);
    CFRelease(inp);

    BOOL ok = ane_interop_eval(h);
    printf("{\"ok\":%s,\"hw_ns\":%llu}\n", ok ? "true" : "false",
           (unsigned long long)ane_interop_last_hw_execution_time_ns(h));
    ane_interop_free(h);
    return ok ? 0 : 7;
  }
}
'''


def blob(path: Path, arr: np.ndarray) -> None:
    arr = np.array(arr, dtype=np.float16, copy=False).reshape(-1)
    total = bytearray(128 + arr.nbytes)
    total[0] = 1
    total[4] = 2
    total[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    total[68] = 1
    total[72:76] = int(arr.nbytes).to_bytes(4, "little")
    total[80:84] = (128).to_bytes(4, "little")
    total[128:] = arr.tobytes()
    path.write_bytes(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    layer = model.model.layers[0]
    dim = 2560
    hidden = 9728
    token_ids = tokenizer.encode("Explain why MLX is useful for local inference.")[:8]
    seq_len = len(token_ids)
    tokens = mx.array([token_ids], dtype=mx.int32)
    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()
    ffn_in = layer.post_attention_layernorm(h)

    mlp = layer.mlp
    w1 = mx.dequantize(
        mlp.gate_proj.weight,
        scales=mlp.gate_proj.scales,
        biases=mlp.gate_proj.biases,
        group_size=mlp.gate_proj.group_size,
        bits=mlp.gate_proj.bits,
        mode=mlp.gate_proj.mode,
    )
    w3 = mx.dequantize(
        mlp.up_proj.weight,
        scales=mlp.up_proj.scales,
        biases=mlp.up_proj.biases,
        group_size=mlp.up_proj.group_size,
        bits=mlp.up_proj.bits,
        mode=mlp.up_proj.mode,
    )
    w2 = mx.dequantize(
        mlp.down_proj.weight,
        scales=mlp.down_proj.scales,
        biases=mlp.down_proj.biases,
        group_size=mlp.down_proj.group_size,
        bits=mlp.down_proj.bits,
        mode=mlp.down_proj.mode,
    )
    mx.eval(w1, w2, w3)
    mx.synchronize()

    w1_np = np.array(w1.astype(mx.float32).tolist(), dtype=np.float32).astype(np.float16).reshape(hidden, dim, 1, 1)
    w3_np = np.array(w3.astype(mx.float32).tolist(), dtype=np.float32).astype(np.float16).reshape(hidden, dim, 1, 1)
    w2_np = np.array(w2.astype(mx.float32).tolist(), dtype=np.float32).astype(np.float16).reshape(dim, hidden, 1, 1)
    inp = np.array(
        ffn_in.transpose(0, 2, 1).reshape(1, dim, 1, seq_len).astype(mx.float32).tolist(),
        dtype=np.float32,
    ).astype(np.float16)

    work = Path(tempfile.mkdtemp(prefix="josie_interop_probe_"))
    try:
        root = work / "artifact"
        (root / "weights").mkdir(parents=True)
        (root / "input.bin").write_bytes(inp.tobytes())
        blob(root / "weights" / "w1.bin", w1_np)
        blob(root / "weights" / "w3.bin", w3_np)
        blob(root / "weights" / "w2.bin", w2_np)
        mil = f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
 func main<ios18>(tensor<fp16, [1,{dim},1,{seq_len}]> x) {{
  string pt = const()[name=string("pt"), val=string("valid")];
  tensor<int32,[2]> st = const()[name=string("st"), val=tensor<int32,[2]>([1,1])];
  tensor<int32,[4]> pd = const()[name=string("pd"), val=tensor<int32,[4]>([0,0,0,0])];
  tensor<int32,[2]> dl = const()[name=string("dl"), val=tensor<int32,[2]>([1,1])];
  int32 gr = const()[name=string("gr"), val=int32(1)];
  tensor<fp16,[{hidden},{dim},1,1]> W1 = const()[name=string("W1"), val=tensor<fp16,[{hidden},{dim},1,1]>(BLOBFILE(path=string("@model_path/weights/w1.bin"), offset=uint64(64)))];
  tensor<fp16,[{hidden},{dim},1,1]> W3 = const()[name=string("W3"), val=tensor<fp16,[{hidden},{dim},1,1]>(BLOBFILE(path=string("@model_path/weights/w3.bin"), offset=uint64(64)))];
  tensor<fp16,[{dim},{hidden},1,1]> W2 = const()[name=string("W2"), val=tensor<fp16,[{dim},{hidden},1,1]>(BLOBFILE(path=string("@model_path/weights/w2.bin"), offset=uint64(64)))];
  tensor<fp16,[1,{hidden},1,{seq_len}]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=x)[name=string("h1")];
  tensor<fp16,[1,{hidden},1,{seq_len}]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=x)[name=string("h3")];
  tensor<fp16,[1,{hidden},1,{seq_len}]> sig = sigmoid(x=h1)[name=string("sig")];
  tensor<fp16,[1,{hidden},1,{seq_len}]> silu = mul(x=h1,y=sig)[name=string("silu")];
  tensor<fp16,[1,{hidden},1,{seq_len}]> gate = mul(x=silu,y=h3)[name=string("gate")];
  tensor<fp16,[1,{dim},1,{seq_len}]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string("out")];
 }} -> (out);
}}
'''
        (root / "model.mil").write_text(mil, encoding="utf-8")
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
        paths = ["inmem", "client", "clientDirect", "realtime"]
        models = []
        for path_name in paths:
            env = dict(os.environ)
            env["ANE_EVAL_PATH"] = path_name
            proc = subprocess.run(
                [str(exe), str(root), str(dim * seq_len * 2)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            models.append(
                {
                    "path": path_name,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )
        report = {"runtime": "josie_interop_ffn_probe", "models": models}
        if args.json:
            print(json.dumps(report))
            return
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
