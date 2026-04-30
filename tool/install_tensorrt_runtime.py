#!/usr/bin/env python3
"""Install TensorRT runtime libraries for the local ORT TensorRT EP path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


WHEELS = {
    "10.9.0.34": (
        "tensorrt_cu12_libs-10.9.0.34-py2.py3-none-"
        "manylinux_2_28_x86_64.whl",
        "https://pypi.nvidia.com/tensorrt-cu12-libs/"
        "tensorrt_cu12_libs-10.9.0.34-py2.py3-none-"
        "manylinux_2_28_x86_64.whl",
        3103291777,
    ),
    "10.13.3.9.post1": (
        "tensorrt_cu12_libs-10.13.3.9.post1-py2.py3-none-"
        "manylinux_2_28_x86_64.whl",
        "https://pypi.nvidia.com/tensorrt-cu12-libs/"
        "tensorrt_cu12_libs-10.13.3.9.post1-py2.py3-none-"
        "manylinux_2_28_x86_64.whl",
        3115267783,
    ),
    "10.16.1.11": (
        "tensorrt_cu12_libs-10.16.1.11-py3-none-"
        "manylinux_2_28_x86_64.whl",
        "https://pypi.nvidia.com/tensorrt-cu12-libs/"
        "tensorrt_cu12_libs-10.16.1.11-py3-none-"
        "manylinux_2_28_x86_64.whl",
        4304294549,
    ),
}

CUDA_PACKAGES = {
    "cuda_runtime": ("libcudart.so.12",),
    "cublas": ("libcublas.so.12", "libcublasLt.so.12"),
    "curand": ("libcurand.so.10",),
    "cufft": ("libcufft.so.11",),
    "cudnn": ("libcudnn.so.9",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="UniFrontend root. Installs into artifacts/runtime/{tensorrt,cuda}/lib.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache/dart_inference/tensorrt",
    )
    parser.add_argument(
        "--version",
        choices=sorted(WHEELS),
        default="10.9.0.34",
        help="TensorRT wheel version. 10.9 matches ORT's CUDA12 TRT test baseline.",
    )
    parser.add_argument(
        "--download-seconds",
        type=float,
        default=0.0,
        help="Download time budget. 0 means no download attempt.",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="Parallel range-download connections for large NVIDIA wheels.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-link-cuda", action="store_true")
    parser.add_argument("--uv-cache", type=Path, default=Path.home() / ".cache/uv")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def log(args: argparse.Namespace, message: str) -> None:
    if not args.json:
        print(message, file=sys.stderr)


def download_resume(
    url: str,
    dst: Path,
    expected_size: int,
    seconds: float,
    connections: int = 1,
) -> None:
    if seconds <= 0:
        return
    if connections > 1 and not dst.exists():
        download_parallel(url, dst, expected_size, seconds, connections)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    existing = dst.stat().st_size if dst.exists() else 0
    mode = "ab" if existing else "wb"
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        if error.code == 416 and existing == expected_size:
            return
        raise
    with response:
        if existing and response.status == 200:
            existing = 0
            mode = "wb"
        with dst.open(mode) as out:
            while True:
                if time.monotonic() - start >= seconds:
                    return
                chunk = response.read(1024 * 1024)
                if not chunk:
                    return
                out.write(chunk)


def download_parallel(
    url: str,
    dst: Path,
    expected_size: int,
    seconds: float,
    connections: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part_dir = dst.parent / f"{dst.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + seconds
    workers = max(1, min(connections, 16))
    segment_size = (expected_size + workers - 1) // workers
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(index: int, start: int, end: int) -> None:
        length = end - start + 1
        part = part_dir / f"{index:02d}.part"
        existing = part.stat().st_size if part.exists() else 0
        if existing >= length:
            return
        request_start = start + existing
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={request_start}-{end}"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=30)
            with response:
                if response.status != 206:
                    raise RuntimeError(
                        f"server ignored range request for segment {index}"
                    )
                with part.open("ab") as out:
                    while time.monotonic() < deadline:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            return
                        out.write(chunk)
        except BaseException as error:
            with errors_lock:
                errors.append(error)

    threads: list[threading.Thread] = []
    for index in range(workers):
        start = index * segment_size
        if start >= expected_size:
            break
        end = min(expected_size - 1, start + segment_size - 1)
        thread = threading.Thread(
            target=worker,
            args=(index, start, end),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if _parts_complete(part_dir, expected_size, segment_size):
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as out:
            for index in range(len(threads)):
                part = part_dir / f"{index:02d}.part"
                with part.open("rb") as src:
                    shutil.copyfileobj(src, out)
        tmp.replace(dst)
        shutil.rmtree(part_dir)


def _parts_complete(part_dir: Path, expected_size: int, segment_size: int) -> bool:
    index = 0
    start = 0
    while start < expected_size:
        end = min(expected_size - 1, start + segment_size - 1)
        part = part_dir / f"{index:02d}.part"
        if not part.is_file() or part.stat().st_size != end - start + 1:
            return False
        index += 1
        start += segment_size
    return True


def wheel_complete(path: Path, expected_size: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_size and zipfile.is_zipfile(path)


def extract_tensorrt(wheel: Path, target: Path, force: bool) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.startswith("tensorrt_libs/")]
        for name in names:
            if name.endswith("/"):
                continue
            basename = Path(name).name
            if sys.platform != "win32" and "_win.so" in basename:
                continue
            dst = target / basename
            if dst.exists() and not force:
                written.append(str(dst))
                continue
            with archive.open(name) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(str(dst))
    return written


def find_cuda_package_root(uv_cache: Path) -> Path | None:
    archive_root = uv_cache / "archive-v0"
    if not archive_root.is_dir():
        return None
    for archive in sorted(archive_root.iterdir()):
        if not archive.is_dir():
            continue
        for nvidia in archive.glob("lib/python*/site-packages/nvidia"):
            if all((nvidia / pkg / "lib" / libs[0]).is_file() for pkg, libs in CUDA_PACKAGES.items()):
                return nvidia
    return None


def link_cuda_runtime(nvidia: Path, target: Path, force: bool) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for package in CUDA_PACKAGES:
        src_dir = nvidia / package / "lib"
        for src in src_dir.glob("*.so*"):
            dst = target / src.name
            if dst.exists() or dst.is_symlink():
                if force:
                    dst.unlink()
                else:
                    linked.append(str(dst))
                    continue
            os.symlink(src, dst)
            linked.append(str(dst))
    return linked


def write_runtime_env(root: Path, tensorrt_dir: Path, cuda_dir: Path | None) -> Path:
    env = {
        "DART_INFERENCE_TENSORRT_LIBRARY_DIRS": str(tensorrt_dir),
    }
    if cuda_dir is not None:
        env["DART_INFERENCE_CUDA_LIBRARY_DIRS"] = str(cuda_dir)
    path = root / ".dart_inference_runtime_env.json"
    path.write_text(json.dumps(env, indent=2) + "\n")
    return path


def main() -> int:
    args = parse_args()
    filename, url, expected_size = WHEELS[args.version]
    wheel = args.cache_dir.expanduser() / filename
    result: dict[str, object] = {
        "version": args.version,
        "wheel": str(wheel),
        "expectedSize": expected_size,
        "downloadConnections": max(1, args.connections),
    }

    if args.download_seconds > 0:
        log(args, f"resuming {filename} for up to {args.download_seconds:.0f}s")
        download_resume(
            url,
            wheel,
            expected_size,
            args.download_seconds,
            max(1, args.connections),
        )

    size = wheel.stat().st_size if wheel.exists() else 0
    complete = wheel_complete(wheel, expected_size)
    result["downloadedSize"] = size
    result["wheelComplete"] = complete

    if args.root is None:
        print(json.dumps(result) if args.json else result)
        return 0 if complete else 78

    root = args.root.expanduser().resolve()
    trt_dir = root / "artifacts/runtime/tensorrt/lib"
    cuda_dir = root / "artifacts/runtime/cuda/lib"
    result["tensorrtDir"] = str(trt_dir)
    result["cudaDir"] = str(cuda_dir)

    if complete:
        result["tensorrtLibraries"] = extract_tensorrt(wheel, trt_dir, args.force)
    else:
        result["error"] = "TensorRT wheel is incomplete; rerun with --download-seconds to resume."

    if not args.no_link_cuda:
        nvidia = find_cuda_package_root(args.uv_cache.expanduser())
        result["cudaSource"] = str(nvidia) if nvidia is not None else None
        if nvidia is not None:
            result["cudaLibraries"] = link_cuda_runtime(nvidia, cuda_dir, args.force)

    if complete:
        result["runtimeEnvFile"] = str(write_runtime_env(root, trt_dir, cuda_dir))

    print(json.dumps(result, indent=2) if args.json else result)
    return 0 if complete else 78


if __name__ == "__main__":
    raise SystemExit(main())
