from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent
TOOLS_DIR = ROOT / "benchmark" / "artifacts" / "tools" / "onnxruntime"
TOOLS_DIR_FALLBACK = ROOT / "benchmark" / "artifacts_local" / "tools" / "onnxruntime"
TOOLS_DIR_ENV = "DART_MLX_ORT_TOOLS_DIR"
REQUIRED_HEADERS = ["onnxruntime_c_api.h", "onnxruntime_ep_c_api.h"]
DEFAULT_ORT_VERSION = "1.25.0"
MAVEN_ANDROID_AAR_URLS = (
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/"
    "onnxruntime-android/{version}/onnxruntime-android-{version}.aar",
    "https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/"
    "onnxruntime-android/{version}/onnxruntime-android-{version}.aar",
)


@dataclass(frozen=True)
class OrtEnvironment:
    version: str
    include_dir: Path | None
    library: Path | None
    runtime_library: Path | None
    package_root: Path | None

    @property
    def ready(self) -> bool:
        return self.include_dir is not None and self.library is not None

    def to_env(self) -> dict[str, str]:
        values = {
            "DART_INFERENCE_ENABLE_ORT": "1",
            "DART_MLX_ENABLE_ORT": "1",
        }
        if self.include_dir is not None:
            values["DART_INFERENCE_ORT_INCLUDE_DIR"] = str(self.include_dir)
            values["DART_MLX_ORT_INCLUDE_DIR"] = str(self.include_dir)
        if self.library is not None:
            values["DART_INFERENCE_ORT_LIBRARY"] = str(self.library)
            values["DART_MLX_ORT_LIBRARY"] = str(self.library)
        if self.runtime_library is not None:
            values["DART_INFERENCE_ORT_RUNTIME_LIBRARY"] = str(self.runtime_library)
            values["DART_MLX_ORT_RUNTIME_LIBRARY"] = str(self.runtime_library)
        return values

    def to_json(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "version": self.version,
            "include_dir": str(self.include_dir) if self.include_dir else None,
            "library": str(self.library) if self.library else None,
            "runtime_library": str(self.runtime_library)
            if self.runtime_library
            else None,
            "package_root": str(self.package_root) if self.package_root else None,
            "env": self.to_env(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover ONNX Runtime C API paths for native backend builds."
    )
    parser.add_argument("--fetch-headers", action="store_true")
    parser.add_argument(
        "--target-os",
        default="host",
        choices=["host", "android"],
        help="Resolve ONNX Runtime environment for host or Android cross-build.",
    )
    parser.add_argument(
        "--target-arch",
        default=None,
        help="Target architecture or Android ABI (e.g. arm64-v8a, x86_64).",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    env = resolve_ort_environment(
        fetch_headers=args.fetch_headers,
        target_os=args.target_os,
        target_arch=args.target_arch,
    )
    if args.shell:
        for key, value in env.to_env().items():
            print(f"export {key}={_shell_quote(value)}")
    else:
        print(json.dumps(env.to_json(), indent=2, ensure_ascii=False))
    if not env.ready:
        raise SystemExit(1)


def resolve_ort_environment(
    *,
    fetch_headers: bool = False,
    target_os: str = "host",
    target_arch: str | None = None,
) -> OrtEnvironment:
    package_root, version = _onnxruntime_package()
    resolved_version = _resolve_version(package_root, version, target_os)
    include_dir = _first_include_dir(_include_candidates(package_root, resolved_version))
    if (
        (include_dir is None or not _has_required_headers(include_dir))
        and fetch_headers
        and resolved_version != "unknown"
    ):
        include_dir = _fetch_c_header(resolved_version)
    if target_os == "android":
        library = _resolve_android_library(
            version=resolved_version,
            target_arch=target_arch,
            fetch_headers=fetch_headers,
        )
    else:
        library = _first_existing(_library_candidates(package_root))
    return OrtEnvironment(
        version=resolved_version,
        include_dir=include_dir,
        library=library,
        runtime_library=library,
        package_root=package_root,
    )


def _resolve_version(
    package_root: Path | None,
    package_version: str,
    target_os: str,
) -> str:
    env = os.environ
    if target_os == "android":
        explicit = env.get("DART_INFERENCE_ORT_ANDROID_VERSION") or env.get("DART_INFERENCE_ORT_VERSION")
        if explicit:
            return explicit
    explicit = env.get("DART_INFERENCE_ORT_VERSION")
    if explicit:
        return explicit
    if package_root is not None and package_version != "unknown":
        return package_version
    return DEFAULT_ORT_VERSION


def _onnxruntime_package() -> tuple[Path | None, str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "unknown"
    return Path(ort.__file__).resolve().parent, getattr(ort, "__version__", "unknown")


def _library_candidates(package_root: Path | None) -> list[Path]:
    env = os.environ
    candidates = [
        _path(env.get("DART_INFERENCE_ORT_RUNTIME_LIBRARY")),
        _path(env.get("DART_INFERENCE_ORT_LIBRARY")),
    ]
    if package_root is not None:
        capi = package_root / "capi"
        candidates.extend(sorted(capi.glob("libonnxruntime*.dylib")))
        candidates.extend(sorted(capi.glob("libonnxruntime*.so*")))
        candidates.extend(sorted(capi.glob("onnxruntime.dll")))
    candidates.extend(
        [
            Path("/opt/homebrew/lib/libonnxruntime.dylib"),
            Path("/usr/local/lib/libonnxruntime.dylib"),
            Path("/usr/lib/libonnxruntime.so"),
        ]
    )
    return [candidate for candidate in candidates if candidate is not None]


def _include_candidates(package_root: Path | None, version: str) -> list[Path]:
    env = os.environ
    candidates = [
        _path(env.get("DART_INFERENCE_ORT_INCLUDE_DIR")),
    ]
    if package_root is not None:
        candidates.extend(
            [
                package_root / "include",
                package_root / "capi",
                package_root.parent / "onnxruntime" / "include",
            ]
        )
    if version != "unknown":
        candidates.append(_download_include_dir(version))
    candidates.extend(
        [
            Path("/opt/homebrew/include/onnxruntime"),
            Path("/opt/homebrew/include"),
            Path("/usr/local/include/onnxruntime"),
            Path("/usr/local/include"),
        ]
    )
    return [candidate for candidate in candidates if candidate is not None]


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _first_include_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if _has_required_headers(candidate):
            return candidate.resolve()
    return None


def _has_required_headers(include_dir: Path | None) -> bool:
    if include_dir is None:
        return False
    return all((include_dir / header).exists() for header in REQUIRED_HEADERS)


def _fetch_c_header(version: str) -> Path:
    include_dir = _download_include_dir(version)
    if _has_required_headers(include_dir):
        return include_dir
    include_dir.mkdir(parents=True, exist_ok=True)
    for header in REQUIRED_HEADERS:
        target = include_dir / header
        if target.exists():
            continue
        url = (
            "https://raw.githubusercontent.com/microsoft/onnxruntime/"
            f"v{version}/include/onnxruntime/core/session/{header}"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                target.write_bytes(response.read())
        except Exception as error:
            raise RuntimeError(
                f"Unable to fetch ONNX Runtime C API header: {url}"
            ) from error
    return include_dir


def _download_include_dir(version: str) -> Path:
    return _tools_dir() / f"v{version}" / "include"


def _resolve_android_library(
    *,
    version: str,
    target_arch: str | None,
    fetch_headers: bool,
) -> Path | None:
    env = os.environ
    explicit = _path(env.get("DART_INFERENCE_ORT_LIBRARY"))
    if explicit is not None and explicit.exists():
        return explicit.resolve()
    if version == "unknown":
        return None
    abi = _normalize_android_abi(target_arch)
    lib_path = _tools_dir() / f"v{version}" / "android" / abi / "libonnxruntime.so"
    if lib_path.exists():
        return lib_path.resolve()
    if not fetch_headers:
        return None
    _download_android_aar(version)
    extracted = _extract_android_library(version, abi)
    return extracted.resolve() if extracted.exists() else None


def _normalize_android_abi(target_arch: str | None) -> str:
    if target_arch is None:
        return "arm64-v8a"
    normalized = target_arch.strip().lower()
    if normalized in {"arm64-v8a", "arm64", "aarch64"}:
        return "arm64-v8a"
    if normalized in {"armeabi-v7a", "armv7", "arm"}:
        return "armeabi-v7a"
    if normalized in {"x86_64", "x64"}:
        return "x86_64"
    if normalized in {"x86"}:
        return "x86"
    return normalized


def _download_android_aar(version: str) -> Path:
    aar_dir = _tools_dir(ensure_exists=True) / f"v{version}" / "android"
    aar_dir.mkdir(parents=True, exist_ok=True)
    aar_path = aar_dir / f"onnxruntime-android-{version}.aar"
    if aar_path.exists() and aar_path.stat().st_size > 0:
        return aar_path
    temp_path = aar_path.with_suffix(".aar.tmp")
    if temp_path.exists():
        temp_path.unlink()
    errors: list[str] = []
    for template in MAVEN_ANDROID_AAR_URLS:
        url = template.format(version=version)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with temp_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            temp_path.replace(aar_path)
            return aar_path
        except Exception as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError(
        "Unable to download onnxruntime-android AAR:\n" + "\n".join(errors)
    )


def _extract_android_library(version: str, abi: str) -> Path:
    tools_dir = _tools_dir()
    aar_path = tools_dir / f"v{version}" / "android" / f"onnxruntime-android-{version}.aar"
    if not aar_path.exists():
        raise RuntimeError(f"Missing downloaded AAR: {aar_path}")
    target = tools_dir / f"v{version}" / "android" / abi / "libonnxruntime.so"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    member = f"jni/{abi}/libonnxruntime.so"
    with zipfile.ZipFile(aar_path) as archive:
        if member not in archive.namelist():
            raise RuntimeError(f"AAR does not contain {member}: {aar_path}")
        with archive.open(member) as source, target.open("wb") as out:
            shutil.copyfileobj(source, out)
    return target


def _path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def _tools_dir(*, ensure_exists: bool = False) -> Path:
    for candidate in _tools_dir_candidates():
        if _has_broken_symlink_ancestor(candidate):
            continue
        if ensure_exists:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
        return candidate
    fallback = _tools_dir_candidates()[-1]
    if ensure_exists:
        fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _tools_dir_candidates() -> list[Path]:
    override = os.environ.get(TOOLS_DIR_ENV)
    if override:
        return [Path(override).expanduser()]
    return [TOOLS_DIR, TOOLS_DIR_FALLBACK]


def _has_broken_symlink_ancestor(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink() and not current.exists():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
