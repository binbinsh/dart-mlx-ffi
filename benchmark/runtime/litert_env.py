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
TOOLS_DIR = ROOT / "benchmark" / "artifacts" / "tools" / "litert"
GOOGLE_ANDROID_MAVEN_BASE = "https://dl.google.com/dl/android/maven2"
DEFAULT_HOST_LITERT_VERSION = "2.16.1"
DEFAULT_ANDROID_LITERT_VERSION = "1.4.2"
MAVEN_ANDROID_AAR_URLS = (
    f"{GOOGLE_ANDROID_MAVEN_BASE}/com/google/ai/edge/litert/"
    "litert/{version}/litert-{version}.aar",
    "https://repo1.maven.org/maven2/org/tensorflow/"
    "tensorflow-lite/{version}/tensorflow-lite-{version}.aar",
    "https://repo.maven.apache.org/maven2/org/tensorflow/"
    "tensorflow-lite/{version}/tensorflow-lite-{version}.aar",
)
MAVEN_ANDROID_SELECT_OPS_AAR_URLS = (
    f"{GOOGLE_ANDROID_MAVEN_BASE}/com/google/ai/edge/litert/"
    "litert-select-tf-ops/{version}/litert-select-tf-ops-{version}.aar",
    "https://repo1.maven.org/maven2/org/tensorflow/"
    "tensorflow-lite-select-tf-ops/{version}/"
    "tensorflow-lite-select-tf-ops-{version}.aar",
    "https://repo.maven.apache.org/maven2/org/tensorflow/"
    "tensorflow-lite-select-tf-ops/{version}/"
    "tensorflow-lite-select-tf-ops-{version}.aar",
)


@dataclass(frozen=True)
class LiteRtEnvironment:
    version: str
    library: Path | None
    runtime_library: Path | None
    extra_libraries: tuple[Path, ...]
    package_root: Path | None

    @property
    def ready(self) -> bool:
        return self.library is not None

    def to_env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        if self.library is not None:
            values["DART_MLX_LITERT_LIBRARY"] = str(self.library)
        if self.runtime_library is not None:
            values["DART_MLX_TFLITE_LIBRARY"] = str(self.runtime_library)
        if self.extra_libraries:
            values["DART_MLX_LITERT_EXTRA_LIBRARIES"] = os.pathsep.join(
                str(path) for path in self.extra_libraries
            )
        return values

    def to_json(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "version": self.version,
            "library": str(self.library) if self.library else None,
            "runtime_library": str(self.runtime_library)
            if self.runtime_library
            else None,
            "extra_libraries": [str(path) for path in self.extra_libraries],
            "package_root": str(self.package_root) if self.package_root else None,
            "env": self.to_env(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover LiteRT/TFLite C API runtime paths for native backends."
    )
    parser.add_argument(
        "--target-os",
        default="host",
        choices=["host", "android"],
        help="Resolve LiteRT environment for host or Android cross-build.",
    )
    parser.add_argument(
        "--target-arch",
        default=None,
        help="Target architecture or Android ABI (e.g. arm64-v8a, x86_64).",
    )
    parser.add_argument(
        "--fetch-library",
        action="store_true",
        help="Download Android AAR runtime when the local cache is empty.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    env = resolve_litert_environment(
        target_os=args.target_os,
        target_arch=args.target_arch,
        fetch_library=args.fetch_library,
    )
    if args.shell:
        for key, value in env.to_env().items():
            print(f"export {key}={_shell_quote(value)}")
    else:
        print(json.dumps(env.to_json(), indent=2, ensure_ascii=False))
    if not env.ready:
        raise SystemExit(1)


def resolve_litert_environment(
    *,
    target_os: str = "host",
    target_arch: str | None = None,
    fetch_library: bool = False,
) -> LiteRtEnvironment:
    package_root, package_version = _litert_package()
    version = _resolve_version(package_version, target_os)
    if target_os == "android":
        library = _resolve_android_library(
            version=version,
            target_arch=target_arch,
            fetch_library=fetch_library,
        )
        extras = _resolve_android_extra_libraries(
            version=version,
            target_arch=target_arch,
            fetch_library=fetch_library,
        )
    else:
        library = _first_existing(_library_candidates(package_root))
        extras = tuple(_extra_library_paths_from_env())
    return LiteRtEnvironment(
        version=version,
        library=library,
        runtime_library=library,
        extra_libraries=extras,
        package_root=package_root,
    )


def _resolve_version(package_version: str, target_os: str) -> str:
    env = os.environ
    if target_os == "android":
        explicit = env.get("DART_MLX_LITERT_ANDROID_VERSION") or env.get(
            "DART_MLX_TFLITE_ANDROID_VERSION"
        )
        if explicit:
            return explicit
        explicit = env.get("DART_MLX_LITERT_VERSION") or env.get(
            "DART_MLX_TFLITE_VERSION"
        )
        if explicit:
            return explicit
        default_android = env.get("DART_MLX_LITERT_ANDROID_DEFAULT_VERSION")
        if default_android:
            return default_android
        return DEFAULT_ANDROID_LITERT_VERSION
    explicit = env.get("DART_MLX_LITERT_VERSION") or env.get("DART_MLX_TFLITE_VERSION")
    if explicit:
        return explicit
    if package_version != "unknown":
        return package_version
    return DEFAULT_HOST_LITERT_VERSION


def _litert_package() -> tuple[Path | None, str]:
    try:
        import tflite_runtime as package
    except ImportError:
        return None, "unknown"
    return (
        Path(package.__file__).resolve().parent,
        getattr(package, "__version__", "unknown"),
    )


def _library_candidates(package_root: Path | None) -> list[Path]:
    env = os.environ
    candidates = [
        _path(env.get("DART_MLX_LITERT_LIBRARY")),
        _path(env.get("DART_MLX_TFLITE_LIBRARY")),
    ]
    if package_root is not None:
        candidates.extend(
            sorted(package_root.glob("**/libtensorflowlite_c*.so*"))
            + sorted(package_root.glob("**/libtensorflowlite*.so*"))
            + sorted(package_root.glob("**/tensorflowlite*.dll"))
            + sorted(package_root.glob("**/libtensorflowlite*.dylib"))
        )
    candidates.extend(
        [
            Path("/opt/homebrew/lib/libtensorflowlite_c.dylib"),
            Path("/opt/homebrew/lib/libtensorflowlite.dylib"),
            Path("/usr/local/lib/libtensorflowlite_c.dylib"),
            Path("/usr/local/lib/libtensorflowlite.dylib"),
            Path("/usr/lib/libtensorflowlite_c.so"),
            Path("/usr/lib/libtensorflowlite.so"),
        ]
    )
    return [candidate for candidate in candidates if candidate is not None]


def _resolve_android_library(
    *,
    version: str,
    target_arch: str | None,
    fetch_library: bool,
) -> Path | None:
    env = os.environ
    explicit = _path(
        env.get("DART_MLX_LITERT_LIBRARY") or env.get("DART_MLX_TFLITE_LIBRARY")
    )
    if explicit is not None and explicit.exists():
        return explicit.resolve()
    if version == "unknown":
        return None
    abi = _normalize_android_abi(target_arch)
    lib_path = TOOLS_DIR / f"v{version}" / "android" / abi / "libtensorflowlite_jni.so"
    if lib_path.exists():
        return lib_path.resolve()
    if not fetch_library:
        return None
    _download_android_aar(version)
    extracted = _extract_android_library(version, abi)
    return extracted.resolve() if extracted.exists() else None


def _resolve_android_extra_libraries(
    *,
    version: str,
    target_arch: str | None,
    fetch_library: bool,
) -> tuple[Path, ...]:
    explicit = _extra_library_paths_from_env()
    if explicit:
        return tuple(path.resolve() for path in explicit if path.exists())
    if version == "unknown":
        return ()
    abi = _normalize_android_abi(target_arch)
    lib_path = TOOLS_DIR / f"v{version}" / "android" / abi / "libtensorflowlite_flex_jni.so"
    if lib_path.exists():
        return (lib_path.resolve(),)
    if not fetch_library:
        return ()
    try:
        _download_android_select_ops_aar(version)
        extracted = _extract_android_select_ops_library(version, abi)
    except RuntimeError:
        return ()
    if extracted.exists():
        return (extracted.resolve(),)
    return ()


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
    aar_dir = TOOLS_DIR / f"v{version}" / "android"
    aar_dir.mkdir(parents=True, exist_ok=True)
    candidates = _android_runtime_aar_candidates(version)
    existing = _first_existing_file(candidates)
    if existing is not None:
        return existing
    aar_path = candidates[0]
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
        "Unable to download tensorflow-lite Android AAR:\n" + "\n".join(errors)
    )


def _download_android_select_ops_aar(version: str) -> Path:
    aar_dir = TOOLS_DIR / f"v{version}" / "android"
    aar_dir.mkdir(parents=True, exist_ok=True)
    candidates = _android_select_ops_aar_candidates(version)
    existing = _first_existing_file(candidates)
    if existing is not None:
        return existing
    aar_path = candidates[0]
    temp_path = aar_path.with_suffix(".aar.tmp")
    if temp_path.exists():
        temp_path.unlink()
    errors: list[str] = []
    for template in MAVEN_ANDROID_SELECT_OPS_AAR_URLS:
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
        "Unable to download tensorflow-lite-select-tf-ops Android AAR:\n"
        + "\n".join(errors)
    )


def _extract_android_library(version: str, abi: str) -> Path:
    aar_path = _first_existing_file(_android_runtime_aar_candidates(version))
    if aar_path is None:
        raise RuntimeError(f"Missing downloaded AAR for LiteRT runtime v{version}")
    target = TOOLS_DIR / f"v{version}" / "android" / abi / "libtensorflowlite_jni.so"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    member = f"jni/{abi}/libtensorflowlite_jni.so"
    with zipfile.ZipFile(aar_path) as archive:
        if member not in archive.namelist():
            raise RuntimeError(f"AAR does not contain {member}: {aar_path}")
        with archive.open(member) as source, target.open("wb") as out:
            shutil.copyfileobj(source, out)
    return target


def _extract_android_select_ops_library(version: str, abi: str) -> Path:
    aar_path = _first_existing_file(_android_select_ops_aar_candidates(version))
    if aar_path is None:
        raise RuntimeError(f"Missing downloaded AAR for LiteRT select ops v{version}")
    target = TOOLS_DIR / f"v{version}" / "android" / abi / "libtensorflowlite_flex_jni.so"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    member = f"jni/{abi}/libtensorflowlite_flex_jni.so"
    with zipfile.ZipFile(aar_path) as archive:
        if member not in archive.namelist():
            raise RuntimeError(f"AAR does not contain {member}: {aar_path}")
        with archive.open(member) as source, target.open("wb") as out:
            shutil.copyfileobj(source, out)
    return target


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _extra_library_paths_from_env() -> list[Path]:
    raw = os.environ.get("DART_MLX_LITERT_EXTRA_LIBRARIES", "")
    if not raw:
        return []
    items = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    return [Path(item).expanduser() for item in items]


def _path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def _first_existing_file(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _android_runtime_aar_candidates(version: str) -> tuple[Path, ...]:
    root = TOOLS_DIR / f"v{version}" / "android"
    return (
        root / f"litert-runtime-{version}.aar",
        root / f"litert-{version}.aar",
        root / f"tensorflow-lite-{version}.aar",
    )


def _android_select_ops_aar_candidates(version: str) -> tuple[Path, ...]:
    root = TOOLS_DIR / f"v{version}" / "android"
    return (
        root / f"litert-select-ops-{version}.aar",
        root / f"litert-select-tf-ops-{version}.aar",
        root / f"tensorflow-lite-select-tf-ops-{version}.aar",
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
